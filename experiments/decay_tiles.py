"""
EXPERIMENT: Memory Decay in Tile Fields

Hypothesis: Adding score decay (tiles slowly revert toward 0.5 without
reinforcement) will IMPROVE performance by pruning stale strategies and
keeping the field adaptive.

Protocol:
- Train TicTacToe tile field for 500 games with score decay
- At game 300, switch opponent strategy (random → optimal-ish)
- Measure win rates before/after switch, adaptation speed, tile churn
- Test decay rates: [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
- Also test: does decay hurt if opponent DOESN'T change?
"""

import random
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zeroclaw.tile_field import TileField
from zeroclaw.games import TicTacToe


# ---------------------------------------------------------------------------
# Opponent strategies
# ---------------------------------------------------------------------------

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def find_winning_move(board, player):
    for a, b, c in WIN_LINES:
        cells = [board[a], board[b], board[c]]
        if cells.count(player) == 2 and cells.count(' ') == 1:
            return [a, b, c][cells.index(' ')]
    return None

def semi_optimal_action(game: TicTacToe, optimal_prob: float = 0.6) -> str:
    """Semi-optimal opponent: plays well `optimal_prob` of the time, random otherwise."""
    if random.random() > optimal_prob:
        return random.choice(game.legal_actions())

    board = game.board
    me = game.current   # O
    them = 'X'

    # Win
    move = find_winning_move(board, me)
    if move is not None:
        return str(move)
    # Block
    move = find_winning_move(board, them)
    if move is not None:
        return str(move)
    # Center
    if board[4] == ' ':
        return '4'
    # Corners
    corners = [i for i in [0, 2, 6, 8] if board[i] == ' ']
    if corners:
        return str(random.choice(corners))
    return random.choice(game.legal_actions())


# ---------------------------------------------------------------------------
# Decaying tile field
# ---------------------------------------------------------------------------

class DecayingTileField(TileField):
    def __init__(self, decay_rate: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.decay_rate = decay_rate

    def apply_decay(self):
        for tile in self.tiles.values():
            for action_data in tile.values():
                score = action_data["score"]
                action_data["score"] = score + self.decay_rate * (0.5 - score)

    def train_game(self, game, opponent_fn=None, evolve_every: int = 25) -> str:
        game.reset()
        history = []

        while not game.done:
            state = game.state()
            actions = game.legal_actions()
            if not actions:
                break

            state_str = str(state.state_str)
            if game.current == 'X':
                action = self.choose_action(game, state_str, actions)
            else:
                if opponent_fn:
                    action = opponent_fn(game)
                else:
                    action = random.choice(actions)

            game.step(action)
            history.append((state_str, action, game.current))

        won = getattr(game, 'winner', None) == 'X'
        for state_str, action, player in history:
            if player == 'X':
                self.record(state_str, action, won)

        self._game_count += 1
        if self._game_count % evolve_every == 0:
            self.evolve()

        self.apply_decay()
        return getattr(game, 'winner', None)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_tile_churn(snap_before, snap_after):
    churn = 0
    total = 0
    for state_str in snap_before:
        if state_str in snap_after:
            b_best = max(snap_before[state_str], key=lambda a: snap_before[state_str][a]["score"])
            a_best = max(snap_after[state_str], key=lambda a: snap_after[state_str][a]["score"])
            if b_best != a_best:
                churn += 1
            total += 1
    return churn, max(total, 1)

def snapshot_best_actions(field):
    snap = {}
    for state_str, tile in field.tiles.items():
        snap[state_str] = {a: dict(d) for a, d in tile.items()}
    return snap


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def run_trial(decay_rate, switch_opponent, seed):
    """Run one trial with a specific seed."""
    random.seed(seed)
    game = TicTacToe()
    field = DecayingTileField(decay_rate=decay_rate, n_simulations=20, temperature=0.3)

    total_games = 500
    switch_point = 300
    window_size = 20

    game_results = []  # (game_num, x_won)
    snap_before = None

    for i in range(total_games):
        if switch_opponent and i >= switch_point:
            opponent_fn = lambda g: semi_optimal_action(g, 0.6)
        else:
            opponent_fn = None

        winner = field.train_game(game, opponent_fn=opponent_fn, evolve_every=25)
        game_results.append((i, 1 if winner == 'X' else 0))

        if switch_opponent and i == switch_point - 1:
            snap_before = snapshot_best_actions(field)

    # Compute windowed win rates
    windows = []
    for start in range(0, total_games, window_size):
        end = min(start + window_size, total_games)
        wins = sum(w for _, w in game_results[start:end])
        mid_game = (start + end) // 2
        windows.append((mid_game, wins / (end - start)))

    # Summary metrics
    before_wins = [wr for g, wr in windows if 200 <= g <= 300]
    after_wins = [wr for g, wr in windows if 400 <= g <= 500]
    # Immediately after switch: games 300-340
    immediate_after = [wr for g, wr in windows if 300 < g <= 340]

    pre_rate = sum(before_wins) / len(before_wins) if before_wins else 0
    post_rate = sum(after_wins) / len(after_wins) if after_wins else 0
    immediate_rate = sum(immediate_after) / len(immediate_after) if immediate_after else 0

    # Adaptation: first window after switch >= 80% of pre-switch rate
    adaptation_game = None
    if switch_opponent and pre_rate > 0:
        post_windows = [(g, wr) for g, wr in windows if g > switch_point]
        for g, wr in post_windows:
            if wr >= 0.8 * pre_rate:
                adaptation_game = g
                break

    # Tile churn
    churn_rate = None
    if switch_opponent and snap_before:
        snap_after = snapshot_best_actions(field)
        churn, total = compute_tile_churn(snap_before, snap_after)
        churn_rate = churn / total if total > 0 else 0

    return {
        "pre_switch_winrate": round(pre_rate, 4),
        "post_switch_winrate": round(post_rate, 4),
        "immediate_after_winrate": round(immediate_rate, 4),
        "adaptation_game": adaptation_game,
        "adaptation_speed": (adaptation_game - switch_point) if adaptation_game else None,
        "tile_churn_rate": round(churn_rate, 4) if churn_rate is not None else None,
        "final_tiles": len(field.tiles),
        "windows": [(g, round(wr, 4)) for g, wr in windows],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    decay_rates = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    n_trials = 5  # Run multiple trials per rate for statistical robustness
    base_seeds = [42, 137, 256, 999, 1337]

    all_results = {
        "experiment": "memory_decay",
        "decay_rates": decay_rates,
        "n_trials": n_trials,
        "conditions": {},
    }

    print("=" * 70)
    print("EXPERIMENT: Memory Decay in Tile Fields")
    print(f"Decay rates: {decay_rates}")
    print(f"Trials per rate: {n_trials}")
    print("=" * 70)

    # --- Condition A: With opponent switch ---
    print("\n--- CONDITION A: Opponent switches (random → semi-optimal) at game 300 ---")
    switch_results = {}
    for rate in decay_rates:
        trials = []
        for t in range(n_trials):
            seed = base_seeds[t] + int(rate * 10000)
            result = run_trial(rate, switch_opponent=True, seed=seed)
            trials.append(result)
        switch_results[str(rate)] = trials

        avg_pre = sum(t["pre_switch_winrate"] for t in trials) / n_trials
        avg_post = sum(t["post_switch_winrate"] for t in trials) / n_trials
        avg_imm = sum(t["immediate_after_winrate"] for t in trials) / n_trials
        adapted = [t["adaptation_speed"] for t in trials if t["adaptation_speed"] is not None]
        avg_adapt = sum(adapted) / len(adapted) if adapted else None
        churns = [t["tile_churn_rate"] for t in trials if t["tile_churn_rate"] is not None]
        avg_churn = sum(churns) / len(churns) if churns else 0

        print(f"\n  Decay={rate:.3f} ({n_trials} trials):")
        print(f"    Pre-switch WR:    {avg_pre:.1%}")
        print(f"    Post-switch WR:   {avg_post:.1%}")
        print(f"    Immediate after:  {avg_imm:.1%}")
        print(f"    Adapt speed:      {f'{avg_adapt:.0f} games' if avg_adapt else 'NEVER recovered'}")
        print(f"    Tile churn:       {avg_churn:.1%}")

    all_results["conditions"]["with_switch"] = switch_results

    # --- Condition B: Without opponent switch ---
    print("\n--- CONDITION B: Opponent stays random ---")
    no_switch_results = {}
    for rate in decay_rates:
        trials = []
        for t in range(n_trials):
            seed = base_seeds[t] + int(rate * 10000)
            result = run_trial(rate, switch_opponent=False, seed=seed)
            trials.append(result)
        no_switch_results[str(rate)] = trials

        avg_pre = sum(t["pre_switch_winrate"] for t in trials) / n_trials
        avg_post = sum(t["post_switch_winrate"] for t in trials) / n_trials
        delta = avg_post - avg_pre

        print(f"\n  Decay={rate:.3f}: Early={avg_pre:.1%}, Late={avg_post:.1%}, Delta={delta:+.1%}")

    all_results["conditions"]["without_switch"] = no_switch_results

    # --- Summary table ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nWITH opponent switch (adaptation test):")
    print(f"  {'Rate':>6s} | {'Pre':>6s} | {'Post':>6s} | {'Drop':>6s} | {'Adapt':>8s} | {'Churn':>6s}")
    print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}")
    for rate in decay_rates:
        trials = switch_results[str(rate)]
        avg_pre = sum(t["pre_switch_winrate"] for t in trials) / n_trials
        avg_post = sum(t["post_switch_winrate"] for t in trials) / n_trials
        drop = avg_post - avg_pre
        adapted = [t["adaptation_speed"] for t in trials if t["adaptation_speed"] is not None]
        adapt_str = f"{sum(adapted)/len(adapted):.0f}g" if adapted else "never"
        churns = [t["tile_churn_rate"] for t in trials if t["tile_churn_rate"] is not None]
        churn_str = f"{sum(churns)/len(churns):.1%}" if churns else "N/A"
        print(f"  {rate:>6.3f} | {avg_pre:>6.1%} | {avg_post:>6.1%} | {drop:>+6.1%} | {adapt_str:>8s} | {churn_str:>6s}")

    print("\nWITHOUT switch (decay cost/benefit):")
    print(f"  {'Rate':>6s} | {'Early':>6s} | {'Late':>6s} | {'Delta':>7s}")
    print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}")
    for rate in decay_rates:
        trials = no_switch_results[str(rate)]
        avg_pre = sum(t["pre_switch_winrate"] for t in trials) / n_trials
        avg_post = sum(t["post_switch_winrate"] for t in trials) / n_trials
        delta = avg_post - avg_pre
        print(f"  {rate:>6.3f} | {avg_pre:>6.1%} | {avg_post:>6.1%} | {delta:>+7.1%}")

    # Save
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "decay-results.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")

    return all_results


if __name__ == "__main__":
    main()
