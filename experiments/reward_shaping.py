"""
EXPERIMENT: Reward Shaping in Tile Fields

Hypothesis: Binary win/loss may be too sparse a signal. Richer reward
schemes could accelerate learning or produce more robust policies — or
they could confuse the field with noisy gradients.

Protocol:
- Train TTT tile fields with 6 reward schemes (300 games each)
- Evaluate vs random AND vs greedy opponents (500 games each)
- Compare win rates, tile counts, exploration patterns, policy robustness

Reward schemes:
  a. Binary:       win=1, loss=0 (baseline)
  b. Margin:       win by more = higher reward (0.5 + 0.5 * moves_remaining/9)
  c. Progressive:  reward for "good" moves (center, blocking, threatening)
  d. Opponent-qual: higher reward for beating strong opponents
  e. Anti-draw:    penalize draws (draw=-0.5 instead of 0)
  f. Exploration:  small reward for visiting new tiles
"""

import random
import json
import math
import os
import sys
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zeroclaw.tile_field import TileField
from zeroclaw.games import TicTacToe


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]


# ---------------------------------------------------------------------------
# Helper game analysis
# ---------------------------------------------------------------------------

def find_winning_move(board, player):
    for a, b, c in WIN_LINES:
        cells = [board[a], board[b], board[c]]
        if cells.count(player) == 2 and cells.count(' ') == 1:
            return [a, b, c][cells.index(' ')]
    return None

def is_blocking_move(board_before, pos, player):
    """Did `player` just block opponent's winning line?"""
    opp = 'O' if player == 'X' else 'X'
    return find_winning_move(list(board_before), opp) == pos

def is_threatening_move(board_after, pos, player):
    """Did `player` create a two-in-a-row threat?"""
    for a, b, c in WIN_LINES:
        if pos in (a, b, c):
            cells = [board_after[a], board_after[b], board_after[c]]
            if cells.count(player) == 2 and cells.count(' ') == 1:
                return True
    return False

def count_threats(board, player):
    """Count number of two-in-a-row lines for player."""
    threats = 0
    for a, b, c in WIN_LINES:
        cells = [board[a], board[b], board[c]]
        if cells.count(player) == 2 and cells.count(' ') == 1:
            threats += 1
    return threats


# ---------------------------------------------------------------------------
# Opponent strategies for training and evaluation
# ---------------------------------------------------------------------------

def greedy_opponent(game: TicTacToe) -> str:
    """Greedy: win > block > center > corner > edge."""
    board = game.board
    me = game.current
    them = 'X' if me == 'O' else 'O'

    # Win if possible
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
    # Any
    return random.choice(game.legal_actions())


# ---------------------------------------------------------------------------
# Shaped-reward tile field
# ---------------------------------------------------------------------------

class ShapedRewardTileField(TileField):
    """TileField with configurable reward shaping."""

    def __init__(self, reward_scheme: str = "binary", **kwargs):
        super().__init__(**kwargs)
        self.reward_scheme = reward_scheme
        self._opponent_strength = 0.0  # Track opponent quality for scheme d
        self._visited_states = set()    # For exploration bonus (scheme f)

    def _compute_reward(self, game: TicTacToe, history: list, winner):
        """
        Compute shaped reward based on the scheme.
        History: list of (state_str, action, board_before, board_after, player, turn)
        """
        scheme = self.reward_scheme
        x_won = (winner == 'X')
        x_lost = (winner == 'O')
        is_draw = (winner is None and game.done)

        if scheme == "binary":
            # Baseline: simple win/loss
            return 1.0 if x_won else 0.0

        elif scheme == "margin":
            # Win faster = higher reward
            if x_won:
                moves_remaining = 9 - game.turn
                return 0.5 + 0.5 * moves_remaining / 9.0
            elif is_draw:
                return 0.3
            else:
                return 0.0

        elif scheme == "progressive":
            # Base reward + bonus for good moves during play
            base = 1.0 if x_won else (0.2 if is_draw else 0.0)
            bonus = 0.0
            for state_str, action, board_before, board_after, player, turn in history:
                if player != 'X':
                    continue
                pos = int(action)
                # Center move bonus
                if pos == 4 and board_before[4] == ' ':
                    bonus += 0.05
                # Blocking bonus
                if is_blocking_move(board_before, pos, 'X'):
                    bonus += 0.1
                # Threatening bonus
                if is_threatening_move(board_after, pos, 'X'):
                    bonus += 0.08
            return base + bonus

        elif scheme == "opponent_quality":
            # Reward scales with opponent strength estimate
            if x_won:
                # Higher reward for beating strong opponents
                return 0.5 + 0.5 * self._opponent_strength
            elif is_draw:
                return 0.2 * self._opponent_strength
            else:
                return 0.0

        elif scheme == "anti_draw":
            # Penalize draws
            if x_won:
                return 1.0
            elif is_draw:
                return -0.5
            else:
                return 0.0

        elif scheme == "exploration":
            # Base reward + small bonus for new state visits
            base = 1.0 if x_won else 0.0
            new_visits = 0
            for state_str, action, board_before, board_after, player, turn in history:
                if player != 'X':
                    continue
                if state_str not in self._visited_states:
                    new_visits += 1
            return base + 0.03 * new_visits

        return 1.0 if x_won else 0.0

    def train_game(self, game: TicTacToe, opponent_fn=None, evolve_every: int = 25) -> str:
        """Play one training game with shaped rewards."""
        game.reset()
        history = []
        # Track opponent quality: fraction of non-random moves
        opp_moves = 0
        opp_good_moves = 0

        while not game.done:
            state = game.state()
            actions = game.legal_actions()
            if not actions:
                break

            state_str = str(state.state_str)
            board_before = game.board[:]

            if game.current == 'X':
                action = self.choose_action(game, state_str, actions)
                player = 'X'
            else:
                if opponent_fn:
                    action = opponent_fn(game)
                    opp_moves += 1
                else:
                    action = random.choice(actions)
                    opp_moves += 1
                player = 'O'

            game.step(action)
            board_after = game.board[:]
            history.append((state_str, action, board_before, board_after, player, game.turn))

        winner = getattr(game, 'winner', None)
        shaped_reward = self._compute_reward(game, history, winner)

        # Record with shaped reward
        for state_str, action, board_before, board_after, player, turn in history:
            if player == 'X':
                if state_str in self.tiles and action in self.tiles[state_str]:
                    self.tiles[state_str][action]["chosen"] += 1
                    # Use shaped reward instead of binary won
                    if shaped_reward > 0:
                        self.tiles[state_str][action]["won"] += shaped_reward
                    # Track exploration for scheme f
                    if self.reward_scheme == "exploration":
                        self._visited_states.add(state_str)

        # Update opponent strength estimate (for scheme d)
        if self.reward_scheme == "opponent_quality" and opp_moves > 0:
            # Use a running average estimate — stronger opponents produce closer games
            game_length = game.turn
            # Heuristic: strong opponent → game is longer (draws/narrow wins)
            self._opponent_strength = min(1.0, game_length / 9.0)

        self._game_count += 1
        if self._game_count % evolve_every == 0:
            self.evolve()

        return winner


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_vs_opponent(field: ShapedRewardTileField, game: TicTacToe,
                          opponent_fn, n_games: int = 500) -> dict:
    """Evaluate a trained field against a specific opponent."""
    wins = 0
    losses = 0
    draws = 0

    for _ in range(n_games):
        game.reset()
        while not game.done:
            state = game.state()
            actions = game.legal_actions()
            if not actions:
                break

            state_str = str(state.state_str)
            if game.current == 'X':
                action = field.choose_action(game, state_str, actions)
            else:
                action = opponent_fn(game)

            game.step(action)

        w = getattr(game, 'winner', None)
        if w == 'X':
            wins += 1
        elif w == 'O':
            losses += 1
        else:
            draws += 1

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": round(wins / n_games, 4),
        "loss_rate": round(losses / n_games, 4),
        "draw_rate": round(draws / n_games, 4),
    }


def measure_exploration(field: ShapedRewardTileField) -> dict:
    """Analyze the exploration pattern of a trained field."""
    if not field.tiles:
        return {"tile_count": 0}

    scores = []
    visit_counts = []
    action_counts = []

    for state_str, tile in field.tiles.items():
        for action, data in tile.items():
            if data["chosen"] > 0:
                scores.append(data["score"])
                visit_counts.append(data["chosen"])
        action_counts.append(len(tile))

    return {
        "tile_count": len(field.tiles),
        "total_visits": sum(visit_counts),
        "avg_score": round(sum(scores) / len(scores), 4) if scores else 0,
        "score_std": round(
            (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)) ** 0.5, 4
        ) if len(scores) > 1 else 0,
        "max_visits": max(visit_counts) if visit_counts else 0,
        "avg_visits": round(sum(visit_counts) / len(visit_counts), 2) if visit_counts else 0,
        "avg_actions_per_tile": round(sum(action_counts) / len(action_counts), 2),
    }


# ---------------------------------------------------------------------------
# Single scheme trial
# ---------------------------------------------------------------------------

def run_scheme_trial(scheme: str, seed: int, train_games: int = 300,
                     eval_games: int = 500) -> dict:
    """Train and evaluate one reward scheme."""
    random.seed(seed)

    game = TicTacToe()
    field = ShapedRewardTileField(
        reward_scheme=scheme,
        n_simulations=20,
        temperature=0.3,
    )

    # Track training progress
    training_history = []
    window = 25

    for i in range(train_games):
        winner = field.train_game(game, opponent_fn=None, evolve_every=25)
        training_history.append(1 if winner == 'X' else 0)

    # Compute training win rate in windows
    training_windows = []
    for start in range(0, train_games, window):
        end = min(start + window, train_games)
        wins = sum(training_history[start:end])
        training_windows.append({
            "game_range": f"{start}-{end}",
            "win_rate": round(wins / (end - start), 4),
        })

    # Evaluate vs random
    random.seed(seed + 1000)
    vs_random = evaluate_vs_opponent(field, game, lambda g: random.choice(g.legal_actions()), eval_games)

    # Evaluate vs greedy
    random.seed(seed + 2000)
    vs_greedy = evaluate_vs_opponent(field, game, greedy_opponent, eval_games)

    # Exploration analysis
    exploration = measure_exploration(field)

    return {
        "scheme": scheme,
        "seed": seed,
        "train_games": train_games,
        "training_windows": training_windows,
        "vs_random": vs_random,
        "vs_greedy": vs_greedy,
        "exploration": exploration,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    schemes = {
        "binary": "Binary: win=1, loss=0 (baseline)",
        "margin": "Margin: win faster = higher reward",
        "progressive": "Progressive: bonus for good moves",
        "opponent_quality": "Opponent-quality: reward scales with opponent strength",
        "anti_draw": "Anti-draw: penalize draws (draw=-0.5)",
        "exploration": "Exploration: bonus for new tile visits",
    }

    n_trials = 5
    seeds = [42, 137, 256, 999, 1337]
    train_games = 300
    eval_games = 500

    all_results = {
        "experiment": "reward_shaping",
        "schemes": schemes,
        "n_trials": n_trials,
        "train_games": train_games,
        "eval_games": eval_games,
        "data": {},
    }

    print("=" * 70)
    print("EXPERIMENT: Reward Shaping in Tile Fields")
    print(f"Schemes: {list(schemes.keys())}")
    print(f"Trials per scheme: {n_trials}")
    print(f"Training games: {train_games}, Eval games: {eval_games}")
    print("=" * 70)

    for scheme_name, scheme_desc in schemes.items():
        print(f"\n{'='*50}")
        print(f"Scheme: {scheme_name}")
        print(f"  {scheme_desc}")
        print(f"{'='*50}")

        trials = []
        for t in range(n_trials):
            seed = seeds[t]
            print(f"\n  Trial {t+1}/{n_trials} (seed={seed})...", end=" ", flush=True)
            result = run_scheme_trial(scheme_name, seed, train_games, eval_games)
            trials.append(result)
            print(f"vs_random={result['vs_random']['win_rate']:.1%} | vs_greedy={result['vs_greedy']['win_rate']:.1%} | tiles={result['exploration']['tile_count']}")

        # Aggregate
        avg_random = sum(t["vs_random"]["win_rate"] for t in trials) / n_trials
        avg_greedy = sum(t["vs_greedy"]["win_rate"] for t in trials) / n_trials
        avg_random_loss = sum(t["vs_random"]["loss_rate"] for t in trials) / n_trials
        avg_greedy_loss = sum(t["vs_greedy"]["loss_rate"] for t in trials) / n_trials
        avg_tiles = sum(t["exploration"]["tile_count"] for t in trials) / n_trials
        avg_score = sum(t["exploration"]["avg_score"] for t in trials) / n_trials
        avg_score_std = sum(t["exploration"]["score_std"] for t in trials) / n_trials

        # Compute learning curve (average across trials)
        avg_curve = []
        for w_idx in range(len(trials[0]["training_windows"])):
            avg_wr = sum(t["training_windows"][w_idx]["win_rate"] for t in trials) / n_trials
            avg_curve.append(round(avg_wr, 4))

        all_results["data"][scheme_name] = {
            "trials": trials,
            "summary": {
                "avg_win_rate_vs_random": round(avg_random, 4),
                "avg_win_rate_vs_greedy": round(avg_greedy, 4),
                "avg_loss_rate_vs_random": round(avg_random_loss, 4),
                "avg_loss_rate_vs_greedy": round(avg_greedy_loss, 4),
                "robustness": round((avg_random + avg_greedy) / 2, 4),
                "avg_tiles": round(avg_tiles, 1),
                "avg_score": round(avg_score, 4),
                "avg_score_std": round(avg_score_std, 4),
                "avg_learning_curve": avg_curve,
            },
        }

        print(f"\n  SUMMARY ({n_trials} trials):")
        print(f"    vs Random:  {avg_random:.1%} wins, {avg_random_loss:.1%} losses")
        print(f"    vs Greedy:  {avg_greedy:.1%} wins, {avg_greedy_loss:.1%} losses")
        print(f"    Robustness: {(avg_random + avg_greedy) / 2:.1%}")
        print(f"    Tiles:      {avg_tiles:.0f}")
        print(f"    Avg Score:  {avg_score:.3f} (std={avg_score_std:.3f})")

    # --- Final comparison ---
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"\n  {'Scheme':<18s} | {'vs Rand':>7s} | {'vs Grdy':>7s} | {'Robust':>7s} | {'Tiles':>6s} | {'AvgScr':>6s} | {'ScoreStd':>8s}")
    print(f"  {'-'*18}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}")

    for scheme_name in schemes:
        s = all_results["data"][scheme_name]["summary"]
        print(f"  {scheme_name:<18s} | {s['avg_win_rate_vs_random']:>7.1%} | {s['avg_win_rate_vs_greedy']:>7.1%} | {s['robustness']:>7.1%} | {s['avg_tiles']:>6.0f} | {s['avg_score']:>6.3f} | {s['avg_score_std']:>8.4f}")

    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Best robust policy
    best_robust = max(all_results["data"].items(),
                      key=lambda x: x[1]["summary"]["robustness"])
    print(f"\n  Most robust: {best_robust[0]} (robustness={best_robust[1]['summary']['robustness']:.1%})")

    best_random = max(all_results["data"].items(),
                      key=lambda x: x[1]["summary"]["avg_win_rate_vs_random"])
    print(f"  Best vs random: {best_random[0]} ({best_random[1]['summary']['avg_win_rate_vs_random']:.1%})")

    best_greedy = max(all_results["data"].items(),
                      key=lambda x: x[1]["summary"]["avg_win_rate_vs_greedy"])
    print(f"  Best vs greedy: {best_greedy[0]} ({best_greedy[1]['summary']['avg_win_rate_vs_greedy']:.1%})")

    # Exploration comparison
    print(f"\n  Exploration (tile counts):")
    for scheme_name in schemes:
        s = all_results["data"][scheme_name]["summary"]
        print(f"    {scheme_name:<18s}: {s['avg_tiles']:.0f} tiles, score_std={s['avg_score_std']:.4f}")

    # Save
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(repo_dir, "reward-shaping-results.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")

    return all_results


if __name__ == "__main__":
    main()
