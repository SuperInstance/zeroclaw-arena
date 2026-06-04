"""
Tile Interference Experiment
=============================
Do similar states interfere with each other?

Each state gets its own tile (hashed independently). But what if two very similar
states get OPPOSITE optimal actions? The field has to learn them independently —
no generalization.

Method:
1. Train TTT tile field for 500 games
2. After training, find all pairs of states that differ by EXACTLY ONE move
3. For each pair: do they agree on the best action? Or recommend opposite actions?
4. Measure: interference rate = % of adjacent state pairs that disagree on best action

Tests:
- Does high interference correlate with slow learning or poor performance?
- Are high-interference tiles visited more often (harder to learn)?
- Does increasing hash size (grouping similar states together) reduce interference?

This tests whether TILE INDEPENDENCE is a feature or a bug. If interference is low,
tiles don't need to communicate. If high, we need function approximation or state grouping.
"""

import json
import hashlib
import random
import time
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional, List, Dict, Tuple, Any
from itertools import combinations


# ─── TTT Game ─────────────────────────────────────────────

class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9
        self.current = 'X'
        self.turn = 0
        self.done = False
        self.winner = None

    def state(self) -> str:
        return ''.join(self.board)

    def legal_actions(self) -> list:
        if self.done:
            return []
        return [str(i) for i in range(9) if self.board[i] == ' ']

    def step(self, action: str) -> float:
        pos = int(action)
        if self.board[pos] != ' ':
            return -1.0
        self.board[pos] = self.current
        self.turn += 1
        lines = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
        for a, b, c in lines:
            if self.board[a] == self.board[b] == self.board[c] != ' ':
                self.done = True
                self.winner = self.current
                return 1.0 if self.current == 'X' else -1.0
        if self.turn >= 9:
            self.done = True
            return 0.0
        self.current = 'O' if self.current == 'X' else 'X'
        return 0.0

    def copy(self):
        g = TicTacToe.__new__(TicTacToe)
        g.board = self.board[:]
        g.current = self.current
        g.turn = self.turn
        g.done = self.done
        g.winner = self.winner
        return g

    def reset(self):
        self.board = [' '] * 9
        self.current = 'X'
        self.turn = 0
        self.done = False
        self.winner = None


# ─── Tile Field ───────────────────────────────────────────

class TileField:
    """Standard tile field — one tile per exact state string."""

    def __init__(self, n_simulations=20, temperature=0.3, hash_grouping=0):
        """
        hash_grouping: if > 0, group states by truncating their string to first N chars
                       or by hashing into N buckets. This simulates function approximation.
        """
        self.tiles = {}  # state_key -> {action: {"score": float, "chosen": int, "won": int}}
        self.n_simulations = n_simulations
        self.temperature = temperature
        self.hash_grouping = hash_grouping  # 0 = no grouping (exact state)

    def _state_key(self, state_str: str) -> str:
        if self.hash_grouping <= 0:
            return state_str
        # Group by hashing into N buckets
        h = hashlib.blake2b(state_str.encode(), digest_size=4).hexdigest()
        bucket = int(h, 16) % self.hash_grouping
        return f"bucket_{bucket}"

    def get_or_create(self, state_str, legal_actions):
        key = self._state_key(state_str)
        if key not in self.tiles:
            self.tiles[key] = {
                a: {"score": 0.5, "chosen": 0, "won": 0} for a in legal_actions
            }
        tile = self.tiles[key]
        for a in legal_actions:
            if a not in tile:
                tile[a] = {"score": 0.5, "chosen": 0, "won": 0}
        return tile

    def choose_action(self, game, state_str, legal_actions):
        if len(legal_actions) <= 1:
            return legal_actions[0] if legal_actions else ''
        tile = self.get_or_create(state_str, legal_actions)
        action_values = {}
        sims_per = max(1, self.n_simulations // len(legal_actions))
        for action in legal_actions:
            sim_wins = 0
            for _ in range(sims_per):
                g = game.copy()
                g.step(action)
                while not g.done:
                    acts = g.legal_actions()
                    if not acts:
                        break
                    g.step(random.choice(acts))
                if g.winner == 'X':
                    sim_wins += 1
            sim_score = sim_wins / max(sims_per, 1)
            learned_score = tile[action]["score"]
            n_chosen = tile[action]["chosen"]
            confidence = min(n_chosen / 20.0, 0.8)
            action_values[action] = confidence * learned_score + (1 - confidence) * sim_score
        actions_list = list(action_values.keys())
        values = np.array([action_values[a] for a in actions_list])
        exp_vals = np.exp(values / self.temperature)
        probs = exp_vals / exp_vals.sum()
        return np.random.choice(actions_list, p=probs)

    def record(self, state_str, action, won):
        key = self._state_key(state_str)
        if key in self.tiles and action in self.tiles[key]:
            self.tiles[key][action]["chosen"] += 1
            if won:
                self.tiles[key][action]["won"] += 1

    def evolve(self):
        for tile in self.tiles.values():
            for action, data in tile.items():
                if data["chosen"] > 0:
                    wr = data["won"] / data["chosen"]
                    data["score"] += 0.05 * (wr - data["score"])
                    data["score"] = max(0.05, min(0.95, data["score"]))

    def best_action(self, state_str, legal_actions) -> Optional[str]:
        """Return the best action for a given state (or None if unlearned)."""
        key = self._state_key(state_str)
        if key not in self.tiles:
            return None
        tile = self.tiles[key]
        best = None
        best_score = -1
        for a in legal_actions:
            if a in tile and tile[a]["chosen"] > 0:
                s = tile[a]["score"]
                if s > best_score:
                    best_score = s
                    best = a
        return best

    def tile_visit_count(self, state_str) -> int:
        key = self._state_key(state_str)
        if key not in self.tiles:
            return 0
        return sum(d["chosen"] for d in self.tiles[key].values())


# ─── Training ─────────────────────────────────────────────

def train_tile_field(num_games=500, n_simulations=20, evolve_every=25,
                     hash_grouping=0) -> Tuple[TileField, dict]:
    """Train tile field and track per-epoch performance."""
    game = TicTacToe()
    field = TileField(n_simulations=n_simulations, hash_grouping=hash_grouping)

    x_wins, o_wins, draws = 0, 0, 0
    learning_curve = []
    epoch_stats = []

    for i in range(num_games):
        game.reset()
        history = []
        while not game.done:
            state_str = game.state()
            actions = game.legal_actions()
            if not actions:
                break
            if game.current == 'X':
                action = field.choose_action(game, state_str, actions)
            else:
                action = random.choice(actions)
            game.step(action)
            history.append((state_str, action, 'X' if len(history) % 2 == 0 else 'O'))

        won_x = game.winner == 'X'
        for state_str, action, player in history:
            if player == 'X':
                field.record(state_str, action, won_x)

        if game.winner == 'X':
            x_wins += 1
        elif game.winner == 'O':
            o_wins += 1
        else:
            draws += 1

        if (i + 1) % evolve_every == 0:
            field.evolve()

        # Track learning curve every 10 games
        if (i + 1) % 10 == 0:
            window_x = 0
            for j in range(max(0, i - 9), i + 1):
                # Replay recent outcomes — we track inline
                pass
            learning_curve.append({
                "game": i + 1,
                "x_wins": x_wins,
                "o_wins": o_wins,
                "draws": draws,
                "x_wr": x_wins / (i + 1),
            })

    return field, {
        "x_wins": x_wins, "o_wins": o_wins, "draws": draws,
        "x_win_rate": x_wins / num_games,
        "num_tiles": len(field.tiles),
        "learning_curve": learning_curve,
    }


# ─── Adjacent State Discovery ─────────────────────────────

def find_adjacent_state_pairs(field: TileField) -> List[Tuple[str, str, List[int]]]:
    """
    Find all pairs of X-to-move states that differ by EXACTLY TWO cells.

    In TTT, X-to-move states have 0, 2, 4, or 6 pieces. Two states with the
    same piece count that differ by exactly 2 cells are 'adjacent' — one cell's
    contents swapped with another. This is the smallest meaningful change
    between two X-to-move positions.

    Example: State A has X at pos 0, O at pos 4.
             State B has X at pos 4, O at pos 0.
    Same piece count, 2 cells differ. The optimal move could be completely different.

    Returns list of (state_a, state_b, [diff_positions]).
    """
    states = [s for s in field.tiles.keys() if len(s) == 9]
    # Group by piece count (same turn = same number of pieces for X-to-move)
    by_piece_count = defaultdict(list)
    for s in states:
        n_pieces = 9 - s.count(' ')
        by_piece_count[n_pieces].append(s)

    pairs = []
    for pc, group in by_piece_count.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                sa, sb = group[i], group[j]
                diffs = [k for k in range(9) if sa[k] != sb[k]]
                if len(diffs) == 2:
                    pairs.append((sa, sb, diffs))

    return pairs


def measure_interference(field: TileField, pairs: List[Tuple[str, str, List[int]]]) -> dict:
    """
    For each adjacent pair (same piece count, differ by 2 cells), check if they
    agree on best action. If two very similar boards recommend different moves,
    that's interference — the tile field can't generalize across them.
    Returns interference metrics.
    """
    agree_count = 0
    disagree_count = 0
    unlearned_count = 0
    interference_details = []

    for sa, sb, diff_positions in pairs:
        legal_a = [str(i) for i in range(9) if sa[i] == ' ']
        legal_b = [str(i) for i in range(9) if sb[i] == ' ']

        best_a = field.best_action(sa, legal_a)
        best_b = field.best_action(sb, legal_b)

        if best_a is None or best_b is None:
            unlearned_count += 1
            continue

        if best_a == best_b:
            agree_count += 1
        else:
            # "Opposite" = one state's best action is one of the differing cells
            diff_actions = [str(p) for p in diff_positions]
            is_opposite = (best_a in diff_actions) != (best_b in diff_actions)

            disagree_count += 1
            interference_details.append({
                "state_a": sa,
                "state_b": sb,
                "diff_positions": diff_positions,
                "best_a": best_a,
                "best_b": best_b,
                "is_opposite": is_opposite,
                "visits_a": field.tile_visit_count(sa),
                "visits_b": field.tile_visit_count(sb),
            })

    total = agree_count + disagree_count
    return {
        "total_adjacent_pairs": len(pairs),
        "agree": agree_count,
        "disagree": disagree_count,
        "unlearned": unlearned_count,
        "interference_rate": disagree_count / total if total > 0 else 0,
        "opposite_rate": sum(1 for d in interference_details if d["is_opposite"]) / total if total > 0 else 0,
        "interference_details": interference_details[:50],  # cap for JSON size
    }


def analyze_interference_vs_learning(field: TileField, interference: dict, training_stats: dict) -> dict:
    """Does high interference correlate with visit count? Learning difficulty?"""
    details = interference.get("interference_details", [])
    if not details:
        return {"note": "No interference details to analyze"}

    visits_interfering = []
    visits_agreeing = []

    # For interfering pairs, average visit counts
    for d in details:
        avg_v = (d["visits_a"] + d["visits_b"]) / 2
        visits_interfering.append(avg_v)

    # Sample some non-interfering tiles for comparison
    all_visits = [field.tile_visit_count(s) for s in field.tiles]
    avg_all_visits = np.mean(all_visits) if all_visits else 0
    avg_interfering_visits = np.mean(visits_interfering) if visits_interfering else 0

    return {
        "avg_visits_all_tiles": round(avg_all_visits, 2),
        "avg_visits_interfering_tiles": round(avg_interfering_visits, 2),
        "interference_ratio_visits": round(avg_interfering_visits / avg_all_visits, 2) if avg_all_visits > 0 else 0,
        "final_x_win_rate": training_stats["x_win_rate"],
        "total_tiles_learned": training_stats["num_tiles"],
    }


# ─── Hash Grouping Experiment ─────────────────────────────

def run_hash_grouping_experiment(num_games=500, groupings=[0, 10, 50, 100, 500]) -> List[dict]:
    """Test whether grouping similar states (larger hash buckets) reduces interference."""
    results = []
    for g in groupings:
        print(f"  Training with hash_grouping={g}...")
        field, stats = train_tile_field(num_games=num_games, hash_grouping=g)
        pairs = find_adjacent_state_pairs(field)
        interference = measure_interference(field, pairs)
        analysis = analyze_interference_vs_learning(field, interference, stats)

        results.append({
            "hash_grouping": g,
            "training_stats": stats,
            "interference": {k: v for k, v in interference.items() if k != "interference_details"},
            "analysis": analysis,
        })
    return results


# ─── Main ─────────────────────────────────────────────────

def main():
    random.seed(42)
    np.random.seed(42)

    print("═══ Tile Interference Experiment ═══")
    print()

    # Phase 1: Train baseline tile field
    print("Phase 1: Training baseline tile field (500 games)...")
    t0 = time.time()
    field, training_stats = train_tile_field(num_games=500)
    print(f"  Trained in {time.time()-t0:.1f}s — {training_stats['num_tiles']} tiles, "
          f"X win rate: {training_stats['x_win_rate']:.2%}")

    # Phase 2: Find adjacent state pairs and measure interference
    print("\nPhase 2: Finding adjacent state pairs...")
    pairs = find_adjacent_state_pairs(field)
    print(f"  Found {len(pairs)} adjacent pairs (differ by exactly 1 move)")

    print("  Measuring interference...")
    interference = measure_interference(field, pairs)
    print(f"  Agree: {interference['agree']}, Disagree: {interference['disagree']}, "
          f"Unlearned: {interference['unlearned']}")
    print(f"  Interference rate: {interference['interference_rate']:.2%}")
    print(f"  Opposite action rate: {interference['opposite_rate']:.2%}")

    # Phase 3: Interference vs learning analysis
    print("\nPhase 3: Interference vs learning analysis...")
    analysis = analyze_interference_vs_learning(field, interference, training_stats)
    print(f"  Avg visits (all tiles): {analysis['avg_visits_all_tiles']}")
    print(f"  Avg visits (interfering): {analysis['avg_visits_interfering_tiles']}")
    print(f"  Interference/All visit ratio: {analysis['interference_ratio_visits']}")

    # Phase 4: Hash grouping experiment
    print("\nPhase 4: Hash grouping experiment...")
    grouping_results = run_hash_grouping_experiment(
        num_games=500, groupings=[0, 10, 50, 100, 500]
    )
    for r in grouping_results:
        g = r["hash_grouping"]
        ir = r["interference"]["interference_rate"]
        wr = r["training_stats"]["x_win_rate"]
        nt = r["training_stats"]["num_tiles"]
        print(f"  grouping={g:>4d}: interference={ir:.2%}, X_wr={wr:.2%}, tiles={nt}")

    # Save results
    results = {
        "experiment": "tile_interference",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "baseline": {
            "training_stats": {k: v for k, v in training_stats.items() if k != "learning_curve"},
            "adjacent_pairs": len(pairs),
            "interference": {k: v for k, v in interference.items() if k != "interference_details"},
            "analysis": analysis,
        },
        "hash_grouping_experiment": grouping_results,
        "conclusion": {
            "interference_is_low": interference['interference_rate'] < 0.3,
            "tile_independence_sufficient": interference['interference_rate'] < 0.3,
            "hash_grouping_helps": any(
                r["interference"]["interference_rate"] < interference['interference_rate']
                for r in grouping_results if r["hash_grouping"] > 0
            ),
        }
    }

    out_path = os.path.join(os.path.dirname(__file__), '..', 'tile-interference-results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


import os

if __name__ == "__main__":
    main()
