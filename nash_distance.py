"""
Nash Distance Experiment
========================
Measure how close divergent strategies get to Nash equilibrium in tic-tac-toe.

The core insight: the tile field achieves ~79.4% win rate as X — far from Nash (draw).
But is it converging on the NEGATIVE SPACE (what NOT to do) even while diverging
on the positive space (what TO do)?

Test:
1. Train tile field on TTT (500 games as X)
2. Build a perfect minimax player for ground truth
3. For each reachable state, compare:
   - Tile field's WORST action vs Nash WORST action
   - Tile field's BEST action vs Nash BEST action
4. Measure negative space agreement rate

If negative space agreement > 80%, the system converges to Nash on what NOT to do,
even while diverging on what TO do.
"""

import json
import random
import numpy as np
from collections import defaultdict
from copy import deepcopy
from itertools import product
import time
import os

# ─── Tic-Tac-Toe ──────────────────────────────────────────

class TicTacToe:
    """Tic-tac-toe with full state copying."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [' '] * 9
        self.current = 'X'
        self.turn = 0
        self.done = False
        self.winner = None

    def clone(self):
        g = TicTacToe()
        g.board = self.board[:]
        g.current = self.current
        g.turn = self.turn
        g.done = self.done
        g.winner = self.winner
        return g

    def state_str(self) -> str:
        return ''.join(self.board)

    def state_hash(self) -> str:
        import hashlib
        return hashlib.blake2b(self.state_str().encode(), digest_size=8).hexdigest()

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


# ─── Perfect Minimax Player (Nash Equilibrium) ───────────

class MinimaxOracle:
    """
    Perfect tic-tac-toe solver using minimax with alpha-beta pruning.
    Caches all state evaluations for instant lookup.

    Returns the game-theoretic value for X:
      +1 = X wins with perfect play
       0 = draw with perfect play
      -1 = O wins with perfect play
    """

    def __init__(self):
        self.cache = {}
        self._build_cache()

    def _board_key(self, board, current):
        return (''.join(board), current)

    def _minimax(self, board, current, alpha=-2, beta=2) -> float:
        """Returns value from X's perspective."""
        key = self._board_key(board, current)
        if key in self.cache:
            return self.cache[key]

        # Check terminal
        lines = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
        for a, b, c in lines:
            if board[a] == board[b] == board[c] != ' ':
                val = 1.0 if board[a] == 'X' else -1.0
                self.cache[key] = val
                return val

        if ' ' not in board:
            self.cache[key] = 0.0
            return 0.0

        legal = [i for i in range(9) if board[i] == ' ']

        if current == 'X':
            # X maximizes
            best = -2.0
            for move in legal:
                board[move] = 'X'
                val = self._minimax(board, 'O', alpha, beta)
                board[move] = ' '
                best = max(best, val)
                alpha = max(alpha, val)
                if beta <= alpha:
                    break
            self.cache[key] = best
            return best
        else:
            # O minimizes
            best = 2.0
            for move in legal:
                board[move] = 'O'
                val = self._minimax(board, 'X', alpha, beta)
                board[move] = ' '
                best = min(best, val)
                beta = min(beta, val)
                if beta <= alpha:
                    break
            self.cache[key] = best
            return best

    def _build_cache(self):
        """Pre-compute all reachable states."""
        print("  Building minimax oracle (all reachable states)...")
        board = [' '] * 9
        self._minimax(board, 'X')
        print(f"  Oracle built: {len(self.cache)} states cached")

    def state_value(self, board, current) -> float:
        """Game-theoretic value of state from X's perspective."""
        return self.cache.get(self._board_key(board, current), 0.0)

    def action_values(self, game: TicTacToe) -> dict:
        """
        For each legal action in current state, return the minimax value.
        From X's perspective: +1 = winning move, 0 = drawing, -1 = losing.
        """
        result = {}
        for action_str in game.legal_actions():
            move = int(action_str)
            board = game.board[:]
            board[move] = game.current
            next_player = 'O' if game.current == 'X' else 'X'
            val = self.state_value(board, next_player)
            result[action_str] = val
        return result

    def best_actions(self, game: TicTacToe) -> list:
        """Nash optimal actions (could be multiple if tied)."""
        av = self.action_values(game)
        if game.current == 'X':
            best_val = max(av.values())
        else:
            best_val = min(av.values())
        return [a for a, v in av.items() if v == best_val]

    def worst_actions(self, game: TicTacToe) -> list:
        """Nash worst actions (the action(s) that lose against perfect play)."""
        av = self.action_values(game)
        if game.current == 'X':
            worst_val = min(av.values())
        else:
            worst_val = max(av.values())
        return [a for a, v in av.items() if v == worst_val]


# ─── Tile Field (simplified from zeroclaw.py) ─────────────

class StateTile:
    """A tile representing one game state with scored reflexes."""

    def __init__(self, state_hash, state_str, actions):
        self.state_hash = state_hash
        self.state_str = state_str
        self.reflexes = {
            a: {"score": 0.5, "chosen": 0, "won": 0} for a in actions
        }
        self.entropy = 1.0

    def best_action(self, legal_actions, n_simulations=20, game=None):
        if not legal_actions:
            return ''
        if len(legal_actions) == 1:
            return legal_actions[0]
        for a in legal_actions:
            if a not in self.reflexes:
                self.reflexes[a] = {"score": 0.5, "chosen": 0, "won": 0}

        action_values = {}
        for action in legal_actions:
            sim_wins = 0
            sims_per = max(1, n_simulations // len(legal_actions))
            if game is not None:
                for _ in range(sims_per):
                    winner = self._simulate_playout(game, action)
                    if winner == 'X':
                        sim_wins += 1
            sim_score = sim_wins / max(sims_per, 1)
            learned_score = self.reflexes[action]["score"]
            n_chosen = self.reflexes[action]["chosen"]
            confidence = min(n_chosen / 20.0, 0.8)
            action_values[action] = confidence * learned_score + (1 - confidence) * sim_score

        actions_list = list(action_values.keys())
        values = np.array([action_values[a] for a in actions_list])
        temp = 0.3
        exp_vals = np.exp(values / temp)
        probs = exp_vals / exp_vals.sum()
        return np.random.choice(actions_list, p=probs)

    def record(self, action, won):
        if action in self.reflexes:
            self.reflexes[action]["chosen"] += 1
            if won:
                self.reflexes[action]["won"] += 1

    def evolve(self):
        for action, data in self.reflexes.items():
            if data["chosen"] > 0:
                wr = data["won"] / data["chosen"]
                data["score"] += 0.05 * (wr - data["score"])
                data["score"] = max(0.05, min(0.95, data["score"]))

    def _simulate_playout(self, real_game, first_action):
        g = real_game.clone()
        g.step(first_action)
        while not g.done:
            actions = g.legal_actions()
            if not actions:
                break
            g.step(random.choice(actions))
        return g.winner

    def tile_worst_action(self, legal_actions):
        """The action the tile field thinks is worst (lowest score)."""
        if not legal_actions:
            return None
        if len(legal_actions) == 1:
            return legal_actions[0]
        scored = {a: self.reflexes.get(a, {}).get("score", 0.5) for a in legal_actions}
        return min(scored, key=scored.get)

    def tile_best_action(self, legal_actions):
        """The action the tile field thinks is best (highest score)."""
        if not legal_actions:
            return None
        if len(legal_actions) == 1:
            return legal_actions[0]
        scored = {a: self.reflexes.get(a, {}).get("score", 0.5) for a in legal_actions}
        return max(scored, key=scored.get)

    def tile_action_ranking(self, legal_actions):
        """Return actions ranked by tile field score (best first)."""
        scored = [(a, self.reflexes.get(a, {}).get("score", 0.5)) for a in legal_actions]
        scored.sort(key=lambda x: -x[1])
        return scored


# ─── Tile Field Training ──────────────────────────────────

def train_tile_field(num_games=500, n_simulations=20, evolve_every=25):
    """Train tile field on tic-tac-toe as X, O plays random."""
    print(f"\n{'='*60}")
    print(f"TRAINING TILE FIELD: {num_games} games (X=tile, O=random)")
    print(f"{'='*60}")

    game = TicTacToe()
    tiles = {}  # state_hash -> StateTile
    stats = {"wins": 0, "losses": 0, "draws": 0, "total": 0}

    for i in range(num_games):
        game.reset()
        history = []

        while not game.done:
            state = game.state_str()
            shash = game.state_hash()
            actions = game.legal_actions()
            if not actions:
                break

            if shash not in tiles:
                tiles[shash] = StateTile(shash, state, actions)
            tile = tiles[shash]

            if game.current == 'X':
                action = tile.best_action(actions, n_simulations, game)
            else:
                action = random.choice(actions)

            game.step(action)
            history.append((shash, action, game.current))

        stats["total"] += 1
        won = game.winner == 'X'
        lost = game.winner == 'O'
        if won:
            stats["wins"] += 1
        elif lost:
            stats["losses"] += 1
        else:
            stats["draws"] += 1

        # Record outcomes for X moves only
        for shash, action, player in history:
            if player == 'X' and shash in tiles:
                tiles[shash].record(action, won)

        if (i + 1) % evolve_every == 0:
            for tile in tiles.values():
                tile.evolve()

        if (i + 1) % 100 == 0:
            wr = stats["wins"] / stats["total"]
            print(f"  {i+1}/{num_games} | W={stats['wins']} L={stats['losses']} D={stats['draws']} "
                  f"| WR={wr:.1%} | tiles={len(tiles)}")

    wr = stats["wins"] / stats["total"]
    print(f"\n  Final: W={stats['wins']} L={stats['losses']} D={stats['draws']} "
          f"| WR={wr:.1%} | tiles={len(tiles)}")
    return tiles, stats


# ─── Nash Distance Measurement ────────────────────────────

def enumerate_reachable_states():
    """Enumerate all unique board states where it's X's turn."""
    oracle = MinimaxOracle()
    seen = set()
    states = []

    def dfs(board, current, depth):
        key = (''.join(board), current)
        if key in seen:
            return
        seen.add(key)

        # Check terminal
        lines = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
        for a, b, c in lines:
            if board[a] == board[b] == board[c] != ' ':
                return
        if ' ' not in board:
            return

        if current == 'X':
            game = TicTacToe()
            game.board = board[:]
            game.current = 'X'
            game.turn = depth
            states.append(game)

        # Expand
        for i in range(9):
            if board[i] == ' ':
                board[i] = current
                dfs(board, 'O' if current == 'X' else 'X', depth + 1)
                board[i] = ' '

    dfs([' '] * 9, 'X', 0)
    return oracle, states


def measure_nash_distance(tiles, oracle, x_states):
    """
    Compare tile field vs Nash on all reachable X-turn states.

    Returns metrics for both positive space (best action) and
    negative space (worst action).
    """
    print(f"\n{'='*60}")
    print("MEASURING NASH DISTANCE")
    print(f"{'='*60}")

    results = {
        "states_evaluated": 0,
        "states_with_tiles": 0,
        # Negative space: worst action agreement
        "negative_agreement": 0,       # tile worst == nash worst
        "negative_overlap": 0,         # tile worst in nash worst set
        # Positive space: best action agreement
        "positive_agreement": 0,       # tile best == nash best
        "positive_overlap": 0,         # tile best in nash best set
        # Ranking correlation
        "spearman_correlations": [],
        "kendall_correlations": [],
        # Per-action value analysis
        "value_errors": [],            # |tile_score - nash_value| per action
        # Detailed per-state info
        "state_details": [],
    }

    for game in x_states:
        actions = game.legal_actions()
        if len(actions) <= 1:
            continue  # trivial state

        shash = game.state_hash()
        results["states_evaluated"] += 1

        # Nash ground truth
        nash_av = oracle.action_values(game)
        nash_best = oracle.best_actions(game)
        nash_worst = oracle.worst_actions(game)

        # Tile field assessment
        tile = tiles.get(shash)
        if tile is None:
            continue
        results["states_with_tiles"] += 1

        tile_worst = tile.tile_worst_action(actions)
        tile_best = tile.tile_best_action(actions)
        tile_ranking = tile.tile_action_ranking(actions)

        # Negative space agreement
        if tile_worst in nash_worst:
            results["negative_agreement"] += 1
            results["negative_overlap"] += 1
        # Also check partial overlap (if nash_worst is a set)
        # Already handled above since we check membership

        # Positive space agreement
        if tile_best in nash_best:
            results["positive_agreement"] += 1
            results["positive_overlap"] += 1

        # Ranking correlation (Spearman)
        nash_ranked = sorted(nash_av.items(), key=lambda x: -x[1])
        tile_rank_dict = {a: s for a, s in tile_ranking}

        # Compute Spearman rank correlation
        try:
            from scipy.stats import spearmanr, kendalltau
            # Align actions
            all_actions = sorted(actions)
            nash_vals = [nash_av.get(a, 0) for a in all_actions]
            tile_vals = [tile.reflexes.get(a, {}).get("score", 0.5) for a in all_actions]

            if len(all_actions) > 1:
                sp_corr, _ = spearmanr(nash_vals, tile_vals)
                kt_corr, _ = kendalltau(nash_vals, tile_vals)
                if not np.isnan(sp_corr):
                    results["spearman_correlations"].append(sp_corr)
                if not np.isnan(kt_corr):
                    results["kendall_correlations"].append(kt_corr)
        except ImportError:
            # Manual Spearman
            all_actions = sorted(actions)
            nash_vals = [nash_av.get(a, 0) for a in all_actions]
            tile_vals = [tile.reflexes.get(a, {}).get("score", 0.5) for a in all_actions]
            if len(all_actions) > 1:
                # Rank them
                def rank(vals):
                    sorted_idx = sorted(range(len(vals)), key=lambda i: vals[i])
                    ranks = [0] * len(vals)
                    for r, i in enumerate(sorted_idx):
                        ranks[i] = r
                    return ranks

                nr = rank(nash_vals)
                tr = rank(tile_vals)
                n = len(nr)
                d_sq = sum((a - b) ** 2 for a, b in zip(nr, tr))
                sp = 1 - 6 * d_sq / (n * (n**2 - 1)) if n > 1 else 0
                results["spearman_correlations"].append(sp)

        # Value errors
        for a in actions:
            nash_val = nash_av.get(a, 0)
            tile_val = tile.reflexes.get(a, {}).get("score", 0.5)
            results["value_errors"].append(abs(tile_val - (nash_val + 1) / 2))  # normalize nash to [0,1]

        # Store details for interesting states
        if results["states_evaluated"] <= 20 or tile_worst not in nash_worst:
            results["state_details"].append({
                "board": game.board[:],
                "nash_values": {a: round(v, 3) for a, v in nash_av.items()},
                "tile_scores": {a: round(tile.reflexes.get(a, {}).get("score", 0.5), 3)
                                for a in actions},
                "nash_best": nash_best,
                "nash_worst": nash_worst,
                "tile_best": tile_best,
                "tile_worst": tile_worst,
                "negative_match": tile_worst in nash_worst,
                "positive_match": tile_best in nash_best,
            })

    return results


def print_results(results):
    """Print formatted results."""
    n = results["states_with_tiles"]
    total = results["states_evaluated"]

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  Total X-turn states: {total}")
    print(f"  States with trained tiles: {n} ({n/total:.1%})")

    if n == 0:
        print("  No overlapping states — nothing to compare.")
        return

    neg_rate = results["negative_agreement"] / n
    pos_rate = results["positive_agreement"] / n
    avg_spearman = np.mean(results["spearman_correlations"]) if results["spearman_correlations"] else 0
    avg_kendall = np.mean(results["kendall_correlations"]) if results["kendall_correlations"] else 0
    avg_val_err = np.mean(results["value_errors"]) if results["value_errors"] else 0

    print(f"\n  ┌─────────────────────────────────────────────────────┐")
    print(f"  │ NEGATIVE SPACE (worst action agreement): {neg_rate:.1%}     │")
    print(f"  │ POSITIVE SPACE (best action agreement):  {pos_rate:.1%}     │")
    print(f"  │ Spearman rank correlation:                {avg_spearman:+.3f}   │")
    print(f"  │ Kendall rank correlation:                 {avg_kendall:+.3f}   │")
    print(f"  │ Mean value error (normalized):            {avg_val_err:.3f}    │")
    print(f"  └─────────────────────────────────────────────────────┘")

    print(f"\n  INTERPRETATION:")
    if neg_rate > 0.8:
        print(f"  ✅ Negative space agreement {neg_rate:.1%} > 80%")
        print(f"     CONFIRMED: Tile field converges to Nash on what NOT to do")
        print(f"     even while diverging on what TO do ({pos_rate:.1%} positive agreement)")
    elif neg_rate > 0.6:
        print(f"  ⚠️  Negative space agreement {neg_rate:.1%} — moderate, not conclusive")
        print(f"     Some convergence on negative space but below 80% threshold")
    else:
        print(f"  ❌ Negative space agreement {neg_rate:.1%} < 60%")
        print(f"     No evidence of negative space convergence to Nash")

    print(f"\n  SAMPLE MISMATCHES:")
    mismatches = [d for d in results["state_details"] if not d["negative_match"]]
    for d in mismatches[:5]:
        board = d["board"]
        print(f"  Board: {board[0]}|{board[1]}|{board[2]} / {board[3]}|{board[4]}|{board[5]} / {board[6]}|{board[7]}|{board[8]}")
        print(f"    Nash worst: {d['nash_worst']} | Tile worst: {d['tile_worst']}")
        print(f"    Nash vals: {d['nash_values']}")
        print(f"    Tile scores: {d['tile_scores']}")
        print()


def save_results(results, stats, filename="nash-distance-results.json"):
    """Save results to JSON."""
    n = results["states_with_tiles"]

    output = {
        "experiment": "nash_distance",
        "description": "Measure how close tile field strategies get to Nash equilibrium on negative space",
        "training": stats,
        "metrics": {
            "states_evaluated": results["states_evaluated"],
            "states_with_tiles": results["states_with_tiles"],
            "negative_agreement_rate": results["negative_agreement"] / n if n else 0,
            "positive_agreement_rate": results["positive_agreement"] / n if n else 0,
            "spearman_correlation_mean": float(np.mean(results["spearman_correlations"])) if results["spearman_correlations"] else 0,
            "spearman_correlation_std": float(np.std(results["spearman_correlations"])) if results["spearman_correlations"] else 0,
            "kendall_correlation_mean": float(np.mean(results["kendall_correlations"])) if results["kendall_correlations"] else 0,
            "mean_value_error": float(np.mean(results["value_errors"])) if results["value_errors"] else 0,
            "negative_hypothesis_confirmed": (results["negative_agreement"] / n > 0.8) if n else False,
        },
        "sample_mismatches": [d for d in results["state_details"] if not d["negative_match"]][:10],
        "sample_matches": [d for d in results["state_details"] if d["negative_match"]][:5],
    }

    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {filename}")
    return output


# ─── Extended Analysis: Per-Depth Breakdown ───────────────

def analyze_by_depth(tiles, oracle):
    """Break down agreement by game depth (early vs late game)."""
    print(f"\n{'='*60}")
    print("ANALYSIS BY GAME DEPTH")
    print(f"{'='*60}")

    depth_buckets = defaultdict(lambda: {
        "count": 0, "neg_agree": 0, "pos_agree": 0, "spearman": []
    })

    game = TicTacToe()
    seen = set()

    def dfs(board, current, depth):
        key = (''.join(board), current)
        if key in seen:
            return
        seen.add(key)

        lines = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
        for a, b, c in lines:
            if board[a] == board[b] == board[c] != ' ':
                return
        if ' ' not in board:
            return

        if current == 'X':
            g = TicTacToe()
            g.board = board[:]
            g.current = 'X'
            g.turn = depth
            actions = g.legal_actions()

            if len(actions) > 1:
                shash = g.state_hash()
                tile = tiles.get(shash)
                if tile:
                    nash_av = oracle.action_values(g)
                    nash_worst = oracle.worst_actions(g)
                    nash_best = oracle.best_actions(game=g)
                    tile_worst = tile.tile_worst_action(actions)
                    tile_best = tile.tile_best_action(actions)

                    bucket = depth_buckets[depth]
                    bucket["count"] += 1
                    if tile_worst in nash_worst:
                        bucket["neg_agree"] += 1
                    if tile_best in nash_best:
                        bucket["pos_agree"] += 1

        for i in range(9):
            if board[i] == ' ':
                board[i] = current
                dfs(board, 'O' if current == 'X' else 'X', depth + 1)
                board[i] = ' '

    dfs([' '] * 9, 'X', 0)

    print(f"\n  {'Depth':>6} {'States':>8} {'Neg Agree':>12} {'Pos Agree':>12}")
    print(f"  {'-'*6} {'-'*8} {'-'*12} {'-'*12}")
    for depth in sorted(depth_buckets.keys()):
        b = depth_buckets[depth]
        n = b["count"]
        neg = b["neg_agree"] / n if n else 0
        pos = b["pos_agree"] / n if n else 0
        print(f"  {depth:>6} {n:>8} {neg:>11.1%} {pos:>11.1%}")

    return dict(depth_buckets)


# ─── Main ─────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   NASH DISTANCE EXPERIMENT                              ║")
    print("║   Measuring negative space convergence to equilibrium   ║")
    print("╚══════════════════════════════════════════════════════════╝")

    start = time.time()

    # Step 1: Build minimax oracle
    print("\n[1/4] Building minimax oracle (perfect play ground truth)...")
    oracle = MinimaxOracle()
    print(f"  Total states in oracle: {len(oracle.cache)}")

    # Step 2: Train tile field
    print("\n[2/4] Training tile field on 500 games as X...")
    tiles, stats = train_tile_field(num_games=500, n_simulations=20)

    # Step 3: Enumerate all X-turn states
    print("\n[3/4] Enumerating all reachable X-turn states...")
    oracle2, x_states = enumerate_reachable_states()
    print(f"  Found {len(x_states)} unique X-turn states")

    # Step 4: Measure Nash distance
    print("\n[4/4] Measuring Nash distance...")
    results = measure_nash_distance(tiles, oracle, x_states)
    print_results(results)

    # Depth analysis
    analyze_by_depth(tiles, oracle)

    # Save
    output = save_results(results, stats)

    elapsed = time.time() - start
    print(f"\n  Total time: {elapsed:.1f}s")

    print(f"\n{'='*60}")
    print("CONCLUSION")
    print(f"{'='*60}")
    neg = output["metrics"]["negative_agreement_rate"]
    pos = output["metrics"]["positive_agreement_rate"]
    print(f"  Negative space agreement: {neg:.1%}")
    print(f"  Positive space agreement: {pos:.1%}")
    print(f"  Hypothesis (neg > 80%): {'CONFIRMED ✅' if output['metrics']['negative_hypothesis_confirmed'] else 'NOT CONFIRMED ❌'}")


if __name__ == "__main__":
    main()
