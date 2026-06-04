"""
Holographic Transfer — Can the negative space from one game accelerate learning in another?

If the holographic principle holds, then the NEGATIVE SPACE from tic-tac-toe
should help learn Connect4 faster. Not the specific strategies (those are degenerate),
but the SHAPE of the decision landscape (what bad looks like).

Test:
1. Train tile field on TTT (source)
2. Extract negative space (bottom 25% of scores per tile)
3. Start learning C4 (target) WITH the TTT negative space as prior
4. Compare: C4 with TTT prior vs C4 from scratch

If the negative space transfers, the prior-assisted C4 should converge faster
(even though TTT and C4 are different games).
"""

import random
import numpy as np
import hashlib
import json
import os
import time
from collections import defaultdict

from zeroclaw import TicTacToe, Connect4


class TransferTileField:
    """Tile field that can receive negative space from another field."""
    
    def __init__(self, source_negative_space=None, transfer_weight=0.3):
        self.tiles = {}
        self.source_negative_space = source_negative_space  # {(stage_features): {"bad_actions": [...], "neg_scores": {...}}}
        self.transfer_weight = transfer_weight
    
    def get_or_create(self, state_str, legal_actions):
        h = hashlib.blake2b(state_str.encode(), digest_size=8).hexdigest()
        if h not in self.tiles:
            self.tiles[h] = {
                "state": state_str[:80],
                "actions": {a: {"score": 0.5, "chosen": 0, "won": 0} for a in legal_actions}
            }
            # Apply negative space transfer if available
            if self.source_negative_space and self.transfer_weight > 0:
                # Find closest source tile by action count similarity
                n_actions = len(legal_actions)
                for src_key, src_data in self.source_negative_space.items():
                    if abs(src_data.get("n_actions", 0) - n_actions) <= 1:
                        # Transfer: actions that were bad in source → lower initial score here
                        for bad_action_idx, bad_score in src_data.get("neg_scores", {}).items():
                            if int(bad_action_idx) < len(legal_actions):
                                action = legal_actions[int(bad_action_idx)]
                                current = self.tiles[h]["actions"][action]["score"]
                                # Blend: mostly keep 0.5 but pull toward source's negative signal
                                transferred = current * (1 - self.transfer_weight) + bad_score * self.transfer_weight
                                self.tiles[h]["actions"][action]["score"] = max(0.05, min(0.95, transferred))
                        break  # Only use first matching source tile
        else:
            for a in legal_actions:
                if a not in self.tiles[h]["actions"]:
                    self.tiles[h]["actions"][a] = {"score": 0.5, "chosen": 0, "won": 0}
        return h
    
    def choose(self, h, actions, T=0.3, eps=0.05):
        tile = self.tiles[h]
        if random.random() < eps:
            return min(actions, key=lambda a: tile["actions"].get(a, {}).get("chosen", 0))
        scores = np.array([tile["actions"].get(a, {}).get("score", 0.5) for a in actions])
        if T > 0.01:
            p = np.exp(scores / T); p /= p.sum()
            return actions[np.random.choice(len(actions), p=p)]
        return actions[np.argmax(scores)]
    
    def record(self, h, action, won):
        if h in self.tiles and action in self.tiles[h]["actions"]:
            self.tiles[h]["actions"][action]["chosen"] += 1
            if won: self.tiles[h]["actions"][action]["won"] += 1
    
    def evolve(self, lr=0.05, cap=0.05):
        for tile in self.tiles.values():
            for d in tile["actions"].values():
                if d["chosen"] > 0:
                    wr = d["won"] / d["chosen"]
                    delta = max(-cap, min(cap, lr * (wr - d["score"])))
                    d["score"] = max(0.05, min(0.95, d["score"] + delta))
    
    def extract_negative_space(self):
        """Extract the negative space (bottom 25% scores per tile)."""
        neg_space = {}
        for h, tile in self.tiles.items():
            actions_sorted = sorted(tile["actions"].items(), key=lambda x: x[1]["score"])
            n = max(1, len(actions_sorted) // 4)
            neg_scores = {str(i): actions_sorted[i][1]["score"] for i in range(n)}
            neg_space[h] = {
                "n_actions": len(actions_sorted),
                "neg_scores": neg_scores,
            }
        return neg_space
    
    def stats(self):
        all_scores = [d["score"] for t in self.tiles.values() for d in t["actions"].values()]
        return {
            "tiles": len(self.tiles),
            "avg_score": np.mean(all_scores) if all_scores else 0.5,
            "score_std": np.std(all_scores) if all_scores else 0,
            "score_range": f"{min(all_scores):.2f}-{max(all_scores):.2f}" if all_scores else "N/A",
        }


def play_games(field, GameClass, n_games, temperature=0.3):
    """Play n_games and return win rate."""
    wins = 0
    for _ in range(n_games):
        game = GameClass()
        history = []
        while not game.done:
            actions = game.legal_actions()
            if not actions: break
            if game.current == 'X':
                h = field.get_or_create(str(game.state()), actions)
                a = field.choose(h, actions, temperature)
                history.append((h, a))
            else:
                a = random.choice(actions)
            game.step(a)
        won = getattr(game, 'winner', None) == 'X'
        if won: wins += 1
        for h, a in history:
            field.record(h, a, won)
    return wins / n_games


def run_transfer_experiment():
    print("=" * 70)
    print("HOLOGRAPHIC TRANSFER — Negative Space Transfer Between Games")
    print("=" * 70)
    
    # Step 1: Train source field on TTT
    print("\n--- Step 1: Training source on TTT (500 games) ---")
    source_field = TransferTileField(transfer_weight=0)
    for batch in range(5):
        wr = play_games(source_field, TicTacToe, 100, temperature=0.4)
        source_field.evolve()
        print(f"  TTT batch {batch+1}: wr={wr:.1%} tiles={len(source_field.tiles)}")
    
    # Extract negative space
    neg_space = source_field.extract_negative_space()
    print(f"  Extracted negative space: {len(neg_space)} tiles")
    
    # Step 2: Train C4 from scratch (baseline)
    print("\n--- Step 2: Training C4 from scratch (10 rounds × 100 games) ---")
    c4_scratch = TransferTileField(transfer_weight=0)
    scratch_results = []
    
    for rnd in range(10):
        T = max(0.15, 0.5 - rnd * 0.04)
        wr = play_games(c4_scratch, Connect4, 100, temperature=T)
        c4_scratch.evolve()
        scratch_results.append(wr)
        stats = c4_scratch.stats()
        print(f"  Round {rnd+1}: wr={wr:.1%} tiles={stats['tiles']} "
              f"scores=[{stats['score_range']}]")
    
    # Step 3: Train C4 WITH TTT negative space (transfer)
    print("\n--- Step 3: Training C4 with TTT negative space (10 rounds × 100 games) ---")
    c4_transfer = TransferTileField(source_negative_space=neg_space, transfer_weight=0.3)
    transfer_results = []
    
    for rnd in range(10):
        T = max(0.15, 0.5 - rnd * 0.04)
        wr = play_games(c4_transfer, Connect4, 100, temperature=T)
        c4_transfer.evolve()
        transfer_results.append(wr)
        stats = c4_transfer.stats()
        print(f"  Round {rnd+1}: wr={wr:.1%} tiles={stats['tiles']} "
              f"scores=[{stats['score_range']}]")
    
    # Step 4: Train C4 with INVERTED negative space (anti-transfer)
    print("\n--- Step 4: Training C4 with INVERTED TTT negative space (anti-transfer) ---")
    inverted_neg = {}
    for k, v in neg_space.items():
        inverted_neg[k] = {
            "n_actions": v["n_actions"],
            "neg_scores": {idx: 1.0 - score for idx, score in v["neg_scores"].items()},
        }
    
    c4_anti = TransferTileField(source_negative_space=inverted_neg, transfer_weight=0.3)
    anti_results = []
    
    for rnd in range(10):
        T = max(0.15, 0.5 - rnd * 0.04)
        wr = play_games(c4_anti, Connect4, 100, temperature=T)
        c4_anti.evolve()
        anti_results.append(wr)
        stats = c4_anti.stats()
        print(f"  Round {rnd+1}: wr={wr:.1%} tiles={stats['tiles']} "
              f"scores=[{stats['score_range']}]")
    
    # Results
    print(f"\n{'=' * 70}")
    print("HOLOGRAPHIC TRANSFER RESULTS")
    print(f"{'=' * 70}")
    
    # Cumulative win rates
    scratch_cum = [np.mean(scratch_results[:i+1]) for i in range(10)]
    transfer_cum = [np.mean(transfer_results[:i+1]) for i in range(10)]
    anti_cum = [np.mean(anti_results[:i+1]) for i in range(10)]
    
    print(f"\n  {'Round':>6s} {'Scratch':>10s} {'Transfer':>10s} {'Anti':>10s} {'Δ Transfer':>12s}")
    for i in range(10):
        delta = (transfer_cum[i] - scratch_cum[i]) * 100
        print(f"  {i+1:>6d} {scratch_cum[i]:>9.1%} {transfer_cum[i]:>9.1%} {anti_cum[i]:>9.1%} {delta:>+11.1f}pp")
    
    final_scratch = scratch_cum[-1]
    final_transfer = transfer_cum[-1]
    final_anti = anti_cum[-1]
    
    print(f"\n  Final Scratch:  {final_scratch:.1%}")
    print(f"  Final Transfer: {final_transfer:.1%} ({(final_transfer-final_scratch)*100:+.1f}pp)")
    print(f"  Final Anti:     {final_anti:.1%} ({(final_anti-final_scratch)*100:+.1f}pp)")
    
    # Save
    output = {
        "scratch_final": final_scratch,
        "transfer_final": final_transfer,
        "anti_final": final_anti,
        "transfer_advantage_pp": (final_transfer - final_scratch) * 100,
        "anti_penalty_pp": (final_anti - final_scratch) * 100,
        "scratch_per_round": scratch_results,
        "transfer_per_round": transfer_results,
        "anti_per_round": anti_results,
    }
    out = os.path.expanduser("~/repos/zeroclaw-arena/holographic-transfer-results.json")
    with open(out, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    run_transfer_experiment()
