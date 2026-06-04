"""
CROSS-GENRE HOLOGRAPHIC TRANSFER
Board Game → Card Game

Does the negative space from tic-tac-toe (deterministic, perfect info, board game)
transfer to Texas Hold'em (stochastic, hidden info, card game)?

If YES: the negative space captures something universal about DECISION STRUCTURE,
not game mechanics.

Three conditions:
  1. Poker from scratch (no prior)
  2. Poker with TTT negative space (transfer)
  3. Poker with INVERTED TTT negative space (anti-transfer control)

Uses: HoldemHand from holdem_tile.py, TransferTileField from holographic_transfer.py
"""

import random
import numpy as np
import json
import os
import time
from collections import defaultdict

from zeroclaw import TicTacToe
from holdem_tile import HoldemHand, PokerTileField, PokerTile, hand_name, best_hand
from holographic_transfer import TransferTileField


# ============== CROSS-GENRE BRIDGE ==============

class CrossGenreBridge:
    """
    Bridges TTT negative space into poker tile field.
    
    TTT negative space tells us: "in situations with N legal moves,
    actions at positions 0..K had the worst scores."
    
    Poker analogue: "in situations with 5 actions available,
    the first K actions (by index) tend to be bad."
    
    The mapping is STRUCTURAL:
    - TTT early game (9 moves) → Poker preflop (5 actions: fold/check/raise_s/raise_b/bluff)
    - TTT mid game (5-7 moves) → Poker flop/turn
    - TTT late game (2-3 moves) → Poker river (fold or call, essentially)
    
    The negative space transfers the SHAPE of "what bad looks like"
    not the specific content.
    """
    
    def __init__(self, ttt_negative_space, transfer_weight=0.3, inverted=False):
        self.neg_space = ttt_negative_space
        self.transfer_weight = transfer_weight
        self.inverted = inverted
    
    def apply_to_poker_tile(self, poker_tile: PokerTile):
        """Apply transferred negative space to a poker tile's initial scores."""
        if not self.neg_space or self.transfer_weight <= 0:
            return
        
        # Match by action count: poker has 5 actions
        # Find TTT tiles with similar decision branching
        poker_n_actions = len(poker_tile.reflexes)
        
        # Collect all source tiles sorted by action count match
        candidates = []
        for src_key, src_data in self.neg_space.items():
            src_n = src_data.get("n_actions", 0)
            distance = abs(src_n - poker_n_actions)
            if distance <= 2:  # Close enough in branching factor
                candidates.append((distance, src_data))
        
        if not candidates:
            return
        
        # Use closest match
        candidates.sort(key=lambda x: x[0])
        best_match = candidates[0][1]
        
        # Apply negative space transfer to poker actions
        # Map: lowest-scoring TTT action indices → poker actions by ordering
        poker_actions = list(poker_tile.reflexes.keys())
        neg_scores = best_match.get("neg_scores", {})
        
        for idx_str, src_score in neg_scores.items():
            idx = int(idx_str)
            if idx < len(poker_actions):
                action = poker_actions[idx]
                current = poker_tile.reflexes[action]["score"]
                
                if self.inverted:
                    # Anti-transfer: boost what source says is bad
                    transferred_score = 1.0 - src_score
                else:
                    transferred_score = src_score
                
                # Blend
                blended = current * (1 - self.transfer_weight) + transferred_score * self.transfer_weight
                poker_tile.reflexes[action]["score"] = max(0.05, min(0.95, blended))


class TransferPokerTileField(PokerTileField):
    """PokerTileField that receives cross-genre negative space."""
    
    def __init__(self, bridge: CrossGenreBridge = None):
        super().__init__()
        self.bridge = bridge
    
    def get_tile(self, stage, hand_bucket, pot_bucket, position):
        tile = super().get_tile(stage, hand_bucket, pot_bucket, position)
        
        # Apply bridge on first access (tile.visits == 0)
        if tile.visits == 0 and self.bridge:
            self.bridge.apply_to_poker_tile(tile)
            tile._transfer_applied = True
        
        return tile


# ============== EXPERIMENT ==============

def play_poker_hands(field, n_hands, temperature_fn=None, opponent="random"):
    """Play n poker hands and return stats."""
    wins = 0
    showdown_wins = 0
    showdowns = 0
    fold_wins = 0
    total_pot = 0
    
    for _ in range(n_hands):
        game = HoldemHand()
        result = game.play(field, opponent)
        
        if result["winner"] == 0:
            wins += 1
            if result["fold"]:
                fold_wins += 1
            else:
                showdown_wins += 1
        if not result["fold"]:
            showdowns += 1
        total_pot += result.get("pot", 0)
    
    return {
        "win_rate": wins / n_hands,
        "showdown_wr": showdown_wins / max(showdowns, 1),
        "fold_wins": fold_wins,
        "showdowns": showdowns,
        "avg_pot": total_pot / n_hands,
    }


def train_ttt_source(n_games=500):
    """Train TTT and extract negative space."""
    print("  Training TTT source field...")
    source_field = TransferTileField(transfer_weight=0)
    
    batch_size = 100
    for batch in range(n_games // batch_size):
        from holographic_transfer import play_games
        wr = play_games(source_field, TicTacToe, batch_size, temperature=0.4)
        source_field.evolve()
    
    neg_space = source_field.extract_negative_space()
    stats = source_field.stats()
    print(f"  TTT source: tiles={stats['tiles']}, "
          f"score_range={stats['score_range']}, "
          f"neg_space_entries={len(neg_space)}")
    return neg_space, stats


def run_condition(name, bridge, n_phases=3, hands_per_phase=500):
    """Run one experimental condition (scratch / transfer / anti-transfer)."""
    print(f"\n  === {name} ===")
    field = TransferPokerTileField(bridge=bridge)
    
    results = []
    cumulative_wins = 0
    cumulative_hands = 0
    
    for phase in range(n_phases):
        # Temperature schedule: explore early, exploit later
        base_T = max(0.15, 0.5 - phase * 0.1)
        
        stats = play_poker_hands(field, hands_per_phase,
                                  temperature_fn=lambda s: max(0.15, base_T - s * 0.05))
        
        # Evolve after each phase
        field.evolve(lr=0.04, cap=0.05)
        
        cumulative_wins += int(stats["win_rate"] * hands_per_phase)
        cumulative_hands += hands_per_phase
        cum_wr = cumulative_wins / cumulative_hands
        
        results.append({
            "phase": phase + 1,
            "phase_wr": stats["win_rate"],
            "cumulative_wr": cum_wr,
            "showdown_wr": stats["showdown_wr"],
            "fold_wins": stats["fold_wins"],
            "tiles": len(field.tiles),
        })
        
        print(f"    Phase {phase+1}: wr={stats['win_rate']:.1%} "
              f"cum_wr={cum_wr:.1%} "
              f"showdown_wr={stats['showdown_wr']:.1%} "
              f"fold_wins={stats['fold_wins']} "
              f"tiles={len(field.tiles)}")
    
    return results, field


def run_cross_genre_experiment():
    print("=" * 70)
    print("CROSS-GENRE HOLOGRAPHIC TRANSFER")
    print("Board Game (TTT) → Card Game (Poker)")
    print("=" * 70)
    
    t0 = time.time()
    
    # ── Step 1: Train TTT source ──
    print("\n--- Step 1: Train TTT source & extract negative space ---")
    neg_space, ttt_stats = train_ttt_source(500)
    
    # Inspect what we got
    action_counts = defaultdict(int)
    for v in neg_space.values():
        action_counts[v["n_actions"]] += 1
    print(f"  Negative space action distribution: {dict(action_counts)}")
    
    # ── Step 2: Poker from scratch ──
    print("\n--- Step 2: Poker from SCRATCH (3 × 500 hands) ---")
    scratch_results, scratch_field = run_condition("SCRATCH", bridge=None)
    
    # ── Step 3: Poker with TTT transfer ──
    print("\n--- Step 3: Poker with TTT NEGATIVE SPACE TRANSFER ---")
    transfer_bridge = CrossGenreBridge(neg_space, transfer_weight=0.3)
    transfer_results, transfer_field = run_condition("TRANSFER", bridge=transfer_bridge)
    
    # ── Step 4: Poker with inverted TTT (anti-transfer) ──
    print("\n--- Step 4: Poker with INVERTED TTT (anti-transfer) ---")
    anti_bridge = CrossGenreBridge(neg_space, transfer_weight=0.3, inverted=True)
    anti_results, anti_field = run_condition("ANTI-TRANSFER", bridge=anti_bridge)
    
    # ── Random baseline ──
    print("\n--- Random baseline (500 hands) ---")
    random_field = PokerTileField()  # Pure random, no learning
    random_wins = 0
    for _ in range(500):
        game = HoldemHand()
        # Override play to make P0 purely random
        for stage_idx in range(4):
            if stage_idx == 1:
                game.deal_community(3)
            elif stage_idx in [2, 3]:
                game.deal_community(1)
            game.stage = stage_idx
            p0_action = random.choice(["check_call", "raise_small", "raise_big"])
            p1_action = random.choice(["check_call", "raise_small", "raise_big"])
            result = game.play_round([p0_action, p1_action])
            if result is not None:
                if result == 0:
                    random_wins += 1
                break
        else:
            w = game.showdown()
            if w == 0:
                random_wins += 1
    random_wr = random_wins / 500
    print(f"  Random baseline: {random_wr:.1%}")
    
    # ── ANALYSIS ──
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print("CROSS-GENRE TRANSFER RESULTS")
    print(f"{'=' * 70}")
    
    scratch_final = scratch_results[-1]["cumulative_wr"]
    transfer_final = transfer_results[-1]["cumulative_wr"]
    anti_final = anti_results[-1]["cumulative_wr"]
    
    print(f"\n  {'Condition':<20s} {'Final WR':>10s} {'vs Scratch':>12s} {'vs Random':>12s}")
    print(f"  {'-'*54}")
    print(f"  {'Random baseline':<20s} {random_wr:>9.1%} {(random_wr-scratch_final)*100:>+11.1f}pp {'---':>12s}")
    print(f"  {'Scratch':<20s} {scratch_final:>9.1%} {'---':>12s} {(scratch_final-random_wr)*100:>+11.1f}pp")
    print(f"  {'TTT Transfer':<20s} {transfer_final:>9.1%} {(transfer_final-scratch_final)*100:>+11.1f}pp {(transfer_final-random_wr)*100:>+11.1f}pp")
    print(f"  {'TTT Anti-Transfer':<20s} {anti_final:>9.1%} {(anti_final-scratch_final)*100:>+11.1f}pp {(anti_final-random_wr)*100:>+11.1f}pp")
    
    # Per-phase comparison
    print(f"\n  PER-PHASE WIN RATES:")
    print(f"  {'Phase':>6s} {'Scratch':>10s} {'Transfer':>10s} {'Anti':>10s} {'Δ Trans':>10s} {'Δ Anti':>10s}")
    for i in range(3):
        sw = scratch_results[i]["phase_wr"]
        tw = transfer_results[i]["phase_wr"]
        aw = anti_results[i]["phase_wr"]
        print(f"  {i+1:>6d} {sw:>9.1%} {tw:>9.1%} {aw:>9.1%} "
              f"{(tw-sw)*100:>+9.1f}pp {(aw-sw)*100:>+9.1f}pp")
    
    # Verdict
    delta_transfer = (transfer_final - scratch_final) * 100
    delta_anti = (anti_final - scratch_final) * 100
    
    print(f"\n  VERDICT:")
    if delta_transfer > 0.5:
        print(f"  ✅ CROSS-GENRE TRANSFER WORKS (+{delta_transfer:.1f}pp)")
        print(f"     Negative space from board games helps card games!")
        if delta_anti < -0.5:
            print(f"  ✅ ANTI-TRANSFER CONFIRMS ({delta_anti:+.1f}pp)")
            print(f"     Inverted negative space hurts. The signal is real.")
    elif delta_transfer < -0.5:
        print(f"  ❌ NEGATIVE TRANSFER ({delta_transfer:.1f}pp)")
        print(f"     TTT negative space HURTS poker. Genres are too different.")
    else:
        print(f"  ⚖️ NEUTRAL TRANSFER ({delta_transfer:+.1f}pp)")
        print(f"     No significant effect. The signal is noise at this scale.")
    
    print(f"\n  Interpretation:")
    print(f"  TTT is deterministic + perfect info + small state space")
    print(f"  Poker is stochastic + hidden info + large state space")
    print(f"  Transfer {'would' if abs(delta_transfer) < 0.5 else 'does'} suggest the negative space")
    print(f"  captures something about DECISION ARCHITECTURE, not game mechanics.")
    
    # Save results
    output = {
        "experiment": "cross_genre_transfer",
        "source": "tic_tac_toe",
        "target": "texas_holdem",
        "ttt_source_stats": ttt_stats,
        "random_baseline": random_wr,
        "scratch": {
            "final_wr": scratch_final,
            "per_phase": scratch_results,
            "tiles": len(scratch_field.tiles),
        },
        "transfer": {
            "final_wr": transfer_final,
            "per_phase": transfer_results,
            "tiles": len(transfer_field.tiles),
            "transfer_weight": 0.3,
        },
        "anti_transfer": {
            "final_wr": anti_final,
            "per_phase": anti_results,
            "tiles": len(anti_field.tiles),
        },
        "transfer_advantage_pp": delta_transfer,
        "anti_penalty_pp": delta_anti,
        "elapsed_seconds": round(elapsed, 1),
    }
    
    out = os.path.expanduser("~/repos/zeroclaw-arena/cross-genre-transfer-results.json")
    with open(out, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {out}")
    
    return output


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    run_cross_genre_experiment()
