"""
Deception Layer — When Both Players Can Read Each Other's Tile Fields

If both players can observe the other's reflex scores, does a deception
strategy emerge naturally? Can a player benefit from manipulating what
the opponent sees?

This is the "keystone" of competitive intelligence: reading the opponent's
negative space AND controlling what they read about yours.

Experiment:
1. Player A can see B's scores, B plays random (A has "the keen eye")
2. Both can see each other's scores (mutual observation)
3. A can SEE B's scores, but A also PRETENDS to have certain scores
   (deception layer — A's true scores differ from visible scores)
"""

import random
import numpy as np
import hashlib
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from holdem_tile import (
    PokerTileField, HoldemHand, hand_name, best_hand, card_str, 
    RANK_VAL, RANKS, SUITS
)


class DeceptiveTileField(PokerTileField):
    """Tile field that can present a DIFFERENT face to opponents."""
    
    def __init__(self):
        super().__init__()
        self.visible_scores = {}  # What opponents see (can differ from real)
        self.deception_active = False
        self.deception_intensity = 0.0  # 0 = honest, 1 = full deception
    
    def get_tile(self, stage: str, hand_bucket: int, pot_bucket: int, position: int):
        tile = super().get_tile(stage, hand_bucket, pot_bucket, position)
        key = tile.state_str
        if key not in self.visible_scores:
            self.visible_scores[key] = {
                a: d["score"] for a, d in tile.reflexes.items()
            }
        return tile
    
    def update_visible_scores(self, strategy="honest"):
        """Update what opponents see."""
        for key, tile in self.tiles.items():
            if strategy == "honest":
                self.visible_scores[key] = {
                    a: d["score"] for a, d in tile.reflexes.items()
                }
            elif strategy == "invert":
                # Show inverted scores — make strong hands look weak and vice versa
                self.visible_scores[key] = {
                    a: 1.0 - d["score"] for a, d in tile.reflexes.items()
                }
            elif strategy == "randomize":
                # Show random scores — give opponent nothing to read
                self.visible_scores[key] = {
                    a: random.uniform(0.2, 0.8) for a in tile.reflexes
                }
            elif strategy == "aggressive_mask":
                # Show always-aggressive — opponent can't tell when you're bluffing
                self.visible_scores[key] = {
                    a: 0.7 if a in ["raise_small", "raise_big", "bluff"] else 0.3
                    for a in tile.reflexes
                }
            elif strategy == "adaptive":
                # Mix real and fake based on deception_intensity
                real = {a: d["score"] for a, d in tile.reflexes.items()}
                fake = {a: 1.0 - d["score"] for a, d in tile.reflexes.items()}
                alpha = self.deception_intensity
                self.visible_scores[key] = {
                    a: (1 - alpha) * real[a] + alpha * fake[a]
                    for a in tile.reflexes
                }
    
    def opponent_reads(self, stage, hand_bucket, pot_bucket):
        """What an opponent would see for this state."""
        key = f"{stage}:h{hand_bucket}:p{pot_bucket}:pos1"
        if key in self.visible_scores:
            return self.visible_scores[key]
        return None


def play_deceptive_hand(
    field_a: DeceptiveTileField, 
    field_b: DeceptiveTileField,
    b_can_read_a: bool = False,
    a_can_read_b: bool = False,
) -> dict:
    """Play one hand where B can optionally read A's visible scores."""
    game = HoldemHand()
    tile_decisions_a = []
    tile_decisions_b = []
    
    for stage_idx in range(4):
        game.stage = stage_idx
        if stage_idx == 1: game.deal_community(3)
        elif stage_idx in [2, 3]: game.deal_community(1)
        
        # Player A decision
        h_a = game.hand_strength_bucket(0)
        p_bucket = game.pot_bucket()
        tile_a = field_a.get_tile(HoldemHand.STAGES[stage_idx], h_a, p_bucket, 0)
        
        T_a = max(0.15, 0.5 - stage_idx * 0.1)
        
        # If B can read A, A's action selection may be influenced
        if b_can_read_a and field_a.deception_active:
            # B reads A's visible scores and counters
            visible = field_a.opponent_reads(
                HoldemHand.STAGES[stage_idx], h_a, p_bucket)
            # A still uses real scores for own decisions
            action_a = field_a.choose_action(tile_a, T_a, epsilon=0.03)
        else:
            action_a = field_a.choose_action(tile_a, T_a, epsilon=0.05)
        
        tile_decisions_a.append((tile_a, action_a))
        
        # Player B decision
        h_b = game.hand_strength_bucket(1)
        tile_b = field_b.get_tile(HoldemHand.STAGES[stage_idx], h_b, p_bucket, 1)
        
        T_b = max(0.15, 0.5 - stage_idx * 0.1)
        
        # If B can read A, adjust B's strategy
        if b_can_read_a:
            visible = field_a.opponent_reads(
                HoldemHand.STAGES[stage_idx], h_a, p_bucket)
            if visible:
                # B knows A's likely action → counter it
                a_most_likely = max(visible, key=visible.get)
                if a_most_likely in ["raise_small", "raise_big", "bluff"]:
                    # A is likely aggressive → B plays tighter
                    if h_b < 2:
                        action_b = "fold"
                    else:
                        action_b = "check_call"
                else:
                    action_b = field_b.choose_action(tile_b, T_b, epsilon=0.05)
            else:
                action_b = field_b.choose_action(tile_b, T_b, epsilon=0.05)
        else:
            action_b = field_b.choose_action(tile_b, T_b, epsilon=0.05)
        
        tile_decisions_b.append((tile_b, action_b))
        
        # Resolve
        result = game.play_round([action_a, action_b])
        if result is not None:
            won_a = result == 0
            for t, a in tile_decisions_a:
                field_a.record(t, a, won_a)
            for t, a in tile_decisions_b:
                field_b.record(t, a, not won_a)
            return {"winner": result, "fold": True, 
                    "stage": HoldemHand.STAGES[stage_idx],
                    "a_action": action_a, "b_action": action_b}
    
    # Showdown
    w = game.showdown()
    won_a = w == 0
    for t, a in tile_decisions_a:
        field_a.record(t, a, won_a)
    for t, a in tile_decisions_b:
        field_b.record(t, a, not won_a)
    
    return {"winner": w, "fold": False,
            "hand_a": hand_name(best_hand(game.hole[0], game.community)),
            "hand_b": hand_name(best_hand(game.hole[1], game.community))}


def run_deception_experiment():
    print("=" * 70)
    print("DECEPTION LAYER — Reading & Manipulating the Opponent's Tile Field")
    print("=" * 70)
    
    n_hands = 1000
    strategies = [
        ("blind", False, False, "honest"),        # Neither can read
        ("b_reads_a", True, False, "honest"),      # B reads A's real scores
        ("b_reads_deception", True, False, "invert"),  # B reads A's FAKE scores
        ("mutual_reading", True, True, "honest"),   # Both read each other
        ("adaptive_deception", True, False, "adaptive"),  # A adapts deception intensity
    ]
    
    results = {}
    
    for strat_name, b_reads, a_reads, deception in strategies:
        print(f"\n--- Strategy: {strat_name} ---")
        
        field_a = DeceptiveTileField()
        field_b = DeceptiveTileField()
        
        if deception == "adaptive":
            field_a.deception_active = True
        
        wins_a = 0
        wins_b = 0
        folds_a = 0  # Times A forced B to fold
        
        for hand_num in range(n_hands):
            # Update deception
            if deception == "adaptive":
                # Increase deception over time as field learns
                field_a.deception_intensity = min(0.8, hand_num / n_hands * 0.8)
                field_a.update_visible_scores("adaptive")
            elif deception != "honest":
                field_a.update_visible_scores(deception)
            
            result = play_deceptive_hand(field_a, field_b, b_reads, a_reads)
            
            if result["winner"] == 0:
                wins_a += 1
                if result.get("fold") and result.get("b_action") == "fold":
                    folds_a += 1
            elif result["winner"] == 1:
                wins_b += 1
            
            # Evolve every 200 hands
            if (hand_num + 1) % 200 == 0:
                field_a.evolve(lr=0.04, cap=0.05)
                field_b.evolve(lr=0.04, cap=0.05)
                
                if deception == "adaptive":
                    field_a.update_visible_scores("adaptive")
                elif deception != "honest":
                    field_a.update_visible_scores(deception)
                
                wr = wins_a / (hand_num + 1)
                print(f"  Hand {hand_num+1}: A={wr:.1%} folds_forced={folds_a}")
        
        wr_a = wins_a / n_hands
        wr_b = wins_b / n_hands
        
        results[strat_name] = {
            "a_win_rate": wr_a,
            "b_win_rate": wr_b,
            "folds_forced_by_a": folds_a,
            "a_tiles": len(field_a.tiles),
            "b_tiles": len(field_b.tiles),
            "deception": deception,
        }
        
        print(f"  FINAL: A={wr_a:.1%} B={wr_b:.1%} folds_forced={folds_a}")
    
    # Summary
    print(f"\n{'=' * 70}")
    print("DECEPTION LAYER RESULTS")
    print(f"{'=' * 70}")
    print(f"{'Strategy':<25s} {'A Win%':>8s} {'B Win%':>8s} {'Folds':>8s}")
    print("-" * 50)
    for name, r in results.items():
        print(f"  {name:<23s} {r['a_win_rate']:>7.1%} {r['b_win_rate']:>7.1%} {r['folds_forced_by_a']:>7d}")
    
    # Save
    out = os.path.expanduser("~/repos/zeroclaw-arena/deception-layer-results.json")
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    run_deception_experiment()
