"""
Evolutionary Arms Race — The Red Queen Hypothesis in Poker

"In the land of the blind, the one-eyed man is king. 
 But in an arms race, everyone keeps growing eyes."

Generation 0: Two default fields (A and B) — both naive
Generation 1: A evolves from gen 0 outcomes, B stays frozen
Generation 2: B evolves from gen 1 outcomes, A stays frozen
...alternate for 10 generations, 500 hands each.

The Red Queen hypothesis: you have to keep running (evolving) just to stay in the same place.
"""

import random
import numpy as np
import json
import os
import copy
from collections import defaultdict
from itertools import combinations
from typing import List, Dict, Tuple, Optional

# Import from existing module
from holdem_tile import (
    PokerTileField, PokerTile, HoldemHand,
    make_deck, best_hand, hand_name, hand_rank, RANK_VAL
)


# ============== FIELD vs FIELD GAME ENGINE ==============

class ArmsRaceHand(HoldemHand):
    """Extended hand that supports tile-field vs tile-field play."""
    
    def play_dual(self, field_a: PokerTileField, field_b: PokerTileField) -> dict:
        """Play a hand with two tile-field players."""
        tiles_a = []  # Tiles activated by player A
        tiles_b = []  # Tiles activated by player B
        actions_a = []
        actions_b = []
        
        for stage_idx in range(4):
            self.stage = stage_idx
            
            # Deal community cards
            if stage_idx == 1:
                self.deal_community(3)
            elif stage_idx in [2, 3]:
                self.deal_community(1)
            
            # Player A (field_a) acts
            h_bucket_a = self.hand_strength_bucket(0)
            p_bucket = self.pot_bucket()
            tile_a = field_a.get_tile(self.STAGES[self.stage], h_bucket_a, p_bucket, 0)
            T_a = max(0.15, 0.5 - stage_idx * 0.1)
            action_a = field_a.choose_action(tile_a, T_a, epsilon=0.05)
            tiles_a.append(tile_a)
            actions_a.append(action_a)
            
            # Player B (field_b) acts
            h_bucket_b = self.hand_strength_bucket(1)
            tile_b = field_b.get_tile(self.STAGES[self.stage], h_bucket_b, p_bucket, 1)
            T_b = max(0.15, 0.5 - stage_idx * 0.1)
            action_b = field_b.choose_action(tile_b, T_b, epsilon=0.05)
            tiles_b.append(tile_b)
            actions_b.append(action_b)
            
            # Resolve actions
            result = self.play_round([action_a, action_b])
            if result is not None:
                # Someone folded
                for t, a in zip(tiles_a, actions_a):
                    field_a.record(t, a, result == 0)
                for t, a in zip(tiles_b, actions_b):
                    field_b.record(t, a, result == 1)
                return {
                    "winner": result, "fold": True,
                    "stage": self.STAGES[stage_idx],
                    "pot": self.pot,
                    "actions_a": list(zip([t.state_str for t in tiles_a], actions_a)),
                    "actions_b": list(zip([t.state_str for t in tiles_b], actions_b)),
                }
        
        # Showdown
        winner = self.showdown()
        for t, a in zip(tiles_a, actions_a):
            field_a.record(t, a, winner == 0)
        for t, a in zip(tiles_b, actions_b):
            field_b.record(t, a, winner == 1)
        
        return {
            "winner": winner, "fold": False,
            "stage": "showdown",
            "pot": self.pot,
            "hand_a": hand_name(best_hand(self.hole[0], self.community)) if winner != -1 else "tie",
            "hand_b": hand_name(best_hand(self.hole[1], self.community)) if winner != -1 else "tie",
            "actions_a": list(zip([t.state_str for t in tiles_a], actions_a)),
            "actions_b": list(zip([t.state_str for t in tiles_b], actions_b)),
        }


# ============== METRICS ==============

def strategy_distance(field_a: PokerTileField, field_b: PokerTileField) -> float:
    """Compute L2 distance between two fields' reflex scores on shared tiles."""
    all_keys = set(field_a.tiles.keys()) | set(field_b.tiles.keys())
    if not all_keys:
        return 0.0
    
    dist_sq = 0.0
    count = 0
    for key in all_keys:
        ta = field_a.tiles.get(key)
        tb = field_b.tiles.get(key)
        actions = ["fold", "check_call", "raise_small", "raise_big", "bluff"]
        for action in actions:
            sa = ta.reflexes[action]["score"] if ta else 0.5
            sb = tb.reflexes[action]["score"] if tb else 0.5
            dist_sq += (sa - sb) ** 2
            count += 1
    
    return np.sqrt(dist_sq / max(count, 1))


def self_distance(field_a: PokerTileField, field_b: PokerTileField) -> float:
    """Distance of each field from the default (gen 0) baseline."""
    default_field = PokerTileField()
    da = strategy_distance(field_a, default_field)
    db = strategy_distance(field_b, default_field)
    return da, db


def count_bluffs(field: PokerTileField) -> dict:
    """Count bluff usage across all tiles."""
    total_chosen = 0
    bluff_chosen = 0
    bluff_won = 0
    for tile in field.tiles.values():
        for action, data in tile.reflexes.items():
            total_chosen += data["chosen"]
            if action == "bluff":
                bluff_chosen += data["chosen"]
                bluff_won += data["won"]
    
    return {
        "bluff_count": bluff_chosen,
        "total_actions": total_chosen,
        "bluff_rate": bluff_chosen / max(total_chosen, 1),
        "bluff_wr": bluff_won / max(bluff_chosen, 1),
    }


def dominant_strategy(field: PokerTileField) -> dict:
    """Get the average reflex scores across all tiles."""
    action_scores = defaultdict(list)
    for tile in field.tiles.values():
        for action, data in tile.reflexes.items():
            if data["chosen"] > 0:
                action_scores[action].append(data["score"])
    
    result = {}
    for action, scores in action_scores.items():
        result[action] = np.mean(scores) if scores else 0.5
    return result


# ============== MAIN EXPERIMENT ==============

def run_arms_race(num_generations=10, hands_per_gen=500, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    print("=" * 75)
    print("EVOLUTIONARY ARMS RACE — Red Queen Hypothesis in Texas Hold'em")
    print("=" * 75)
    print(f"\n  Generations: {num_generations}")
    print(f"  Hands per generation: {hands_per_gen}")
    print(f"  Alternating evolution: A evolves on odd gens, B on even (after gen 0)")
    print()
    
    # Generation 0: Both start as default fields
    field_a = PokerTileField()
    field_b = PokerTileField()
    
    # Track history
    history = []
    
    # Save gen-0 state for distance measurement
    baseline_scores_a = {}
    baseline_scores_b = {}
    
    for gen in range(num_generations):
        print(f"--- Generation {gen} ---")
        
        # Determine who evolves this generation
        if gen == 0:
            evolver = "both"  # Both learn from scratch
        elif gen % 2 == 1:
            evolver = "A"  # A adapts to B's frozen strategy
        else:
            evolver = "B"  # B adapts to A's frozen strategy
        
        # Snapshot the frozen player's field for distance tracking
        frozen_snapshot_b = copy.deepcopy(field_b) if evolver == "A" else None
        frozen_snapshot_a = copy.deepcopy(field_a) if evolver == "B" else None
        
        wins_a = 0
        wins_b = 0
        ties = 0
        folds = 0
        showdowns = 0
        pots = []
        bluff_actions_a = 0
        bluff_actions_b = 0
        total_actions = 0
        
        for hand_num in range(hands_per_gen):
            game = ArmsRaceHand()
            result = game.play_dual(field_a, field_b)
            
            if result["winner"] == 0:
                wins_a += 1
            elif result["winner"] == 1:
                wins_b += 1
            else:
                ties += 1
            
            if result["fold"]:
                folds += 1
            else:
                showdowns += 1
            
            pots.append(result["pot"])
            
            # Count bluffs this hand
            for _, a in result["actions_a"]:
                total_actions += 1
                if a == "bluff":
                    bluff_actions_a += 1
            for _, a in result["actions_b"]:
                total_actions += 1
                if a == "bluff":
                    bluff_actions_b += 1
        
        # Evolve the appropriate field
        if evolver in ("A", "both"):
            field_a.evolve(lr=0.04, cap=0.05)
        if evolver in ("B", "both"):
            field_b.evolve(lr=0.04, cap=0.05)
        
        # Compute metrics
        wr_a = wins_a / hands_per_gen
        wr_b = wins_b / hands_per_gen
        dist = strategy_distance(field_a, field_b)
        bluff_a = count_bluffs(field_a)
        bluff_b = count_bluffs(field_b)
        dom_a = dominant_strategy(field_a)
        dom_b = dominant_strategy(field_b)
        dist_from_default_a, dist_from_default_b = self_distance(field_a, field_b)
        
        gen_data = {
            "generation": gen,
            "evolver": evolver,
            "wins_a": wins_a,
            "wins_b": wins_b,
            "ties": ties,
            "wr_a": round(wr_a, 4),
            "wr_b": round(wr_b, 4),
            "delta_wr": round(wr_a - wr_b, 4),
            "folds": folds,
            "showdowns": showdowns,
            "avg_pot": round(np.mean(pots), 1),
            "strategy_distance": round(dist, 4),
            "dist_from_default_a": round(dist_from_default_a, 4),
            "dist_from_default_b": round(dist_from_default_b, 4),
            "bluff_rate_a": round(bluff_a["bluff_rate"], 4),
            "bluff_rate_b": round(bluff_b["bluff_rate"], 4),
            "bluff_wr_a": round(bluff_a["bluff_wr"], 4),
            "bluff_wr_b": round(bluff_b["bluff_wr"], 4),
            "bluff_actions_a": bluff_actions_a,
            "bluff_actions_b": bluff_actions_b,
            "dominant_a": {k: round(v, 3) for k, v in sorted(dom_a.items(), key=lambda x: -x[1])},
            "dominant_b": {k: round(v, 3) for k, v in sorted(dom_b.items(), key=lambda x: -x[1])},
            "tiles_a": len(field_a.tiles),
            "tiles_b": len(field_b.tiles),
        }
        history.append(gen_data)
        
        leader = "A" if wr_a > wr_b else ("B" if wr_b > wr_a else "TIE")
        print(f"  Evolver: {evolver:4s} | A wins: {wins_a:3d} ({wr_a:.1%}) | "
              f"B wins: {wins_b:3d} ({wr_b:.1%}) | Leader: {leader}")
        print(f"  Strategy distance: {dist:.4f} | "
              f"Dist from default: A={dist_from_default_a:.3f} B={dist_from_default_b:.3f}")
        print(f"  Bluff rate: A={bluff_a['bluff_rate']:.2%} (wr={bluff_a['bluff_wr']:.0%}) | "
              f"B={bluff_b['bluff_rate']:.2%} (wr={bluff_b['bluff_wr']:.0%})")
        print(f"  Avg pot: {np.mean(pots):.1f} | Tiles: A={len(field_a.tiles)} B={len(field_b.tiles)}")
        print()
    
    # ============== ANALYSIS ==============
    print("\n" + "=" * 75)
    print("ARMS RACE ANALYSIS")
    print("=" * 75)
    
    # 1. Win rate per generation
    print("\n1. WIN RATE PER GENERATION")
    print(f"   {'Gen':>3} {'Evolver':>8} {'A WR':>7} {'B WR':>7} {'Delta':>7} {'Leader':>6}")
    print("   " + "-" * 45)
    for g in history:
        leader = "A" if g["wr_a"] > g["wr_b"] else ("B" if g["wr_b"] > g["wr_a"] else "TIE")
        print(f"   {g['generation']:3d} {g['evolver']:>8} {g['wr_a']:>7.1%} {g['wr_b']:>7.1%} "
              f"{g['delta_wr']:>+7.1%} {leader:>6}")
    
    # 2. Strategy distance
    print("\n2. STRATEGY DISTANCE (L2 between A and B)")
    for g in history:
        bar = "█" * int(g["strategy_distance"] * 50)
        print(f"   Gen {g['generation']:2d}: {g['strategy_distance']:.4f} {bar}")
    
    # 3. Does the lead oscillate?
    print("\n3. LEAD OSCILLATION (Red Queen Dynamics)")
    leads = []
    for g in history:
        if g["wr_a"] > g["wr_b"]:
            leads.append("A")
        elif g["wr_b"] > g["wr_a"]:
            leads.append("B")
        else:
            leads.append("TIE")
    
    switches = sum(1 for i in range(1, len(leads)) if leads[i] != leads[i-1])
    print(f"   Lead sequence: {' → '.join(leads)}")
    print(f"   Lead switches: {switches}")
    
    if switches >= 4:
        print("   ✅ OSCILLATION DETECTED — classic Red Queen dynamics!")
        print("   Each player adapts to the other's frozen strategy, flipping the lead.")
    elif switches >= 2:
        print("   ⚠️  PARTIAL oscillation — some adaptation but not full cycle")
    else:
        print("   ❌ No oscillation — one side dominates throughout")
    
    # 4. Bluff evolution
    print("\n4. BLUFF EVOLUTION")
    print(f"   {'Gen':>3} {'A bluff%':>10} {'A wr':>6} {'B bluff%':>10} {'B wr':>6}")
    print("   " + "-" * 40)
    for g in history:
        print(f"   {g['generation']:3d} {g['bluff_rate_a']:>10.2%} {g['bluff_wr_a']:>6.0%} "
              f"{g['bluff_rate_b']:>10.2%} {g['bluff_wr_b']:>6.0%}")
    
    bluff_trend_a = history[-1]["bluff_rate_a"] - history[0]["bluff_rate_a"]
    bluff_trend_b = history[-1]["bluff_rate_b"] - history[0]["bluff_rate_b"]
    print(f"\n   Bluff rate trend: A {'↑' if bluff_trend_a > 0 else '↓'} "
          f"({bluff_trend_a:+.2%}), B {'↑' if bluff_trend_b > 0 else '↓'} "
          f"({bluff_trend_b:+.2%})")
    
    if bluff_trend_a > 0.02 or bluff_trend_b > 0.02:
        print("   → Bluffs INCREASE over generations — arms race in deception")
    elif bluff_trend_a < -0.02 or bluff_trend_b < -0.02:
        print("   → Bluffs DECREASE over generations — arms race in honesty (mutual detection)")
    else:
        print("   → Bluffs remain STABLE — equilibrium in deception frequency")
    
    # 5. Does the system settle into a cycle?
    print("\n5. CYCLE DETECTION")
    # Check if last 4 generations show repeating win rate patterns
    if len(history) >= 4:
        last_deltas = [g["delta_wr"] for g in history[-4:]]
        last_signs = [np.sign(d) for d in last_deltas]
        
        # Check for alternating signs (ABAB pattern)
        alternations = sum(1 for i in range(1, len(last_signs)) 
                          if last_signs[i] * last_signs[i-1] < 0)
        
        if alternations >= 2:
            print("   ✅ CYCLE DETECTED — win rates oscillate in the last 4 generations")
            print(f"   Delta pattern: {' → '.join(f'{d:+.1%}' for d in last_deltas)}")
        else:
            print("   No clear cycle in recent generations")
            print(f"   Delta pattern: {' → '.join(f'{d:+.1%}' for d in last_deltas)}")
    
    # Check convergence of strategy distance
    dists = [g["strategy_distance"] for g in history]
    if len(dists) >= 3:
        recent_std = np.std(dists[-3:])
        overall_std = np.std(dists)
        if recent_std < overall_std * 0.3:
            print(f"   Strategy distance STABILIZED (σ_recent={recent_std:.4f} vs σ_overall={overall_std:.4f})")
        else:
            print(f"   Strategy distance still DIVERGING/FLUCTUATING (σ_recent={recent_std:.4f})")
    
    # Distance from default over time
    print("\n6. DISTANCE FROM DEFAULT (Evolution From Naive)")
    for g in history:
        bar_a = "▓" * int(g["dist_from_default_a"] * 30)
        bar_b = "░" * int(g["dist_from_default_b"] * 30)
        print(f"   Gen {g['generation']:2d}: A {g['dist_from_default_a']:.3f} {bar_a}")
        print(f"          B {g['dist_from_default_b']:.3f} {bar_b}")
    
    # ============== SAVE ==============
    output = {
        "experiment": "evolutionary_arms_race",
        "params": {
            "num_generations": num_generations,
            "hands_per_gen": hands_per_gen,
            "seed": seed,
        },
        "summary": {
            "lead_switches": switches,
            "lead_sequence": leads,
            "bluff_trend_a": round(bluff_trend_a, 4),
            "bluff_trend_b": round(bluff_trend_b, 4),
            "final_strategy_distance": dists[-1],
            "max_strategy_distance": max(dists),
            "final_wr_a": history[-1]["wr_a"],
            "final_wr_b": history[-1]["wr_b"],
            "oscillation_detected": switches >= 4,
            "cycle_detected": alternations >= 2 if len(history) >= 4 else False,
        },
        "generations": history,
    }
    
    out_path = os.path.expanduser("~/repos/zeroclaw-arena/evolutionary-arms-race-results.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")
    
    return output


if __name__ == "__main__":
    run_arms_race(num_generations=10, hands_per_gen=500, seed=42)
