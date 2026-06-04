"""
Texas Hold'em × Rival Intelligence — Two Tile Fields at War

Two independent PokerTileField instances (Player A and Player B) compete.
Each learns from playing against the other.

Phase 1 (1000 hands): Both use default scores (all ~0.5) — raw emergence
Phase 2 (1000 hands): A uses learned scores, B still default — asymmetric exploitation
Phase 3 (1000 hands): Both use learned scores — the arms race

Key questions:
  - Does mutual learning converge to a stable equilibrium, or oscillate forever?
  - Does bluffing emerge as a viable strategy?
  - Does the field read the "negative space" of the opponent?
"""

import random
import numpy as np
import json
import os
import sys
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

# Import the existing poker infrastructure
from holdem_tile import (
    PokerTileField, PokerTile, HoldemHand,
    best_hand, hand_name, hand_rank, RANK_VAL,
    make_deck, SUITS, RANKS
)


class RivalHand(HoldemHand):
    """Extended HoldemHand that supports two tile-field players."""

    def play_rival(self, field_a: PokerTileField, field_b: PokerTileField,
                   a_active: bool = True, b_active: bool = True) -> dict:
        """
        Play a complete hand with two tile-field players.
        a_active/b_active: whether each player uses learned scores or defaults.
        """
        tiles_used_a = []
        tiles_used_b = []
        actions_a = []
        actions_b = []

        for stage_idx in range(4):
            self.stage = stage_idx

            if stage_idx == 1:
                self.deal_community(3)
            elif stage_idx in [2, 3]:
                self.deal_community(1)

            # Player A acts
            h_bucket_a = self.hand_strength_bucket(0)
            h_bucket_b = self.hand_strength_bucket(1)
            p_bucket = self.pot_bucket()

            tile_a = field_a.get_tile(self.STAGES[self.stage], h_bucket_a, p_bucket, 0)
            tile_b = field_b.get_tile(self.STAGES[self.stage], h_bucket_b, p_bucket, 1)

            T = max(0.15, 0.5 - stage_idx * 0.1)

            if a_active:
                action_a = field_a.choose_action(tile_a, T, epsilon=0.05)
            else:
                # Default: mostly check/call with slight randomness
                action_a = np.random.choice(
                    ["fold", "check_call", "raise_small", "raise_big", "bluff"],
                    p=[0.02, 0.60, 0.20, 0.10, 0.08]
                )

            if b_active:
                action_b = field_b.choose_action(tile_b, T, epsilon=0.05)
            else:
                action_b = np.random.choice(
                    ["fold", "check_call", "raise_small", "raise_big", "bluff"],
                    p=[0.02, 0.60, 0.20, 0.10, 0.08]
                )

            tiles_used_a.append(tile_a)
            tiles_used_b.append(tile_b)
            actions_a.append(action_a)
            actions_b.append(action_b)

            result = self.play_round([action_a, action_b])
            if result is not None:
                a_won = result == 0
                for t, a in zip(tiles_used_a, actions_a):
                    field_a.record(t, a, a_won)
                for t, a in zip(tiles_used_b, actions_b):
                    field_b.record(t, a, not a_won)

                return {
                    "winner": result,
                    "fold": True,
                    "stage": self.STAGES[stage_idx],
                    "pot": self.pot,
                    "actions_a": actions_a,
                    "actions_b": actions_b,
                    "fold_by": 0 if action_a == "fold" else 1,
                }

        # Showdown
        winner = self.showdown()
        a_won = winner == 0
        b_won = winner == 1
        tie = winner == -1

        for t, a in zip(tiles_used_a, actions_a):
            field_a.record(t, a, a_won)
        for t, a in zip(tiles_used_b, actions_b):
            field_b.record(t, a, b_won)

        return {
            "winner": winner,
            "fold": False,
            "stage": "showdown",
            "pot": self.pot,
            "hand_a": hand_name(best_hand(self.hole[0], self.community)),
            "hand_b": hand_name(best_hand(self.hole[1], self.community)),
            "actions_a": actions_a,
            "actions_b": actions_b,
        }


def analyze_field(field: PokerTileField, name: str) -> dict:
    """Analyze a tile field's evolved strategy."""
    analysis = {
        "name": name,
        "tiles": len(field.tiles),
        "total_visits": sum(t.visits for t in field.tiles.values()),
        "by_stage": {},
        "bluff_analysis": {},
        "top_reflexes": {},
    }

    for stage in ["preflop", "flop", "turn", "river"]:
        stage_tiles = [t for t in field.tiles.values() if t.stage == stage and t.visits > 2]
        if not stage_tiles:
            continue

        reflex_agg = defaultdict(lambda: {"score": 0, "chosen": 0, "won": 0, "count": 0})
        for t in stage_tiles:
            for a, d in t.reflexes.items():
                reflex_agg[a]["score"] += d["score"]
                reflex_agg[a]["chosen"] += d["chosen"]
                reflex_agg[a]["won"] += d["won"]
                reflex_agg[a]["count"] += 1

        analysis["by_stage"][stage] = {
            a: {
                "avg_score": round(d["score"] / d["count"], 3),
                "wr": round(d["won"] / max(d["chosen"], 1), 3),
                "times_chosen": d["chosen"],
            }
            for a, d in reflex_agg.items()
        }

        # Best action
        best = max(reflex_agg.items(),
                   key=lambda x: x[1]["score"] / x[1]["count"])
        analysis["top_reflexes"][stage] = best[0]

    # Bluffing analysis
    bluff_tiles = [(t.state_str, t.reflexes["bluff"]["score"],
                    t.reflexes["bluff"]["chosen"],
                    t.reflexes["bluff"]["won"])
                   for t in field.tiles.values()
                   if t.reflexes["bluff"]["chosen"] > 2]
    bluff_tiles.sort(key=lambda x: -x[1])
    analysis["bluff_analysis"]["top_bluff_states"] = bluff_tiles[:5]
    analysis["bluff_analysis"]["total_bluffs"] = sum(
        t.reflexes["bluff"]["chosen"] for t in field.tiles.values())
    analysis["bluff_analysis"]["bluff_wr"] = round(
        sum(t.reflexes["bluff"]["won"] for t in field.tiles.values()) /
        max(sum(t.reflexes["bluff"]["chosen"] for t in field.tiles.values()), 1), 3)

    return analysis


def compute_strategy_distance(field_a: PokerTileField, field_b: PokerTileField) -> float:
    """Measure how different two fields' strategies are (L2 distance of score vectors)."""
    all_keys = set(field_a.tiles.keys()) | set(field_b.tiles.keys())
    if not all_keys:
        return 0.0

    dist_sq = 0.0
    for key in all_keys:
        ta = field_a.tiles.get(key)
        tb = field_b.tiles.get(key)
        for action in ["fold", "check_call", "raise_small", "raise_big", "bluff"]:
            sa = ta.reflexes[action]["score"] if ta else 0.5
            sb = tb.reflexes[action]["score"] if tb else 0.5
            dist_sq += (sa - sb) ** 2

    return np.sqrt(dist_sq)


def run_rival_arena():
    print("=" * 72)
    print("RIVAL INTELLIGENCE — Two Tile Fields at War")
    print("Texas Hold'em: Player A vs Player B")
    print("=" * 72)

    field_a = PokerTileField()
    field_b = PokerTileField()

    results = {
        "phase1": {}, "phase2": {}, "phase3": {},
        "convergence": [], "strategy_snapshots": {},
    }

    # ─── PHASE 1: Both Default (1000 hands) ─────────────────────────────
    print("\n" + "─" * 72)
    print("PHASE 1: Both Default — Raw Emergence (1000 hands)")
    print("─" * 72)

    p1_a_wins = 0
    p1_b_wins = 0
    p1_ties = 0
    p1_folds_a = 0  # A won by fold
    p1_folds_b = 0  # B won by fold
    p1_showdowns = 0
    p1_pot_a = 0
    p1_pot_b = 0
    p1_convergence = []

    for hand_num in range(1000):
        game = RivalHand()
        result = game.play_rival(field_a, field_b, a_active=False, b_active=False)

        if result["winner"] == 0:
            p1_a_wins += 1
            p1_pot_a += result["pot"]
        elif result["winner"] == 1:
            p1_b_wins += 1
            p1_pot_b += result["pot"]
        else:
            p1_ties += 1

        if result["fold"]:
            if result["fold_by"] == 1:
                p1_folds_a += 1
            else:
                p1_folds_b += 1
        else:
            p1_showdowns += 1

        # Periodic evolution and convergence tracking
        if (hand_num + 1) % 100 == 0:
            # Both fields still learn from outcomes even though they don't use learned scores yet
            field_a.evolve(lr=0.03, cap=0.04)
            field_b.evolve(lr=0.03, cap=0.04)

            dist = compute_strategy_distance(field_a, field_b)
            p1_convergence.append(dist)

            wr_a = p1_a_wins / (hand_num + 1)
            wr_b = p1_b_wins / (hand_num + 1)
            print(f"  Hand {hand_num+1:4d}: A_wr={wr_a:.1%} B_wr={wr_b:.1%} "
                  f"ties={p1_ties} dist={dist:.2f} "
                  f"tiles_A={len(field_a.tiles)} tiles_B={len(field_b.tiles)}")

    results["phase1"] = {
        "a_wins": p1_a_wins, "b_wins": p1_b_wins, "ties": p1_ties,
        "a_wr": round(p1_a_wins / 1000, 3),
        "b_wr": round(p1_b_wins / 1000, 3),
        "folds_won_by_a": p1_folds_a, "folds_won_by_b": p1_folds_b,
        "showdowns": p1_showdowns,
        "avg_pot_won_a": round(p1_pot_a / max(p1_a_wins, 1), 2),
        "avg_pot_won_b": round(p1_pot_b / max(p1_b_wins, 1), 2),
        "convergence_distances": p1_convergence,
    }

    print(f"\n  Phase 1 Summary:")
    print(f"    A wins: {p1_a_wins} ({p1_a_wins/10:.1f}%)  B wins: {p1_b_wins} ({p1_b_wins/10:.1f}%)  Ties: {p1_ties}")
    print(f"    A won by fold: {p1_folds_a}  B won by fold: {p1_folds_b}")

    # ─── PHASE 2: A Learned, B Default (1000 hands) ─────────────────────
    print("\n" + "─" * 72)
    print("PHASE 2: A Learned vs B Default — Asymmetric Exploitation (1000 hands)")
    print("─" * 72)

    p2_a_wins = 0
    p2_b_wins = 0
    p2_ties = 0
    p2_folds_a = 0
    p2_folds_b = 0
    p2_showdowns = 0
    p2_bluffs_a = 0
    p2_bluffs_a_won = 0
    p2_pot_a = 0
    p2_pot_b = 0
    p2_convergence = []

    for hand_num in range(1000):
        game = RivalHand()
        result = game.play_rival(field_a, field_b, a_active=True, b_active=False)

        if result["winner"] == 0:
            p2_a_wins += 1
            p2_pot_a += result["pot"]
        elif result["winner"] == 1:
            p2_b_wins += 1
            p2_pot_b += result["pot"]
        else:
            p2_ties += 1

        if result["fold"]:
            if result["fold_by"] == 1:
                p2_folds_a += 1
            else:
                p2_folds_b += 1
        else:
            p2_showdowns += 1

        # Count A's bluffs
        for a in result.get("actions_a", []):
            if a == "bluff":
                p2_bluffs_a += 1
                if result["winner"] == 0:
                    p2_bluffs_a_won += 1

        if (hand_num + 1) % 100 == 0:
            field_a.evolve(lr=0.03, cap=0.04)
            field_b.evolve(lr=0.03, cap=0.04)

            dist = compute_strategy_distance(field_a, field_b)
            p2_convergence.append(dist)

            wr_a = p2_a_wins / (hand_num + 1)
            wr_b = p2_b_wins / (hand_num + 1)
            print(f"  Hand {hand_num+1:4d}: A_wr={wr_a:.1%} B_wr={wr_b:.1%} "
                  f"A_bluffs={p2_bluffs_a} bluff_wr={p2_bluffs_a_won/max(p2_bluffs_a,1):.1%} "
                  f"dist={dist:.2f}")

    results["phase2"] = {
        "a_wins": p2_a_wins, "b_wins": p2_b_wins, "ties": p2_ties,
        "a_wr": round(p2_a_wins / 1000, 3),
        "b_wr": round(p2_b_wins / 1000, 3),
        "folds_won_by_a": p2_folds_a, "folds_won_by_b": p2_folds_b,
        "showdowns": p2_showdowns,
        "bluffs_by_a": p2_bluffs_a,
        "bluff_wr_a": round(p2_bluffs_a_won / max(p2_bluffs_a, 1), 3),
        "avg_pot_won_a": round(p2_pot_a / max(p2_a_wins, 1), 2),
        "avg_pot_won_b": round(p2_pot_b / max(p2_b_wins, 1), 2),
        "convergence_distances": p2_convergence,
    }

    print(f"\n  Phase 2 Summary:")
    print(f"    A wins: {p2_a_wins} ({p2_a_wins/10:.1f}%)  B wins: {p2_b_wins} ({p2_b_wins/10:.1f}%)  Ties: {p2_ties}")
    print(f"    A bluffs: {p2_bluffs_a} (win rate: {p2_bluffs_a_won/max(p2_bluffs_a,1):.1%})")
    print(f"    A avg pot won: {p2_pot_a/max(p2_a_wins,1):.1f}  B avg pot won: {p2_pot_b/max(p2_b_wins,1):.1f}")

    # ─── PHASE 3: Both Learned — Arms Race (1000 hands) ─────────────────
    print("\n" + "─" * 72)
    print("PHASE 3: Both Learned — THE ARMS RACE (1000 hands)")
    print("─" * 72)

    p3_a_wins = 0
    p3_b_wins = 0
    p3_ties = 0
    p3_folds_a = 0
    p3_folds_b = 0
    p3_showdowns = 0
    p3_bluffs_a = 0
    p3_bluffs_b = 0
    p3_bluffs_a_won = 0
    p3_bluffs_b_won = 0
    p3_pot_a = 0
    p3_pot_b = 0
    p3_convergence = []

    # Track rolling win rates to detect oscillation
    rolling_wr_a = []
    window = 50
    recent_results = []

    for hand_num in range(1000):
        game = RivalHand()
        result = game.play_rival(field_a, field_b, a_active=True, b_active=True)

        if result["winner"] == 0:
            p3_a_wins += 1
            p3_pot_a += result["pot"]
            recent_results.append(1)
        elif result["winner"] == 1:
            p3_b_wins += 1
            p3_pot_b += result["pot"]
            recent_results.append(0)
        else:
            p3_ties += 1
            recent_results.append(0.5)

        if result["fold"]:
            if result["fold_by"] == 1:
                p3_folds_a += 1
            else:
                p3_folds_b += 1
        else:
            p3_showdowns += 1

        for a in result.get("actions_a", []):
            if a == "bluff":
                p3_bluffs_a += 1
                if result["winner"] == 0:
                    p3_bluffs_a_won += 1
        for a in result.get("actions_b", []):
            if a == "bluff":
                p3_bluffs_b += 1
                if result["winner"] == 1:
                    p3_bluffs_b_won += 1

        if len(recent_results) > window:
            recent_results = recent_results[-window:]
        if len(recent_results) >= window:
            rolling_wr_a.append(sum(recent_results) / len(recent_results))

        if (hand_num + 1) % 100 == 0:
            field_a.evolve(lr=0.03, cap=0.04)
            field_b.evolve(lr=0.03, cap=0.04)

            dist = compute_strategy_distance(field_a, field_b)
            p3_convergence.append(dist)

            wr_a = p3_a_wins / (hand_num + 1)
            wr_b = p3_b_wins / (hand_num + 1)
            print(f"  Hand {hand_num+1:4d}: A_wr={wr_a:.1%} B_wr={wr_b:.1%} "
                  f"bluffs A={p3_bluffs_a} B={p3_bluffs_b} "
                  f"dist={dist:.2f} "
                  f"recent_wr={rolling_wr_a[-1]:.1%}" if rolling_wr_a else
                  f"  Hand {hand_num+1:4d}: A_wr={wr_a:.1%} B_wr={wr_b:.1%} "
                  f"bluffs A={p3_bluffs_a} B={p3_bluffs_b} "
                  f"dist={dist:.2f}")

    results["phase3"] = {
        "a_wins": p3_a_wins, "b_wins": p3_b_wins, "ties": p3_ties,
        "a_wr": round(p3_a_wins / 1000, 3),
        "b_wr": round(p3_b_wins / 1000, 3),
        "folds_won_by_a": p3_folds_a, "folds_won_by_b": p3_folds_b,
        "showdowns": p3_showdowns,
        "bluffs_by_a": p3_bluffs_a, "bluffs_by_b": p3_bluffs_b,
        "bluff_wr_a": round(p3_bluffs_a_won / max(p3_bluffs_a, 1), 3),
        "bluff_wr_b": round(p3_bluffs_b_won / max(p3_bluffs_b, 1), 3),
        "avg_pot_won_a": round(p3_pot_a / max(p3_a_wins, 1), 2),
        "avg_pot_won_b": round(p3_pot_b / max(p3_b_wins, 1), 2),
        "convergence_distances": p3_convergence,
        "rolling_wr_a": rolling_wr_a,
    }

    print(f"\n  Phase 3 Summary:")
    print(f"    A wins: {p3_a_wins} ({p3_a_wins/10:.1f}%)  B wins: {p3_b_wins} ({p3_b_wins/10:.1f}%)  Ties: {p3_ties}")
    print(f"    A bluffs: {p3_bluffs_a} (wr: {p3_bluffs_a_won/max(p3_bluffs_a,1):.1%})  "
          f"B bluffs: {p3_bluffs_b} (wr: {p3_bluffs_b_won/max(p3_bluffs_b,1):.1%})")
    print(f"    A avg pot won: {p3_pot_a/max(p3_a_wins,1):.1f}  B avg pot won: {p3_pot_b/max(p3_b_wins,1):.1f}")

    # ─── CONVERGENCE ANALYSIS ────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("CONVERGENCE ANALYSIS")
    print("=" * 72)

    all_distances = (results["phase1"]["convergence_distances"] +
                     results["phase2"]["convergence_distances"] +
                     results["phase3"]["convergence_distances"])

    # Check if distance is decreasing (converging) or oscillating
    if len(all_distances) >= 6:
        first_half = all_distances[:len(all_distances)//2]
        second_half = all_distances[len(all_distances)//2:]
        avg_first = np.mean(first_half)
        avg_second = np.mean(second_half)
        trend = "CONVERGING" if avg_second < avg_first else "DIVERGING"

        # Oscillation check: count sign changes in diffs
        diffs = np.diff(all_distances)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        oscillation_ratio = sign_changes / max(len(diffs) - 1, 1)

        print(f"  Strategy distance trend: {trend}")
        print(f"  Avg distance first half:  {avg_first:.3f}")
        print(f"  Avg distance second half: {avg_second:.3f}")
        print(f"  Oscillation ratio: {oscillation_ratio:.2f} "
              f"({'HIGH oscillation' if oscillation_ratio > 0.5 else 'LOW oscillation'})")
        print(f"  Final distance: {all_distances[-1]:.3f}")

        results["convergence"] = {
            "trend": trend,
            "avg_first_half": round(float(avg_first), 3),
            "avg_second_half": round(float(avg_second), 3),
            "oscillation_ratio": round(float(oscillation_ratio), 3),
            "final_distance": round(all_distances[-1], 3),
            "all_distances": [round(d, 3) for d in all_distances],
        }

    # Check Nash-like equilibrium: rolling win rates in phase 3
    if rolling_wr_a:
        wr_std = np.std(rolling_wr_a)
        wr_mean = np.mean(rolling_wr_a)
        nash_like = wr_std < 0.05 and abs(wr_mean - 0.5) < 0.05
        print(f"\n  Phase 3 rolling win rate (A): mean={wr_mean:.3f} std={wr_std:.3f}")
        print(f"  Nash-like equilibrium (std<0.05, |mean-0.5|<0.05): {'YES' if nash_like else 'NO'}")

        results["convergence"]["rolling_wr_mean"] = round(float(wr_mean), 3)
        results["convergence"]["rolling_wr_std"] = round(float(wr_std), 3)
        results["convergence"]["nash_like"] = nash_like

    # ─── STRATEGY ANALYSIS ───────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("STRATEGY EVOLUTION")
    print("=" * 72)

    analysis_a = analyze_field(field_a, "Player A")
    analysis_b = analyze_field(field_b, "Player B")

    for ana in [analysis_a, analysis_b]:
        print(f"\n  {ana['name']}: {ana['tiles']} tiles, {ana['total_visits']} total visits")
        print(f"    Bluff usage: {ana['bluff_analysis']['total_bluffs']} "
              f"(win rate: {ana['bluff_analysis']['bluff_wr']:.1%})")
        print(f"    Top reflexes by stage:")
        for stage, action in ana["top_reflexes"].items():
            detail = ana["by_stage"].get(stage, {}).get(action, {})
            print(f"      {stage:8s}: {action:12s} (score={detail.get('avg_score',0):.3f}, "
                  f"wr={detail.get('wr',0):.3f}, chosen={detail.get('times_chosen',0)})")

    results["strategy_a"] = analysis_a
    results["strategy_b"] = analysis_b

    # ─── CROSS-PHASE COMPARISON ──────────────────────────────────────────
    print("\n" + "=" * 72)
    print("CROSS-PHASE WIN RATES")
    print("=" * 72)

    phases = [
        ("Phase 1 (both default)", results["phase1"]),
        ("Phase 2 (A learned)", results["phase2"]),
        ("Phase 3 (arms race)", results["phase3"]),
    ]
    for label, data in phases:
        print(f"  {label:30s}: A={data['a_wr']:.1%} B={data['b_wr']:.1%} "
              f"ties={data['ties']}")

    # ─── SAVE ────────────────────────────────────────────────────────────
    out_path = os.path.expanduser("~/repos/zeroclaw-arena/holdem-rival-results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    run_rival_arena()
