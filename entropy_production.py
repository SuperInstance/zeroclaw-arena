"""
Entropy Production Theorem — Does training INCREASE or DECREASE total score entropy?

For each game (TTT, C4, Hold'em), we train a tile field and measure:
1. Shannon entropy of the score distribution at regular intervals
2. Conditional entropy H(negative space | positive space)
3. Mutual information I(negative space; positive space)
4. How these evolve over training

Key question: does training produce or destroy information?
- If entropy DECREASES: learning = compression (fitting a model)
- If entropy INCREASES: learning = differentiation (creating structure)
- If entropy is conserved: learning = redistribution (same info, different shape)
"""

import random
import numpy as np
import json
import os
import sys
import math
from collections import defaultdict, Counter
from copy import deepcopy

sys.path.insert(0, os.path.dirname(__file__))
from zeroclaw import TicTacToe, Connect4, ZeroClaw, StateTile
from holdem_tile import PokerTileField, HoldemHand


# ─── Entropy Measurement Utilities ─────────────────────────

def shannon_entropy(scores: list[float], bins: int = 20) -> float:
    """Shannon entropy of a continuous score distribution, discretized into bins."""
    if len(scores) < 2:
        return 0.0
    hist, _ = np.histogram(scores, bins=bins, range=(0, 1))
    probs = hist / hist.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def score_entropy_from_reflexes(reflexes: dict) -> float:
    """Entropy of the score distribution across reflexes of a single tile."""
    if not reflexes:
        return 0.0
    scores = [d["score"] for d in reflexes.values()]
    if len(scores) < 2:
        return 0.0
    # Use scores as a probability distribution (normalize)
    scores_arr = np.array(scores)
    total = scores_arr.sum()
    if total == 0:
        return 0.0
    probs = scores_arr / total
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def partition_scores(tile_field: dict) -> tuple[list[float], list[float]]:
    """Partition all scores into positive (>0.5) and negative (<=0.5) space."""
    positive = []
    negative = []
    for tile in tile_field.values():
        for action, data in tile.reflexes.items():
            s = data["score"]
            if s > 0.5:
                positive.append(s)
            else:
                negative.append(s)
    return positive, negative


def joint_entropy_2d(positive: list[float], negative: list[float],
                     bins: int = 10) -> float:
    """Joint entropy H(P, N) of positive and negative score distributions."""
    if not positive or not negative:
        return 0.0
    # Create a 2D histogram by pairing (we need matched pairs)
    # Since they come from different tiles, we'll use the marginal distributions
    # and compute H(P,N) = H(P) + H(N|P)
    # For simplicity: use concatenated distribution entropy
    all_scores = positive + negative
    return shannon_entropy(all_scores, bins=bins)


def conditional_entropy(positive: list[float], negative: list[float],
                        bins: int = 10) -> float:
    """H(N|P) = H(P,N) - H(P). Approximate via discretized distributions."""
    if not positive or not negative:
        return 0.0
    h_pos = shannon_entropy(positive, bins=bins)
    h_all = shannon_entropy(positive + negative, bins=bins)
    return max(0.0, h_all - h_pos)


def mutual_information(positive: list[float], negative: list[float],
                       bins: int = 10) -> float:
    """I(N;P) = H(N) + H(P) - H(P,N)."""
    if not positive or not negative:
        return 0.0
    h_pos = shannon_entropy(positive, bins=bins)
    h_neg = shannon_entropy(negative, bins=bins)
    h_joint = shannon_entropy(positive + negative, bins=bins)
    return max(0.0, h_neg + h_pos - h_joint)


def tile_field_entropy_snapshot(tile_field: dict) -> dict:
    """Take a full entropy snapshot of a tile field."""
    all_scores = []
    per_tile_entropies = []
    n_reflexes_per_tile = []

    for tile in tile_field.values():
        scores = [d["score"] for d in tile.reflexes.values()]
        all_scores.extend(scores)
        per_tile_entropies.append(score_entropy_from_reflexes(tile.reflexes))
        n_reflexes_per_tile.append(len(tile.reflexes))

    positive, negative = partition_scores(tile_field)

    if not all_scores:
        return {
            "n_tiles": 0,
            "n_scores": 0,
            "shannon_entropy": 0.0,
            "score_mean": 0.0,
            "score_std": 0.0,
            "avg_tile_entropy": 0.0,
            "positive_count": 0,
            "negative_count": 0,
            "h_positive": 0.0,
            "h_negative": 0.0,
            "h_joint": 0.0,
            "conditional_entropy": 0.0,
            "mutual_information": 0.0,
        }

    scores_arr = np.array(all_scores)
    return {
        "n_tiles": len(tile_field),
        "n_scores": len(all_scores),
        "shannon_entropy": shannon_entropy(all_scores, bins=20),
        "score_mean": float(scores_arr.mean()),
        "score_std": float(scores_arr.std()),
        "avg_tile_entropy": float(np.mean(per_tile_entropies)) if per_tile_entropies else 0.0,
        "positive_count": len(positive),
        "negative_count": len(negative),
        "h_positive": shannon_entropy(positive, bins=10) if positive else 0.0,
        "h_negative": shannon_entropy(negative, bins=10) if negative else 0.0,
        "h_joint": joint_entropy_2d(positive, negative, bins=10),
        "conditional_entropy": conditional_entropy(positive, negative, bins=10),
        "mutual_information": mutual_information(positive, negative, bins=10),
    }


# ─── TTT & C4 Training with Snapshots ──────────────────────

def train_ttt_with_snapshots(num_games: int = 1000, interval: int = 50) -> list[dict]:
    """Train TTT tile field and take entropy snapshots at intervals."""
    random.seed(42)
    np.random.seed(42)

    game = TicTacToe()
    claw = ZeroClaw("entropy-ttt", "tictactoe",
                    sandbox_dir=f"/tmp/entropy-production/ttt")

    snapshots = []

    for batch in range(0, num_games, interval):
        # Train a batch
        claw.explore_tile_field(game, num_games=interval, n_simulations=20)

        # Take snapshot
        snap = tile_field_entropy_snapshot(claw.tile_field)
        snap["games_played"] = batch + interval
        snap["game"] = "TTT"
        snapshots.append(snap)
        print(f"  TTT {batch + interval}/{num_games}: H={snap['shannon_entropy']:.4f} "
              f"MI={snap['mutual_information']:.4f} tiles={snap['n_tiles']}")

    return snapshots


def train_c4_with_snapshots(num_games: int = 1000, interval: int = 50) -> list[dict]:
    """Train Connect4 tile field and take entropy snapshots at intervals."""
    random.seed(42)
    np.random.seed(42)

    game = Connect4()
    claw = ZeroClaw("entropy-c4", "connect4",
                    sandbox_dir=f"/tmp/entropy-production/c4")

    snapshots = []

    for batch in range(0, num_games, interval):
        claw.explore_tile_field(game, num_games=interval, n_simulations=20)

        snap = tile_field_entropy_snapshot(claw.tile_field)
        snap["games_played"] = batch + interval
        snap["game"] = "C4"
        snapshots.append(snap)
        print(f"  C4 {batch + interval}/{num_games}: H={snap['shannon_entropy']:.4f} "
              f"MI={snap['mutual_information']:.4f} tiles={snap['n_tiles']}")

    return snapshots


# ─── Hold'em Training with Snapshots ──────────────────────

def train_holdem_with_snapshots(num_games: int = 1000, interval: int = 50) -> list[dict]:
    """Train Hold'em tile field and take entropy snapshots at intervals."""
    random.seed(42)
    np.random.seed(42)

    field = PokerTileField()
    snapshots = []

    for batch in range(0, num_games, interval):
        for _ in range(interval):
            hand = HoldemHand()
            result = hand.play(field, opponent_strategy="random")

            # Evolve periodically
            if hand.turn % 5 == 0:
                field.evolve()

        # Take snapshot from poker tiles
        all_scores = []
        per_tile_entropies = []

        for key, tile in field.tiles.items():
            for action, data in tile.reflexes.items():
                all_scores.append(data["score"])
            per_tile_entropies.append(score_entropy_from_reflexes(tile.reflexes))

        positive = [s for s in all_scores if s > 0.5]
        negative = [s for s in all_scores if s <= 0.5]

        if not all_scores:
            snap = {"n_tiles": 0, "n_scores": 0, "shannon_entropy": 0.0,
                    "score_mean": 0.0, "score_std": 0.0,
                    "avg_tile_entropy": 0.0,
                    "positive_count": 0, "negative_count": 0,
                    "h_positive": 0.0, "h_negative": 0.0,
                    "h_joint": 0.0, "conditional_entropy": 0.0,
                    "mutual_information": 0.0}
        else:
            scores_arr = np.array(all_scores)
            snap = {
                "n_tiles": len(field.tiles),
                "n_scores": len(all_scores),
                "shannon_entropy": shannon_entropy(all_scores, bins=20),
                "score_mean": float(scores_arr.mean()),
                "score_std": float(scores_arr.std()),
                "avg_tile_entropy": float(np.mean(per_tile_entropies)) if per_tile_entropies else 0.0,
                "positive_count": len(positive),
                "negative_count": len(negative),
                "h_positive": shannon_entropy(positive, bins=10) if positive else 0.0,
                "h_negative": shannon_entropy(negative, bins=10) if negative else 0.0,
                "h_joint": joint_entropy_2d(positive, negative, bins=10),
                "conditional_entropy": conditional_entropy(positive, negative, bins=10),
                "mutual_information": mutual_information(positive, negative, bins=10),
            }

        snap["games_played"] = batch + interval
        snap["game"] = "Holdem"
        snapshots.append(snap)
        print(f"  Hold'em {batch + interval}/{num_games}: H={snap['shannon_entropy']:.4f} "
              f"MI={snap['mutual_information']:.4f} tiles={snap['n_tiles']}")

    return snapshots


# ─── Results Analysis & Display ─────────────────────────────

def analyze_entropy_trajectory(snapshots: list[dict], game_name: str) -> dict:
    """Analyze the entropy trajectory and classify the learning behavior."""
    if len(snapshots) < 2:
        return {"game": game_name, "verdict": "insufficient_data"}

    entropies = [s["shannon_entropy"] for s in snapshots]
    mi_values = [s["mutual_information"] for s in snapshots]
    cond_entropies = [s["conditional_entropy"] for s in snapshots]
    avg_tile_entropies = [s["avg_tile_entropy"] for s in snapshots]

    # Compute delta: first half vs second half
    mid = len(entropies) // 2
    h_first = np.mean(entropies[:mid]) if mid > 0 else entropies[0]
    h_second = np.mean(entropies[mid:]) if mid < len(entropies) else entropies[-1]
    delta_h = h_second - h_first

    mi_first = np.mean(mi_values[:mid]) if mid > 0 else mi_values[0]
    mi_second = np.mean(mi_values[mid:]) if mid < len(mi_values) else mi_values[-1]
    delta_mi = mi_second - mi_first

    # Classify
    if delta_h < -0.05:
        verdict = "COMPRESSION — entropy decreased, learning fits a model"
    elif delta_h > 0.05:
        verdict = "DIFFERENTIATION — entropy increased, learning creates structure"
    else:
        if delta_mi > 0.05:
            verdict = "RESTRUCTURING — entropy conserved, mutual information grew"
        else:
            verdict = "CONSERVATION — entropy stable, information redistributed"

    return {
        "game": game_name,
        "entropy_start": entropies[0],
        "entropy_end": entropies[-1],
        "entropy_delta": delta_h,
        "entropy_trend": "decreasing" if delta_h < -0.01 else "increasing" if delta_h > 0.01 else "stable",
        "mi_start": mi_values[0],
        "mi_end": mi_values[-1],
        "mi_delta": delta_mi,
        "conditional_entropy_start": cond_entropies[0],
        "conditional_entropy_end": cond_entropies[-1],
        "avg_tile_entropy_start": avg_tile_entropies[0],
        "avg_tile_entropy_end": avg_tile_entropies[-1],
        "verdict": verdict,
    }


def print_entropy_table(snapshots: list[dict], game_name: str):
    """Print a formatted table of entropy vs training progress."""
    print(f"\n{'='*80}")
    print(f"  ENTROPY PRODUCTION: {game_name}")
    print(f"{'='*80}")
    print(f"{'Games':>6} | {'H(score)':>10} | {'H(pos)':>8} | {'H(neg)':>8} | "
          f"{'H(N|P)':>8} | {'I(N;P)':>8} | {'<H_tile>':>10} | {'Tiles':>6}")
    print(f"{'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*6}")

    for s in snapshots:
        print(f"{s['games_played']:>6} | {s['shannon_entropy']:>10.4f} | "
              f"{s['h_positive']:>8.4f} | {s['h_negative']:>8.4f} | "
              f"{s['conditional_entropy']:>8.4f} | {s['mutual_information']:>8.4f} | "
              f"{s['avg_tile_entropy']:>10.4f} | {s['n_tiles']:>6}")


def print_summary(analyses: list[dict]):
    """Print the grand summary with verdict."""
    print(f"\n{'='*80}")
    print(f"  ENTROPY PRODUCTION THEOREM — RESULTS")
    print(f"{'='*80}")
    print(f"{'Game':>10} | {'H_start':>8} | {'H_end':>8} | {'ΔH':>8} | "
          f"{'MI_start':>8} | {'MI_end':>8} | {'ΔMI':>8} | Verdict")
    print(f"{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*30}")

    for a in analyses:
        print(f"{a['game']:>10} | {a['entropy_start']:>8.4f} | {a['entropy_end']:>8.4f} | "
              f"{a['entropy_delta']:>+8.4f} | {a['mi_start']:>8.4f} | {a['mi_end']:>8.4f} | "
              f"{a['mi_delta']:>+8.4f} | {a['verdict'][:40]}")

    print(f"\n{'='*80}")
    # Overall verdict
    deltas = [a['entropy_delta'] for a in analyses]
    avg_delta = np.mean(deltas)
    if avg_delta < -0.05:
        overall = "COMPRESSION — Learning decreases entropy. The tile field compresses game knowledge."
    elif avg_delta > 0.05:
        overall = "DIFFERENTIATION — Learning increases entropy. The tile field creates structure."
    else:
        overall = "CONSERVATION/RESTRUCTURING — Learning preserves entropy while reorganizing information."
    print(f"  OVERALL: {overall}")
    print(f"  Average ΔH = {avg_delta:+.4f}")
    print(f"{'='*80}")


# ─── Main ───────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  ENTROPY PRODUCTION THEOREM EXPERIMENT")
    print("  Does training INCREASE or DECREASE total score entropy?")
    print("=" * 80)

    NUM_GAMES = 1000
    INTERVAL = 50
    all_results = {}
    all_analyses = []

    # TTT
    print("\n>>> Tic-Tac-Toe Training")
    ttt_snaps = train_ttt_with_snapshots(NUM_GAMES, INTERVAL)
    print_entropy_table(ttt_snaps, "Tic-Tac-Toe")
    ttt_analysis = analyze_entropy_trajectory(ttt_snaps, "TTT")
    all_analyses.append(ttt_analysis)
    all_results["ttt"] = {"snapshots": ttt_snaps, "analysis": ttt_analysis}

    # C4
    print("\n>>> Connect-4 Training")
    c4_snaps = train_c4_with_snapshots(NUM_GAMES, INTERVAL)
    print_entropy_table(c4_snaps, "Connect-4")
    c4_analysis = analyze_entropy_trajectory(c4_snaps, "C4")
    all_analyses.append(c4_analysis)
    all_results["c4"] = {"snapshots": c4_snaps, "analysis": c4_analysis}

    # Hold'em
    print("\n>>> Texas Hold'em Training")
    holdem_snaps = train_holdem_with_snapshots(NUM_GAMES, INTERVAL)
    print_entropy_table(holdem_snaps, "Texas Hold'em")
    holdem_analysis = analyze_entropy_trajectory(holdem_snaps, "Holdem")
    all_analyses.append(holdem_analysis)
    all_results["holdem"] = {"snapshots": holdem_snaps, "analysis": holdem_analysis}

    # Grand Summary
    print_summary(all_analyses)

    # Save results
    results_path = os.path.join(os.path.dirname(__file__), "entropy-production-results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
