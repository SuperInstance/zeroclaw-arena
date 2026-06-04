"""
Tile Conservation Law — Connect4 Edition

Tests the "degenerate positive space" hypothesis:
Connect4 has LESS symmetry than tic-tac-toe (center column is genuinely best
in most positions, not just one of several equally-good options).

If the conservation law depends on game structure:
- Connect4 should have HIGHER top-reflex agreement than TTT (41%)
- Fewer equally-good moves → more unique optimal policies
- Target: >60% reflex agreement on C4

Runs tile exploration on Connect4 5 times (different random seeds),
300 games per run, then compares score distributions, top reflexes, and correlations.
"""

import random
import numpy as np
import json
import os
import sys
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(__file__))
from zeroclaw import Connect4, ZeroClaw


def run_tile_exploration(seed: int, num_games: int = 300) -> dict:
    """Run tile exploration with a given random seed on Connect4."""
    random.seed(seed)
    np.random.seed(seed)

    game = Connect4()
    claw = ZeroClaw(f"c4-conservation-seed-{seed}", "connect4",
                    sandbox_dir=f"/tmp/tile-conservation-c4/seed-{seed}")

    # Run tile-field exploration
    claw.explore_tile_field(game, num_games=num_games, n_simulations=20)

    # Collect stats from tile field
    tile_field = claw.tile_field
    if not tile_field:
        return {"error": "no tiles learned", "seed": seed}

    # Score distribution across all reflexes
    all_scores = []
    top_reflexes_per_tile = {}
    action_rankings = defaultdict(list)

    for state_hash, tile in tile_field.items():
        for action, data in tile.reflexes.items():
            all_scores.append(data["score"])

        best = max(tile.reflexes.items(), key=lambda x: x[1]["score"])
        top_reflexes_per_tile[state_hash] = (best[0], best[1]["score"])

        ranked = sorted(tile.reflexes.items(), key=lambda x: -x[1]["score"])
        action_rankings[state_hash] = [(a, d["score"]) for a, d in ranked]

    scores = np.array(all_scores)

    return {
        "seed": seed,
        "num_tiles": len(tile_field),
        "score_distribution": {
            "min": float(scores.min()),
            "max": float(scores.max()),
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "median": float(np.median(scores)),
            "p25": float(np.percentile(scores, 25)),
            "p75": float(np.percentile(scores, 75)),
            "p90": float(np.percentile(scores, 90)),
            "p10": float(np.percentile(scores, 10)),
        },
        "top_reflexes": top_reflexes_per_tile,
        "action_rankings": action_rankings,
        "all_scores": all_scores,
        "tile_hashes": set(tile_field.keys()),
    }


def compare_runs(results: list[dict]) -> dict:
    """Compare multiple runs for convergence."""
    valid = [r for r in results if "error" not in r]
    if len(valid) < 2:
        return {"error": "not enough valid runs"}

    # 1. Score distribution comparison
    print("\n" + "=" * 70)
    print("SCORE DISTRIBUTION COMPARISON (Connect4)")
    print("=" * 70)

    dists = [r["score_distribution"] for r in valid]
    for i, (r, d) in enumerate(zip(valid, dists)):
        print(f"\n  Run {i+1} (seed={r['seed']}): tiles={r['num_tiles']}")
        print(f"    Mean={d['mean']:.4f}  Std={d['std']:.4f}  "
              f"Min={d['min']:.4f}  Max={d['max']:.4f}")
        print(f"    Median={d['median']:.4f}  "
              f"P25={d['p25']:.4f}  P75={d['p75']:.4f}")

    means = [d["mean"] for d in dists]
    stds = [d["std"] for d in dists]
    medians = [d["median"] for d in dists]

    mean_of_means = np.mean(means)
    std_of_means = np.std(means)
    mean_of_stds = np.mean(stds)
    std_of_stds = np.std(stds)

    print(f"\n  Cross-run convergence:")
    print(f"    Means:     {mean_of_means:.4f} ± {std_of_means:.4f}  "
          f"(CV={std_of_means/max(mean_of_means,1e-10):.4f})")
    print(f"    StdDevs:   {mean_of_stds:.4f} ± {std_of_stds:.4f}  "
          f"(CV={std_of_stds/max(mean_of_stds,1e-10):.4f})")
    print(f"    Medians:   {np.mean(medians):.4f} ± {np.std(medians):.4f}")

    # 2. Top reflex agreement (conservation test)
    print("\n" + "=" * 70)
    print("TOP REFLEX AGREEMENT (Conservation Test — Connect4)")
    print("=" * 70)

    all_tile_sets = [r["tile_hashes"] for r in valid]
    shared_tiles = all_tile_sets[0]
    for s in all_tile_sets[1:]:
        shared_tiles = shared_tiles & s

    print(f"\n  Tile overlap: {len(shared_tiles)} shared states across all runs")
    print(f"  Per-run tile counts: {[len(s) for s in all_tile_sets]}")

    overall_agreement = 0
    full_agreement_pct = 0

    if shared_tiles:
        agreement_count = 0
        disagreement_examples = []
        state_agreements = []

        for state_hash in shared_tiles:
            top_actions = []
            for r in valid:
                if state_hash in r["top_reflexes"]:
                    top_actions.append(r["top_reflexes"][state_hash][0])

            if top_actions:
                most_common = Counter(top_actions).most_common(1)[0]
                agreement = most_common[1] / len(top_actions)
                state_agreements.append(agreement)
                if agreement == 1.0:
                    agreement_count += 1
                elif len(disagreement_examples) < 5:
                    disagreement_examples.append({
                        "state": state_hash,
                        "top_actions": top_actions,
                    })

        overall_agreement = np.mean(state_agreements) if state_agreements else 0
        full_agreement_pct = agreement_count / len(shared_tiles) * 100

        print(f"\n  States where ALL runs agree on best action: "
              f"{agreement_count}/{len(shared_tiles)} ({full_agreement_pct:.1f}%)")
        print(f"  Average agreement rate: {overall_agreement:.1%}")

        if disagreement_examples:
            print(f"\n  Disagreement examples (first {len(disagreement_examples)}):")
            for ex in disagreement_examples:
                print(f"    State {ex['state'][:12]}...: top actions = {ex['top_actions']}")

    # 3. Action ranking correlation (Pearson r)
    print("\n" + "=" * 70)
    print("ACTION RANKING CORRELATION (Pearson r)")
    print("=" * 70)

    tau_scores = []
    if shared_tiles:
        shared_list = list(shared_tiles)[:30]

        for state_hash in shared_list:
            rankings = []
            for r in valid:
                if state_hash in r["action_rankings"]:
                    rankings.append(r["action_rankings"][state_hash])

            if len(rankings) >= 2:
                r1_actions = {a: s for a, s in rankings[0]}
                r2_actions = {a: s for a, s in rankings[1]}
                common_actions = set(r1_actions.keys()) & set(r2_actions.keys())

                if len(common_actions) >= 2:
                    s1 = np.array([r1_actions[a] for a in sorted(common_actions)])
                    s2 = np.array([r2_actions[a] for a in sorted(common_actions)])
                    corr = np.corrcoef(s1, s2)[0, 1]
                    if not np.isnan(corr):
                        tau_scores.append(corr)

    if tau_scores:
        print(f"\n  Score correlation across runs (Pearson r):")
        print(f"    Mean: {np.mean(tau_scores):.4f}")
        print(f"    Std:  {np.std(tau_scores):.4f}")
        print(f"    Min:  {np.min(tau_scores):.4f}")
        print(f"    Max:  {np.max(tau_scores):.4f}")

    # 4. Tile count stability
    print("\n" + "=" * 70)
    print("TILE COUNT STABILITY")
    print("=" * 70)

    tile_counts = [r["num_tiles"] for r in valid]
    print(f"\n  Tile counts: {tile_counts}")
    print(f"  Mean: {np.mean(tile_counts):.1f} ± {np.std(tile_counts):.1f}")
    print(f"  Range: {min(tile_counts)} - {max(tile_counts)}")

    # 5. Compare with TTT baseline
    print("\n" + "=" * 70)
    print("CROSS-GAME COMPARISON (Connect4 vs Tic-Tac-Toe)")
    print("=" * 70)

    TTT_AGREEMENT = 0.4128  # from tile-conservation-results.json
    print(f"\n  TTT top-reflex agreement:    {TTT_AGREEMENT:.1%}")
    print(f"  C4  top-reflex agreement:    {overall_agreement:.1%}")
    delta = overall_agreement - TTT_AGREEMENT
    print(f"  Delta:                       {delta:+.1%}")

    if overall_agreement > 0.60:
        print(f"\n  ★ HYPOTHESIS CONFIRMED: C4 agreement ({overall_agreement:.1%}) > 60%")
        print(f"    Less symmetric game → more unique optimal policies.")
        print(f"    The conservation law depends on game structure.")
    elif overall_agreement > TTT_AGREEMENT:
        print(f"\n  ◆ PARTIAL SUPPORT: C4 agreement ({overall_agreement:.1%}) > TTT ({TTT_AGREEMENT:.1%})")
        print(f"    But below the 60% target. Direction supports hypothesis.")
    else:
        print(f"\n  ✗ HYPOTHESIS REJECTED: C4 agreement ({overall_agreement:.1%}) ≤ TTT ({TTT_AGREEMENT:.1%})")
        print(f"    More complex game does NOT produce more agreement.")

    # 6. Conservation verdict
    print("\n" + "=" * 70)
    print("CONSERVATION LAW VERDICT (Connect4)")
    print("=" * 70)

    score_converges = std_of_means < 0.05
    reflex_converges = overall_agreement > 0.7 if shared_tiles else False
    count_stable = np.std(tile_counts) / max(np.mean(tile_counts), 1) < 0.3

    print(f"\n  Score distributions converge:     {'YES ✓' if score_converges else 'NO ✗'}"
          f"  (std_of_means={std_of_means:.4f})")
    print(f"  Top reflexes converge:            {'YES ✓' if reflex_converges else 'NO ✗'}"
          f"  (agreement={overall_agreement:.1%})")
    print(f"  Tile counts are stable:           {'YES ✓' if count_stable else 'NO ✗'}"
          f"  (cv={np.std(tile_counts)/max(np.mean(tile_counts),1):.2f})")

    if score_converges and reflex_converges:
        print("\n  ★ CONSERVATION LAW HOLDS: The tile field converges to a UNIQUE solution.")
    elif score_converges and not reflex_converges:
        print("\n  ◆ PARTIAL CONSERVATION: Scores converge but optimal strategies differ.")
        print("    The tile field has multiple equivalent configurations (degenerate).")
    elif not score_converges and reflex_converges:
        print("\n  ◆ PARTIAL CONSERVATION: Reflexes agree but score magnitudes vary.")
    else:
        print("\n  ✗ NO CONSERVATION: The tile field produces different solutions each time.")

    return {
        "score_converges": bool(score_converges),
        "reflex_converges": bool(reflex_converges),
        "count_stable": bool(count_stable),
        "mean_agreement": float(overall_agreement),
        "full_agreement_pct": float(full_agreement_pct),
        "std_of_means": float(std_of_means),
        "mean_correlation": float(np.mean(tau_scores)) if tau_scores else None,
        "c4_vs_ttt_delta": float(delta),
        "hypothesis_confirmed": bool(overall_agreement > 0.60),
        "hypothesis_direction": bool(overall_agreement > TTT_AGREEMENT),
    }


def main():
    print("=" * 70)
    print("TILE CONSERVATION LAW EXPERIMENT — CONNECT4")
    print("Testing: Does less symmetry → more unique optimal policy?")
    print("=" * 70)
    print(f"\n  Baseline: TTT top-reflex agreement = 41.3%")
    print(f"  Target:   C4  top-reflex agreement > 60%")

    seeds = [42, 1337, 9999, 7, 2024]
    num_games = 300
    results = []

    for seed in seeds:
        print(f"\n{'─' * 50}")
        print(f"Running seed={seed} ({num_games} games of Connect4)...")
        result = run_tile_exploration(seed, num_games)
        results.append(result)
        if "error" not in result:
            d = result["score_distribution"]
            print(f"  Done: {result['num_tiles']} tiles, "
                  f"score mean={d['mean']:.4f} std={d['std']:.4f}")
        else:
            print(f"  ERROR: {result['error']}")

    # Compare
    verdict = compare_runs(results)

    # Save
    output = {
        "experiment": "tile_conservation_law_connect4",
        "hypothesis": "less_symmetric_game_more_unique_policy",
        "baseline_ttt_agreement": 0.4128,
        "seeds": seeds,
        "num_games": num_games,
        "verdict": verdict,
        "run_summaries": [
            {
                "seed": r["seed"],
                "num_tiles": r.get("num_tiles"),
                "score_distribution": r.get("score_distribution"),
            }
            for r in results
            if "error" not in r
        ],
    }

    out_path = os.path.expanduser(
        "~/repos/zeroclaw-arena/tile-conservation-connect4-results.json"
    )
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
