"""
EXPERIMENT: Temperature Sweep — What's the optimal softmax temperature?

Hypothesis: The default T=0.3 may not be optimal. Lower T gives sharper
exploitation but less exploration; higher T explores more but may waste
training on bad moves. There may be an ideal exploration T that differs
from the best eval T, justifying annealing schedules.

Protocol:
- Train TTT tile fields at temperatures: [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 5.0]
- For each T: train 300 games, evaluate vs random (500 games)
- Evaluate at both training T AND greedy T=0.01
- Measure: win rates, unique tiles discovered, score distribution entropy
"""

import random
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zeroclaw.tile_field import TileField
from zeroclaw.games import TicTacToe

TEMPERATURES = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 5.0]
TRAIN_GAMES = 300
EVAL_GAMES = 500
GREEDY_T = 0.01


def play_vs_random(field: TileField, game: TicTacToe, player_field: str = 'X'):
    """Play one game: field uses its policy, opponent plays random."""
    g = TicTacToe()
    history = []

    while not g.done:
        actions = g.legal_actions()
        if not actions:
            break

        if g.current == player_field:
            action = field.choose_action(g, g.state().state_str, actions)
            history.append((g.state().state_str, action))
        else:
            action = random.choice(actions)

        g.step(action)

    return g.winner, history


def score_entropy(field: TileField) -> float:
    """Compute entropy of the score distribution across all tiles."""
    scores = []
    for tile in field.tiles.values():
        for action_data in tile.values():
            scores.append(action_data["score"])

    if not scores:
        return 0.0

    # Bin scores into 20 buckets [0, 0.05, ..., 1.0]
    n_bins = 20
    counts = [0] * n_bins
    for s in scores:
        idx = min(int(s * n_bins), n_bins - 1)
        counts[idx] += 1

    total = sum(counts)
    if total == 0:
        return 0.0

    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)

    return entropy


def run_temperature_experiment(temp: float) -> dict:
    """Train and evaluate at a single temperature."""
    print(f"\n{'='*60}")
    print(f"Temperature T={temp}")
    print(f"{'='*60}")

    # --- Train ---
    field = TileField(n_simulations=20, temperature=temp)
    print(f"  Training {TRAIN_GAMES} games at T={temp}...")
    train_wins = {'X': 0, 'O': 0, 'draw': 0}
    for i in range(TRAIN_GAMES):
        g = TicTacToe()
        winner = field.train_game(g)
        if winner in train_wins:
            train_wins[winner] += 1
        else:
            train_wins['draw'] += 1
        if (i + 1) % 100 == 0:
            total = sum(train_wins.values())
            wr = (train_wins['X'] + train_wins.get('B', 0)) / total
            print(f"    {i+1}/{TRAIN_GAMES} | P1 win rate={wr:.1%} | tiles={len(field.tiles)}")

    field.evolve()
    tiles_after_train = len(field.tiles)
    entropy_after_train = score_entropy(field)

    # --- Evaluate at training T ---
    print(f"  Evaluating {EVAL_GAMES} games at T={temp} (training temperature)...")
    eval_wins_train_t = {'X': 0, 'O': 0, 'draw': 0}
    for _ in range(EVAL_GAMES):
        winner, _ = play_vs_random(field, TicTacToe())
        if winner in eval_wins_train_t:
            eval_wins_train_t[winner] += 1
        else:
            eval_wins_train_t['draw'] += 1

    total_eval = sum(eval_wins_train_t.values())
    wr_at_train_t = (eval_wins_train_t['X'] + eval_wins_train_t.get('B', 0)) / total_eval

    # --- Evaluate at greedy T=0.01 ---
    print(f"  Evaluating {EVAL_GAMES} games at T={GREEDY_T} (greedy exploitation)...")
    # Save original temp, set greedy
    original_temp = field.temperature
    field.temperature = GREEDY_T

    eval_wins_greedy = {'X': 0, 'O': 0, 'draw': 0}
    for _ in range(EVAL_GAMES):
        winner, _ = play_vs_random(field, TicTacToe())
        if winner in eval_wins_greedy:
            eval_wins_greedy[winner] += 1
        else:
            eval_wins_greedy['draw'] += 1

    field.temperature = original_temp

    wr_at_greedy = (eval_wins_greedy['X'] + eval_wins_greedy.get('B', 0)) / total_eval

    # --- Unique tiles discovered ---
    unique_tiles = tiles_after_train

    # --- Score distribution ---
    all_scores = []
    for tile in field.tiles.values():
        for action_data in tile.values():
            all_scores.append(action_data["score"])
    all_scores.sort()

    # Score stats
    mean_score = sum(all_scores) / len(all_scores) if all_scores else 0
    score_variance = sum((s - mean_score)**2 for s in all_scores) / len(all_scores) if all_scores else 0

    result = {
        "temperature": temp,
        "train_games": TRAIN_GAMES,
        "eval_games": EVAL_GAMES,
        "train_win_rate": (train_wins['X'] + train_wins.get('B', 0)) / sum(train_wins.values()),
        "eval_win_rate_at_train_T": wr_at_train_t,
        "eval_win_rate_at_greedy_T": wr_at_greedy,
        "unique_tiles": unique_tiles,
        "score_entropy": round(entropy_after_train, 4),
        "mean_score": round(mean_score, 4),
        "score_std": round(math.sqrt(score_variance), 4),
        "train_wins": train_wins,
        "eval_wins_at_train_T": eval_wins_train_t,
        "eval_wins_at_greedy_T": eval_wins_greedy,
    }

    print(f"  Results: train_wr={result['train_win_rate']:.1%} | "
          f"eval@T={wr_at_train_t:.1%} | eval@greedy={wr_at_greedy:.1%} | "
          f"tiles={unique_tiles} | entropy={entropy_after_train:.3f}")

    return result


def main():
    random.seed(42)  # Reproducibility
    results = []

    print("TEMPERATURE SWEEP EXPERIMENT")
    print(f"Temperatures: {TEMPERATURES}")
    print(f"Train games per T: {TRAIN_GAMES}")
    print(f"Eval games per T: {EVAL_GAMES}")
    print(f"Greedy eval T: {GREEDY_T}")

    for temp in TEMPERATURES:
        result = run_temperature_experiment(temp)
        results.append(result)

    # --- Analysis ---
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'T':>6} | {'Train WR':>9} | {'Eval@T':>9} | {'Eval@Greedy':>12} | {'Tiles':>6} | {'Entropy':>8}")
    print("-"*80)

    best_eval_t = max(results, key=lambda r: r["eval_win_rate_at_train_T"])
    best_greedy = max(results, key=lambda r: r["eval_win_rate_at_greedy_T"])
    best_explore = max(results, key=lambda r: r["unique_tiles"])

    for r in results:
        marker = ""
        if r == best_eval_t:
            marker += " ← best eval@T"
        if r == best_greedy:
            marker += " ← best eval@greedy"
        print(f"{r['temperature']:>6.2f} | {r['train_win_rate']:>8.1%} | "
              f"{r['eval_win_rate_at_train_T']:>8.1%} | {r['eval_win_rate_at_greedy_T']:>11.1%} | "
              f"{r['unique_tiles']:>6} | {r['score_entropy']:>8.3f}{marker}")

    print(f"\nBest eval@training T: T={best_eval_t['temperature']} ({best_eval_t['eval_win_rate_at_train_T']:.1%})")
    print(f"Best eval@greedy T:   T={best_greedy['temperature']} ({best_greedy['eval_win_rate_at_greedy_T']:.1%})")
    print(f"Most exploration:     T={best_explore['temperature']} ({best_explore['unique_tiles']} tiles)")

    # Check if optimal train T differs from optimal eval T
    annealing_justified = best_eval_t["temperature"] != best_greedy["temperature"]
    print(f"\nAnnealing justified: {annealing_justified}")
    if annealing_justified:
        print(f"  → Train at T={best_eval_t['temperature']} but eval at T={best_greedy['temperature']} differs from greedy best T={best_greedy['temperature']}")
        print(f"  → Consider annealing from {best_eval_t['temperature']} → {best_greedy['temperature']}")
    else:
        print(f"  → Same optimal T for training and evaluation; annealing may not help much")

    output = {
        "experiment": "temperature_sweep",
        "description": "Sweep softmax temperature to find optimal exploration vs exploitation balance",
        "temperatures": TEMPERATURES,
        "train_games": TRAIN_GAMES,
        "eval_games": EVAL_GAMES,
        "greedy_eval_T": GREEDY_T,
        "results": results,
        "analysis": {
            "best_eval_at_train_T": {"temperature": best_eval_t["temperature"], "win_rate": best_eval_t["eval_win_rate_at_train_T"]},
            "best_eval_at_greedy_T": {"temperature": best_greedy["temperature"], "win_rate": best_greedy["eval_win_rate_at_greedy_T"]},
            "most_exploration": {"temperature": best_explore["temperature"], "tiles": best_explore["unique_tiles"]},
            "annealing_justified": annealing_justified,
        },
    }

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "temperature-sweep-results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
