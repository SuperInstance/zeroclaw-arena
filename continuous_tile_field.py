"""
Continuous Tile Field Experiment
================================
Test the tile field on CONTINUOUS action spaces by discretizing them into bins.

Game: 2D navigation. Agent starts at (0,0), must reach target at (1,1).
Actions are continuous (dx, dy) where dx,dy ∈ [-0.3, 0.3].
We discretize the action space into bins and test with different bin counts.

Hypothesis: as bins increase, conservation WEAKENS because the negative space
becomes more fragmented (more near-identical actions that are equally bad/good).
There should be a critical bin count where conservation breaks down.

Test bin counts: [4, 8, 16, 32, 64]
For each: train tile field for 500 episodes, 5 runs each.
Measure: convergence speed, score distribution CV, conservation law strength.
"""

import random
import numpy as np
import json
import os
import sys
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

# ─── 2D Navigation Game ───────────────────────────────────

class NavigationGame:
    """Simple 2D navigation: start at (0,0), reach (1,1)."""

    def __init__(self, n_bins: int = 4, max_step: float = 0.3, max_steps: int = 50):
        self.n_bins = n_bins
        self.max_step = max_step
        self.max_steps = max_steps
        self.target = np.array([1.0, 1.0])
        self.reset()

    def reset(self):
        self.pos = np.array([0.0, 0.0])
        self.done = False
        self.winner = None
        self.turn = 0
        self.done = False

    def _discrete_actions(self) -> list[str]:
        """Generate discretized action labels."""
        # Create bins in [-max_step, max_step] for each axis
        vals = np.linspace(-self.max_step, self.max_step, self.n_bins)
        actions = []
        for dx in vals:
            for dy in vals:
                actions.append(f"{dx:.4f},{dy:.4f}")
        return actions

    def legal_actions(self) -> list[str]:
        return self._discrete_actions() if not self.done else []

    def state(self):
        from zeroclaw import GameState
        state_str = f"({self.pos[0]:.3f},{self.pos[1]:.3f})"
        return GameState(state_str, self.turn, "player")

    def step(self, action: str) -> tuple:
        parts = action.split(",")
        dx, dy = float(parts[0]), float(parts[1])
        self.pos = self.pos + np.array([dx, dy])
        # Clamp to [0, 2] box
        self.pos = np.clip(self.pos, 0.0, 2.0)
        self.turn += 1

        dist = np.linalg.norm(self.pos - self.target)
        reward = -dist  # negative distance as reward

        if dist < 0.1:
            self.done = True
            self.winner = "player"
            reward = 1.0
        elif self.turn >= self.max_steps:
            self.done = True
            self.winner = None
            reward = -dist

        return reward, self.done


# ─── Tile Field for Continuous Actions ────────────────────

class ContinuousStateTile:
    """StateTile adapted for continuous action spaces."""

    def __init__(self, state_hash: str, state_str: str, actions: list[str]):
        self.state_hash = state_hash
        self.state_str = state_str
        self.reflexes: dict[str, dict] = {
            a: {"score": 0.5, "chosen": 0, "total_reward": 0.0} for a in actions
        }
        self.entropy = 1.0

    def best_action(self, legal_actions: list[str], scores: dict = None) -> str:
        """Epsilon-greedy with learned scores. No MC simulation for continuous."""
        if not legal_actions:
            return ""
        if len(legal_actions) == 1:
            return legal_actions[0]

        # Ensure all actions have reflex entries
        for a in legal_actions:
            if a not in self.reflexes:
                self.reflexes[a] = {"score": 0.5, "chosen": 0, "total_reward": 0.0}

        # Epsilon-greedy with decay
        total_chosen = sum(d["chosen"] for d in self.reflexes.values())
        epsilon = max(0.1, 0.5 - total_chosen * 0.005)
        if random.random() < epsilon:
            return random.choice(legal_actions)

        # Pick best scored action
        scores_list = [(a, self.reflexes[a]["score"]) for a in legal_actions]
        scores_list.sort(key=lambda x: -x[1])
        return scores_list[0][0]

    def record(self, action: str, reward: float):
        if action in self.reflexes:
            self.reflexes[action]["chosen"] += 1
            self.reflexes[action]["total_reward"] += reward

    def evolve(self):
        """Update scores based on average reward."""
        for action, data in self.reflexes.items():
            if data["chosen"] > 0:
                avg_reward = data["total_reward"] / data["chosen"]
                # Map reward to [0,1] score: reward ∈ [-1.5, 1.0]
                normalized = (avg_reward + 1.5) / 2.5
                normalized = max(0.05, min(0.95, normalized))
                data["score"] += 0.1 * (normalized - data["score"])
                data["score"] = max(0.05, min(0.95, data["score"]))


def run_episode(game: NavigationGame, tile_field: dict, n_bins: int,
                train: bool = True) -> dict:
    """Run one episode of navigation with tile field."""
    game.reset()
    total_reward = 0.0
    steps = 0
    history = []

    while not game.done:
        state = game.state()
        actions = game.legal_actions()
        if not actions:
            break

        state_hash = state.hash()

        # Quantize position to create state tiles (grid-based)
        # Round to 0.1 resolution for tile grouping
        qx = round(game.pos[0], 1)
        qy = round(game.pos[1], 1)
        tile_key = f"({qx:.1f},{qy:.1f})"

        if tile_key not in tile_field:
            tile_field[tile_key] = ContinuousStateTile(
                tile_key, str(state), actions
            )
        tile = tile_field[tile_key]

        action = tile.best_action(actions)
        reward, done = game.step(action)
        total_reward += reward
        steps += 1

        if train:
            # Propagate reward to all tiles in history
            for h_key, h_action in history:
                if h_key in tile_field:
                    tile_field[h_key].record(h_action, reward * 0.9)
            tile.record(action, reward)

        history.append((tile_key, action))

    dist = np.linalg.norm(game.pos - game.target)
    reached = dist < 0.1

    return {
        "total_reward": total_reward,
        "steps": steps,
        "final_distance": dist,
        "reached_target": reached,
        "final_pos": game.pos.tolist(),
    }


def run_experiment(n_bins: int, num_episodes: int = 500, seed: int = 42) -> dict:
    """Run full experiment for one bin count."""
    random.seed(seed)
    np.random.seed(seed)

    game = NavigationGame(n_bins=n_bins)
    tile_field = {}

    episode_results = []
    convergence_episode = None  # episode where agent first reaches target

    for ep in range(num_episodes):
        result = run_episode(game, tile_field, n_bins, train=True)
        result["episode"] = ep
        episode_results.append(result)

        if result["reached_target"] and convergence_episode is None:
            convergence_episode = ep

        # Evolve tiles every 10 episodes
        if (ep + 1) % 10 == 0:
            for tile in tile_field.values():
                tile.evolve()

    # Analyze tile field
    all_scores = []
    tile_score_distributions = []
    action_score_variance = []

    for tile_key, tile in tile_field.items():
        scores = [d["score"] for d in tile.reflexes.values()]
        all_scores.extend(scores)
        tile_score_distributions.append({
            "tile": tile_key,
            "mean": np.mean(scores),
            "std": np.std(scores),
            "min": min(scores),
            "max": max(scores),
            "n_actions": len(scores),
        })
        # Variance within each tile's action scores
        if len(scores) > 1:
            action_score_variance.append(np.var(scores))

    all_scores = np.array(all_scores)

    # Conservation metrics
    # 1. Score distribution stats
    score_stats = {
        "mean": float(np.mean(all_scores)) if len(all_scores) > 0 else 0,
        "std": float(np.std(all_scores)) if len(all_scores) > 0 else 0,
        "min": float(np.min(all_scores)) if len(all_scores) > 0 else 0,
        "max": float(np.max(all_scores)) if len(all_scores) > 0 else 0,
    }

    # 2. Negative space fraction: actions with score < 0.3
    negative_fraction = float(np.mean(all_scores < 0.3)) if len(all_scores) > 0 else 0

    # 3. Positive space fraction: actions with score > 0.7
    positive_fraction = float(np.mean(all_scores > 0.7)) if len(all_scores) > 0 else 0

    # 4. Within-tile variance (how much actions differ within a state)
    mean_within_tile_var = float(np.mean(action_score_variance)) if action_score_variance else 0

    # 5. Convergence: rolling average of final distances
    window = 50
    final_distances = [r["final_distance"] for r in episode_results]
    rolling_avg = [
        np.mean(final_distances[max(0, i-window):i+1])
        for i in range(len(final_distances))
    ]

    # Performance stats
    reached_count = sum(1 for r in episode_results if r["reached_target"])
    last_100_reached = sum(1 for r in episode_results[-100:] if r["reached_target"])
    last_100_avg_dist = float(np.mean(final_distances[-100:]))

    return {
        "n_bins": n_bins,
        "n_actions": n_bins * n_bins,
        "seed": seed,
        "convergence_episode": convergence_episode,
        "episodes_run": num_episodes,
        "tiles_created": len(tile_field),
        "performance": {
            "total_reached": reached_count,
            "last_100_reached": last_100_reached,
            "reached_rate": reached_count / num_episodes,
            "last_100_reached_rate": last_100_reached / 100,
            "avg_final_distance_last_100": last_100_avg_dist,
        },
        "conservation": {
            "score_distribution": score_stats,
            "negative_fraction": negative_fraction,
            "positive_fraction": positive_fraction,
            "mean_within_tile_variance": mean_within_tile_var,
        },
        "score_cv": float(np.std(all_scores) / np.mean(all_scores)) if np.mean(all_scores) > 0 else float('inf'),
        "total_score_count": len(all_scores),
    }


def main():
    print("=" * 70)
    print("CONTINUOUS ACTION SPACE — TILE FIELD EXPERIMENT")
    print("=" * 70)

    bin_counts = [4, 8, 16, 32, 64]
    n_runs = 5
    base_episodes = 500

    all_results = {}

    for n_bins in bin_counts:
        # Scale episodes with action space — more bins need more exploration
        num_episodes = base_episodes + (n_bins * n_bins * 10)
        print(f"\n{'─' * 60}")
        print(f"BIN COUNT: {n_bins} ({n_bins*n_bins} discrete actions, {num_episodes} episodes)")
        print(f"{'─' * 60}")

        run_results = []
        for run_idx in range(n_runs):
            seed = 42 + run_idx * 100
            print(f"  Run {run_idx+1}/{n_runs} (seed={seed})...", end=" ", flush=True)
            result = run_experiment(n_bins, num_episodes, seed)
            run_results.append(result)
            print(f"reached={result['performance']['last_100_reached_rate']:.1%} "
                  f"convergence={result['convergence_episode']} "
                  f"neg_frac={result['conservation']['negative_fraction']:.3f}")

        # Aggregate across runs
        convergence_episodes = [r["convergence_episode"] for r in run_results if r["convergence_episode"] is not None]
        reached_rates = [r["performance"]["last_100_reached_rate"] for r in run_results]
        neg_fracs = [r["conservation"]["negative_fraction"] for r in run_results]
        pos_fracs = [r["conservation"]["positive_fraction"] for r in run_results]
        within_vars = [r["conservation"]["mean_within_tile_variance"] for r in run_results]
        score_cvs = [r["score_cv"] for r in run_results]

        # CV of score CVs (conservation of conservation)
        cv_of_scores = np.std(score_cvs) / np.mean(score_cvs) if np.mean(score_cvs) > 0 else float('inf')

        summary = {
            "n_bins": n_bins,
            "n_actions": n_bins * n_bins,
            "convergence": {
                "mean_episode": float(np.mean(convergence_episodes)) if convergence_episodes else None,
                "std_episode": float(np.std(convergence_episodes)) if convergence_episodes else None,
                "converged_runs": len(convergence_episodes),
                "total_runs": n_runs,
            },
            "performance": {
                "mean_reached_rate": float(np.mean(reached_rates)),
                "std_reached_rate": float(np.std(reached_rates)),
                "cv_reached_rate": float(np.std(reached_rates) / np.mean(reached_rates)) if np.mean(reached_rates) > 0 else float('inf'),
            },
            "conservation": {
                "negative_fraction": {
                    "mean": float(np.mean(neg_fracs)),
                    "std": float(np.std(neg_fracs)),
                    "cv": float(np.std(neg_fracs) / np.mean(neg_fracs)) if np.mean(neg_fracs) > 0 else float('inf'),
                },
                "positive_fraction": {
                    "mean": float(np.mean(pos_fracs)),
                    "std": float(np.std(pos_fracs)),
                },
                "mean_within_tile_variance": {
                    "mean": float(np.mean(within_vars)),
                    "std": float(np.std(within_vars)),
                },
                "score_cv_across_runs": {
                    "mean": float(np.mean(score_cvs)),
                    "cv_of_cv": float(cv_of_scores),
                },
            },
            "runs": run_results,
        }

        all_results[str(n_bins)] = summary

        print(f"\n  SUMMARY ({n_bins} bins = {n_bins*n_bins} actions):")
        conv = summary['convergence']
        if conv['mean_episode'] is not None:
            print(f"    Convergence: {conv['mean_episode']:.0f} ± {conv['std_episode']:.0f} episodes "
                  f"({conv['converged_runs']}/{n_runs} converged)")
        else:
            print(f"    Convergence: NONE (0/{n_runs} converged)")
        print(f"    Reached rate (last 100): {summary['performance']['mean_reached_rate']:.1%} ± {summary['performance']['std_reached_rate']:.1%}")
        print(f"    Negative fraction: {summary['conservation']['negative_fraction']['mean']:.3f} ± {summary['conservation']['negative_fraction']['std']:.3f}")
        print(f"    Score CV: {summary['conservation']['score_cv_across_runs']['mean']:.4f}")
        print(f"    CV-of-CV (conservation of conservation): {cv_of_scores:.4f}")

    # ─── Analysis ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("CROSS-BIN ANALYSIS")
    print("=" * 70)

    bins = bin_counts
    neg_means = [all_results[str(b)]["conservation"]["negative_fraction"]["mean"] for b in bins]
    neg_cvs = [all_results[str(b)]["conservation"]["negative_fraction"]["cv"] for b in bins]
    score_cvs = [all_results[str(b)]["conservation"]["score_cv_across_runs"]["mean"] for b in bins]
    cv_of_cvs = [all_results[str(b)]["conservation"]["score_cv_across_runs"]["cv_of_cv"] for b in bins]
    reached = [all_results[str(b)]["performance"]["mean_reached_rate"] for b in bins]

    print(f"\n{'Bins':>5} {'Actions':>8} {'Neg%':>8} {'NegCV':>8} {'ScoreCV':>8} {'CVofCV':>8} {'Reached':>8}")
    print("─" * 60)
    for i, b in enumerate(bins):
        print(f"{b:>5} {b*b:>8} {neg_means[i]:>8.3f} {neg_cvs[i]:>8.4f} "
              f"{score_cvs[i]:>8.4f} {cv_of_cvs[i]:>8.4f} {reached[i]:>8.1%}")

    # Find critical bin count where conservation breaks down
    print("\nCONSERVATION BREAKDOWN ANALYSIS:")
    # Conservation holds if CV of negative fraction < 0.05
    for b in bins:
        neg_cv = all_results[str(b)]["conservation"]["negative_fraction"]["cv"]
        status = "HOLDS" if neg_cv < 0.05 else "WEAKENS" if neg_cv < 0.15 else "BROKEN"
        print(f"  {b:>3} bins ({b*b:>4} actions): negative CV = {neg_cv:.4f} → {status}")

    # Identify critical point
    critical_bin = None
    for i, b in enumerate(bins):
        neg_cv = all_results[str(b)]["conservation"]["negative_fraction"]["cv"]
        if neg_cv > 0.05 and critical_bin is None:
            critical_bin = b
    if critical_bin:
        print(f"\n  → CRITICAL BIN COUNT: {critical_bin} ({critical_bin*critical_bin} actions)")
    else:
        print(f"\n  → Conservation HOLDS across all tested bin counts")

    # Final results structure
    final_results = {
        "experiment": "continuous_tile_field",
        "description": "Tile field on continuous action space (2D navigation), discretized into bins",
        "hypothesis": "Conservation weakens as bins increase, breaking down at a critical count",
        "parameters": {
            "game": "2D navigation: (0,0) → (1,1), step ∈ [-0.3, 0.3]",
            "bin_counts": bin_counts,
            "n_runs": n_runs,
            "num_episodes": num_episodes,
        },
        "results": all_results,
        "analysis": {
            "negative_fraction_trend": neg_means,
            "negative_cv_trend": neg_cvs,
            "score_cv_trend": score_cvs,
            "cv_of_cv_trend": cv_of_cvs,
            "reached_rate_trend": reached,
            "critical_bin_count": critical_bin,
            "conclusion": "",
        },
    }

    # Write conclusion
    if critical_bin:
        final_results["analysis"]["conclusion"] = (
            f"Conservation BREAKS at {critical_bin} bins ({critical_bin*critical_bin} actions). "
            f"Negative space CV exceeds 0.05 threshold. Hypothesis CONFIRMED: continuous spaces "
            f"fragment the negative space, weakening conservation."
        )
    else:
        final_results["analysis"]["conclusion"] = (
            f"Conservation HOLDS across all tested bin counts (up to {bins[-1]} bins = {bins[-1]**2} actions). "
            f"Hypothesis NOT confirmed at this scale. May need higher bin counts."
        )

    # Save results
    results_path = os.path.join(os.path.dirname(__file__), "continuous-tile-results.json")
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    return final_results


if __name__ == "__main__":
    main()
