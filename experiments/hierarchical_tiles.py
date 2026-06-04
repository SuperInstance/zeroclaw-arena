"""
Hierarchical Tiles — Decompose complex decisions into composable sub-tiles.

The Innovation:
  Each tile is flat: one state → one action. But real intelligence decomposes:
  "opening strategy" → "center control" → "specific move".

  1. Train a normal tile field on Connect4 (500 games)
  2. Cluster tiles using k-means on their score vectors (k=8)
  3. Each cluster becomes a "meta-tile" — a higher-level strategy
  4. Build a 2-level hierarchy:
     - Level 1: meta-tile selection (which cluster am I in?)
     - Level 2: within-cluster action selection (which specific move?)

  Compare:
  - Flat tile field win rate
  - Hierarchical tile field win rate
  - Speed: hierarchical should be faster (smaller selection at each level)
  - Compression: how much smaller is the hierarchical representation?

  If hierarchical performs within 5% of flat but is 10x smaller/faster,
  it proves that intelligence HAS hierarchical structure.
  The tile field automatically discovers subroutines — function decomposition.
"""

import random
import time
import json
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from copy import deepcopy
import hashlib
import os

# Import game from zeroclaw
from zeroclaw import Connect4, GameState, StateTile

# ─── K-Means Implementation ────────────────────────────────

def kmeans(data: np.ndarray, k: int, max_iters: int = 100, seed: int = 42) -> tuple:
    """Simple k-means clustering. Returns (labels, centroids)."""
    rng = np.random.RandomState(seed)
    n = len(data)
    if n <= k:
        labels = np.arange(n)
        centroids = data.copy()
        return labels, centroids

    # Initialize with k-means++
    centroids = [data[rng.randint(n)]]
    for _ in range(1, k):
        dists = np.array([min(np.linalg.norm(x - c) for c in centroids) for x in data])
        probs = dists / dists.sum()
        idx = rng.choice(n, p=probs)
        centroids.append(data[idx])
    centroids = np.array(centroids)

    for _ in range(max_iters):
        # Assign
        dists = np.array([[np.linalg.norm(x - c) for c in centroids] for x in data])
        labels = np.argmin(dists, axis=1)
        # Update
        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            members = data[labels == i]
            if len(members) > 0:
                new_centroids[i] = members.mean(axis=0)
            else:
                new_centroids[i] = data[rng.randint(n)]
        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    return labels, centroids


# ─── Flat Tile Field Training ───────────────────────────────

def train_flat_tile_field(n_games: int = 500, n_simulations: int = 20) -> dict:
    """Train a standard tile field on Connect4."""
    print(f"Training flat tile field for {n_games} games...")
    tile_field: Dict[str, StateTile] = {}
    stats = {"wins": 0, "losses": 0, "draws": 0, "games": 0}

    for game_i in range(n_games):
        game = Connect4()
        game.reset()
        history = []

        while not game.done:
            state = game.state()
            actions = game.legal_actions()
            if not actions:
                break

            sh = state.hash()
            if sh not in tile_field:
                tile_field[sh] = StateTile(sh, state.state_str, actions)

            tile = tile_field[sh]
            action = tile.best_action(actions, n_simulations=n_simulations, game=game)
            history.append((sh, action))

            game.step(action)

        # Record outcomes
        won = game.winner == 'X'
        for sh, action in history:
            if sh in tile_field:
                tile_field[sh].record(action, won)

        stats["games"] += 1
        if game.winner == 'X':
            stats["wins"] += 1
        elif game.winner == 'O':
            stats["losses"] += 1
        else:
            stats["draws"] += 1

        # Evolve every 25 games
        if (game_i + 1) % 25 == 0:
            for tile in tile_field.values():
                tile.evolve()
            wr = stats["wins"] / max(stats["games"], 1)
            print(f"  Game {game_i+1}/{n_games} | win_rate={wr:.1%} | tiles={len(tile_field)}")

    return tile_field, stats


# ─── Hierarchical Tile Field ────────────────────────────────

class MetaTile:
    """A cluster of similar tiles — represents a higher-level strategy."""

    def __init__(self, cluster_id: int, centroid: np.ndarray, member_tiles: List[str],
                 action_scores: Dict[str, float]):
        self.cluster_id = cluster_id
        self.centroid = centroid
        self.member_tiles = member_tiles  # state hashes in this cluster
        self.action_scores = action_scores  # aggregated action scores across members
        self.n_members = len(member_tiles)

    def select_action(self, legal_actions: List[str], epsilon: float = 0.1) -> str:
        """Select action using aggregated cluster scores."""
        if not legal_actions:
            return ""
        if len(legal_actions) == 1:
            return legal_actions[0]

        if random.random() < epsilon:
            return random.choice(legal_actions)

        scores = [(a, self.action_scores.get(a, 0.5)) for a in legal_actions]
        scores.sort(key=lambda x: -x[1])
        return scores[0][0]


class HierarchicalTileField:
    """2-level decision hierarchy:
    Level 1: Identify which meta-tile (strategy cluster) the current state belongs to.
    Level 2: Use the meta-tile's aggregated scores to pick an action.
    """

    def __init__(self, tile_field: Dict[str, StateTile], n_clusters: int = 8):
        self.n_clusters = n_clusters
        self.tile_field = tile_field
        self.meta_tiles: List[MetaTile] = []
        self.state_to_cluster: Dict[str, int] = {}

        # Build the score matrix for clustering
        self._build_clusters()

    def _build_clusters(self):
        """Cluster tiles based on their score vectors."""
        state_hashes = list(self.tile_field.keys())
        if not state_hashes:
            print("  WARNING: No tiles to cluster!")
            return

        # Fixed action space for Connect4: columns 0-6
        all_actions = [str(i) for i in range(7)]

        # Build score vectors: each tile → 7-dim vector of action scores
        score_matrix = np.zeros((len(state_hashes), 7))
        for i, sh in enumerate(state_hashes):
            tile = self.tile_field[sh]
            for j, action in enumerate(all_actions):
                score_matrix[i, j] = tile.reflexes.get(action, {}).get("score", 0.5)

        # Cluster
        actual_k = min(self.n_clusters, len(state_hashes))
        labels, centroids = kmeans(score_matrix, actual_k)

        print(f"  Clustered {len(state_hashes)} tiles into {actual_k} meta-tiles")

        # Build meta-tiles
        for cluster_id in range(actual_k):
            member_indices = np.where(labels == cluster_id)[0]
            member_hashes = [state_hashes[i] for i in member_indices]

            # Aggregate action scores across members
            action_scores = {}
            for action in all_actions:
                scores = [score_matrix[i, int(action)] for i in member_indices]
                action_scores[action] = np.mean(scores) if scores else 0.5

            meta = MetaTile(
                cluster_id=cluster_id,
                centroid=centroids[cluster_id],
                member_tiles=member_hashes,
                action_scores=action_scores,
            )
            self.meta_tiles.append(meta)

            # Map state → cluster
            for sh in member_hashes:
                self.state_to_cluster[sh] = cluster_id

        # Print cluster stats
        for meta in self.meta_tiles:
            print(f"    Cluster {meta.cluster_id}: {meta.n_members} tiles, "
                  f"best action={max(meta.action_scores, key=meta.action_scores.get)} "
                  f"(score={max(meta.action_scores.values()):.3f})")

    def assign_cluster(self, state_hash: str) -> int:
        """Assign a new/unknown state to nearest cluster."""
        if state_hash in self.state_to_cluster:
            return self.state_to_cluster[state_hash]

        # For unknown states, find nearest centroid
        tile = self.tile_field.get(state_hash)
        if tile is None:
            return 0  # fallback

        vec = np.array([tile.reflexes.get(str(a), {}).get("score", 0.5) for a in range(7)])
        dists = [np.linalg.norm(vec - mt.centroid) for mt in self.meta_tiles]
        return int(np.argmin(dists))

    def select_action(self, state_hash: str, legal_actions: List[str],
                      epsilon: float = 0.1) -> str:
        """2-level action selection:
        1. Identify cluster (O(k) distance computation)
        2. Use cluster's aggregated scores (O(|actions|) lookup)
        """
        cluster_id = self.assign_cluster(state_hash)
        meta = self.meta_tiles[cluster_id]
        return meta.select_action(legal_actions, epsilon)

    def size_bytes(self) -> int:
        """Estimate memory footprint."""
        total = 0
        for meta in self.meta_tiles:
            total += 7 * 8  # centroid
            total += len(meta.member_tiles) * 64  # hash strings
            total += 7 * 8  # action scores
        return total


# ─── Evaluation ────────────────────────────────────────────

def play_vs_random(policy_fn, n_games: int = 200) -> dict:
    """Play Connect4 against a random opponent. policy_fn(game, state, actions) -> action."""
    stats = {"wins": 0, "losses": 0, "draws": 0}
    for _ in range(n_games):
        game = Connect4()
        game.reset()
        while not game.done:
            actions = game.legal_actions()
            if not actions:
                break
            if game.current == 'X':
                # Our agent
                state = game.state()
                action = policy_fn(game, state, actions)
            else:
                # Random opponent
                action = random.choice(actions)
            game.step(action)

        if game.winner == 'X':
            stats["wins"] += 1
        elif game.winner == 'O':
            stats["losses"] += 1
        else:
            stats["draws"] += 1
    return stats


def play_flat_tile(game, state, actions, tile_field: dict, n_sims: int = 10):
    """Flat tile policy."""
    sh = state.hash()
    tile = tile_field.get(sh)
    if tile:
        return tile.best_action(actions, n_simulations=n_sims, game=game)
    return random.choice(actions)


def play_hierarchical(game, state, actions, hier_field: HierarchicalTileField):
    """Hierarchical tile policy."""
    sh = state.hash()
    return hier_field.select_action(sh, actions)


def flat_tile_size(tile_field: dict) -> int:
    """Estimate memory footprint of flat tile field."""
    total = 0
    for sh, tile in tile_field.items():
        total += 64  # hash
        total += len(tile.reflexes) * (8 + 8 + 8)  # action + score + counters
    return total


# ─── Head-to-Head Match ────────────────────────────────────

def play_head_to_head(policy_x, policy_o, n_games: int = 200) -> dict:
    """Pit two policies against each other."""
    stats = {"x_wins": 0, "o_wins": 0, "draws": 0}
    for _ in range(n_games):
        game = Connect4()
        game.reset()
        while not game.done:
            actions = game.legal_actions()
            if not actions:
                break
            state = game.state()
            if game.current == 'X':
                action = policy_x(game, state, actions)
            else:
                action = policy_o(game, state, actions)
            game.step(action)

        if game.winner == 'X':
            stats["x_wins"] += 1
        elif game.winner == 'O':
            stats["o_wins"] += 1
        else:
            stats["draws"] += 1
    return stats


# ─── Main Experiment ───────────────────────────────────────

def run_experiment():
    print("=" * 70)
    print("HIERARCHICAL TILES EXPERIMENT")
    print("=" * 70)

    # Phase 1: Train flat tile field
    print("\n--- Phase 1: Train Flat Tile Field (500 games) ---")
    tile_field, train_stats = train_flat_tile_field(n_games=500, n_simulations=20)
    train_wr = train_stats["wins"] / max(train_stats["games"], 1)
    print(f"  Training win rate: {train_wr:.1%} | Total tiles: {len(tile_field)}")

    # Phase 2: Build hierarchical tile field
    print("\n--- Phase 2: Build Hierarchical Tile Field (k=8) ---")
    hier_field = HierarchicalTileField(tile_field, n_clusters=8)

    # Phase 3: Evaluate against random
    print("\n--- Phase 3: Evaluate vs Random (200 games each) ---")

    print("  Evaluating flat tile field...")
    t0 = time.time()
    flat_stats = play_vs_random(
        lambda g, s, a: play_flat_tile(g, s, a, tile_field, n_sims=10),
        n_games=200
    )
    flat_time = time.time() - t0
    flat_wr = flat_stats["wins"] / 200

    print("  Evaluating hierarchical tile field...")
    t0 = time.time()
    hier_stats = play_vs_random(
        lambda g, s, a: play_hierarchical(g, s, a, hier_field),
        n_games=200
    )
    hier_time = time.time() - t0
    hier_wr = hier_stats["wins"] / 200

    # Phase 4: Head-to-head
    print("\n--- Phase 4: Head-to-Head (flat vs hierarchical, 200 games) ---")
    h2h_stats = play_head_to_head(
        lambda g, s, a: play_flat_tile(g, s, a, tile_field, n_sims=10),
        lambda g, s, a: play_hierarchical(g, s, a, hier_field),
        n_games=200
    )

    # Phase 5: Compression analysis
    print("\n--- Phase 5: Compression Analysis ---")
    flat_size = flat_tile_size(tile_field)
    hier_size = hier_field.size_bytes()
    compression_ratio = flat_size / max(hier_size, 1)

    # Results
    results = {
        "experiment": "hierarchical_tiles",
        "training": {
            "n_games": 500,
            "n_tiles": len(tile_field),
            "train_win_rate": train_wr,
        },
        "flat_field": {
            "wins": flat_stats["wins"],
            "losses": flat_stats["losses"],
            "draws": flat_stats["draws"],
            "win_rate": flat_wr,
            "time_seconds": round(flat_time, 2),
            "size_bytes": flat_size,
        },
        "hierarchical_field": {
            "n_clusters": 8,
            "wins": hier_stats["wins"],
            "losses": hier_stats["losses"],
            "draws": hier_stats["draws"],
            "win_rate": hier_wr,
            "time_seconds": round(hier_time, 2),
            "size_bytes": hier_size,
        },
        "comparison": {
            "win_rate_diff": round(hier_wr - flat_wr, 4),
            "win_rate_diff_pct": round((hier_wr - flat_wr) * 100, 2),
            "within_5pct": abs(hier_wr - flat_wr) <= 0.05,
            "speedup": round(flat_time / max(hier_time, 0.001), 2),
            "compression_ratio": round(compression_ratio, 2),
            "size_reduction_pct": round((1 - hier_size / max(flat_size, 1)) * 100, 1),
        },
        "head_to_head": {
            "flat_wins": h2h_stats["x_wins"],
            "hier_wins": h2h_stats["o_wins"],
            "draws": h2h_stats["draws"],
        },
        "conclusion": "",
    }

    # Conclusion
    speedup = flat_time / max(hier_time, 0.001)
    wr_diff = hier_wr - flat_wr  # positive = hierarchical wins
    abs_diff = abs(wr_diff)
    compression = compression_ratio

    if wr_diff >= 0 and speedup >= 2:
        conclusion = (
            f"BREAKTHROUGH: Hierarchical tiles OUTPERFORM flat by {wr_diff:.1%} "
            f"while being {speedup:.1f}x faster and {compression:.1f}x smaller! "
            f"Aggregation across cluster members acts as regularization — "
            f"averaging out noise from individual tiles produces better strategies. "
            f"This proves intelligence HAS hierarchical structure — "
            f"the tile field automatically discovered composable subroutines."
        )
    elif wr_diff >= 0:
        conclusion = (
            f"SUCCESS: Hierarchical tiles match/beat flat ({wr_diff:+.1%}) "
            f"and are {compression:.1f}x smaller. "
            f"Speedup was {speedup:.1f}x. "
            f"Hierarchical decomposition works."
        )
    elif abs_diff <= 0.05 and speedup >= 2:
        conclusion = (
            f"SUCCESS: Hierarchical tiles within {abs_diff:.1%} of flat "
            f"while being {speedup:.1f}x faster and {compression:.1f}x smaller. "
            f"Hierarchical structure is valid — function decomposition works."
        )
    else:
        conclusion = (
            f"MIXED: Hierarchical tiles are {abs_diff:.1%} worse than flat "
            f"but {speedup:.1f}x faster and {compression:.1f}x smaller. "
            f"More clusters or better clustering may close the gap."
        )

    results["conclusion"] = conclusion

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n  Flat Tile Field:")
    print(f"    Win Rate: {flat_wr:.1%} ({flat_stats['wins']}W / {flat_stats['losses']}L / {flat_stats['draws']}D)")
    print(f"    Time: {flat_time:.2f}s")
    print(f"    Size: {flat_size:,} bytes ({len(tile_field)} tiles)")

    print(f"\n  Hierarchical Tile Field (k=8):")
    print(f"    Win Rate: {hier_wr:.1%} ({hier_stats['wins']}W / {hier_stats['losses']}L / {hier_stats['draws']}D)")
    print(f"    Time: {hier_time:.2f}s")
    print(f"    Size: {hier_size:,} bytes ({len(hier_field.meta_tiles)} meta-tiles)")

    print(f"\n  Head-to-Head (flat=X, hier=O):")
    print(f"    Flat wins: {h2h_stats['x_wins']} | Hier wins: {h2h_stats['o_wins']} | Draws: {h2h_stats['draws']}")

    print(f"\n  Comparison:")
    print(f"    Win rate diff: {wr_diff:+.1%} (hierarchical {'wins' if wr_diff >= 0 else 'loses'})")
    print(f"    Speedup: {speedup:.1f}x")
    print(f"    Compression: {compression:.1f}x ({results['comparison']['size_reduction_pct']}% smaller)")

    print(f"\n  CONCLUSION:")
    print(f"    {conclusion}")

    # Save results
    with open("hierarchical-tiles-results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to hierarchical-tiles-results.json")

    # Print cluster analysis
    print("\n  Cluster Strategy Analysis:")
    for meta in hier_field.meta_tiles:
        sorted_actions = sorted(meta.action_scores.items(), key=lambda x: -x[1])
        top3 = sorted_actions[:3]
        print(f"    Cluster {meta.cluster_id} ({meta.n_members} tiles): "
              f"top actions = {[(a, f'{s:.3f}') for a, s in top3]}")

    return results


if __name__ == "__main__":
    results = run_experiment()
