"""
GPU Tile Factory — Mass-produce compiled tile policies using CUDA + Ryzen parallelism.

Architecture:
  - 24 game variants (different boards, rewards, win conditions)
  - 24 CPU cores train in parallel (multiprocessing.Pool)
  - Each process: 200 games → evolve → compile to lookup table
  - GPU (torch.cuda) batch-SVD factorization of all score matrices
  - 5 generations of evolutionary cross-pollination

The Ryzen/GPU synergy: CPU handles parallel game simulation,
GPU handles batch linear algebra on all policies at once.
"""

import torch
import numpy as np
import random
import json
import os
import sys
import time
import hashlib
import multiprocessing as mp
from collections import defaultdict
from copy import deepcopy
from typing import List, Dict, Tuple, Optional

# ─── Game Variants ────────────────────────────────────────────────

class TicTacToeVariant:
    """Configurable tic-tac-toe variant."""
    def __init__(self, board_size=3, win_length=None,
                 win_reward=1.0, loss_penalty=-1.0, draw_reward=0.0,
                 center_bonus=0.0, corner_bonus=0.0, edge_penalty=0.0,
                 early_win_bonus=0.0, blockade_bonus=0.0,
                 name="3x3_standard"):
        self.board_size = board_size
        self.win_length = win_length or board_size
        self.win_reward = win_reward
        self.loss_penalty = loss_penalty
        self.draw_reward = draw_reward
        self.center_bonus = center_bonus
        self.corner_bonus = corner_bonus
        self.edge_penalty = edge_penalty
        self.early_win_bonus = early_win_bonus
        self.blockade_bonus = blockade_bonus
        self.name = name
        self.n_cells = board_size * board_size
        self._build_lines()

    def _build_lines(self):
        """Build all winning lines for this board size."""
        n = self.board_size
        w = self.win_length
        lines = []
        # Rows
        for r in range(n):
            for c in range(n - w + 1):
                lines.append(tuple(r * n + c + i for i in range(w)))
        # Columns
        for c in range(n):
            for r in range(n - w + 1):
                lines.append(tuple((r + i) * n + c for i in range(w)))
        # Diag down-right
        for r in range(n - w + 1):
            for c in range(n - w + 1):
                lines.append(tuple((r + i) * n + (c + i) for i in range(w)))
        # Diag down-left
        for r in range(n - w + 1):
            for c in range(w - 1, n):
                lines.append(tuple((r + i) * n + (c - i) for i in range(w)))
        self.lines = lines
        # Position metadata
        center = n // 2
        self.center_pos = center * n + center
        self.corner_positions = set()
        for r, c in [(0, 0), (0, n-1), (n-1, 0), (n-1, n-1)]:
            self.corner_positions.add(r * n + c)

    def new_game(self):
        g = _VariantGame(self)
        return g


class _VariantGame:
    """A game instance for a specific variant."""
    def __init__(self, variant: TicTacToeVariant):
        self.variant = variant
        self.board = [' '] * variant.n_cells
        self.current = 'X'
        self.turn = 0
        self.done = False
        self.winner = None

    def state(self):
        return ''.join(self.board)

    def legal_actions(self):
        if self.done:
            return []
        return [str(i) for i in range(self.variant.n_cells) if self.board[i] == ' ']

    def step(self, action):
        pos = int(action)
        if self.board[pos] != ' ':
            return -1.0, True  # illegal

        self.board[pos] = self.current
        self.turn += 1

        # Check wins
        for line in self.variant.lines:
            if all(self.board[i] == self.current for i in line):
                self.done = True
                self.winner = self.current
                reward = self.variant.win_reward
                # Early win bonus
                if self.variant.early_win_bonus > 0:
                    max_turns = self.variant.n_cells
                    reward += self.variant.early_win_bonus * (1 - self.turn / max_turns)
                final = reward if self.current == 'X' else self.variant.loss_penalty
                return final, True

        # Draw
        if self.turn >= self.variant.n_cells:
            self.done = True
            return self.variant.draw_reward, True

        # Blockade bonus: did X just block O's win?
        if self.variant.blockade_bonus > 0 and self.current == 'X':
            # Check if O was about to win before this move
            opp = 'O'
            for line in self.variant.lines:
                if pos in line:
                    opp_count = sum(1 for i in line if self.board[i] == opp)
                    if opp_count == self.variant.win_length - 1:
                        # This was a blocking move (but we already placed, so opp_count was win_length-1 before)
                        # Actually check: before our move, O had w-1 in this line and the empty was pos
                        pass

        # Positional reward shaping
        shaping = 0.0
        if self.current == 'X':
            if pos == self.variant.center_pos:
                shaping += self.variant.center_bonus
            if pos in self.variant.corner_positions:
                shaping += self.variant.corner_bonus
            if pos not in self.variant.corner_positions and pos != self.variant.center_pos:
                shaping += self.variant.edge_penalty

        self.current = 'O' if self.current == 'X' else 'X'
        return shaping, False

    def reset(self):
        self.board = [' '] * self.variant.n_cells
        self.current = 'X'
        self.turn = 0
        self.done = False
        self.winner = None

    def copy(self):
        g = _VariantGame(self.variant)
        g.board = self.board[:]
        g.current = self.current
        g.turn = self.turn
        g.done = self.done
        g.winner = self.winner
        return g


# ─── 24 Variants ──────────────────────────────────────────────────

def create_variants():
    """Create 24 game variants covering different board sizes, rewards, win conditions."""
    variants = []

    # === 3x3 variants (12) ===
    variants.append(TicTacToeVariant(3, name="3x3_standard"))
    variants.append(TicTacToeVariant(3, win_reward=2.0, name="3x3_high_stakes"))
    variants.append(TicTacToeVariant(3, center_bonus=0.2, name="3x3_center_lover"))
    variants.append(TicTacToeVariant(3, corner_bonus=0.15, name="3x3_corner_hugger"))
    variants.append(TicTacToeVariant(3, edge_penalty=-0.1, name="3x3_edge_hater"))
    variants.append(TicTacToeVariant(3, early_win_bonus=0.5, name="3x3_speed_demon"))
    variants.append(TicTacToeVariant(3, loss_penalty=-2.0, name="3x3_loss_averse"))
    variants.append(TicTacToeVariant(3, draw_reward=0.3, name="3x3_draw_friendly"))
    variants.append(TicTacToeVariant(3, win_reward=1.5, center_bonus=0.1, corner_bonus=0.05, name="3x3_positional"))
    variants.append(TicTacToeVariant(3, win_reward=3.0, loss_penalty=-3.0, name="3x3_extreme"))
    variants.append(TicTacToeVariant(3, win_length=2, name="3x3_fast_win"))  # Only 2 in a row needed
    variants.append(TicTacToeVariant(3, center_bonus=0.3, corner_bonus=-0.1, name="3x3_center_fetish"))

    # === 4x4 variants (6) ===
    variants.append(TicTacToeVariant(4, win_length=3, name="4x4_win3"))
    variants.append(TicTacToeVariant(4, win_length=3, center_bonus=0.1, name="4x4_win3_center"))
    variants.append(TicTacToeVariant(4, win_length=4, name="4x4_full"))
    variants.append(TicTacToeVariant(4, win_length=3, early_win_bonus=0.3, name="4x4_speed"))
    variants.append(TicTacToeVariant(4, win_length=3, win_reward=2.0, name="4x4_high_stakes"))
    variants.append(TicTacToeVariant(4, win_length=3, corner_bonus=0.1, name="4x4_corners"))

    # === 5x5 variants (6) ===
    variants.append(TicTacToeVariant(5, win_length=4, name="5x5_win4"))
    variants.append(TicTacToeVariant(5, win_length=3, name="5x5_win3"))
    variants.append(TicTacToeVariant(5, win_length=4, center_bonus=0.1, name="5x5_win4_center"))
    variants.append(TicTacToeVariant(5, win_length=4, early_win_bonus=0.4, name="5x5_speed"))
    variants.append(TicTacToeVariant(5, win_length=5, name="5x5_full"))
    variants.append(TicTacToeVariant(5, win_length=4, win_reward=2.0, corner_bonus=0.05, name="5x5_mixed"))

    assert len(variants) == 24, f"Expected 24 variants, got {len(variants)}"
    return variants


# ─── Tile Field (self-contained for multiprocessing) ──────────────

class TileField:
    """Lightweight tile field with Monte Carlo simulation for action selection."""

    def __init__(self, n_cells=9, n_simulations=30, temperature=0.3,
                 priors=None):
        self.n_cells = n_cells
        self.n_simulations = n_simulations
        self.temperature = temperature
        # score_table: state_hash -> {action_index -> score}
        self.score_table = {}
        if priors:
            self.score_table = {k: dict(v) for k, v in priors.items()}

    def _hash(self, board_str, n_cells):
        # Pad/truncate to n_cells for consistency
        padded = board_str.ljust(n_cells)[:n_cells]
        return padded

    def _rollout(self, game: _VariantGame) -> float:
        """Random rollout from current state to terminal."""
        g = game.copy()
        while not g.done:
            actions = g.legal_actions()
            if not actions:
                break
            g.step(random.choice(actions))
        if g.winner == 'X':
            return 1.0
        elif g.winner == 'O':
            return -1.0
        return 0.0

    def choose_action(self, game: _VariantGame) -> str:
        """Select action via Monte Carlo simulation."""
        actions = game.legal_actions()
        if not actions:
            return '0'
        if len(actions) == 1:
            return actions[0]

        state_key = game.state()
        scores = {}

        for action in actions:
            total = 0.0
            for _ in range(self.n_simulations):
                g = game.copy()
                g.step(action)
                total += self._rollout(g)
            scores[action] = total / self.n_simulations

        # Blend with prior knowledge
        if state_key in self.score_table:
            for a in scores:
                if a in self.score_table[state_key]:
                    scores[a] = 0.6 * scores[a] + 0.4 * self.score_table[state_key][a]

        # Softmax selection with temperature
        vals = np.array([scores[a] for a in actions])
        if self.temperature > 0:
            exp_vals = np.exp(vals / self.temperature)
            probs = exp_vals / (exp_vals.sum() + 1e-10)
            idx = np.random.choice(len(actions), p=probs)
        else:
            idx = int(np.argmax(vals))

        # Store
        self.score_table[state_key] = scores
        return actions[idx]

    def compile(self) -> dict:
        """Compile to deterministic lookup table: state -> best_action."""
        lookup = {}
        for state_key, scores in self.score_table.items():
            if scores:
                best_action = max(scores, key=scores.get)
                lookup[state_key] = best_action
        return lookup


# ─── Training Worker (runs in separate process) ──────────────────

def train_variant(args):
    """Train a tile field for one variant. Returns (variant_name, score_table, win_rate)."""
    variant_config, n_games, seed, priors_json = args

    # Reconstruct variant
    v = TicTacToeVariant(**variant_config)

    random.seed(seed)
    np.random.seed(seed)

    priors = None
    if priors_json:
        priors = json.loads(priors_json)

    field = TileField(
        n_cells=v.n_cells,
        n_simulations=20,
        temperature=0.3,
        priors=priors,
    )

    wins = 0
    draws = 0
    losses = 0

    for game_idx in range(n_games):
        game = v.new_game()

        while not game.done:
            actions = game.legal_actions()
            if not actions:
                break

            if game.current == 'X':
                action = field.choose_action(game)
            else:
                action = random.choice(actions)

            reward, done = game.step(action)

        if game.winner == 'X':
            wins += 1
        elif game.winner == 'O':
            losses += 1
        else:
            draws += 1

    win_rate = wins / n_games
    compiled = field.compile()

    # Serialize score table
    score_data = {}
    for state_key, scores in field.score_table.items():
        score_data[state_key] = {str(k): float(v) for k, v in scores.items()}

    return {
        'name': v.name,
        'board_size': v.board_size,
        'n_cells': v.n_cells,
        'win_rate': win_rate,
        'wins': wins,
        'draws': draws,
        'losses': losses,
        'n_states': len(score_data),
        'score_table_json': json.dumps(score_data),
        'compiled_lookup': json.dumps(compiled),
    }


# ─── GPU SVD Factorization ────────────────────────────────────────

def gpu_batch_svd(policies: List[dict], device='cuda') -> dict:
    """
    Given compiled policies, extract score matrices and batch-SVD them on GPU.
    This is the GPU part of the Ryzen/GPU synergy.
    """
    print(f"\n{'='*60}")
    print(f"GPU BATCH SVD — Processing {len(policies)} policies on {device}")
    print(f"{'='*60}")

    # Collect all score vectors into a batch matrix
    all_states = set()
    for p in policies:
        scores = json.loads(p['score_table_json'])
        all_states.update(scores.keys())

    n_states = len(all_states)
    n_policies = len(policies)
    state_list = sorted(all_states)

    print(f"  Total unique states: {n_states}")
    print(f"  Total policies: {n_policies}")
    print(f"  Matrix shape: ({n_policies}, {n_states})")

    if n_states == 0:
        return {'status': 'no_states', 'singular_values': []}

    # Build batch matrix: [n_policies, n_states]
    matrix = np.zeros((n_policies, n_states), dtype=np.float32)
    for i, p in enumerate(policies):
        scores = json.loads(p['score_table_json'])
        for j, state in enumerate(state_list):
            if state in scores:
                # Average score across all actions for this state
                vals = list(scores[state].values())
                matrix[i, j] = np.mean(vals) if vals else 0.0

    tensor = torch.from_numpy(matrix).to(device)
    print(f"  Tensor on {tensor.device}: {tensor.shape}")

    # Batch SVD
    t0 = time.perf_counter()
    U, S, Vt = torch.linalg.svd(tensor, full_matrices=False)
    torch.cuda.synchronize() if device == 'cuda' else None
    elapsed = time.perf_counter() - t0

    print(f"  SVD completed in {elapsed*1000:.1f}ms")
    print(f"  Top 10 singular values: {S[:10].cpu().numpy()}")

    # Energy analysis
    total_energy = (S ** 2).sum().item()
    top3_energy = (S[:3] ** 2).sum().item()
    top5_energy = (S[:5] ** 2).sum().item()
    print(f"  Energy in top-3 components: {top3_energy/total_energy:.1%}")
    print(f"  Energy in top-5 components: {top5_energy/total_energy:.1%}")

    # Reconstruct low-rank approximations (rank-5)
    rank = min(5, len(S))
    approx = U[:, :rank] @ torch.diag(S[:rank]) @ Vt[:rank, :]
    approx_np = approx.cpu().numpy()

    return {
        'status': 'success',
        'singular_values': S.cpu().tolist(),
        'total_energy': total_energy,
        'top3_energy_pct': top3_energy / total_energy,
        'top5_energy_pct': top5_energy / total_energy,
        'svd_time_ms': elapsed * 1000,
        'matrix_shape': [n_policies, n_states],
        'low_rank_approx': approx_np,
        'state_list': state_list,
    }


# ─── Cross-Pollination ────────────────────────────────────────────

def cross_pollinate(results: List[dict], top_k=5) -> dict:
    """
    Take the top-k policies and share their score vectors as priors
    for the next generation. Better policies get more weight.
    """
    # Rank by win rate
    ranked = sorted(results, key=lambda x: -x['win_rate'])
    top = ranked[:top_k]

    print(f"\n  Cross-pollinating top {len(top)} policies:")
    for p in top:
        print(f"    {p['name']}: {p['win_rate']:.1%} ({p['n_states']} states)")

    # Merge score tables with quality-weighted averaging
    merged = {}
    weights = []
    for rank_i, p in enumerate(top):
        w = (top_k - rank_i) / sum(range(1, top_k + 1))  # Higher rank = more weight
        weights.append(w)
        scores = json.loads(p['score_table_json'])
        for state, action_scores in scores.items():
            if state not in merged:
                merged[state] = {}
            for action, score in action_scores.items():
                if action not in merged[state]:
                    merged[state][action] = 0.0
                merged[state][action] += w * score

    # Normalize
    for state in merged:
        for action in merged[state]:
            merged[state][action] /= sum(weights)

    print(f"  Merged prior: {len(merged)} states")
    return merged


# ─── Main Factory ─────────────────────────────────────────────────

def run_factory():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("GPU TILE FACTORY — 24 Ryzen Cores + CUDA SVD")
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"CPU cores: {mp.cpu_count()}")
    print(f"Pool workers: 24")
    print("=" * 60)

    variants = create_variants()
    n_games = 200
    n_generations = 5
    pool_size = 24

    all_results = []

    for gen in range(n_generations):
        print(f"\n{'='*60}")
        print(f"GENERATION {gen + 1}/{n_generations}")
        print(f"{'='*60}")
        gen_start = time.perf_counter()

        # Build work items
        work = []
        shared_prior = None
        if gen > 0 and all_results:
            shared_prior = cross_pollinate(all_results[-1])

        for i, v in enumerate(variants):
            # Serialize variant config
            config = {
                'board_size': v.board_size,
                'win_length': v.win_length,
                'win_reward': v.win_reward,
                'loss_penalty': v.loss_penalty,
                'draw_reward': v.draw_reward,
                'center_bonus': v.center_bonus,
                'corner_bonus': v.corner_bonus,
                'edge_penalty': v.edge_penalty,
                'early_win_bonus': v.early_win_bonus,
                'blockade_bonus': v.blockade_bonus,
                'name': v.name,
            }
            priors_json = json.dumps(shared_prior) if shared_prior else None
            seed = 42 + gen * 1000 + i * 17
            work.append((config, n_games, seed, priors_json))

        # Parallel training on 24 cores
        print(f"\n  Dispatching {len(work)} training jobs to {pool_size} cores...")
        dispatch_t = time.perf_counter()

        with mp.Pool(processes=pool_size) as pool:
            results = pool.map(train_variant, work)

        train_elapsed = time.perf_counter() - dispatch_t

        # Sort and display
        results.sort(key=lambda x: -x['win_rate'])
        print(f"\n  Training completed in {train_elapsed:.1f}s")
        print(f"\n  {'Rank':<5} {'Variant':<25} {'Win%':<8} {'W/D/L':<12} {'States':<8}")
        print(f"  {'-'*60}")
        for rank, r in enumerate(results, 1):
            wdl = f"{r['wins']}/{r['draws']}/{r['losses']}"
            print(f"  {rank:<5} {r['name']:<25} {r['win_rate']:<8.1%} {wdl:<12} {r['n_states']:<8}")

        # GPU batch SVD on all policies
        svd_result = gpu_batch_svd(results, device=device)

        # Store results for next generation's cross-pollination
        all_results.append(results)

        gen_elapsed = time.perf_counter() - gen_start
        print(f"\n  Generation {gen+1} elapsed: {gen_elapsed:.1f}s")

    # ─── Final Summary ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FACTORY COMPLETE — Final Rankings")
    print(f"{'='*60}")

    # Best across all generations
    final_ranked = sorted(all_results[-1], key=lambda x: -x['win_rate'])
    print(f"\n  Top 10 policies (final generation):")
    for rank, r in enumerate(final_ranked[:10], 1):
        print(f"    {rank}. {r['name']}: {r['win_rate']:.1%} ({r['n_states']} states)")

    # Save results
    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_variants': 24,
        'n_games_per_variant': n_games,
        'n_generations': n_generations,
        'device': device,
        'final_rankings': [{
            'rank': i + 1,
            'name': r['name'],
            'board_size': r['board_size'],
            'win_rate': r['win_rate'],
            'wins': r['wins'],
            'draws': r['draws'],
            'losses': r['losses'],
            'n_states': r['n_states'],
        } for i, r in enumerate(final_ranked)],
        'svd_singular_values': svd_result.get('singular_values', []),
        'svd_energy_top3': svd_result.get('top3_energy_pct', 0),
        'svd_energy_top5': svd_result.get('top5_energy_pct', 0),
        'svd_time_ms': svd_result.get('svd_time_ms', 0),
    }

    results_path = os.path.join(os.path.dirname(__file__), 'gpu-tile-factory-results.json')
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    return output


if __name__ == '__main__':
    run_factory()
