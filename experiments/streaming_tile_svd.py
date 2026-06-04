"""
Streaming Tile SVD — Incremental Factorization of a Growing Tile Field
======================================================================

Standard SVD requires the full matrix. But a tile field grows over time
(new states discovered during training). This experiment asks:

Can we update the SVD incrementally as new tiles arrive?
Does the dominant strategy axis (rank-1 component) stabilize EARLY?

If rank-1 stabilizes at game 100, we can stop training early —
the rest is just refinement.

Protocol:
1. Train a tile field on TTT incrementally (50 games per batch)
2. After each batch, compute SVD of the score matrix
3. Track how singular values evolve over time
4. Compare: full SVD vs incremental SVD (warm-start from previous U,S,V)
5. Measure: does the rank-1 component stabilize? When?

Uses torch.linalg.svd for GPU-accelerated decomposition.
"""

import random
import hashlib
import numpy as np
import json
import os
import time
import sys
from collections import defaultdict
from copy import deepcopy

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from zeroclaw import TicTacToe, Connect4


# ─── Minimal Tile Field ──────────────────────────────────

class StreamingTileField:
    """Lightweight tile field for streaming SVD experiments."""

    def __init__(self):
        self.tiles = {}  # state_hash -> {action: {score, chosen, won}}
        self.state_order = []  # ordered list of discovered states
        self.action_set = set()  # all actions ever seen

    def get_or_create(self, state_str, legal_actions):
        h = hashlib.blake2b(state_str.encode(), digest_size=8).hexdigest()
        if h not in self.tiles:
            self.tiles[h] = {
                a: {"score": 0.5, "chosen": 0, "won": 0}
                for a in legal_actions
            }
            self.state_order.append(h)
        else:
            for a in legal_actions:
                if a not in self.tiles[h]:
                    self.tiles[h][a] = {"score": 0.5, "chosen": 0, "won": 0}
        for a in legal_actions:
            self.action_set.add(a)
        return h

    def choose(self, h, actions, temperature=0.3, epsilon=0.05):
        tile = self.tiles[h]
        if random.random() < epsilon:
            # Explore least-visited
            return min(actions, key=lambda a: tile[a]["chosen"])
        scores = np.array([tile[a]["score"] for a in actions])
        if temperature > 0.01:
            exp_s = np.exp(scores / temperature)
            probs = exp_s / exp_s.sum()
            return actions[np.random.choice(len(actions), p=probs)]
        return actions[np.argmax(scores)]

    def record(self, h, action, won):
        if h in self.tiles and action in self.tiles[h]:
            self.tiles[h][action]["chosen"] += 1
            if won:
                self.tiles[h][action]["won"] += 1

    def evolve(self, lr=0.05, cap=0.05):
        for tile in self.tiles.values():
            for d in tile.values():
                if d["chosen"] > 0:
                    wr = d["won"] / d["chosen"]
                    delta = max(-cap, min(cap, lr * (wr - d["score"])))
                    d["score"] = max(0.05, min(0.95, d["score"] + delta))


# ─── Score Matrix Construction ───────────────────────────

def build_score_matrix(field):
    """Build a (states × actions) score matrix from the tile field.
    
    Rows = states (ordered by discovery)
    Columns = actions (global set, sorted)
    Values = learned scores (0.5 for unseen actions = prior)
    """
    if not field.tiles or not field.action_set:
        return None, [], []

    actions = sorted(field.action_set)
    states = field.state_order
    n_states = len(states)
    n_actions = len(actions)

    matrix = np.full((n_states, n_actions), 0.5)  # prior score
    for i, h in enumerate(states):
        tile = field.tiles[h]
        for j, a in enumerate(actions):
            if a in tile:
                matrix[i, j] = tile[a]["score"]

    return matrix, states, actions


# ─── SVD Analysis ────────────────────────────────────────

def compute_svd(matrix, use_torch=True):
    """Compute SVD, optionally on GPU via torch."""
    if use_torch and TORCH_AVAILABLE:
        t = torch.tensor(matrix, dtype=torch.float32)
        if torch.cuda.is_available():
            t = t.cuda()
        U, S, Vh = torch.linalg.svd(t, full_matrices=False)
        return U.cpu().numpy(), S.cpu().numpy(), Vh.cpu().numpy()
    else:
        U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
        return U, S, Vh


def incremental_svd_update(prev_U, prev_S, prev_Vh, new_matrix):
    """Warm-start incremental SVD using subspace tracking.
    
    Strategy: project new rows onto previous singular space,
    then do a cheap rank-k update.
    
    This is an approximation — exact incremental SVD requires
    the brand algorithm, but this captures the key idea.
    """
    if prev_U is None:
        return compute_svd(new_matrix)

    k = len(prev_S)  # rank of previous decomposition
    n_new_states = new_matrix.shape[0]
    n_actions = new_matrix.shape[1]

    # Previous reconstruction (rank-k approximation)
    # prev_U: (old_states × k), prev_S: (k,), prev_Vh: (k × actions)
    
    # Split new matrix into old states (updated) and genuinely new states
    n_old = prev_U.shape[0]
    
    if n_new_states <= n_old:
        # Only updates to existing states — recompute SVD
        # but warm-start by projecting onto previous V
        updated = new_matrix[:n_new_states]
        # Project onto previous right singular vectors
        projected = updated @ prev_Vh.T  # (states × k)
        # Residual
        reconstructed = projected @ np.diag(prev_S) @ prev_Vh
        residual = updated - reconstructed
        # Combine: [projected | residual] and do SVD on the small matrix
        Q = np.hstack([projected, residual])
        Uq, Sq, Vq = np.linalg.svd(Q, full_matrices=False)
        U_new = Uq
        # Pad Vh to include residual subspace
        n_res = residual.shape[1]
        V_new = np.zeros((len(Sq), n_actions))
        V_new[:, :k] = Vq[:, :k] @ prev_Vh
        if n_res > 0 and Vq.shape[1] > k:
            V_new[:, k:] = Vq[:, k:]  # residual directions
        return U_new, Sq, V_new
    
    # New states added — extend the matrix
    old_part = new_matrix[:n_old]
    new_part = new_matrix[n_old:]
    
    # Project old part onto previous basis
    old_projected = old_part @ prev_Vh.T @ np.diag(1.0 / (prev_S + 1e-10))
    
    # For new part, compute residuals against previous basis
    new_approx = new_part @ prev_Vh.T  # projection
    new_residual = new_part - new_approx @ np.diag(prev_S) @ prev_Vh
    
    # QR of new residual to get orthogonal basis
    if new_residual.shape[0] > 0 and np.linalg.norm(new_residual) > 1e-10:
        Q_res, R_res = np.linalg.qr(new_residual)
        # Trim to actual rank
        r = min(Q_res.shape[1], R_res.shape[0])
        Q_res = Q_res[:, :r]
        R_res = R_res[:r, :]
    else:
        Q_res = np.zeros((new_part.shape[0], 0))
        R_res = np.zeros((0, n_actions))
    
    # Build combined small matrix for SVD
    # [prev_S     prev_Vh @ Q_res.T]  (this is the key insight)
    # Actually, let's just do a proper block update
    # Combined coefficient matrix in the [prev_basis | residual] space
    k_new = Q_res.shape[1]
    combined_rows = n_old + new_part.shape[0]
    
    # Build the block matrix
    # Top: old states in prev basis
    # Bottom: new states in prev basis + residual
    block = np.zeros((combined_rows, k + k_new))
    block[:n_old, :k] = old_projected @ np.diag(prev_S)
    block[n_old:, :k] = new_approx
    if k_new > 0:
        block[n_old:, k:k+k_new] = Q_res * np.linalg.norm(R_res, axis=1, keepdims=True).T if R_res.shape[0] > 0 else Q_res
    
    # SVD of the small matrix
    Ub, Sb, Vhb = np.linalg.svd(block, full_matrices=False)
    
    # Recover full U, Vh
    U_full = np.zeros((combined_rows, len(Sb)))
    U_full[:n_old, :] = Ub[:n_old, :] 
    U_full[n_old:, :] = Ub[n_old:, :]
    
    # Vh: map back to original action space
    # basis = [prev_Vh.T | Q_res_cols] → shape (n_actions, k + k_new)
    basis = np.zeros((n_actions, k + k_new))
    basis[:, :k] = prev_Vh.T
    if k_new > 0 and R_res.shape[0] > 0:
        basis[:, k:] = R_res.T
    
    Vh_full = Vhb @ basis.T  # (rank × actions)
    
    return U_full, Sb, Vh_full


def rank1_stability(history):
    """Measure how stable the rank-1 component is across batches.
    
    Returns stability metric: cosine similarity of V[0] between
    consecutive batches. 1.0 = perfectly stable.
    """
    if len(history) < 2:
        return []
    
    stabilities = []
    for i in range(1, len(history)):
        v_prev = history[i-1]["Vh"][0]  # first right singular vector
        v_curr = history[i]["Vh"][0]
        
        # Pad to same length
        max_len = max(len(v_prev), len(v_curr))
        vp = np.zeros(max_len)
        vc = np.zeros(max_len)
        vp[:len(v_prev)] = v_prev
        vc[:len(v_curr)] = v_curr
        
        cos_sim = abs(np.dot(vp, vc)) / (np.linalg.norm(vp) * np.linalg.norm(vc) + 1e-10)
        stabilities.append(cos_sim)
    
    return stabilities


def singular_value_concentration(S):
    """How much of the energy is in the top k components?
    
    Returns the fraction of total singular value energy in rank-1, rank-2, etc.
    """
    total = np.sum(S**2)
    if total < 1e-10:
        return [0.0] * len(S)
    return [(S[i]**2 / total) for i in range(len(S))]


# ─── Main Experiment ─────────────────────────────────────

def run_streaming_svd_experiment():
    print("=" * 70)
    print("STREAMING TILE SVD — Incremental Factorization")
    print("=" * 70)
    print(f"PyTorch: {'✓' if TORCH_AVAILABLE else '✗'} | "
          f"CUDA: {'✓' if TORCH_AVAILABLE and torch.cuda.is_available() else '✗'}")
    print()
    
    BATCH_SIZE = 50
    TOTAL_GAMES = 500
    N_BATCHES = TOTAL_GAMES // BATCH_SIZE
    
    games_to_test = [
        ("tictactoe", TicTacToe),
        ("connect4", Connect4),
    ]
    
    all_results = {}
    
    for game_name, GameClass in games_to_test:
        print(f"\n{'='*60}")
        print(f"  {game_name.upper()} — Streaming SVD Analysis")
        print(f"{'='*60}")
        
        field = StreamingTileField()
        
        svd_history = []       # full SVD at each batch
        inc_svd_history = []   # incremental SVD at each batch
        prev_U, prev_S, prev_Vh = None, None, None
        
        batch_results = []
        
        for batch_idx in range(N_BATCHES):
            games_in_batch = list(range(batch_idx * BATCH_SIZE, (batch_idx + 1) * BATCH_SIZE))
            
            # ─── Train one batch ─────────────────────────
            for _ in games_in_batch:
                game = GameClass()
                history_x = []
                history_o = []
                
                while not game.done:
                    actions = game.legal_actions()
                    if not actions:
                        break
                    
                    state_str = str(game.state())
                    h = field.get_or_create(state_str, actions)
                    a = field.choose(h, actions)
                    
                    if game.current == 'X':
                        history_x.append((h, a))
                    else:
                        history_o.append((h, a))
                    
                    game.step(a)
                
                # Record outcomes
                winner = getattr(game, 'winner', None)
                for h, a in history_x:
                    field.record(h, a, winner == 'X')
                for h, a in history_o:
                    field.record(h, a, winner == 'O')
            
            # Evolve after each batch
            field.evolve()
            
            # ─── Build score matrix ──────────────────────
            matrix, states, actions = build_score_matrix(field)
            if matrix is None:
                continue
            
            # ─── Full SVD ────────────────────────────────
            t0 = time.time()
            U_full, S_full, Vh_full = compute_svd(matrix)
            t_full = time.time() - t0
            
            # ─── Incremental SVD ─────────────────────────
            t0 = time.time()
            U_inc, S_inc, Vh_inc = incremental_svd_update(
                prev_U, prev_S, prev_Vh, matrix
            )
            t_inc = time.time() - t0
            
            prev_U, prev_S, prev_Vh = U_full, S_full, Vh_full
            
            # ─── Analysis ────────────────────────────────
            concentration = singular_value_concentration(S_full)
            rank1_energy = concentration[0] if concentration else 0
            rank2_energy = concentration[1] if len(concentration) > 1 else 0
            
            # Compare full vs incremental
            n_common = min(len(S_full), len(S_inc))
            if n_common > 0:
                sv_corr = np.corrcoef(S_full[:n_common], S_inc[:n_common])[0, 1]
            else:
                sv_corr = 0.0
            
            # Cosine similarity of rank-1 directions
            v1_full = Vh_full[0] if len(Vh_full) > 0 else np.array([])
            v1_inc = Vh_inc[0] if len(Vh_inc) > 0 else np.array([])
            if len(v1_full) > 0 and len(v1_inc) > 0:
                max_l = max(len(v1_full), len(v1_inc))
                v1f = np.zeros(max_l); v1i = np.zeros(max_l)
                v1f[:len(v1_full)] = v1_full
                v1i[:len(v1_inc)] = v1_inc
                rank1_cos = abs(np.dot(v1f, v1i) / (np.linalg.norm(v1f) * np.linalg.norm(v1i) + 1e-10))
            else:
                rank1_cos = 0.0
            
            entry = {
                "batch": batch_idx + 1,
                "games_played": (batch_idx + 1) * BATCH_SIZE,
                "n_states": len(states),
                "n_actions": len(actions),
                "matrix_shape": list(matrix.shape),
                "singular_values": S_full.tolist()[:10],
                "rank1_energy": float(rank1_energy),
                "rank2_energy": float(rank2_energy),
                "effective_rank": float(np.sum(S_full > S_full[0] * 0.01)),
                "sv_correlation_full_vs_inc": float(sv_corr),
                "rank1_cosine_full_vs_inc": float(rank1_cos),
                "time_full_svd_ms": t_full * 1000,
                "time_inc_svd_ms": t_inc * 1000,
                "speedup": t_full / max(t_inc, 1e-10),
            }
            
            svd_history.append({
                "S": S_full,
                "Vh": Vh_full,
                "entry": entry,
            })
            inc_svd_history.append({
                "S": S_inc,
                "Vh": Vh_inc,
            })
            batch_results.append(entry)
            
            print(f"  Batch {batch_idx+1:2d}/{N_BATCHES} | "
                  f"states={len(states):4d} actions={len(actions):2d} | "
                  f"σ₁={S_full[0]:.4f} σ₂={S_full[1] if len(S_full)>1 else 0:.4f} | "
                  f"rank1%={rank1_energy*100:5.1f}% | "
                  f"full={t_full*1000:6.1f}ms inc={t_inc*1000:6.1f}ms | "
                  f"cos(R1)={rank1_cos:.4f}")
        
        # ─── Rank-1 stability over time ──────────────────
        stabilities = rank1_stability(svd_history)
        inc_stabilities = rank1_stability(inc_svd_history)
        
        # Find when rank-1 stabilizes (cos > 0.99 for 3 consecutive batches)
        stabilization_batch = None
        for i in range(2, len(stabilities)):
            if stabilities[i] > 0.99 and stabilities[i-1] > 0.99 and stabilities[i-2] > 0.99:
                stabilization_batch = i + 1  # 1-indexed
                break
        
        # Spectral evolution: track top-5 singular values across batches
        spectral_evolution = []
        for h in svd_history:
            top5 = h["S"][:min(5, len(h["S"]))].tolist()
            while len(top5) < 5:
                top5.append(0.0)
            spectral_evolution.append(top5)
        
        game_result = {
            "game": game_name,
            "total_games": TOTAL_GAMES,
            "batch_size": BATCH_SIZE,
            "batches": batch_results,
            "rank1_stability": [float(s) for s in stabilities],
            "rank1_stability_inc": [float(s) for s in inc_stabilities],
            "spectral_evolution": spectral_evolution,
            "rank1_stabilization_batch": stabilization_batch,
            "rank1_stabilization_game": stabilization_batch * BATCH_SIZE if stabilization_batch else None,
            "final_rank1_energy": batch_results[-1]["rank1_energy"] if batch_results else None,
            "final_effective_rank": batch_results[-1]["effective_rank"] if batch_results else None,
        }
        
        all_results[game_name] = game_result
        
        # ─── Summary ────────────────────────────────────
        print(f"\n  {'─'*56}")
        if stabilization_batch:
            print(f"  ★ Rank-1 STABILIZED at batch {stabilization_batch} "
                  f"(game {stabilization_batch * BATCH_SIZE})")
        else:
            print(f"  ✗ Rank-1 did NOT fully stabilize in {TOTAL_GAMES} games")
        
        if batch_results:
            print(f"  Final rank-1 energy: {batch_results[-1]['rank1_energy']*100:.1f}%")
            print(f"  Final effective rank: {batch_results[-1]['effective_rank']:.1f}")
            print(f"  Avg full SVD: {np.mean([b['time_full_svd_ms'] for b in batch_results]):.1f}ms")
            print(f"  Avg inc SVD:  {np.mean([b['time_inc_svd_ms'] for b in batch_results]):.1f}ms")
            avg_speedup = np.mean([b['speedup'] for b in batch_results[1:]])  # skip first (no prev)
            print(f"  Avg speedup:  {avg_speedup:.2f}x")
    
    # ─── Cross-Game Comparison ────────────────────────────
    print(f"\n{'='*70}")
    print("  CROSS-GAME COMPARISON")
    print(f"{'='*70}")
    
    for game_name, res in all_results.items():
        stab = res["rank1_stabilization_batch"]
        print(f"\n  {game_name.upper()}:")
        print(f"    Rank-1 stabilized: {'batch ' + str(stab) + f' (game {stab*50})' if stab else 'NOT YET'}")
        print(f"    Final rank-1 energy: {res['final_rank1_energy']*100:.1f}%")
        print(f"    Final effective rank: {res['final_effective_rank']:.1f}")
    
    # ─── Key Findings ─────────────────────────────────────
    print(f"\n{'='*70}")
    print("  KEY FINDINGS")
    print(f"{'='*70}")
    
    for game_name, res in all_results.items():
        batches = res["batches"]
        if not batches:
            continue
        
        # Energy trajectory
        energies = [b["rank1_energy"] for b in batches]
        early = energies[:3]  # first 3 batches (150 games)
        late = energies[-3:]  # last 3 batches
        
        print(f"\n  {game_name.upper()}:")
        print(f"    Rank-1 energy: {np.mean(early)*100:.1f}% (early) → {np.mean(late)*100:.1f}% (late)")
        
        if res["rank1_stabilization_game"]:
            pct = res["rank1_stabilization_game"] / TOTAL_GAMES * 100
            print(f"    ★ Dominant axis emerges at {pct:.0f}% of training — "
                  f"early stopping viable!")
        else:
            # Check if trending toward stability
            if len(res["rank1_stability"]) > 0:
                max_stab = max(res["rank1_stability"])
                print(f"    Max rank-1 stability: {max_stab:.4f} "
                      f"(need >0.99 for 3 consecutive)")
                print(f"    Not yet stable — may need more games")
    
    # ─── Save results ─────────────────────────────────────
    # Strip numpy arrays for JSON serialization
    json_results = {}
    for game_name, res in all_results.items():
        jr = {}
        for k, v in res.items():
            if isinstance(v, np.ndarray):
                jr[k] = v.tolist()
            elif isinstance(v, np.floating):
                jr[k] = float(v)
            elif isinstance(v, np.integer):
                jr[k] = int(v)
            elif isinstance(v, list):
                jr[k] = [float(x) if isinstance(x, (np.floating, np.float32, np.float64)) else x for x in v]
            else:
                jr[k] = v
        json_results[game_name] = jr
    
    results_path = os.path.join(os.path.dirname(__file__) or ".", "streaming-tile-svd-results.json")
    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\n  Results saved to {results_path}")
    
    return all_results


if __name__ == "__main__":
    results = run_streaming_svd_experiment()
