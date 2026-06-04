"""
Parallel ZeroClaw Arena — Uses all 24 Ryzen cores.

Runs multiple games in parallel using multiprocessing.Pool.
Each core runs its own game instance, exploring independently.
Results merge at the end of each generation.

Target: 24x speedup for exploration phase.
"""

import multiprocessing as mp
import time
import json
import os
import sys
import random
import hashlib
import numpy as np
from collections import defaultdict
from functools import partial

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def hash_embed(text, dim=64):
    h = hashlib.blake2b(text.encode(), digest_size=dim).digest()
    v = np.array([b/255.0 for b in h], dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-10)


def run_single_game(args):
    """Run a single game episode. Called by worker processes."""
    game_type, game_id, seed = args
    random.seed(seed)
    np.random.seed(seed)
    
    # Import games locally in each worker
    import zeroclaw
    
    if game_type == 'tictactoe':
        game = zeroclaw.TicTacToe()
    elif game_type == 'connect4':
        game = zeroclaw.Connect4()
    elif game_type == 'go9x9':
        game = zeroclaw.Go9x9()
    else:
        return []
    
    transitions = []
    while not game.done:
        actions = game.legal_actions()
        if not actions:
            break
        action = random.choice(actions)
        state_before = str(game.state())
        reward, done = game.step(action)
        state_after = str(game.state())
        
        transitions.append({
            'state': state_before,
            'action': action,
            'reward': reward,
            'next_state': state_after,
            'done': done,
            'game': game_type,
            'game_id': game_id,
        })
    
    return transitions


def run_parallel_exploration(game_type, num_games=200, num_workers=None):
    """Run exploration in parallel across all cores."""
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    print(f"  Spawning {num_games} games across {num_workers} cores...")
    
    args = [(game_type, i, random.randint(0, 2**31)) for i in range(num_games)]
    
    start = time.perf_counter()
    with mp.Pool(num_workers) as pool:
        results = pool.map(run_single_game, args)
    
    all_transitions = []
    for transitions in results:
        all_transitions.extend(transitions)
    
    elapsed = time.perf_counter() - start
    
    print(f"  {len(all_transitions)} transitions in {elapsed:.1f}s ({len(all_transitions)/elapsed:.0f} trans/s)")
    
    return all_transitions, elapsed


def run_single_threaded(game_type, num_games=200):
    """Same workload, single-threaded for comparison."""
    start = time.perf_counter()
    all_transitions = []
    for i in range(num_games):
        result = run_single_game((game_type, i, random.randint(0, 2**31)))
        all_transitions.extend(result)
    elapsed = time.perf_counter() - start
    return all_transitions, elapsed


def benchmark():
    print("=" * 60)
    print("PARALLEL ARENA BENCHMARK — 24 Ryzen Cores")
    print("=" * 60)
    print(f"CPU cores: {mp.cpu_count()}")
    print()
    
    results = {}
    
    for game in ['tictactoe', 'connect4', 'go9x9']:
        print(f"\n=== {game.upper()} ===")
        
        # Single-threaded
        print(f"  Single-threaded (200 games)...")
        _, st_time = run_single_threaded(game, 200)
        
        # Parallel
        print(f"  Parallel ({mp.cpu_count()} cores, 200 games)...")
        _, pt_time = run_parallel_exploration(game, 200)
        
        speedup = st_time / pt_time
        print(f"  Speedup: {speedup:.1f}x")
        
        results[game] = {
            'single_threaded_s': st_time,
            'parallel_s': pt_time,
            'speedup': speedup,
            'cores': mp.cpu_count(),
        }
    
    # Scale test
    print(f"\n=== SCALE TEST: 5000 Connect4 games ===")
    _, st_5k = run_single_threaded('connect4', 500)
    _, pt_5k = run_parallel_exploration('connect4', 5000)
    
    print(f"  Single (500 games): {st_5k:.1f}s")
    print(f"  Parallel (5000 games): {pt_5k:.1f}s")
    print(f"  Throughput: {5000/pt_5k:.0f} games/s")
    
    results['scale_test'] = {
        'single_500_s': st_5k,
        'parallel_5000_s': pt_5k,
        'throughput_games_per_s': 5000/pt_5k,
    }
    
    # Save
    out = os.path.expanduser("~/repos/zeroclaw-arena/parallel-benchmarks.json")
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")
    
    return results


if __name__ == "__main__":
    benchmark()
