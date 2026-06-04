"""
Holographic Bound Conjecture Experiment
========================================
Conjecture: A tile field of N tiles can be reconstructed from O(√N) tiles' negative space alone.

Test:
1. Train a full tile field on tic-tac-toe (500 games, ~800 tiles)
2. Take random subsets: [10, 25, 50, 100, 200, 400, 600]
3. Reconstruct full field: subset tiles copied directly, others predicted from nearest subset tile
4. Measure: what minimum subset achieves >95% of full field's win rate?

Hypothesis: O(√N) ≈ 28 tiles should be sufficient.
"""

import random
import json
import numpy as np
from copy import deepcopy
from zeroclaw import TicTacToe, ZeroClaw, StateTile

# ─── Train a full tile field ─────────────────────────────

def train_full_field(n_games=500):
    """Train a ZeroClaw agent on tic-tac-toe and return the tile field + agent."""
    agent = ZeroClaw("holographic_test", "tictactoe")

    for i in range(n_games):
        game = TicTacToe()
        game.reset()
        transitions = []

        while not game.done:
            state = game.state()
            actions = game.legal_actions()
            if not actions:
                break

            shash = state.hash()
            if shash not in agent.tile_field:
                agent.tile_field[shash] = StateTile(shash, str(state), actions)

            tile = agent.tile_field[shash]
            action = tile.best_action(actions, n_simulations=10, game=game)

            reward, done = game.step(action)
            tile.record(action, game.winner == 'X' if game.winner else False)
            transitions.append((shash, action, game.winner))

            if done:
                break

        # Backpropagate win/loss
        winner = game.winner
        for shash, action, _ in transitions:
            if shash in agent.tile_field:
                agent.tile_field[shash].record(action, winner == 'X')

        # Evolve every 25 games
        if (i + 1) % 25 == 0:
            for tile in agent.tile_field.values():
                tile.evolve()

    return agent


def evaluate_win_rate(tile_field, n_games=1000):
    """Evaluate a tile field's win rate as X against random O."""
    wins = 0
    draws = 0
    losses = 0

    for _ in range(n_games):
        game = TicTacToe()
        game.reset()

        while not game.done:
            state = game.state()
            actions = game.legal_actions()
            if not actions:
                break

            shash = state.hash()

            if game.current == 'X':
                # Use tile field
                if shash in tile_field:
                    tile = tile_field[shash]
                    best_a = max(actions, key=lambda a: tile.reflexes.get(a, {}).get("score", 0.5))
                else:
                    best_a = random.choice(actions)
                game.step(best_a)
            else:
                # Random opponent
                game.step(random.choice(actions))

        if game.winner == 'X':
            wins += 1
        elif game.winner == 'O':
            losses += 1
        else:
            draws += 1

    return wins / n_games, draws / n_games, losses / n_games


def reconstruct_field(full_field, subset_keys):
    """Reconstruct full field from subset using nearest-neighbor prediction."""
    if not subset_keys:
        return {}

    # Collect state vectors for subset tiles
    subset_vectors = {}
    for key in subset_keys:
        tile = full_field[key]
        # Use a simple state representation as "position"
        # Parse state_str to get board position for spatial distance
        vec = state_to_vector(tile.state_str)
        subset_vectors[key] = vec

    subset_keys_list = list(subset_keys)
    subset_vecs = np.array([subset_vectors[k] for k in subset_keys_list])

    reconstructed = {}

    # Copy subset tiles directly
    for key in subset_keys:
        reconstructed[key] = deepcopy(full_field[key])

    # For all other tiles, predict from nearest subset tile
    for key, tile in full_field.items():
        if key in subset_keys:
            continue

        vec = state_to_vector(tile.state_str)

        # Find nearest tile in subset by Hamming distance on board
        dists = np.sum(subset_vecs != vec, axis=1)
        nearest_idx = np.argmin(dists)
        nearest_key = subset_keys_list[nearest_idx]

        # Copy reflexes from nearest, but adjust scores based on distance
        nearest_tile = full_field[nearest_key]
        distance = dists[nearest_idx]

        new_tile = StateTile(tile.state_hash, tile.state_str, list(tile.reflexes.keys()))

        # Transfer scores with distance-based decay
        # As distance increases, regress toward 0.5 (uncertainty)
        decay = max(0.0, 1.0 - distance / 9.0)  # 9 cells max

        for action in new_tile.reflexes:
            if action in nearest_tile.reflexes:
                source_score = nearest_tile.reflexes[action]["score"]
                new_tile.reflexes[action]["score"] = 0.5 + decay * (source_score - 0.5)
            # else keep default 0.5

        reconstructed[key] = new_tile

    return reconstructed


def state_to_vector(state_str):
    """Convert state string like '[turn=0|X]         ' to a numeric board vector."""
    # Extract board part after the ']'
    parts = state_str.split(']')
    if len(parts) >= 2:
        board_str = parts[1]
    else:
        board_str = state_str

    # Pad/truncate to 9 chars
    board_str = board_str.ljust(9)[:9]

    # Convert to numeric: X=1, O=-1, space=0
    vec = np.zeros(9)
    for i, ch in enumerate(board_str):
        if ch == 'X':
            vec[i] = 1
        elif ch == 'O':
            vec[i] = -1

    # Return as categorical array for Hamming distance
    return vec


def run_experiment():
    print("=" * 70)
    print("HOLOGRAPHIC BOUND CONJECTURE EXPERIMENT")
    print("=" * 70)

    # Step 1: Train full tile field
    print("\n[1] Training full tile field (500 games)...")
    random.seed(42)
    np.random.seed(42)
    agent = train_full_field(n_games=500)
    full_field = agent.tile_field
    N = len(full_field)
    print(f"    Total tiles: {N}")
    print(f"    √N ≈ {int(N**0.5)}")

    # Step 2: Evaluate full field baseline
    print("\n[2] Evaluating full field win rate (1000 games)...")
    full_wr, full_dr, full_lr = evaluate_win_rate(full_field, n_games=1000)
    print(f"    Win rate: {full_wr:.3f}, Draw rate: {full_dr:.3f}, Loss rate: {full_lr:.3f}")

    # Step 3: Test subsets
    subset_sizes = [10, 25, 50, 100, 200, 400, 600]
    sqrt_n = int(N**0.5)

    # Add √N specifically
    if sqrt_n not in subset_sizes:
        subset_sizes.append(sqrt_n)
        subset_sizes.sort()

    print(f"\n[3] Testing subsets: {subset_sizes}")
    print(f"    Hypothesis: √N = {sqrt_n} tiles should achieve >95% win rate")

    results = []
    all_keys = list(full_field.keys())

    for size in subset_sizes:
        if size > N:
            continue

        # Multiple random trials for robustness
        trial_wrs = []
        for trial in range(5):
            random.seed(42 + trial)
            subset_keys = set(random.sample(all_keys, min(size, N)))

            reconstructed = reconstruct_field(full_field, subset_keys)
            wr, dr, lr = evaluate_win_rate(reconstructed, n_games=500)
            trial_wrs.append(wr)

        avg_wr = np.mean(trial_wrs)
        std_wr = np.std(trial_wrs)
        pct_of_full = avg_wr / full_wr * 100 if full_wr > 0 else 0

        result = {
            "subset_size": size,
            "fraction": f"{size}/{N}",
            "pct_of_field": f"{size/N*100:.1f}%",
            "avg_win_rate": round(avg_wr, 4),
            "std_win_rate": round(std_wr, 4),
            "pct_of_full_wr": round(pct_of_full, 1),
            "trials": [round(w, 4) for w in trial_wrs],
        }
        results.append(result)

        marker = " ← √N" if size == sqrt_n else ""
        threshold = " ✅ >95%" if pct_of_full >= 95 else ""
        print(f"    {size:4d} tiles ({size/N*100:5.1f}%): "
              f"WR={avg_wr:.3f}±{std_wr:.3f}  "
              f"({pct_of_full:.1f}% of full){marker}{threshold}")

    # Step 4: Linear scan from small to large for exact threshold
    # (Binary search failed because the relationship isn't monotonic
    # with random subsets — it depends WHICH tiles you pick)
    print("\n[4] Scanning for minimum reliable 95% threshold...")
    scan_sizes = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300]
    
    first_95 = None
    for size in scan_sizes:
        if size > N:
            break
        # More trials for reliability (10 trials)
        trial_wrs = []
        for trial in range(10):
            random.seed(2000 + size * 100 + trial)
            subset_keys = set(random.sample(all_keys, min(size, N)))
            reconstructed = reconstruct_field(full_field, subset_keys)
            wr, _, _ = evaluate_win_rate(reconstructed, n_games=300)
            trial_wrs.append(wr)
        
        avg_wr = np.mean(trial_wrs)
        std_wr = np.std(trial_wrs)
        pct = avg_wr / full_wr * 100 if full_wr > 0 else 0
        
        # Use lower bound (mean - 1 std) as conservative estimate
        lb_pct = (avg_wr - std_wr) / full_wr * 100 if full_wr > 0 else 0
        
        marker = " ← √N" if size == sqrt_n else ""
        reliable = " ✅ reliable" if lb_pct >= 95 else ""
        print(f"    {size:4d} tiles: WR={avg_wr:.3f}±{std_wr:.3f} ({pct:.1f}%, LB={lb_pct:.1f}%){marker}{reliable}")
        
        if first_95 is None and lb_pct >= 95:
            first_95 = size
    
    best_min = first_95 if first_95 else N

    # Step 5: Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Total tiles (N):           {N}")
    print(f"  √N:                        {sqrt_n}")
    print(f"  Full field win rate:        {full_wr:.3f}")
    print(f"  Minimum for >95% wr:       {best_min} tiles")
    print(f"  Ratio (min/N):             {best_min/N:.3f}")
    print(f"  Ratio (min/√N):            {best_min/sqrt_n:.2f}")
    print()

    if best_min <= sqrt_n * 1.5:
        print("  🎯 CONJECTURE SUPPORTED: O(√N) tiles sufficient!")
    elif best_min <= sqrt_n * 3:
        print("  📊 CONJECTURE PARTIALLY SUPPORTED: ~O(√N) order correct, constant > 1")
    else:
        print("  ❌ CONJECTURE NOT SUPPORTED: need significantly more than √N tiles")

    # Save results
    output = {
        "experiment": "holographic_bound",
        "N": N,
        "sqrt_N": sqrt_n,
        "full_win_rate": full_wr,
        "minimum_95pct": best_min,
        "conjecture_supported": best_min <= sqrt_n * 1.5,
        "subset_results": results,
    }

    with open("holographic-bound-results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to holographic-bound-results.json")

    return output


if __name__ == "__main__":
    run_experiment()
