"""
Tile Dead Code Elimination
==========================
Like a compiler's dead code elimination pass, this finds and removes tiles
that never contribute to winning — then measures whether the "compiled" field
loses any performance.

1. Train tile field on Connect4 (500 games, ~4000 tiles)
2. Profile: play 200 more games, track which tiles are ACTUALLY visited
3. Identify:
   - Dead tiles: never visited (unreachable states)
   - Dead reflexes: actions with score < 0.1 (never worth choosing)
   - Redundant tiles: tiles whose top action matches nearest neighbor's top action
4. Prune: remove dead tiles and reflexes
5. Compare: full field vs pruned field win rate, size, speed

If pruning 80% of tiles loses < 5% performance, the tile field is massively
redundant — the "compiled" version only needs the essential tiles.
This is tree-shaking for game AI.
"""

import random
import time
import json
import os
import sys
import numpy as np
from copy import deepcopy
from collections import defaultdict

# Add repo to path
sys.path.insert(0, os.path.dirname(__file__))
from zeroclaw import Connect4, StateTile


# ─── Connect4 Helpers ─────────────────────────────────────

def play_game(tile_field: dict, opponent_field: dict = None, 
              player_uses_tiles: bool = True, opponent_uses_tiles: bool = True,
              profile: dict = None) -> dict:
    """Play one Connect4 game between tile-field player (X) and opponent (O).
    
    Args:
        tile_field: Player X's tile field
        opponent_field: Player O's tile field (None = random)
        player_uses_tiles: If False, X plays randomly
        opponent_uses_tiles: If False, O plays randomly  
        profile: If provided, dict to accumulate profiling data
    
    Returns:
        {"winner": "X"/"O"/None, "moves": int, "history": [(state_hash, action), ...]}
    """
    game = Connect4()
    history = []
    
    while not game.done:
        state = game.state()
        actions = game.legal_actions()
        if not actions:
            break
        
        state_hash = state.hash()
        is_x = game.current == 'X'
        
        # Select action
        if is_x and player_uses_tiles:
            if state_hash in tile_field:
                action = tile_field[state_hash].best_action(actions, n_simulations=10, game=game)
            else:
                action = random.choice(actions)
        elif not is_x and opponent_uses_tiles and opponent_field is not None:
            if state_hash in opponent_field:
                action = opponent_field[state_hash].best_action(actions, n_simulations=10, game=game)
            else:
                action = random.choice(actions)
        else:
            action = random.choice(actions)
        
        # Profile: track visits
        if profile is not None:
            if state_hash not in profile["tile_visits"]:
                profile["tile_visits"][state_hash] = 0
            profile["tile_visits"][state_hash] += 1
            
            if is_x and state_hash in tile_field:
                tile = tile_field[state_hash]
                if action in tile.reflexes:
                    key = f"{state_hash}:{action}"
                    if key not in profile["reflex_used"]:
                        profile["reflex_used"][key] = 0
                    profile["reflex_used"][key] += 1
        
        game.step(action)
        history.append((state_hash, action, is_x))
    
    return {
        "winner": game.winner,
        "moves": game.turn,
        "history": history,
    }


def train_tile_field(n_games: int = 500, seed: int = 42) -> dict:
    """Train a tile field on Connect4 by self-play."""
    random.seed(seed)
    np.random.seed(seed)
    
    tile_field = {}
    wins = {"X": 0, "O": 0, "draw": 0}
    
    for g in range(n_games):
        game = Connect4()
        game_history = []  # [(state_hash, action, player)]
        
        while not game.done:
            state = game.state()
            actions = game.legal_actions()
            if not actions:
                break
            
            state_hash = state.hash()
            
            # Create tile if needed
            if state_hash not in tile_field:
                tile_field[state_hash] = StateTile(state_hash, str(state), actions)
            
            tile = tile_field[state_hash]
            action = tile.best_action(actions, n_simulations=10, game=game)
            game.step(action)
            game_history.append((state_hash, action, game.current))
        
        # Record outcomes
        winner = game.winner
        if winner == 'X':
            wins["X"] += 1
        elif winner == 'O':
            wins["O"] += 1
        else:
            wins["draw"] += 1
        
        # Update tiles based on outcome
        for state_hash, action, player in game_history:
            if state_hash in tile_field:
                won = (player == winner) if winner else False
                tile_field[state_hash].record(action, won)
        
        # Evolve every 20 games
        if (g + 1) % 20 == 0:
            for tile in tile_field.values():
                tile.evolve()
        
        if (g + 1) % 100 == 0:
            print(f"  Training game {g+1}/{n_games}: "
                  f"{len(tile_field)} tiles, "
                  f"X={wins['X']} O={wins['O']} D={wins['draw']}")
    
    return tile_field, wins


def profile_tile_field(tile_field: dict, n_games: int = 200, seed: int = 123) -> dict:
    """Play games and profile which tiles/reflexes are actually used."""
    random.seed(seed)
    np.random.seed(seed)
    
    profile = {
        "tile_visits": {},       # state_hash -> visit count
        "reflex_used": {},       # "state_hash:action" -> use count
        "games_played": n_games,
    }
    
    results = {"X": 0, "O": 0, "draw": 0}
    
    for g in range(n_games):
        result = play_game(tile_field, profile=profile)
        w = result["winner"]
        if w:
            results[w] += 1
        else:
            results["draw"] += 1
    
    return profile, results


def find_dead_tiles(tile_field: dict, profile: dict) -> list:
    """Tiles that were never visited during profiling."""
    dead = []
    for state_hash, tile in tile_field.items():
        visits = profile["tile_visits"].get(state_hash, 0)
        if visits == 0:
            dead.append(state_hash)
    return dead


def find_dead_reflexes(tile_field: dict) -> dict:
    """Actions with score < 0.1 — effectively never worth choosing."""
    dead_reflexes = {}  # state_hash -> [action, ...]
    for state_hash, tile in tile_field.items():
        dead_actions = []
        for action, data in tile.reflexes.items():
            if data["score"] < 0.1:
                dead_actions.append(action)
        if dead_actions:
            dead_reflexes[state_hash] = dead_actions
    return dead_reflexes


def find_redundant_tiles(tile_field: dict) -> list:
    """Tiles whose top action matches their nearest neighbor's top action.
    
    Uses state_hash similarity (edit distance on hash strings) as a proxy
    for state similarity. If two nearby states have the same best action,
    one can be "merged" into the other.
    """
    # Build a mapping of tile -> top action
    tile_top_actions = {}
    for state_hash, tile in tile_field.items():
        if tile.reflexes:
            best = max(tile.reflexes.items(), key=lambda x: x[1]["score"])
            tile_top_actions[state_hash] = best[0]  # action name
    
    # For each tile, check if any "neighbor" (similar hash) has same top action
    hashes = list(tile_top_actions.keys())
    redundant = []
    
    # Group by first 6 chars of hash (coarse similarity)
    groups = defaultdict(list)
    for h in hashes:
        prefix = h[:6]
        groups[prefix].append(h)
    
    # Also check adjacent prefixes
    for prefix, group in groups.items():
        # Check within group
        if len(group) > 1:
            for i, h1 in enumerate(group):
                for h2 in group[i+1:]:
                    if tile_top_actions[h1] == tile_top_actions[h2]:
                        # Mark the one with fewer visits as redundant
                        t1_chosen = sum(d["chosen"] for d in tile_field[h1].reflexes.values())
                        t2_chosen = sum(d["chosen"] for d in tile_field[h2].reflexes.values())
                        redundant.append(h1 if t1_chosen <= t2_chosen else h2)
    
    return list(set(redundant))


def prune_field(tile_field: dict, dead_tiles: list, dead_reflexes: dict, 
                redundant_tiles: list) -> dict:
    """Remove dead tiles, dead reflexes, and redundant tiles."""
    pruned = {}
    
    all_dead = set(dead_tiles) | set(redundant_tiles)
    
    for state_hash, tile in tile_field.items():
        if state_hash in all_dead:
            continue  # Skip entirely
        
        # Deep copy tile
        new_tile = StateTile(state_hash, tile.state_str, list(tile.reflexes.keys()))
        # Copy over scores
        for action, data in tile.reflexes.items():
            if action in new_tile.reflexes:
                new_tile.reflexes[action] = deepcopy(data)
        
        # Remove dead reflexes
        if state_hash in dead_reflexes:
            for action in dead_reflexes[state_hash]:
                if action in new_tile.reflexes and len(new_tile.reflexes) > 1:
                    del new_tile.reflexes[action]
        
        pruned[state_hash] = new_tile
    
    return pruned


def evaluate_field(tile_field: dict, n_games: int = 200, seed: int = 999) -> dict:
    """Evaluate a tile field by playing against random opponent."""
    random.seed(seed)
    np.random.seed(seed)
    
    results = {"X": 0, "O": 0, "draw": 0}
    total_time = 0.0
    
    for g in range(n_games):
        game = Connect4()
        start = time.perf_counter()
        
        while not game.done:
            state = game.state()
            actions = game.legal_actions()
            if not actions:
                break
            
            state_hash = state.hash()
            
            if game.current == 'X':
                # Tile field player
                if state_hash in tile_field:
                    action = tile_field[state_hash].best_action(actions, n_simulations=10, game=game)
                else:
                    # Tile not found — fall back to MC simulation
                    # Pick best action by simulation
                    best_action = actions[0]
                    best_wins = -1
                    for a in actions:
                        wins = 0
                        for _ in range(5):
                            gc = Connect4()
                            gc.board = [row[:] for row in game.board]
                            gc.current = game.current
                            gc.done = game.done
                            gc.winner = game.winner
                            gc.turn = game.turn
                            gc.step(a)
                            while not gc.done:
                                la = gc.legal_actions()
                                if not la:
                                    break
                                gc.step(random.choice(la))
                            if gc.winner == 'X':
                                wins += 1
                        if wins > best_wins:
                            best_wins = wins
                            best_action = a
                    action = best_action
            else:
                # Random opponent
                action = random.choice(actions)
            
            game.step(action)
        
        elapsed = time.perf_counter() - start
        total_time += elapsed
        
        if game.winner:
            results[game.winner] += 1
        else:
            results["draw"] += 1
    
    return {
        "results": results,
        "win_rate": results["X"] / n_games,
        "total_time": total_time,
        "avg_time_per_game": total_time / n_games,
        "games": n_games,
    }


def count_reflexes(tile_field: dict) -> int:
    """Total number of state-action pairs in the field."""
    return sum(len(tile.reflexes) for tile in tile_field.values())


def main():
    print("=" * 70)
    print("TILE DEAD CODE ELIMINATION — Connect4")
    print("=" * 70)
    print()
    print("Like a compiler's dead code elimination pass, we find and remove")
    print("tiles that never contribute to winning. If pruning 80% loses < 5%")
    print("performance, the tile field is massively redundant.")
    print()
    
    # ─── Phase 1: Train ────────────────────────────────
    print("─" * 60)
    print("PHASE 1: Train tile field (500 games)")
    print("─" * 60)
    
    tile_field, train_wins = train_tile_field(n_games=500, seed=42)
    
    full_tiles = len(tile_field)
    full_reflexes = count_reflexes(tile_field)
    train_total = sum(train_wins.values())
    
    print(f"\n  Training complete:")
    print(f"    Tiles created: {full_tiles}")
    print(f"    Total reflexes: {full_reflexes}")
    print(f"    Training results: X={train_wins['X']} O={train_wins['O']} D={train_wins['draw']}")
    print(f"    X win rate: {train_wins['X']/train_total:.1%}")
    
    # ─── Phase 2: Profile ──────────────────────────────
    print(f"\n{'─' * 60}")
    print("PHASE 2: Profile tile usage (200 games)")
    print("─" * 60)
    
    profile, prof_wins = profile_tile_field(tile_field, n_games=200, seed=123)
    prof_total = sum(prof_wins.values())
    
    visited_tiles = len([v for v in profile["tile_visits"].values() if v > 0])
    visited_reflexes = len(profile["reflex_used"])
    
    print(f"  Profile complete:")
    print(f"    Tiles visited: {visited_tiles}/{full_tiles} ({visited_tiles/full_tiles:.1%})")
    print(f"    Reflexes used: {visited_reflexes}")
    print(f"    Profile games: X={prof_wins['X']} O={prof_wins['O']} D={prof_wins['draw']}")
    print(f"    X win rate: {prof_wins['X']/prof_total:.1%}")
    
    # ─── Phase 3: Identify Dead Code ───────────────────
    print(f"\n{'─' * 60}")
    print("PHASE 3: Identify dead code")
    print("─" * 60)
    
    dead_tiles = find_dead_tiles(tile_field, profile)
    dead_reflexes = find_dead_reflexes(tile_field)
    redundant_tiles = find_redundant_tiles(tile_field)
    
    total_dead_reflex_count = sum(len(v) for v in dead_reflexes.values())
    
    print(f"  Dead tiles (never visited): {len(dead_tiles)} ({len(dead_tiles)/full_tiles:.1%})")
    print(f"  Dead reflexes (score < 0.1): {total_dead_reflex_count} across {len(dead_reflexes)} tiles")
    print(f"  Redundant tiles (same top action as neighbor): {len(redundant_tiles)} ({len(redundant_tiles)/full_tiles:.1%})")
    
    # ─── Phase 4: Prune ────────────────────────────────
    print(f"\n{'─' * 60}")
    print("PHASE 4: Prune dead code")
    print("─" * 60)
    
    pruned_field = prune_field(tile_field, dead_tiles, dead_reflexes, redundant_tiles)
    
    pruned_tiles = len(pruned_field)
    pruned_reflexes = count_reflexes(pruned_field)
    
    tile_reduction = (1 - pruned_tiles / full_tiles) * 100
    reflex_reduction = (1 - pruned_reflexes / full_reflexes) * 100
    
    print(f"  Full field: {full_tiles} tiles, {full_reflexes} reflexes")
    print(f"  Pruned field: {pruned_tiles} tiles, {pruned_reflexes} reflexes")
    print(f"  Tile reduction: {tile_reduction:.1f}%")
    print(f"  Reflex reduction: {reflex_reduction:.1f}%")
    
    # ─── Phase 5: Compare ──────────────────────────────
    print(f"\n{'─' * 60}")
    print("PHASE 5: Performance comparison (200 games each)")
    print("─" * 60)
    
    print("  Evaluating FULL field...")
    full_eval = evaluate_field(tile_field, n_games=200, seed=999)
    
    print("  Evaluating PRUNED field...")
    pruned_eval = evaluate_field(pruned_field, n_games=200, seed=999)
    
    win_rate_delta = full_eval["win_rate"] - pruned_eval["win_rate"]
    speedup = full_eval["avg_time_per_game"] / pruned_eval["avg_time_per_game"] if pruned_eval["avg_time_per_game"] > 0 else 1.0
    
    print(f"\n  {'Metric':<25} {'Full':>12} {'Pruned':>12} {'Delta':>12}")
    print(f"  {'─'*61}")
    print(f"  {'Win rate':<25} {full_eval['win_rate']:>11.1%} {pruned_eval['win_rate']:>11.1%} {win_rate_delta:>+11.1%}")
    print(f"  {'Tiles':<25} {full_tiles:>12} {pruned_tiles:>12} {pruned_tiles-full_tiles:>+12}")
    print(f"  {'Reflexes':<25} {full_reflexes:>12} {pruned_reflexes:>12} {pruned_reflexes-full_reflexes:>+12}")
    print(f"  {'Avg time/game (ms)':<25} {full_eval['avg_time_per_game']*1000:>11.1f} {pruned_eval['avg_time_per_game']*1000:>11.1f} {'':>12}")
    print(f"  {'Speedup':<25} {'1.00x':>12} {speedup:>11.2f}x {'':>12}")
    
    # ─── Verdict ───────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("VERDICT")
    print(f"{'═' * 60}")
    
    performance_held = abs(win_rate_delta) < 0.05
    massive_prune = tile_reduction > 50
    
    if performance_held and massive_prune:
        verdict = "TREESHAKING SUCCESS"
        emoji = "🔥"
        msg = (f"Pruned {tile_reduction:.0f}% of tiles while losing only "
               f"{abs(win_rate_delta):.1%} win rate. The tile field is "
               f"MASSIVELY REDUNDANT — the compiled version only needs "
               f"{pruned_tiles} essential tiles.")
    elif performance_held:
        verdict = "PRUNING EFFECTIVE"
        emoji = "✅"
        msg = (f"Pruned {tile_reduction:.0f}% of tiles with < 5% win rate loss. "
               f"Dead code elimination works, though redundancy is moderate.")
    elif massive_prune:
        verdict = "TOO AGGRESSIVE"
        emoji = "⚠️"
        msg = (f"Pruned {tile_reduction:.0f}% of tiles but lost {abs(win_rate_delta):.1%} win rate. "
               f"The field has redundancy, but some 'dead' tiles matter.")
    else:
        verdict = "FIELD IS LEAN"
        emoji = "📊"
        msg = (f"Only {tile_reduction:.0f}% of tiles could be pruned. "
               f"The tile field is already efficient — most tiles earn their keep.")
    
    print(f"  {emoji} {verdict}")
    print(f"  {msg}")
    
    # ─── Save Results ──────────────────────────────────
    results = {
        "experiment": "tile_dead_code_elimination",
        "description": "Find and remove tiles that never contribute to winning",
        "analogy": "Like tree-shaking in JS bundlers or dead code elimination in compilers",
        "training": {
            "games": 500,
            "tiles_created": full_tiles,
            "reflexes_created": full_reflexes,
            "results": train_wins,
        },
        "profiling": {
            "games": 200,
            "tiles_visited": visited_tiles,
            "tiles_total": full_tiles,
            "visit_rate": visited_tiles / full_tiles,
            "results": prof_wins,
        },
        "dead_code": {
            "dead_tiles": len(dead_tiles),
            "dead_tiles_pct": len(dead_tiles) / full_tiles,
            "dead_reflexes": total_dead_reflex_count,
            "dead_reflexes_in_tiles": len(dead_reflexes),
            "redundant_tiles": len(redundant_tiles),
            "redundant_tiles_pct": len(redundant_tiles) / full_tiles,
        },
        "pruning": {
            "full_tiles": full_tiles,
            "full_reflexes": full_reflexes,
            "pruned_tiles": pruned_tiles,
            "pruned_reflexes": pruned_reflexes,
            "tile_reduction_pct": tile_reduction,
            "reflex_reduction_pct": reflex_reduction,
        },
        "comparison": {
            "full": {
                "win_rate": full_eval["win_rate"],
                "results": full_eval["results"],
                "avg_time_per_game_ms": full_eval["avg_time_per_game"] * 1000,
            },
            "pruned": {
                "win_rate": pruned_eval["win_rate"],
                "results": pruned_eval["results"],
                "avg_time_per_game_ms": pruned_eval["avg_time_per_game"] * 1000,
            },
            "win_rate_delta": win_rate_delta,
            "speedup": speedup,
        },
        "verdict": verdict,
        "verdict_detail": msg,
    }
    
    results_path = os.path.join(os.path.dirname(__file__), "tile-dead-code-results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_path}")
    
    return results


if __name__ == "__main__":
    main()
