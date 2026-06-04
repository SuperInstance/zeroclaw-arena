"""
Minimum Opponent Exposure — How few games to reach competence?
==============================================================

Train TTT tile field in increments of 10 games, evaluating after each increment.
Also compare memorization (same games repeated) vs diversity (different games each time).

Key questions:
- How many games until 55% win rate?
- How many games until plateau?
- Does tile count or visit count matter more?
- Is there a "click" moment?
- Memorization vs diversity?
"""

import random
import json
import sys
import os
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from zeroclaw import TicTacToe, TileField


def evaluate_vs_random(field, n_games=500):
    """Evaluate trained tile field vs random player. Returns stats."""
    wins = 0
    losses = 0
    draws = 0
    total_moves_field = 0

    for _ in range(n_games):
        game = TicTacToe()
        while not game.done:
            state_str = str(game.state().state_str)
            actions = game.legal_actions()
            if not actions:
                break

            if game.current == 'X':
                # Tile field plays X
                action = field.choose_action(game, state_str, actions)
            else:
                # Random plays O
                action = random.choice(actions)

            game.step(action)
            if game.current == 'X':
                total_moves_field += 1

        if game.winner == 'X':
            wins += 1
        elif game.winner == 'O':
            losses += 1
        else:
            draws += 1

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / n_games,
        "loss_rate": losses / n_games,
        "draw_rate": draws / n_games,
    }


def count_total_visits(field):
    """Count total action visits across all tiles."""
    total = 0
    for tile in field.tiles.values():
        for action_data in tile.values():
            total += action_data["chosen"]
    return total


def avg_tile_visits(field):
    """Average visits per tile."""
    if not field.tiles:
        return 0
    return count_total_visits(field) / len(field.tiles)


def score_distribution(field):
    """Distribution of action scores."""
    scores = []
    for tile in field.tiles.values():
        for action_data in tile.values():
            scores.append(action_data["score"])

    if not scores:
        return {"mean": 0.5, "std": 0, "min": 0.5, "max": 0.5, "n": 0}

    import statistics
    return {
        "mean": round(statistics.mean(scores), 4),
        "std": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0,
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "n": len(scores),
    }


def generate_fixed_games(n_games=10):
    """Generate a fixed set of game trajectories for memorization test."""
    trajectories = []
    for _ in range(n_games):
        game = TicTacToe()
        history = []
        while not game.done:
            actions = game.legal_actions()
            if not actions:
                break
            action = random.choice(actions)
            history.append((str(game.state().state_str), action, game.current))
            game.step(action)
        history.append(game.winner)
        trajectories.append(history)
    return trajectories


def train_from_trajectories(field, trajectories, evolve_every=10):
    """Replay fixed game trajectories into the field (memorization)."""
    for traj in trajectories:
        winner = traj[-1]
        won = winner == 'X'
        for state_str, action, player in traj[:-1]:
            if player == 'X':
                tile = field.get_or_create(state_str, [str(i) for i in range(9)])
                field.record(state_str, action, won)
        field._game_count += 1
        if field._game_count % evolve_every == 0:
            field.evolve()


def run_incremental_experiment(mode="diverse", max_games=200, eval_games=500, seed=42):
    """Train incrementally and evaluate at each step."""
    random.seed(seed)
    results = []

    field = TileField(n_simulations=20, temperature=0.3)
    game = TicTacToe()

    # For memorization mode, pre-generate 10 fixed games
    fixed_trajectories = generate_fixed_games(10)

    for n_total in range(10, max_games + 1, 10):
        # Train 10 more games
        if mode == "diverse":
            for _ in range(10):
                game.reset()
                field.train_game(game)
        elif mode == "memorize":
            train_from_trajectories(field, fixed_trajectories)

        # Evaluate
        random.seed(seed + 10000)  # Separate seed for eval
        eval_stats = evaluate_vs_random(field, eval_games)
        random.seed(seed + n_total)  # Restore different seed for next training

        checkpoint = {
            "training_games": n_total,
            "mode": mode,
            "num_tiles": len(field.tiles),
            "total_visits": count_total_visits(field),
            "avg_visits_per_tile": round(avg_tile_visits(field), 2),
            "score_distribution": score_distribution(field),
            "evaluation": eval_stats,
        }
        results.append(checkpoint)

        wr = eval_stats["win_rate"]
        tiles = checkpoint["num_tiles"]
        visits = checkpoint["total_visits"]
        print(f"  [{mode:9s}] {n_total:3d} games | tiles={tiles:5d} | "
              f"visits={visits:6d} | WR={wr:.1%} | "
              f"W={eval_stats['wins']} D={eval_stats['draws']} L={eval_stats['losses']}")

    return results


def find_milestones(results):
    """Find key milestones in results."""
    first_55 = None
    plateau = None
    click_moment = None

    # Find first 55% win rate
    for r in results:
        if r["evaluation"]["win_rate"] >= 0.55:
            first_55 = r["training_games"]
            break

    # Find plateau: where improvement drops below 1% per 10 games
    for i in range(1, len(results)):
        delta = results[i]["evaluation"]["win_rate"] - results[i-1]["evaluation"]["win_rate"]
        if delta < 0.01 and results[i]["evaluation"]["win_rate"] > 0.50:
            plateau = results[i]["training_games"]
            break

    # Find "click" moment: largest single jump in win rate
    max_delta = 0
    for i in range(1, len(results)):
        delta = results[i]["evaluation"]["win_rate"] - results[i-1]["evaluation"]["win_rate"]
        if delta > max_delta:
            max_delta = delta
            click_moment = {
                "from_games": results[i-1]["training_games"],
                "to_games": results[i]["training_games"],
                "win_rate_jump": round(delta, 4),
                "from_tiles": results[i-1]["num_tiles"],
                "to_tiles": results[i]["num_tiles"],
            }

    return {
        "first_55_pct": first_55,
        "plateau_start": plateau,
        "click_moment": click_moment,
        "peak_win_rate": max(r["evaluation"]["win_rate"] for r in results),
        "final_win_rate": results[-1]["evaluation"]["win_rate"],
    }


def main():
    print("=" * 70)
    print("MINIMUM OPPONENT EXPOSURE EXPERIMENT")
    print("=" * 70)

    all_results = {}

    # --- Experiment 1: Diverse training ---
    print("\n--- DIVERSE training (different games each time) ---")
    diverse_results = run_incremental_experiment(mode="diverse", max_games=200, eval_games=500, seed=42)
    diverse_milestones = find_milestones(diverse_results)

    # --- Experiment 2: Memorization training ---
    print("\n--- MEMORIZE training (same 10 games repeated) ---")
    memorize_results = run_incremental_experiment(mode="memorize", max_games=200, eval_games=500, seed=42)
    memorize_milestones = find_milestones(memorize_results)

    # --- Analysis ---
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    print(f"\n📊 DIVERSE training:")
    print(f"   First 55%: {diverse_milestones['first_55_pct']} games")
    print(f"   Plateau:   {diverse_milestones['plateau_start']} games")
    print(f"   Peak WR:   {diverse_milestones['peak_win_rate']:.1%}")
    if diverse_milestones['click_moment']:
        c = diverse_milestones['click_moment']
        print(f"   Click:     {c['from_games']}→{c['to_games']} games "
              f"(+{c['win_rate_jump']:.1%}, tiles {c['from_tiles']}→{c['to_tiles']})")

    print(f"\n📊 MEMORIZE training:")
    print(f"   First 55%: {memorize_milestones['first_55_pct']} games")
    print(f"   Plateau:   {memorize_milestones['plateau_start']} games")
    print(f"   Peak WR:   {memorize_milestones['peak_win_rate']:.1%}")
    if memorize_milestones['click_moment']:
        c = memorize_milestones['click_moment']
        print(f"   Click:     {c['from_games']}→{c['to_games']} games "
              f"(+{c['win_rate_jump']:.1%}, tiles {c['from_tiles']}→{c['to_tiles']})")

    # --- Tile vs Visit correlation ---
    print(f"\n📈 Correlation analysis:")
    diverse_tiles = [r["num_tiles"] for r in diverse_results]
    diverse_visits = [r["total_visits"] for r in diverse_results]
    diverse_wrs = [r["evaluation"]["win_rate"] for r in diverse_results]
    print(f"   Tiles range: {min(diverse_tiles)} → {max(diverse_tiles)}")
    print(f"   Visits range: {min(diverse_visits)} → {max(diverse_visits)}")
    print(f"   WR range: {min(diverse_wrs):.1%} → {max(diverse_wrs):.1%}")

    # Does tile count or visit count matter more?
    # Simple check: when does tile growth slow down vs when does WR improve?
    tile_growth = []
    for i in range(1, len(diverse_results)):
        tile_growth.append(diverse_results[i]["num_tiles"] - diverse_results[i-1]["num_tiles"])
    if tile_growth:
        print(f"   Avg new tiles/10 games: {sum(tile_growth)/len(tile_growth):.0f}")
        print(f"   Early (10-50) avg: {sum(tile_growth[:4])/len(tile_growth[:4]):.0f} tiles/step")
        print(f"   Late (150-200) avg: {sum(tile_growth[-5:])/len(tile_growth[-5:]):.0f} tiles/step")

    # Save results
    output = {
        "experiment": "min_opponent_exposure",
        "description": "How few training games to reach 55% win rate on TTT?",
        "diverse": {
            "checkpoints": diverse_results,
            "milestones": diverse_milestones,
        },
        "memorize": {
            "checkpoints": memorize_results,
            "milestones": memorize_milestones,
        },
        "conclusions": {
            "min_games_for_55_diverse": diverse_milestones["first_55_pct"],
            "min_games_for_55_memorize": memorize_milestones["first_55_pct"],
            "diversity_advantage": (
                "diverse better"
                if (diverse_milestones["peak_win_rate"] > memorize_milestones["peak_win_rate"])
                else "memorize better or equal"
            ),
            "plateau_games_diverse": diverse_milestones["plateau_start"],
            "plateau_games_memorize": memorize_milestones["plateau_start"],
        },
    }

    output_path = os.path.join(os.path.dirname(__file__), "..", "min-exposure-results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Results saved to min-exposure-results.json")


if __name__ == "__main__":
    main()
