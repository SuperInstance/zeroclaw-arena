"""
Tile Capacity Scaling — How many tiles does a game ACTUALLY need?

Trains a tile field at full capacity, then incrementally prunes tiles
by visit count (and by score) to find the true performance cliff.

Key question: Is there a SHARP cliff where performance drops, or is it gradual?
"""

import sys
import os
import json
import random
import copy
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from zeroclaw.tile_field import TileField
from zeroclaw.games import TicTacToe, Connect4


# ── helpers ──────────────────────────────────────────────────────────────────

def tile_visits(tile_data: dict) -> int:
    """Total visits for a tile (sum of 'chosen' across all actions)."""
    return sum(a["chosen"] for a in tile_data.values())


def tile_avg_score(tile_data: dict) -> float:
    """Average learned score across actions."""
    if not tile_data:
        return 0.0
    return sum(a["score"] for a in tile_data.values()) / len(tile_data)


def prune_by_visit_threshold(field: TileField, min_visits: int) -> TileField:
    """Return a copy of the field with tiles whose total visits < min_visits removed."""
    pruned = copy.deepcopy(field)
    to_remove = [sid for sid, td in pruned.tiles.items() if tile_visits(td) < min_visits]
    for sid in to_remove:
        del pruned.tiles[sid]
    return pruned


def prune_keep_top_n_by_visits(field: TileField, n: int) -> TileField:
    """Keep only the top-n tiles by visit count."""
    pruned = copy.deepcopy(field)
    sorted_tiles = sorted(pruned.tiles.items(), key=lambda x: tile_visits(x[1]), reverse=True)
    keep = set(sid for sid, _ in sorted_tiles[:n])
    for sid in list(pruned.tiles.keys()):
        if sid not in keep:
            del pruned.tiles[sid]
    return pruned


def prune_keep_top_n_by_score(field: TileField, n: int) -> TileField:
    """Keep only the top-n tiles by average learned score."""
    pruned = copy.deepcopy(field)
    sorted_tiles = sorted(pruned.tiles.items(), key=lambda x: tile_avg_score(x[1]), reverse=True)
    keep = set(sid for sid, _ in sorted_tiles[:n])
    for sid in list(pruned.tiles.keys()):
        if sid not in keep:
            del pruned.tiles[sid]
    return pruned


def random_player_action(game):
    """Choose a random legal action."""
    actions = game.legal_actions()
    return random.choice(actions) if actions else ''


def evaluate_vs_random(game_class, field: TileField, n_games: int = 200) -> dict:
    """Evaluate tile field playing as X (P1) vs random O. Returns win/draw/loss."""
    wins = draws = losses = 0
    for _ in range(n_games):
        game = game_class()
        game.reset()
        # Alternate who goes first to be fair, but field always plays as X
        while not game.done:
            state = game.state()
            actions = game.legal_actions()
            if not actions:
                break

            if game.current == 'X':
                # Tile field player
                state_str = str(state.state_str)
                if state_str in field.tiles:
                    action = field.choose_action(game, state_str, actions)
                else:
                    action = random.choice(actions)
            else:
                action = random.choice(actions)

            game.step(action)

        if game.winner == 'X':
            wins += 1
        elif game.winner is None or game.winner == 'draw':
            draws += 1
        else:
            losses += 1

    return {"wins": wins, "draws": draws, "losses": losses,
            "win_rate": wins / n_games, "total": n_games}


def train_field(game_class, n_games: int = 1000) -> TileField:
    """Train a tile field for n_games."""
    field = TileField(n_simulations=20, temperature=0.3)
    game = game_class()
    field.train(game, num_games=n_games, evolve_every=25)
    return field


# ── main experiment ──────────────────────────────────────────────────────────

PRUNE_THRESHOLDS = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500]

GAMES_CONFIG = {
    "TTT": {"class": TicTacToe, "train_games": 1000, "eval_games": 200},
    "Connect4": {"class": Connect4, "train_games": 1000, "eval_games": 200},
}


def run_capacity_experiment(game_name: str, config: dict) -> dict:
    """Run full capacity experiment for one game."""
    game_class = config["class"]
    train_n = config["train_games"]
    eval_n = config["eval_games"]

    print(f"\n{'='*60}")
    print(f"  {game_name} — Tile Capacity Experiment")
    print(f"{'='*60}")

    # 1. Train full field
    print(f"\n[1] Training on {train_n} games...")
    t0 = time.time()
    field = train_field(game_class, train_n)
    train_time = time.time() - t0
    full_tile_count = field.size
    print(f"    Full field: {full_tile_count} tiles ({train_time:.1f}s)")

    # Count total visits
    total_visits = sum(tile_visits(td) for td in field.tiles.values())
    avg_visits = total_visits / max(full_tile_count, 1)
    print(f"    Total visits: {total_visits}, avg visits/tile: {avg_visits:.1f}")

    # 2. Baseline evaluation (full field)
    print(f"\n[2] Baseline eval ({eval_n} games vs random)...")
    baseline = evaluate_vs_random(game_class, field, eval_n)
    print(f"    Win rate: {baseline['win_rate']:.1%} "
          f"(W={baseline['wins']} D={baseline['draws']} L={baseline['losses']})")

    results = {
        "game": game_name,
        "train_games": train_n,
        "full_tiles": full_tile_count,
        "total_visits": total_visits,
        "train_time_s": round(train_time, 2),
        "baseline": baseline,
        "prune_by_visits": [],
        "prune_by_score": [],
    }

    # 3. Prune by visit threshold
    print(f"\n[3] Pruning by visit threshold...")
    for min_visits in PRUNE_THRESHOLDS:
        pruned = prune_by_visit_threshold(field, min_visits)
        remaining = pruned.size
        if remaining == 0:
            print(f"    min_visits={min_visits:>4d} → 0 tiles (skipped)")
            continue
        eval_result = evaluate_vs_random(game_class, pruned, eval_n)
        pct_of_full = remaining / full_tile_count * 100
        print(f"    min_visits={min_visits:>4d} → {remaining:>5d} tiles ({pct_of_full:5.1f}%) "
              f"win={eval_result['win_rate']:.1%}")
        results["prune_by_visits"].append({
            "min_visits": min_visits,
            "tiles_remaining": remaining,
            "pct_of_full": round(pct_of_full, 2),
            **eval_result,
        })

    # 4. Prune by score — keep top N tiles by score vs by visits
    print(f"\n[4] Pruning by keeping top-N tiles (by visits vs by score)...")

    # Determine interesting N values based on the visit-prune curve
    visit_tiles = [r["tiles_remaining"] for r in results["prune_by_visits"]]
    interesting_n = sorted(set([1, 2, 3, 5, 10, 20, 50, 100, 200, 500] + visit_tiles))
    # Cap at full capacity
    interesting_n = [n for n in interesting_n if 0 < n <= full_tile_count]
    # Sample at most 15 points
    if len(interesting_n) > 15:
        step = len(interesting_n) // 15
        interesting_n = interesting_n[::step]
        if interesting_n[-1] != full_tile_count:
            interesting_n.append(full_tile_count)

    for n in interesting_n:
        # Top N by visits
        pruned_v = prune_keep_top_n_by_visits(field, n)
        ev_v = evaluate_vs_random(game_class, pruned_v, eval_n)

        # Top N by score
        pruned_s = prune_keep_top_n_by_score(field, n)
        ev_s = evaluate_vs_random(game_class, pruned_s, eval_n)

        print(f"    top-{n:>4d} by visits: {ev_v['win_rate']:.1%}  |  "
              f"top-{n:>4d} by score: {ev_s['win_rate']:.1%}")

        results["prune_by_score"].append({
            "keep_n": n,
            "pct_of_full": round(n / full_tile_count * 100, 2),
            "by_visits": {"tiles_remaining": pruned_v.size, **ev_v},
            "by_score": {"tiles_remaining": pruned_s.size, **ev_s},
        })

    return results


def plot_results(results: dict, out_dir: str):
    """Generate plots for one game's results."""
    game = results["game"]
    full = results["full_tiles"]

    # ── Plot 1: Visit threshold pruning ──
    fig, ax = plt.subplots(figsize=(10, 6))
    pv = results["prune_by_visits"]
    if pv:
        tiles = [r["tiles_remaining"] for r in pv]
        win_rates = [r["win_rate"] for r in pv]
        ax.plot(tiles, win_rates, 'bo-', linewidth=2, markersize=8, label='Prune by visits')

    ax.axhline(y=results["baseline"]["win_rate"], color='g', linestyle='--',
               alpha=0.7, label=f'Full field ({full} tiles)')
    ax.set_xlabel('Tiles Remaining')
    ax.set_ylabel('Win Rate vs Random')
    ax.set_title(f'{game}: Tile Capacity — Prune by Visit Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"tile_capacity_{game.lower()}_visits.png"), dpi=150)
    plt.close(fig)

    # ── Plot 2: Top-N by visits vs score ──
    fig, ax = plt.subplots(figsize=(10, 6))
    ps = results["prune_by_score"]
    if ps:
        n_vals = [r["keep_n"] for r in ps]
        wr_visits = [r["by_visits"]["win_rate"] for r in ps]
        wr_score = [r["by_score"]["win_rate"] for r in ps]
        ax.plot(n_vals, wr_visits, 'rs-', linewidth=2, markersize=6, label='Top-N by visits')
        ax.plot(n_vals, wr_score, 'm^-', linewidth=2, markersize=6, label='Top-N by score')

    ax.axhline(y=results["baseline"]["win_rate"], color='g', linestyle='--',
               alpha=0.7, label=f'Full field ({full} tiles)')
    ax.set_xlabel('Tiles Kept (N)')
    ax.set_ylabel('Win Rate vs Random')
    ax.set_title(f'{game}: Top-N Tiles — Visits vs Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    ax.set_xscale('log')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"tile_capacity_{game.lower()}_topn.png"), dpi=150)
    plt.close(fig)


def analyze_cliff(results: dict) -> str:
    """Analyze whether there's a sharp cliff or gradual decline."""
    lines = []
    game = results["game"]
    lines.append(f"\n{'='*60}")
    lines.append(f"  {game} — Cliff Analysis")
    lines.append(f"{'='*60}")

    pv = results["prune_by_visits"]
    if len(pv) < 2:
        lines.append("  Not enough data points for cliff analysis.")
        return '\n'.join(lines)

    # Find biggest drop in win rate between consecutive points
    max_drop = 0
    max_drop_idx = 0
    for i in range(1, len(pv)):
        drop = pv[i-1]["win_rate"] - pv[i]["win_rate"]
        if drop > max_drop:
            max_drop = drop
            max_drop_idx = i

    if max_drop >= 0.15:
        lines.append(f"  🔴 SHARP CLIFF detected!")
        lines.append(f"     Drop of {max_drop:.1%} between "
                     f"{pv[max_drop_idx-1]['tiles_remaining']} tiles "
                     f"({pv[max_drop_idx-1]['win_rate']:.1%}) → "
                     f"{pv[max_drop_idx]['tiles_remaining']} tiles "
                     f"({pv[max_drop_idx]['win_rate']:.1%})")
        lines.append(f"     True capacity: ~{pv[max_drop_idx-1]['tiles_remaining']} tiles")
    elif max_drop >= 0.05:
        lines.append(f"  🟡 MODERATE cliff (drop={max_drop:.1%})")
        lines.append(f"     Between {pv[max_drop_idx-1]['tiles_remaining']} → "
                     f"{pv[max_drop_idx]['tiles_remaining']} tiles")
    else:
        lines.append(f"  🟢 GRADUAL decline (max drop={max_drop:.1%})")
        lines.append(f"     Performance degrades smoothly — even few tiles carry signal")

    # What's the minimum tiles for >50% win rate?
    for r in reversed(pv):
        if r["win_rate"] > 0.5:
            lines.append(f"  Minimum for >50% win rate: {r['tiles_remaining']} tiles "
                         f"(win={r['win_rate']:.1%})")
            break
    else:
        lines.append(f"  Even 0 tiles could beat random... (baseline randomness)")

    # Top-N analysis
    ps = results["prune_by_score"]
    if ps:
        best_visits = max(ps, key=lambda r: r["by_visits"]["win_rate"])
        best_score = max(ps, key=lambda r: r["by_score"]["win_rate"])
        lines.append(f"\n  Top-N by visits best: {best_visits['keep_n']} tiles → "
                     f"{best_visits['by_visits']['win_rate']:.1%}")
        lines.append(f"  Top-N by score best:  {best_score['keep_n']} tiles → "
                     f"{best_score['by_score']['win_rate']:.1%}")

        # Does score-based selection outperform visit-based?
        score_wins = sum(1 for r in ps if r["by_score"]["win_rate"] > r["by_visits"]["win_rate"])
        visit_wins = sum(1 for r in ps if r["by_visits"]["win_rate"] > r["by_score"]["win_rate"])
        lines.append(f"\n  Score > Visits: {score_wins}/{len(ps)} points")
        lines.append(f"  Visits > Score: {visit_wins}/{len(ps)} points")
        if score_wins > visit_wins:
            lines.append(f"  → Score-based selection is BETTER at this game size")
        else:
            lines.append(f"  → Visit-based selection is BETTER (experience matters)")

    return '\n'.join(lines)


def main():
    os.chdir(os.path.dirname(__file__) or '.')
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}
    all_analysis = []

    for game_name, config in GAMES_CONFIG.items():
        result = run_capacity_experiment(game_name, config)
        all_results[game_name] = result
        analysis = analyze_cliff(result)
        all_analysis.append(analysis)
        print(analysis)
        plot_results(result, out_dir)

    # Save combined results
    out_path = os.path.join(os.path.dirname(__file__), '..', 'tile-capacity-results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for game_name, result in all_results.items():
        full = result["full_tiles"]
        baseline_wr = result["baseline"]["win_rate"]
        pv = result["prune_by_visits"]
        min_tiles = "N/A"
        for r in reversed(pv):
            if r["win_rate"] > baseline_wr * 0.9:  # within 90% of baseline
                min_tiles = r["tiles_remaining"]
                break
        print(f"  {game_name}: {full} tiles trained, baseline win={baseline_wr:.1%}, "
              f"essential tiles (90% perf): {min_tiles}")


if __name__ == "__main__":
    main()
