"""
Arena — run ZeroClaw game-learning experiments.

Supports multiple modes:
- explore: random play to collect transitions
- evolve: tile field learning with Monte Carlo simulation
- exploit: play with a compiled policy
- tile: full train → compile → evaluate pipeline
- random: baseline random play
"""

import json
import random
import time
from pathlib import Path

from zeroclaw.games import TicTacToe, Connect4, Go9x9, HoldemHand
from zeroclaw.tile_field import TileField
from zeroclaw.compiled_policy import CompiledPolicy


GAME_REGISTRY = {
    "tictactoe": TicTacToe,
    "connect4": Connect4,
    "go9x9": Go9x9,
    "holdem": HoldemHand,
}


def run_arena(
    games: list[str] | None = None,
    mode: str = "tile",
    num_explore: int = 50,
    num_evolve: int = 3,
    num_exploit: int = 50,
    num_train: int = 500,
    num_eval: int = 1000,
    output_dir: str | None = None,
) -> dict:
    """Run ZeroClaw arena experiments.

    Args:
        games: list of game names (default: all)
        mode: one of 'explore', 'evolve', 'exploit', 'tile', 'random'
        num_explore: games per exploration phase (explore/evolve modes)
        num_evolve: number of evolution generations (evolve mode)
        num_exploit: games in exploit phase (exploit mode)
        num_train: games to train tile field (tile mode)
        num_eval: games to evaluate compiled policy (tile mode)
        output_dir: directory to save results (default: ./results)

    Returns:
        dict of results per game
    """
    if games is None:
        games = list(GAME_REGISTRY.keys())
    if output_dir is None:
        output_dir = "./results"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = {}
    for game_name in games:
        if game_name not in GAME_REGISTRY:
            print(f"  Unknown game: {game_name}, skipping")
            continue

        print(f"\n{'=' * 50}")
        print(f"  {game_name} — mode={mode}")
        print(f"{'=' * 50}")

        game_cls = GAME_REGISTRY[game_name]
        game = game_cls()

        if mode == "tile":
            result = _run_tile_mode(game, game_name, num_train, num_eval)
        elif mode == "explore":
            result = _run_explore(game, game_name, num_explore)
        elif mode == "evolve":
            result = _run_evolve(game, game_name, num_explore, num_evolve)
        elif mode == "exploit":
            result = _run_exploit(game, game_name, num_exploit)
        elif mode == "random":
            result = _run_random(game, game_name, num_exploit)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        results[game_name] = result

        # Save results
        out_path = Path(output_dir) / f"{game_name}-{mode}-results.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  Results saved to {out_path}")

    return results


def _run_tile_mode(game, game_name: str, num_train: int, num_eval: int) -> dict:
    """Full pipeline: train tile field → compile → evaluate."""
    t0 = time.time()

    # Train
    print(f"  Training tile field ({num_train} games)...")
    field = TileField(n_simulations=20, temperature=0.3)
    wins = field.train(game, num_games=num_train)

    # Compile
    print(f"  Compiling policy ({field.size} tiles)...")
    policy = CompiledPolicy.from_tile_field(field)

    # Evaluate
    print(f"  Evaluating ({num_eval} games)...")
    if game_name == "tictactoe":
        eval_results = policy.evaluate(num_games=num_eval)
    else:
        eval_results = _evaluate_generic(game, policy, num_eval)

    elapsed = time.time() - t0

    # Optionally save compiled policy
    policy_path = Path("./results") / f"{game_name}-compiled-policy.py"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    source = policy.to_python()
    with open(policy_path, "w") as f:
        f.write(source)
    print(f"  Compiled policy saved to {policy_path}")

    return {
        "mode": "tile",
        "game": game_name,
        "train_games": num_train,
        "tiles_learned": field.size,
        "compiled_states": policy.size,
        "evaluation": eval_results,
        "elapsed_seconds": round(elapsed, 2),
    }


def _run_explore(game, game_name: str, num_games: int) -> dict:
    """Random exploration baseline."""
    wins = 0
    draws = 0
    losses = 0
    t0 = time.time()

    for i in range(num_games):
        game.reset()
        while not game.done:
            actions = game.legal_actions()
            if not actions:
                break
            game.step(random.choice(actions))

        w = getattr(game, 'winner', None)
        if w in ('X', 'B', 'player', 0):
            wins += 1
        elif w in ('draw', None) or w == 'draw':
            draws += 1
        else:
            losses += 1

    return {
        "mode": "explore",
        "game": game_name,
        "total_games": num_games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / num_games,
        "elapsed_seconds": round(time.time() - t0, 2),
    }


def _run_evolve(game, game_name: str, num_explore: int, num_evolve: int) -> dict:
    """Tile field evolution across generations."""
    field = TileField()
    history = []

    for gen in range(1, num_evolve + 1):
        wins = field.train(game, num_games=num_explore)
        field.evolve()

        # Quick eval
        w, d, l = 0, 0, 0
        for _ in range(100):
            game.reset()
            while not game.done:
                actions = game.legal_actions()
                if not actions:
                    break
                state_str = str(game.state().state_str)
                if hasattr(game, 'current') and game.current in ('X', 'B'):
                    action = field.choose_action(game, state_str, actions)
                else:
                    action = random.choice(actions)
                game.step(action)
            winner = getattr(game, 'winner', None)
            if winner in ('X', 'B', 'player', 0):
                w += 1
            elif winner in ('draw', None):
                d += 1
            else:
                l += 1

        history.append({
            "generation": gen,
            "tiles": field.size,
            "wins": w, "draws": d, "losses": l,
            "win_rate": w / 100,
        })
        print(f"  Gen {gen}: tiles={field.size} win_rate={w / 100:.1%}")

    return {
        "mode": "evolve",
        "game": game_name,
        "generations": num_evolve,
        "final_tiles": field.size,
        "history": history,
    }


def _run_exploit(game, game_name: str, num_games: int) -> dict:
    """Play with tile field (no compilation)."""
    field = TileField()
    field.train(game, num_games=num_games // 2)

    wins = 0
    for _ in range(num_games):
        game.reset()
        while not game.done:
            actions = game.legal_actions()
            if not actions:
                break
            state_str = str(game.state().state_str)
            if hasattr(game, 'current') and game.current in ('X', 'B'):
                action = field.choose_action(game, state_str, actions)
            else:
                action = random.choice(actions)
            game.step(action)
        if getattr(game, 'winner', None) in ('X', 'B', 'player', 0):
            wins += 1

    return {
        "mode": "exploit",
        "game": game_name,
        "total_games": num_games,
        "wins": wins,
        "win_rate": wins / num_games,
    }


def _run_random(game, game_name: str, num_games: int) -> dict:
    """Pure random baseline."""
    return _run_explore(game, game_name, num_games)


def _evaluate_generic(game, policy, num_games: int) -> dict:
    """Evaluate a compiled policy on any game."""
    wins = 0
    draws = 0
    losses = 0

    for _ in range(num_games):
        game.reset()
        while not game.done:
            actions = game.legal_actions()
            if not actions:
                break
            if hasattr(game, 'current') and game.current in ('X', 'B'):
                state_str = str(game.state().state_str)
                action = policy(state_str)
                if action not in actions:
                    action = random.choice(actions)
            else:
                action = random.choice(actions)
            game.step(action)

        w = getattr(game, 'winner', None)
        if w in ('X', 'B', 'player', 0):
            wins += 1
        elif w in ('draw', None) or w == 'draw':
            draws += 1
        else:
            losses += 1

    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / num_games,
        "total_games": num_games,
    }
