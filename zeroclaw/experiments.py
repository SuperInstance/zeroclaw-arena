"""
Experiment runners — one-liner experiment functions.
"""

from zeroclaw.arena import run_arena


def quick_ttt(num_train: int = 200, num_eval: int = 500) -> dict:
    """Quick tic-tac-toe experiment."""
    return run_arena(
        games=["tictactoe"],
        mode="tile",
        num_train=num_train,
        num_eval=num_eval,
    )["tictactoe"]


def quick_connect4(num_train: int = 100, num_eval: int = 200) -> dict:
    """Quick Connect4 experiment."""
    return run_arena(
        games=["connect4"],
        mode="tile",
        num_train=num_train,
        num_eval=num_eval,
    )["connect4"]


def compare_games(num_train: int = 300, num_eval: int = 500) -> dict:
    """Compare tile field learning across all games."""
    return run_arena(
        mode="tile",
        num_train=num_train,
        num_eval=num_eval,
    )


def evolve_experiment(game: str = "tictactoe", generations: int = 5, games_per_gen: int = 100) -> dict:
    """Run evolution across multiple generations."""
    return run_arena(
        games=[game],
        mode="evolve",
        num_explore=games_per_gen,
        num_evolve=generations,
    )[game]


def baseline_random(games: list[str] | None = None, num_games: int = 1000) -> dict:
    """Establish random baselines for all games."""
    if games is None:
        games = ["tictactoe", "connect4"]
    return run_arena(
        games=games,
        mode="random",
        num_exploit=num_games,
    )
