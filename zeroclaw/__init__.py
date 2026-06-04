"""
ZeroClaw Arena — learn games from scratch with tile-based Monte Carlo.

Usage:
    from zeroclaw import TicTacToe, TileField, run_arena

    # Quick experiment
    results = run_arena(games=["tictactoe"], mode="tile")

    # Manual training
    game = TicTacToe()
    field = TileField()
    field.train(game, num_games=500)

    # Compile to a zero-dependency policy
    from zeroclaw import CompiledPolicy
    policy = CompiledPolicy.from_tile_field(field)
    action = policy("X O  X   ")
"""

from zeroclaw.games import TicTacToe, Connect4, Go9x9, HoldemHand, GameState, Transition
from zeroclaw.tile_field import TileField
from zeroclaw.compiled_policy import CompiledPolicy
from zeroclaw.arena import run_arena

__all__ = [
    "TicTacToe",
    "Connect4",
    "Go9x9",
    "HoldemHand",
    "GameState",
    "Transition",
    "TileField",
    "CompiledPolicy",
    "run_arena",
]

__version__ = "0.1.0"
