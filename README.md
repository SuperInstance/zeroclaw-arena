# ZeroClaw Arena — Learn Games from Scratch with Tile-Based Monte Carlo

**ZeroClaw Arena** is a game-learning framework that uses tile-based Monte Carlo self-play to build optimal policies — no neural networks required. It includes built-in games (Tic-Tac-Toe, Connect 4, Go 9×9, Texas Hold'em), a tile field training engine, compiled policy generation, and an experiment arena for running tournaments and comparative studies.

## Why It Matters

AlphaGo and AlphaZero demonstrated that self-play can produce superhuman game-playing AI, but they require massive neural networks and days of GPU training. ZeroClaw Arena proves that for many games — especially those with bounded state spaces — tile-based Monte Carlo achieves strong play with zero neural network dependencies, minutes of training time, and complete interpretability (every decision traces to a lookup table entry). The tile decomposition enables generalization: patterns learned in one part of the game state transfer to analogous positions. This makes tile-based learning ideal for fleet strategy optimization where interpretability and rapid training matter more than handling continuous state spaces.

## How It Works

### Tile Field

The `TileField` decomposes game states into local patterns. For Tic-Tac-Toe, tiles include individual cells, rows, columns, and diagonals. Each tile configuration accumulates statistics from Monte Carlo simulation:

```
tile_value(config) = win_rate(config) over simulation games
```

Training plays `num_games` self-play games. Each game:
1. At each state, extract tiles and look up values
2. Choose moves proportional to tile value (with exploration ε)
3. After game ends, update all visited tiles with the outcome

Training is O(g × s × t) for g games, s states per game, t tiles per state.

### Compiled Policy

After training, `CompiledPolicy.from_tile_field(field)` creates a lookup table:

```
state_hash → best_action
```

Runtime evaluation is O(1) — hash the state, look up the action. Zero neural network inference. Zero matrix multiplication. Zero dependencies.

### Arena Experiments

`run_arena()` runs tournaments between policies:

```python
results = run_arena(
    games=["tictactoe", "connect4"],
    mode="tile",
    num_rounds=100,
)
```

Results include win rates, move quality analysis, and convergence statistics.

### Game Protocol

The `Game` protocol makes ZeroClaw extensible:

- `legal_moves(state) → List[move]`
- `apply(state, move) → new_state`
- `is_terminal(state) → bool`
- `outcome(state) → Optional[score]`

Any game implementing this protocol can be trained and compiled.

## Quick Start

```python
from zeroclaw import TicTacToe, TileField, CompiledPolicy, run_arena

# Quick experiment
results = run_arena(games=["tictactoe"], mode="tile")
print(f"Win rate: {results['tictactoe']['win_rate']:.1%}")

# Manual training
game = TicTacToe()
field = TileField()
field.train(game, num_games=500)

# Compile and use
policy = CompiledPolicy.from_tile_field(field)
action = policy("X O  X   ")
```

```bash
pip install zeroclaw
```

## API

| Type / Function | Description |
|---|---|
| `TileField` | Trainable tile-based value estimator |
| `CompiledPolicy` | Hash → action lookup (O(1) runtime) |
| `run_arena(games, mode)` | Tournament runner with results |
| `TicTacToe` | Built-in 3×3 game |
| `Connect4` | Built-in 7×6 game |
| `Go9x9` | Built-in 9×9 Go |
| `HoldemHand` | Built-in poker hand evaluator |
| `GameState`, `Transition` | Protocol types for custom games |

## Architecture Notes

ZeroClaw Arena is the game-theory research environment in **SuperInstance**. It demonstrates that ternary decision-making (win/loss/draw in Z₃) can be learned without neural networks. The γ + η = C conservation manifests in the training dynamics: exploration (γ = information gain) must balance with exploitation (η = known-value reinforcement). The tile field's convergence to optimal play is the fleet-level analog of reaching the γ-η equilibrium. See [Architecture](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md).

## References:

- Sutton, Richard & Barto, Andrew. *Reinforcement Learning*, 2nd ed., MIT Press, 2018 — Monte Carlo methods.
| Silver, David et al. "Mastering the Game of Go without Human Knowledge," *Nature*, 550, 2017 — self-play learning.
| Schaeffer, Jonathan. *One Jump Ahead*, Springer, 2009 — game-solving methodology.

## License

MIT
