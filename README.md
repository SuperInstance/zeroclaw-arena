# ZeroClaw Arena

Learn games from scratch with tile-based Monte Carlo. No neural networks — pure algorithmic discovery.

## Install

```bash
pip install -e .
```

## Quick Start

```python
from zeroclaw import TicTacToe, TileField, CompiledPolicy, run_arena

# One-liner experiment
results = run_arena(games=["tictactoe"], mode="tile", num_train=500, num_eval=1000)

# Or manual training
game = TicTacToe()
field = TileField(n_simulations=20, temperature=0.3)
field.train(game, num_games=500)

# Compile to zero-dependency policy
policy = CompiledPolicy.from_tile_field(field)
action = policy("X O  X   ")  # → "8"
eval_results = policy.evaluate(num_games=1000)
print(f"Win rate: {eval_results['win_rate']:.1%}")

# Export to standalone Python
source = policy.to_python()
with open("my_policy.py", "w") as f:
    f.write(source)
```

## Package Structure

```
zeroclaw/
├── __init__.py          # Public API: TicTacToe, Connect4, Go9x9, TileField, etc.
├── games.py             # Game implementations (TicTacToe, Connect4, Go9x9, HoldemHand)
├── tile_field.py        # TileField — Monte Carlo tile coding with softmax selection
├── compiled_policy.py   # CompiledPolicy — zero-dependency lookup table
├── arena.py             # run_arena — experiment runner (tile/evolve/explore/random modes)
└── experiments.py       # One-liner experiment functions

experiments/             # Research scripts (36 experiment files)
results/                 # Experiment results (38 JSON files)
tests/                   # Test suite
```

## Games

| Game | Description | State Space |
|------|-------------|-------------|
| `TicTacToe` | 3×3 grid, X/O | ~5,478 legal states |
| `Connect4` | 6×7 grid, drop pieces | ~4.5T legal states |
| `Go9x9` | 9×9 Go with Chinese scoring | ~10^38 |
| `HoldemHand` | Texas Hold'em, simplified | bucketed stages |

## Arena Modes

```python
# Full pipeline: train → compile → evaluate
run_arena(mode="tile")

# Evolution across generations
run_arena(mode="evolve", num_evolve=5)

# Random exploration baseline
run_arena(mode="random")

# Tile field play (no compilation)
run_arena(mode="exploit")
```

## How It Works

1. **Explore**: Play games using Monte Carlo simulation + softmax action selection
2. **Record**: Track (state, action) → win/loss for each tile
3. **Evolve**: Update tile scores from accumulated win rates
4. **Compile**: Extract best action per state into a deterministic lookup table
5. **Deploy**: The compiled policy has zero dependencies — just string matching

## Running Tests

```bash
python3 -m pytest tests/ -v
```

## Architecture

The same recursive self-improvement pattern at every layer:

```
Meta Layer → forks repos, learns from them
ZeroClaw Layer → forks games, learns from them
Game Layer → the world that grounds the learning
```

Zero dependencies for the core library. Pure Python + standard library only.
