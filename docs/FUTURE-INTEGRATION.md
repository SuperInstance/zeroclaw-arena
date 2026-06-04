# Future Integration: zeroclaw-arena

## Current State
An arena for learning games from scratch using tile-based Monte Carlo — no neural networks, pure algorithmic discovery. Provides TicTacToe and extensible game protocol. Compile policies to zero-dependency tables. `run_arena` for one-liner experiments.

## Integration Opportunities

### With room agent competitions
zeroclaw-arena becomes the testing ground where room agents compete. Each room sends its best strategy (compiled policy) to the arena. Strategies compete in games. Winners' rooms gain prestige; losers' rooms evolve new strategies. The arena IS the fleet's fitness test.

### With evolution-ternary
The arena provides the fitness evaluation for evolution-ternary's generational loops. Instead of abstract fitness functions, strategies compete in real games. Fitness = win rate. Natural selection = tournament results. The arena makes evolution concrete.

### With superinstance-spreadsheet
The spreadsheet visualizes arena results: each row is a strategy, each column is an opponent, each cell is the win rate. The arena runs competitions; the spreadsheet shows the results. Press "Evolve" to generate new strategies and compete again.

## Dormant Ideas Now Unlockable
The arena was for board games. Now it's for room strategies. A "game" is any competitive scenario: resource allocation, task scheduling, anomaly response. Strategies that win in the arena are deployed to rooms. The arena IS the fleet's testing environment.

## Potential in Mature Systems
Every new strategy is arena-tested before deployment. Rooms submit candidates; the arena runs tournaments; winners are promoted to production. This is the fleet's CI/CD for strategies: test before deploy, measure before promote.

## Cross-Pollination Ideas
- **tile-compiler**: Compile arena-winning strategies into fast lookup tables
- **evolution-ternary**: Arena provides the selection pressure
- **strategy-ecology**: Species compete in the arena

## Dependencies for Next Steps
- Define room-strategy game protocol (beyond board games)
- Integration with ternary-cell for strategy deployment
- Arena results → room strategy promotion pipeline
