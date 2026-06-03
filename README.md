# ZeroClaw Arena

ZeroClaw agents in sandboxes learn text-based games from scratch. They clone games, build vector DBs of state transitions, discover patterns algorithmically, and write automation scripts that get better over time.

This is the same pattern as the higher layers — us (Forgemaster/Oracle2) fork repos and learn from them, ZeroClaws fork games and learn from them. Recursive self-improvement at every abstraction.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  META LAYER (us)                         │
│  Forgemaster + Oracle2 fork repos, build ecosystem      │
│  open-mind induces any repo, metal-lathe tests hypothes │
└────────────────────────┬────────────────────────────────┘
                         │ same pattern
                         ▼
┌─────────────────────────────────────────────────────────┐
│               ZEROCLAW LAYER (agents)                    │
│  Each ZeroClaw:                                         │
│  1. Forks a text-based game                             │
│  2. Plays it (exploration)                              │
│  3. Records every (state, action, reward, next_state)   │
│  4. Builds vector DB of state transitions               │
│  5. Discovers patterns algorithmically                  │
│  6. Writes automation scripts                           │
│  7. Tests scripts against the game                      │
│  8. Keeps scripts that win, discards ones that lose     │
│  9. Repeats — scripts get better each cycle             │
└─────────────────────────────────────────────────────────┘
                         │ same pattern
                         ▼
┌─────────────────────────────────────────────────────────┐
│              GAME LAYER (the world)                      │
│  Chess, Blackjack, Tic-tac-toe, Connect4, Go (9x9)     │
│  Each game is a closed world with clear win/lose states │
│  The metal that grounds the learning                    │
└─────────────────────────────────────────────────────────┘
```

## The Loop (same as metal-lathe)

1. **EXPLORE**: Play random games, collect state transitions
2. **OBSERVE**: What states lead to wins? What patterns exist?
3. **PATTERN**: Algorithmically discover patterns in the vector DB
4. **SCRIPT**: Write a Python script that encodes the pattern
5. **TEST**: Run the script against the game 100 times
6. **EVALUATE**: Win rate > 50%? Keep it. Otherwise, discard.
7. **FEED**: Results become new observations. Loop.
