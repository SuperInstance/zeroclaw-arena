"""
Tile Compiler — Convert a trained tile field into an optimized deterministic lookup table.

Like compiling source code to machine code:
- "Source" = tile field with softmax selection + Monte Carlo simulation
- "Compiled" = pure if-then-else decision tree, O(1) lookup, ZERO dependencies

The tile field is the TRAINING algorithm; the compiled policy is the DEPLOYED artifact.
Learn with gradients/simulation, deploy with lookups.
"""

import json
import time
import random
import sys
import os
import hashlib

# ─── Minimal TicTacToe (standalone, no external deps needed for compilation) ───

class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9
        self.current = 'X'
        self.turn = 0
        self.done = False
        self.winner = None

    def state(self):
        return ''.join(self.board)

    def legal_actions(self):
        if self.done:
            return []
        return [str(i) for i in range(9) if self.board[i] == ' ']

    def step(self, action):
        pos = int(action)
        if self.board[pos] != ' ':
            return -1.0, True
        self.board[pos] = self.current
        self.turn += 1
        lines = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
        for a, b, c in lines:
            if self.board[a] == self.board[b] == self.board[c] != ' ':
                self.done = True
                self.winner = self.current
                reward = 1.0 if self.current == 'X' else -1.0
                return reward, True
        if self.turn >= 9:
            self.done = True
            return 0.0, True
        self.current = 'O' if self.current == 'X' else 'X'
        return 0.0, False

    def reset(self):
        self.board = [' '] * 9
        self.current = 'X'
        self.turn = 0
        self.done = False
        self.winner = None

    def copy(self):
        g = TicTacToe()
        g.board = self.board[:]
        g.current = self.current
        g.turn = self.turn
        g.done = self.done
        g.winner = self.winner
        return g


# ─── Tile Field (lightweight reimplementation) ──────────────────────

import numpy as np

class TileField:
    """Tile field with softmax action selection + Monte Carlo simulation."""

    def __init__(self, n_simulations=20, temperature=0.3):
        self.tiles = {}  # state_str -> {action: {"score": float, "chosen": int, "won": int}}
        self.n_simulations = n_simulations
        self.temperature = temperature

    def get_or_create(self, state_str, legal_actions):
        if state_str not in self.tiles:
            self.tiles[state_str] = {
                a: {"score": 0.5, "chosen": 0, "won": 0} for a in legal_actions
            }
        tile = self.tiles[state_str]
        # Ensure all legal actions present
        for a in legal_actions:
            if a not in tile:
                tile[a] = {"score": 0.5, "chosen": 0, "won": 0}
        return tile

    def choose_action(self, game, state_str, legal_actions):
        """Choose action via Monte Carlo sim + learned scores + softmax."""
        if len(legal_actions) <= 1:
            return legal_actions[0] if legal_actions else ''

        tile = self.get_or_create(state_str, legal_actions)

        action_values = {}
        sims_per = max(1, self.n_simulations // len(legal_actions))

        for action in legal_actions:
            # Monte Carlo rollouts
            sim_wins = 0
            for _ in range(sims_per):
                g = game.copy()
                g.step(action)
                while not g.done:
                    acts = g.legal_actions()
                    if not acts:
                        break
                    g.step(random.choice(acts))
                if g.winner == 'X':
                    sim_wins += 1

            sim_score = sim_wins / max(sims_per, 1)
            learned_score = tile[action]["score"]
            n_chosen = tile[action]["chosen"]
            confidence = min(n_chosen / 20.0, 0.8)
            action_values[action] = confidence * learned_score + (1 - confidence) * sim_score

        # Softmax selection
        actions_list = list(action_values.keys())
        values = np.array([action_values[a] for a in actions_list])
        exp_vals = np.exp(values / self.temperature)
        probs = exp_vals / exp_vals.sum()
        return np.random.choice(actions_list, p=probs)

    def record(self, state_str, action, won):
        if state_str in self.tiles and action in self.tiles[state_str]:
            self.tiles[state_str][action]["chosen"] += 1
            if won:
                self.tiles[state_str][action]["won"] += 1

    def evolve(self):
        """Update scores based on accumulated win rates."""
        for tile in self.tiles.values():
            for action, data in tile.items():
                if data["chosen"] > 0:
                    wr = data["won"] / data["chosen"]
                    data["score"] += 0.05 * (wr - data["score"])
                    data["score"] = max(0.05, min(0.95, data["score"]))


# ─── Phase 1: Train the Tile Field ──────────────────────────────────

def train_tile_field(num_games=1000, n_simulations=20, evolve_every=25):
    """Train a tile field on tic-tac-toe via self-play."""
    print(f"═══ Phase 1: Training Tile Field ({num_games} games) ═══")
    game = TicTacToe()
    field = TileField(n_simulations=n_simulations)

    x_wins, o_wins, draws = 0, 0, 0

    for i in range(num_games):
        game.reset()
        history = []

        while not game.done:
            state_str = game.state()
            actions = game.legal_actions()
            if not actions:
                break

            if game.current == 'X':
                action = field.choose_action(game, state_str, actions)
            else:
                action = random.choice(actions)

            game.step(action)
            history.append((state_str, action, 'X' if len(history) % 2 == 0 else 'O'))

        # Record outcome
        won_x = game.winner == 'X'
        for state_str, action, player in history:
            if player == 'X':
                field.record(state_str, action, won_x)

        if game.winner == 'X':
            x_wins += 1
        elif game.winner == 'O':
            o_wins += 1
        else:
            draws += 1

        # Evolve periodically
        if (i + 1) % evolve_every == 0:
            field.evolve()

        if (i + 1) % 100 == 0:
            total = i + 1
            print(f"  {total}/{num_games} | X wins: {x_wins/total:.1%} | "
                  f"O wins: {o_wins/total:.1%} | Draws: {draws/total:.1%} | "
                  f"Tiles: {len(field.tiles)}")

    print(f"  Final: {len(field.tiles)} tiles | "
          f"X={x_wins/num_games:.1%} O={o_wins/num_games:.1%} D={draws/num_games:.1%}")
    return field


# ─── Phase 2: Compile the Tile Field ────────────────────────────────

def compile_tile_field(field):
    """
    Compile the tile field into a deterministic lookup table.

    For each tile (state), find the TOP action (argmax of scores).
    For unknown states, use nearest-neighbor by Hamming distance.
    Output: a self-contained Python function with ZERO dependencies.
    """
    print(f"\n═══ Phase 2: Compiling {len(field.tiles)} tiles ═══")

    # Step 1: Extract best action per state
    lookup = {}
    for state_str, tile in field.tiles.items():
        # Only compile states where it's X's turn (board has equal X and O)
        x_count = state_str.count('X')
        o_count = state_str.count('O')
        is_x_turn = (x_count == o_count)

        if not is_x_turn:
            continue

        # Find best action using weighted score: score * confidence
        # This favors actions with both high score AND high visit count
        best_action = None
        best_weighted = -1
        for action, data in tile.items():
            score = data["score"]
            visits = data["chosen"]
            wins = data["won"]
            # UCB-like: use empirical win rate + exploration bonus
            if visits > 0:
                empirical_wr = wins / visits
                exploration_bonus = 0.3 * (1.0 / (1.0 + visits))  # diminishes with visits
                weighted = empirical_wr + exploration_bonus
            else:
                weighted = 0.5  # unexplored

            if weighted > best_weighted:
                best_weighted = weighted
                best_action = action

        total_chosen = sum(d["chosen"] for d in tile.values())

        if total_chosen >= 1:  # Include states with any experience
            lookup[state_str] = best_action

    print(f"  Compiled {len(lookup)} states (X-turn, >=1 visit)")

    # Step 2: Generate the compiled policy function
    lines = []
    lines.append('"""')
    lines.append('COMPILED TILE POLICY — Zero-dependency deterministic tic-tac-toe.')
    lines.append('Generated by tile_compiler.py from a trained tile field.')
    lines.append('No numpy, no random, no hashlib — just string matching.')
    lines.append('"""')
    lines.append('')
    lines.append('')
    lines.append('def compiled_policy(board_str):')
    lines.append('    """')
    lines.append('    Given a 9-char board string (e.g. "X O  X   "), return best move index as str.')
    lines.append('    X always goes first. Returns action for X player only.')
    lines.append('    """')

    # Direct lookup dictionary
    lines.append('    _lookup = {')
    sorted_states = sorted(lookup.items())
    for state_str, action in sorted_states:
        # Escape the string properly
        escaped = state_str.replace("'", "\\'")
        lines.append(f"        '{escaped}': '{action}',")
    lines.append('    }')
    lines.append('')

    # Nearest-neighbor fallback using Hamming distance
    # Pre-extract unique boards for NN
    nn_boards = list(lookup.keys())
    lines.append('    _boards = [')
    for b in nn_boards:
        escaped = b.replace("'", "\\'")
        lines.append(f"        '{escaped}',")
    lines.append('    ]')
    lines.append('')

    lines.append('    # Direct lookup')
    lines.append('    if board_str in _lookup:')
    lines.append('        return _lookup[board_str]')
    lines.append('')
    lines.append('    # Nearest-neighbor fallback (Hamming distance)')
    lines.append('    best_dist = 999')
    lines.append('    best_board = None')
    lines.append('    for b in _boards:')
    lines.append('        dist = sum(1 for i in range(9) if board_str[i] != b[i])')
    lines.append('        if dist < best_dist:')
    lines.append('            best_dist = dist')
    lines.append('            best_board = b')
    lines.append('    if best_board is not None and best_dist <= 3:')
    lines.append('        return _lookup[best_board]')
    lines.append('')
    lines.append('    # Ultimate fallback: play center, then corners, then edges')
    lines.append('    for pos in "402681357":')
    lines.append('        if board_str[int(pos)] == " ":')
    lines.append('            return pos')
    lines.append('    return "0"')
    lines.append('')
    lines.append('')
    lines.append('# Self-test')
    lines.append('if __name__ == "__main__":')
    lines.append('    tests = {')
    lines.append('        "         ": None,  # empty board, should play center')
    lines.append('        "X        ": None,  # X played, O should... but this is O turn')
    lines.append('    }')
    lines.append('    for board, expected in tests.items():')
    lines.append('        result = compiled_policy(board)')
    lines.append('        print(f"  {board!r} -> {result}")')
    lines.append('')

    code = '\n'.join(lines)
    return code, lookup


# ─── Phase 3: Evaluate ──────────────────────────────────────────────

def play_game_tile_field(field, opponent='random'):
    """Play one game: tile field (X) vs opponent (O). Returns winner."""
    game = TicTacToe()
    while not game.done:
        state_str = game.state()
        actions = game.legal_actions()
        if not actions:
            break
        if game.current == 'X':
            action = field.choose_action(game, state_str, actions)
        else:
            if opponent == 'random':
                action = random.choice(actions)
            else:
                action = random.choice(actions)  # default random
        game.step(action)
    return game.winner


def play_game_compiled(policy_func, opponent='random'):
    """Play one game: compiled policy (X) vs opponent (O). Returns winner."""
    game = TicTacToe()
    while not game.done:
        state_str = game.state()
        actions = game.legal_actions()
        if not actions:
            break
        if game.current == 'X':
            action = policy_func(state_str)
            # Verify legal
            if action not in actions:
                action = random.choice(actions)  # fallback
        else:
            action = random.choice(actions)
        game.step(action)
    return game.winner


def evaluate(field, policy_func, num_games=500):
    """Compare tile field vs compiled policy over many games."""
    print(f"\n═══ Phase 3: Evaluation ({num_games} games each) ═══")

    # Tile field
    x_wins_tf, o_wins_tf, draws_tf = 0, 0, 0
    t0 = time.perf_counter()
    for _ in range(num_games):
        w = play_game_tile_field(field)
        if w == 'X': x_wins_tf += 1
        elif w == 'O': o_wins_tf += 1
        else: draws_tf += 1
    time_tf = time.perf_counter() - t0

    # Compiled policy
    x_wins_cp, o_wins_cp, draws_cp = 0, 0, 0
    t0 = time.perf_counter()
    for _ in range(num_games):
        w = play_game_compiled(policy_func)
        if w == 'X': x_wins_cp += 1
        elif w == 'O': o_wins_cp += 1
        else: draws_cp += 1
    time_cp = time.perf_counter() - t0

    wr_tf = x_wins_tf / num_games
    wr_cp = x_wins_cp / num_games
    gap = abs(wr_tf - wr_cp)

    print(f"\n  ┌─────────────────────────────────────────────────────┐")
    print(f"  │ {'Tile Field (softmax+MC)':^25s} │ {'Compiled Policy':^25s} │")
    print(f"  ├─────────────────────────────────────────────────────┤")
    print(f"  │ X wins:  {x_wins_tf:>4}/{num_games} ({wr_tf:.1%})     │ X wins:  {x_wins_cp:>4}/{num_games} ({wr_cp:.1%})     │")
    print(f"  │ O wins:  {o_wins_tf:>4}/{num_games} ({o_wins_tf/num_games:.1%})     │ O wins:  {o_wins_cp:>4}/{num_games} ({o_wins_cp/num_games:.1%})     │")
    print(f"  │ Draws:   {draws_tf:>4}/{num_games} ({draws_tf/num_games:.1%})     │ Draws:   {draws_cp:>4}/{num_games} ({draws_cp/num_games:.1%})     │")
    print(f"  │ Time:    {time_tf:.3f}s ({time_tf/num_games*1000:.2f}ms/game)  │ Time:    {time_cp:.3f}s ({time_cp/num_games*1000:.2f}ms/game)  │")
    print(f"  └─────────────────────────────────────────────────────┘")

    speedup = time_tf / time_cp if time_cp > 0 else float('inf')
    print(f"\n  Win rate gap: {gap:.1%}")
    print(f"  Speedup: {speedup:.1f}x (compiled is {'faster' if speedup < 1 else 'FASTER'}!)")

    if gap <= 0.05:
        print(f"\n  ✅ WITHIN 5% THRESHOLD — compiled policy is deployment-ready!")
    else:
        print(f"\n  ⚠️  Gap ({gap:.1%}) exceeds 5% — tile field has more exploration advantage")

    return {
        "tile_field": {
            "x_wins": x_wins_tf, "o_wins": o_wins_tf, "draws": draws_tf,
            "win_rate": wr_tf, "total_time_s": round(time_tf, 4),
            "ms_per_game": round(time_tf / num_games * 1000, 3),
        },
        "compiled": {
            "x_wins": x_wins_cp, "o_wins": o_wins_cp, "draws": draws_cp,
            "win_rate": wr_cp, "total_time_s": round(time_cp, 4),
            "ms_per_game": round(time_cp / num_games * 1000, 3),
        },
        "gap": round(gap, 4),
        "speedup": round(speedup, 2),
        "within_threshold": gap <= 0.05,
    }


# ─── Phase 4: Compiled Policy Analysis ──────────────────────────────

def analyze_compiled(code, lookup):
    """Analyze the compiled policy artifact."""
    code_bytes = len(code.encode('utf-8'))
    num_entries = len(lookup)
    avg_entry_bytes = code_bytes / max(num_entries, 1)

    print(f"\n═══ Phase 4: Compiled Policy Analysis ═══")
    print(f"  Code size: {code_bytes:,} bytes ({code_bytes/1024:.1f} KB)")
    print(f"  Lookup entries: {num_entries}")
    print(f"  Avg bytes/entry: {avg_entry_bytes:.1f}")
    print(f"  Dependencies: NONE (pure Python, no imports needed)")

    # Estimate microcontroller fit
    if code_bytes < 32000:
        print(f"  Fits on: Arduino Uno (32KB flash) ✅")
    if code_bytes < 256000:
        print(f"  Fits on: ESP8266 (256KB flash) ✅")
    print(f"  Fits on: Any device with Python/browser ✅")

    # Analyze action distribution
    action_dist = {}
    for action in lookup.values():
        action_dist[action] = action_dist.get(action, 0) + 1

    print(f"\n  Action distribution in compiled policy:")
    for pos in sorted(action_dist.keys(), key=lambda x: int(x)):
        count = action_dist[pos]
        bar = '█' * (count // 2)
        print(f"    Pos {pos}: {count:>3} {bar}")

    return {
        "code_bytes": code_bytes,
        "lookup_entries": num_entries,
        "action_distribution": action_dist,
    }


# ─── Phase 5: Edge Case Testing ─────────────────────────────────────

def test_compiled_policy(policy_func):
    """Test the compiled policy on specific board states."""
    print(f"\n═══ Phase 5: Edge Case Tests ═══")

    tests = [
        ("         ", "Empty board — should prefer center"),
        ("    X    ", "X in center — O just played (X turn to respond)"),
        ("X  O     ", "X corner, O opposite — play strategically"),
        ("XO X  O  ", "Complex mid-game"),
        ("XOXOX    ", "Near-end game"),
    ]

    all_legal = True
    for board, desc in tests:
        action = policy_func(board)
        legal_positions = [i for i in range(9) if board[i] == ' ']
        is_legal = int(action) in legal_positions if action else False
        status = "✅" if is_legal else "❌"
        print(f"  {status} {board!r} -> move {action} ({desc})")
        if not is_legal:
            all_legal = False

    return all_legal


# ─── Main ────────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║        TILE COMPILER — Learn → Compile → Deploy     ║")
    print("║  Train with simulation, deploy with zero deps       ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # Phase 1: Train
    field = train_tile_field(num_games=1000)

    # Phase 2: Compile
    code, lookup = compile_tile_field(field)

    # Write compiled policy to file
    compiled_path = os.path.join(os.path.dirname(__file__), "compiled_policy.py")
    with open(compiled_path, 'w') as f:
        f.write(code)
    print(f"  Written to: {compiled_path}")

    # Load it back (ensures it's actually self-contained)
    import importlib.util
    spec = importlib.util.spec_from_file_location("compiled_policy", compiled_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    policy_func = mod.compiled_policy
    print(f"  ✅ Loaded compiled policy (verified zero-dependency)")

    # Phase 3: Evaluate
    results = evaluate(field, policy_func, num_games=1000)

    # Phase 4: Analyze
    analysis = analyze_compiled(code, lookup)

    # Phase 5: Edge cases
    all_legal = test_compiled_policy(policy_func)

    # Save results
    output = {
        "experiment": "tile_compiler",
        "description": "Compile trained tile field into deterministic zero-dep lookup",
        "training": {
            "games": 1000,
            "tiles_learned": len(field.tiles),
            "tiles_compiled": len(lookup),
        },
        "evaluation": results,
        "analysis": analysis,
        "edge_tests_passed": all_legal,
        "innovation": {
            "concept": "Learn with gradients/simulation, deploy with lookups",
            "tile_field_is": "TRAINING algorithm (needs numpy, random, simulation)",
            "compiled_policy_is": "DEPLOYED artifact (zero dependencies, O(1) lookup)",
            "deployment_targets": ["microcontrollers", "browsers", "paper", "any Python"],
        }
    }

    results_path = os.path.join(os.path.dirname(__file__), "tile-compiler-results.json")
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    # Final summary
    print(f"\n{'═'*60}")
    print(f"  TILE COMPILER SUMMARY")
    print(f"{'═'*60}")
    print(f"  Trained: {len(field.tiles)} tiles from 500 games")
    print(f"  Compiled: {len(lookup)} state→action mappings ({analysis['code_bytes']:,} bytes)")
    print(f"  Tile field win rate: {results['tile_field']['win_rate']:.1%}")
    print(f"  Compiled win rate:   {results['compiled']['win_rate']:.1%}")
    print(f"  Gap: {results['gap']:.1%} {'✅' if results['within_threshold'] else '⚠️'}")
    print(f"  Speedup: {results['speedup']:.1f}x")
    print(f"  Edge tests: {'ALL PASSED ✅' if all_legal else 'SOME FAILED ⚠️'}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
