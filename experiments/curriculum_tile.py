"""
Curriculum Learning in Tile Fields — Experiment

Hypothesis: Training against progressively harder opponents
(random → self-play → optimal) produces stronger tile fields than
training against any single opponent type.

4 training regimes, each 500 games:
1. Curriculum: Phase1 (1-100) vs random, Phase2 (101-300) vs mixed, Phase3 (301-500) self-play
2. Random-only: 500 games vs random
3. Self-play only: 500 games vs self
4. Mixed only: 500 games vs 50/50 random/self

Evaluation: 1000 games vs random on both TicTacToe and Connect4.
"""

import json
import random
import copy
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from zeroclaw.games import TicTacToe, Connect4
from zeroclaw.tile_field import TileField


def play_game_vs_random(game, field, player_field_goes_first=True):
    """Play one game: field vs random. Returns winner string."""
    game.reset()
    history = []

    while not game.done:
        actions = game.legal_actions()
        if not actions:
            break

        state_str = game.state().state_str
        is_field_turn = (player_field_goes_first and game.current == 'X') or \
                        (not player_field_goes_first and game.current == 'O')

        if is_field_turn:
            action = field.choose_action(game, state_str, actions)
        else:
            action = random.choice(actions)

        game.step(action)
        history.append((state_str, action, is_field_turn))

    return game.winner


def play_game_vs_field(game, field1, field2):
    """Play one game: field1 (X) vs field2 (O). Returns winner string."""
    game.reset()

    while not game.done:
        actions = game.legal_actions()
        if not actions:
            break

        state_str = game.state().state_str

        if game.current == 'X':
            action = field1.choose_action(game, state_str, actions)
        else:
            action = field2.choose_action(game, state_str, actions)

        game.step(action)

    return game.winner


def play_self_play_game(game, field):
    """Play one self-play game: field plays both sides. Returns winner."""
    game.reset()
    history = []

    while not game.done:
        actions = game.legal_actions()
        if not actions:
            break

        state_str = game.state().state_str
        action = field.choose_action(game, state_str, actions)
        game.step(action)
        history.append((state_str, action))

    won = game.winner in ('X', 'B', 'player', 0)
    for state_str, action in history:
        field.record(state_str, action, won)

    field._game_count += 1
    if field._game_count % 25 == 0:
        field.evolve()

    return game.winner


def play_training_game_vs_random(game, field):
    """Play one training game vs random opponent. Field is X."""
    game.reset()
    history = []

    while not game.done:
        actions = game.legal_actions()
        if not actions:
            break

        state_str = game.state().state_str

        if game.current == 'X':
            action = field.choose_action(game, state_str, actions)
        else:
            action = random.choice(actions)

        game.step(action)
        history.append((state_str, action))

    # Record from field's perspective (X wins = good)
    won = game.winner == 'X'
    for state_str, action in history:
        field.record(state_str, action, won)

    field._game_count += 1
    if field._game_count % 25 == 0:
        field.evolve()

    return game.winner


def play_training_game_vs_field(game, field1, field2):
    """Play one training game: field1 (X) vs field2 (O). Records for field1 only."""
    game.reset()
    history = []

    while not game.done:
        actions = game.legal_actions()
        if not actions:
            break

        state_str = game.state().state_str

        if game.current == 'X':
            action = field1.choose_action(game, state_str, actions)
        else:
            action = field2.choose_action(game, state_str, actions)

        game.step(action)
        history.append((state_str, action))

    won = game.winner == 'X'
    for state_str, action in history:
        field1.record(state_str, action, won)

    field1._game_count += 1
    if field1._game_count % 25 == 0:
        field1.evolve()

    return game.winner


def deep_copy_field(field):
    """Deep copy a tile field for curriculum phases."""
    new_field = TileField(n_simulations=field.n_simulations, temperature=field.temperature)
    new_field.tiles = copy.deepcopy(field.tiles)
    new_field._game_count = field._game_count
    return new_field


def train_curriculum(game_class, num_games=500):
    """Curriculum training: random → mixed → self-play."""
    field = TileField(n_simulations=20, temperature=0.3)
    game = game_class()

    for i in range(num_games):
        if i < 100:
            # Phase 1: vs pure random
            play_training_game_vs_random(game, field)
        elif i < 300:
            # Phase 2: vs mixed (50% random, 50% previous-phase snapshot)
            if i == 100:
                phase1_snapshot = deep_copy_field(field)
            if random.random() < 0.5:
                play_training_game_vs_random(game, field)
            else:
                play_training_game_vs_field(game, field, phase1_snapshot)
        else:
            # Phase 3: self-play
            play_self_play_game(game, field)

        if (i + 1) % 100 == 0:
            print(f"    Curriculum {i+1}/{num_games} done, tiles={field.size}")

    return field


def train_random_only(game_class, num_games=500):
    """Train against pure random opponent."""
    field = TileField(n_simulations=20, temperature=0.3)
    game = game_class()

    for i in range(num_games):
        play_training_game_vs_random(game, field)
        if (i + 1) % 100 == 0:
            print(f"    Random-only {i+1}/{num_games} done, tiles={field.size}")

    return field


def train_self_play_only(game_class, num_games=500):
    """Train via self-play only."""
    field = TileField(n_simulations=20, temperature=0.3)
    game = game_class()

    for i in range(num_games):
        play_self_play_game(game, field)
        if (i + 1) % 100 == 0:
            print(f"    Self-play {i+1}/{num_games} done, tiles={field.size}")

    return field


def train_mixed_only(game_class, num_games=500):
    """Train vs mixed (50% random, 50% self) throughout."""
    field = TileField(n_simulations=20, temperature=0.3)
    game = game_class()

    for i in range(num_games):
        if random.random() < 0.5:
            play_training_game_vs_random(game, field)
        else:
            play_self_play_game(game, field)
        if (i + 1) % 100 == 0:
            print(f"    Mixed {i+1}/{num_games} done, tiles={field.size}")

    return field


def evaluate(field, game_class, num_games=1000):
    """Evaluate field vs random opponent. Returns win/draw/loss counts."""
    game = game_class()
    wins = 0
    draws = 0
    losses = 0

    for _ in range(num_games):
        # Field plays both sides to be fair
        winner = play_game_vs_random(game, field, player_field_goes_first=True)
        if winner == 'X':
            wins += 1
        elif winner is None or winner == 'draw':
            draws += 1
        else:
            losses += 1

    return {"wins": wins, "draws": draws, "losses": losses,
            "win_rate": wins / num_games, "total": num_games}


def evaluate_both_sides(field, game_class, num_games=1000):
    """Evaluate field as both X and O vs random, return combined."""
    game = game_class()
    wins = 0
    draws = 0
    losses = 0

    for i in range(num_games):
        as_x = (i % 2 == 0)
        winner = play_game_vs_random(game, field, player_field_goes_first=as_x)
        # Field won if it was X and X won, or O and O won
        if as_x:
            field_won = (winner == 'X')
        else:
            field_won = (winner == 'O')

        if field_won:
            wins += 1
        elif winner is None or winner == 'draw':
            draws += 1
        else:
            losses += 1

    return {"wins": wins, "draws": draws, "losses": losses,
            "win_rate": wins / num_games, "total": num_games}


def run_experiment():
    """Run the full curriculum learning experiment."""
    results = {
        "experiment": "curriculum_learning",
        "hypothesis": "Curriculum training produces stronger tile fields than single-regime training",
        "training_games": 500,
        "eval_games": 1000,
        "regimes": {}
    }

    trainers = {
        "curriculum": train_curriculum,
        "random_only": train_random_only,
        "self_play_only": train_self_play_only,
        "mixed_only": train_mixed_only,
    }

    game_classes = {"TicTacToe": TicTacToe, "Connect4": Connect4}

    for regime_name, trainer in trainers.items():
        print(f"\n{'='*60}")
        print(f"Training regime: {regime_name}")
        print(f"{'='*60}")

        results["regimes"][regime_name] = {"evaluations": {}}

        for game_name, game_class in game_classes.items():
            print(f"\n  Training on {game_name}...")
            field = trainer(game_class, num_games=500)

            print(f"  Evaluating {regime_name} on {game_name} (1000 games vs random)...")
            eval_result = evaluate_both_sides(field, game_class, num_games=1000)
            results["regimes"][regime_name]["evaluations"][game_name] = eval_result
            print(f"    → W:{eval_result['wins']} D:{eval_result['draws']} L:{eval_result['losses']} "
                  f"({eval_result['win_rate']:.1%})")
            print(f"    Tiles learned: {field.size}")

        # Cross-game: train on TicTacToe, eval on Connect4 and vice versa
        for train_game, eval_game in [("TicTacToe", "Connect4"), ("Connect4", "TicTacToe")]:
            tag = f"train_{train_game}_eval_{eval_game}"
            print(f"\n  Cross-game: train {train_game} → eval {eval_game}...")
            field = trainers[regime_name](game_classes[train_game], num_games=500)
            eval_result = evaluate_both_sides(field, game_classes[eval_game], num_games=1000)
            results["regimes"][regime_name]["evaluations"][tag] = eval_result
            print(f"    → W:{eval_result['wins']} D:{eval_result['draws']} L:{eval_result['losses']} "
                  f"({eval_result['win_rate']:.1%})")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Regime':<20} {'TTT Win%':<12} {'C4 Win%':<12}")
    print("-" * 44)
    for regime in trainers:
        ttt_wr = results["regimes"][regime]["evaluations"]["TicTacToe"]["win_rate"]
        c4_wr = results["regimes"][regime]["evaluations"]["Connect4"]["win_rate"]
        print(f"{regime:<20} {ttt_wr:<12.1%} {c4_wr:<12.1%}")

    # Determine winner
    best_regime = None
    best_score = -1
    for regime in trainers:
        score = (results["regimes"][regime]["evaluations"]["TicTacToe"]["win_rate"] +
                 results["regimes"][regime]["evaluations"]["Connect4"]["win_rate"])
        if score > best_score:
            best_score = score
            best_regime = regime

    results["best_regime"] = best_regime
    results["best_combined_win_rate"] = best_score / 2
    print(f"\nBest regime: {best_regime} (combined win rate: {best_score/2:.1%})")

    # Hypothesis check
    curriculum_score = (results["regimes"]["curriculum"]["evaluations"]["TicTacToe"]["win_rate"] +
                        results["regimes"]["curriculum"]["evaluations"]["Connect4"]["win_rate"]) / 2
    random_score = (results["regimes"]["random_only"]["evaluations"]["TicTacToe"]["win_rate"] +
                    results["regimes"]["random_only"]["evaluations"]["Connect4"]["win_rate"]) / 2
    self_score = (results["regimes"]["self_play_only"]["evaluations"]["TicTacToe"]["win_rate"] +
                  results["regimes"]["self_play_only"]["evaluations"]["Connect4"]["win_rate"]) / 2
    mixed_score = (results["regimes"]["mixed_only"]["evaluations"]["TicTacToe"]["win_rate"] +
                   results["regimes"]["mixed_only"]["evaluations"]["Connect4"]["win_rate"]) / 2

    hypothesis_supported = curriculum_score >= max(random_score, self_score, mixed_score)
    results["hypothesis_supported"] = hypothesis_supported
    print(f"\nHypothesis {'SUPPORTED ✓' if hypothesis_supported else 'NOT SUPPORTED ✗'}")
    print(f"  Curriculum: {curriculum_score:.1%} | Random: {random_score:.1%} | "
          f"Self-play: {self_score:.1%} | Mixed: {mixed_score:.1%}")

    # Save
    output_path = os.path.join(os.path.dirname(__file__), "..", "curriculum-results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    return results


if __name__ == "__main__":
    random.seed(42)
    run_experiment()
