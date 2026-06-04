"""
EXPERIMENT: Noisy Opponent Generalization

Hypothesis: Training against diverse opponent types produces more ROBUST tile fields
than training against any single type.

Question: Does multi-opponent training sacrifice peak performance against any single
type for better average performance?

Uses TicTacToe (fast, deterministic outcomes) with 7 opponent archetypes:
  - Random: uniform random
  - Greedy: picks highest immediate reward
  - Center-preferring: biased toward center actions
  - Edge-preferring: biased toward edges
  - Copycat: copies player's last move
  - Anti-copycat: avoids player's last move
  - Weighted random: normally distributed action weights
"""

import json
import random
import math
import sys
from pathlib import Path

# Add parent to path so we can import zeroclaw
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zeroclaw.games import TicTacToe
from zeroclaw.tile_field import TileField


# ── Opponent Archetypes ─────────────────────────────────────────────────────

def opponent_random(game, legal_actions, player_last_action=None):
    """Uniform random action selection."""
    return random.choice(legal_actions)


def opponent_greedy(game, legal_actions, player_last_action=None):
    """Picks the action with highest immediate reward via lookahead."""
    best_action = random.choice(legal_actions)
    best_reward = -999.0

    for action in legal_actions:
        g = game.copy()
        reward, done = g.step(action)
        # If this move wins, take it immediately
        if done and g.winner == 'O':
            return action
        # Evaluate board position (count lines where O can still win)
        score = _evaluate_for_o(g) if not done else reward
        if score > best_reward:
            best_reward = score
            best_action = action
    return best_action


def _evaluate_for_o(game):
    """Simple board evaluation for O (player 2)."""
    if not hasattr(game, 'board'):
        return 0.0
    lines = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
             (0, 3, 6), (1, 4, 7), (2, 5, 8),
             (0, 4, 8), (2, 4, 6)]
    score = 0.0
    for a, b, c in lines:
        cells = game.board[a], game.board[b], game.board[c]
        if 'X' not in cells:
            o_count = cells.count('O')
            score += o_count * 0.3 + (0.1 if o_count == 0 else 0)
        if 'O' in cells and 'X' not in cells:
            score += 0.2
    return score


def opponent_center_preferring(game, legal_actions, player_last_action=None):
    """Biased toward center (action 4), then corners, then edges."""
    center = '4'
    corners = ['0', '2', '6', '8']
    edges = ['1', '3', '5', '7']

    # Build weighted preference
    weights = []
    for a in legal_actions:
        if a == center:
            weights.append(5.0)
        elif a in corners:
            weights.append(3.0)
        else:
            weights.append(1.0)

    total = sum(weights)
    r = random.random() * total
    cumulative = 0.0
    for a, w in zip(legal_actions, weights):
        cumulative += w
        if r <= cumulative:
            return a
    return legal_actions[-1]


def opponent_edge_preferring(game, legal_actions, player_last_action=None):
    """Biased toward edge actions (1,3,5,7), then corners, then center."""
    edges = ['1', '3', '5', '7']
    corners = ['0', '2', '6', '8']
    center = '4'

    weights = []
    for a in legal_actions:
        if a in edges:
            weights.append(5.0)
        elif a in corners:
            weights.append(2.0)
        else:
            weights.append(1.0)

    total = sum(weights)
    r = random.random() * total
    cumulative = 0.0
    for a, w in zip(legal_actions, weights):
        cumulative += w
        if r <= cumulative:
            return a
    return legal_actions[-1]


def opponent_copycat(game, legal_actions, player_last_action=None):
    """Copies the player's last move if possible; otherwise random."""
    if player_last_action is not None and player_last_action in legal_actions:
        return player_last_action
    # If can't copy, prefer same row/col as player's last move
    if player_last_action is not None:
        pos = int(player_last_action)
        row, col = pos // 3, pos % 3
        # Try same row or column
        related = [str(r * 3 + c) for r in range(3) for c in range(3)
                    if (r == row or c == col) and str(r * 3 + c) in legal_actions]
        if related:
            return random.choice(related)
    return random.choice(legal_actions)


def opponent_anti_copycat(game, legal_actions, player_last_action=None):
    """Avoids the player's last move and adjacent positions."""
    if player_last_action is None:
        return random.choice(legal_actions)

    pos = int(player_last_action)
    row, col = pos // 3, pos % 3
    # Avoid same position + neighbors
    avoided = set()
    avoided.add(player_last_action)
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = row + dr, col + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            avoided.add(str(nr * 3 + nc))

    safe = [a for a in legal_actions if a not in avoided]
    if safe:
        return random.choice(safe)
    return random.choice(legal_actions)


def opponent_weighted_random(game, legal_actions, player_last_action=None):
    """Normally distributed weights seeded by action index."""
    rng = random.Random(hash(tuple(legal_actions)) + id(game) % 10000)
    weights = [max(0.01, rng.gauss(1.0, 0.5)) for _ in legal_actions]
    total = sum(weights)
    r = rng.random() * total
    cumulative = 0.0
    for a, w in zip(legal_actions, weights):
        cumulative += w
        if r <= cumulative:
            return a
    return legal_actions[-1]


# Registry
OPPONENT_TYPES = {
    'random': opponent_random,
    'greedy': opponent_greedy,
    'center_preferring': opponent_center_preferring,
    'edge_preferring': opponent_edge_preferring,
    'copycat': opponent_copycat,
    'anti_copycat': opponent_anti_copycat,
    'weighted_random': opponent_weighted_random,
}


# ── Training with specific opponent ─────────────────────────────────────────

def train_with_opponent(game_cls, opponent_fn, num_games=500, n_simulations=15):
    """Train a TileField against a specific opponent function.

    The tile field plays as X (player 1). The opponent plays as O (player 2).
    """
    field = TileField(n_simulations=n_simulations, temperature=0.3)

    for game_idx in range(num_games):
        game = game_cls()
        history = []  # (state_str, action) for X moves
        player_last_x_action = None

        while not game.done:
            state = game.state()
            actions = game.legal_actions()
            if not actions:
                break

            state_str = state.state_str

            if game.current == 'X':
                # Tile field chooses for X
                action = field.choose_action(game, state_str, actions)
                player_last_x_action = action
            else:
                # Opponent chooses for O
                action = opponent_fn(game, actions, player_last_x_action)

            game.step(action)
            if game.current == 'O' or game.done:
                # Record X's move after opponent response
                pass
            history.append((state_str, action, game.current))

        # Record X's moves only
        won = game.winner == 'X'
        for state_str, action, player in history:
            if player == 'X' and state_str in field.tiles and action in field.tiles[state_str]:
                field.record(state_str, action, won)

        if (game_idx + 1) % 25 == 0:
            field.evolve()

    field.evolve()
    return field


def train_multi_opponent(game_cls, opponent_fns, num_games=500, strategy='rotate', n_simulations=15):
    """Train a TileField against multiple opponent types.

    strategy:
        'rotate' - cycle through opponents each game
        'round_robin' - strict round-robin scheduling
    """
    field = TileField(n_simulations=n_simulations, temperature=0.3)
    opponents = list(opponent_fns)

    for game_idx in range(num_games):
        if strategy == 'rotate':
            opp_idx = game_idx % len(opponents)
        else:  # round_robin
            opp_idx = game_idx % len(opponents)

        opponent_fn = opponents[opp_idx]
        game = game_cls()
        history = []
        player_last_x_action = None

        while not game.done:
            actions = game.legal_actions()
            if not actions:
                break

            state = game.state()
            state_str = state.state_str

            if game.current == 'X':
                action = field.choose_action(game, state_str, actions)
                player_last_x_action = action
            else:
                action = opponent_fn(game, actions, player_last_x_action)

            game.step(action)
            history.append((state_str, action, game.current))

        won = game.winner == 'X'
        for state_str, action, player in history:
            if player == 'X' and state_str in field.tiles and action in field.tiles[state_str]:
                field.record(state_str, action, won)

        if (game_idx + 1) % 25 == 0:
            field.evolve()

    field.evolve()
    return field


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_field(field, game_cls, opponent_fn, num_games=100):
    """Evaluate a trained tile field against a specific opponent.

    Returns dict with win/draw/loss counts and rates.
    """
    wins = draws = losses = 0

    for _ in range(num_games):
        game = game_cls()
        player_last_x_action = None

        while not game.done:
            actions = game.legal_actions()
            if not actions:
                break

            state_str = game.state().state_str

            if game.current == 'X':
                # Tile field plays (use learned policy, no exploration)
                action = field.choose_action(game, state_str, actions)
                player_last_x_action = action
            else:
                action = opponent_fn(game, actions, player_last_x_action)

            game.step(action)

        if game.winner == 'X':
            wins += 1
        elif game.winner is None:
            draws += 1
        else:
            losses += 1

    total = wins + draws + losses
    return {
        'wins': wins,
        'draws': draws,
        'losses': losses,
        'total': total,
        'win_rate': wins / total if total > 0 else 0,
        'draw_rate': draws / total if total > 0 else 0,
        'loss_rate': losses / total if total > 0 else 0,
    }


# ── Main Experiment ─────────────────────────────────────────────────────────

def run_experiment():
    print("=" * 60)
    print("EXPERIMENT: Noisy Opponent Generalization")
    print("=" * 60)

    game_cls = TicTacToe
    num_train = 500
    num_eval = 100
    opponent_names = list(OPPONENT_TYPES.keys())
    opponent_fns = [OPPONENT_TYPES[n] for n in opponent_names]

    # ── Phase 1: Training ────────────────────────────────────────────────
    print("\n── Phase 1: Training ──\n")

    trained_fields = {}

    # Single-opponent: train vs random only
    print("Training: single-opponent (random only)...")
    trained_fields['single_random'] = train_with_opponent(
        game_cls, OPPONENT_TYPES['random'], num_games=num_train
    )
    print(f"  Done. Tiles: {trained_fields['single_random'].size}")

    # Single-opponent: train vs greedy only (for comparison)
    print("Training: single-opponent (greedy only)...")
    trained_fields['single_greedy'] = train_with_opponent(
        game_cls, OPPONENT_TYPES['greedy'], num_games=num_train
    )
    print(f"  Done. Tiles: {trained_fields['single_greedy'].size}")

    # Multi-opponent: rotate through all 7 types
    print("Training: multi-opponent (rotate through all 7)...")
    trained_fields['multi_rotate'] = train_multi_opponent(
        game_cls, opponent_fns, num_games=num_train, strategy='rotate'
    )
    print(f"  Done. Tiles: {trained_fields['multi_rotate'].size}")

    # Tournament: round-robin
    print("Training: tournament (round-robin through all 7)...")
    trained_fields['tournament'] = train_multi_opponent(
        game_cls, opponent_fns, num_games=num_train, strategy='round_robin'
    )
    print(f"  Done. Tiles: {trained_fields['tournament'].size}")

    # ── Phase 2: Evaluation ──────────────────────────────────────────────
    print("\n── Phase 2: Cross-Evaluation ──\n")

    results = {
        'config': {
            'game': 'TicTacToe',
            'num_train': num_train,
            'num_eval': num_eval,
            'opponent_types': opponent_names,
            'training_strategies': list(trained_fields.keys()),
        },
        'training_tile_counts': {
            name: field.size for name, field in trained_fields.items()
        },
        'evaluation': {},
        'analysis': {},
    }

    for field_name, field in trained_fields.items():
        print(f"\nEvaluating field: {field_name}")
        results['evaluation'][field_name] = {}

        for opp_name in opponent_names:
            opp_fn = OPPONENT_TYPES[opp_name]
            eval_result = evaluate_field(field, game_cls, opp_fn, num_games=num_eval)
            results['evaluation'][field_name][opp_name] = eval_result
            wr = eval_result['win_rate']
            print(f"  vs {opp_name:20s}: win={eval_result['wins']:3d}/{num_eval} "
                  f"({wr:.1%})  draw={eval_result['draws']}  loss={eval_result['losses']}")

    # ── Phase 3: Analysis ────────────────────────────────────────────────
    print("\n── Phase 3: Analysis ──\n")

    for field_name in trained_fields:
        evals = results['evaluation'][field_name]
        win_rates = [e['win_rate'] for e in evals.values()]
        avg_wr = sum(win_rates) / len(win_rates)
        min_wr = min(win_rates)
        max_wr = max(win_rates)
        std_wr = (sum((w - avg_wr) ** 2 for w in win_rates) / len(win_rates)) ** 0.5
        consistency = max_wr - min_wr  # lower = more consistent

        best_opp = max(evals.keys(), key=lambda k: evals[k]['win_rate'])
        worst_opp = min(evals.keys(), key=lambda k: evals[k]['win_rate'])

        results['analysis'][field_name] = {
            'avg_win_rate': avg_wr,
            'min_win_rate': min_wr,
            'max_win_rate': max_wr,
            'std_win_rate': std_wr,
            'consistency_range': consistency,
            'best_opponent': best_opp,
            'worst_opponent': worst_opp,
        }

        print(f"{field_name}:")
        print(f"  Avg win rate:   {avg_wr:.1%}")
        print(f"  Min win rate:   {min_wr:.1%} (vs {worst_opp})")
        print(f"  Max win rate:   {max_wr:.1%} (vs {best_opp})")
        print(f"  Std win rate:   {std_wr:.3f}")
        print(f"  Consistency:    {consistency:.3f} (lower=better)")

    # ── Conclusion ───────────────────────────────────────────────────────
    print("\n── Conclusion ──\n")

    analysis = results['analysis']

    # Most consistent
    most_consistent = min(analysis.keys(), key=lambda k: analysis[k]['consistency_range'])
    # Best average
    best_avg = max(analysis.keys(), key=lambda k: analysis[k]['avg_win_rate'])
    # Best peak
    best_peak = max(analysis.keys(), key=lambda k: analysis[k]['max_win_rate'])

    results['conclusion'] = {
        'most_consistent': most_consistent,
        'best_average_performance': best_avg,
        'best_peak_performance': best_peak,
        'multi_vs_single': {},
    }

    # Compare multi-opponent vs single-opponent
    multi_analysis = analysis['multi_rotate']
    single_analysis = analysis['single_random']

    avg_tradeoff = multi_analysis['avg_win_rate'] - single_analysis['avg_win_rate']
    peak_tradeoff = multi_analysis['max_win_rate'] - single_analysis['max_win_rate']
    consistency_tradeoff = multi_analysis['consistency_range'] - single_analysis['consistency_range']

    results['conclusion']['multi_vs_single'] = {
        'avg_win_rate_diff': avg_tradeoff,
        'peak_win_rate_diff': peak_tradeoff,
        'consistency_diff': consistency_tradeoff,
        'multi_sacrifices_peak': peak_tradeoff < 0,
        'multi_more_consistent': consistency_tradeoff < 0,
        'multi_better_average': avg_tradeoff > 0,
        'answer': None,
    }

    print(f"Most consistent:      {most_consistent}")
    print(f"Best avg performance: {best_avg}")
    print(f"Best peak:            {best_peak}")
    print()
    print(f"Multi vs Single (random):")
    print(f"  Avg WR diff:     {avg_tradeoff:+.1%}")
    print(f"  Peak WR diff:    {peak_tradeoff:+.1%}")
    print(f"  Consistency:     {consistency_tradeoff:+.3f}")

    answer_parts = []
    if avg_tradeoff > 0:
        answer_parts.append(f"Multi-opponent training IMPROVES average performance ({avg_tradeoff:+.1%})")
    else:
        answer_parts.append(f"Multi-opponent training HURTS average performance ({avg_tradeoff:+.1%})")

    if peak_tradeoff < 0:
        answer_parts.append(f"Multi-opponent training SACRIFICES peak performance ({peak_tradeoff:+.1%})")
    else:
        answer_parts.append(f"Multi-opponent training does NOT sacrifice peak performance ({peak_tradeoff:+.1%})")

    if consistency_tradeoff < 0:
        answer_parts.append(f"Multi-opponent training IS more consistent (range diff: {consistency_tradeoff:+.3f})")
    else:
        answer_parts.append(f"Multi-opponent training is NOT more consistent (range diff: {consistency_tradeoff:+.3f})")

    answer = " | ".join(answer_parts)
    results['conclusion']['answer'] = answer
    print(f"\n  → {answer}")

    # ── Save ─────────────────────────────────────────────────────────────
    output_path = Path(__file__).parent / 'noisy-opponents-results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == '__main__':
    run_experiment()
