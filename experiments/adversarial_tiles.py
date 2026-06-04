"""
EXPERIMENT: Adversarial Tile Fields

Hypothesis: Two tile fields trained to exploit each other's weaknesses will
BOTH be stronger than fields trained cooperatively or independently.

Setup:
1. Phase 1: Train two independent tile fields (A and B) on TicTacToe for 200 games
2. Phase 2: Adversarial phase — 300 games where:
   - A plays to win normally
   - B targets states where A has high confidence and tries to flip them
3. Cooperative control: Two fields that share score matrices after each game

Evaluation:
- A vs random (1000 games)
- B vs random (1000 games)
- A vs B (1000 games)
- Both vs independently-trained field
- Cooperative pair vs random
"""

import sys
import os
import json
import random
import math
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from zeroclaw.tile_field import TileField
from zeroclaw.games import TicTacToe


class AdversarialField(TileField):
    """Tile field that plays to exploit another field's high-confidence states."""

    def __init__(self, target_field=None, **kwargs):
        super().__init__(**kwargs)
        self.target_field = target_field

    def set_target(self, target_field):
        self.target_field = target_field

    def choose_exploit_action(self, game, state_str, legal_actions):
        """Choose action that targets the opponent's most-visited/confident tiles.

        Strategy: Find states where the target (opponent) has high confidence
        (many visits, high score). Pick actions that lead to those states being
        disrupted — i.e., actions the opponent would NOT expect.
        """
        if not self.target_field or not legal_actions:
            return self.choose_action(game, state_str, legal_actions)

        if len(legal_actions) == 1:
            return legal_actions[0]

        # Score each action by how much it disrupts the target's confidence
        action_scores = {}
        for action in legal_actions:
            # Simulate the resulting state
            sim = game.copy()
            sim.step(action)
            next_state = str(sim.state().state_str)

            exploit_score = 0.0

            # Check if target has strong opinion about the next state
            if next_state in self.target_field.tiles:
                target_tile = self.target_field.tiles[next_state]
                for a, data in target_tile.items():
                    visits = data["chosen"]
                    score = data["score"]
                    # High visits + high score = high confidence = high value to disrupt
                    confidence = min(visits / 10.0, 1.0)
                    exploit_score += confidence * score

            # Also use our own learned knowledge
            my_tile = self.get_or_create(state_str, legal_actions)
            my_score = my_tile[action]["score"]

            # Weight: 40% exploitation, 60% our own winning knowledge
            action_scores[action] = 0.4 * exploit_score + 0.6 * my_score

        # Greedy with some noise for exploration
        actions_list = list(action_scores.keys())
        values = [action_scores[a] for a in actions_list]

        # Softmax with moderate temperature
        temp = 0.5
        max_val = max(values)
        exp_vals = [math.exp(v - max_val) / temp for v in values]
        total = sum(exp_vals)
        probs = [e / total for e in exp_vals]

        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return actions_list[i]
        return actions_list[-1]


class CooperativeField(TileField):
    """Tile field that shares knowledge with a partner after each game."""

    def __init__(self, partner=None, **kwargs):
        super().__init__(**kwargs)
        self.partner = partner

    def set_partner(self, partner):
        self.partner = partner

    def share_knowledge(self):
        """Merge scores from partner into self (average)."""
        if not self.partner:
            return
        for state_str, tile in self.partner.tiles.items():
            if state_str not in self.tiles:
                self.tiles[state_str] = {
                    a: dict(data) for a, data in tile.items()
                }
            else:
                my_tile = self.tiles[state_str]
                for action, data in tile.items():
                    if action in my_tile:
                        # Average the scores
                        my_tile[action]["score"] = (
                            my_tile[action]["score"] + data["score"]
                        ) / 2.0
                    else:
                        my_tile[action] = dict(data)


def play_game_field_vs_field(field_x, field_o, game=None):
    """Play a game between two tile fields. Returns winner."""
    if game is None:
        game = TicTacToe()
    game.reset()

    history_x = []
    history_o = []

    while not game.done:
        state = game.state()
        actions = game.legal_actions()
        if not actions:
            break
        state_str = str(state.state_str)

        if game.current == 'X':
            action = field_x.choose_action(game, state_str, actions)
            history_x.append((state_str, action))
        else:
            action = field_o.choose_action(game, state_str, actions)
            history_o.append((state_str, action))

        game.step(action)

    x_won = game.winner == 'X'
    o_won = game.winner == 'O'

    for state_str, action in history_x:
        field_x.record(state_str, action, x_won)
    for state_str, action in history_o:
        field_o.record(state_str, action, o_won)

    return game.winner


def play_game_adversarial(field_a, field_b, game=None):
    """Adversarial game: A plays to win, B plays to exploit A."""
    if game is None:
        game = TicTacToe()
    game.reset()

    history_a = []
    history_b = []

    while not game.done:
        state = game.state()
        actions = game.legal_actions()
        if not actions:
            break
        state_str = str(state.state_str)

        if game.current == 'X':
            # Player A: plays to win
            action = field_a.choose_action(game, state_str, actions)
            history_a.append((state_str, action))
        else:
            # Player B: plays to exploit A's confidence
            action = field_b.choose_exploit_action(game, state_str, actions)
            history_b.append((state_str, action))

        game.step(action)

    a_won = game.winner == 'X'
    b_won = game.winner == 'O'

    for state_str, action in history_a:
        field_a.record(state_str, action, a_won)
    for state_str, action in history_b:
        field_b.record(state_str, action, b_won)

    return game.winner


def play_game_cooperative(field_x, field_o, game=None):
    """Cooperative game: fields share knowledge after each game."""
    winner = play_game_field_vs_field(field_x, field_o, game)
    field_x.share_knowledge()
    field_o.share_knowledge()
    return winner


def evaluate_vs_random(field, num_games=1000):
    """Evaluate a field against random play."""
    game = TicTacToe()
    wins = 0
    draws = 0
    losses = 0

    for _ in range(num_games):
        game.reset()
        while not game.done:
            state = game.state()
            actions = game.legal_actions()
            if not actions:
                break
            state_str = str(state.state_str)

            if game.current == 'X':
                action = field.choose_action(game, state_str, actions)
            else:
                action = random.choice(actions)

            game.step(action)

        if game.winner == 'X':
            wins += 1
        elif game.winner == 'O':
            losses += 1
        else:
            draws += 1

    return {"wins": wins, "draws": draws, "losses": losses,
            "win_rate": wins / num_games, "draw_rate": draws / num_games}


def evaluate_field_vs_field(field_x, field_o, num_games=1000):
    """Evaluate two fields against each other."""
    game = TicTacToe()
    x_wins = 0
    o_wins = 0
    draws = 0

    for _ in range(num_games):
        game.reset()
        while not game.done:
            state = game.state()
            actions = game.legal_actions()
            if not actions:
                break
            state_str = str(state.state_str)

            if game.current == 'X':
                action = field_x.choose_action(game, state_str, actions)
            else:
                action = field_o.choose_action(game, state_str, actions)

            game.step(action)

        if game.winner == 'X':
            x_wins += 1
        elif game.winner == 'O':
            o_wins += 1
        else:
            draws += 1

    return {"x_wins": x_wins, "o_wins": o_wins, "draws": draws}


def compute_robustness(field):
    """Compute robustness metrics for a field's strategy."""
    if not field.tiles:
        return {"tile_count": 0, "score_variance": 0, "visit_balance": 0,
                "avg_score": 0}

    all_scores = []
    all_visits = []

    for tile in field.tiles.values():
        for action, data in tile.items():
            if data["chosen"] > 0:
                all_scores.append(data["score"])
                all_visits.append(data["chosen"])

    if not all_scores:
        return {"tile_count": len(field.tiles), "score_variance": 0,
                "visit_balance": 0, "avg_score": 0}

    mean_score = sum(all_scores) / len(all_scores)
    score_variance = sum((s - mean_score) ** 2 for s in all_scores) / len(all_scores)

    mean_visits = sum(all_visits) / len(all_visits)
    if mean_visits > 0:
        visit_balance = 1.0 - (sum((v - mean_visits) ** 2 for v in all_visits) / len(all_visits)) ** 0.5 / mean_visits
        visit_balance = max(0, visit_balance)
    else:
        visit_balance = 0

    return {
        "tile_count": len(field.tiles),
        "score_variance": round(score_variance, 6),
        "visit_balance": round(visit_balance, 4),
        "avg_score": round(mean_score, 4),
        "total_states_explored": len(field.tiles),
    }


def main():
    random.seed(42)
    results = {}

    print("=" * 60)
    print("ADVERSARIAL TILE FIELDS EXPERIMENT")
    print("=" * 60)

    # =========================================================================
    # PHASE 1: Independent Training (200 games each)
    # =========================================================================
    print("\n--- PHASE 1: Independent Training (200 games each) ---")

    field_a = TileField(n_simulations=20, temperature=0.3)
    field_b = TileField(n_simulations=20, temperature=0.3)
    game = TicTacToe()

    print("Training field A (200 games)...")
    wins_a = {'X': 0, 'O': 0, 'draw': 0, None: 0}
    for i in range(200):
        w = field_a.train_game(game)
        if w in wins_a:
            wins_a[w] += 1
        else:
            wins_a[None] += 1
        if (i + 1) % 50 == 0:
            total = sum(wins_a.values())
            x_w = wins_a.get('X', 0) + wins_a.get('B', 0)
            print(f"  A: {i+1}/200 | tiles={field_a.size} | P1 wins={x_w/total:.1%}")

    print("Training field B (200 games)...")
    wins_b = {'X': 0, 'O': 0, 'draw': 0, None: 0}
    for i in range(200):
        w = field_b.train_game(game)
        if w in wins_b:
            wins_b[w] += 1
        else:
            wins_b[None] += 1
        if (i + 1) % 50 == 0:
            total = sum(wins_b.values())
            x_w = wins_b.get('X', 0) + wins_b.get('B', 0)
            print(f"  B: {i+1}/200 | tiles={field_b.size} | P1 wins={x_w/total:.1%}")

    # Train independent control field
    print("Training independent control field (200 games)...")
    field_independent = TileField(n_simulations=20, temperature=0.3)
    for i in range(200):
        field_independent.train_game(game)

    # Train cooperative control pair
    print("Training cooperative control pair (200 games)...")
    field_coop_x = CooperativeField(n_simulations=20, temperature=0.3)
    field_coop_o = CooperativeField(n_simulations=20, temperature=0.3)
    field_coop_x.set_partner(field_coop_o)
    field_coop_o.set_partner(field_coop_x)
    for i in range(200):
        play_game_cooperative(field_coop_x, field_coop_o, game)
        if (i + 1) % 25 == 0:
            field_coop_x.evolve()
            field_coop_o.evolve()

    print(f"\nPhase 1 tile counts: A={field_a.size}, B={field_b.size}, "
          f"Indep={field_independent.size}, CoopX={field_coop_x.size}")

    # =========================================================================
    # PHASE 2: Adversarial Training (300 games)
    # =========================================================================
    print("\n--- PHASE 2: Adversarial Training (300 games) ---")

    # Create adversarial field from B's knowledge
    adv_b = AdversarialField(target_field=field_a, n_simulations=20, temperature=0.3)
    # Copy B's learned knowledge into adversarial B
    for state_str, tile in field_b.tiles.items():
        adv_b.tiles[state_str] = {a: dict(d) for a, d in tile.items()}

    adv_a = copy.deepcopy(field_a)

    adv_wins = {'X': 0, 'O': 0, 'draw': 0}
    for i in range(300):
        w = play_game_adversarial(adv_a, adv_b, game)
        if w == 'X':
            adv_wins['X'] += 1
        elif w == 'O':
            adv_wins['O'] += 1
        else:
            adv_wins['draw'] += 1

        if (i + 1) % 25 == 0:
            adv_a.evolve()
            adv_b.evolve()

        if (i + 1) % 100 == 0:
            total = sum(adv_wins.values())
            print(f"  {i+1}/300 | A wins={adv_wins['X']/total:.1%} "
                  f"B wins={adv_wins['O']/total:.1%} "
                  f"Draws={adv_wins['draw']/total:.1%} "
                  f"| tiles: A={adv_a.size} B={adv_b.size}")

    # Continue cooperative training for 300 more games
    print("\nContinuing cooperative training (300 more games)...")
    coop_wins = {'X': 0, 'O': 0, 'draw': 0}
    for i in range(300):
        w = play_game_cooperative(field_coop_x, field_coop_o, game)
        if w == 'X':
            coop_wins['X'] += 1
        elif w == 'O':
            coop_wins['O'] += 1
        else:
            coop_wins['draw'] += 1
        if (i + 1) % 25 == 0:
            field_coop_x.evolve()
            field_coop_o.evolve()
        if (i + 1) % 100 == 0:
            total = sum(coop_wins.values())
            print(f"  Coop {i+1}/300 | X={coop_wins['X']/total:.1%} "
                  f"O={coop_wins['O']/total:.1%} Draws={coop_wins['draw']/total:.1%}")

    # Continue independent training for 300 more games
    print("Continuing independent training (300 more games)...")
    for i in range(300):
        field_independent.train_game(game)

    print(f"\nPhase 2 tile counts: AdvA={adv_a.size}, AdvB={adv_b.size}, "
          f"Indep={field_independent.size}, CoopX={field_coop_x.size}")

    # =========================================================================
    # PHASE 3: Evaluation
    # =========================================================================
    print("\n--- PHASE 3: Evaluation ---")

    N = 1000

    print(f"\n1. Adversarial A vs random ({N} games)...")
    eval_adv_a_random = evaluate_vs_random(adv_a, N)
    print(f"   Win={eval_adv_a_random['win_rate']:.1%} "
          f"Draw={eval_adv_a_random['draw_rate']:.1%}")

    print(f"2. Adversarial B vs random ({N} games)...")
    eval_adv_b_random = evaluate_vs_random(adv_b, N)
    print(f"   Win={eval_adv_b_random['win_rate']:.1%} "
          f"Draw={eval_adv_b_random['draw_rate']:.1%}")

    print(f"3. Adversarial A vs B ({N} games)...")
    eval_a_vs_b = evaluate_field_vs_field(adv_a, adv_b, N)
    print(f"   A wins={eval_a_vs_b['x_wins']} B wins={eval_a_vs_b['o_wins']} "
          f"Draws={eval_a_vs_b['draws']}")

    print(f"4. Adversarial A vs Independent ({N} games)...")
    eval_adv_a_vs_indep = evaluate_field_vs_field(adv_a, field_independent, N)
    print(f"   AdvA={eval_adv_a_vs_indep['x_wins']} Indep={eval_adv_a_vs_indep['o_wins']} "
          f"Draws={eval_adv_a_vs_indep['draws']}")

    print(f"5. Adversarial B vs Independent ({N} games)...")
    eval_adv_b_vs_indep = evaluate_field_vs_field(adv_b, field_independent, N)
    print(f"   AdvB={eval_adv_b_vs_indep['x_wins']} Indep={eval_adv_b_vs_indep['o_wins']} "
          f"Draws={eval_adv_b_vs_indep['draws']}")

    print(f"6. Independent vs random ({N} games)...")
    eval_indep_random = evaluate_vs_random(field_independent, N)
    print(f"   Win={eval_indep_random['win_rate']:.1%} "
          f"Draw={eval_indep_random['draw_rate']:.1%}")

    print(f"7. Cooperative X vs random ({N} games)...")
    eval_coop_random = evaluate_vs_random(field_coop_x, N)
    print(f"   Win={eval_coop_random['win_rate']:.1%} "
          f"Draw={eval_coop_random['draw_rate']:.1%}")

    print(f"8. Cooperative vs Independent ({N} games)...")
    eval_coop_vs_indep = evaluate_field_vs_field(field_coop_x, field_independent, N)
    print(f"   Coop={eval_coop_vs_indep['x_wins']} Indep={eval_coop_vs_indep['o_wins']} "
          f"Draws={eval_coop_vs_indep['draws']}")

    # =========================================================================
    # Robustness Analysis
    # =========================================================================
    print("\n--- ROBUSTNESS ANALYSIS ---")

    robustness = {
        "adversarial_a": compute_robustness(adv_a),
        "adversarial_b": compute_robustness(adv_b),
        "independent": compute_robustness(field_independent),
        "cooperative_x": compute_robustness(field_coop_x),
    }

    for name, r in robustness.items():
        print(f"  {name}: tiles={r['tile_count']}, "
              f"score_var={r['score_variance']:.6f}, "
              f"visit_balance={r['visit_balance']:.4f}, "
              f"avg_score={r['avg_score']:.4f}")

    # =========================================================================
    # Results
    # =========================================================================
    results = {
        "experiment": "adversarial_tiles",
        "hypothesis": "Adversarially-trained fields develop more robust strategies",
        "phases": {
            "phase1_independent_games": 200,
            "phase2_adversarial_games": 300,
        },
        "adversarial_training_summary": adv_wins,
        "cooperative_training_summary": coop_wins,
        "evaluation": {
            "adversarial_a_vs_random": eval_adv_a_random,
            "adversarial_b_vs_random": eval_adv_b_random,
            "adversarial_a_vs_b": eval_a_vs_b,
            "adversarial_a_vs_independent": eval_adv_a_vs_indep,
            "adversarial_b_vs_independent": eval_adv_b_vs_indep,
            "independent_vs_random": eval_indep_random,
            "cooperative_vs_random": eval_coop_random,
            "cooperative_vs_independent": eval_coop_vs_indep,
        },
        "robustness": robustness,
        "conclusions": {},
    }

    # Auto-analyze conclusions
    adv_a_wr = eval_adv_a_random["win_rate"]
    adv_b_wr = eval_adv_b_random["win_rate"]
    indep_wr = eval_indep_random["win_rate"]
    coop_wr = eval_coop_random["win_rate"]

    best_method = max(
        [("adversarial_a", adv_a_wr), ("adversarial_b", adv_b_wr),
         ("independent", indep_wr), ("cooperative", coop_wr)],
        key=lambda x: x[1]
    )

    # Lower score variance = more robust
    adv_avg_var = (robustness["adversarial_a"]["score_variance"] +
                   robustness["adversarial_b"]["score_variance"]) / 2
    indep_var = robustness["independent"]["score_variance"]
    coop_var = robustness["cooperative_x"]["score_variance"]

    most_robust = min(
        [("adversarial", adv_avg_var), ("independent", indep_var), ("cooperative", coop_var)],
        key=lambda x: x[1]
    )

    results["conclusions"] = {
        "best_win_rate_vs_random": {
            "method": best_method[0],
            "win_rate": round(best_method[1], 4),
        },
        "most_robust_strategy": {
            "method": most_robust[0],
            "score_variance": round(most_robust[1], 6),
        },
        "adversarial_a_outperforms_independent": adv_a_wr > indep_wr,
        "adversarial_b_outperforms_independent": adv_b_wr > indep_wr,
        "adversarial_pair_balanced": abs(eval_a_vs_b['x_wins'] - eval_a_vs_b['o_wins']) < 100,
        "adv_a_win_rate": round(adv_a_wr, 4),
        "adv_b_win_rate": round(adv_b_wr, 4),
        "indep_win_rate": round(indep_wr, 4),
        "coop_win_rate": round(coop_wr, 4),
        "hypothesis_supported": (
            adv_avg_var < indep_var and
            (adv_a_wr > indep_wr or adv_b_wr > indep_wr)
        ),
    }

    # Save
    output_path = os.path.join(os.path.dirname(__file__), '..', 'adversarial-results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Win rates vs random:")
    print(f"    Adversarial A:  {adv_a_wr:.1%}")
    print(f"    Adversarial B:  {adv_b_wr:.1%}")
    print(f"    Independent:    {indep_wr:.1%}")
    print(f"    Cooperative:    {coop_wr:.1%}")
    print(f"  Best method: {best_method[0]} ({best_method[1]:.1%})")
    print(f"  Most robust: {most_robust[0]} (var={most_robust[1]:.6f})")
    hyp = results["conclusions"]["hypothesis_supported"]
    print(f"  Hypothesis supported: {hyp}")


if __name__ == "__main__":
    main()
