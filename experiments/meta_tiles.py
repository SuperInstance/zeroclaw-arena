"""
Meta-Learning in Tile Fields — Learning to Learn

Hypothesis: A tile field trained on many different games will develop a
"meta-prior" — score initialization patterns that make it learn faster on
NEW games.

Method:
1. Meta-training: Train ONE tile field across 5 games sequentially
   - After each game, keep the score INITIALIZATION patterns (not the actual scores)
   - Track: which states tend to get high initial scores? Which actions?
2. Extract meta-prior: average score initialization across all games
3. Meta-test: Train on a NEW game using meta-prior as initialization
   - Compare: meta-initialized vs random-initialized (0.5)
   - Measure: learning speed (games to reach X% win rate), final performance

Also tests:
- Negative transfer: train on TTT only, test on C4
- Positive transfer: train on TTT variants, test on TTT
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import random
import math
import json
import time
from collections import defaultdict
from zeroclaw.tile_field import TileField
from zeroclaw.games import TicTacToe, Connect4, HoldemHand


# ─── Game Variants for diversity ────────────────────────────────────────

class SmallConnect4:
    """Connect 4 on a 4×5 board (smaller, faster)."""

    def __init__(self):
        self.rows = 4
        self.cols = 5
        self.reset()

    def reset(self):
        self.board = [[' '] * self.cols for _ in range(self.rows)]
        self.current = 'X'
        self.turn = 0
        self.done = False
        self.winner = None

    def state(self):
        from zeroclaw.games import GameState
        board_str = ''.join(''.join(row) for row in self.board)
        return GameState(board_str, self.turn, self.current)

    def legal_actions(self):
        if self.done:
            return []
        return [str(c) for c in range(self.cols) if self.board[0][c] == ' ']

    def step(self, action: str):
        col = int(action)
        if col < 0 or col >= self.cols or self.board[0][col] != ' ':
            return -1.0, True
        row = self.rows - 1
        while row >= 0 and self.board[row][col] != ' ':
            row -= 1
        if row < 0:
            return -1.0, True
        self.board[row][col] = self.current
        self.turn += 1
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for sign in [1, -1]:
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < self.rows and 0 <= c < self.cols and self.board[r][c] == self.current:
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= 4:
                self.done = True
                self.winner = self.current
                return (1.0 if self.current == 'X' else -1.0), True
        if self.turn >= self.rows * self.cols:
            self.done = True
            return 0.0, True
        self.current = 'O' if self.current == 'X' else 'X'
        return 0.0, False

    def copy(self):
        g = SmallConnect4()
        g.board = [row[:] for row in self.board]
        g.current = self.current
        g.turn = self.turn
        g.done = self.done
        g.winner = self.winner
        return g


class MisereTTT:
    """Misère tic-tac-toe: you LOSE if you get 3 in a row. Player X tries to LOSE."""

    def __init__(self):
        self.board = [' '] * 9
        self.current = 'X'
        self.turn = 0
        self.done = False
        self.winner = None

    def state(self):
        from zeroclaw.games import GameState
        return GameState('M' + ''.join(self.board), self.turn, self.current)

    def legal_actions(self):
        if self.done:
            return []
        return [str(i) for i in range(9) if self.board[i] == ' ']

    def step(self, action: str):
        pos = int(action)
        if self.board[pos] != ' ':
            return -1.0, True
        self.board[pos] = self.current
        self.turn += 1
        lines = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
        for a, b, c in lines:
            if self.board[a] == self.board[b] == self.board[c] != ' ':
                self.done = True
                # In misère, the player who completes the line LOSES
                loser = self.current
                self.winner = 'O' if loser == 'X' else 'X'
                return (1.0 if self.winner == 'X' else -1.0), True
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
        g = MisereTTT()
        g.board = self.board[:]
        g.current = self.current
        g.turn = self.turn
        g.done = self.done
        g.winner = self.winner
        return g


class RestrictedTTT:
    """TTT where first move must be a corner. Forces different opening patterns."""

    def __init__(self):
        self.board = [' '] * 9
        self.current = 'X'
        self.turn = 0
        self.done = False
        self.winner = None

    def state(self):
        from zeroclaw.games import GameState
        return GameState('R' + ''.join(self.board), self.turn, self.current)

    def legal_actions(self):
        if self.done:
            return []
        if self.turn == 0:
            return ['0', '2', '6', '8']  # Corners only for first move
        return [str(i) for i in range(9) if self.board[i] == ' ']

    def step(self, action: str):
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
                return (1.0 if self.current == 'X' else -1.0), True
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
        g = RestrictedTTT()
        g.board = self.board[:]
        g.current = self.current
        g.turn = self.turn
        g.done = self.done
        g.winner = self.winner
        return g


# ─── Meta Tile Field ────────────────────────────────────────────────────

class MetaTileField(TileField):
    """Tile field that can extract and apply meta-priors."""

    def __init__(self, n_simulations=20, temperature=0.3, init_prior=None):
        super().__init__(n_simulations, temperature)
        self.init_prior = init_prior or {}  # state_pattern -> {action -> initial_score}

    def get_or_create(self, state_str, legal_actions):
        if state_str not in self.tiles:
            # Apply meta-prior if available
            init_scores = {}
            for a in legal_actions:
                # Check if we have a prior for this kind of state
                prior_score = self._lookup_prior(state_str, a)
                init_scores[a] = prior_score
            self.tiles[state_str] = {
                a: {"score": s, "chosen": 0, "won": 0}
                for a, s in init_scores.items()
            }
        tile = self.tiles[state_str]
        for a in legal_actions:
            if a not in tile:
                prior_score = self._lookup_prior(state_str, a)
                tile[a] = {"score": prior_score, "chosen": 0, "won": 0}
        return tile

    def _lookup_prior(self, state_str, action):
        """Look up initialization score from meta-prior."""
        if not self.init_prior:
            return 0.5

        # Check direct match
        if state_str in self.init_prior:
            action_priors = self.init_prior[state_str]
            if action in action_priors:
                return action_priors[action]

        # Check pattern-level priors (feature-based)
        # Extract features from state_str to generalize
        features = self._extract_features(state_str)
        if features:
            for pattern, action_priors in self.init_prior.items():
                if isinstance(action_priors, dict) and action in action_priors:
                    pat_features = self._extract_features(pattern)
                    if pat_features and self._feature_similarity(features, pat_features) > 0.7:
                        return action_priors[action]

        return 0.5

    def _extract_features(self, state_str):
        """Extract positional features from a state string."""
        # Count occupied positions, detect center control, etc.
        occupied = sum(1 for c in state_str if c in 'XO')
        center_control = 0
        if len(state_str) > 4:
            mid = len(state_str) // 2
            if state_str[mid] == 'X':
                center_control = 1
            elif state_str[mid] == 'O':
                center_control = -1
        return {"occupied": occupied, "center": center_control, "length": len(state_str)}

    def _feature_similarity(self, f1, f2):
        """Simple feature similarity."""
        sim = 0.0
        if f1.get("length") == f2.get("length"):
            sim += 0.4
        if f1.get("center") == f2.get("center"):
            sim += 0.3
        occ_diff = abs(f1.get("occupied", 0) - f2.get("occupied", 0))
        sim += max(0, 0.3 - occ_diff * 0.05)
        return sim

    def extract_init_patterns(self):
        """Extract the current score patterns as initialization prior."""
        patterns = {}
        for state_str, tile in self.tiles.items():
            patterns[state_str] = {a: d["score"] for a, d in tile.items()}
        return patterns


def train_and_measure(field, game, num_games, eval_interval=25, eval_games=50):
    """Train and measure win rate at intervals. Returns list of (games_trained, win_rate)."""
    results = []
    wins_accum = defaultdict(int)
    total_accum = defaultdict(int)

    for i in range(num_games):
        game.reset()
        history = []

        while not game.done:
            state = game.state()
            actions = game.legal_actions()
            if not actions:
                break
            state_str = str(state.state_str)
            if hasattr(game, 'current') and game.current in ('X', 'B', 'player', 0):
                action = field.choose_action(game, state_str, actions)
            else:
                action = random.choice(actions)
            game.step(action)
            history.append((state_str, action))

        won = getattr(game, 'winner', None) in ('X', 'B', 'player', 0)
        for s, a in history:
            field.record(s, a, won)

        field._game_count += 1
        if field._game_count % 25 == 0:
            field.evolve()

        if (i + 1) % eval_interval == 0:
            # Evaluate current performance
            eval_wins = 0
            for _ in range(eval_games):
                game.reset()
                while not game.done:
                    state = game.state()
                    actions = game.legal_actions()
                    if not actions:
                        break
                    state_str = str(state.state_str)
                    if hasattr(game, 'current') and game.current in ('X', 'B', 'player', 0):
                        action = field.choose_action(game, state_str, actions)
                    else:
                        action = random.choice(actions)
                    game.step(action)
                if getattr(game, 'winner', None) in ('X', 'B', 'player', 0):
                    eval_wins += 1
            wr = eval_wins / eval_games
            results.append((i + 1, wr))

    return results


def extract_meta_prior(all_patterns):
    """Average initialization patterns across multiple games.
    
    Returns a dict mapping state patterns to action scores,
    averaged across all game experiences.
    """
    action_scores = defaultdict(list)
    
    for patterns in all_patterns:
        for state_str, action_map in patterns.items():
            for action, score in action_map.items():
                action_scores[(state_str, action)].append(score)
    
    meta_prior = {}
    for (state_str, action), scores in action_scores.items():
        if state_str not in meta_prior:
            meta_prior[state_str] = {}
        meta_prior[state_str][action] = sum(scores) / len(scores)
    
    return meta_prior


def extract_feature_prior(all_patterns):
    """Extract feature-level meta-prior (more generalizable).
    
    Groups states by turn count and computes average score per action index.
    """
    # Group by turn number and action
    turn_action_scores = defaultdict(list)
    
    for patterns in all_patterns:
        for state_str, action_map in patterns.items():
            # Try to extract turn from state
            turn = 0
            if 'turn=' in state_str:
                try:
                    turn = int(state_str.split('turn=')[1].split('|')[0].split(']')[0])
                except (ValueError, IndexError):
                    pass
            elif len(state_str) >= 9:
                # Board games: count occupied squares
                turn = sum(1 for c in state_str if c in 'XO')
            
            for action, score in action_map.items():
                turn_action_scores[(turn, action)].append(score)
    
    # Average
    feature_prior = {}
    for (turn, action), scores in turn_action_scores.items():
        key = f"turn_{turn}"
        if key not in feature_prior:
            feature_prior[key] = {}
        feature_prior[key][action] = sum(scores) / len(scores)
    
    return feature_prior


# ─── Main Experiment ────────────────────────────────────────────────────

def run_meta_learning():
    random.seed(42)
    RESULTS = {}

    print("=" * 70)
    print("META-LEARNING IN TILE FIELDS — Can a field learn HOW to learn?")
    print("=" * 70)

    # ── Phase 1: Meta-training on 5 diverse games ──────────────────────
    meta_games = [
        ("TicTacToe", TicTacToe),
        ("Misère TTT", MisereTTT),
        ("Restricted TTT", RestrictedTTT),
        ("Small Connect 4", SmallConnect4),
        ("Connect 4", Connect4),
    ]
    GAMES_PER_META = 150
    all_init_patterns = []
    per_game_patterns = {}

    print(f"\n{'─' * 70}")
    print("PHASE 1: Meta-training on 5 diverse games")
    print(f"{'─' * 70}")

    for game_name, GameClass in meta_games:
        game = GameClass()
        field = TileField(n_simulations=15, temperature=0.3)
        print(f"\n  Training on {game_name} ({GAMES_PER_META} games)...")
        wins = field.train(game, num_games=GAMES_PER_META, evolve_every=25)
        total = sum(wins.values())
        x_wins = wins.get('X', 0) + wins.get('B', 0)
        wr = x_wins / total if total else 0
        print(f"  → {game_name}: P1 win rate = {wr:.1%}, tiles = {field.size}")

        patterns = field.extract_init_patterns() if hasattr(field, 'extract_init_patterns') else {}
        # Manually extract from tiles
        patterns = {}
        for state_str, tile in field.tiles.items():
            patterns[state_str] = {a: d["score"] for a, d in tile.items()}
        
        all_init_patterns.append(patterns)
        per_game_patterns[game_name] = patterns

    # ── Phase 2: Extract meta-priors ───────────────────────────────────
    print(f"\n{'─' * 70}")
    print("PHASE 2: Extracting meta-priors")
    print(f"{'─' * 70}")

    # Full meta-prior (all games)
    meta_prior_full = extract_meta_prior(all_init_patterns)
    feature_prior = extract_feature_prior(all_init_patterns)

    # TTT-only prior (for negative transfer test)
    ttt_prior = extract_meta_prior([per_game_patterns["TicTacToe"]])

    # TTT-variants prior (for positive transfer test)
    variant_patterns = [per_game_patterns["Misère TTT"], per_game_patterns["Restricted TTT"]]
    variants_prior = extract_meta_prior(variant_patterns)

    total_states = sum(len(p) for p in all_init_patterns)
    print(f"  Total state patterns collected: {total_states}")
    print(f"  Meta-prior (full): {len(meta_prior_full)} unique states")
    print(f"  Feature-level prior: {len(feature_prior)} turn-action groups")
    print(f"  TTT-only prior: {len(ttt_prior)} states")
    print(f"  Variants-only prior: {len(variants_prior)} states")

    # Analyze score distribution in meta-prior
    all_scores = []
    for state_priors in meta_prior_full.values():
        for score in state_priors.values():
            all_scores.append(score)
    if all_scores:
        print(f"\n  Score distribution in meta-prior:")
        print(f"    Mean:   {sum(all_scores)/len(all_scores):.3f}")
        print(f"    Min:    {min(all_scores):.3f}")
        print(f"    Max:    {max(all_scores):.3f}")
        above_half = sum(1 for s in all_scores if s > 0.5)
        print(f"    Above 0.5: {above_half}/{len(all_scores)} ({above_half/len(all_scores):.1%})")

    # ── Phase 3: Meta-test on NEW game (Hold'em) ───────────────────────
    print(f"\n{'─' * 70}")
    print("PHASE 3: Meta-test on Hold'em (unseen game)")
    print(f"{'─' * 70}")

    EVAL_GAMES_TRAIN = 200
    EVAL_INTERVAL = 25
    EVAL_GAMES_MEASURE = 80
    TARGET_WR = 0.45  # Target win rate to measure learning speed

    # 3a: Meta-initialized field
    print("\n  Training META-INITIALIZED field on Hold'em...")
    meta_field = MetaTileField(n_simulations=15, temperature=0.3, init_prior=meta_prior_full)
    holdem_meta = HoldemHand()
    meta_learning_curve = train_and_measure(
        meta_field, holdem_meta, EVAL_GAMES_TRAIN, EVAL_INTERVAL, EVAL_GAMES_MEASURE
    )

    # 3b: Random-initialized field (baseline)
    print("  Training RANDOM-INITIALIZED field on Hold'em...")
    random_field = MetaTileField(n_simulations=15, temperature=0.3, init_prior={})
    holdem_random = HoldemHand()
    random_learning_curve = train_and_measure(
        random_field, holdem_random, EVAL_GAMES_TRAIN, EVAL_INTERVAL, EVAL_GAMES_MEASURE
    )

    # 3c: Feature-prior field
    print("  Training FEATURE-PRIOR field on Hold'em...")
    feature_field = MetaTileField(n_simulations=15, temperature=0.3, init_prior=feature_prior)
    holdem_feature = HoldemHand()
    feature_learning_curve = train_and_measure(
        feature_field, holdem_feature, EVAL_GAMES_TRAIN, EVAL_INTERVAL, EVAL_GAMES_MEASURE
    )

    print("\n  Hold'em Learning Curves:")
    print(f"  {'Games':>6} | {'Meta':>6} | {'Random':>6} | {'Feature':>7}")
    print(f"  {'─'*6}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*7}")
    for (g1, m), (_, r), (_, f) in zip(meta_learning_curve, random_learning_curve, feature_learning_curve):
        print(f"  {g1:>6} | {m:>6.1%} | {r:>6.1%} | {f:>7.1%}")

    # Measure learning speed: games to reach target win rate
    def games_to_target(curve, target):
        for games, wr in curve:
            if wr >= target:
                return games
        return None

    meta_speed = games_to_target(meta_learning_curve, TARGET_WR)
    random_speed = games_to_target(random_learning_curve, TARGET_WR)
    feature_speed = games_to_target(feature_learning_curve, TARGET_WR)

    # Final performance
    meta_final = meta_learning_curve[-1][1] if meta_learning_curve else 0
    random_final = random_learning_curve[-1][1] if random_learning_curve else 0
    feature_final = feature_learning_curve[-1][1] if feature_learning_curve else 0

    print(f"\n  Learning Speed (games to reach {TARGET_WR:.0%}):")
    print(f"    Meta-init:    {meta_speed or 'NOT REACHED'}")
    print(f"    Random-init:  {random_speed or 'NOT REACHED'}")
    print(f"    Feature-init: {feature_speed or 'NOT REACHED'}")
    print(f"\n  Final Win Rates:")
    print(f"    Meta-init:    {meta_final:.1%}")
    print(f"    Random-init:  {random_final:.1%}")
    print(f"    Feature-init: {feature_final:.1%}")

    RESULTS["holdem_meta_test"] = {
        "meta_learning_curve": meta_learning_curve,
        "random_learning_curve": random_learning_curve,
        "feature_learning_curve": feature_learning_curve,
        "meta_speed": meta_speed,
        "random_speed": random_speed,
        "feature_speed": feature_speed,
        "meta_final": round(meta_final, 3),
        "random_final": round(random_final, 3),
        "feature_final": round(feature_final, 3),
    }

    # ── Phase 4: Negative Transfer Test (TTT → C4) ────────────────────
    print(f"\n{'─' * 70}")
    print("PHASE 4: Negative Transfer — TTT-only prior on Connect 4")
    print(f"{'─' * 70}")

    NEG_TRAIN = 150

    print("  Training TTT-PRIOR field on Connect 4...")
    neg_field = MetaTileField(n_simulations=15, temperature=0.3, init_prior=ttt_prior)
    c4_neg = Connect4()
    neg_curve = train_and_measure(neg_field, c4_neg, NEG_TRAIN, 25, 50)

    print("  Training FRESH field on Connect 4...")
    fresh_field = MetaTileField(n_simulations=15, temperature=0.3, init_prior={})
    c4_fresh = Connect4()
    fresh_curve = train_and_measure(fresh_field, c4_fresh, NEG_TRAIN, 25, 50)

    print(f"\n  {'Games':>6} | {'TTT-Prior':>9} | {'Fresh':>6}")
    print(f"  {'─'*6}─┼─{'─'*9}─┼─{'─'*6}")
    for (g1, n), (_, f) in zip(neg_curve, fresh_curve):
        print(f"  {g1:>6} | {n:>9.1%} | {f:>6.1%}")

    neg_final = neg_curve[-1][1] if neg_curve else 0
    fresh_final = fresh_curve[-1][1] if fresh_curve else 0
    neg_speed = games_to_target(neg_curve, 0.45)
    fresh_speed = games_to_target(fresh_curve, 0.45)

    transfer_effect = neg_final - fresh_final
    if transfer_effect < -0.03:
        verdict_neg = "NEGATIVE TRANSFER DETECTED"
    elif transfer_effect > 0.03:
        verdict_neg = "Unexpected POSITIVE transfer"
    else:
        verdict_neg = "Neutral — no significant transfer"

    print(f"\n  TTT-prior final: {neg_final:.1%}  |  Fresh final: {fresh_final:.1%}")
    print(f"  Transfer effect: {transfer_effect:+.1%} → {verdict_neg}")

    RESULTS["negative_transfer"] = {
        "ttt_prior_curve": neg_curve,
        "fresh_curve": fresh_curve,
        "ttt_prior_final": round(neg_final, 3),
        "fresh_final": round(fresh_final, 3),
        "transfer_effect": round(transfer_effect, 3),
        "verdict": verdict_neg,
        "ttt_prior_speed": neg_speed,
        "fresh_speed": fresh_speed,
    }

    # ── Phase 5: Positive Transfer Test (TTT variants → TTT) ──────────
    print(f"\n{'─' * 70}")
    print("PHASE 5: Positive Transfer — TTT-variant prior on standard TTT")
    print(f"{'─' * 70}")

    POS_TRAIN = 150

    print("  Training VARIANTS-PRIOR field on TicTacToe...")
    pos_field = MetaTileField(n_simulations=15, temperature=0.3, init_prior=variants_prior)
    ttt_pos = TicTacToe()
    pos_curve = train_and_measure(pos_field, ttt_pos, POS_TRAIN, 25, 50)

    print("  Training FRESH field on TicTacToe...")
    fresh_ttt_field = MetaTileField(n_simulations=15, temperature=0.3, init_prior={})
    ttt_fresh = TicTacToe()
    fresh_ttt_curve = train_and_measure(fresh_ttt_field, ttt_fresh, POS_TRAIN, 25, 50)

    print(f"\n  {'Games':>6} | {'Var-Prior':>9} | {'Fresh':>6}")
    print(f"  {'─'*6}─┼─{'─'*9}─┼─{'─'*6}")
    for (g1, p), (_, f) in zip(pos_curve, fresh_ttt_curve):
        print(f"  {g1:>6} | {p:>9.1%} | {f:>6.1%}")

    pos_final = pos_curve[-1][1] if pos_curve else 0
    fresh_ttt_final = fresh_ttt_curve[-1][1] if fresh_ttt_curve else 0
    pos_speed = games_to_target(pos_curve, 0.55)
    fresh_ttt_speed = games_to_target(fresh_ttt_curve, 0.55)

    pos_effect = pos_final - fresh_ttt_final
    if pos_effect > 0.03:
        verdict_pos = "POSITIVE TRANSFER — variant diversity helps!"
    elif pos_effect < -0.03:
        verdict_pos = "Negative — variant prior hurts"
    else:
        verdict_pos = "Neutral — no significant effect"

    print(f"\n  Variant-prior final: {pos_final:.1%}  |  Fresh final: {fresh_ttt_final:.1%}")
    print(f"  Transfer effect: {pos_effect:+.1%} → {verdict_pos}")

    RESULTS["positive_transfer"] = {
        "variant_prior_curve": pos_curve,
        "fresh_curve": fresh_ttt_curve,
        "variant_prior_final": round(pos_final, 3),
        "fresh_final": round(fresh_ttt_final, 3),
        "transfer_effect": round(pos_effect, 3),
        "verdict": verdict_pos,
        "variant_prior_speed": pos_speed,
        "fresh_speed": fresh_ttt_speed,
    }

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SUMMARY: Meta-Learning in Tile Fields")
    print(f"{'=' * 70}")

    print(f"\n  Meta-prior statistics:")
    print(f"    States in prior: {len(meta_prior_full)}")
    print(f"    Feature groups: {len(feature_prior)}")

    print(f"\n  Hold'em (unseen game) results:")
    print(f"    Meta-init final:    {meta_final:.1%}")
    print(f"    Random-init final:  {random_final:.1%}")
    print(f"    Feature-init final: {feature_final:.1%}")
    meta_advantage = meta_final - random_final
    feature_advantage = feature_final - random_final
    print(f"    Meta advantage:     {meta_advantage:+.1%}")
    print(f"    Feature advantage:  {feature_advantage:+.1%}")

    if meta_speed and random_speed:
        speedup = random_speed / meta_speed
        print(f"    Learning speedup:   {speedup:.2f}x")
    else:
        speedup = None
        print(f"    Learning speedup:   N/A (target not reached)")

    print(f"\n  Transfer effects:")
    print(f"    TTT→C4 (negative test):  {transfer_effect:+.1%} — {verdict_neg}")
    print(f"    Variants→TTT (positive): {pos_effect:+.1%} — {verdict_pos}")

    # Overall conclusion
    if meta_advantage > 0.02:
        print(f"\n  ✅ META-LEARNING CONFIRMED: Multi-game prior improves learning on unseen game")
    elif meta_advantage > 0:
        print(f"\n  ⚠️ MARGINAL META-LEARNING: Slight improvement from multi-game prior")
    else:
        print(f"\n  ❌ NO META-LEARNING: Multi-game prior did not help")

    RESULTS["summary"] = {
        "meta_prior_size": len(meta_prior_full),
        "feature_prior_size": len(feature_prior),
        "meta_advantage": round(meta_advantage, 3),
        "feature_advantage": round(feature_advantage, 3),
        "speedup": round(speedup, 2) if speedup else None,
        "negative_transfer_effect": round(transfer_effect, 3),
        "positive_transfer_effect": round(pos_effect, 3),
        "conclusion": "confirmed" if meta_advantage > 0.02 else ("marginal" if meta_advantage > 0 else "none"),
    }

    return RESULTS


if __name__ == "__main__":
    t0 = time.time()
    results = run_meta_learning()
    elapsed = time.time() - t0
    results["elapsed_seconds"] = round(elapsed, 1)
    print(f"\n  Total time: {elapsed:.1f}s")

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), '..', 'meta-learning-results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {out_path}")
