"""
Dice Flavors Experiment — How do different randomness distributions
affect the tile field conservation law?

Casey's insight: games use different "flavors of randomness" — dice, cards,
weighted probabilities. The tile field should handle all of them, but the
conservation law might vary.

Tests 5 distributions:
1. Uniform dice (1d6)        — tic-tac-toe style
2. Weighted dice (center)    — Connect4 style
3. Card draw (no replace)    — Hold'em style
4. Normal distribution       — Go style
5. Power law                 — real-world style

For each: 2-player game, tile field training, measure convergence speed,
conservation CV, and negative space clarity.

Hypothesis: Power law has WEAKEST conservation; uniform has STRONGEST.
"""

import random
import numpy as np
import json
import os
import sys
import time
from collections import defaultdict, Counter
from copy import deepcopy
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))
from zeroclaw import GameState, StateTile


# ─── Distribution Samplers ────────────────────────────────────────

class DiceSampler:
    """Base class for reward distribution samplers."""
    def __init__(self, name: str, n_actions: int = 6):
        self.name = name
        self.n_actions = n_actions

    def sample(self, rng: np.random.Generator) -> float:
        raise NotImplementedError

    def description(self) -> str:
        raise NotImplementedError


class UniformDice(DiceSampler):
    """1d6: each outcome equally likely."""
    def __init__(self):
        super().__init__("uniform", 6)
    def sample(self, rng):
        return float(rng.integers(1, 7))
    def description(self):
        return "Uniform 1d6 — flat probability, every face equal"


class WeightedDice(DiceSampler):
    """Biased toward center values (3-4 more likely than 1 or 6)."""
    def __init__(self, n_actions=6):
        super().__init__("weighted", n_actions)
        # weights peak at center
        self.weights = np.array([1, 3, 6, 6, 3, 1], dtype=float)
        self.weights /= self.weights.sum()

    def sample(self, rng):
        return float(rng.choice([1, 2, 3, 4, 5, 6], p=self.weights))
    def description(self):
        return "Weighted dice — center-biased (Connect4-style)"


class CardDeck(DiceSampler):
    """Draw without replacement from a deck. Exhausts and reshuffles."""
    def __init__(self, n_actions=6):
        super().__init__("card_deck", n_actions)
        self.deck = list(range(1, 11)) * 4  # 40 cards, values 1-10
        self._reshuffle()

    def _reshuffle(self):
        self.deck = list(range(1, 11)) * 4
        random.shuffle(self.deck)

    def sample(self, rng):
        if len(self.deck) == 0:
            self._reshuffle()
        return float(self.deck.pop())
    def description(self):
        return "Card deck — draw without replacement (Hold'em-style)"


class NormalDice(DiceSampler):
    """Normal distribution, most outcomes near center, rare extremes."""
    def __init__(self):
        super().__init__("normal", 6)
    def sample(self, rng):
        # mean=3.5, std=1.5, clipped to [1,6]
        v = rng.normal(3.5, 1.5)
        return float(max(1, min(6, round(v))))
    def description(self):
        return "Normal distribution — bell curve (Go-style)"


class PowerLawDice(DiceSampler):
    """Power law: few extreme outcomes, many mundane."""
    def __init__(self):
        super().__init__("power_law", 6)
    def sample(self, rng):
        # Pareto distribution, shape=1.5, scaled to roughly [1,10]
        v = (rng.pareto(1.5) + 1) * 1.0
        return float(min(10, max(1, round(v))))
    def description(self):
        return "Power law — rare extremes dominate (real-world style)"


# ─── Simple 2-Player Reward Game ──────────────────────────────────

class RewardGame:
    """
    Simple 2-player game:
    - Players alternate turns
    - Each turn, player picks one of N actions
    - Each action has a reward sampled from the given distribution
    - After a fixed number of rounds, player with highest total reward wins

    State = (player_turn, round_number)
    Actions = "pick_0", "pick_1", ..., "pick_{n-1}"
    """

    def __init__(self, sampler: DiceSampler, n_actions: int = 4, n_rounds: int = 8):
        self.sampler = sampler
        self.n_actions = n_actions
        self.n_rounds = n_rounds
        self.reset()

    def reset(self):
        self.current = 'X'
        self.done = False
        self.winner = None
        self.round = 0
        self.scores = {'X': 0.0, 'O': 0.0}
        self.turn_in_round = 0
        self._draw_rewards()

    def _draw_rewards(self):
        """Draw fresh rewards for each action this round."""
        rng = np.random.default_rng()
        self.current_rewards = [self.sampler.sample(rng) for _ in range(self.n_actions)]

    def state(self) -> GameState:
        state_str = f"round={self.round}|turn={self.current}|scoreX={self.scores['X']:.1f}|scoreO={self.scores['O']:.1f}"
        return GameState(state_str, self.round, self.current)

    def legal_actions(self) -> list[str]:
        return [f"pick_{i}" for i in range(self.n_actions)]

    def step(self, action: str) -> tuple[float, bool]:
        idx = int(action.split('_')[1])
        reward = self.current_rewards[idx]
        self.scores[self.current] += reward

        # Switch player
        if self.current == 'X':
            self.current = 'O'
            self.turn_in_round += 1
        else:
            self.current = 'X'
            self.round += 1
            self.turn_in_round = 0
            self._draw_rewards()  # new rewards each round

        # Check game end
        if self.round >= self.n_rounds:
            self.done = True
            if self.scores['X'] > self.scores['O']:
                self.winner = 'X'
            elif self.scores['O'] > self.scores['X']:
                self.winner = 'O'
            else:
                self.winner = 'draw'

        return reward, self.done


# ─── Tile Field Trainer for Reward Games ──────────────────────────

class TileFieldTrainer:
    """Train a tile field on a RewardGame, track convergence."""

    def __init__(self, game: RewardGame, n_simulations: int = 15, evolve_every: int = 5):
        self.game = game
        self.tile_field: dict[str, StateTile] = {}
        self.n_simulations = n_simulations
        self.evolve_every = evolve_every
        self.win_history = []  # 1 if X won, 0 otherwise, per game
        self.stats = {"games": 0, "wins": 0, "losses": 0, "draws": 0}

    def _play_game(self) -> bool:
        """Play one game, return True if X won."""
        self.game.reset()
        history = []

        while not self.game.done:
            state = self.game.state()
            actions = self.game.legal_actions()
            if not actions:
                break

            state_hash = state.hash()

            # Get or create tile
            if state_hash not in self.tile_field:
                self.tile_field[state_hash] = StateTile(
                    state_hash, str(state), actions
                )
            tile = self.tile_field[state_hash]

            # X uses tile field, O plays randomly
            if self.game.current == 'X':
                # Simple epsilon-greedy using tile scores
                scores = {a: tile.reflexes.get(a, {}).get("score", 0.5) for a in actions}
                epsilon = 0.1
                if random.random() < epsilon:
                    action = random.choice(actions)
                else:
                    action = max(scores, key=scores.get)
            else:
                action = random.choice(actions)

            self.game.step(action)
            history.append((state_hash, action))

        # Record outcome
        self.stats["games"] += 1
        won = False
        if self.game.winner == 'X':
            self.stats["wins"] += 1
            won = True
        elif self.game.winner == 'draw':
            self.stats["draws"] += 1
        else:
            self.stats["losses"] += 1

        self.win_history.append(1.0 if won else 0.0)

        # Update tile records
        for state_hash, action in history:
            if state_hash in self.tile_field:
                self.tile_field[state_hash].record(action, won)

        return won

    def evolve(self):
        """Evolve all tiles."""
        for tile in self.tile_field.values():
            tile.evolve()

    def train(self, num_games: int = 500):
        """Train for num_games, evolving periodically."""
        for i in range(num_games):
            self._play_game()
            if (i + 1) % self.evolve_every == 0:
                self.evolve()
        return self

    def convergence_rounds(self, window: int = 50, threshold: float = 0.05) -> int:
        """Rounds until win rate stabilizes (std of last `window` games < threshold)."""
        if len(self.win_history) < window:
            return len(self.win_history)
        for i in range(window, len(self.win_history)):
            recent = self.win_history[i - window:i]
            if np.std(recent) < threshold:
                return i
        return len(self.win_history)

    def conservation_cv(self) -> float:
        """CV of score distribution across all tiles."""
        all_scores = []
        for tile in self.tile_field.values():
            for action, data in tile.reflexes.items():
                if data["chosen"] > 0:
                    all_scores.append(data["score"])
        if len(all_scores) < 2:
            return float('inf')
        arr = np.array(all_scores)
        mean = arr.mean()
        if mean == 0:
            return float('inf')
        return float(arr.std() / mean)

    def negative_space_clarity(self) -> float:
        """Separation between bottom 25% and top 75% scores."""
        all_scores = []
        for tile in self.tile_field.values():
            for action, data in tile.reflexes.items():
                if data["chosen"] > 0:
                    all_scores.append(data["score"])
        if len(all_scores) < 4:
            return 0.0
        arr = np.array(all_scores)
        p25 = np.percentile(arr, 25)
        p75 = np.percentile(arr, 75)
        return float(p75 - p25)


# ─── Run Experiment ───────────────────────────────────────────────

def run_single_experiment(sampler: DiceSampler, seed: int,
                          num_games: int = 500, n_actions: int = 4,
                          n_rounds: int = 8) -> dict:
    """Run one experiment with a given sampler and seed."""
    random.seed(seed)
    np.random.seed(seed)

    game = RewardGame(sampler, n_actions=n_actions, n_rounds=n_rounds)
    trainer = TileFieldTrainer(game, n_simulations=15, evolve_every=5)

    start = time.time()
    trainer.train(num_games=num_games)
    elapsed = time.time() - start

    convergence = trainer.convergence_rounds()
    cv = trainer.conservation_cv()
    clarity = trainer.negative_space_clarity()

    final_wr = trainer.stats["wins"] / max(trainer.stats["games"], 1)

    return {
        "sampler": sampler.name,
        "seed": seed,
        "num_games": num_games,
        "convergence_rounds": convergence,
        "conservation_cv": cv,
        "negative_space_clarity": clarity,
        "final_win_rate": final_wr,
        "num_tiles": len(trainer.tile_field),
        "total_scores_recorded": sum(
            sum(1 for d in t.reflexes.values() if d["chosen"] > 0)
            for t in trainer.tile_field.values()
        ),
        "elapsed_seconds": round(elapsed, 2),
    }


def run_all_flavors(n_seeds: int = 5, num_games: int = 500) -> dict:
    """Run experiment for all 5 distribution flavors across multiple seeds."""
    flavors = [
        UniformDice(),
        WeightedDice(),
        CardDeck(),
        NormalDice(),
        PowerLawDice(),
    ]

    all_results = {}
    summary = {}

    for sampler in flavors:
        print(f"\n{'='*60}")
        print(f"  FLAVOR: {sampler.name}")
        print(f"  {sampler.description()}")
        print(f"{'='*60}")

        seed_results = []
        for seed_idx in range(n_seeds):
            seed = 42 + seed_idx * 1000
            print(f"  Seed {seed}...", end=" ", flush=True)
            result = run_single_experiment(sampler, seed, num_games=num_games)
            seed_results.append(result)
            print(f"conv={result['convergence_rounds']}  cv={result['conservation_cv']:.4f}  "
                  f"clarity={result['negative_space_clarity']:.4f}  wr={result['final_win_rate']:.1%}")

        # Aggregate across seeds
        cvs = [r["conservation_cv"] for r in seed_results]
        convs = [r["convergence_rounds"] for r in seed_results]
        clarities = [r["negative_space_clarity"] for r in seed_results]
        wrs = [r["final_win_rate"] for r in seed_results]

        agg = {
            "description": sampler.description(),
            "convergence": {
                "mean": float(np.mean(convs)),
                "std": float(np.std(convs)),
                "min": int(np.min(convs)),
                "max": int(np.max(convs)),
            },
            "conservation_cv": {
                "mean": float(np.mean(cvs)),
                "std": float(np.std(cvs)),
                "min": float(np.min(cvs)),
                "max": float(np.max(cvs)),
            },
            "negative_space_clarity": {
                "mean": float(np.mean(clarities)),
                "std": float(np.std(clarities)),
                "min": float(np.min(clarities)),
                "max": float(np.max(clarities)),
            },
            "win_rate": {
                "mean": float(np.mean(wrs)),
                "std": float(np.std(wrs)),
            },
        }

        summary[sampler.name] = agg
        all_results[sampler.name] = seed_results

        print(f"  → Convergence: {agg['convergence']['mean']:.0f} ± {agg['convergence']['std']:.0f} rounds")
        print(f"  → Conservation CV: {agg['conservation_cv']['mean']:.4f} ± {agg['conservation_cv']['std']:.4f}")
        print(f"  → Neg Space Clarity: {agg['negative_space_clarity']['mean']:.4f} ± {agg['negative_space_clarity']['std']:.4f}")

    return {"summary": summary, "raw": all_results}


def analyze_results(results: dict) -> str:
    """Analyze and interpret the experiment results."""
    summary = results["summary"]
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("  ANALYSIS: Conservation Law vs Randomness Flavor")
    lines.append("=" * 60)

    # Rank by conservation CV (lower = tighter = stronger conservation)
    by_cv = sorted(summary.items(), key=lambda x: x[1]["conservation_cv"]["mean"])
    lines.append("\n📊 Conservation Strength (CV — lower = stronger):")
    for rank, (name, data) in enumerate(by_cv, 1):
        lines.append(f"  {rank}. {name:12s}  CV = {data['conservation_cv']['mean']:.4f} ± {data['conservation_cv']['std']:.4f}")

    # Rank by convergence speed (lower = faster)
    by_conv = sorted(summary.items(), key=lambda x: x[1]["convergence"]["mean"])
    lines.append("\n⚡ Convergence Speed (rounds — lower = faster):")
    for rank, (name, data) in enumerate(by_conv, 1):
        lines.append(f"  {rank}. {name:12s}  rounds = {data['convergence']['mean']:.0f} ± {data['convergence']['std']:.0f}")

    # Rank by negative space clarity (higher = better separation)
    by_clarity = sorted(summary.items(), key=lambda x: -x[1]["negative_space_clarity"]["mean"])
    lines.append("\n🔍 Negative Space Clarity (higher = better separation):")
    for rank, (name, data) in enumerate(by_clarity, 1):
        lines.append(f"  {rank}. {name:12s}  clarity = {data['negative_space_clarity']['mean']:.4f} ± {data['negative_space_clarity']['std']:.4f}")

    # Hypothesis check
    lines.append("\n🧪 HYPOTHESIS CHECK:")
    best_cv_name = by_cv[0][0]
    worst_cv_name = by_cv[-1][0]
    lines.append(f"  Strongest conservation: {best_cv_name}")
    lines.append(f"  Weakest conservation:   {worst_cv_name}")

    uniform_rank = next(i for i, (n, _) in enumerate(by_cv) if n == "uniform")
    powerlaw_rank = next(i for i, (n, _) in enumerate(by_cv) if n == "power_law")

    if uniform_rank < powerlaw_rank:
        lines.append(f"  ✅ CONFIRMED: Uniform (rank {uniform_rank+1}) has stronger conservation than Power Law (rank {powerlaw_rank+1})")
    else:
        lines.append(f"  ❌ REJECTED: Power Law (rank {powerlaw_rank+1}) has stronger conservation than Uniform (rank {uniform_rank+1})")

    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🎲 Dice Flavors Experiment")
    print("   Testing how randomness distributions affect tile field conservation")
    print()

    results = run_all_flavors(n_seeds=5, num_games=500)

    # Save raw results
    with open("dice-flavors-results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n✅ Results saved to dice-flavors-results.json")

    # Analysis
    analysis = analyze_results(results)
    print(analysis)

    # Save analysis as text
    with open("dice-flavors-analysis.txt", "w") as f:
        f.write(analysis)
    print("✅ Analysis saved to dice-flavors-analysis.txt")
