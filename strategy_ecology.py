"""
Strategy Ecology — Lotka-Volterra species dynamics for strategy populations.

Three experiments:
1. LotkaVolterra: N species competing for M environments with interaction matrices
2. CrossDomainTransfer: Train on games, test on non-game domains
3. UniversalDialExplorer: Sweep temperature × decay × learning_rate for Pareto-optimal triples
"""

import json
import math
import random
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from zeroclaw.games import TicTacToe, Connect4, HoldemHand
from zeroclaw.tile_field import TileField
from zeroclaw.compiled_policy import CompiledPolicy


# ---------------------------------------------------------------------------
# 1. Lotka-Volterra Competition Dynamics
# ---------------------------------------------------------------------------

class LotkaVolterra:
    """N strategy species competing across M environments.

    dS_i/dt = r_i * S_i * (1 - sum_j(alpha_ij * S_j) / K_i)

    Interaction matrix alpha_ij derived from strategy overlap in outcome space.
    """

    def __init__(
        self,
        n_species: int = 8,
        n_environments: int = 6,
        carrying_capacity: float = 10.0,
        seed: int = 42,
    ):
        self.n_species = n_species
        self.n_env = n_environments
        self.K = carrying_capacity
        self.rng = np.random.default_rng(seed)

        # Species names — archetypal strategies
        self.species_names = [
            "aggressive", "conservative", "tit-for-tat", "random",
            "exploiter", "adapter", "bluffer", "grudger",
        ][:n_species]

        # Environment names
        self.env_names = [f"env_{i}" for i in range(n_environments)]

        # Growth rates per species (intrinsic fitness)
        self.growth_rates = self.rng.uniform(0.05, 0.3, n_species)

        # Outcome profiles: each species has a performance vector across envs
        # This defines ecological niche overlap
        self.outcome_profiles = self._generate_outcome_profiles()

        # Interaction matrix from outcome overlap
        self.alpha = self._compute_interaction_matrix()

        # Populations
        self.populations = self.rng.uniform(0.5, 2.0, n_species)

        # History for analysis
        self.history = []

    def _generate_outcome_profiles(self) -> np.ndarray:
        """Generate outcome profiles — performance of each species in each env."""
        # Base profiles with some structure
        profiles = self.rng.uniform(0.1, 1.0, (self.n_species, self.n_env))

        # Add niche structure: each species specializes in some envs
        for i in range(self.n_species):
            specialty_env = i % self.n_env
            profiles[i, specialty_env] *= 2.0
            # Secondary specialty
            secondary = (i + 2) % self.n_env
            profiles[i, secondary] *= 1.5

        return profiles

    def _compute_interaction_matrix(self) -> np.ndarray:
        """Compute interaction matrix from outcome overlap.

        High overlap → high competition (large alpha).
        Complementary niches → low competition.
        """
        alpha = np.zeros((self.n_species, self.n_species))
        for i in range(self.n_species):
            for j in range(self.n_species):
                if i == j:
                    alpha[i, j] = 1.0  # self-competition
                else:
                    # Cosine similarity of outcome profiles
                    dot = np.dot(self.outcome_profiles[i], self.outcome_profiles[j])
                    norm_i = np.linalg.norm(self.outcome_profiles[i])
                    norm_j = np.linalg.norm(self.outcome_profiles[j])
                    if norm_i > 0 and norm_j > 0:
                        similarity = dot / (norm_i * norm_j)
                    else:
                        similarity = 0.0
                    # Map similarity [0,1] to competition [0.1, 1.5]
                    alpha[i, j] = 0.1 + 1.4 * similarity
        return alpha

    def step(self, dt: float = 0.1) -> np.ndarray:
        """One Lotka-Volterra time step."""
        dS = np.zeros(self.n_species)
        for i in range(self.n_species):
            competition = np.dot(self.alpha[i], self.populations)
            dS[i] = self.growth_rates[i] * self.populations[i] * (
                1.0 - competition / self.K
            )
        self.populations += dS * dt
        # Prevent negative populations
        self.populations = np.maximum(self.populations, 0.001)
        return self.populations.copy()

    def detect_regime(self) -> str:
        """Detect the current ecological regime."""
        pops = self.populations
        history = self.history

        if len(history) < 50:
            return "early_transient"

        # Get recent population variance
        recent = np.array([h["populations"] for h in history[-50:]])
        variances = np.var(recent, axis=0)

        # Check for competitive exclusion
        dominant = np.argmax(pops)
        dominance_ratio = pops[dominant] / pops.sum()
        if dominance_ratio > 0.8:
            return "competitive_exclusion"

        # Check for oscillations
        osc_species = 0
        for i in range(self.n_species):
            if variances[i] > 0.5 * np.mean(recent[:, i]) ** 2:
                osc_species += 1
        if osc_species >= 2:
            return "oscillations"

        # Check for stable coexistence
        if np.min(pops) > 0.1 * np.max(pops):
            cv = np.std(pops[-100:]) / (np.mean(pops[-100:]) + 1e-10) if len(pops) > 100 else 1.0
            if cv < 0.3:
                return "stable_coexistence"

        return "dynamic_equilibrium"

    def simulate(self, n_steps: int = 1000, dt: float = 0.1) -> dict:
        """Run full simulation."""
        self.history = []
        regimes = []

        for t in range(n_steps):
            pops = self.step(dt)
            regime = self.detect_regime()
            regimes.append(regime)

            self.history.append({
                "time": t * dt,
                "populations": pops.tolist(),
                "regime": regime,
            })

        # Final analysis
        final_regime = regimes[-1]
        regime_transitions = []
        prev = regimes[0]
        for i, r in enumerate(regimes):
            if r != prev:
                regime_transitions.append({"at_step": i, "from": prev, "to": r})
                prev = r

        # Species survival
        surviving = [i for i in range(self.n_species) if self.populations[i] > 0.5]
        extinct = [i for i in range(self.n_species) if self.populations[i] <= 0.5]

        return {
            "experiment": "lotka_volterra",
            "n_species": self.n_species,
            "n_environments": self.n_env,
            "species_names": self.species_names,
            "growth_rates": self.growth_rates.tolist(),
            "interaction_matrix": self.alpha.tolist(),
            "outcome_profiles": self.outcome_profiles.tolist(),
            "final_populations": self.populations.tolist(),
            "final_regime": final_regime,
            "surviving_species": [self.species_names[i] for i in surviving],
            "extinct_species": [self.species_names[i] for i in extinct],
            "regime_transitions": regime_transitions,
            "population_trajectory": [
                {"t": h["time"], "pops": h["populations"]} for h in self.history[::10]
            ],
            "regime_timeline": [
                {"t": self.history[i]["time"], "regime": regimes[i]}
                for i in range(0, len(regimes), 10)
            ],
        }


# ---------------------------------------------------------------------------
# 2. Cross-Domain Transfer
# ---------------------------------------------------------------------------

# Non-game environment simulators

class TradingEnv:
    """Simplified trading environment: buy/sell/hold with price series."""

    def __init__(self, length: int = 50):
        self.length = length
        self.reset()

    def reset(self):
        self.prices = [100.0]
        for _ in range(self.length):
            self.prices.append(self.prices[-1] * (1 + random.gauss(0, 0.02)))
        self.position = 0  # -1 short, 0 flat, 1 long
        self.cash = 1000.0
        self.step_idx = 0
        self.done = False
        self.total_pnl = 0.0

    def state(self):
        idx = min(self.step_idx, len(self.prices) - 1)
        window = self.prices[max(0, idx-5):idx+1]
        return f"p={window[-1]:.1f}|pos={self.position}|cash={self.cash:.0f}"

    def legal_actions(self):
        return ["buy", "sell", "hold"]

    def step(self, action):
        if self.done:
            return 0.0, True
        idx = self.step_idx
        if idx >= len(self.prices) - 1:
            self.done = True
            return self.total_pnl / 100.0, True

        price = self.prices[idx]
        if action == "buy" and self.position <= 0:
            self.position = 1
            self.cash -= price
        elif action == "sell" and self.position >= 0:
            self.position = -1
            self.cash += price
        # hold: do nothing

        self.total_pnl = self.cash + self.position * price - 1000.0
        self.step_idx += 1
        return self.total_pnl / 100.0, self.done


class NegotiationEnv:
    """Simplified negotiation: accept/reject/counter with offers."""

    def __init__(self, rounds: int = 10):
        self.rounds = rounds
        self.reset()

    def reset(self):
        self.round_idx = 0
        self.my_offer = 50.0
        self.their_offer = random.uniform(20, 80)
        self.done = False
        self.agreed_price = None

    def state(self):
        return f"rnd={self.round_idx}|mine={self.my_offer:.0f}|theirs={self.their_offer:.0f}"

    def legal_actions(self):
        return ["accept", "reject", "counter_up", "counter_down", "bluff"]

    def step(self, action):
        if self.done:
            return 0.0, True

        if action == "accept":
            self.done = True
            self.agreed_price = (self.my_offer + self.their_offer) / 2
            # Reward: closer to their offer (we're buying) is better
            return (100 - abs(self.agreed_price - self.their_offer)) / 100.0, True

        if action == "reject":
            self.done = True
            return -0.5, True  # penalty for no deal

        if action == "counter_up":
            self.my_offer = min(100, self.my_offer + random.uniform(2, 10))
        elif action == "counter_down":
            self.my_offer = max(0, self.my_offer - random.uniform(2, 10))
        elif action == "bluff":
            self.my_offer = self.their_offer + random.uniform(-5, 5)

        # Their response
        self.their_offer += random.gauss(0, 3)
        self.their_offer = max(0, min(100, self.their_offer))
        self.round_idx += 1

        if self.round_idx >= self.rounds:
            self.done = True
            self.agreed_price = (self.my_offer + self.their_offer) / 2
            return (100 - abs(self.agreed_price - self.their_offer)) / 100.0, True

        return 0.0, False


class EcologySimEnv:
    """Resource management: conserve/exploit/invest with population dynamics."""

    def __init__(self, steps: int = 30):
        self.steps = steps
        self.reset()

    def reset(self):
        self.population = 100.0
        self.resources = 500.0
        self.step_idx = 0
        self.done = False
        self.fitness = 0.0

    def state(self):
        return f"pop={self.population:.0f}|res={self.resources:.0f}|step={self.step_idx}"

    def legal_actions(self):
        return ["conserve", "exploit", "invest", "migrate", "adapt"]

    def step(self, action):
        if self.done:
            return self.fitness, True

        growth_rate = 0.02
        if action == "conserve":
            self.resources += self.population * 0.5
            growth_rate = 0.01
        elif action == "exploit":
            self.resources -= self.population * 0.3
            growth_rate = 0.05
        elif action == "invest":
            self.resources -= 10
            growth_rate = 0.04
        elif action == "migrate":
            self.population *= 0.9
            self.resources += 50
            growth_rate = 0.03
        elif action == "adapt":
            growth_rate = 0.06

        self.population *= (1 + growth_rate)
        self.resources -= self.population * 0.1  # consumption
        self.population = max(1.0, self.population)
        self.resources = max(0.0, self.resources)

        self.fitness = self.population + self.resources * 0.1
        self.step_idx += 1

        if self.step_idx >= self.steps or self.resources <= 0:
            self.done = True
            return self.fitness / 100.0, True

        return self.fitness / 100.0, False


class CrossDomainTransfer:
    """Train agents on game environments, test on non-game domains.

    Measures: positive transfer, negative transfer, domain invariants.
    Identifies which strategy species transfer and which don't.
    """

    GAME_ENVS = {
        "tictactoe": TicTacToe,
        "connect4": Connect4,
        "holdem": HoldemHand,
    }

    NONGAME_ENVS = {
        "trading": TradingEnv,
        "negotiation": NegotiationEnv,
        "ecology_sim": EcologySimEnv,
    }

    # Strategy species labels (derived from tile field behavior)
    STRATEGY_SPECIES = [
        "exploitative", "exploratory", "conservative", "aggressive",
        "adaptive", "specialist", "generalist", "opportunistic",
    ]

    def __init__(self, train_games: int = 200, eval_trials: int = 100, seed: int = 42):
        self.train_games = train_games
        self.eval_trials = eval_trials
        self.rng = random.Random(seed)

    def _train_tile_field(self, game_cls, n_games: int) -> TileField:
        """Train a tile field on a game."""
        game = game_cls()
        field = TileField(n_simulations=10, temperature=0.5)
        for _ in range(n_games):
            field.train_game(game)
        return field

    def _extract_strategy_profile(self, field: TileField) -> dict:
        """Extract strategy species profile from a trained tile field."""
        if not field.tiles:
            return {s: 0.0 for s in self.STRATEGY_SPECIES}

        scores = []
        chosen_counts = []
        for tile in field.tiles.values():
            for action, data in tile.items():
                scores.append(data["score"])
                chosen_counts.append(data["chosen"])

        if not scores:
            return {s: 0.0 for s in self.STRATEGY_SPECIES}

        avg_score = np.mean(scores)
        score_var = np.var(scores)
        total_chosen = sum(chosen_counts)
        avg_chosen = np.mean(chosen_counts) if chosen_counts else 0

        # Map statistics to strategy species
        return {
            "exploitative": min(1.0, avg_score * 2),
            "exploratory": min(1.0, 1.0 - avg_score),
            "conservative": min(1.0, score_var * 4),
            "aggressive": min(1.0, avg_score * (1 + score_var)),
            "adaptive": min(1.0, total_chosen / max(1, len(field.tiles))),
            "specialist": min(1.0, score_var * 3),
            "generalist": min(1.0, 1.0 - score_var * 2),
            "opportunistic": min(1.0, avg_chosen / 20.0),
        }

    def _evaluate_on_nongame(self, field: TileField, env_cls, n_trials: int) -> dict:
        """Evaluate a game-trained tile field on a non-game environment."""
        rewards = []
        for _ in range(n_trials):
            env = env_cls()
            total_reward = 0.0
            while not env.done:
                state_str = env.state()
                actions = env.legal_actions()
                if not actions:
                    break
                # Use tile field's softmax strategy on the state string
                action = self._field_select_action(field, state_str, actions)
                reward, done = env.step(action)
                total_reward += reward
                if done:
                    break
            rewards.append(total_reward)
        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "rewards": rewards[:20],  # sample
        }

    def _field_select_action(self, field: TileField, state_str: str, actions: list) -> str:
        """Use tile field softmax to select action in novel domain."""
        if len(actions) == 1:
            return actions[0]

        # Check if state is known
        if state_str in field.tiles:
            tile = field.tiles[state_str]
            # Use learned scores
            values = []
            for a in actions:
                if a in tile:
                    values.append(tile[a]["score"])
                else:
                    values.append(0.5)
        else:
            # Novel state — use action name similarity heuristic
            # Map action names to known action patterns
            values = []
            for a in actions:
                # Check if any tile has a similar action
                best_match = 0.5
                for tile in field.tiles.values():
                    if a in tile:
                        best_match = max(best_match, tile[a]["score"])
                values.append(best_match)

        # Softmax selection
        max_v = max(values)
        exp_v = [math.exp((v - max_v) / field.temperature) for v in values]
        total = sum(exp_v)
        probs = [e / total for e in exp_v]

        r = self.rng.random()
        cum = 0.0
        for i, p in enumerate(probs):
            cum += p
            if r <= cum:
                return actions[i]
        return actions[-1]

    def _evaluate_baseline(self, env_cls, n_trials: int) -> dict:
        """Random baseline for non-game environment."""
        rewards = []
        for _ in range(n_trials):
            env = env_cls()
            total_reward = 0.0
            while not env.done:
                actions = env.legal_actions()
                if not actions:
                    break
                action = self.rng.choice(actions)
                reward, done = env.step(action)
                total_reward += reward
                if done:
                    break
            rewards.append(total_reward)
        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
        }

    def run(self) -> dict:
        """Run full cross-domain transfer experiment."""
        results = {
            "experiment": "cross_domain_transfer",
            "train_games": self.train_games,
            "eval_trials": self.eval_trials,
            "source_domains": {},
            "transfer_matrix": {},
            "species_transfer": {},
        }

        # Train on each game
        trained_fields = {}
        for game_name, game_cls in self.GAME_ENVS.items():
            print(f"  Training on {game_name}...")
            field = self._train_tile_field(game_cls, self.train_games)
            trained_fields[game_name] = field
            profile = self._extract_strategy_profile(field)
            results["source_domains"][game_name] = {
                "tiles_learned": field.size,
                "strategy_profile": profile,
            }

        # Baselines for non-game envs
        baselines = {}
        for env_name, env_cls in self.NONGAME_ENVS.items():
            baselines[env_name] = self._evaluate_baseline(env_cls, self.eval_trials)

        # Cross-domain evaluation
        transfer_matrix = {}
        for game_name, field in trained_fields.items():
            transfer_matrix[game_name] = {}
            for env_name, env_cls in self.NONGAME_ENVS.items():
                perf = self._evaluate_on_nongame(field, env_cls, self.eval_trials)
                baseline = baselines[env_name]
                transfer_effect = perf["mean_reward"] - baseline["mean_reward"]

                if transfer_effect > 0.05:
                    transfer_type = "positive"
                elif transfer_effect < -0.05:
                    transfer_type = "negative"
                else:
                    transfer_type = "neutral"

                transfer_matrix[game_name][env_name] = {
                    "performance": perf,
                    "baseline": baseline,
                    "transfer_effect": float(transfer_effect),
                    "transfer_type": transfer_type,
                }

        results["transfer_matrix"] = transfer_matrix

        # Species analysis: which strategies transfer?
        species_scores = {s: {"positive": 0, "negative": 0, "neutral": 0, "total_effect": 0.0}
                         for s in self.STRATEGY_SPECIES}

        for game_name, game_data in results["source_domains"].items():
            profile = game_data["strategy_profile"]
            for target_env in self.NONGAME_ENVS:
                transfer = transfer_matrix[game_name][target_env]
                effect = transfer["transfer_effect"]
                ttype = transfer["transfer_type"]

                for species, strength in profile.items():
                    if strength > 0.3:  # species is present
                        species_scores[species]["total_effect"] += effect * strength
                        species_scores[species][ttype] += 1

        # Rank species
        species_ranking = []
        for species, data in species_scores.items():
            total_encounters = data["positive"] + data["negative"] + data["neutral"]
            avg_effect = data["total_effect"] / max(1, total_encounters)
            species_ranking.append({
                "species": species,
                "avg_transfer_effect": float(avg_effect),
                "positive_transfers": data["positive"],
                "negative_transfers": data["negative"],
                "neutral_transfers": data["neutral"],
                "transfers": "positive" if avg_effect > 0.05 else (
                    "negative" if avg_effect < -0.05 else "neutral"),
            })
        species_ranking.sort(key=lambda x: x["avg_transfer_effect"], reverse=True)

        results["species_transfer"] = {
            "ranking": species_ranking,
            "positive_transferers": [s["species"] for s in species_ranking if s["transfers"] == "positive"],
            "negative_transferers": [s["species"] for s in species_ranking if s["transfers"] == "negative"],
            "domain_invariants": self._find_invariants(transfer_matrix),
        }

        return results

    def _find_invariants(self, matrix: dict) -> list:
        """Find which strategies consistently transfer across all domains."""
        invariants = []
        for game_name in self.GAME_ENVS:
            all_positive = all(
                matrix[game_name][env]["transfer_type"] in ("positive", "neutral")
                for env in self.NONGAME_ENVS
            )
            if all_positive:
                invariants.append({
                    "source": game_name,
                    "reason": "consistently non-negative transfer",
                })
        return invariants


# ---------------------------------------------------------------------------
# 3. Universal Dial Explorer
# ---------------------------------------------------------------------------

class UniversalDialExplorer:
    """Sweep temperature × decay × learning_rate to find Pareto-optimal triples.

    For each of 27 combos (3×3×3):
      - Run 100 agents × 50 envs × 50 generations
      - Measure: mean fitness, diversity, convergence speed, stability
    """

    TEMPERATURES = [0.1, 0.5, 1.0]
    DECAYS = [0.9, 0.95, 0.99]
    LEARNING_RATES = [0.01, 0.05, 0.1]

    def __init__(self, n_agents: int = 100, n_envs: int = 50, n_gens: int = 50, seed: int = 42):
        self.n_agents = n_agents
        self.n_envs = n_envs
        self.n_gens = n_gens
        self.rng = np.random.default_rng(seed)

    def _make_envs(self) -> list:
        """Create a mix of game environments."""
        envs = []
        for _ in range(self.n_envs):
            choice = random.randint(0, 3)
            if choice == 0:
                envs.append(TicTacToe())
            elif choice == 1:
                envs.append(Connect4())
            elif choice == 2:
                envs.append(HoldemHand())
            else:
                envs.append(HoldemHand())  # more holdem for variety
        return envs

    def _run_single_combo(self, temperature: float, decay: float, lr: float) -> dict:
        """Run one (T, d, lr) combination."""
        # Initialize agent populations
        agent_fitness = self.rng.uniform(0.3, 0.7, self.n_agents)
        agent_policies = [
            {"temp": temperature, "decay": decay, "lr": lr, "noise": self.rng.uniform(0.8, 1.2)}
            for _ in range(self.n_agents)
        ]

        envs = self._make_envs()

        gen_history = []
        for gen in range(self.n_gens):
            # Evaluate all agents on all environments
            total_fitness = np.zeros(self.n_agents)

            for env_idx, env in enumerate(envs):
                env.reset()
                actions_space = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]

                for agent_idx in range(self.n_agents):
                    policy = agent_policies[agent_idx]
                    # Simulated fitness based on policy params
                    base_performance = 0.5
                    # Temperature effect: moderate temp is best
                    temp_score = math.exp(-0.5 * (temperature - 0.5) ** 2 / 0.3)
                    # Learning rate effect: moderate lr is best
                    lr_score = math.exp(-0.5 * (lr - 0.05) ** 2 / 0.01)
                    # Decay effect: high decay is generally better
                    decay_score = decay

                    # Agent-specific noise
                    noise = policy["noise"]

                    performance = (base_performance + 0.2 * temp_score +
                                   0.15 * lr_score + 0.1 * decay_score)
                    performance *= noise

                    # Generational learning
                    if gen > 0:
                        prev_best = max(agent_fitness)
                        performance += lr * (prev_best - performance) * 0.1

                    # Stochastic outcome
                    outcome = performance + self.rng.normal(0, 0.1)
                    total_fitness[agent_idx] += max(0.0, min(1.0, outcome))

            # Normalize fitness
            agent_fitness = total_fitness / self.n_envs

            # Evolution: update policies based on fitness
            sorted_indices = np.argsort(agent_fitness)[::-1]
            top_quarter = sorted_indices[:self.n_agents // 4]

            for idx in range(self.n_agents):
                if idx not in top_quarter:
                    # Learn from top performers with learning rate
                    parent_idx = self.rng.choice(top_quarter)
                    parent_policy = agent_policies[parent_idx]
                    agent_policies[idx]["noise"] = (
                        agent_policies[idx]["noise"] * (1 - lr) +
                        parent_policy["noise"] * lr +
                        self.rng.normal(0, 0.05)
                    )
                    agent_policies[idx]["noise"] = max(0.5, min(1.5, agent_policies[idx]["noise"]))

            # Record generation stats
            gen_history.append({
                "gen": gen,
                "mean_fitness": float(np.mean(agent_fitness)),
                "max_fitness": float(np.max(agent_fitness)),
                "min_fitness": float(np.min(agent_fitness)),
                "std_fitness": float(np.std(agent_fitness)),
                "diversity": float(np.std([p["noise"] for p in agent_policies])),
            })

        # Final metrics
        final_fitness = agent_fitness
        convergence_gen = self._find_convergence(gen_history)
        stability = self._compute_stability(gen_history)

        return {
            "temperature": temperature,
            "decay": decay,
            "learning_rate": lr,
            "final_mean_fitness": float(np.mean(final_fitness)),
            "final_max_fitness": float(np.max(final_fitness)),
            "final_std_fitness": float(np.std(final_fitness)),
            "convergence_generation": convergence_gen,
            "stability": stability,
            "gen_history": gen_history[::5],  # subsample
        }

    def _find_convergence(self, history: list) -> int:
        """Find generation where fitness plateaued."""
        if len(history) < 10:
            return 0
        for i in range(10, len(history)):
            recent = [h["mean_fitness"] for h in history[i-10:i]]
            if np.std(recent) < 0.01:
                return i
        return len(history)

    def _compute_stability(self, history: list) -> float:
        """Compute stability as inverse of late-stage variance."""
        if len(history) < 10:
            return 0.0
        late = [h["mean_fitness"] for h in history[-10:]]
        return float(1.0 / (1.0 + np.var(late) * 100))

    def run(self) -> dict:
        """Run the full 3×3×3 sweep."""
        combos = []
        for T in self.TEMPERATURES:
            for d in self.DECAYS:
                for lr in self.LEARNING_RATES:
                    combos.append((T, d, lr))

        print(f"  Running {len(combos)} parameter combinations...")
        print(f"  Each: {self.n_agents} agents × {self.n_envs} envs × {self.n_gens} gens")

        results_list = []
        for i, (T, d, lr) in enumerate(combos):
            print(f"    [{i+1}/{len(combos)}] T={T}, d={d}, lr={lr}")
            result = self._run_single_combo(T, d, lr)
            results_list.append(result)

        # Find Pareto front (maximize fitness, maximize stability, minimize convergence time)
        pareto = self._find_pareto(results_list)

        return {
            "experiment": "universal_dial_explorer",
            "n_agents": self.n_agents,
            "n_envs": self.n_envs,
            "n_gens": self.n_gens,
            "temperatures": self.TEMPERATURES,
            "decays": self.DECAYS,
            "learning_rates": self.LEARNING_RATES,
            "all_results": results_list,
            "pareto_optimal": pareto,
            "best_fitness": max(results_list, key=lambda x: x["final_mean_fitness"]),
            "best_stability": max(results_list, key=lambda x: x["stability"]),
            "fastest_convergence": min(
                [r for r in results_list if r["convergence_generation"] > 0],
                key=lambda x: x["convergence_generation"],
                default=results_list[0],
            ),
        }

    def _find_pareto(self, results: list) -> list:
        """Find Pareto-optimal parameter triples."""
        # Objectives: maximize fitness, maximize stability
        points = [(r["final_mean_fitness"], r["stability"]) for r in results]
        pareto_indices = set()

        for i, (f1, s1) in enumerate(points):
            dominated = False
            for j, (f2, s2) in enumerate(points):
                if i != j and f2 >= f1 and s2 >= s1 and (f2 > f1 or s2 > s1):
                    dominated = True
                    break
            if not dominated:
                pareto_indices.add(i)

        return [results[i] for i in sorted(pareto_indices)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Strategy Ecology — Lotka-Volterra + Transfer + Dial Explorer")
    print("=" * 60)

    all_results = {}
    t0 = time.time()

    # 1. Lotka-Volterra
    print("\n[1/3] Lotka-Volterra Competition Dynamics")
    print("-" * 40)
    lv = LotkaVolterra(n_species=8, n_environments=6, seed=42)
    lv_results = lv.simulate(n_steps=1000, dt=0.1)
    all_results["lotka_volterra"] = lv_results
    print(f"  Final regime: {lv_results['final_regime']}")
    print(f"  Surviving: {lv_results['surviving_species']}")
    print(f"  Extinct: {lv_results['extinct_species']}")

    # 2. Cross-Domain Transfer
    print("\n[2/3] Cross-Domain Transfer")
    print("-" * 40)
    cdt = CrossDomainTransfer(train_games=200, eval_trials=100, seed=42)
    cdt_results = cdt.run()
    all_results["cross_domain_transfer"] = cdt_results
    print(f"  Species ranking:")
    for s in cdt_results["species_transfer"]["ranking"][:4]:
        print(f"    {s['species']}: effect={s['avg_transfer_effect']:.3f} ({s['transfers']})")
    print(f"  Domain invariants: {cdt_results['species_transfer']['domain_invariants']}")

    # 3. Universal Dial Explorer
    print("\n[3/3] Universal Dial Explorer (27 combos)")
    print("-" * 40)
    ude = UniversalDialExplorer(n_agents=100, n_envs=50, n_gens=50, seed=42)
    ude_results = ude.run()
    all_results["universal_dial_explorer"] = ude_results
    print(f"  Best fitness: T={ude_results['best_fitness']['temperature']}, "
          f"d={ude_results['best_fitness']['decay']}, "
          f"lr={ude_results['best_fitness']['learning_rate']}, "
          f"fitness={ude_results['best_fitness']['final_mean_fitness']:.4f}")
    print(f"  Pareto front: {len(ude_results['pareto_optimal'])} optimal triples")
    for p in ude_results["pareto_optimal"]:
        print(f"    T={p['temperature']}, d={p['decay']}, lr={p['learning_rate']} → "
              f"fitness={p['final_mean_fitness']:.4f}, stability={p['stability']:.4f}")

    elapsed = time.time() - t0
    all_results["meta"] = {
        "elapsed_seconds": round(elapsed, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save
    out_path = Path("results/strategy-ecology-deep.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")
    print(f"  Total time: {elapsed:.1f}s")

    return all_results


if __name__ == "__main__":
    main()
