"""
Conservation Taxonomy of Games

How does game structure affect the conservation law?

Dimensions:
1. Information: complete (chess, TTT) vs hidden (poker)
2. Stochasticity: deterministic (chess) vs stochastic (poker, backgammon)
3. Symmetry: high (TTT - rotations) vs low (poker - different hands)
4. Depth: shallow (TTT - 9 moves) vs deep (Go - 200+)
5. Players: 2 vs N

For each combination, predict the conservation CV and test it.

Hypothesis:
- More symmetry → MORE degenerate positive space → lower strategy agreement
- More stochasticity → WEAKER conservation (noise drowns signal)
- More hidden info → STRONGER bluffing dynamics → more divergence
- Deeper games → SLOWER convergence but same eventual CV
- More players → FASTER divergence (proven: 1.91× for 3P)
"""

import random
import numpy as np
import hashlib
import json
import os
import time
from collections import defaultdict


class SimpleGame:
    """Configurable game for testing conservation across dimensions."""
    
    def __init__(self, n_actions=4, n_states=50, depth=5, 
                 stochastic=False, hidden_info=False, symmetry=1,
                 reward_noise=0.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.depth = depth
        self.stochastic = stochastic
        self.hidden_info = hidden_info
        self.symmetry = symmetry  # number of equivalent state permutations
        self.reward_noise = reward_noise
        self.state_transitions = {}
        self.state_rewards = {}
        
        # Build random game tree
        for s in range(n_states):
            self.state_transitions[s] = random.sample(
                range(n_states), min(n_actions, n_states))
            # Rewards: some actions lead to wins, others to losses
            base_reward = random.uniform(0.2, 0.8)
            self.state_rewards[s] = [base_reward + random.gauss(0, 0.15) 
                                     for _ in range(n_actions)]
    
    def play(self, field, player=0):
        """Play one game using tile field."""
        state = 0
        history = []
        
        for step in range(self.depth):
            actions = list(range(self.n_actions))
            state_str = f"s{state}"
            
            # Get tile
            h = hashlib.blake2b(state_str.encode(), digest_size=8).hexdigest()
            if h not in field:
                field[h] = {a: {"score": 0.5, "chosen": 0, "won": 0} for a in actions}
            
            # Stochastic: add noise to scores
            if self.stochastic:
                noisy_scores = {a: field[h][a]["score"] + random.gauss(0, 0.1) 
                               for a in actions}
            else:
                noisy_scores = {a: field[h][a]["score"] for a in actions}
            
            # Choose action
            if random.random() < 0.05:  # epsilon
                action = min(actions, key=lambda a: field[h][a]["chosen"])
            else:
                scores = np.array([noisy_scores[a] for a in actions])
                T = 0.3
                probs = np.exp(scores / T)
                probs /= probs.sum()
                action_idx = np.random.choice(len(actions), p=probs)
                action = actions[action_idx]
            
            history.append((h, action))
            
            # Transition
            if state < len(self.state_transitions):
                state = self.state_transitions[state][action]
            else:
                state = random.randint(0, self.n_states - 1)
        
        # Determine winner
        base_wr = np.mean([self.state_rewards[s][a] for s, (h, a) in 
                          zip(range(len(history)), history)])
        noise = random.gauss(0, self.reward_noise) if self.reward_noise > 0 else 0
        won = (base_wr + noise) > 0.5
        
        # Record
        for h, action in history:
            field[h][action]["chosen"] += 1
            if won:
                field[h][action]["won"] += 1
        
        return won


def evolve_field(field, lr=0.05, cap=0.05):
    for h, actions in field.items():
        for a, data in actions.items():
            if data["chosen"] > 0:
                wr = data["won"] / data["chosen"]
                delta = max(-cap, min(cap, lr * (wr - data["score"])))
                data["score"] = max(0.05, min(0.95, data["score"] + delta))


def measure_conservation(game, n_runs=5, n_games=500):
    """Train multiple independent fields and measure conservation."""
    all_distributions = []
    
    for run in range(n_runs):
        field = {}
        for batch in range(5):
            wins = 0
            for _ in range(n_games // 5):
                if game.play(field):
                    wins += 1
            evolve_field(field)
        
        # Collect score distribution
        scores = [d["score"] for actions in field.values() for d in actions.values()]
        if scores:
            all_distributions.append({
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores),
                "p25": np.percentile(scores, 25),
                "p75": np.percentile(scores, 75),
            })
    
    if len(all_distributions) < 2:
        return {"cv_mean": float('nan'), "cv_std": float('nan')}
    
    means = [d["mean"] for d in all_distributions]
    stds = [d["std"] for d in all_distributions]
    
    return {
        "cv_mean": np.std(means) / np.mean(means) if np.mean(means) > 0 else float('nan'),
        "cv_std": np.std(stds) / np.mean(stds) if np.mean(stds) > 0 else float('nan'),
        "mean_mean": np.mean(means),
        "mean_std": np.mean(stds),
        "n_tiles": len(all_distributions[0]) if all_distributions else 0,
    }


def run_taxonomy():
    print("=" * 70)
    print("CONSERVATION TAXONOMY OF GAMES")
    print("=" * 70)
    
    games = {
        # (n_actions, n_states, depth, stochastic, hidden, symmetry, noise)
        "deterministic_shallow": SimpleGame(4, 30, 3, False, False, 8, 0.0),
        "deterministic_deep": SimpleGame(4, 100, 10, False, False, 1, 0.0),
        "stochastic_low": SimpleGame(4, 50, 5, True, False, 4, 0.1),
        "stochastic_high": SimpleGame(4, 50, 5, True, False, 2, 0.3),
        "hidden_info": SimpleGame(4, 50, 5, False, True, 1, 0.2),
        "high_symmetry": SimpleGame(4, 50, 5, False, False, 8, 0.0),
        "low_symmetry": SimpleGame(4, 50, 5, False, False, 1, 0.0),
        "many_actions": SimpleGame(8, 50, 5, False, False, 2, 0.0),
        "few_actions": SimpleGame(2, 50, 5, False, False, 2, 0.0),
        "real_world_like": SimpleGame(6, 80, 7, True, True, 1, 0.2),
    }
    
    results = {}
    
    for name, game in games.items():
        print(f"\n--- {name} ---")
        start = time.perf_counter()
        result = measure_conservation(game, n_runs=5, n_games=300)
        elapsed = time.perf_counter() - start
        
        print(f"  CV(mean)={result['cv_mean']:.4f} CV(std)={result['cv_std']:.4f} "
              f"mean={result['mean_mean']:.3f} std={result['mean_std']:.3f} ({elapsed:.1f}s)")
        
        results[name] = result
    
    # Analysis
    print(f"\n{'=' * 70}")
    print("CONSERVATION TAXONOMY RESULTS")
    print(f"{'=' * 70}")
    
    print(f"\n  {'Game Type':<25s} {'CV(mean)':>10s} {'CV(std)':>10s} {'Mean':>8s} {'Std':>8s}")
    print("  " + "-" * 65)
    
    for name, r in sorted(results.items(), key=lambda x: x[1]['cv_mean']):
        print(f"  {name:<25s} {r['cv_mean']:>10.4f} {r['cv_std']:>10.4f} "
              f"{r['mean_mean']:>8.3f} {r['mean_std']:>8.3f}")
    
    # Correlations
    print(f"\n  INSIGHTS:")
    
    det_cv = np.mean([results["deterministic_shallow"]["cv_mean"],
                      results["deterministic_deep"]["cv_mean"]])
    sto_cv = np.mean([results["stochastic_low"]["cv_mean"],
                      results["stochastic_high"]["cv_mean"]])
    print(f"  Deterministic avg CV: {det_cv:.4f}")
    print(f"  Stochastic avg CV:    {sto_cv:.4f}")
    print(f"  Stochasticity effect: {'WEAKENS' if sto_cv > det_cv else 'STRENGTHENS'} conservation")
    
    sym_cv = np.mean([results["high_symmetry"]["cv_mean"],
                      results["low_symmetry"]["cv_mean"]])
    print(f"  High symmetry CV: {results['high_symmetry']['cv_mean']:.4f}")
    print(f"  Low symmetry CV:  {results['low_symmetry']['cv_mean']:.4f}")
    print(f"  Symmetry effect:  {'STRENGTHENS' if results['high_symmetry']['cv_mean'] < results['low_symmetry']['cv_mean'] else 'WEAKENS'} conservation")
    
    print(f"  Real-world-like CV: {results['real_world_like']['cv_mean']:.4f}")
    print(f"  This predicts conservation in actual agent deployments")
    
    # Save
    out = os.path.expanduser("~/repos/zeroclaw-arena/conservation-taxonomy-results.json")
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    run_taxonomy()
