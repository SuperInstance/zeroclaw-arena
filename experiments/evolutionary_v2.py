"""
Evolutionary Strategy v2 — Higher Signal, Harder Game

v1: +6.3pp on TTT with 100-game evals (noisy)
v2: Target >10pp on Connect4 with 300-game evals

Changes:
- 300 games per evaluation (3x less noise)
- Connect4 (harder game, more room for strategy)
- Larger population (50)
- More generations (20)
"""

import random
import numpy as np
import json
import os
import time
from collections import defaultdict


class StrategyGenome:
    def __init__(self, params=None):
        if params is None:
            self.exploration_rate = random.uniform(0.1, 0.9)
            self.temperature = random.uniform(0.1, 5.0)
            self.reward_weight = random.uniform(0.1, 2.0)
            self.center_bonus = random.uniform(0.0, 1.0)
            self.blocking_weight = random.uniform(0.0, 1.0)
            self.random_noise = random.uniform(0.0, 0.3)
        else:
            self.__dict__.update(params)
    
    def mutate(self, rate=0.15):
        new = {}
        for k, v in self.__dict__.items():
            if random.random() < rate:
                new[k] = max(0, min(2, v + random.gauss(0, 0.1)))
            else:
                new[k] = v
        return StrategyGenome(new)
    
    def crossover(self, other):
        child = {}
        for k in self.__dict__:
            child[k] = getattr(self, k) if random.random() < 0.5 else getattr(other, k)
        return StrategyGenome(child)


def evaluate_on_connect4(genome, n_games=300):
    """Evaluate genome on Connect4."""
    from zeroclaw import Connect4
    
    wins = 0
    # Simple Q-learning with genome-controlled hyperparams
    q_values = defaultdict(lambda: defaultdict(float))
    
    for _ in range(n_games):
        game = Connect4()
        state_history = []
        
        while not game.done:
            actions = game.legal_actions()
            if not actions: break
            
            state = str(game.state())
            
            if game.current == 'X':
                # Strategy: weighted combination of Q-values, center bonus, and exploration
                action_scores = {}
                for a in actions:
                    score = q_values[state].get(a, 0) * genome.reward_weight
                    
                    # Center bonus (genome-controlled)
                    try:
                        col = int(a)
                        center_dist = abs(col - 3)
                        score += genome.center_bonus / (1 + center_dist)
                    except: pass
                    
                    action_scores[a] = score
                
                # Exploration
                if random.random() < genome.exploration_rate:
                    action = random.choice(actions)
                else:
                    # Temperature-controlled selection
                    vals = np.array([action_scores.get(a, 0) for a in actions])
                    if genome.temperature > 0.01:
                        vals_shifted = vals - vals.max()  # numerical stability
                        probs = np.exp(vals_shifted / genome.temperature)
                        probs = np.nan_to_num(probs, nan=0.0)
                        s = probs.sum()
                        if s > 0:
                            probs /= s
                        else:
                            probs = np.ones(len(actions)) / len(actions)
                        action = np.random.choice(actions, p=probs)
                    else:
                        action = actions[np.argmax(vals)]
                
                # Random noise
                if random.random() < genome.random_noise:
                    action = random.choice(actions)
            else:
                action = random.choice(actions)
            
            reward, done = game.step(action)
            state_history.append((state, action))
        
        winner = getattr(game, 'winner', None)
        if winner == 'X':
            wins += 1
        
        # Update Q-values
        for state, action in state_history:
            if winner == 'X':
                q_values[state][action] += 0.1
            elif winner == 'O':
                q_values[state][action] -= 0.05
    
    return wins / n_games


def main():
    print("=" * 60)
    print("EVOLUTIONARY STRATEGY v2 — Connect4, 300-game evals")
    print("=" * 60)
    
    # Baseline
    print("\nBaseline (random)...")
    random_g = StrategyGenome({'exploration_rate': 1.0, 'temperature': 1.0, 
                                'reward_weight': 0, 'center_bonus': 0, 
                                'blocking_weight': 0, 'random_noise': 0})
    baseline_wr = evaluate_on_connect4(random_g, 300)
    print(f"  Random: {baseline_wr:.1%}")
    
    # Evolution
    pop_size = 40
    n_gens = 15
    population = [StrategyGenome() for _ in range(pop_size)]
    
    print(f"\nEvolving {pop_size} genomes for {n_gens} generations...")
    history = []
    
    for gen in range(n_gens):
        start = time.perf_counter()
        fitness = [evaluate_on_connect4(g, 300) for g in population]
        elapsed = time.perf_counter() - start
        
        paired = sorted(zip(fitness, population), key=lambda x: -x[0])
        best_wr, avg_wr, worst_wr = paired[0][0], np.mean(fitness), paired[-1][0]
        
        best_params = {k: round(v, 3) for k, v in paired[0][1].__dict__.items()}
        print(f"  Gen {gen:2d}: best={best_wr:.1%} avg={avg_wr:.1%} ({elapsed:.0f}s) {best_params}")
        
        history.append({'gen': gen, 'best': best_wr, 'avg': avg_wr, 'params': best_params})
        
        # Selection + breeding
        n_elite = max(3, pop_size // 5)
        elite = [g for _, g in paired[:n_elite]]
        
        new_pop = list(elite)
        while len(new_pop) < pop_size:
            p1, p2 = random.choice(elite), random.choice(elite)
            child = p1.crossover(p2).mutate()
            new_pop.append(child)
        
        population = new_pop
    
    # Final
    best = StrategyGenome(history[-1]['params'])
    final_wr = evaluate_on_connect4(best, 500)
    print(f"\n=== FINAL ===")
    print(f"  Evolved: {final_wr:.1%} (500 games)")
    print(f"  Random: {baseline_wr:.1%}")
    print(f"  Improvement: {(final_wr - baseline_wr)*100:+.1f}pp")
    
    if final_wr > baseline_wr + 0.10:
        print("  ✅ >10pp improvement!")
    elif final_wr > baseline_wr + 0.05:
        print("  ⚠️ Modest improvement")
    else:
        print("  ❌ No significant improvement")
    
    output = {'history': history, 'final': final_wr, 'baseline': baseline_wr,
              'improvement_pp': (final_wr - baseline_wr) * 100, 'best_params': history[-1]['params']}
    
    with open(os.path.expanduser("~/repos/zeroclaw-arena/evolutionary-v2-results.json"), 'w') as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()
