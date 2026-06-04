"""
GPU-Accelerated Evolutionary Strategy Optimization for ZeroClaw

Instead of random exploration, EVOLVE the exploration parameters:
- Exploration rate (how often to try random vs best-known)
- Temperature (how much to weight rewards)
- Mutation rate (how much to deviate from parent scripts)
- Selection pressure (how many top scripts to breed from)

Each "genome" is a set of learning hyperparameters.
Fitness = win rate after 100 games.
Evolve on GPU using batch evaluation.

Target: Evolved strategy beats random exploration by >10%.
"""

import torch
import numpy as np
import random
import hashlib
import sqlite3
import json
import os
import time
from collections import defaultdict


class StrategyGenome:
    """A set of learning hyperparameters to evolve."""
    def __init__(self, params=None):
        if params is None:
            # Random initialization
            self.exploration_rate = random.uniform(0.1, 0.9)
            self.temperature = random.uniform(0.1, 5.0)
            self.mutation_rate = random.uniform(0.01, 0.3)
            self.selection_pressure = random.uniform(0.1, 0.5)
            self.reward_decay = random.uniform(0.8, 0.99)
            self.action_noise = random.uniform(0.0, 0.5)
        else:
            self.__dict__.update(params)
    
    def to_tensor(self):
        return torch.tensor([
            self.exploration_rate,
            self.temperature,
            self.mutation_rate,
            self.selection_pressure,
            self.reward_decay,
            self.action_noise,
        ], dtype=torch.float32)
    
    @staticmethod
    def from_tensor(t):
        return StrategyGenome({
            'exploration_rate': float(t[0]),
            'temperature': float(t[1]),
            'mutation_rate': float(t[2]),
            'selection_pressure': float(t[3]),
            'reward_decay': float(t[4]),
            'action_noise': float(t[5]),
        })
    
    def mutate(self, rate=0.1):
        new_params = {}
        for key, val in self.__dict__.items():
            if random.random() < rate:
                delta = random.gauss(0, 0.1)
                new_params[key] = max(0, min(1, val + delta))
            else:
                new_params[key] = val
        return StrategyGenome(new_params)
    
    def crossover(self, other):
        child_params = {}
        for key in self.__dict__:
            if random.random() < 0.5:
                child_params[key] = getattr(self, key)
            else:
                child_params[key] = getattr(other, key)
        return StrategyGenome(child_params)


def evaluate_genome_tictactoe(genome, n_games=100):
    """Evaluate a strategy genome by playing tic-tac-toe."""
    from zeroclaw import TicTacToe
    
    wins = 0
    
    # Build a simple Q-table from the genome's parameters
    q_table = {}  # state -> {action -> value}
    
    for game_idx in range(n_games):
        game = TicTacToe()
        history = []
        
        while not game.done:
            actions = game.legal_actions()
            if not actions:
                break
            
            state = str(game.state())
            
            # Initialize Q-values if needed
            if state not in q_table:
                q_table[state] = {a: 0.0 for a in actions}
            
            # Exploration vs exploitation (genome-controlled)
            if random.random() < genome.exploration_rate:
                # Explore
                action = random.choice(actions)
            else:
                # Exploit with temperature
                q_vals = {a: q_table[state].get(a, 0) for a in actions}
                
                # Softmax with temperature
                vals = np.array(list(q_vals.values()))
                if genome.temperature > 0:
                    probs = np.exp(vals / genome.temperature)
                    probs /= probs.sum() + 1e-10
                    action = np.random.choice(actions, p=probs)
                else:
                    action = max(q_vals, key=q_vals.get)
            
            # Add noise (genome-controlled)
            if random.random() < genome.action_noise:
                action = random.choice(actions)
            
            reward, done = game.step(action)
            history.append((state, action, reward))
        
        winner = getattr(game, 'winner', None)
        if winner == 'X':
            wins += 1
        
        # Update Q-table with genome-controlled learning params
        for i, (state, action, reward) in enumerate(history):
            if state not in q_table:
                continue
            # Discounted reward
            future = genome.reward_decay ** (len(history) - i - 1)
            final_reward = reward * future if winner == 'X' else reward * future * -0.5
            q_table[state][action] = q_table[state].get(action, 0) + 0.1 * final_reward
    
    return wins / n_games


def run_evolution():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 60)
    print("EVOLUTIONARY STRATEGY OPTIMIZATION")
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    
    population_size = 30
    generations = 15
    elite_frac = 0.3
    
    # Initialize population
    population = [StrategyGenome() for _ in range(population_size)]
    
    # Baseline: random exploration
    print("\nComputing baseline (random strategy)...")
    baseline_genome = StrategyGenome({
        'exploration_rate': 1.0,  # Always random
        'temperature': 1.0,
        'mutation_rate': 0.1,
        'selection_pressure': 0.3,
        'reward_decay': 0.9,
        'action_noise': 0.0,
    })
    baseline_wr = evaluate_genome_tictactoe(baseline_genome, 200)
    print(f"  Baseline (random): {baseline_wr:.1%}")
    
    # Evolution
    print(f"\nEvolving {population_size} genomes for {generations} generations...")
    history = []
    
    for gen in range(generations):
        start = time.perf_counter()
        
        # Evaluate all genomes
        fitness = []
        for genome in population:
            wr = evaluate_genome_tictactoe(genome, 100)
            fitness.append(wr)
        
        # Sort by fitness
        paired = list(zip(fitness, population))
        paired.sort(key=lambda x: -x[0])
        
        best_wr = paired[0][0]
        avg_wr = np.mean(fitness)
        worst_wr = paired[-1][0]
        elapsed = time.perf_counter() - start
        
        print(f"  Gen {gen:2d}: best={best_wr:.1%} avg={avg_wr:.1%} worst={worst_wr:.1%} ({elapsed:.1f}s)")
        
        history.append({
            'generation': gen,
            'best': best_wr,
            'avg': avg_wr,
            'worst': worst_wr,
            'best_params': {k: v for k, v in paired[0][1].__dict__.items()},
        })
        
        # Selection
        n_elite = max(2, int(population_size * elite_frac))
        elite = [g for _, g in paired[:n_elite]]
        
        # Create next generation
        new_population = list(elite)  # Elitism
        
        while len(new_population) < population_size:
            # Tournament selection
            parent1 = random.choice(elite)
            parent2 = random.choice(elite)
            child = parent1.crossover(parent2)
            child = child.mutate(rate=0.2)
            new_population.append(child)
        
        population = new_population
    
    # Final evaluation
    print("\n=== FINAL EVALUATION ===")
    best_genome = history[-1]['best_params']
    best = StrategyGenome(best_genome)
    final_wr = evaluate_genome_tictactoe(best, 500)
    
    print(f"  Evolved strategy: {final_wr:.1%} (500 games)")
    print(f"  Random baseline: {baseline_wr:.1%}")
    print(f"  Improvement: {(final_wr - baseline_wr)*100:+.1f}pp")
    
    print(f"\n  Evolved parameters:")
    for k, v in best_genome.items():
        print(f"    {k}: {v:.3f}")
    
    if final_wr > baseline_wr + 0.10:
        print(f"\n✅ EVOLUTION WORKS: >10pp improvement over random!")
    elif final_wr > baseline_wr + 0.05:
        print(f"\n⚠️ MODEST IMPROVEMENT: {final_wr - baseline_wr:.1%} better than random")
    else:
        print(f"\n❌ No significant improvement")
    
    # Save
    output = {
        'history': history,
        'final_win_rate': final_wr,
        'baseline_win_rate': baseline_wr,
        'improvement_pp': (final_wr - baseline_wr) * 100,
        'best_genome': best_genome,
    }
    
    out = os.path.expanduser("~/repos/zeroclaw-arena/evolutionary-results.json")
    with open(out, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    run_evolution()
