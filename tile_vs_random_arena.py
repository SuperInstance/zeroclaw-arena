"""
Tile vs Random Arena — A/B test on actual ZeroClaw games.

Player X: Tile-field exploration (Monte Carlo simulation + score evolution)
Player O: Random

Compare with:
Player X: Random exploration (standard ZeroClaw)
Player O: Random

The difference is the value of the tile field.
"""

import random
import numpy as np
import hashlib
import json
import os
import time
from collections import defaultdict


def hash_embed(text, dim=64):
    h = hashlib.blake2b(text.encode(), digest_size=dim).digest()
    v = np.array([b/255.0 for b in h], dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-10)


class TilePlayer:
    """Player that uses tile field exploration."""
    
    def __init__(self, n_simulations=30):
        self.tiles = {}  # state_hash -> {action: {"score": 0.5, "chosen": 0, "won": 0}}
        self.n_simulations = n_simulations
    
    def choose(self, game) -> str:
        state = str(game.state())
        actions = game.legal_actions()
        if not actions:
            return ''
        if len(actions) == 1:
            return actions[0]
        
        state_hash = hashlib.blake2b(state.encode(), digest_size=8).hexdigest()
        
        if state_hash not in self.tiles:
            self.tiles[state_hash] = {a: {"score": 0.5, "chosen": 0, "won": 0, "state": state} for a in actions}
        else:
            for a in actions:
                if a not in self.tiles[state_hash]:
                    self.tiles[state_hash][a] = {"score": 0.5, "chosen": 0, "won": 0, "state": state}
        
        # Monte Carlo: simulate each action
        action_values = {}
        for action in actions:
            wins = 0
            sims = max(1, self.n_simulations // len(actions))
            
            for _ in range(sims):
                # Copy game state by replaying moves
                sim_game = type(game)()
                winner = self._simulate(game, action, sim_game)
                if winner == 'X':
                    wins += 1
            
            # Combine simulation with learned score
            sim_score = wins / max(sims, 1)
            learned_score = self.tiles[state_hash][action]["score"]
            
            n_chosen = self.tiles[state_hash][action]["chosen"]
            confidence = min(n_chosen / 20.0, 0.8)
            
            action_values[action] = confidence * learned_score + (1 - confidence) * sim_score
        
        # Softmax selection
        actions_list = list(action_values.keys())
        values = np.array([action_values[a] for a in actions_list])
        temperature = 0.3
        probs = np.exp(values / temperature)
        probs /= probs.sum()
        
        return np.random.choice(actions_list, p=probs)
    
    def record(self, state: str, action: str, won: bool):
        state_hash = hashlib.blake2b(state.encode(), digest_size=8).hexdigest()
        if state_hash in self.tiles and action in self.tiles[state_hash]:
            self.tiles[state_hash][action]["chosen"] += 1
            if won:
                self.tiles[state_hash][action]["won"] += 1
    
    def evolve(self):
        """Update scores based on win rates."""
        for state_hash, actions in self.tiles.items():
            for action, data in actions.items():
                if data["chosen"] > 0:
                    wr = data["won"] / data["chosen"]
                    data["score"] += 0.05 * (wr - data["score"])
                    data["score"] = max(0.05, min(0.95, data["score"]))
    
    def _simulate(self, real_game, first_action, sim_game):
        """Run a random playout from the current state + first_action."""
        game_copy = type(real_game)()
        
        # Copy board state
        if hasattr(real_game, 'board'):
            game_copy.board = [row[:] for row in real_game.board]
        if hasattr(real_game, 'current'):
            game_copy.current = real_game.current
        if hasattr(real_game, 'done'):
            game_copy.done = real_game.done
        if hasattr(real_game, 'winner'):
            game_copy.winner = real_game.winner
        if hasattr(real_game, 'turn'):
            game_copy.turn = real_game.turn
        
        # Apply first action
        game_copy.step(first_action)
        
        # Play randomly to completion
        while not game_copy.done:
            actions = game_copy.legal_actions()
            if not actions: break
            game_copy.step(random.choice(actions))
        
        return getattr(game_copy, 'winner', None)
    
    @property
    def stats(self):
        total_chosen = sum(d["chosen"] for actions in self.tiles.values() for d in actions.values())
        total_won = sum(d["won"] for actions in self.tiles.values() for d in actions.values())
        return {"tiles": len(self.tiles), "total_decisions": total_chosen, 
                "total_wins_recorded": total_won}


def run_arena():
    from zeroclaw import TicTacToe, Connect4
    
    print("=" * 70)
    print("TILE vs RANDOM ARENA — Direct A/B Test")
    print("=" * 70)
    
    results = {}
    
    for game_name, GameClass in [("tictactoe", TicTacToe), ("connect4", Connect4)]:
        print(f"\n=== {game_name.upper()} ===")
        
        n_games = 500
        evolve_every = 50
        
        # Tile player
        tile_player = TilePlayer(n_simulations=20)
        tile_wins = 0
        
        # Random baseline
        random_wins = 0
        
        for i in range(n_games):
            # Tile player game
            game = GameClass()
            history = []
            
            while not game.done:
                actions = game.legal_actions()
                if not actions: break
                
                if game.current == 'X':
                    state = str(game.state())
                    action = tile_player.choose(game)
                    history.append((state, action))
                else:
                    action = random.choice(actions)
                
                game.step(action)
            
            won = getattr(game, 'winner', None) == 'X'
            if won: tile_wins += 1
            
            for state, action in history:
                tile_player.record(state, action, won)
            
            # Evolve periodically
            if (i + 1) % evolve_every == 0:
                tile_player.evolve()
            
            # Random baseline game
            game = GameClass()
            while not game.done:
                actions = game.legal_actions()
                if not actions: break
                game.step(random.choice(actions))
            if getattr(game, 'winner', None) == 'X':
                random_wins += 1
            
            if (i + 1) % 100 == 0:
                tile_wr = tile_wins / (i + 1)
                rand_wr = random_wins / (i + 1)
                print(f"  Game {i+1}: tile={tile_wr:.1%}, random={rand_wr:.1%}, "
                      f"tiles_known={tile_player.stats['tiles']}")
        
        tile_wr = tile_wins / n_games
        random_wr = random_wins / n_games
        
        print(f"\n  FINAL: tile={tile_wr:.1%}, random={random_wr:.1%}, "
              f"advantage={(tile_wr-random_wr)*100:+.1f}pp")
        
        results[game_name] = {
            "tile_win_rate": tile_wr,
            "random_win_rate": random_wr,
            "advantage_pp": (tile_wr - random_wr) * 100,
            "tiles_learned": tile_player.stats["tiles"],
        }
    
    # Save
    out = os.path.expanduser("~/repos/zeroclaw-arena/tile-vs-random-results.json")
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    run_arena()
