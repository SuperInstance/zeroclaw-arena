"""
Cooperative Tile Field — Does conservation hold when players cooperate?

All our experiments have been competitive. What happens when two tile fields
WORK TOGETHER toward a shared goal?

Predictions:
- Cooperative = STRONGER conservation (shared signal, no adversarial noise)
- Emergent specialization = players divide the action space
- No divergence (no arms race) — strategies should converge
"""

import random
import numpy as np
import hashlib
import json
import os
from collections import defaultdict

from zeroclaw import TicTacToe, Connect4


class CoopTileField:
    def __init__(self, name="agent"):
        self.name = name
        self.tiles = {}
    
    def get_or_create(self, state_str, legal_actions):
        h = hashlib.blake2b(state_str.encode(), digest_size=8).hexdigest()
        if h not in self.tiles:
            self.tiles[h] = {a: {"score": 0.5, "chosen": 0, "won": 0} for a in legal_actions}
        else:
            for a in legal_actions:
                if a not in self.tiles[h]:
                    self.tiles[h][a] = {"score": 0.5, "chosen": 0, "won": 0}
        return h
    
    def choose(self, h, actions, T=0.3, eps=0.05):
        tile = self.tiles[h]
        if random.random() < eps:
            return min(actions, key=lambda a: tile[a]["chosen"])
        scores = np.array([tile[a]["score"] for a in actions])
        if T > 0.01:
            p = np.exp(scores/T); p /= p.sum()
            return actions[np.random.choice(len(actions), p=p)]
        return actions[np.argmax(scores)]
    
    def record(self, h, action, won):
        if h in self.tiles and action in self.tiles[h]:
            self.tiles[h][action]["chosen"] += 1
            if won: self.tiles[h][action]["won"] += 1
    
    def evolve(self, lr=0.05, cap=0.05):
        for tile in self.tiles.values():
            for d in tile.values():
                if d["chosen"] > 0:
                    wr = d["won"] / d["chosen"]
                    delta = max(-cap, min(cap, lr * (wr - d["score"])))
                    d["score"] = max(0.05, min(0.95, d["score"] + delta))
    
    def share_knowledge(self, other, weight=0.2):
        """Share learned scores with another field (cooperation)."""
        shared = 0
        for h in self.tiles:
            if h in other.tiles:
                for a in self.tiles[h]:
                    if a in other.tiles[h]:
                        # Average scores
                        s1 = self.tiles[h][a]["score"]
                        s2 = other.tiles[h][a]["score"]
                        self.tiles[h][a]["score"] = (1-weight) * s1 + weight * s2
                        other.tiles[h][a]["score"] = weight * s1 + (1-weight) * s2
                        shared += 1
        return shared


def run_cooperative_experiment():
    print("=" * 70)
    print("COOPERATIVE TILE FIELD — Shared Goals vs Competition")
    print("=" * 70)
    
    n_games = 500
    evolve_every = 50
    
    conditions = [
        ("solo", False, False),           # One agent plays both X and O turns
        ("parallel_no_share", False, False),  # Two agents, no knowledge sharing
        ("parallel_share", True, False),      # Two agents WITH knowledge sharing
        ("competitive", False, True),         # Two agents competing
    ]
    
    results = {}
    
    for game_name, GameClass in [("tictactoe", TicTacToe), ("connect4", Connect4)]:
        print(f"\n=== {game_name.upper()} ===")
        
        for cond_name, share, competitive in conditions:
            # Solo: one field plays as X, O is random
            # Parallel: two fields, one for X one for O, cooperating
            # Share: parallel with knowledge transfer
            # Competitive: two fields playing against each other
            
            if cond_name == "solo":
                field = CoopTileField("solo")
                wins = 0
                for _ in range(n_games):
                    game = GameClass()
                    history = []
                    while not game.done:
                        actions = game.legal_actions()
                        if not actions: break
                        if game.current == 'X':
                            h = field.get_or_create(str(game.state()), actions)
                            a = field.choose(h, actions)
                            history.append((h, a))
                        else:
                            a = random.choice(actions)
                        game.step(a)
                    won = getattr(game, 'winner', None) == 'X'
                    if won: wins += 1
                    for h, a in history:
                        field.record(h, a, won)
                    if (_ + 1) % evolve_every == 0:
                        field.evolve()
                
            elif cond_name == "parallel_no_share":
                field_x = CoopTileField("X")
                field_o = CoopTileField("O")
                wins_x = 0
                wins_o = 0
                for _ in range(n_games):
                    game = GameClass()
                    hist_x, hist_o = [], []
                    while not game.done:
                        actions = game.legal_actions()
                        if not actions: break
                        if game.current == 'X':
                            h = field_x.get_or_create(str(game.state()), actions)
                            a = field_x.choose(h, actions)
                            hist_x.append((h, a))
                        else:
                            h = field_o.get_or_create(str(game.state()), actions)
                            a = field_o.choose(h, actions)
                            hist_o.append((h, a))
                        game.step(a)
                    winner = getattr(game, 'winner', None)
                    won_x = winner == 'X'
                    won_o = winner == 'O'
                    if won_x: wins_x += 1
                    if won_o: wins_o += 1
                    for h, a in hist_x: field_x.record(h, a, won_x)
                    for h, a in hist_o: field_o.record(h, a, won_o)
                    if (_ + 1) % evolve_every == 0:
                        field_x.evolve()
                        field_o.evolve()
                
            elif cond_name == "parallel_share":
                field_x = CoopTileField("X")
                field_o = CoopTileField("O")
                wins_x = 0
                wins_o = 0
                shared_count = 0
                for _ in range(n_games):
                    game = GameClass()
                    hist_x, hist_o = [], []
                    while not game.done:
                        actions = game.legal_actions()
                        if not actions: break
                        if game.current == 'X':
                            h = field_x.get_or_create(str(game.state()), actions)
                            a = field_x.choose(h, actions)
                            hist_x.append((h, a))
                        else:
                            h = field_o.get_or_create(str(game.state()), actions)
                            a = field_o.choose(h, actions)
                            hist_o.append((h, a))
                        game.step(a)
                    winner = getattr(game, 'winner', None)
                    won_x = winner == 'X'
                    won_o = winner == 'O'
                    if won_x: wins_x += 1
                    if won_o: wins_o += 1
                    for h, a in hist_x: field_x.record(h, a, won_x)
                    for h, a in hist_o: field_o.record(h, a, won_o)
                    if (_ + 1) % evolve_every == 0:
                        field_x.evolve()
                        field_o.evolve()
                        shared = field_x.share_knowledge(field_o, weight=0.2)
                        shared_count += shared
                
            elif cond_name == "competitive":
                field_a = CoopTileField("A")
                field_b = CoopTileField("B")
                wins_a = 0
                for _ in range(n_games):
                    game = GameClass()
                    hist_a, hist_b = [], []
                    while not game.done:
                        actions = game.legal_actions()
                        if not actions: break
                        if game.current == 'X':
                            h = field_a.get_or_create(str(game.state()), actions)
                            a = field_a.choose(h, actions)
                            hist_a.append((h, a))
                        else:
                            h = field_b.get_or_create(str(game.state()), actions)
                            a = field_b.choose(h, actions)
                            hist_b.append((h, a))
                        game.step(a)
                    winner = getattr(game, 'winner', None)
                    if winner == 'X': wins_a += 1
                    for h, a in hist_a: field_a.record(h, a, winner == 'X')
                    for h, a in hist_b: field_b.record(h, a, winner == 'O')
                    if (_ + 1) % evolve_every == 0:
                        field_a.evolve()
                        field_b.evolve()
            
            # Collect results
            if cond_name == "solo":
                wr = wins / n_games
                all_scores = [d["score"] for t in field.tiles.values() for d in t.values()]
                results[f"{game_name}_{cond_name}"] = {
                    "x_wr": wr,
                    "cv_field": 0,  # Single field
                    "tiles": len(field.tiles),
                    "score_std": np.std(all_scores) if all_scores else 0,
                }
                print(f"  {cond_name:<20s}: X_wr={wr:.1%} tiles={len(field.tiles)}")
            elif cond_name in ["parallel_no_share", "parallel_share"]:
                wr_x = wins_x / n_games
                wr_o = wins_o / n_games
                all_x = [d["score"] for t in field_x.tiles.values() for d in t.values()]
                all_o = [d["score"] for t in field_o.tiles.values() for d in t.values()]
                results[f"{game_name}_{cond_name}"] = {
                    "x_wr": wr_x, "o_wr": wr_o,
                    "tiles": len(field_x.tiles) + len(field_o.tiles),
                    "x_std": np.std(all_x) if all_x else 0,
                    "o_std": np.std(all_o) if all_o else 0,
                    "shared": shared_count if cond_name == "parallel_share" else 0,
                }
                print(f"  {cond_name:<20s}: X={wr_x:.1%} O={wr_o:.1%} tiles={len(field_x.tiles)+len(field_o.tiles)}")
            elif cond_name == "competitive":
                wr_a = wins_a / n_games
                results[f"{game_name}_{cond_name}"] = {
                    "a_wr": wr_a, "b_wr": 1-wr_a,
                    "tiles": len(field_a.tiles) + len(field_b.tiles),
                }
                print(f"  {cond_name:<20s}: A={wr_a:.1%} B={1-wr_a:.1%}")
    
    # Save
    out = os.path.expanduser("~/repos/zeroclaw-arena/cooperative-results.json")
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    run_cooperative_experiment()
