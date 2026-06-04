"""
Reflex Evolution via Stochastic Simulation

The tile field has scored reflexes (probabilities of what to do next).
We evolve those scores by:
1. Running N stochastic simulations from each tile
2. Measuring which reflex-led paths achieve the best outcomes
3. Adjusting scores to favor high-outcome reflexes
4. Adding noise (mutations) to explore the score space
5. Repeating

This is the algorithmic version of "learning from negative space":
reflexes that lead to bad outcomes get their scores reduced (not eliminated —
they stay in the distribution as negative knowledge).
"""

import json
import random
import numpy as np
import time
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any
from collections import defaultdict


@dataclass
class Reflex:
    name: str
    action: str
    score: float
    cost: float
    conditions: Dict[str, Any] = field(default_factory=dict)
    times_chosen: int = 0
    times_succeeded: int = 0
    
    @property
    def success_rate(self) -> float:
        return self.times_succeeded / max(self.times_chosen, 1)
    
    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class Tile:
    id: str
    type: str
    state: Dict[str, Any] = field(default_factory=dict)
    reflexes: List[Reflex] = field(default_factory=list)
    neighbors: List[str] = field(default_factory=list)
    momentum: float = 0.0
    entropy: float = 1.0
    
    def to_json(self) -> dict:
        return {
            "id": self.id, "type": self.type, "state": self.state,
            "reflexes": [r.to_json() for r in self.reflexes],
            "neighbors": self.neighbors, "momentum": self.momentum, "entropy": self.entropy,
        }


class TileField:
    """A field of tiles for the evolutionary experiment."""
    
    def __init__(self):
        self.tiles: Dict[str, Tile] = {}
        self.outcome_history: List[dict] = []
    
    def add_tile(self, tile: Tile):
        self.tiles[tile.id] = tile
    
    def connect(self, a: str, b: str):
        if a in self.tiles and b in self.tiles:
            if b not in self.tiles[a].neighbors:
                self.tiles[a].neighbors.append(b)
            if a not in self.tiles[b].neighbors:
                self.tiles[b].neighbors.append(a)
    
    def simulate_path(self, start: str, t_minus: float, max_depth: int = 8) -> dict:
        """Run one path through tiles, choosing reflexes stochastically."""
        path = []
        total_reward = 0.0
        current = start
        
        for _ in range(max_depth):
            if t_minus <= 0:
                break
            
            tile = self.tiles.get(current)
            if not tile or not tile.reflexes:
                break
            
            # Choose reflex by score
            scores = np.array([r.score for r in tile.reflexes])
            probs = scores / (scores.sum() + 1e-10)
            idx = np.random.choice(len(tile.reflexes), p=probs)
            reflex = tile.reflexes[idx]
            
            # Track usage
            reflex.times_chosen += 1
            
            # Simulate outcome (reward based on reflex success probability + noise)
            outcome = reflex.score + random.gauss(0, 0.1)
            success = outcome > 0.5
            if success:
                reflex.times_succeeded += 1
                total_reward += outcome
            
            t_minus -= reflex.cost * 0.1
            path.append({"tile": current, "reflex": reflex.name, "success": success, "reward": outcome})
            
            # Move to neighbor
            if tile.neighbors:
                weights = [max(self.tiles[n].momentum, 0.1) for n in tile.neighbors]
                weights = np.array(weights)
                weights /= weights.sum()
                current = np.random.choice(tile.neighbors, p=weights)
            else:
                break
        
        return {"path": path, "total_reward": total_reward, "depth": len(path)}
    
    def evolve_scores(self, n_simulations: int = 500, learning_rate: float = 0.1):
        """Evolve reflex scores based on simulation outcomes."""
        start_tiles = list(self.tiles.keys())
        
        for _ in range(n_simulations):
            start = random.choice(start_tiles)
            result = self.simulate_path(start, t_minus=1.0)
            
            # Update scores based on outcomes
            for step in result["path"]:
                tile = self.tiles.get(step["tile"])
                if not tile:
                    continue
                for reflex in tile.reflexes:
                    if reflex.name == step["reflex"]:
                        # Reinforce successful reflexes, weaken failed ones
                        if step["success"]:
                            reflex.score = min(reflex.score + learning_rate * 0.01, 1.0)
                        else:
                            reflex.score = max(reflex.score - learning_rate * 0.005, 0.01)
                        break
            
            self.outcome_history.append(result)
    
    def mutate_scores(self, rate: float = 0.05):
        """Add noise to scores (mutation for exploration)."""
        for tile in self.tiles.values():
            for reflex in tile.reflexes:
                if random.random() < rate:
                    delta = random.gauss(0, 0.05)
                    reflex.score = max(0.01, min(1.0, reflex.score + delta))
    
    def get_score_snapshot(self) -> dict:
        """Capture current scores for comparison."""
        return {
            tid: {r.name: round(r.score, 3) for r in t.reflexes}
            for tid, t in self.tiles.items()
        }


def build_game_field() -> TileField:
    """Build a tile field representing a game decision space."""
    field = TileField()
    
    # Opening moves
    field.add_tile(Tile(
        id="opening",
        type="decider",
        state={"phase": "opening"},
        reflexes=[
            Reflex("center_control", "play_center", 0.5, 0.1),
            Reflex("corner_play", "play_corner", 0.4, 0.1),
            Reflex("edge_play", "play_edge", 0.3, 0.1),
            Reflex("aggressive_open", "attack_immediately", 0.2, 0.2),
        ],
        neighbors=["mid_game_aggressive", "mid_game_defensive", "mid_game_balanced"],
    ))
    
    field.add_tile(Tile(
        id="mid_game_aggressive",
        type="decider",
        state={"phase": "mid", "style": "aggressive"},
        reflexes=[
            Reflex("full_attack", "push_hard", 0.5, 0.15),
            Reflex("feint", "fake_then_attack", 0.4, 0.1),
            Reflex("retreat", "fall_back", 0.3, 0.1),
        ],
        neighbors=["end_game_winning", "end_game_losing"],
        entropy=0.6,
    ))
    
    field.add_tile(Tile(
        id="mid_game_defensive",
        type="decider",
        state={"phase": "mid", "style": "defensive"},
        reflexes=[
            Reflex("fortify", "build_defense", 0.6, 0.1),
            Reflex("counter_attack", "wait_and_strike", 0.5, 0.15),
            Reflex("full_retreat", "give_ground", 0.2, 0.05),
        ],
        neighbors=["end_game_winning", "end_game_losing"],
        entropy=0.4,
    ))
    
    field.add_tile(Tile(
        id="mid_game_balanced",
        type="decider",
        state={"phase": "mid", "style": "balanced"},
        reflexes=[
            Reflex("probe", "test_weakness", 0.5, 0.1),
            Reflex("consolidate", "strengthen_position", 0.4, 0.1),
            Reflex("adapt", "switch_strategy", 0.6, 0.15),
        ],
        neighbors=["end_game_winning", "end_game_losing"],
        entropy=0.5,
    ))
    
    field.add_tile(Tile(
        id="end_game_winning",
        type="actor",
        state={"phase": "end", "advantage": "winning"},
        reflexes=[
            Reflex("close_out", "finish_game", 0.7, 0.1),
            Reflex("play_safe", "avoid_risks", 0.5, 0.05),
            Reflex("style_points", "go_for_broke", 0.3, 0.2),
        ],
        entropy=0.2,
    ))
    
    field.add_tile(Tile(
        id="end_game_losing",
        type="actor",
        state={"phase": "end", "advantage": "losing"},
        reflexes=[
            Reflex("desperate_attack", "all_in", 0.4, 0.2),
            Reflex("extend_game", "delay_loss", 0.3, 0.1),
            Reflex("resign", "concede", 0.2, 0.01),
            Reflex("miracle_play", "hope_for_best", 0.1, 0.15),
        ],
        entropy=0.8,
    ))
    
    # Connect
    field.connect("opening", "mid_game_aggressive")
    field.connect("opening", "mid_game_defensive")
    field.connect("opening", "mid_game_balanced")
    field.connect("mid_game_aggressive", "end_game_winning")
    field.connect("mid_game_aggressive", "end_game_losing")
    field.connect("mid_game_defensive", "end_game_winning")
    field.connect("mid_game_defensive", "end_game_losing")
    field.connect("mid_game_balanced", "end_game_winning")
    field.connect("mid_game_balanced", "end_game_losing")
    
    return field


def run_evolution():
    print("=" * 70)
    print("STOCHASTIC REFLEX EVOLUTION — Evolving Tile Field Scores")
    print("=" * 70)
    
    field = build_game_field()
    
    print(f"\nTiles: {len(field.tiles)}")
    print(f"Initial scores:")
    for tid, tile in field.tiles.items():
        scores = {r.name: f"{r.score:.2f}" for r in tile.reflexes}
        print(f"  {tid}: {scores}")
    
    # Evolve for 10 generations
    print(f"\nEvolving for 10 generations (500 simulations each)...")
    
    score_history = [field.get_score_snapshot()]
    
    for gen in range(1, 11):
        start = time.perf_counter()
        field.evolve_scores(n_simulations=500, learning_rate=0.1)
        field.mutate_scores(rate=0.05)
        elapsed = time.perf_counter() - start
        
        # Collect stats
        all_scores = []
        for tile in field.tiles.values():
            for r in tile.reflexes:
                all_scores.append(r.score)
        
        snapshot = field.get_score_snapshot()
        score_history.append(snapshot)
        
        # Top reflex per tile
        top_reflexes = {}
        for tid, tile in field.tiles.items():
            best = max(tile.reflexes, key=lambda r: r.score)
            top_reflexes[tid] = f"{best.name} ({best.score:.3f})"
        
        print(f"  Gen {gen:2d}: avg_score={np.mean(all_scores):.3f}, "
              f"max={max(all_scores):.3f}, min={min(all_scores):.3f} ({elapsed:.1f}s)")
        print(f"    Top: {top_reflexes}")
    
    # Final analysis
    print(f"\n{'=' * 70}")
    print("FINAL SCORES (evolved)")
    print(f"{'=' * 70}")
    
    for tid, tile in field.tiles.items():
        print(f"\n  {tid}:")
        for r in sorted(tile.reflexes, key=lambda x: -x.score):
            sr = f"{r.success_rate:.1%}" if r.times_chosen > 0 else "N/A"
            print(f"    {r.name}: score={r.score:.3f}, chosen={r.times_chosen}, "
                  f"success={sr}")
    
    # What changed?
    print(f"\n{'=' * 70}")
    print("EVOLUTION ANALYSIS (initial → final)")
    print(f"{'=' * 70}")
    
    initial = score_history[0]
    final = score_history[-1]
    
    for tid in field.tiles:
        if tid in initial and tid in final:
            print(f"\n  {tid}:")
            for reflex_name in initial[tid]:
                old = initial[tid][reflex_name]
                new = final[tid][reflex_name]
                delta = new - old
                arrow = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "→")
                print(f"    {reflex_name}: {old:.3f} → {new:.3f} ({arrow} {delta:+.3f})")
    
    # Save
    output = {
        "score_history": score_history,
        "final_scores": field.get_score_snapshot(),
        "outcome_count": len(field.outcome_history),
    }
    
    out = os.path.expanduser("~/repos/zeroclaw-arena/reflex-evolution-results.json")
    with open(out, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    run_evolution()
