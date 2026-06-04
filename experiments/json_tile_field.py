"""
JSON Tile Field — Agents as JSONs in a Stochastic Decision Web

Core abstraction:
- A Tile is a JSON object with: id, type, state, reflexes (scored variations), neighbors
- A Room is a set of tiles that reference each other
- The Interpreter jumps between tiles, never "running" one continuously
- Each jump is a decision: which tile, which reflex, based on stochastic simulation
- T-minus: the countdown until action must be taken (forces early termination of simulation)
- Empty space: the gaps between tiles in the decision tree (modeled as uncertainty)

The stochastic simulation:
1. At each timestep, the interpreter reads the room state
2. It runs N Monte Carlo simulations of possible futures (each a path through tiles)
3. Each simulation has a probability score based on reflex scores + momentum + room read
4. T-minus constrains how deep simulations can go
5. The highest-expected-value path is chosen
6. Empty space between tiles = uncertainty = where negative space intelligence lives
"""

import json
import hashlib
import random
import time
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from collections import defaultdict, Counter
import os


@dataclass
class Reflex:
    """A scored algorithmic variation — one possible action with expected value."""
    name: str
    action: str
    score: float  # 0-1, higher = better expected outcome
    cost: float   # time/energy cost
    conditions: Dict[str, Any] = field(default_factory=dict)  # when this reflex fires
    
    def to_json(self) -> dict:
        return asdict(self)


@dataclass 
class Tile:
    """An agent as JSON — a node in the decision web."""
    id: str
    type: str  # "sensor", "decider", "actor", "memory", "planner"
    state: Dict[str, Any] = field(default_factory=dict)
    reflexes: List[Reflex] = field(default_factory=list)
    neighbors: List[str] = field(default_factory=list)  # IDs of connected tiles
    momentum: float = 0.0  # how much "force" is behind this tile's activation
    entropy: float = 1.0   # uncertainty level (1.0 = maximum uncertainty)
    
    def to_json(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "state": self.state,
            "reflexes": [r.to_json() for r in self.reflexes],
            "neighbors": self.neighbors,
            "momentum": self.momentum,
            "entropy": self.entropy,
        }
    
    @classmethod
    def from_json(cls, data: dict) -> 'Tile':
        reflexes = [Reflex(**r) for r in data.get('reflexes', [])]
        return cls(
            id=data['id'],
            type=data['type'],
            state=data.get('state', {}),
            reflexes=reflexes,
            neighbors=data.get('neighbors', []),
            momentum=data.get('momentum', 0.0),
            entropy=data.get('entropy', 1.0),
        )


class Room:
    """A field of JSON tiles — the decision web."""
    
    def __init__(self):
        self.tiles: Dict[str, Tile] = {}
        self.history: List[dict] = []  # log of tile activations
        self.global_momentum: float = 0.0
    
    def add_tile(self, tile: Tile):
        self.tiles[tile.id] = tile
    
    def connect(self, tile_a: str, tile_b: str):
        if tile_a in self.tiles and tile_b in self.tiles:
            if tile_b not in self.tiles[tile_a].neighbors:
                self.tiles[tile_a].neighbors.append(tile_b)
            if tile_a not in self.tiles[tile_b].neighbors:
                self.tiles[tile_b].neighbors.append(tile_a)
    
    def get_state_hash(self) -> str:
        """Hash of the entire room state — for cache/comparison."""
        state = {tid: t.state for tid, t in sorted(self.tiles.items())}
        return hashlib.blake2b(json.dumps(state, sort_keys=True).encode(), digest_size=8).hexdigest()
    
    def read_room(self) -> dict:
        """Read the room — what's the current state of all tiles?"""
        return {
            "n_tiles": len(self.tiles),
            "active_tiles": sum(1 for t in self.tiles.values() if t.momentum > 0.1),
            "total_momentum": sum(t.momentum for t in self.tiles.values()),
            "avg_entropy": np.mean([t.entropy for t in self.tiles.values()]) if self.tiles else 1.0,
            "state_hash": self.get_state_hash(),
            "type_distribution": dict(Counter(t.type for t in self.tiles.values())),
        }


class StochasticSimulator:
    """Monte Carlo simulation of possible futures through the tile field."""
    
    def __init__(self, room: Room, n_simulations: int = 100):
        self.room = room
        self.n_simulations = n_simulations
        self.empty_space_model: Dict[str, float] = {}  # tile_id → uncertainty
    
    def simulate_path(self, start_tile: str, t_minus: float, max_depth: int = 10) -> dict:
        """Run one Monte Carlo simulation of a path through tiles."""
        path = [start_tile]
        total_score = 0.0
        total_cost = 0.0
        current_tile = start_tile
        momentum = self.room.tiles[current_tile].momentum
        
        for depth in range(max_depth):
            if t_minus <= 0:
                break  # Time's up — must act
            
            tile = self.room.tiles.get(current_tile)
            if not tile or not tile.reflexes:
                break
            
            # Choose reflex stochastically (weighted by score)
            scores = np.array([r.score for r in tile.reflexes])
            probs = scores / (scores.sum() + 1e-10)
            chosen_idx = np.random.choice(len(tile.reflexes), p=probs)
            reflex = tile.reflexes[chosen_idx]
            
            # Apply momentum: high momentum → more likely to continue chain
            momentum_factor = 0.5 + 0.5 * min(momentum, 1.0)
            
            # Apply room read: high entropy → more exploration
            room_state = self.room.read_room()
            exploration_bonus = room_state["avg_entropy"] * 0.2
            
            # Empty space: uncertainty between tiles
            empty_space = tile.entropy * 0.1
            total_score += reflex.score * momentum_factor + exploration_bonus - empty_space
            total_cost += reflex.cost
            
            # Decay t_minus
            t_minus -= reflex.cost * 0.1
            
            # Move to next tile
            if tile.neighbors:
                # Weighted by neighbor momentum
                neighbor_weights = []
                for n_id in tile.neighbors:
                    n_tile = self.room.tiles.get(n_id)
                    w = max(n_tile.momentum, 0.1) if n_tile else 0.1
                    neighbor_weights.append(w)
                
                weights = np.array(neighbor_weights)
                weights /= weights.sum()
                next_idx = np.random.choice(len(tile.neighbors), p=weights)
                current_tile = tile.neighbors[next_idx]
                path.append(current_tile)
                
                # Update momentum (chain gains momentum)
                momentum = min(momentum + 0.1, 2.0)
            else:
                break
        
        return {
            "path": path,
            "total_score": total_score,
            "total_cost": total_cost,
            "depth": len(path),
            "final_momentum": momentum,
        }
    
    def run_simulations(self, start_tile: str, t_minus: float) -> dict:
        """Run N Monte Carlo simulations and aggregate results."""
        results = []
        for _ in range(self.n_simulations):
            result = self.simulate_path(start_tile, t_minus)
            results.append(result)
        
        # Aggregate
        paths = [r["path"] for r in results]
        scores = [r["total_score"] for r in results]
        costs = [r["total_cost"] for r in results]
        
        # Find best path
        best_idx = np.argmax(scores)
        best_path = paths[best_idx]
        
        # Count path frequencies (most common = most probable)
        path_counter = defaultdict(int)
        for p in paths:
            path_key = "→".join(p)
            path_counter[path_key] += 1
        
        most_common_path = max(path_counter, key=path_counter.get)
        most_common_count = path_counter[most_common_path]
        
        return {
            "n_simulations": self.n_simulations,
            "best_path": best_path,
            "best_score": scores[best_idx],
            "avg_score": np.mean(scores),
            "most_common_path": most_common_path.split("→"),
            "most_common_frequency": most_common_count,
            "most_common_probability": most_common_count / self.n_simulations,
            "avg_depth": np.mean([r["depth"] for r in results]),
            "avg_cost": np.mean(costs),
            "all_scores": scores,
        }


class TMinusClock:
    """T-minus countdown — how much time before action is forced."""
    
    def __init__(self, total_time: float = 1.0):
        self.total_time = total_time
        self.remaining = total_time
        self.started_at = time.perf_counter()
    
    def tick(self, cost: float = 0.01) -> float:
        """Advance time. Returns remaining time."""
        self.remaining -= cost
        return max(self.remaining, 0.0)
    
    @property
    def urgency(self) -> float:
        """0.0 = plenty of time, 1.0 = must act NOW."""
        return 1.0 - (self.remaining / self.total_time)
    
    @property
    def is_expired(self) -> bool:
        return self.remaining <= 0


class Interpreter:
    """The model that jumps between JSON tiles in the room."""
    
    def __init__(self, room: Room, n_simulations: int = 50):
        self.room = room
        self.simulator = StochasticSimulator(room, n_simulations)
        self.current_tile: Optional[str] = None
        self.decision_log: List[dict] = []
    
    def activate(self, tile_id: str, t_minus_time: float = 1.0) -> dict:
        """Activate a tile — run stochastic simulation and choose action."""
        self.current_tile = tile_id
        tile = self.room.tiles[tile_id]
        
        # Start T-minus clock
        clock = TMinusClock(t_minus_time)
        
        # Run Monte Carlo simulations
        sim_results = self.simulator.run_simulations(tile_id, clock.remaining)
        
        # Choose best action based on simulation results
        if tile.reflexes:
            # Weight by both score and simulation evidence
            best_path = sim_results["best_path"]
            path_score = sim_results["best_score"]
            common_prob = sim_results["most_common_probability"]
            
            # Decision: use best-scored path, but prefer high-probability paths
            # when urgency is high (T-minus running out)
            if clock.urgency > 0.7:
                # High urgency: go with most probable path
                chosen_path = sim_results["most_common_path"]
                confidence = common_prob
            else:
                # Low urgency: explore best-scored path
                chosen_path = best_path
                confidence = path_score / (max(sim_results["all_scores"]) + 1e-10)
            
            # Choose first reflex of first tile in chosen path
            first_tile_id = chosen_path[0] if chosen_path else tile_id
            first_tile = self.room.tiles.get(first_tile_id, tile)
            
            if first_tile.reflexes:
                # Pick highest-scored reflex that matches conditions
                best_reflex = max(first_tile.reflexes, key=lambda r: r.score)
                action = best_reflex.action
            else:
                action = "idle"
        else:
            action = "idle"
            chosen_path = [tile_id]
            confidence = 0.0
        
        # Update momentum
        for tid in chosen_path:
            if tid in self.room.tiles:
                self.room.tiles[tid].momentum = min(
                    self.room.tiles[tid].momentum + 0.1, 2.0
                )
        
        # Decay entropy (each activation reduces uncertainty)
        tile.entropy = max(tile.entropy - 0.05, 0.0)
        
        decision = {
            "tile": tile_id,
            "action": action,
            "path": chosen_path,
            "confidence": confidence,
            "urgency": clock.urgency,
            "simulations": sim_results["n_simulations"],
            "avg_score": sim_results["avg_score"],
            "remaining_time": clock.remaining,
        }
        
        self.decision_log.append(decision)
        return decision


def build_robot_room() -> Room:
    """Build a room simulating a robot navigating a space."""
    room = Room()
    
    # Sensor tiles
    room.add_tile(Tile(
        id="lidar_front",
        type="sensor",
        state={"distance": 2.5, "object": "wall"},
        reflexes=[
            Reflex("approach_wall", "slow_down", 0.8, 0.1, {"distance_lt": 3.0}),
            Reflex("wall_clear", "maintain_speed", 0.6, 0.05),
        ],
        entropy=0.3,
    ))
    
    room.add_tile(Tile(
        id="lidar_left",
        type="sensor",
        state={"distance": 5.0, "object": "clear"},
        reflexes=[
            Reflex("left_clear", "turn_left", 0.7, 0.1),
            Reflex("maintain", "go_straight", 0.5, 0.05),
        ],
        entropy=0.1,
    ))
    
    room.add_tile(Tile(
        id="camera",
        type="sensor",
        state={"objects": ["person", "chair"], "person_distance": 4.0},
        reflexes=[
            Reflex("person_ahead", "stop_and_wait", 0.9, 0.2, {"person_distance_lt": 5.0}),
            Reflex("person_far", "proceed_cautiously", 0.6, 0.1),
        ],
        entropy=0.5,
    ))
    
    # Decider tiles
    room.add_tile(Tile(
        id="path_planner",
        type="decider",
        state={"destination": "kitchen", "obstacles": 2},
        reflexes=[
            Reflex("reroute", "calculate_new_path", 0.8, 0.3, {"obstacles_gt": 1}),
            Reflex("continue", "follow_path", 0.5, 0.1),
            Reflex("explore", "scan_alternatives", 0.3, 0.2),
        ],
        neighbors=["lidar_front", "lidar_left", "camera"],
        entropy=0.4,
    ))
    
    room.add_tile(Tile(
        id="safety_checker",
        type="decider",
        state={"risk_level": "medium", "last_incident": 30},
        reflexes=[
            Reflex("high_risk", "full_stop", 0.95, 0.05, {"risk_eq": "high"}),
            Reflex("medium_risk", "reduce_speed", 0.7, 0.1, {"risk_eq": "medium"}),
            Reflex("low_risk", "proceed", 0.4, 0.05, {"risk_eq": "low"}),
        ],
        neighbors=["path_planner"],
        entropy=0.2,
    ))
    
    # Actor tiles
    room.add_tile(Tile(
        id="motor_control",
        type="actor",
        state={"speed": 0.5, "direction": "forward"},
        reflexes=[
            Reflex("accelerate", "speed_up", 0.3, 0.1),
            Reflex("decelerate", "slow_down", 0.6, 0.1),
            Reflex("stop", "full_stop", 0.9, 0.05),
            Reflex("turn_left", "rotate_left", 0.5, 0.15),
            Reflex("turn_right", "rotate_right", 0.5, 0.15),
        ],
        neighbors=["safety_checker", "path_planner"],
        entropy=0.1,
    ))
    
    room.add_tile(Tile(
        id="voice",
        type="actor",
        state={"last_utterance": "excuse_me", "volume": "normal"},
        reflexes=[
            Reflex("announce", "say_excuse_me", 0.7, 0.2, {"person_nearby": True}),
            Reflex("quiet", "silent_mode", 0.3, 0.0),
        ],
        neighbors=["camera"],
        entropy=0.3,
    ))
    
    # Memory tile
    room.add_tile(Tile(
        id="spatial_memory",
        type="memory",
        state={"known_paths": 3, "dead_ends": 1, "last_location": "hallway"},
        reflexes=[
            Reflex("recall_path", "use_known_route", 0.8, 0.1, {"known_paths_gt": 0}),
            Reflex("explore_new", "try_unexplored", 0.4, 0.3),
        ],
        neighbors=["path_planner", "safety_checker"],
        entropy=0.6,
    ))
    
    # Connect tiles
    room.connect("lidar_front", "path_planner")
    room.connect("lidar_left", "path_planner")
    room.connect("camera", "path_planner")
    room.connect("camera", "voice")
    room.connect("path_planner", "safety_checker")
    room.connect("safety_checker", "motor_control")
    room.connect("spatial_memory", "path_planner")
    
    return room


def run_experiment():
    print("=" * 70)
    print("JSON TILE FIELD — Stochastic Decision Web")
    print("=" * 70)
    
    room = build_robot_room()
    interpreter = Interpreter(room, n_simulations=200)
    
    print(f"\nRoom: {len(room.tiles)} tiles")
    for tid, tile in room.tiles.items():
        print(f"  {tid} ({tile.type}): {len(tile.reflexes)} reflexes, "
              f"entropy={tile.entropy:.1f}, momentum={tile.momentum:.1f}")
    
    print(f"\nRoom state: {json.dumps(room.read_room(), indent=2)}")
    
    # Simulate T-minus decisions
    print("\n" + "=" * 70)
    print("T-MINUS SIMULATION — Robot navigating to kitchen")
    print("=" * 70)
    
    scenarios = [
        ("plenty of time", 2.0),
        ("moderate urgency", 0.5),
        ("must act NOW", 0.1),
        ("crisis mode", 0.02),
    ]
    
    for scenario_name, t_minus in scenarios:
        print(f"\n--- Scenario: {scenario_name} (T-{t_minus}s) ---")
        
        # Start from path planner
        decision = interpreter.activate("path_planner", t_minus_time=t_minus)
        
        print(f"  Action: {decision['action']}")
        print(f"  Path: {' → '.join(decision['path'])}")
        print(f"  Confidence: {decision['confidence']:.2f}")
        print(f"  Urgency: {decision['urgency']:.1%}")
        print(f"  Avg simulation score: {decision['avg_score']:.2f}")
        print(f"  Time remaining: {decision['remaining_time']:.3f}s")
    
    # Run iterative learning — 10 rounds of decision-making
    print("\n" + "=" * 70)
    print("ITERATIVE LEARNING — 10 rounds with momentum accumulation")
    print("=" * 70)
    
    for round_num in range(1, 11):
        # Random starting tile
        start = random.choice(list(room.tiles.keys()))
        decision = interpreter.activate(start, t_minus_time=0.5)
        
        room_state = room.read_room()
        print(f"  Round {round_num}: start={start}, action={decision['action']}, "
              f"confidence={decision['confidence']:.2f}, "
              f"active_tiles={room_state['active_tiles']}, "
              f"total_momentum={room_state['total_momentum']:.1f}")
    
    # Analyze empty space
    print("\n" + "=" * 70)
    print("EMPTY SPACE ANALYSIS — Uncertainty in the decision web")
    print("=" * 70)
    
    for tid, tile in room.tiles.items():
        gap_uncertainty = tile.entropy * len(tile.neighbors) * 0.1
        print(f"  {tid}: entropy={tile.entropy:.2f}, "
              f"neighbors={len(tile.neighbors)}, "
              f"gap_uncertainty={gap_uncertainty:.2f}, "
              f"momentum={tile.momentum:.2f}")
    
    total_entropy = sum(t.entropy for t in room.tiles.values())
    total_momentum = sum(t.momentum for t in room.tiles.values())
    print(f"\n  Total entropy: {total_entropy:.2f}")
    print(f"  Total momentum: {total_momentum:.2f}")
    print(f"  Entropy/momentum ratio: {total_entropy / (total_momentum + 1e-10):.2f}")
    print(f"  → {'Entropy dominates (exploration mode)' if total_entropy > total_momentum else 'Momentum dominates (exploitation mode)'}")
    
    # Save everything as JSON (the whole system IS the JSON)
    output = {
        "room": {tid: t.to_json() for tid, t in room.tiles.items()},
        "decisions": interpreter.decision_log,
        "room_state": room.read_room(),
        "total_entropy": total_entropy,
        "total_momentum": total_momentum,
    }
    
    out = os.path.expanduser("~/repos/zeroclaw-arena/json-tile-field-results.json")
    with open(out, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    run_experiment()
