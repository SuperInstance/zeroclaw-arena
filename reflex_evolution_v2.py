"""
Reflex Evolution v2 — Fixed Polarization Problem

v1 polarized to 0/1 because reward was coupled to score:
  outcome = reflex.score + noise  → self-reinforcing feedback loop.

v2 fixes:
1. Reward decoupled from score — outcomes use game-like win/loss based on
   path quality + noise, never raw score.
2. Temperature decay: start at 2.0, decay 0.1/gen down to 0.1 (softmax temp).
3. Epsilon-greedy: 5% chance of choosing LEAST-CHOSEN reflex (forced exploration).
4. Capped score deltas: max ±0.05 per generation per reflex.
5. Scores clamped to [0.05, 0.95] — never hit absolute 0 or 1.
"""

import json
import random
import numpy as np
import time
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any
from collections import defaultdict


SCORE_MIN = 0.05
SCORE_MAX = 0.95
MAX_DELTA = 0.05


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

    def clamp_score(self):
        self.score = max(SCORE_MIN, min(SCORE_MAX, self.score))

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
    """A field of tiles for the evolutionary experiment — v2."""

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

    def _choose_reflex(self, tile: Tile, epsilon: float = 0.05) -> 'Reflex':
        """Choose reflex with temperature-based softmax + epsilon-greedy exploration."""
        # Epsilon-greedy: pick least-chosen reflex
        if random.random() < epsilon and tile.reflexes:
            min_chosen = min(r.times_chosen for r in tile.reflexes)
            least = [r for r in tile.reflexes if r.times_chosen == min_chosen]
            return random.choice(least)

        # Otherwise softmax with current temperature
        return None  # filled in by caller with temperature

    def simulate_path(self, start: str, t_minus: float, max_depth: int = 8,
                      temperature: float = 1.0, epsilon: float = 0.05) -> dict:
        """Run one path through tiles, choosing reflexes stochastically."""
        path = []
        path_quality = 0.0
        current = start

        for _ in range(max_depth):
            if t_minus <= 0:
                break

            tile = self.tiles.get(current)
            if not tile or not tile.reflexes:
                break

            # Epsilon-greedy: 5% chance pick least-chosen
            if random.random() < epsilon:
                min_chosen = min(r.times_chosen for r in tile.reflexes)
                least = [r for r in tile.reflexes if r.times_chosen == min_chosen]
                reflex = random.choice(least)
            else:
                # Softmax with temperature
                scores = np.array([r.score for r in tile.reflexes])
                logits = scores / max(temperature, 1e-6)
                # Numerically stable softmax
                logits = logits - logits.max()
                probs = np.exp(logits)
                probs = probs / probs.sum()
                idx = np.random.choice(len(tile.reflexes), p=probs)
                reflex = tile.reflexes[idx]

            reflex.times_chosen += 1

            # v2 FIX #1: Reward decoupled from score.
            # Use intrinsic reflex quality (a hidden property) + noise.
            # The "intrinsic quality" is based on the reflex's action type and conditions,
            # NOT its evolving score. This breaks the feedback loop.
            intrinsic = self._intrinsic_outcome(reflex, tile)
            noise = random.gauss(0, 0.15)
            outcome = intrinsic + noise

            # Win/loss threshold at 0.5
            success = outcome > 0.5
            if success:
                reflex.times_succeeded += 1

            path_quality += outcome
            t_minus -= reflex.cost * 0.1
            path.append({
                "tile": current,
                "reflex": reflex.name,
                "success": success,
                "outcome": outcome,
            })

            # Move to neighbor
            if tile.neighbors:
                weights = [max(self.tiles[n].momentum, 0.1) for n in tile.neighbors]
                weights = np.array(weights)
                weights /= weights.sum()
                current = np.random.choice(tile.neighbors, p=weights)
            else:
                break

        # Path-level win/loss
        path_win = path_quality > 0  # positive total quality = win

        return {
            "path": path,
            "path_quality": path_quality,
            "path_win": path_win,
            "depth": len(path),
        }

    def _intrinsic_outcome(self, reflex: Reflex, tile: Tile) -> float:
        """
        v2: Generate a game-like outcome INDEPENDENT of reflex.score.

        Uses the reflex's action type, tile context, and a fixed quality signal
        so that the evolutionary signal comes from actual game dynamics, not
        from the score that's being evolved.
        """
        # Hash the reflex name + tile state to get a deterministic but varied quality
        # This simulates "some actions are genuinely better in some contexts"
        seed_str = f"{reflex.name}:{tile.id}:{tile.state}"
        h = hash(seed_str)
        # Map to [0.2, 0.8] range — no reflex is trivially always-good or always-bad
        base_quality = 0.2 + 0.6 * ((h % 1000) / 1000.0)

        # Contextual modifiers (simulating game dynamics)
        phase = tile.state.get("phase", "")
        style = tile.state.get("style", "")
        advantage = tile.state.get("advantage", "")

        modifier = 0.0
        if phase == "opening":
            if "center" in reflex.name or "balanced" in reflex.action:
                modifier += 0.1
            if "aggressive" in reflex.name:
                modifier -= 0.05
        elif phase == "mid":
            if style == "defensive" and "defense" in reflex.action:
                modifier += 0.08
            if style == "aggressive" and "attack" in reflex.action:
                modifier += 0.08
            if "adapt" in reflex.action:
                modifier += 0.05  # adaptability bonus
        elif phase == "end":
            if advantage == "winning" and "finish" in reflex.action:
                modifier += 0.1
            if advantage == "losing" and "all_in" in reflex.action:
                modifier += 0.05  # desperate measures can work
            if advantage == "losing" and "concede" in reflex.action:
                modifier -= 0.1  # resignation is bad for learning

        return max(0.05, min(0.95, base_quality + modifier))

    def evolve_scores(self, n_simulations: int = 300, temperature: float = 1.0,
                      epsilon: float = 0.05):
        """Evolve reflex scores based on simulation outcomes — v2 with capped deltas."""
        start_tiles = list(self.tiles.keys())

        # Accumulate score deltas per (tile_id, reflex_name)
        delta_accum: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        count_accum: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for _ in range(n_simulations):
            start = random.choice(start_tiles)
            result = self.simulate_path(
                start, t_minus=1.0,
                temperature=temperature, epsilon=epsilon,
            )

            path_win = result["path_win"]

            for step in result["path"]:
                tile_id = step["tile"]
                reflex_name = step["reflex"]
                step_success = step["success"]

                # Reward signal: path-level outcome + step-level signal
                # path_win is the dominant signal (decoupled from score!)
                if path_win:
                    delta_accum[tile_id][reflex_name] += 0.02
                else:
                    delta_accum[tile_id][reflex_name] -= 0.01

                # Step-level: bonus for individual success in winning paths
                if path_win and step_success:
                    delta_accum[tile_id][reflex_name] += 0.01

                count_accum[tile_id][reflex_name] += 1

            self.outcome_history.append(result)

        # Apply deltas with capping (v2 FIX #4)
        for tile_id, tile in self.tiles.items():
            for reflex in tile.reflexes:
                raw_delta = delta_accum[tile_id].get(reflex.name, 0.0)
                # Normalize by times chosen
                if count_accum[tile_id].get(reflex.name, 0) > 0:
                    normalized = raw_delta / count_accum[tile_id][reflex.name]
                else:
                    normalized = 0.0

                # Cap at ±MAX_DELTA
                capped = max(-MAX_DELTA, min(MAX_DELTA, normalized))
                reflex.score += capped
                reflex.clamp_score()  # v2 FIX #5

    def mutate_scores(self, rate: float = 0.05, temperature: float = 1.0):
        """Add noise to scores — mutation scaled by temperature."""
        noise_scale = 0.03 * max(temperature, 0.1)
        for tile in self.tiles.values():
            for reflex in tile.reflexes:
                if random.random() < rate:
                    delta = random.gauss(0, noise_scale)
                    # Cap mutation delta too
                    delta = max(-MAX_DELTA, min(MAX_DELTA, delta))
                    reflex.score += delta
                    reflex.clamp_score()

    def get_score_snapshot(self) -> dict:
        return {
            tid: {r.name: round(r.score, 3) for r in t.reflexes}
            for tid, t in self.tiles.items()
        }

    def get_polarization_metric(self) -> float:
        """How close are scores to 0 or 1? 0 = healthy, 1 = fully polarized."""
        all_scores = [r.score for t in self.tiles.values() for r in t.reflexes]
        if not all_scores:
            return 0.0
        # Distance from 0.5, normalized to [0, 1]
        return sum(abs(s - 0.5) / 0.5 for s in all_scores) / len(all_scores)

    def get_entropy_metric(self) -> float:
        """Average entropy of score distributions per tile. Higher = more diverse."""
        entropies = []
        for tile in self.tiles.values():
            scores = np.array([r.score for r in tile.reflexes])
            probs = scores / (scores.sum() + 1e-10)
            ent = -np.sum(probs * np.log(probs + 1e-10))
            entropies.append(ent)
        return np.mean(entropies) if entropies else 0.0


def build_game_field() -> TileField:
    """Build a tile field representing a game decision space."""
    field = TileField()

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
    N_GENERATIONS = 20
    GAMES_PER_GEN = 300
    TEMP_START = 2.0
    TEMP_DECAY = 0.1
    TEMP_MIN = 0.1
    EPSILON = 0.05
    MUTATION_RATE = 0.05

    print("=" * 70)
    print("STOCHASTIC REFLEX EVOLUTION v2 — Fixed Polarization")
    print("=" * 70)
    print(f"\nv2 fixes applied:")
    print(f"  1. Reward decoupled from score (intrinsic outcomes)")
    print(f"  2. Temperature decay: {TEMP_START} → {TEMP_MIN}")
    print(f"  3. Epsilon-greedy: {EPSILON:.0%} least-chosen exploration")
    print(f"  4. Score delta cap: ±{MAX_DELTA}")
    print(f"  5. Score bounds: [{SCORE_MIN}, {SCORE_MAX}]")

    field = build_game_field()

    print(f"\nTiles: {len(field.tiles)}")
    print(f"Initial scores:")
    for tid, tile in field.tiles.items():
        scores = {r.name: f"{r.score:.2f}" for r in tile.reflexes}
        print(f"  {tid}: {scores}")

    print(f"\nEvolving for {N_GENERATIONS} generations ({GAMES_PER_GEN} games each)...")

    score_history = [field.get_score_snapshot()]
    polarization_history = [field.get_polarization_metric()]
    entropy_history = [field.get_entropy_metric()]

    for gen in range(1, N_GENERATIONS + 1):
        temperature = max(TEMP_MIN, TEMP_START - TEMP_DECAY * (gen - 1))
        start = time.perf_counter()

        field.evolve_scores(
            n_simulations=GAMES_PER_GEN,
            temperature=temperature,
            epsilon=EPSILON,
        )
        field.mutate_scores(rate=MUTATION_RATE, temperature=temperature)
        elapsed = time.perf_counter() - start

        # Stats
        all_scores = [r.score for t in field.tiles.values() for r in t.reflexes]
        snapshot = field.get_score_snapshot()
        score_history.append(snapshot)

        polarization = field.get_polarization_metric()
        polarization_history.append(polarization)
        entropy = field.get_entropy_metric()
        entropy_history.append(entropy)

        top_reflexes = {}
        for tid, tile in field.tiles.items():
            best = max(tile.reflexes, key=lambda r: r.score)
            top_reflexes[tid] = f"{best.name} ({best.score:.3f})"

        print(f"  Gen {gen:2d} [T={temperature:.1f}]: "
              f"avg={np.mean(all_scores):.3f}, "
              f"range=[{min(all_scores):.3f}, {max(all_scores):.3f}], "
              f"polar={polarization:.3f}, "
              f"entropy={entropy:.3f} ({elapsed:.1f}s)")
        print(f"    Top: {top_reflexes}")

    # Final analysis
    print(f"\n{'=' * 70}")
    print("FINAL SCORES (v2 evolved)")
    print(f"{'=' * 70}")

    for tid, tile in field.tiles.items():
        print(f"\n  {tid}:")
        for r in sorted(tile.reflexes, key=lambda x: -x.score):
            sr = f"{r.success_rate:.1%}" if r.times_chosen > 0 else "N/A"
            print(f"    {r.name}: score={r.score:.3f}, chosen={r.times_chosen}, "
                  f"success={sr}")

    # Evolution deltas
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

    # Polarization check
    print(f"\n{'=' * 70}")
    print("POLARIZATION CHECK")
    print(f"{'=' * 70}")
    print(f"  Initial polarization: {polarization_history[0]:.3f}")
    print(f"  Final polarization:   {polarization_history[-1]:.3f}")
    polarized_count = sum(
        1 for t in field.tiles.values() for r in t.reflexes
        if r.score >= 0.9 or r.score <= 0.1
    )
    total = sum(len(t.reflexes) for t in field.tiles.values())
    print(f"  Reflexes near boundaries (≤0.1 or ≥0.9): {polarized_count}/{total}")
    if polarized_count == 0:
        print(f"  ✅ No polarization! All scores in healthy range.")
    else:
        print(f"  ⚠️  Some drift toward boundaries, but clamped.")

    # Save results
    output = {
        "version": "v2",
        "config": {
            "n_generations": N_GENERATIONS,
            "games_per_gen": GAMES_PER_GEN,
            "temp_start": TEMP_START,
            "temp_decay": TEMP_DECAY,
            "temp_min": TEMP_MIN,
            "epsilon": EPSILON,
            "max_delta": MAX_DELTA,
            "score_bounds": [SCORE_MIN, SCORE_MAX],
        },
        "score_history": score_history,
        "polarization_history": [round(p, 4) for p in polarization_history],
        "entropy_history": [round(e, 4) for e in entropy_history],
        "final_scores": field.get_score_snapshot(),
        "outcome_count": len(field.outcome_history),
    }

    out = os.path.expanduser("~/repos/zeroclaw-arena/reflex-evolution-v2-results.json")
    with open(out, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    run_evolution()
