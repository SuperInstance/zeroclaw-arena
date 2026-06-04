"""
Multiplayer Divergence Experiment - Does the divergence theorem hold for 3+ players?

The 2-player divergence theorem: strategies diverge monotonically, no Nash equilibrium.

3-player simplified poker variant:
  - 3 players, each with their own tile field
  - Ante + one round of betting + showdown
  - Track pairwise strategy distances and win rates

Phases:
  1. (500 hands) All 3 default - baseline
  2. (500 hands) All 3 learning against each other
  3. (500 hands) Continued arms race

Hypothesis: 3-player is MORE divergent than 2-player.
The strategy space is larger, creating more exploitable patterns.
"""

import random
import numpy as np
import hashlib
import json
import os
from collections import defaultdict
from itertools import combinations
from typing import List, Dict, Tuple, Optional


# ============== CARD / HAND EVALUATION ==============

SUITS = ['♠', '♥', '♦', '♣']
RANKS = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
RANK_VAL = {r: i for i, r in enumerate(RANKS)}

def make_deck():
    return [(r, s) for s in SUITS for r in RANKS]

def hand_rank(cards):
    ranks = sorted([RANK_VAL[c[0]] for c in cards], reverse=True)
    suits = [c[1] for c in cards]
    is_flush = len(set(suits)) == 1
    unique = sorted(set(ranks), reverse=True)
    is_straight = False
    straight_high = 0
    if len(unique) == 5:
        if unique[0] - unique[4] == 4:
            is_straight = True
            straight_high = unique[0]
        if unique == [12, 3, 2, 1, 0]:
            is_straight = True
            straight_high = 3
    counts = defaultdict(int)
    for r in ranks:
        counts[r] += 1
    groups = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    if is_straight and is_flush:
        return (8, straight_high)
    if groups[0][1] == 4:
        return (7, groups[0][0], groups[1][0])
    if groups[0][1] == 3 and groups[1][1] == 2:
        return (6, groups[0][0], groups[1][0])
    if is_flush:
        return (5,) + tuple(ranks)
    if is_straight:
        return (4, straight_high)
    if groups[0][1] == 3:
        return (3, groups[0][0], groups[1][0], groups[2][0])
    if groups[0][1] == 2 and groups[1][1] == 2:
        return (2, groups[0][0], groups[1][0], groups[2][0])
    if groups[0][1] == 2:
        return (1, groups[0][0], groups[1][0], groups[2][0], groups[3][0])
    return (0,) + tuple(ranks)

def best_hand(hole, community):
    all_cards = hole + community
    if len(all_cards) < 5:
        return hand_rank(all_cards[:5])
    best = None
    for combo in combinations(all_cards, 5):
        r = hand_rank(list(combo))
        if best is None or r > best:
            best = r
    return best

def hand_name(rank):
    names = {8: "Straight Flush", 7: "Four of a Kind", 6: "Full House",
             5: "Flush", 4: "Straight", 3: "Three of a Kind",
             2: "Two Pair", 1: "Pair", 0: "High Card"}
    return names.get(rank[0], "Unknown")


# ============== TILE FIELD (per player) ==============

class PokerTile:
    """Decision tile: state = (hand_bucket, position, num_active_players)"""
    def __init__(self, hand_bucket: int, position: int, n_active: int):
        self.hand_bucket = hand_bucket
        self.position = position
        self.n_active = n_active
        self.state_str = f"h{hand_bucket}:pos{position}:act{n_active}"
        self.hash = hashlib.blake2b(self.state_str.encode(), digest_size=8).hexdigest()

        self.reflexes = {
            "fold":       {"score": 0.2, "chosen": 0, "won": 0},
            "check_call": {"score": 0.5, "chosen": 0, "won": 0},
            "raise":      {"score": 0.5, "chosen": 0, "won": 0},
            "bluff":      {"score": 0.3, "chosen": 0, "won": 0},
        }
        self.momentum = 0.0
        self.visits = 0

    def score_vector(self):
        """Return score vector for distance computation."""
        return np.array([self.reflexes[a]["score"] for a in sorted(self.reflexes.keys())])

    def to_json(self):
        return {
            "state": self.state_str,
            "visits": self.visits,
            "momentum": round(self.momentum, 3),
            "reflexes": {a: {"score": round(d["score"],3), "chosen": d["chosen"],
                            "won": d["won"]}
                        for a, d in self.reflexes.items()}
        }


class PlayerTileField:
    """Each player has their own tile field."""
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.tiles: Dict[str, PokerTile] = {}

    def get_tile(self, hand_bucket: int, position: int, n_active: int) -> PokerTile:
        key = f"h{hand_bucket}:pos{position}:act{n_active}"
        if key not in self.tiles:
            self.tiles[key] = PokerTile(hand_bucket, position, n_active)
        return self.tiles[key]

    def choose_action(self, tile: PokerTile, temperature: float = 0.3,
                      epsilon: float = 0.05) -> str:
        actions = sorted(tile.reflexes.keys())
        if random.random() < epsilon:
            return min(actions, key=lambda a: tile.reflexes[a]["chosen"])
        scores = np.array([tile.reflexes[a]["score"] for a in actions])
        if temperature > 0.01:
            probs = np.exp(scores / temperature)
            probs /= probs.sum()
            return actions[np.random.choice(len(actions), p=probs)]
        return actions[np.argmax(scores)]

    def record(self, tile: PokerTile, action: str, won: bool):
        tile.visits += 1
        tile.reflexes[action]["chosen"] += 1
        if won:
            tile.reflexes[action]["won"] += 1
            tile.momentum = min(tile.momentum + 0.1, 2.0)
        else:
            tile.momentum = max(tile.momentum - 0.05, -0.5)

    def evolve(self, lr=0.04, cap=0.05):
        for tile in self.tiles.values():
            for action, data in tile.reflexes.items():
                if data["chosen"] > 0:
                    wr = data["won"] / data["chosen"]
                    delta = max(-cap, min(cap, lr * (wr - data["score"])))
                    data["score"] = max(0.05, min(0.95, data["score"] + delta))

    def strategy_vector(self) -> np.ndarray:
        """Flatten all tile scores into a single vector for distance comparison."""
        if not self.tiles:
            return np.zeros(4)
        # Sort by key for consistency
        vecs = []
        for key in sorted(self.tiles.keys()):
            vecs.append(self.tiles[key].score_vector())
        return np.concatenate(vecs)

    def to_json(self):
        return {
            "player_id": self.player_id,
            "n_tiles": len(self.tiles),
            "tiles": {k: v.to_json() for k, v in sorted(self.tiles.items())}
        }


# ============== 3-PLAYER GAME ENGINE ==============

class ThreePlayerPoker:
    """
    Simplified 3-player poker:
    - Ante (1 chip each)
    - Deal hole cards
    - Deal community (flop + turn + river all at once for simplicity)
    - One round of betting (each player acts once in order)
    - Showdown among non-folded players

    Returns winner index (0, 1, 2) or -1 for tie.
    """

    def __init__(self):
        self.deck = make_deck()
        random.shuffle(self.deck)
        self.holes = [[], [], []]
        self.community = []
        self.pot = 0
        self.bets = [0, 0, 0]
        self.folded = [False, False, False]
        self.done = False

        # Deal hole cards
        for _ in range(2):
            for p in range(3):
                self.holes[p].append(self.deck.pop())

        # Deal all 5 community cards
        for _ in range(5):
            self.community.append(self.deck.pop())

        # Ante
        self.pot = 3
        self.bets = [1, 1, 1]

    def hand_strength_bucket(self, player: int) -> int:
        """Bucket hand strength 0-4."""
        rank = best_hand(self.holes[player], self.community)
        # rank[0] is 0-8, map to 0-4
        return min(4, rank[0] // 2)

    def play(self, fields: List[PlayerTileField], learning: bool = True) -> dict:
        """
        Play one hand with 3 players.
        Each player uses their tile field to decide.
        """
        tiles_used = [[] for _ in range(3)]
        actions_taken = [[] for _ in range(3)]

        n_active = 3
        for pos in range(3):
            if self.folded[pos]:
                continue

            h_bucket = self.hand_strength_bucket(pos)
            n_act = sum(1 for f in self.folded if not f)

            tile = fields[pos].get_tile(h_bucket, pos, n_act)
            tiles_used[pos].append(tile)

            T = 0.3 if learning else 0.1
            eps = 0.08 if learning else 0.02
            action = fields[pos].choose_action(tile, T, epsilon=eps)
            actions_taken[pos].append(action)

            if action == "fold":
                self.folded[pos] = True
                n_active -= 1
                if n_active <= 1:
                    # Everyone else folded, remaining player wins
                    winner = next(i for i in range(3) if not self.folded[i])
                    self._record_results(fields, tiles_used, actions_taken, winner)
                    return {"winner": winner, "fold": True, "pot": self.pot}
            elif action == "check_call":
                call_amount = max(self.bets) - self.bets[pos]
                self.bets[pos] += call_amount
                self.pot += call_amount
            elif action == "raise":
                call_amount = max(self.bets) - self.bets[pos]
                raise_amount = max(2, self.pot // 3)
                self.bets[pos] += call_amount + raise_amount
                self.pot += call_amount + raise_amount
            elif action == "bluff":
                call_amount = max(self.bets) - self.bets[pos]
                raise_amount = max(3, self.pot // 2)
                self.bets[pos] += call_amount + raise_amount
                self.pot += call_amount + raise_amount

        # Showdown
        active = [i for i in range(3) if not self.folded[i]]
        if len(active) == 0:
            # Shouldn't happen
            self._record_results(fields, tiles_used, actions_taken, -1)
            return {"winner": -1, "fold": False, "pot": self.pot}

        best_rank = None
        winner = active[0]
        for p in active:
            r = best_hand(self.holes[p], self.community)
            if best_rank is None or r > best_rank:
                best_rank = r
                winner = p

        self._record_results(fields, tiles_used, actions_taken, winner)
        return {
            "winner": winner, "fold": False, "pot": self.pot,
            "hands": {p: hand_name(best_hand(self.holes[p], self.community)) for p in active}
        }

    def _record_results(self, fields, tiles_used, actions_taken, winner):
        for p in range(3):
            for tile, action in zip(tiles_used[p], actions_taken[p]):
                fields[p].record(tile, action, p == winner)


# ============== STRATEGY DISTANCE ==============

def strategy_distance(f1: PlayerTileField, f2: PlayerTileField) -> float:
    """Euclidean distance between two players' strategy vectors."""
    v1 = f1.strategy_vector()
    v2 = f2.strategy_vector()
    # Pad to same length
    max_len = max(len(v1), len(v2))
    if max_len == 0:
        return 0.0
    v1 = np.pad(v1, (0, max_len - len(v1)))
    v2 = np.pad(v2, (0, max_len - len(v2)))
    return float(np.linalg.norm(v1 - v2))


def pairwise_distances(fields: List[PlayerTileField]) -> Dict[Tuple[int,int], float]:
    """Compute all pairwise distances."""
    dists = {}
    for i, j in [(0,1), (0,2), (1,2)]:
        dists[(i,j)] = strategy_distance(fields[i], fields[j])
    return dists


# ============== MAIN EXPERIMENT ==============

def run_experiment():
    print("=" * 70)
    print("MULTIPLAYER DIVERGENCE EXPERIMENT")
    print("3-Player Simplified Poker - Does Divergence Scale?")
    print("=" * 70)

    random.seed(42)
    np.random.seed(42)

    fields = [PlayerTileField(i) for i in range(3)]
    player_names = ["Alice", "Bob", "Carol"]

    # Tracking
    all_distances = []  # [(phase, hand, d01, d02, d12)]
    all_winrates = []   # [(phase, hand, wr0, wr1, wr2)]
    phase_results = {}

    HANDS_PER_PHASE = 500
    TOTAL_PHASES = 3

    for phase in range(1, TOTAL_PHASES + 1):
        phase_label = ["DEFAULT", "LEARNING", "ARMS RACE"][phase - 1]
        learning = phase >= 2
        print(f"\n{'─' * 60}")
        print(f"Phase {phase}: {phase_label} ({HANDS_PER_PHASE} hands)")
        print(f"{'─' * 60}")

        phase_wins = [0, 0, 0]
        phase_folds = 0
        phase_showdowns = 0

        for hand in range(HANDS_PER_PHASE):
            game = ThreePlayerPoker()
            result = game.play(fields, learning=learning)

            if result["winner"] >= 0:
                phase_wins[result["winner"]] += 1
            if result["fold"]:
                phase_folds += 1
            else:
                phase_showdowns += 1

            # Evolve every 50 hands during learning phases
            if learning and (hand + 1) % 50 == 0:
                for f in fields:
                    f.evolve(lr=0.04, cap=0.05)

            # Record metrics every 25 hands
            if (hand + 1) % 25 == 0:
                dists = pairwise_distances(fields)
                total_hands = (phase - 1) * HANDS_PER_PHASE + hand + 1
                wr = [w / (hand + 1) for w in phase_wins]
                all_distances.append((phase, total_hands,
                                      dists[(0,1)], dists[(0,2)], dists[(1,2)]))
                all_winrates.append((phase, total_hands, wr[0], wr[1], wr[2]))

        # Final phase stats
        total = HANDS_PER_PHASE
        print(f"\n  Win rates:  " +
              "  ".join(f"{player_names[i]}={phase_wins[i]/total:.1%}" for i in range(3)))
        print(f"  Folds: {phase_folds}  Showdowns: {phase_showdowns}")

        dists = pairwise_distances(fields)
        print(f"  Pairwise distances:")
        for (i,j), d in sorted(dists.items()):
            print(f"    {player_names[i]}↔{player_names[j]}: {d:.4f}")

        # Tile counts
        for i in range(3):
            print(f"  {player_names[i]} tiles: {len(fields[i].tiles)}")

        phase_results[phase] = {
            "wins": phase_wins,
            "win_rates": [w / total for w in phase_wins],
            "folds": phase_folds,
            "showdowns": phase_showdowns,
            "distances": {f"{i}-{j}": d for (i,j), d in dists.items()},
            "tiles": [len(f.tiles) for f in fields],
        }

    # ============== ANALYSIS ==============
    print(f"\n{'=' * 70}")
    print("DIVERGENCE ANALYSIS")
    print(f"{'=' * 70}")

    # Distance trajectory
    print("\n  Strategy Distance Over Time:")
    print(f"  {'Hand':>6s}  {'Alice↔Bob':>10s}  {'Alice↔Carol':>12s}  {'Bob↔Carol':>10s}  {'Mean':>8s}  {'Phase':>8s}")

    prev_mean = None
    monotonic_increasing = True
    for phase, hand, d01, d02, d12 in all_distances:
        mean_d = (d01 + d02 + d12) / 3
        if prev_mean is not None and mean_d < prev_mean - 0.001:
            monotonic_increasing = False
        prev_mean = mean_d
        phase_label = ["DEFAULT", "LEARN", "ARMS"][phase - 1]
        print(f"  {hand:6d}  {d01:10.4f}  {d02:12.4f}  {d12:10.4f}  {mean_d:8.4f}  {phase_label:>8s}")

    # Check convergence: does ANY pair converge?
    print(f"\n  Convergence Check (does any pair converge?):")
    pair_labels = [(0,1, "Alice↔Bob"), (0,2, "Alice↔Carol"), (1,2, "Bob↔Carol")]
    any_converged = False
    for i, j, label in pair_labels:
        # Extract distances for this pair during learning phases (2+)
        phase2_dists = [d for d in all_distances if d[0] >= 2]
        if phase2_dists:
            p2_pair = [{(0,1): d[2], (0,2): d[3], (1,2): d[4]}[(i,j)] for d in phase2_dists]
            early = np.mean(p2_pair[:3])
            late = np.mean(p2_pair[-3:])
            converged = late < early - 0.01
            trend = "CONVERGING ↓" if converged else "DIVERGING ↑"
            if converged:
                any_converged = True
            print(f"    {label}: early={early:.4f} → late={late:.4f}  {trend}")

    # Dominant player check
    print(f"\n  Dominant Player Check:")
    for phase in range(1, 4):
        wr = phase_results[phase]["win_rates"]
        best_p = max(range(3), key=lambda i: wr[i])
        best_wr = wr[best_p]
        expected = 1/3
        dominance = best_wr - expected
        print(f"    Phase {phase}: {player_names[best_p]} dominant ({best_wr:.1%}, "
              f"+{dominance:.1%} above expected 33.3%)")

    # Final summary
    final_dists = pairwise_distances(fields)
    mean_final = np.mean(list(final_dists.values()))

    # Initial distances (after phase 1)
    phase1_dists_data = [d for d in all_distances if d[0] == 1]
    if phase1_dists_data:
        last_p1 = phase1_dists_data[-1]
        mean_initial = np.mean([last_p1[2], last_p1[3], last_p1[4]])
    else:
        mean_initial = 0

    print(f"\n{'=' * 70}")
    print("VERDICT")
    print(f"{'=' * 70}")

    divergence_ratio = mean_final / mean_initial if mean_initial > 0 else float('inf')
    print(f"  Mean distance:  Phase 1 end = {mean_initial:.4f}  →  Phase 3 end = {mean_final:.4f}")
    print(f"  Divergence ratio: {divergence_ratio:.2f}x")
    print(f"  Monotonically increasing: {monotonic_increasing}")
    print(f"  Any pair converged: {any_converged}")

    if divergence_ratio > 1.5:
        verdict = "STRONGLY DIVERGENT - 3-player amplifies divergence vs 2-player"
    elif divergence_ratio > 1.0:
        verdict = "DIVERGENT - divergence holds but not dramatically stronger than 2-player"
    elif divergence_ratio > 0.8:
        verdict = "WEAKLY DIVERGENT - some convergence pressure from multiplayer dynamics"
    else:
        verdict = "CONVERGENT - multiplayer dynamics create equilibrium pressure"

    print(f"\n  → {verdict}")

    # ============== SAVE RESULTS ==============
    output = {
        "experiment": "multiplayer_divergence",
        "description": "3-player poker divergence test",
        "hypothesis": "3-player is MORE divergent than 2-player due to larger strategy space",
        "phases": phase_results,
        "distance_trajectory": [
            {"phase": p, "hand": h, "d_alice_bob": d01, "d_alice_carol": d02, "d_bob_carol": d12}
            for p, h, d01, d02, d12 in all_distances
        ],
        "winrate_trajectory": [
            {"phase": p, "hand": h, "wr_alice": w0, "wr_bob": w1, "wr_carol": w2}
            for p, h, w0, w1, w2 in all_winrates
        ],
        "analysis": {
            "mean_distance_initial": round(mean_initial, 4),
            "mean_distance_final": round(mean_final, 4),
            "divergence_ratio": round(divergence_ratio, 4),
            "monotonic_increasing": monotonic_increasing,
            "any_pair_converged": any_converged,
        },
        "verdict": verdict,
        "player_fields": {f"player_{i}": fields[i].to_json() for i in range(3)},
    }

    out_path = os.path.expanduser("~/repos/zeroclaw-arena/multiplayer-divergence-results.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return output


if __name__ == "__main__":
    run_experiment()
