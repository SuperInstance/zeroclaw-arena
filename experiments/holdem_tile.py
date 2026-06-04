"""
Texas Hold'em × Tile Field — The Penrose Model of Competitive Intelligence

Each betting round is a tile activation:
  Opening (hole cards) → Flop (3 community) → Turn (1) → River (1)
  
Each round reveals information and constrains the negative space.
The "keen eye" reads what the opponent's bets say about their HIDDEN cards.
Bluffing = weaponizing your own negative space.

Different "dice" = different randomness flavors:
  - Hole cards: uniform random (52 choose 2)
  - Community cards: conditional probability (given what's visible)
  - Opponent actions: strategic (game-theoretic mixed strategy)
  - Bluffing: deception layer (negative space weaponized)

The Penrose spiral: each zoom level reveals the same structure.
  - Micro: one hand (5 cards visible, 47 unknown)
  - Meso: one session (patterns emerge, player styles identified)
  - Macro: career (the negative space of ALL players converges to Nash equilibrium)
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

def hand_rank(cards: List[Tuple[str,str]]) -> Tuple:
    """Evaluate a 5-card hand. Returns (category, tiebreakers...)"""
    ranks = sorted([RANK_VAL[c[0]] for c in cards], reverse=True)
    suits = [c[1] for c in cards]
    is_flush = len(set(suits)) == 1
    
    # Check straight
    unique = sorted(set(ranks), reverse=True)
    is_straight = False
    straight_high = 0
    if len(unique) == 5:
        if unique[0] - unique[4] == 4:
            is_straight = True
            straight_high = unique[0]
        # Ace-low straight
        if unique == [12, 3, 2, 1, 0]:
            is_straight = True
            straight_high = 3
    
    # Count ranks
    counts = defaultdict(int)
    for r in ranks:
        counts[r] += 1
    groups = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    
    if is_straight and is_flush:
        return (8, straight_high)  # Straight flush
    if groups[0][1] == 4:
        return (7, groups[0][0], groups[1][0])  # Four of a kind
    if groups[0][1] == 3 and groups[1][1] == 2:
        return (6, groups[0][0], groups[1][0])  # Full house
    if is_flush:
        return (5,) + tuple(ranks)  # Flush
    if is_straight:
        return (4, straight_high)  # Straight
    if groups[0][1] == 3:
        return (3, groups[0][0], groups[1][0], groups[2][0])  # Three of a kind
    if groups[0][1] == 2 and groups[1][1] == 2:
        return (2, groups[0][0], groups[1][0], groups[2][0])  # Two pair
    if groups[0][1] == 2:
        return (1, groups[0][0], groups[1][0], groups[2][0], groups[3][0])  # One pair
    return (0,) + tuple(ranks)  # High card

def best_hand(hole: List[Tuple], community: List[Tuple]) -> Tuple:
    """Find best 5-card hand from hole + community."""
    all_cards = hole + community
    if len(all_cards) < 5:
        return hand_rank(all_cards[:5])
    best = None
    for combo in combinations(all_cards, 5):
        r = hand_rank(list(combo))
        if best is None or r > best:
            best = r
    return best

def hand_name(rank: Tuple) -> str:
    names = {8: "Straight Flush", 7: "Four of a Kind", 6: "Full House",
             5: "Flush", 4: "Straight", 3: "Three of a Kind",
             2: "Two Pair", 1: "Pair", 0: "High Card"}
    return names.get(rank[0], "Unknown")

def card_str(card):
    return f"{card[0]}{card[1]}"


# ============== TILE FIELD FOR POKER ==============

class PokerTile:
    """
    A poker decision point as a tile in the Penrose field.
    
    State = (stage, hand_strength, pot_size_relative, position)
    Each tile contains:
      - Reflexes: fold, check/call, raise_small, raise_big, bluff
      - Score: learned from outcomes
      - Negative space: what the opponent DOESN'T know about our hand
    """
    def __init__(self, stage: str, hand_bucket: int, pot_bucket: int, position: int):
        self.stage = stage
        self.hand_bucket = hand_bucket  # 0=worst, 4=best
        self.pot_bucket = pot_bucket    # 0=small, 2=large
        self.position = position        # 0=first, 1=last (dealer)
        
        self.state_str = f"{stage}:h{hand_bucket}:p{pot_bucket}:pos{position}"
        self.hash = hashlib.blake2b(self.state_str.encode(), digest_size=8).hexdigest()
        
        self.reflexes = {
            "fold": {"score": 0.3, "chosen": 0, "won": 0},
            "check_call": {"score": 0.5, "chosen": 0, "won": 0},
            "raise_small": {"score": 0.5, "chosen": 0, "won": 0},
            "raise_big": {"score": 0.4, "chosen": 0, "won": 0},
            "bluff": {"score": 0.3, "chosen": 0, "won": 0},
        }
        self.momentum = 0.0
        self.visits = 0
    
    def to_json(self):
        return {
            "state": self.state_str,
            "visits": self.visits,
            "momentum": round(self.momentum, 3),
            "reflexes": {a: {"score": round(d["score"],3), "chosen": d["chosen"],
                            "won": d["won"], "wr": f"{d['won']/max(d['chosen'],1):.0%}"}
                        for a, d in self.reflexes.items()}
        }


class PokerTileField:
    """The Penrose field of poker decisions."""
    
    def __init__(self):
        self.tiles: Dict[str, PokerTile] = {}
        self.STAGES = ["preflop", "flop", "turn", "river"]
    
    def get_tile(self, stage: str, hand_bucket: int, pot_bucket: int, position: int) -> PokerTile:
        key = f"{stage}:h{hand_bucket}:p{pot_bucket}:pos{position}"
        if key not in self.tiles:
            self.tiles[key] = PokerTile(stage, hand_bucket, pot_bucket, position)
        return self.tiles[key]
    
    def choose_action(self, tile: PokerTile, temperature: float = 0.3, 
                      epsilon: float = 0.05) -> str:
        """Choose action using softmax + epsilon-greedy."""
        actions = list(tile.reflexes.keys())
        
        # Epsilon-greedy: explore least-chosen
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


# ============== POKER GAME ENGINE ==============

class HoldemHand:
    """One hand of Texas Hold'em."""
    
    STAGES = ["preflop", "flop", "turn", "river"]
    
    def __init__(self):
        self.deck = make_deck()
        random.shuffle(self.deck)
        self.hole = [[], []]  # Player 0 (tile), Player 1 (random)
        self.community = []
        self.pot = 0
        self.bets = [0, 0]
        self.folded = [False, False]
        self.stage = 0  # 0=preflop, 1=flop, 2=turn, 3=river
        self.done = False
        self.actions_history = []
        
        # Deal hole cards
        for _ in range(2):
            self.hole[0].append(self.deck.pop())
            self.hole[1].append(self.deck.pop())
        self.pot = 2  # Blinds
    
    def deal_community(self, n: int):
        """Deal community cards."""
        for _ in range(n):
            self.community.append(self.deck.pop())
    
    def hand_strength_bucket(self, player: int) -> int:
        """Bucket hand strength 0-4 (for tile state)."""
        if not self.community:
            # Preflop: use hole card rank
            r1 = RANK_VAL[self.hole[player][0][0]]
            r2 = RANK_VAL[self.hole[player][1][0]]
            suited = self.hole[player][0][1] == self.hole[player][1][1]
            score = (r1 + r2) / 24.0 + (0.1 if suited else 0) + (0.1 if r1 == r2 else 0)
            return min(4, int(score * 5))
        
        # With community: evaluate current best hand
        rank = best_hand(self.hole[player], self.community)
        return min(4, rank[0])  # 0-4 based on hand category
    
    def pot_bucket(self) -> int:
        """Bucket pot size 0-2."""
        if self.pot < 6: return 0
        if self.pot < 15: return 1
        return 2
    
    def play_round(self, actions: List[str]) -> Optional[int]:
        """Play one betting round. Returns winner if hand is over."""
        for i, action in enumerate(actions):
            player = i % 2
            if self.folded[player]:
                continue
            
            if action == "fold":
                self.folded[player] = True
                self.done = True
                return 1 - player  # Other player wins
            elif action == "check_call":
                call_amount = max(self.bets) - self.bets[player]
                self.bets[player] += call_amount
                self.pot += call_amount
            elif action == "raise_small":
                call_amount = max(self.bets) - self.bets[player]
                raise_amount = max(2, self.pot // 3)
                self.bets[player] += call_amount + raise_amount
                self.pot += call_amount + raise_amount
            elif action == "raise_big":
                call_amount = max(self.bets) - self.bets[player]
                raise_amount = max(4, self.pot)
                self.bets[player] += call_amount + raise_amount
                self.pot += call_amount + raise_amount
            elif action == "bluff":
                # Bluff = raise_big but with weak hand (tile field decides intent)
                call_amount = max(self.bets) - self.bets[player]
                raise_amount = max(4, self.pot)
                self.bets[player] += call_amount + raise_amount
                self.pot += call_amount + raise_amount
            
            self.actions_history.append((self.STAGES[self.stage], player, action))
        
        return None
    
    def showdown(self) -> int:
        """Compare hands at showdown."""
        r0 = best_hand(self.hole[0], self.community)
        r1 = best_hand(self.hole[1], self.community)
        if r0 > r1: return 0
        if r1 > r0: return 1
        return -1  # Tie (split pot)
    
    def play(self, field: PokerTileField, opponent_strategy="random") -> dict:
        """Play a complete hand."""
        tiles_used = []
        tile_actions = []
        
        for stage_idx in range(4):
            self.stage = stage_idx
            
            # Deal community cards
            if stage_idx == 1:
                self.deal_community(3)  # Flop
            elif stage_idx in [2, 3]:
                self.deal_community(1)  # Turn/River
            
            # Player 0 (tile field) acts
            h_bucket = self.hand_strength_bucket(0)
            p_bucket = self.pot_bucket()
            tile = field.get_tile(self.STAGES[self.stage], h_bucket, p_bucket, 0)
            
            # Temperature decreases as hand progresses (more info = more decisive)
            T = max(0.15, 0.5 - stage_idx * 0.1)
            action = field.choose_action(tile, T, epsilon=0.05)
            tiles_used.append(tile)
            tile_actions.append(action)
            
            # Player 1 acts
            if opponent_strategy == "random":
                p1_actions = ["fold", "check_call", "raise_small", "raise_big", "bluff"]
                p1_action = random.choice(p1_actions[1:])  # Don't randomly fold
            else:
                p1_action = "check_call"
            
            result = self.play_round([action, p1_action])
            if result is not None:
                # Someone folded
                for t, a in zip(tiles_used, tile_actions):
                    field.record(t, a, result == 0)
                return {
                    "winner": result, "fold": True, "stage": self.STAGES[stage_idx],
                    "pot": self.pot, "tiles": len(tiles_used),
                    "tile_actions": [(t.state_str, a) for t, a in zip(tiles_used, tile_actions)],
                    "p1_action": p1_action,
                }
        
        # Showdown
        winner = self.showdown()
        won = winner == 0
        for t, a in zip(tiles_used, tile_actions):
            field.record(t, a, won)
        
        return {
            "winner": winner, "fold": False, 
            "stage": "showdown",
            "pot": self.pot, "tiles": len(tiles_used),
            "hand0": hand_name(best_hand(self.hole[0], self.community)),
            "hand1": hand_name(best_hand(self.hole[1], self.community)),
            "tile_actions": [(t.state_str, a) for t, a in zip(tiles_used, tile_actions)],
        }


# ============== MAIN ==============

def run_holdem_arena():
    print("=" * 70)
    print("TEXAS HOLD'EM × TILE FIELD — Penrose Model of Competitive Intelligence")
    print("=" * 70)
    
    field = PokerTileField()
    
    # Phase 1: Learning (1000 hands)
    print("\n--- Phase 1: Learning (1000 hands) ---")
    wins = 0
    folds_won = 0
    showdowns = 0
    showdown_wins = 0
    
    for hand_num in range(1000):
        game = HoldemHand()
        result = game.play(field, "random")
        
        if result["winner"] == 0:
            wins += 1
            if result["fold"]:
                folds_won += 1
            else:
                showdown_wins += 1
        if not result["fold"]:
            showdowns += 1
        
        if (hand_num + 1) % 200 == 0:
            field.evolve(lr=0.04, cap=0.05)
            wr = wins / (hand_num + 1)
            n_tiles = len(field.tiles)
            print(f"  Hand {hand_num+1}: wr={wr:.1%} folds_won={folds_won} "
                  f"showdown_wr={showdown_wins}/{showdowns} tiles={n_tiles}")
    
    # Phase 2: Exploitation (500 hands, low temperature)
    print("\n--- Phase 2: Exploitation (500 hands, low T) ---")
    phase2_wins = 0
    phase2_showdowns = 0
    phase2_showdown_wins = 0
    
    for hand_num in range(500):
        game = HoldemHand()
        result = game.play(field, "random")
        
        if result["winner"] == 0:
            phase2_wins += 1
            if not result["fold"]:
                phase2_showdown_wins += 1
        if not result["fold"]:
            phase2_showdowns += 1
    
    # Random baseline
    random_wins = 0
    for _ in range(500):
        game = HoldemHand()
        # Both players random
        for stage_idx in range(4):
            if stage_idx == 1: game.deal_community(3)
            elif stage_idx in [2,3]: game.deal_community(1)
            actions = [random.choice(["check_call","raise_small","raise_big"]),
                      random.choice(["check_call","raise_small","raise_big"])]
            result = game.play_round(actions)
            if result is not None:
                if result == 0: random_wins += 1
                break
        else:
            w = game.showdown()
            if w == 0: random_wins += 1
    
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    
    p1_wr = wins / 1000
    p2_wr = phase2_wins / 500
    rand_wr = random_wins / 500
    
    print(f"  Learning (1000 hands):  {p1_wr:.1%}")
    print(f"  Exploitation (500):     {p2_wr:.1%}")
    print(f"  Random baseline:        {rand_wr:.1%}")
    print(f"  Tile advantage:         {(p2_wr - rand_wr)*100:+.1f}pp")
    print(f"  Tiles learned:          {len(field.tiles)}")
    
    # Show evolved strategy by stage
    print(f"\n  EVOLVED STRATEGY BY STAGE:")
    for stage in ["preflop", "flop", "turn", "river"]:
        stage_tiles = [t for t in field.tiles.values() if t.stage == stage and t.visits > 5]
        if not stage_tiles:
            continue
        
        # Aggregate reflex scores
        reflex_totals = defaultdict(lambda: {"score": 0, "count": 0})
        for t in stage_tiles:
            for a, d in t.reflexes.items():
                reflex_totals[a]["score"] += d["score"]
                reflex_totals[a]["count"] += 1
        
        avg_scores = {a: d["score"]/d["count"] for a, d in reflex_totals.items()}
        best = max(avg_scores, key=avg_scores.get)
        
        print(f"    {stage:8s}: best={best:12s} (avg={avg_scores[best]:.3f}) | "
              + " | ".join(f"{a[:5]}={s:.2f}" for a, s in sorted(avg_scores.items(), key=lambda x: -x[1])))
    
    # Show bluffing patterns
    print(f"\n  BLUFFING ANALYSIS:")
    bluff_tiles = [(t.state_str, t.reflexes["bluff"]["score"]) 
                   for t in field.tiles.values() if t.reflexes["bluff"]["chosen"] > 3]
    bluff_tiles.sort(key=lambda x: -x[1])
    for state, score in bluff_tiles[:5]:
        print(f"    {state}: bluff_score={score:.3f}")
    
    # Save
    output = {
        "learning_wr": p1_wr,
        "exploitation_wr": p2_wr,
        "random_wr": rand_wr,
        "advantage_pp": (p2_wr - rand_wr) * 100,
        "tiles_learned": len(field.tiles),
        "tiles": {k: v.to_json() for k, v in field.tiles.items()},
    }
    out = os.path.expanduser("~/repos/zeroclaw-arena/holdem-tile-results.json")
    with open(out, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    run_holdem_arena()
