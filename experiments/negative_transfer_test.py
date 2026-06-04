"""
Negative Transfer Test — Can learning from one game HURT another?

The cross-game mining found reward anti-correlation between TTT and C4.
This experiment tests: does loading TTT knowledge into a C4 player 
actually make it WORSE than random?

Hypothesis: Some TTT patterns actively hurt C4 performance (negative transfer).
If confirmed, transfer learning needs a FILTER, not just a pipe.

Method:
1. Build a C4 player that uses TTT vectors (like transfer_learning.py)
2. Build a "filtered" player that only transfers POSITIVE-reward patterns
3. Build a "reversed" player that inverts TTT rewards (win↔lose)
4. Play 500 games each vs random
5. Compare: unfiltered transfer vs filtered transfer vs reversed vs random
"""

import sqlite3
import json
import hashlib
import numpy as np
import random
import time
import os


def hash_embed(text, dim=64):
    h = hashlib.blake2b(text.encode(), digest_size=dim).digest()
    v = np.array([b/255.0 for b in h], dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-10)


class TransferPlayer:
    """Uses source game vectors to play target game."""
    def __init__(self, source_db, strategy="unfiltered"):
        self.strategy = strategy
        self.entries = []
        
        conn = sqlite3.connect(source_db)
        for row in conn.execute("SELECT vector, metadata FROM vectors"):
            vec = [b/255.0 for b in row[0]]
            meta = json.loads(row[1])
            
            if strategy == "positive_only":
                if meta.get('reward', 0) > 0.3:
                    self.entries.append((vec, meta))
            elif strategy == "negative_only":
                if meta.get('reward', 0) < -0.3:
                    self.entries.append((vec, meta))
            elif strategy == "reversed":
                meta['reward'] = -meta.get('reward', 0)
                self.entries.append((vec, meta))
            else:  # unfiltered
                self.entries.append((vec, meta))
        
        conn.close()
        
        self.vectors = np.array([e[0] for e in self.entries]) if self.entries else np.array([]).reshape(0, 64)
        self.rewards = np.array([e[1].get('reward', 0) for e in self.entries])
    
    def choose_action(self, state_str, legal_actions):
        if len(self.vectors) == 0 or not legal_actions:
            return random.choice(legal_actions) if legal_actions else ''
        
        q = hash_embed(state_str)
        sims = self.vectors @ q
        
        # Weight actions by similarity to high-reward source states
        action_scores = {a: 0.0 for a in legal_actions}
        
        top_k = min(20, len(self.entries))
        top_indices = np.argsort(sims)[-top_k:]
        
        for idx in top_indices:
            source_meta = self.entries[idx][1]
            source_action = source_meta.get('action', '')
            weight = sims[idx] * self.rewards[idx]
            
            # Map source action to target actions
            # For grid games, similar positions suggest similar strategies
            # Center preference, blocking, etc.
            if source_action in legal_actions:
                action_scores[source_action] += weight
            else:
                # Distribute weight based on action similarity
                for a in legal_actions:
                    # Simple: prefer center-adjacent actions
                    try:
                        action_num = int(a)
                        center = 3  # Connect4 center column
                        proximity = 1.0 / (1 + abs(action_num - center))
                        action_scores[a] += weight * proximity * 0.3
                    except:
                        pass
        
        best = max(action_scores, key=action_scores.get)
        if action_scores[best] == 0:
            return random.choice(legal_actions)
        return best


def play_games(game_class, player_fn, n_games=500):
    """Play n games using player_fn to choose X's moves, random for O."""
    wins = 0
    losses = 0
    draws = 0
    
    for _ in range(n_games):
        game = game_class()
        while not game.done:
            actions = game.legal_actions()
            if not actions:
                break
            
            if hasattr(game, 'current') and game.current == 'X':
                action = player_fn(str(game.state()), actions)
            else:
                action = random.choice(actions)
            
            game.step(action)
        
        winner = getattr(game, 'winner', None)
        if winner == 'X':
            wins += 1
        elif winner == 'O':
            losses += 1
        else:
            draws += 1
    
    return wins, losses, draws


def run_experiment():
    print("=" * 60)
    print("NEGATIVE TRANSFER TEST")
    print("Can learning from one game HURT another?")
    print("=" * 60)
    
    # Load source DB
    ttt_db = "/tmp/zeroclaw-sandbox/zeroclaw-tictactoe/vectors.db"
    if not os.path.exists(ttt_db):
        print("No tic-tac-toe DB found — run arena first")
        return
    
    from zeroclaw import Connect4, TicTacToe
    
    strategies = {
        "random": None,
        "unfiltered": TransferPlayer(ttt_db, "unfiltered"),
        "positive_only": TransferPlayer(ttt_db, "positive_only"),
        "negative_only": TransferPlayer(ttt_db, "negative_only"),
        "reversed": TransferPlayer(ttt_db, "reversed"),
    }
    
    print(f"\nSource: TTT ({len(strategies['unfiltered'].entries)} transitions)")
    print(f"Target: Connect4 (500 games each)")
    print()
    
    results = {}
    for name, player in strategies.items():
        print(f"Testing {name}...")
        
        if player is None:
            # Random baseline
            def player_fn(state, actions): return random.choice(actions)
        else:
            def player_fn(state, actions, p=player): return p.choose_action(state, actions)
        
        wins, losses, draws = play_games(Connect4, player_fn, 500)
        wr = wins / 500
        results[name] = {"wins": wins, "losses": losses, "draws": draws, "win_rate": wr}
        print(f"  {name}: {wins}W/{losses}L/{draws}D = {wr:.1%}")
    
    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    random_wr = results["random"]["win_rate"]
    for name, r in results.items():
        delta = (r["win_rate"] - random_wr) * 100
        symbol = "✅" if delta > 2 else ("❌" if delta < -2 else "➡️")
        print(f"  {symbol} {name}: {r['win_rate']:.1%} ({delta:+.1f}pp vs random)")
    
    # Key questions
    unf_wr = results["unfiltered"]["win_rate"]
    pos_wr = results["positive_only"]["win_rate"]
    neg_wr = results["negative_only"]["win_rate"]
    rev_wr = results["reversed"]["win_rate"]
    
    print()
    if neg_wr > random_wr + 0.05:
        print("⚠️ NEGATIVE TRANSFER: Loss patterns from TTT help C4 more than random!")
        print("   This suggests the vector space has meaningful structure.")
    elif rev_wr < random_wr - 0.05:
        print("✅ REWARD SIGNAL MATTERS: Reversed rewards hurt performance.")
        print("   Transfer learning needs reward-aware filtering.")
    elif pos_wr > unf_wr + 0.05:
        print("✅ FILTERING HELPS: Positive-only transfer beats unfiltered.")
        print("   Reward-based filtering is essential for transfer.")
    else:
        print("➡️ NEUTRAL: Transfer doesn't significantly help or hurt at this scale.")
    
    # Save
    out = os.path.expanduser("~/repos/zeroclaw-arena/negative-transfer-results.json")
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    run_experiment()
