"""
Reward Filter Ablation — Find the Optimal Transfer Threshold

Previous result: positive-only (reward > 0.3) = 67.2%
Question: What threshold maximizes transfer performance?

Test thresholds: -1.0 (everything), -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 0.8, 1.0 (only perfect wins)

For each threshold:
1. Filter source vectors to only those with reward > threshold
2. Play 500 Connect4 games using filtered TTT knowledge
3. Measure win rate

This directly answers: how much negative space should we model?
"""

import sqlite3
import json
import hashlib
import numpy as np
import random
import os


def hash_embed(text, dim=64):
    h = hashlib.blake2b(text.encode(), digest_size=dim).digest()
    v = np.array([b/255.0 for b in h], dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-10)


def test_threshold(source_db, threshold, n_games=500):
    """Test a specific reward filter threshold."""
    entries = []
    conn = sqlite3.connect(source_db)
    for row in conn.execute("SELECT vector, metadata FROM vectors"):
        meta = json.loads(row[1])
        if meta.get('reward', 0) > threshold:
            vec = [b/255.0 for b in row[0]]
            entries.append((vec, meta))
    conn.close()
    
    if not entries:
        return None, 0
    
    vectors = np.array([e[0] for e in entries])
    rewards = np.array([e[1].get('reward', 0) for e in entries])
    
    from zeroclaw import Connect4
    
    wins = 0
    for _ in range(n_games):
        game = Connect4()
        while not game.done:
            actions = game.legal_actions()
            if not actions: break
            
            if game.current == 'X':
                state = str(game.state())
                q = hash_embed(state)
                sims = vectors @ q
                
                action_scores = {}
                for a in actions:
                    score = 0.0
                    # Center preference (transferred)
                    if a in ['3', '4']:
                        score += 0.2
                    # Similarity-weighted reward
                    top_k = min(10, len(entries))
                    top_idx = np.argsort(sims)[-top_k:]
                    for idx in top_idx:
                        score += sims[idx] * rewards[idx] * 0.1
                    action_scores[a] = score
                
                action = max(action_scores, key=action_scores.get) if action_scores else random.choice(actions)
            else:
                action = random.choice(actions)
            
            game.step(action)
        
        if getattr(game, 'winner', None) == 'X':
            wins += 1
    
    return wins / n_games, len(entries)


def main():
    print("=" * 60)
    print("REWARD FILTER ABLATION — Optimal Transfer Threshold")
    print("=" * 60)
    
    ttt_db = "/tmp/zeroclaw-sandbox/zeroclaw-tictactoe/vectors.db"
    if not os.path.exists(ttt_db):
        print("No TTT DB found")
        return
    
    thresholds = [-1.0, -0.5, -0.3, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
    
    print(f"\n{'Threshold':>10} {'Vectors':>8} {'Win Rate':>10} {'vs Random':>10}")
    print("-" * 45)
    
    results = []
    for t in thresholds:
        wr, n_vecs = test_threshold(ttt_db, t, 500)
        if wr is not None:
            results.append((t, wr, n_vecs))
            vs_random = wr - 0.548  # baseline from previous experiment
            print(f"{t:>10.1f} {n_vecs:>8d} {wr:>10.1%} {vs_random:>+10.1%}")
    
    # Find optimal
    best = max(results, key=lambda x: x[1])
    print(f"\n🏆 Optimal threshold: {best[0]:.1f} → {best[1]:.1%} with {best[2]} vectors")
    
    # Save
    output = {
        'thresholds': [{'threshold': t, 'win_rate': wr, 'n_vectors': n} for t, wr, n in results],
        'optimal_threshold': best[0],
        'optimal_win_rate': best[1],
    }
    
    out = os.path.expanduser("~/repos/zeroclaw-arena/reward-filter-ablation.json")
    with open(out, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
