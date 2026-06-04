"""
Cross-Game GPU Pattern Mining — RTX 4050

Loads ALL game vector DBs into GPU memory simultaneously.
Finds cross-game patterns: states in tic-tac-toe that resemble states in Connect4.
Uses the GPU for massive parallel similarity computation.

This is novel: most game AI treats each game independently.
We're testing if knowledge can transfer across game boundaries.
"""

import torch
import sqlite3
import hashlib
import numpy as np
import json
import os
import time
from collections import defaultdict


def hash_embed(text, dim=64):
    h = hashlib.blake2b(text.encode(), digest_size=dim).digest()
    v = np.array([b/255.0 for b in h], dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-10)


class CrossGameMiner:
    def __init__(self, device='cuda'):
        self.device = device
        self.games = {}  # game_name -> (vectors_gpu, metadata)
        
    def load_game(self, name, db_path):
        """Load a game's vector DB into GPU memory."""
        if not os.path.exists(db_path):
            print(f"  {name}: DB not found at {db_path}")
            return False
        
        conn = sqlite3.connect(db_path)
        vectors = []
        metadata = []
        for row in conn.execute("SELECT vector, metadata FROM vectors"):
            v = np.array([b/255.0 for b in row[0]], dtype=np.float32)
            v /= (np.linalg.norm(v) + 1e-10)
            vectors.append(v)
            import json as _json
            meta = _json.loads(row[1])
            metadata.append(meta)
        conn.close()
        
        if not vectors:
            print(f"  {name}: no vectors in DB")
            return False
        
        vec_tensor = torch.from_numpy(np.array(vectors)).to(self.device)
        self.games[name] = (vec_tensor, metadata)
        
        vram_mb = vec_tensor.element_size() * vec_tensor.nelement() / 1024**2
        print(f"  {name}: {len(vectors)} vectors loaded, {vram_mb:.1f} MB VRAM")
        return True
    
    def cross_game_similarity(self, game_a, game_b, top_k=20):
        """Find most similar states between two games on GPU."""
        vecs_a, meta_a = self.games[game_a]
        vecs_b, meta_b = self.games[game_b]
        
        # Full cross-similarity matrix: [N_a, N_b]
        # Chunk to avoid OOM
        chunk_size = 2000
        top_results = []
        
        for i in range(0, len(vecs_a), chunk_size):
            chunk = vecs_a[i:i+chunk_size]
            sims = chunk @ vecs_b.T  # [chunk, N_b]
            
            vals, idxs = torch.topk(sims.flatten(), min(top_k, sims.numel()))
            for val, idx in zip(vals, idxs):
                r = idx.item() // len(vecs_b)
                c = idx.item() % len(vecs_b)
                top_results.append((i + r, c, val.item()))
        
        top_results.sort(key=lambda x: -x[2])
        return top_results[:top_k]
    
    def mine_all_pairs(self, top_k=10):
        """Mine patterns across all game pairs."""
        game_names = list(self.games.keys())
        all_patterns = {}
        
        for i, game_a in enumerate(game_names):
            for j, game_b in enumerate(game_names):
                if i >= j:
                    continue
                
                print(f"\n  Mining {game_a} ↔ {game_b}...")
                start = time.perf_counter()
                patterns = self.cross_game_similarity(game_a, game_b, top_k)
                elapsed = time.perf_counter() - start
                
                vecs_a, meta_a = self.games[game_a]
                vecs_b, meta_b = self.games[game_b]
                
                print(f"    Top patterns ({elapsed*1000:.0f}ms):")
                for idx_a, idx_b, sim in patterns[:5]:
                    ma = meta_a[idx_a] if idx_a < len(meta_a) else {}
                    mb = meta_b[idx_b] if idx_b < len(meta_b) else {}
                    print(f"      [{game_a}:{idx_a}] ↔ [{game_b}:{idx_b}] sim={sim:.4f}")
                    print(f"        A: action={ma.get('action','?')} reward={ma.get('reward','?'):.2f}")
                    print(f"        B: action={mb.get('action','?')} reward={mb.get('reward','?'):.2f}")
                
                all_patterns[f"{game_a}_vs_{game_b}"] = [
                    {"idx_a": a, "idx_b": b, "similarity": s,
                     "meta_a": meta_a[a] if a < len(meta_a) else {},
                     "meta_b": meta_b[b] if b < len(meta_b) else {}}
                    for a, b, s in patterns
                ]
        
        return all_patterns
    
    def find_high_reward_patterns(self, min_reward=0.5, min_sim=0.6):
        """Find states that are similar across games AND had high rewards."""
        game_names = list(self.games.keys())
        insights = []
        
        for i, game_a in enumerate(game_names):
            for j, game_b in enumerate(game_names):
                if i >= j:
                    continue
                
                vecs_a, meta_a = self.games[game_a]
                vecs_b, meta_b = self.games[game_b]
                
                # Get high-reward states from each game
                high_a = [(idx, meta) for idx, meta in enumerate(meta_a) 
                          if abs(meta.get('reward', 0)) >= min_reward]
                high_b = [(idx, meta) for idx, meta in enumerate(meta_b)
                          if abs(meta.get('reward', 0)) >= min_reward]
                
                if not high_a or not high_b:
                    continue
                
                # Compute cross-similarity of high-reward states only
                a_vecs = torch.stack([vecs_a[idx] for idx, _ in high_a])
                b_vecs = torch.stack([vecs_b[idx] for idx, _ in high_b])
                
                sims = a_vecs @ b_vecs.T
                
                # Find pairs above threshold
                mask = sims > min_sim
                pairs = torch.nonzero(mask, as_tuple=False)
                
                for pair in pairs[:20]:
                    ai, bi = pair[0].item(), pair[1].item()
                    sim = sims[ai, bi].item()
                    ma = high_a[ai][1]
                    mb = high_b[bi][1]
                    
                    insights.append({
                        "game_a": game_a,
                        "game_b": game_b,
                        "similarity": sim,
                        "reward_a": ma.get('reward', 0),
                        "reward_b": mb.get('reward', 0),
                        "action_a": ma.get('action', '?'),
                        "action_b": mb.get('action', '?'),
                    })
        
        insights.sort(key=lambda x: -x['similarity'])
        return insights[:30]


def main():
    print("=" * 60)
    print("CROSS-GAME GPU PATTERN MINING — RTX 4050")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM: {total_vram:.1f} GB")
    
    miner = CrossGameMiner(device)
    
    # Load all game DBs
    print("\nLoading game databases...")
    db_base = "/tmp/zeroclaw-sandbox"
    games_to_load = {
        'tictactoe': f"{db_base}/zeroclaw-tictactoe/vectors.db",
        'connect4': f"{db_base}/zeroclaw-connect4/vectors.db",
        'blackjack': f"{db_base}/zeroclaw-blackjack/vectors.db",
        'chess': f"{db_base}/zeroclaw-chess/vectors.db",
    }
    
    for name, path in games_to_load.items():
        miner.load_game(name, path)
    
    loaded = list(miner.games.keys())
    
    if len(loaded) < 2:
        print("\nNeed at least 2 game DBs loaded!")
        # Create synthetic data for testing
        print("Creating synthetic data...")
        for name in ['game_a', 'game_b', 'game_c']:
            n = 1000
            vecs = torch.randn(n, 64, device=device)
            vecs /= vecs.norm(dim=1, keepdim=True)
            meta = [{'reward': np.random.randn(), 'action': str(np.random.randint(0,9))} for _ in range(n)]
            miner.games[name] = (vecs, meta)
            print(f"  {name}: {n} synthetic vectors")
        loaded = list(miner.games.keys())
    
    print(f"\nVRAM used: {torch.cuda.memory_allocated(0)/1024**2:.1f} MB")
    
    # Mine all pairs
    print("\n" + "=" * 60)
    print("CROSS-GAME PATTERN MINING")
    print("=" * 60)
    
    patterns = miner.mine_all_pairs(top_k=10)
    
    # High-reward patterns
    print("\n" + "=" * 60)
    print("HIGH-REWARD CROSS-GAME INSIGHTS")
    print("=" * 60)
    
    insights = miner.find_high_reward_patterns(min_reward=0.3, min_sim=0.5)
    for ins in insights[:10]:
        print(f"  {ins['game_a']}↔{ins['game_b']} sim={ins['similarity']:.4f} "
              f"r_a={ins['reward_a']:.2f} r_b={ins['reward_b']:.2f} "
              f"a_a={ins['action_a']} a_b={ins['action_b']}")
    
    # Save
    output = {
        "patterns": patterns,
        "insights": insights,
        "games_loaded": loaded,
        "total_vectors": sum(len(v) for v, _ in miner.games.values()),
    }
    
    out = os.path.expanduser("~/repos/zeroclaw-arena/cross-game-patterns.json")
    with open(out, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
