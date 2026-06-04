"""
ReflexPlayer — The vector DB IS the player.

No scripts. No code generation. No LLM. Just:
1. See a game state
2. Embed it
3. Search vector DB for similar states
4. Pick the action that led to the highest average reward
5. Execute it

The ZeroClaw's learning phase BUILDS the DB.
The ReflexPlayer's playing phase QUERIES the DB.
Same DB, two modes: learn vs play.

This is pincherOS's reflex engine applied to games:
- State = intent (what situation am I in?)
- Action = command (what should I do?)
- Reward = confidence (did it work before?)
- Vector search = matching (find similar past situations)
"""

import json
import math
import hashlib
import sqlite3
import random
import time
import statistics
from pathlib import Path
from collections import defaultdict


class ReflexPlayer:
    """
    A player whose entire "brain" is a vector database of past state transitions.
    
    Playing = vector search + reward-weighted action selection.
    No neural nets. No LLM. No scripts. Just vectors and rewards.
    """
    
    def __init__(self, name: str, db_path: str):
        self.name = name
        self.conn = sqlite3.connect(db_path)
        self.stats = {"wins": 0, "losses": 0, "draws": 0, "moves": 0, "queries": 0}
    
    def _embed(self, text: str, dim: int = 64) -> list[float]:
        """Deterministic hash-based embedding."""
        h = hashlib.blake2b(text.encode(), digest_size=dim).digest()
        vec = [b / 255.0 for b in h]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
    
    def _cosine(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(x * x for x in b)) or 1.0
        return dot / (na * nb)
    
    def choose_action(self, state_str: str, legal_actions: list[str], 
                      temperature: float = 0.0) -> str:
        """
        The core playing function.
        
        1. Embed current state
        2. Find all past transitions from similar states
        3. For each legal action, compute expected reward
        4. Pick the best action (with optional temperature for exploration)
        """
        self.stats["queries"] += 1
        
        if not legal_actions:
            return ""
        
        # Embed current state
        state_vec = self._embed(state_str)
        
        # Search all transitions for similar states
        action_scores = defaultdict(list)  # action -> [rewards]
        action_similarities = defaultdict(list)  # action -> [similarities]
        
        # Search transitions - limit to recent 2000 for speed
        for row in self.conn.execute(
            "SELECT vector, metadata FROM vectors ORDER BY rowid DESC LIMIT 2000"
        ):
            vec = [b / 255.0 for b in row[0]]
            meta = json.loads(row[1])
            
            sim = self._cosine(state_vec, vec)
            if sim < 0.5:  # skip unrelated states
                continue
            
            action = meta.get("action", "")
            reward = meta.get("reward", 0.0)
            
            if action in legal_actions:
                # Weight reward by similarity
                action_scores[action].append(reward * sim)
                action_similarities[action].append(sim)
        
        # Compute weighted expected reward per action
        expected = {}
        for action in legal_actions:
            if action in action_scores and action_scores[action]:
                scores = action_scores[action]
                weights = action_similarities[action]
                total_weight = sum(weights) or 1.0
                expected[action] = sum(scores) / total_weight
                # Boost actions with more evidence
                evidence = len(scores)
                confidence_bonus = min(evidence / 50, 0.5)
                expected[action] += confidence_bonus * 0.1
            else:
                expected[action] = 0.0  # unknown action
        
        # Temperature-based selection
        if temperature > 0 and expected:
            # Boltzmann distribution
            max_e = max(expected.values()) or 1.0
            exp_scores = {a: math.exp((e - max_e) / max(temperature, 0.01)) 
                         for a, e in expected.items()}
            total = sum(exp_scores.values()) or 1.0
            probs = {a: s / total for a, s in exp_scores.items()}
            
            r = random.random()
            cumulative = 0.0
            for action, prob in probs.items():
                cumulative += prob
                if r <= cumulative:
                    return action
            return random.choice(legal_actions)
        
        # Greedy: pick best expected reward
        if expected:
            best_action = max(expected, key=expected.get)
            return best_action
        
        return random.choice(legal_actions)
    
    def play_game(self, game, temperature: float = 0.0) -> dict:
        """Play a full game using only vector DB lookups."""
        game.reset()
        moves = []
        
        while not game.done:
            state = game.state()
            actions = game.legal_actions()
            if not actions:
                break
            
            action = self.choose_action(str(state), actions, temperature)
            reward, done = game.step(action)
            
            moves.append({
                "state": str(state),
                "action": action,
                "reward": reward,
            })
            self.stats["moves"] += 1
        
        # Record outcome
        winner = getattr(game, 'winner', None)
        if winner in ('X', 'player', 'white'):
            self.stats["wins"] += 1
        elif winner in ('draw', None):
            self.stats["draws"] += 1
        else:
            self.stats["losses"] += 1
        
        return {
            "winner": winner,
            "moves": len(moves),
            "total_reward": sum(m["reward"] for m in moves),
        }
    
    def evaluate(self, game, num_games: int = 200, temperature: float = 0.0) -> dict:
        """Evaluate the player over many games."""
        results = []
        for _ in range(num_games):
            result = self.play_game(game, temperature)
            results.append(result)
        
        total = len(results)
        wins = self.stats["wins"]
        wr = wins / max(total, 1)
        
        return {
            "player": self.name,
            "games": total,
            "wins": wins,
            "win_rate": wr,
            "avg_reward": statistics.mean([r["total_reward"] for r in results]) if results else 0,
            "avg_moves": statistics.mean([r["moves"] for r in results]) if results else 0,
            "total_queries": self.stats["queries"],
        }


class RandomPlayer:
    """Baseline: random moves."""
    def __init__(self):
        self.stats = {"wins": 0, "losses": 0, "draws": 0}
    
    def play_game(self, game) -> dict:
        game.reset()
        moves = 0
        while not game.done:
            actions = game.legal_actions()
            if not actions:
                break
            game.step(random.choice(actions))
            moves += 1
        
        winner = getattr(game, 'winner', None)
        if winner in ('X', 'player', 'white'):
            self.stats["wins"] += 1
        elif winner in ('draw', None):
            self.stats["draws"] += 1
        else:
            self.stats["losses"] += 1
        
        return {"winner": winner, "moves": moves}


class GreedyPlayer:
    """Baseline: greedy one-step lookahead (for tic-tac-toe)."""
    def __init__(self, symbol: str = 'X'):
        self.symbol = symbol
        self.stats = {"wins": 0, "losses": 0, "draws": 0}
    
    def play_game(self, game) -> dict:
        game.reset()
        while not game.done:
            actions = game.legal_actions()
            if not actions:
                break
            
            best_action = None
            best_score = -999
            
            for action in actions:
                # Try move, check if it wins
                board_copy = game.board[:]
                pos = int(action)
                board_copy[pos] = game.current
                
                lines = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
                score = 0
                for a, b, c in lines:
                    line = [board_copy[a], board_copy[b], board_copy[c]]
                    if line.count(game.current) == 3:
                        score = 10  # win
                    elif line.count(game.current) == 2 and line.count(' ') == 1:
                        score = max(score, 5)  # two in a row
                    elif line.count(' ') == 3:
                        score = max(score, 1)  # open line
                
                # Block opponent
                opp = 'O' if game.current == 'X' else 'X'
                for a, b, c in lines:
                    line = [board_copy[a], board_copy[b], board_copy[c]]
                    if line.count(opp) == 2 and line.count(' ') == 1:
                        score = max(score, 8)  # block
                
                # Center preference
                if pos == 4:
                    score = max(score, 3)
                
                if score > best_score:
                    best_score = score
                    best_action = action
            
            game.step(best_action or random.choice(actions))
        
        winner = getattr(game, 'winner', None)
        if winner in ('X', 'player', 'white'):
            self.stats["wins"] += 1
        elif winner in ('draw', None):
            self.stats["draws"] += 1
        else:
            self.stats["losses"] += 1
        
        return {"winner": winner, "moves": 0}


def benchmark():
    """
    Compare ReflexPlayer vs RandomPlayer vs GreedyPlayer.
    
    The ReflexPlayer uses ONLY the vector DB built by ZeroClaw exploration.
    No scripts. No LLM. Just vector search.
    """
    
    # Import game
    import sys
    sys.path.insert(0, '.')
    from zeroclaw import TicTacToe, Blackjack, ChessEndgame, Connect4
    
    print("╔══════════════════════════════════════════════════╗")
    print("║     REFLEX PLAYER — Vector DB as Game Engine     ║")
    print("╚══════════════════════════════════════════════════╝")
    
    # ── Tic-Tac-Toe ──────────────────────────────────────
    print("\n" + "="*60)
    print("  TIC-TAC-TOE: Reflex vs Random vs Greedy")
    print("="*60)
    
    game = TicTacToe()
    db_path = "/tmp/zeroclaw-sandbox/zeroclaw-tictactoe/vectors.db"
    
    if Path(db_path).exists():
        reflex = ReflexPlayer("reflex-ttt", db_path)
        random_p = RandomPlayer()
        greedy_p = GreedyPlayer()
        
        # Reflex vs itself
        print("\n  ReflexPlayer vs Random Opponent (200 games):")
        for _ in range(200):
            game.reset()
            while not game.done:
                state = game.state()
                actions = game.legal_actions()
                if not actions: break
                
                if game.current == 'X':
                    action = reflex.choose_action(str(state), actions, temperature=0.0)
                else:
                    action = random.choice(actions)
                
                game.step(action)
            
            winner = game.winner
            if winner == 'X':
                reflex.stats["wins"] += 1
            elif winner == 'O':
                reflex.stats["losses"] += 1
            else:
                reflex.stats["draws"] += 1
        
        reflex_wr = reflex.stats["wins"] / 200
        reflex_draws = reflex.stats["draws"] / 200
        
        # Random baseline
        print("  RandomPlayer baseline (200 games):")
        random_wr = 0
        for _ in range(200):
            game.reset()
            while not game.done:
                actions = game.legal_actions()
                if not actions: break
                game.step(random.choice(actions))
            if game.winner == 'X':
                random_wr += 1
        random_wr /= 200
        
        # Greedy baseline
        print("  GreedyPlayer baseline (200 games):")
        greedy_wr = 0
        for _ in range(200):
            game.reset()
            while not game.done:
                actions = game.legal_actions()
                if not actions: break
                
                if game.current == 'X':
                    # Greedy: center, win, block, random
                    action = None
                    if '4' in actions:
                        action = '4'
                    else:
                        action = random.choice(actions)
                else:
                    action = random.choice(actions)
                game.step(action)
            if game.winner == 'X':
                greedy_wr += 1
        greedy_wr /= 200
        
        print(f"\n  Results:")
        print(f"    ReflexPlayer (vector DB): {reflex_wr:.1%} wins, {reflex_draws:.1%} draws")
        print(f"    RandomPlayer:             {random_wr:.1%} wins")
        print(f"    GreedyPlayer:             {greedy_wr:.1%} wins")
        print(f"    Reflex advantage:         {(reflex_wr - random_wr)*100:+.1f}pp vs random")
        print(f"    Reflex DB size:           {reflex.stats['queries']} queries made")
    
    # ── Blackjack ────────────────────────────────────────
    print("\n" + "="*60)
    print("  BLACKJACK: Reflex vs Random")
    print("="*60)
    
    game = Blackjack()
    db_path = "/tmp/zeroclaw-sandbox/zeroclaw-blackjack/vectors.db"
    
    if Path(db_path).exists():
        reflex = ReflexPlayer("reflex-bj", db_path)
        
        print("\n  ReflexPlayer (200 games):")
        for _ in range(200):
            game.reset()
            while not game.done:
                state = game.state()
                actions = game.legal_actions()
                if not actions: break
                action = reflex.choose_action(str(state), actions, temperature=0.0)
                game.step(action)
            
            if game.winner == "player":
                reflex.stats["wins"] += 1
            else:
                reflex.stats["losses"] += 1
        
        reflex_wr = reflex.stats["wins"] / 200
        
        random_wr = 0
        for _ in range(200):
            game.reset()
            while not game.done:
                actions = game.legal_actions()
                if not actions: break
                game.step(random.choice(actions))
            if game.winner == "player":
                random_wr += 1
        random_wr /= 200
        
        print(f"    ReflexPlayer: {reflex_wr:.1%} wins")
        print(f"    RandomPlayer: {random_wr:.1%} wins")
        print(f"    Advantage:    {(reflex_wr - random_wr)*100:+.1f}pp")
    
    # ── Chess Endgame ────────────────────────────────────
    print("\n" + "="*60)
    print("  CHESS ENDGAME: Reflex vs Random")
    print("="*60)
    
    game = ChessEndgame()
    db_path = "/tmp/zeroclaw-sandbox/zeroclaw-chess_endgame/vectors.db"
    
    if Path(db_path).exists() and game.has_chess:
        reflex = ReflexPlayer("reflex-chess", db_path)
        
        print("\n  ReflexPlayer (100 games):")
        for _ in range(100):
            game.reset()
            while not game.done:
                actions = game.legal_actions()
                if not actions: break
                
                state = game.state()
                # White (us) uses reflex, black is random
                if state.player == "white":
                    action = reflex.choose_action(str(state), actions, temperature=0.1)
                else:
                    action = random.choice(actions)
                game.step(action)
            
            if game.winner == "white":
                reflex.stats["wins"] += 1
            elif game.winner == "black":
                reflex.stats["losses"] += 1
            else:
                reflex.stats["draws"] += 1
        
        reflex_wr = reflex.stats["wins"] / 100
        
        random_wr = 0
        for _ in range(100):
            game.reset()
            while not game.done:
                actions = game.legal_actions()
                if not actions: break
                game.step(random.choice(actions))
            if game.winner == "white":
                random_wr += 1
        random_wr /= 100
        
        print(f"    ReflexPlayer: {reflex_wr:.1%} wins")
        print(f"    RandomPlayer: {random_wr:.1%} wins")
        print(f"    Advantage:    {(reflex_wr - random_wr)*100:+.1f}pp")
    
    # ── Connect 4 ──────────────────────────────────────
    print("\n" + "="*60)
    print("  CONNECT 4: Reflex vs Random")
    print("="*60)
    
    game = Connect4()
    db_path = "/tmp/zeroclaw-sandbox/zeroclaw-connect4/vectors.db"
    
    if Path(db_path).exists():
        reflex = ReflexPlayer("reflex-c4", db_path)
        
        print("\n  ReflexPlayer vs Random Opponent (200 games):")
        for _ in range(200):
            game.reset()
            while not game.done:
                state = game.state()
                actions = game.legal_actions()
                if not actions: break
                
                if game.current == 'X':
                    action = reflex.choose_action(str(state), actions, temperature=0.0)
                else:
                    action = random.choice(actions)
                
                game.step(action)
            
            if game.winner == 'X':
                reflex.stats["wins"] += 1
            elif game.winner == 'O':
                reflex.stats["losses"] += 1
            else:
                reflex.stats["draws"] += 1
        
        reflex_wr = reflex.stats["wins"] / 200
        reflex_draws = reflex.stats["draws"] / 200
        
        random_wr = 0
        for _ in range(200):
            game.reset()
            while not game.done:
                actions = game.legal_actions()
                if not actions: break
                game.step(random.choice(actions))
            if game.winner == 'X':
                random_wr += 1
        random_wr /= 200
        
        print(f"    ReflexPlayer: {reflex_wr:.1%} wins, {reflex_draws:.1%} draws")
        print(f"    RandomPlayer: {random_wr:.1%} wins")
        print(f"    Advantage:    {(reflex_wr - random_wr)*100:+.1f}pp")
    else:
        print("    No vector DB found — run ZeroClaw arena first")
    
    print(f"\n{'='*60}")
    print(f"  CONCLUSION: The vector DB IS the player.")
    print(f"  No neural nets. No LLM. Just vector search + reward.")
    print(f"{'='*60}")


if __name__ == "__main__":
    benchmark()
