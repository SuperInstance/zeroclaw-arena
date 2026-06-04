"""
ZeroClaw — A sandboxed agent that learns text-based games from scratch.

Each ZeroClaw:
1. Gets a game environment (state, actions, rewards)
2. Explores by playing randomly
3. Records every (state, action, reward, next_state) in a vector DB
4. Discovers patterns algorithmically (no neural nets — pure math)
5. Writes automation scripts based on patterns
6. Tests scripts, keeps winners, discards losers
7. Repeats — each cycle produces better scripts

No LLM inference during learning. Pure algorithmic discovery.
The LLM is only used to READ patterns and WRITE scripts (the metacognitive layer).
Learning is all vector math and statistical pattern matching.
"""

import json
import time
import hashlib
import os
import sqlite3
import random
import math
import statistics
import numpy as np

# ─── Tile Exploration Flag ─────────────────────────────
USE_TILE_EXPLORATION = True  # When True, use tile-field Monte Carlo instead of random exploration
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
from pathlib import Path
from collections import defaultdict

# GPU Vector Engine — optional, uses CUDA if available
try:
    from gpu_vector_engine import GPUVectorEngine
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# torch-vector-search — optional, replaces SQLite VectorDB with GPU-accelerated backend
try:
    from vector_store import VectorStore
    TORCH_VECTOR_AVAILABLE = True
except ImportError:
    TORCH_VECTOR_AVAILABLE = False


# ─── Vector DB (lightweight, no dependencies) ─────────────

class VectorDB:
    """Simple vector database using SQLite + cosine similarity."""
    
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS vectors (
                id TEXT PRIMARY KEY,
                vector BLOB,
                metadata TEXT
            )
        """)
        self.conn.commit()
    
    def _hash_to_vector(self, text: str, dim: int = 64) -> list[float]:
        """Deterministic hash-based embedding. Same text = same vector."""
        h = hashlib.blake2b(text.encode(), digest_size=dim).digest()
        vec = [b / 255.0 for b in h]
        # Normalize
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
    
    def _cosine_sim(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(x * x for x in b)) or 1.0
        return dot / (na * nb)
    
    def insert(self, id: str, text: str, metadata: dict):
        vec = self._hash_to_vector(text)
        vec_bytes = bytes(int(v * 255) for v in vec)
        self.conn.execute(
            "INSERT OR REPLACE INTO vectors (id, vector, metadata) VALUES (?, ?, ?)",
            (id, vec_bytes, json.dumps(metadata))
        )
        self.conn.commit()
    
    def search(self, query_text: str, top_k: int = 10) -> list[tuple[str, float, dict]]:
        query_vec = self._hash_to_vector(query_text)
        results = []
        for row in self.conn.execute("SELECT id, vector, metadata FROM vectors"):
            vec = [b / 255.0 for b in row[1]]
            sim = self._cosine_sim(query_vec, vec)
            results.append((row[0], sim, json.loads(row[2])))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]
    
    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
    
    def close(self):
        self.conn.close()


# ─── Game Environments ────────────────────────────────────

class GameState:
    """Serialized game state for vector embedding."""
    def __init__(self, state_str: str, turn: int, player: str):
        self.state_str = state_str
        self.turn = turn
        self.player = player
    
    def __str__(self):
        return f"[turn={self.turn}|{self.player}]{self.state_str}"
    
    def hash(self):
        return hashlib.blake2b(str(self).encode(), digest_size=8).hexdigest()


@dataclass
class Transition:
    """One state transition: (state, action) → (reward, next_state)"""
    state_hash: str
    state_str: str
    action: str
    reward: float
    next_state_hash: str
    next_state_str: str
    game_over: bool
    winner: Optional[str] = None


class TicTacToe:
    """Simple tic-tac-toe for initial testing."""
    
    def __init__(self):
        self.board = [' '] * 9
        self.current = 'X'
        self.turn = 0
        self.done = False
        self.winner = None
    
    def state(self) -> GameState:
        return GameState(''.join(self.board), self.turn, self.current)
    
    def legal_actions(self) -> list[str]:
        if self.done:
            return []
        return [str(i) for i in range(9) if self.board[i] == ' ']
    
    def step(self, action: str) -> tuple[float, bool]:
        pos = int(action)
        if self.board[pos] != ' ':
            return -1.0, True  # illegal move
        
        self.board[pos] = self.current
        self.turn += 1
        
        # Check win
        lines = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
        for a, b, c in lines:
            if self.board[a] == self.board[b] == self.board[c] != ' ':
                self.done = True
                self.winner = self.current
                reward = 1.0 if self.current == 'X' else -1.0
                return reward, True
        
        if self.turn >= 9:
            self.done = True
            return 0.0, True  # draw
        
        self.current = 'O' if self.current == 'X' else 'X'
        return 0.0, False
    
    def reset(self):
        self.board = [' '] * 9
        self.current = 'X'
        self.turn = 0
        self.done = False
        self.winner = None


class Blackjack:
    """Simple blackjack for card game testing."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.deck = [min(v, 10) for v in range(1, 14)] * 4
        random.shuffle(self.deck)
        self.player_hand = [self._draw(), self._draw()]
        self.dealer_hand = [self._draw(), self._draw()]
        self.done = False
        self.winner = None
        self.turn = 0
    
    def _draw(self) -> int:
        return self.deck.pop()
    
    def _hand_value(self, hand: list[int]) -> int:
        total = sum(hand)
        # Simplified: no ace logic
        return total
    
    def state(self) -> GameState:
        player_val = self._hand_value(self.player_hand)
        dealer_show = self.dealer_hand[0]
        return GameState(
            f"P={player_val}_D={dealer_show}_cards={len(self.player_hand)}",
            self.turn, "player"
        )
    
    def legal_actions(self) -> list[str]:
        if self.done:
            return []
        return ["hit", "stand"]
    
    def step(self, action: str) -> tuple[float, bool]:
        self.turn += 1
        
        if action == "hit":
            self.player_hand.append(self._draw())
            player_val = self._hand_value(self.player_hand)
            if player_val > 21:
                self.done = True
                self.winner = "dealer"
                return -1.0, True
            return 0.0, False
        
        # stand — dealer plays
        while self._hand_value(self.dealer_hand) < 17:
            self.dealer_hand.append(self._draw())
        
        player_val = self._hand_value(self.player_hand)
        dealer_val = self._hand_value(self.dealer_hand)
        self.done = True
        
        if dealer_val > 21:
            self.winner = "player"
            return 1.0, True
        elif player_val > dealer_val:
            self.winner = "player"
            return 1.0, True
        elif player_val < dealer_val:
            self.winner = "dealer"
            return -1.0, True
        else:
            return 0.0, True  # push


class ChessEndgame:
    """Simplified chess: king + queen vs king endgames."""
    
    def __init__(self):
        try:
            import chess
            self.has_chess = True
        except ImportError:
            self.has_chess = False
        self.reset()
    
    def reset(self):
        if self.has_chess:
            import chess
            # KQ vs K endgame
            self.board = chess.Board("8/8/8/8/8/5k2/8/4K2Q w - - 0 1")
        else:
            self.board = None
        self.done = False
        self.winner = None
        self.turn = 0
        self.moves = []
    
    def state(self) -> GameState:
        if self.has_chess and self.board:
            return GameState(self.board.fen(), self.turn, "white")
        return GameState("no_chess", self.turn, "none")
    
    def legal_actions(self) -> list[str]:
        if self.done or not self.has_chess or not self.board:
            return []
        return [m.uci() for m in self.board.legal_moves]
    
    def step(self, action: str) -> tuple[float, bool]:
        if not self.has_chess or not self.board:
            return 0.0, True
        
        import chess
        try:
            move = chess.Move.from_uci(action)
            if move not in self.board.legal_moves:
                return -1.0, True  # illegal
            self.board.push(move)
        except:
            return -1.0, True
        
        self.turn += 1
        self.moves.append(action)
        
        if self.board.is_checkmate():
            self.done = True
            self.winner = "white" if self.board.result() == "1-0" else "black"
            reward = 1.0 if self.winner == "white" else -1.0
            return reward, True
        
        if self.board.is_stalemate() or self.board.is_insufficient_material() or self.turn > 100:
            self.done = True
            self.winner = "draw"
            return 0.0, True
        
        return 0.0, False
    
    def random_move(self) -> Optional[str]:
        actions = self.legal_actions()
        return random.choice(actions) if actions else None


class Connect4:
    """Connect 4 game for ZeroClaw learning."""
    
    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.reset()
    
    def reset(self):
        self.board = [[' ']*self.cols for _ in range(self.rows)]
        self.current = 'X'  # X=Red, O=Yellow
        self.turn = 0
        self.done = False
        self.winner = None
    
    def state(self) -> GameState:
        board_str = ''.join(''.join(row) for row in self.board)
        return GameState(board_str, self.turn, self.current)
    
    def legal_actions(self) -> list[str]:
        if self.done:
            return []
        return [str(c) for c in range(self.cols) if self.board[0][c] == ' ']
    
    def step(self, action: str) -> tuple[float, bool]:
        col = int(action)
        if col < 0 or col >= self.cols or self.board[0][col] != ' ':
            return -1.0, True  # illegal
        
        # Drop piece
        row = self.rows - 1
        while row >= 0 and self.board[row][col] != ' ':
            row -= 1
        if row < 0:
            return -1.0, True
        
        self.board[row][col] = self.current
        self.turn += 1
        
        # Check win (horizontal, vertical, both diagonals)
        directions = [(0,1),(1,0),(1,1),(1,-1)]
        for dr, dc in directions:
            count = 1
            for sign in [1, -1]:
                r, c = row + sign*dr, col + sign*dc
                while 0 <= r < self.rows and 0 <= c < self.cols and self.board[r][c] == self.current:
                    count += 1
                    r += sign*dr
                    c += sign*dc
            if count >= 4:
                self.done = True
                self.winner = self.current
                reward = 1.0 if self.current == 'X' else -1.0
                return reward, True
        
        # Check draw
        if self.turn >= self.rows * self.cols:
            self.done = True
            return 0.0, True
        
        self.current = 'O' if self.current == 'X' else 'X'
        return 0.0, False


# ─── Go 9x9 ───────────────────────────────────────────────

class Go9x9:
    """Simplified 9x9 Go for ZeroClaw learning.
    
    Rules:
    - Black plays first
    - Simple ko rule (can't repeat previous board state)
    - Game ends when both players pass
    - Score = territory + captures (Chinese scoring)
    - Komi: 5.5 for White
    """
    
    def __init__(self, size=9):
        self.size = size
        self.reset()
    
    def reset(self):
        self.board = [['.' for _ in range(self.size)] for _ in range(self.size)]
        self.current = 'B'  # B=Black, W=White
        self.turn = 0
        self.done = False
        self.winner = None
        self.captures = {'B': 0, 'W': 0}
        self.previous_board = None
        self.passes = 0
        self.komi = 5.5
    
    def state(self) -> GameState:
        board_str = ''.join(''.join(row) for row in self.board)
        return GameState(f"{board_str}_C{self.captures['B']}_{self.captures['W']}", self.turn, self.current)
    
    def legal_actions(self) -> list[str]:
        if self.done:
            return []
        actions = ['pass']
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] == '.':
                    if self._is_legal(r, c):
                        actions.append(f"{r},{c}")
        return actions
    
    def _is_legal(self, r, c) -> bool:
        """Check if placing at (r,c) is legal."""
        test_board = [row[:] for row in self.board]
        test_board[r][c] = self.current
        
        opp = 'W' if self.current == 'B' else 'B'
        captured = 0
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < self.size and 0 <= nc < self.size and test_board[nr][nc] == opp:
                group, liberties = self._get_group(test_board, nr, nc)
                if liberties == 0:
                    captured += len(group)
        
        if captured == 0:
            _, liberties = self._get_group(test_board, r, c)
            if liberties == 0:
                return False
        
        board_str = ''.join(''.join(row) for row in test_board)
        if self.previous_board and board_str == self.previous_board:
            return False
        
        return True
    
    def _get_group(self, board, r, c):
        """Get all stones in the group and count liberties."""
        color = board[r][c]
        if color == '.':
            return [], 0
        visited = set()
        group = []
        liberties = set()
        stack = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            if (cr, cc) in visited:
                continue
            visited.add((cr, cc))
            group.append((cr, cc))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = cr+dr, cc+dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if board[nr][nc] == '.':
                        liberties.add((nr, nc))
                    elif board[nr][nc] == color and (nr, nc) not in visited:
                        stack.append((nr, nc))
        return group, len(liberties)
    
    def step(self, action: str) -> tuple[float, bool]:
        if action == 'pass':
            self.passes += 1
            if self.passes >= 2:
                self._score_game()
                return self._get_reward(), True
            self.previous_board = ''.join(''.join(row) for row in self.board)
            self.current = 'W' if self.current == 'B' else 'B'
            self.turn += 1
            return 0.0, False
        
        r, c = map(int, action.split(','))
        if not self._is_legal(r, c):
            return -1.0, True
        
        self.previous_board = ''.join(''.join(row) for row in self.board)
        
        self.board[r][c] = self.current
        self.passes = 0
        
        opp = 'W' if self.current == 'B' else 'B'
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < self.size and 0 <= nc < self.size and self.board[nr][nc] == opp:
                group, liberties = self._get_group(self.board, nr, nc)
                if liberties == 0:
                    for gr, gc in group:
                        self.board[gr][gc] = '.'
                    self.captures[self.current] += len(group)
        
        self.current = 'W' if self.current == 'B' else 'B'
        self.turn += 1
        
        if self.turn >= self.size * self.size * 2:
            self._score_game()
            return self._get_reward(), True
        
        return 0.0, False
    
    def _score_game(self):
        """Chinese scoring: territory + stones on board."""
        self.done = True
        scores = {'B': 0, 'W': 0}
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] != '.':
                    scores[self.board[r][c]] += 1
                else:
                    owner = self._get_territory(r, c)
                    if owner in scores:
                        scores[owner] += 1
        
        scores['W'] += self.komi
        self.winner = 'B' if scores['B'] > scores['W'] else 'W'
    
    def _get_territory(self, r, c):
        """Flood fill to find territory owner."""
        visited = set()
        colors = set()
        stack = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            if (cr, cc) in visited:
                continue
            visited.add((cr, cc))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = cr+dr, cc+dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if self.board[nr][nc] == '.':
                        stack.append((nr, nc))
                    else:
                        colors.add(self.board[nr][nc])
        if len(colors) == 1:
            return colors.pop()
        return None
    
    def _get_reward(self) -> float:
        if self.winner == 'B':
            return 1.0
        elif self.winner == 'W':
            return -1.0
        return 0.0

# ─── StateTile for Tile-Field Exploration ──────────────────

class StateTile:
    """A tile representing one game state with scored reflexes (legal actions)."""

    def __init__(self, state_hash: str, state_str: str, actions: list[str]):
        self.state_hash = state_hash
        self.state_str = state_str
        self.reflexes: dict[str, dict] = {
            a: {"score": 0.5, "chosen": 0, "won": 0} for a in actions
        }
        self.entropy = 1.0  # high = uncertain

    def best_action(self, legal_actions: list[str], n_simulations: int = 20,
                    game=None) -> str:
        """Pick the best action using Monte Carlo simulation + learned scores."""
        if not legal_actions:
            return ''
        if len(legal_actions) == 1:
            return legal_actions[0]

        # Ensure all legal actions have a reflex entry
        for a in legal_actions:
            if a not in self.reflexes:
                self.reflexes[a] = {"score": 0.5, "chosen": 0, "won": 0}

        action_values = {}
        for action in legal_actions:
            # Monte Carlo simulation
            sim_wins = 0
            sims_per_action = max(1, n_simulations // len(legal_actions))

            if game is not None:
                for _ in range(sims_per_action):
                    winner = self._simulate_playout(game, action)
                    if winner == 'X':
                        sim_wins += 1

            sim_score = sim_wins / max(sims_per_action, 1)
            learned_score = self.reflexes[action]["score"]
            n_chosen = self.reflexes[action]["chosen"]
            confidence = min(n_chosen / 20.0, 0.8)

            action_values[action] = (
                confidence * learned_score + (1 - confidence) * sim_score
            )

        # Softmax selection (temperature=0.3)
        actions_list = list(action_values.keys())
        values = np.array([action_values[a] for a in actions_list])
        temperature = 0.3
        exp_vals = np.exp(values / temperature)
        probs = exp_vals / exp_vals.sum()

        return np.random.choice(actions_list, p=probs)

    def record(self, action: str, won: bool):
        if action in self.reflexes:
            self.reflexes[action]["chosen"] += 1
            if won:
                self.reflexes[action]["won"] += 1

    def evolve(self):
        """Update scores based on accumulated win rates."""
        for action, data in self.reflexes.items():
            if data["chosen"] > 0:
                wr = data["won"] / data["chosen"]
                data["score"] += 0.05 * (wr - data["score"])
                data["score"] = max(0.05, min(0.95, data["score"]))

    def _simulate_playout(self, real_game, first_action) -> Optional[str]:
        """Run a random playout from the current state + first_action."""
        game_copy = type(real_game)()
        # Copy board state
        if hasattr(real_game, 'board'):
            game_copy.board = (
                [row[:] for row in real_game.board]
                if isinstance(real_game.board[0], list)
                else real_game.board[:]
            )
        for attr in ('current', 'done', 'winner', 'turn'):
            if hasattr(real_game, attr):
                setattr(game_copy, attr, getattr(real_game, attr))

        game_copy.step(first_action)

        # Play randomly to completion
        while not game_copy.done:
            actions = game_copy.legal_actions()
            if not actions:
                break
            game_copy.step(random.choice(actions))

        return getattr(game_copy, 'winner', None)


# ─── ZeroClaw Agent ───────────────────────────────────────

class ZeroClaw:
    """
    A sandboxed agent that learns a game algorithmically.
    
    No neural networks. No training loops. Pure:
    - State transition recording
    - Vector DB pattern matching
    - Statistical analysis
    - Script generation from patterns
    """
    
    def __init__(self, name: str, game_name: str, sandbox_dir: str = "/tmp/zeroclaw-sandbox"):
        self.name = name
        self.game_name = game_name
        self.sandbox_dir = Path(sandbox_dir) / name
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        
        # Use torch-vector-search backend when available, fall back to SQLite
        if TORCH_VECTOR_AVAILABLE:
            self.vdb = VectorStore(str(self.sandbox_dir / "vectors.db"))
        else:
            self.vdb = VectorDB(str(self.sandbox_dir / "vectors.db"))
        self.transitions: list[Transition] = []
        self.scripts: list[dict] = []  # discovered automation scripts
        self.generation = 0
        self.stats = {
            "games_played": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "scripts_generated": 0,
            "scripts_passing": 0,
            "best_win_rate": 0.0,
        }
        
        # GPU Vector Engine for batch embedding & pattern mining
        if GPU_AVAILABLE:
            self.gpu_engine = GPUVectorEngine(dim=64)
        else:
            self.gpu_engine = None
        
        # Tile-field state for tile exploration
        self.tile_field: dict[str, StateTile] = {}  # state_hash -> StateTile
        self.tile_evolve_every = 25  # evolve tile scores every N games

        self._load_state()
        self._load_gpu_state()
    
    # ── Phase 1: EXPLORE ──────────────────────────────────
    
    def play_game(self, game, policy: str = "random") -> list[Transition]:
        """Play one game and record all transitions."""
        transitions = []
        game.reset()
        
        while not game.done:
            state = game.state()
            actions = game.legal_actions()
            
            if not actions:
                break
            
            # Choose action
            if policy == "random":
                action = random.choice(actions)
            elif policy == "script" and self.scripts:
                action = self._script_action(state, actions)
            elif policy == "best_pattern":
                action = self._best_pattern_action(state, actions)
            else:
                action = random.choice(actions)
            
            reward, done = game.step(action)
            next_state = game.state()
            
            t = Transition(
                state_hash=state.hash(),
                state_str=str(state),
                action=action,
                reward=reward,
                next_state_hash=next_state.hash(),
                next_state_str=str(next_state),
                game_over=done,
                winner=getattr(game, 'winner', None)
            )
            transitions.append(t)
            
            # Store in vector DB
            self.vdb.insert(
                f"{state.hash()}:{action}",
                f"{state}|{action}",
                {"state": state.state_str, "action": action, "reward": reward,
                 "turn": state.turn, "game_over": done}
            )
        
        # Record outcome
        self.stats["games_played"] += 1
        if hasattr(game, 'winner'):
            if game.winner in ('X', 'player', 'white'):
                self.stats["wins"] += 1
            elif game.winner in ('draw', None, 'draw'):
                self.stats["draws"] += 1
            else:
                self.stats["losses"] += 1
        
        self.transitions.extend(transitions)
        return transitions
    
    def explore(self, game, num_games: int = 100):
        """Play many games to explore the state space."""
        if USE_TILE_EXPLORATION:
            self.explore_tile_field(game, num_games)
            return

        print(f"  {self.name}: Exploring {num_games} games of {self.game_name}...")
        for i in range(num_games):
            self.play_game(game, policy="random")
            if (i + 1) % 25 == 0:
                win_rate = self.stats["wins"] / max(self.stats["games_played"], 1)
                print(f"    {i+1}/{num_games} games | win_rate={win_rate:.1%} | transitions={len(self.transitions)}")
        
        # GPU batch embedding: if we collected enough transitions, batch-embed them
        if self.gpu_engine and len(self.transitions) > 100:
            new_states = [f"{t.state_str}|{t.action}" for t in self.transitions]
            new_metadata = [{"state": t.state_str, "action": t.action, "reward": t.reward,
                             "turn": t.state_str.count("|") - 1 if "|" in t.state_str else 0,
                             "game_over": t.game_over} for t in self.transitions]
            batch_vecs = self.gpu_engine.hash_embed_batch(new_states)
            self.gpu_engine.add_batch(batch_vecs, new_metadata)
            print(f"    GPU batch-embedded {len(new_states)} states ({len(self.gpu_engine)} total in GPU index)")

    def explore_tile_field(self, game, num_games: int = 100, n_simulations: int = 20):
        """Explore using tile-field Monte Carlo instead of random."""
        print(f"  {self.name}: Tile-field exploring {num_games} games of {self.game_name} (sims={n_simulations})...")

        for i in range(num_games):
            game.reset()
            history = []  # (state_hash, action) pairs for this game

            while not game.done:
                state = game.state()
                actions = game.legal_actions()
                if not actions:
                    break

                state_hash = state.hash()

                # Get or create StateTile
                if state_hash not in self.tile_field:
                    self.tile_field[state_hash] = StateTile(
                        state_hash, str(state), actions
                    )
                tile = self.tile_field[state_hash]

                # Use tile to choose action
                if game.current == 'X' or getattr(game, 'current', 'player') == 'player':
                    action = tile.best_action(actions, n_simulations, game)
                else:
                    action = random.choice(actions)

                # Record transition
                reward, done = game.step(action)
                next_state = game.state()
                t = Transition(
                    state_hash=state_hash,
                    state_str=str(state),
                    action=action,
                    reward=reward,
                    next_state_hash=next_state.hash(),
                    next_state_str=str(next_state),
                    game_over=done,
                    winner=getattr(game, 'winner', None)
                )
                self.transitions.append(t)

                # Store in vector DB
                self.vdb.insert(
                    f"{state_hash}:{action}",
                    f"{state}|{action}",
                    {"state": state.state_str, "action": action, "reward": reward,
                     "turn": state.turn, "game_over": done}
                )

                history.append((state_hash, action))

            # Record outcome
            self.stats["games_played"] += 1
            won = False
            if hasattr(game, 'winner'):
                if game.winner in ('X', 'player', 'white'):
                    self.stats["wins"] += 1
                    won = True
                elif game.winner in ('draw', None):
                    self.stats["draws"] += 1
                else:
                    self.stats["losses"] += 1

            # Update tile records with outcome
            for state_hash, action in history:
                if state_hash in self.tile_field:
                    self.tile_field[state_hash].record(action, won)

            # Evolve tile scores periodically
            if (i + 1) % self.tile_evolve_every == 0:
                for tile in self.tile_field.values():
                    tile.evolve()

            if (i + 1) % 25 == 0:
                win_rate = self.stats["wins"] / max(self.stats["games_played"], 1)
                print(f"    {i+1}/{num_games} games | win_rate={win_rate:.1%} | "
                      f"tiles={len(self.tile_field)} | transitions={len(self.transitions)}")

        print(f"  Tile field: {len(self.tile_field)} tiles learned over {num_games} games")

        # GPU batch embedding
        if self.gpu_engine and len(self.transitions) > 100:
            new_states = [f"{t.state_str}|{t.action}" for t in self.transitions[-num_games*10:]]
            new_metadata = [{"state": t.state_str, "action": t.action, "reward": t.reward,
                             "turn": 0, "game_over": t.game_over}
                            for t in self.transitions[-num_games*10:]]
            batch_vecs = self.gpu_engine.hash_embed_batch(new_states)
            self.gpu_engine.add_batch(batch_vecs, new_metadata)
            print(f"    GPU batch-embedded {len(new_states)} states")
    
    # ── Phase 2: OBSERVE ──────────────────────────────────
    
    def analyze_patterns(self) -> list[dict]:
        """Algorithmically discover patterns in the transition data."""
        patterns = []
        
        # Pattern 1: Winning actions per state prefix
        action_rewards = defaultdict(lambda: defaultdict(list))
        for t in self.transitions:
            # Group by first N chars of state (state prefix = position type)
            prefix = t.state_str[:20]
            action_rewards[prefix][t.action].append(t.reward)
        
        for prefix, actions in action_rewards.items():
            for action, rewards in actions.items():
                if len(rewards) >= 3:
                    avg_reward = statistics.mean(rewards)
                    if avg_reward > 0.3:  # winning pattern
                        patterns.append({
                            "type": "winning_action",
                            "state_prefix": prefix,
                            "action": action,
                            "avg_reward": avg_reward,
                            "sample_size": len(rewards),
                            "confidence": min(len(rewards) / 20, 1.0),
                        })
        
        # Pattern 2: State features that predict wins
        feature_wins = defaultdict(lambda: {"wins": 0, "total": 0})
        for t in self.transitions:
            if t.game_over:
                features = self._extract_features(t.state_str)
                for feat in features:
                    feature_wins[feat]["total"] += 1
                    if t.reward > 0:
                        feature_wins[feat]["wins"] += 1
        
        for feat, data in feature_wins.items():
            if data["total"] >= 5:
                win_rate = data["wins"] / data["total"]
                if win_rate > 0.6 or win_rate < 0.2:  # strong signal either way
                    patterns.append({
                        "type": "feature_predictor",
                        "feature": feat,
                        "win_rate": win_rate,
                        "sample_size": data["total"],
                        "confidence": min(data["total"] / 30, 1.0),
                    })
        
        # Pattern 3: Action sequences that lead to wins
        game_sequences = defaultdict(list)
        current_game = []
        for t in self.transitions:
            current_game.append(t)
            if t.game_over:
                game_id = len(game_sequences)
                game_sequences[game_id] = current_game
                current_game = []
        
        for gid, seq in game_sequences.items():
            if seq and seq[-1].reward > 0:  # winning game
                # Extract last 3 actions as "closing pattern"
                closing = [t.action for t in seq[-3:]]
                patterns.append({
                    "type": "closing_pattern",
                    "actions": closing,
                    "win_rate": 1.0,
                    "sample_size": 1,
                    "confidence": 0.3,  # low — need more samples
                })
        
        # Pattern 4: Vector DB similarity — similar states have similar best actions
        if self.vdb.count() > 50:
            for t in self.transitions[-100:]:
                if t.reward > 0:
                    similar = self.vdb.search(t.state_str, top_k=5)
                    for sid, sim, meta in similar:
                        if sim > 0.9 and meta.get("action") != t.action:
                            patterns.append({
                                "type": "similar_state_different_action",
                                "state_a": t.state_str[:30],
                                "state_b": sid[:30],
                                "action_a": t.action,
                                "action_b": meta.get("action"),
                                "similarity": sim,
                                "confidence": sim * 0.5,
                            })
        
        patterns.sort(key=lambda p: p.get("confidence", 0), reverse=True)
        return patterns
    
    # ── Phase 3: SCRIPT ───────────────────────────────────
    
    def generate_scripts(self, patterns: list[dict]) -> list[dict]:
        """Turn patterns into executable automation scripts."""
        scripts = []
        
        for i, pattern in enumerate(patterns[:20]):  # top 20 patterns
            script = {
                "id": f"{self.name}_script_{self.generation}_{i}",
                "pattern": pattern,
                "generation": self.generation,
                "code": self._pattern_to_code(pattern),
                "win_rate": 0.0,
                "games_tested": 0,
                "status": "untested",
            }
            scripts.append(script)
            self.scripts.append(script)
            self.stats["scripts_generated"] += 1
        
        return scripts
    
    def _pattern_to_code(self, pattern: dict) -> str:
        """Convert a discovered pattern into executable decision code."""
        ptype = pattern.get("type")
        
        if ptype == "winning_action":
            return f"""
def choose_action(state_str, legal_actions):
    prefix = state_str[:20]
    if "{pattern['state_prefix']}" in state_str and "{pattern['action']}" in legal_actions:
        return "{pattern['action']}"  # avg_reward={pattern['avg_reward']:.2f}, n={pattern['sample_size']}
    return None  # no opinion
"""
        
        elif ptype == "feature_predictor":
            feat = pattern['feature']
            return f"""
def choose_action(state_str, legal_actions):
    if "{feat}" in state_str:
        # feature '{feat}' has {pattern['win_rate']:.0%} win rate
        if {pattern['win_rate']} > 0.5:
            return random.choice(legal_actions)  # play aggressively
        else:
            return None  # avoid — low win rate
    return None
"""
        
        elif ptype == "closing_pattern":
            actions = pattern['actions']
            return f"""
def choose_action(state_str, legal_actions):
    closing_pattern = {actions}
    if len(legal_actions) > 0:
        for a in closing_pattern:
            if a in legal_actions:
                return a
    return None
"""
        
        return """
def choose_action(state_str, legal_actions):
    return None  # no opinion
"""
    
    # ── Phase 4: TEST ─────────────────────────────────────
    
    def test_scripts(self, game, num_tests: int = 100) -> list[dict]:
        """Test all untested scripts against the game."""
        results = []
        
        for script in self.scripts:
            if script["status"] != "untested":
                continue
            
            wins = 0
            total = 0
            
            # Compile the script code
            try:
                namespace = {"random": random, "math": math, "statistics": statistics, "np": np}
                exec(script["code"], namespace)
                choose_fn = namespace.get("choose_action")
                if not choose_fn:
                    script["status"] = "broken"
                    continue
            except:
                script["status"] = "broken"
                continue
            
            for _ in range(num_tests):
                game.reset()
                while not game.done:
                    state = game.state()
                    actions = game.legal_actions()
                    if not actions:
                        break
                    
                    # Try script first, fall back to random
                    choice = choose_fn(str(state), actions)
                    if choice is None or choice not in actions:
                        choice = random.choice(actions)
                    
                    game.step(choice)
                
                total += 1
                if hasattr(game, 'winner'):
                    if game.winner in ('X', 'player', 'white'):
                        wins += 1
            
            win_rate = wins / max(total, 1)
            script["win_rate"] = win_rate
            script["games_tested"] = total
            script["status"] = "passing" if win_rate > 0.5 else "failing"
            
            if win_rate > 0.5:
                self.stats["scripts_passing"] += 1
            if win_rate > self.stats["best_win_rate"]:
                self.stats["best_win_rate"] = win_rate
            
            results.append({
                "script_id": script["id"],
                "win_rate": win_rate,
                "status": script["status"],
            })
        
        return results
    
    # ── Phase 5: EVOLVE ───────────────────────────────────
    
    def evolve(self) -> float:
        """
        One full evolution cycle:
        1. Analyze patterns from all transitions
        2. Generate scripts from patterns
        3. Test scripts
        4. Keep winners, discard losers
        5. Return best win rate
        """
        self.generation += 1
        print(f"\n{'='*50}")
        print(f"  {self.name} — Generation {self.generation}")
        print(f"{'='*50}")
        
        print(f"  Transitions: {len(self.transitions)}")
        print(f"  Vector DB: {self.vdb.count()} entries")
        
        # Analyze
        patterns = self.analyze_patterns()
        print(f"  Patterns discovered: {len(patterns)}")
        for p in patterns[:5]:
            print(f"    [{p['type']}] conf={p.get('confidence',0):.2f} samples={p.get('sample_size',0)}")
        
        # Generate scripts
        new_scripts = self.generate_scripts(patterns)
        print(f"  Scripts generated: {len(new_scripts)}")
        
        # We need a game for testing — return the pattern info
        # Testing happens in the main loop
        
        self._save_state()
        return patterns
    
    # ── Helpers ────────────────────────────────────────────
    
    def _extract_features(self, state_str: str) -> list[str]:
        """Extract features from a state string for pattern matching."""
        features = []
        # Bigram features
        for i in range(0, len(state_str) - 1, 2):
            features.append(state_str[i:i+2])
        # Positional features
        parts = state_str.split('|')
        for part in parts:
            if '=' in part:
                features.append(part)
        return features
    
    def _script_action(self, state: GameState, legal_actions: list[str]) -> str:
        """Use best passing script to choose an action."""
        for script in reversed(self.scripts):
            if script["status"] == "passing":
                try:
                    namespace = {}
                    exec(script["code"], namespace)
                    choice = namespace["choose_action"](str(state), legal_actions)
                    if choice and choice in legal_actions:
                        return choice
                except:
                    pass
        return random.choice(legal_actions)
    
    def _best_pattern_action(self, state: GameState, legal_actions: list[str]) -> str:
        """Use vector DB to find best action for similar states."""
        best_action = None
        best_reward = -999
        
        similar = self.vdb.search(str(state), top_k=10)
        for _, sim, meta in similar:
            if meta.get("action") in legal_actions:
                if meta.get("reward", 0) > best_reward:
                    best_reward = meta["reward"]
                    best_action = meta["action"]
        
        return best_action if best_action else random.choice(legal_actions)
    
    # ── State ─────────────────────────────────────────────
    
    def _save_state(self):
        state = {
            "name": self.name,
            "game": self.game_name,
            "generation": self.generation,
            "stats": self.stats,
            "scripts": self.scripts,
        }
        with open(self.sandbox_dir / "state.json", "w") as f:
            json.dump(state, f, indent=2, default=str)
        self._save_gpu_state()
    
    def _load_state(self):
        state_file = self.sandbox_dir / "state.json"
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            self.generation = state.get("generation", 0)
            self.stats.update(state.get("stats", {}))
            self.scripts = state.get("scripts", [])
    
    def _save_gpu_state(self):
        """Persist GPU engine vectors alongside the SQLite DB."""
        if self.gpu_engine and len(self.gpu_engine) > 0:
            path = str(self.sandbox_dir / "gpu_engine.pt")
            self.gpu_engine.save(path)
    
    def _load_gpu_state(self):
        """Load GPU engine vectors if a saved state exists."""
        if self.gpu_engine:
            path = self.sandbox_dir / "gpu_engine.pt"
            if path.exists():
                self.gpu_engine.load(str(path))


# ─── Cross-Game Pattern Mining ───────────────────────────

def cross_game_mining(output_path: str = "cross_game_patterns.json"):
    """
    Load tic-tac-toe and Connect4 GPU vector DBs,
    use the GPU engine to find cross-game patterns,
    and save the top 10 most-similar states between games.
    """
    print("\n" + "=" * 60)
    print("  Cross-Game Pattern Mining")
    print("=" * 60)
    
    # Build GPU engines for each game by running exploration
    ttt = TicTacToe()
    c4 = Connect4()
    
    ttt_claw = ZeroClaw("cross-ttt", "tictactoe")
    c4_claw = ZeroClaw("cross-c4", "connect4")
    
    # Collect transitions
    print("  Collecting tic-tac-toe transitions...")
    ttt_claw.explore(ttt, num_games=200)
    print("  Collecting Connect4 transitions...")
    c4_claw.explore(c4, num_games=200)
    
    # Build dedicated GPU engines for each game's states
    ttt_engine = GPUVectorEngine(dim=64)
    c4_engine = GPUVectorEngine(dim=64)
    
    ttt_states = [f"{t.state_str}|{t.action}" for t in ttt_claw.transitions]
    ttt_meta = [{"game": "tictactoe", "state": t.state_str, "action": t.action, "reward": t.reward} for t in ttt_claw.transitions]
    
    c4_states = [f"{t.state_str}|{t.action}" for t in c4_claw.transitions]
    c4_meta = [{"game": "connect4", "state": t.state_str, "action": t.action, "reward": t.reward} for t in c4_claw.transitions]
    
    ttt_vecs = ttt_engine.hash_embed_batch(ttt_states)
    ttt_engine.add_batch(ttt_vecs, ttt_meta)
    
    c4_vecs = c4_engine.hash_embed_batch(c4_states)
    c4_engine.add_batch(c4_vecs, c4_meta)
    
    print(f"  TTT GPU index: {len(ttt_engine)} vectors")
    print(f"  C4 GPU index: {len(c4_engine)} vectors")
    
    # Cross-game search: find most similar states between the two games
    print("  Running cross-game similarity search...")
    cross_results = ttt_engine.cross_game_search(c4_engine, top_k=10)
    
    patterns = []
    for ttt_idx, c4_idx, sim in cross_results:
        ttt_info = ttt_engine.metadata[ttt_idx] if ttt_idx < len(ttt_engine.metadata) else {}
        c4_info = c4_engine.metadata[c4_idx] if c4_idx < len(c4_engine.metadata) else {}
        patterns.append({
            "rank": len(patterns) + 1,
            "similarity": round(sim, 6),
            "ttt_state": ttt_info.get("state", "?")[:50],
            "ttt_action": ttt_info.get("action", "?"),
            "ttt_reward": ttt_info.get("reward", 0),
            "c4_state": c4_info.get("state", "?")[:50],
            "c4_action": c4_info.get("action", "?"),
            "c4_reward": c4_info.get("reward", 0),
        })
    
    # Save results
    with open(output_path, "w") as f:
        json.dump({"cross_game_patterns": patterns, "ttt_vectors": len(ttt_engine), "c4_vectors": len(c4_engine)}, f, indent=2)
    
    print(f"\n  Top 10 cross-game patterns:")
    for p in patterns:
        print(f"    #{p['rank']}: sim={p['similarity']:.4f} | TTT({p['ttt_action']}, r={p['ttt_reward']:.1f}) <-> C4({p['c4_action']}, r={p['c4_reward']:.1f})")
    
    print(f"\n  Saved to {output_path}")
    return patterns


# ─── Main: Run the ZeroClaw Arena ────────────────────────

def run_arena(games=None, num_explore=50, num_evolve=3, num_exploit=50):
    """Run multiple ZeroClaws learning different games simultaneously.
    
    Args:
        games: list of game names to run (default: all)
        num_explore: games per exploration phase
        num_evolve: number of evolution generations  
        num_exploit: games in exploit/test phase
    """
    
    print("╔══════════════════════════════════════════════════╗")
    print("║         ZEROCLAW ARENA — Game Learning           ║")
    print("╚══════════════════════════════════════════════════╝")
    
    all_games = {
        "tictactoe": TicTacToe,
        "blackjack": Blackjack,
        "connect4": Connect4,
        "go9x9": Go9x9,
    }
    
    # Add chess if available
    try:
        import chess
        all_games["chess_endgame"] = ChessEndgame
    except ImportError:
        pass

    # Filter if specific games requested
    if games:
        game_classes = {k: v for k, v in all_games.items() if k in games}
    else:
        game_classes = all_games

    games = {name: cls() for name, cls in game_classes.items()}
    
    # Create ZeroClaws
    claws = {}
    for game_name, game in games.items():
        claw_name = f"zeroclaw-{game_name}"
        claws[game_name] = ZeroClaw(claw_name, game_name)
    
    # ── Run evolution generations ──────────────────────────
    for gen in range(1, num_evolve + 1):
        print(f"\n{'#'*60}")
        print(f"  GENERATION {gen}")
        print(f"{'#'*60}")
        
        for game_name, game in games.items():
            claw = claws[game_name]
            
            # Phase 1: Explore (play games)
            print(f"\n🎮 {game_name} — Exploration")
            explore_n = num_explore if game_name not in ("chess_endgame", "go9x9") else max(num_explore // 5, 3)
            claw.explore(game, num_games=explore_n)
            
            # Phase 2: Analyze and generate scripts
            print(f"\n🔬 {game_name} — Pattern Analysis")
            patterns = claw.evolve()
            
            # Phase 3: Test scripts
            print(f"\n⚡ {game_name} — Script Testing")
            results = claw.test_scripts(game, num_tests=100)
            passing = sum(1 for r in results if r["status"] == "passing")
            print(f"  Scripts tested: {len(results)}, passing: {passing}")
            
            # Phase 4: Play with learned scripts
            print(f"\n🧠 {game_name} — Script-Guided Play")
            prev_stats = dict(claw.stats)
            claw.explore(game, num_games=num_exploit)
            
            # Check improvement
            old_wr = prev_stats["wins"] / max(prev_stats["games_played"], 1)
            new_wr = claw.stats["wins"] / max(claw.stats["games_played"], 1)
            improvement = new_wr - old_wr
            
            print(f"  Win rate: {old_wr:.1%} → {new_wr:.1%} (Δ={improvement:+.1%})")
            print(f"  Best script win rate: {claw.stats['best_win_rate']:.1%}")
            print(f"  Scripts passing: {claw.stats['scripts_passing']}")
        
        # ── Generation Summary ───────────────────────────
        print(f"\n{'='*60}")
        print(f"  Generation {gen} Summary:")
        for game_name, claw in claws.items():
            wr = claw.stats["wins"] / max(claw.stats["games_played"], 1)
            print(f"    {game_name:15s}: win_rate={wr:.1%} scripts={claw.stats['scripts_passing']} best={claw.stats['best_win_rate']:.1%}")
        print(f"{'='*60}")
    
    # ── Final Report ─────────────────────────────────────
    print(f"\n{'#'*60}")
    print(f"  FINAL RESULTS — ZeroClaw Arena")
    print(f"{'#'*60}")
    
    report = {}
    for game_name, claw in claws.items():
        wr = claw.stats["wins"] / max(claw.stats["games_played"], 1)
        report[game_name] = {
            "games_played": claw.stats["games_played"],
            "win_rate": f"{wr:.1%}",
            "best_script_wr": f"{claw.stats['best_win_rate']:.1%}",
            "scripts_passing": claw.stats["scripts_passing"],
            "scripts_total": claw.stats["scripts_generated"],
            "generations": claw.generation,
            "transitions": len(claw.transitions),
            "vector_db_size": claw.vdb.count(),
        }
        
        print(f"\n  {game_name}:")
        for k, v in report[game_name].items():
            print(f"    {k}: {v}")
        
        # Save best scripts
        best = [s for s in claw.scripts if s["status"] == "passing"]
        if best:
            best.sort(key=lambda s: -s["win_rate"])
            with open(claw.sandbox_dir / "best_scripts.json", "w") as f:
                json.dump(best, f, indent=2, default=str)
            print(f"    Best scripts saved to {claw.sandbox_dir / 'best_scripts.json'}")
    
    with open("/tmp/zeroclaw-arena-report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n  Report saved to /tmp/zeroclaw-arena-report.json")
    
    return report


if __name__ == "__main__":
    run_arena()
