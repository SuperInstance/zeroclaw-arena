"""
Game implementations for ZeroClaw Arena.

All games share a common interface:
    - state() -> GameState
    - legal_actions() -> list[str]
    - step(action: str) -> tuple[float, bool]
    - reset()
    - done: bool
    - winner: Optional[str]
    - copy() -> self (for Monte Carlo simulation)
"""

import random
import hashlib
from dataclasses import dataclass
from typing import Optional


@dataclass
class GameState:
    """Serialized game state for vector embedding."""
    state_str: str
    turn: int
    player: str

    def __str__(self):
        return f"[turn={self.turn}|{self.player}]{self.state_str}"

    def __repr__(self):
        return f"GameState({self.state_str!r}, turn={self.turn}, player={self.player!r})"

    def hash(self):
        return hashlib.blake2b(str(self).encode(), digest_size=8).hexdigest()


@dataclass
class Transition:
    """One state transition: (state, action) → (reward, next_state)."""
    state_hash: str
    state_str: str
    action: str
    reward: float
    next_state_hash: str
    next_state_str: str
    game_over: bool
    winner: Optional[str] = None


class TicTacToe:
    """Tic-tac-toe with standard rules. X goes first."""

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

        lines = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                 (0, 3, 6), (1, 4, 7), (2, 5, 8),
                 (0, 4, 8), (2, 4, 6)]
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

    def copy(self):
        g = TicTacToe()
        g.board = self.board[:]
        g.current = self.current
        g.turn = self.turn
        g.done = self.done
        g.winner = self.winner
        return g


class Connect4:
    """Connect 4 on a 6×7 board. X=Red, O=Yellow."""

    def __init__(self, rows: int = 6, cols: int = 7):
        self.rows = rows
        self.cols = cols
        self.reset()

    def reset(self):
        self.board = [[' '] * self.cols for _ in range(self.rows)]
        self.current = 'X'
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

        row = self.rows - 1
        while row >= 0 and self.board[row][col] != ' ':
            row -= 1
        if row < 0:
            return -1.0, True

        self.board[row][col] = self.current
        self.turn += 1

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for sign in [1, -1]:
                r, c = row + sign * dr, col + sign * dc
                while (0 <= r < self.rows and 0 <= c < self.cols
                       and self.board[r][c] == self.current):
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= 4:
                self.done = True
                self.winner = self.current
                reward = 1.0 if self.current == 'X' else -1.0
                return reward, True

        if self.turn >= self.rows * self.cols:
            self.done = True
            return 0.0, True

        self.current = 'O' if self.current == 'X' else 'X'
        return 0.0, False

    def copy(self):
        g = Connect4.__new__(Connect4)
        g.rows = self.rows
        g.cols = self.cols
        g.board = [row[:] for row in self.board]
        g.current = self.current
        g.turn = self.turn
        g.done = self.done
        g.winner = self.winner
        return g


class Go9x9:
    """Simplified 9×9 Go with Chinese scoring and komi 5.5."""

    def __init__(self, size: int = 9):
        self.size = size
        self.reset()

    def reset(self):
        self.board = [['.' for _ in range(self.size)] for _ in range(self.size)]
        self.current = 'B'
        self.turn = 0
        self.done = False
        self.winner = None
        self.captures = {'B': 0, 'W': 0}
        self.previous_board = None
        self.passes = 0
        self.komi = 5.5

    def state(self) -> GameState:
        board_str = ''.join(''.join(row) for row in self.board)
        return GameState(
            f"{board_str}_C{self.captures['B']}_{self.captures['W']}",
            self.turn, self.current,
        )

    def legal_actions(self) -> list[str]:
        if self.done:
            return []
        actions = ['pass']
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] == '.' and self._is_legal(r, c):
                    actions.append(f"{r},{c}")
        return actions

    def _is_legal(self, r: int, c: int) -> bool:
        test_board = [row[:] for row in self.board]
        test_board[r][c] = self.current

        opp = 'W' if self.current == 'B' else 'B'
        captured = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < self.size and 0 <= nc < self.size
                    and test_board[nr][nc] == opp):
                _, liberties = self._get_group(test_board, nr, nc)
                if liberties == 0:
                    captured += 1

        if captured == 0:
            _, liberties = self._get_group(test_board, r, c)
            if liberties == 0:
                return False

        board_str = ''.join(''.join(row) for row in test_board)
        if self.previous_board and board_str == self.previous_board:
            return False

        return True

    def _get_group(self, board, r: int, c: int):
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
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
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
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < self.size and 0 <= nc < self.size
                    and self.board[nr][nc] == opp):
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

    def _get_territory(self, r: int, c: int):
        visited = set()
        colors = set()
        stack = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            if (cr, cc) in visited:
                continue
            visited.add((cr, cc))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
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

    def copy(self):
        import copy
        g = Go9x9.__new__(Go9x9)
        g.size = self.size
        g.board = [row[:] for row in self.board]
        g.current = self.current
        g.turn = self.turn
        g.done = self.done
        g.winner = self.winner
        g.captures = dict(self.captures)
        g.previous_board = self.previous_board
        g.passes = self.passes
        g.komi = self.komi
        return g


class HoldemHand:
    """One hand of Texas Hold'em poker for tile-based learning.

    Simplified: 2 players, 4 stages (preflop/flop/turn/river),
    actions are fold/check_call/raise_small/raise_big/bluff.
    """

    STAGES = ["preflop", "flop", "turn", "river"]

    def __init__(self):
        self.deck = self._make_deck()
        random.shuffle(self.deck)
        self.hole = [[], []]  # Player 0, Player 1
        self.community = []
        self.pot = 0
        self.bets = [0, 0]
        self.folded = [False, False]
        self.stage = 0
        self.done = False
        self.winner = None
        self.actions_history = []

        for _ in range(2):
            self.hole[0].append(self.deck.pop())
            self.hole[1].append(self.deck.pop())
        self.pot = 2  # Blinds

    @staticmethod
    def _make_deck():
        ranks = '23456789TJQKA'
        suits = 'CDHS'
        return [r + s for r in ranks for s in suits]

    def reset(self):
        self.__init__()

    def state(self) -> GameState:
        stage_name = self.STAGES[self.stage]
        return GameState(
            f"stage={stage_name}|pot={self.pot}|bets={self.bets}",
            self.stage, f"player0",
        )

    def legal_actions(self) -> list[str]:
        if self.done:
            return []
        return ["fold", "check_call", "raise_small", "raise_big", "bluff"]

    def deal_community(self, n: int):
        for _ in range(n):
            self.community.append(self.deck.pop())

    def hand_strength_bucket(self, player: int) -> int:
        """Bucket hand strength 0-4."""
        if not self.community:
            ranks = '23456789TJQKA'
            r1 = ranks.index(self.hole[player][0][0])
            r2 = ranks.index(self.hole[player][1][0])
            suited = self.hole[player][0][1] == self.hole[player][1][1]
            score = (r1 + r2) / 24.0 + (0.1 if suited else 0) + (0.1 if r1 == r2 else 0)
            return min(4, int(score * 5))
        # With community cards: simplified strength
        return min(4, random.randint(0, 4))

    def pot_bucket(self) -> int:
        if self.pot < 6:
            return 0
        if self.pot < 15:
            return 1
        return 2

    def play_round(self, actions: list[str]):
        for i, action in enumerate(actions):
            player = i % 2
            if self.folded[player]:
                continue
            if action == "fold":
                self.folded[player] = True
                self.done = True
                self.winner = 1 - player
                return self.winner
            elif action == "check_call":
                call_amount = max(self.bets) - self.bets[player]
                self.bets[player] += call_amount
                self.pot += call_amount
            elif action in ("raise_small", "raise_big", "bluff"):
                call_amount = max(self.bets) - self.bets[player]
                raise_amount = max(2, self.pot // 3) if action == "raise_small" else max(4, self.pot)
                self.bets[player] += call_amount + raise_amount
                self.pot += call_amount + raise_amount
            self.actions_history.append((self.STAGES[self.stage], player, action))
        return None

    def step(self, action: str) -> tuple[float, bool]:
        """Take one action for player 0, random for player 1."""
        p1_actions = ["check_call", "raise_small", "raise_big"]
        p1_action = random.choice(p1_actions)
        result = self.play_round([action, p1_action])
        if result is not None:
            reward = 1.0 if result == 0 else -1.0
            return reward, True
        # Advance stage
        self.stage += 1
        if self.stage == 1:
            self.deal_community(3)
        elif self.stage in (2, 3):
            self.deal_community(1)
        elif self.stage >= 4:
            # Showdown - simplified
            self.done = True
            self.winner = random.choice([0, 1])
            reward = 1.0 if self.winner == 0 else -1.0
            return reward, True
        return 0.0, False

    def copy(self):
        import copy
        g = HoldemHand.__new__(HoldemHand)
        g.deck = self.deck[:]
        g.hole = [h[:] for h in self.hole]
        g.community = self.community[:]
        g.pot = self.pot
        g.bets = self.bets[:]
        g.folded = self.folded[:]
        g.stage = self.stage
        g.done = self.done
        g.winner = self.winner
        g.actions_history = list(self.actions_history)
        return g
