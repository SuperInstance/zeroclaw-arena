"""
TileField — learn game policies via Monte Carlo tile coding.

A TileField maps board states to scored actions. It uses:
- Monte Carlo simulation for exploration
- Softmax selection with temperature
- Learned score evolution from win/loss outcomes
"""

import random
import math


class TileField:
    """Tile field with softmax action selection + Monte Carlo simulation.

    Usage:
        field = TileField()
        # Training
        for _ in range(1000):
            game = TicTacToe()
            field.train_game(game)
        field.evolve()
        # Playing
        action = field.choose_action(game, game.state_str(), game.legal_actions())
    """

    def __init__(self, n_simulations: int = 20, temperature: float = 0.3):
        self.tiles = {}  # state_str -> {action: {"score", "chosen", "won"}}
        self.n_simulations = n_simulations
        self.temperature = temperature
        self._game_count = 0

    def get_or_create(self, state_str: str, legal_actions: list[str]) -> dict:
        if state_str not in self.tiles:
            self.tiles[state_str] = {
                a: {"score": 0.5, "chosen": 0, "won": 0} for a in legal_actions
            }
        tile = self.tiles[state_str]
        for a in legal_actions:
            if a not in tile:
                tile[a] = {"score": 0.5, "chosen": 0, "won": 0}
        return tile

    def choose_action(self, game, state_str: str, legal_actions: list[str]) -> str:
        """Choose action via Monte Carlo sim + learned scores + softmax."""
        if not legal_actions:
            return ''
        if len(legal_actions) == 1:
            return legal_actions[0]

        tile = self.get_or_create(state_str, legal_actions)

        action_values = {}
        sims_per = max(1, self.n_simulations // len(legal_actions))

        for action in legal_actions:
            sim_wins = 0
            for _ in range(sims_per):
                g = game.copy()
                g.step(action)
                while not g.done:
                    acts = g.legal_actions()
                    if not acts:
                        break
                    g.step(random.choice(acts))
                if getattr(g, 'winner', None) in ('X', 'B', 'player', 0):
                    sim_wins += 1

            sim_score = sim_wins / max(sims_per, 1)
            learned_score = tile[action]["score"]
            n_chosen = tile[action]["chosen"]
            confidence = min(n_chosen / 20.0, 0.8)
            action_values[action] = confidence * learned_score + (1 - confidence) * sim_score

        # Softmax selection
        actions_list = list(action_values.keys())
        values = [action_values[a] for a in actions_list]
        max_val = max(values)
        exp_vals = [math.exp(v - max_val) / self.temperature for v in values]
        total = sum(exp_vals)
        probs = [e / total for e in exp_vals]

        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return actions_list[i]
        return actions_list[-1]

    def record(self, state_str: str, action: str, won: bool):
        if state_str in self.tiles and action in self.tiles[state_str]:
            self.tiles[state_str][action]["chosen"] += 1
            if won:
                self.tiles[state_str][action]["won"] += 1

    def evolve(self):
        """Update scores based on accumulated win rates."""
        for tile in self.tiles.values():
            for action, data in tile.items():
                if data["chosen"] > 0:
                    wr = data["won"] / data["chosen"]
                    data["score"] += 0.05 * (wr - data["score"])
                    data["score"] = max(0.05, min(0.95, data["score"]))

    def train_game(self, game, evolve_every: int = 25) -> str:
        """Play one training game using the tile field. Returns winner."""
        game.reset()
        history = []

        while not game.done:
            state = game.state()
            actions = game.legal_actions()
            if not actions:
                break

            state_str = str(state.state_str)
            if hasattr(game, 'current') and game.current in ('X', 'B', 'player', 0):
                action = self.choose_action(game, state_str, actions)
            else:
                action = random.choice(actions)

            game.step(action)
            history.append((state_str, action))

        won = getattr(game, 'winner', None) in ('X', 'B', 'player', 0)
        for state_str, action in history:
            self.record(state_str, action, won)

        self._game_count += 1
        if self._game_count % evolve_every == 0:
            self.evolve()

        return getattr(game, 'winner', None)

    def train(self, game, num_games: int = 100, evolve_every: int = 25):
        """Train the tile field on multiple games."""
        wins = {'X': 0, 'O': 0, 'draw': 0, 'B': 0, 'W': 0, None: 0}
        for i in range(num_games):
            winner = self.train_game(game, evolve_every)
            if winner in wins:
                wins[winner] += 1
            else:
                wins[None] += 1

            if (i + 1) % 50 == 0:
                total = sum(wins.values())
                x_wins = wins.get('X', 0) + wins.get('B', 0)
                print(f"  {i + 1}/{num_games} | tiles={len(self.tiles)} | "
                      f"P1 wins={x_wins / total:.1%}")

        return wins

    @property
    def size(self) -> int:
        return len(self.tiles)
