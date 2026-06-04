"""Tests for zeroclaw package."""
import pytest
from zeroclaw import (
    TicTacToe, Connect4, Go9x9, HoldemHand,
    GameState, Transition, TileField, CompiledPolicy,
)


class TestTicTacToe:
    def test_initial_state(self):
        g = TicTacToe()
        assert g.state().state_str == "         "
        assert len(g.legal_actions()) == 9
        assert not g.done

    def test_play_to_completion(self):
        g = TicTacToe()
        g.step("4")  # X center
        g.step("0")  # O top-left
        g.step("2")  # X top-right
        g.step("1")  # O top-middle
        g.step("6")  # X bottom-left — diagonal win
        assert g.done
        assert g.winner == 'X'

    def test_draw(self):
        g = TicTacToe()
        moves = ["0", "1", "2", "4", "3", "5", "7", "6", "8"]
        for m in moves:
            reward, done = g.step(m)
        assert g.done
        assert g.winner is None  # draw

    def test_illegal_move(self):
        g = TicTacToe()
        g.step("4")
        reward, done = g.step("4")  # occupied
        assert reward == -1.0
        assert done

    def test_reset(self):
        g = TicTacToe()
        g.step("4")
        g.reset()
        assert g.state().state_str == "         "
        assert not g.done

    def test_copy(self):
        g = TicTacToe()
        g.step("4")
        c = g.copy()
        assert c.state().state_str == g.state().state_str
        c.step("0")
        assert c.state().state_str != g.state().state_str

    def test_gamestate_hash(self):
        g = TicTacToe()
        s = g.state()
        assert isinstance(s.hash(), str)
        assert len(s.hash()) > 0


class TestConnect4:
    def test_initial_state(self):
        g = Connect4()
        assert len(g.legal_actions()) == 7
        assert not g.done

    def test_drop_piece(self):
        g = Connect4()
        g.step("3")  # drop in middle
        state = g.state().state_str
        assert state[5 * 7 + 3] == 'X'  # bottom-middle

    def test_win_horizontal(self):
        g = Connect4()
        # X: cols 0,1,2,3  O: cols 0,1,2 (blocking attempts)
        g.step("0")  # X col0
        g.step("0")  # O col0
        g.step("1")  # X col1
        g.step("1")  # O col1
        g.step("2")  # X col2
        g.step("2")  # O col2
        g.step("3")  # X col3 — win!
        assert g.done
        assert g.winner == 'X'

    def test_copy(self):
        g = Connect4()
        g.step("3")
        c = g.copy()
        assert c.state().state_str == g.state().state_str


class TestGo9x9:
    def test_initial_state(self):
        g = Go9x9()
        actions = g.legal_actions()
        assert 'pass' in actions
        assert len(actions) > 1  # many empty positions

    def test_place_stone(self):
        g = Go9x9()
        reward, done = g.step("4,4")
        assert not done
        assert g.board[4][4] == 'B'

    def test_pass_ends_game(self):
        g = Go9x9()
        g.step("pass")  # B passes
        g.step("pass")  # W passes — game over
        assert g.done

    def test_copy(self):
        g = Go9x9()
        g.step("4,4")
        c = g.copy()
        assert c.board[4][4] == 'B'


class TestHoldemHand:
    def test_initial_state(self):
        g = HoldemHand()
        assert len(g.legal_actions()) == 5
        assert not g.done

    def test_fold(self):
        g = HoldemHand()
        reward, done = g.step("fold")
        assert done

    def test_copy(self):
        g = HoldemHand()
        g.step("check_call")
        c = g.copy()
        assert c.stage == g.stage


class TestTileField:
    def test_train_ttt(self):
        game = TicTacToe()
        field = TileField(n_simulations=5, temperature=0.5)
        field.train(game, num_games=10)
        assert field.size > 0

    def test_choose_action(self):
        game = TicTacToe()
        field = TileField(n_simulations=2)
        field.train(game, num_games=5)
        game.reset()
        action = field.choose_action(game, game.state().state_str, game.legal_actions())
        assert action in game.legal_actions()

    def test_evolve(self):
        game = TicTacToe()
        field = TileField(n_simulations=2)
        field.train(game, num_games=10)
        field.evolve()
        # Scores should be updated
        for tile in field.tiles.values():
            for data in tile.values():
                assert 0.05 <= data["score"] <= 0.95


class TestCompiledPolicy:
    def test_compile(self):
        game = TicTacToe()
        field = TileField(n_simulations=5)
        field.train(game, num_games=100)
        policy = CompiledPolicy.from_tile_field(field)
        assert policy.size > 0

    def test_policy_returns_action(self):
        game = TicTacToe()
        field = TileField(n_simulations=5)
        field.train(game, num_games=100)
        policy = CompiledPolicy.from_tile_field(field)
        action = policy("         ")
        assert isinstance(action, str)

    def test_evaluate(self):
        game = TicTacToe()
        field = TileField(n_simulations=5)
        field.train(game, num_games=50)
        policy = CompiledPolicy.from_tile_field(field)
        results = policy.evaluate(num_games=50)
        assert results["total_games"] == 50
        assert "win_rate" in results

    def test_to_python(self):
        game = TicTacToe()
        field = TileField(n_simulations=5)
        field.train(game, num_games=50)
        policy = CompiledPolicy.from_tile_field(field)
        source = policy.to_python()
        assert "def compiled_policy" in source
        assert "_lookup" in source


class TestArenaModes:
    def test_tile_mode(self):
        from zeroclaw.arena import run_arena
        results = run_arena(
            games=["tictactoe"],
            mode="tile",
            num_train=20,
            num_eval=20,
            output_dir="/tmp/zeroclaw-test-results",
        )
        assert "tictactoe" in results
        assert results["tictactote"]["mode"] == "tile" if "tictactote" in results else results["tictactoe"]["mode"] == "tile"

    def test_random_mode(self):
        from zeroclaw.arena import run_arena
        results = run_arena(
            games=["tictactoe"],
            mode="random",
            num_exploit=20,
            output_dir="/tmp/zeroclaw-test-results",
        )
        assert "tictactoe" in results
        assert results["tictactoe"]["win_rate"] >= 0
