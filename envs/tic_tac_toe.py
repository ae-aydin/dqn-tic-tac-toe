import random
from enum import Enum

import numpy as np

from .env import Environment


class Player(Enum):
    X = -1
    O = 1

PLAYERS = (Player.X, Player.O)
PLAYER_ENCODING = {Player.X: 'X', Player.O:'O'}
BOARD_SHAPE = (3, 3)
NUM_PLAYERS = 2

class TicTacToe(Environment):
    def __init__(
        self,
        board_shape: tuple = BOARD_SHAPE,
        num_players: int = NUM_PLAYERS,
        players: tuple = PLAYERS,
        enc_players: dict = PLAYER_ENCODING,
    ):
        super().__init__(board_shape, num_players, players)
        self.enc_players = enc_players

    def reset(self):
        self.board = np.zeros(self.board.shape, dtype=np.int8)
        self.starting_player = random.choice(self.players)
        self.current_player = self.starting_player
        self.valid_actions = np.argwhere(self.board.flatten() == 0).flatten()
        return self.last()[0], self.current_player

    def _sample_action(self):
        if len(self.valid_actions) > 0:
            return random.choice(self.valid_actions)
        return -1

    def _act(self, pos: int, player: Player):
        pos_tuple = divmod(pos, self.board.shape[0])
        if self._is_valid_action(pos_tuple):
            self.board[pos_tuple] = player.value

    def _is_valid_action(self, pos: tuple):
        return (self.board[pos] == 0) and (self._check_done() == -2)

    def _render(self):
        rendered_board = list()
        for r in range(self.board.shape[0]):
            row = self.board[r]
            row_str = " | ".join([self.enc_players.get(e, " ") for e in row])
            if r == 0:
                rendered_board.append("".join(["-" for _ in range(len(row_str))]))
            rendered_board.append(row_str)
            rendered_board.append("".join(["-" for _ in range(len(row_str))]))
        return "\n".join(rendered_board)

    def _check_done(self):
        game_status = -2
        for player in self.players:
            if self._check(player):
                game_status = player.value
                return game_status
        game_status = -2 if np.any(self.board == 0) else 0
        return game_status

    def _check(self, player: Player):
        vertical = np.any(np.sum(self.board, axis=0) == (self.board.shape[0] * player.value))
        horizontal = np.any(
            np.sum(self.board, axis=1) == (self.board.shape[1] * player.value)
        )
        diagonals = self._check_diagonal(player.value)
        return vertical or horizontal or diagonals

    def _check_diagonal(self, player: int):
        diagonal_n, diagonal_p = 0, 0
        for i in range(self.board.shape[0]):
            diagonal_n += self.board[i, i]
            diagonal_p += self.board[i, self.board.shape[0] - 1 - i]
        return (diagonal_n == self.board.shape[0] * player) or (
            diagonal_p == self.board.shape[0] * player
        )

    def update_valid_actions(self):
        self.valid_actions = np.argwhere(self.board.flatten() == 0).flatten()

    def step(self, action: int, player: Player, render=False):
        if (self._check_done() == -2) and action != -1:
            self.update_valid_actions()
            assert player == self.current_player
            self._act(action, player)
            self.update_valid_actions()
            next_player = Player.X if self.current_player == Player.O else Player.O
            self.current_player = next_player
            if render:
                print(self._render(), "\n")
            return self.last()
        else:
            print("Game ended. Status:", self._check_done())
            return

    def last(self):
        state = self.board.flatten()
        done = self._check_done()
        terminate = True if done != -2 else False
        truncate = None
        reward = done if done != -2 else 0
        return state, reward, terminate, truncate
