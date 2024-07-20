import numpy as np
from abc import ABC, abstractmethod
import random


class Environment(ABC):
    def __init__(self, board_shape: tuple, num_players: int, players: tuple):
        self.board_shape = board_shape
        self.num_players = num_players
        self.players = players
        self.board = np.zeros(board_shape, dtype=np.int8)
        self.current_player = self.players[0]
        self.starting_player = random.choice(players)
        self.valid_actions = np.argwhere(self.board.flatten() == 0).flatten()
        self.state_space = np.prod(board_shape)
        self.action_space = len(self.valid_actions)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action: int):
        pass

    @abstractmethod
    def _render(self):
        pass

    @abstractmethod
    def _check_done(self):
        pass

    @abstractmethod
    def last(self):
        pass

    def _update_valid_actions(self):
        self.valid_actions = np.argwhere(self.board.flatten() == 0).flatten()

    def _act(self, pos: int, player: int):
        pos_tuple = divmod(pos, self.board.shape[0])
        if self._is_valid_action(pos_tuple, player):
            self.board[pos_tuple] = player

    def _is_valid_action(self, pos: tuple, player: int):
        return (self.board[pos] == 0) and (self._check_done() == -2)

    def _sample_valid_action(self):
        return random.choice(self.valid_actions) if len(self.valid_actions) > 0 else -1

    def _sample_random_action(self):
        return random.choice(range(self.state_space))
