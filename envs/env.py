import random
from abc import ABC, abstractmethod
from collections import namedtuple
from enum import Enum

import numpy as np

Observation = namedtuple(
    "Observation", ("state", "valid_actions", "reward", "terminate")
)


class Player(Enum):
    X = 1
    O = -1
    NONE = 0


class Environment(ABC):
    def __init__(
        self,
        board_shape: tuple,
        num_players: int,
        players: tuple,
        starting_player: Player,
    ):
        self.board_shape = board_shape
        self.num_players = num_players
        self.players = players
        self.board = np.zeros(board_shape, dtype=np.int8)
        self.starting_player = starting_player
        self.current_player = self.starting_player
        self.valid_actions = self._update_valid_actions()
        self.state_space = np.prod(board_shape)
        self.action_space = len(self.valid_actions)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action: int, player: int):
        pass

    @abstractmethod
    def _check_done(self):
        pass

    @abstractmethod
    def _update_valid_actions(self):
        pass

    @abstractmethod
    def _act(self, pos: int, player: int):
        pass

    @abstractmethod
    def _is_valid_action(self, pos: tuple, player: int):
        pass

    @abstractmethod
    def _sample_action(self):
        pass

    def _sample_random_action(self):
        return random.choice(range(self.state_space))

    @abstractmethod
    def _out(self):
        pass

    @abstractmethod
    def last(self):
        pass
