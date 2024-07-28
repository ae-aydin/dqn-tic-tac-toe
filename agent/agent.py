from abc import ABC, abstractmethod

import numpy as np

from envs.env import Environment, Player


class BaseAgent(ABC):
    def __init__(self, player: Player = Player(0)):
        self.player = player

    @abstractmethod
    def select_action(self, state: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        return "BaseAgent"


class RandomAgent(BaseAgent):
    def __init__(self, player: Player = Player(0)):
        super().__init__(player)

    def select_action(self, valid_actions: list):
        # TODO
        pass

    def __str__(self):
        return "RandomAgent"


class HumanAgent(BaseAgent):
    def __init__(self, player: Player = Player(0)):
        super().__init__(player)

    def select_action(self, n_actions: int):
        while True:
            try:
                action = int(input(f"Enter your move (1-{n_actions}): ")) - 1
                assert 0 <= action < n_actions, "Invalid input, try again."
                return action
            except ValueError:
                print("Please enter a valid number.")
            except AssertionError as e:
                print(e)
            except KeyboardInterrupt:
                print("Terminating game.")
                return None


    def __str__(self):
        return "HumanAgent"
