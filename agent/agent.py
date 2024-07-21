from abc import ABC, abstractmethod

import numpy as np

from env.env import Environment


class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, state: np.ndarray):
        pass


class RandomAgent(BaseAgent):
    def select_action(self, env: Environment):
        return env._sample_action()

    def __str__(self) -> str:
        return "RandomAgent"


class HumanAgent(BaseAgent):
    def select_action(self, env: Environment):
        while True:
            try:
                action = int(input(f"Enter your move (1-{len(env.state_space)}): ")) - 1
                pos = divmod(action)
                if env._is_valid_action(pos):
                    return action
                else:
                    print("Invalid move. Try again.")
            except ValueError:
                print("Invalid input. Enter an integer.")