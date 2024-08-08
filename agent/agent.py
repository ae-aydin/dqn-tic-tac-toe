from abc import ABC, abstractmethod

from envs.env import Environment, Player


# Base Agent class.
class BaseAgent(ABC):
    def __init__(self, player: Player = Player(0)):
        self.player = player

    @abstractmethod
    def step(self):
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        return "BaseAgent"


# Agent that plays randomly.
class RandomAgent(BaseAgent):
    def __init__(self, player: Player = Player(0)):
        super().__init__(player)

    def step(self, env: Environment):
        return env._sample_action()

    def __str__(self):
        return "RandomAgent"


# Agent allowing human input through terminal.
class HumanAgent(BaseAgent):
    def __init__(self, player: Player = Player(0)):
        super().__init__(player)

    def step(self, env: Environment):
        while True:
            try:
                action = int(input(f"Enter your move (1-{env.num_actions}): ")) - 1
                if not (0 <= action < env.num_actions):
                    raise ValueError("Action out of range.")

                if not env._is_valid_action(divmod(action, env.board_shape[0])):
                    print("Invalid move, try again.")
                    continue
                return action
            except ValueError as e:
                print(e)
            except KeyboardInterrupt:
                print("Terminating game.")
                return None

    def __str__(self):
        return "HumanAgent"
