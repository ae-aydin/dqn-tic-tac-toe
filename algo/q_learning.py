import pickle
import random
from collections import defaultdict

import numpy as np

from algo.algo import Algorithm
from algo.epsilon import EpsilonLinearDecay
from envs.env import Observation, Player

QL_STR = "Q-Learning"


def create_q_table():
    return defaultdict(float)


def init_prev():
    return None


# Tabular Q-Learning Implementation.
class QLearning(Algorithm):
    def __init__(
        self,
        approx_steps: int,  # for eps decay
        player: Player = Player(0),  # default player
        eps_init: float = 1.0,  # starting epsilon
        eps_final: float = 0.1,  # final epsilon
        learning_rate: float = 0.1,  # alpha
        gamma: float = 0.99,  # discount
    ) -> None:
        super().__init__()
        self.q = defaultdict(create_q_table)
        self.player = player
        self.eps_schedule = EpsilonLinearDecay(eps_init, eps_final, approx_steps)
        self.epsilon = self.eps_schedule.value
        self.learning_rate = learning_rate
        self.gamma = gamma
        self._prev_state = defaultdict(init_prev)
        self._prev_action = defaultdict(init_prev)

    def step(self, obs: Observation, eval: bool = False, eps: float = 0.0):
        """Progressing the model according to current observation.

        Args:
            obs (Observation): Current observation.
            eval (bool, optional): Evaluation mode. Defaults to False.
            eps (float, optional): Evaluation epsilon. Defaults to 0.0.

        Returns:
            int: Action ID.
        """

        # Select an action if episode is not terminated.
        if not obs.terminate:
            action = self._act_e_greedy(obs.state, obs.valid_actions, eval, eps)
        else:
            action = None

        if not eval:

            # If mid game, update the Q-Table according to observed state transition.
            if self._prev_state[self.player]:
                self._update(obs)

            # Keep the training loop consistent.
            if obs.terminate:
                self._prev_state[self.player] = None
                self._prev_action[self.player] = None
                return
            else:
                self._prev_state[self.player] = obs.state
                self._prev_action[self.player] = action

        return action

    def _update(self, obs: Observation):
        """Update the Q-Table according to current observation.

        Args:
            obs (Observation): Current observation.
        """
        q_current = self.q[self._prev_state[self.player]][
            self._prev_action[self.player]
        ]
        q_next = (
            max([self.q[obs.state][a] for a in obs.valid_actions])
            if not obs.terminate
            else 0
        )
        q_target = obs.reward[self.player] + self.gamma * q_next
        self.q[self._prev_state[self.player]][
            self._prev_action[self.player]
        ] += self.learning_rate * (q_target - q_current)

    def _act_e_greedy(
        self, state: tuple, valid_actions: list, eval: bool, eps: float = 0.0
    ):
        """Making the model act epsilon-greedy.

        Args:
            state (tuple): Information state.
            valid_actions (list): Legal actions to take.
            eval (bool): Evaluation mode.
            eps (float, optional): Evaluation epsilon. Defaults to 0.0.

        Returns:
            int: Action ID.
        """
        epsilon = eps if eval else self.epsilon
        if np.random.rand() < 1 - epsilon:
            action = self._act_greedy(state, valid_actions)
        else:
            action = self._act_random(valid_actions)
        if not eval:
            self.epsilon = self.eps_schedule.step()
        return action

    def _act_greedy(self, state: tuple, valid_actions: list):
        """Acting greedy by choosing the best action.

        Args:
            state (tuple): Information state.
            valid_actions (list): Legal actions to take.

        Returns:
            int: Action ID.
        """
        q_values = self.q[state]
        return max(valid_actions, key=lambda x: q_values[x])

    def _act_random(self, valid_actions: list):
        """Act randomly according to legal actions.

        Args:
            valid_actions (list): Legal actions to take.

        Returns:
            action: Action ID.
        """
        return random.choice(valid_actions)

    def load(self, fname: str):
        """Load saved model weights from a file.

        Args:
            fname (str): File name.
        """
        with open(fname, "rb") as file:
            self.q = pickle.load(file)

    def save(self, fname: str):
        """Save the model with the given filename.

        Args:
            fname (str): File name.
        """
        with open(f"{str(self)}_{fname}.pkl", "wb") as file:
            pickle.dump(self.q, file)

    def __str__(self) -> str:
        return QL_STR
