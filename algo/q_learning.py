import logging
import pickle
import random
from collections import defaultdict

import numpy as np

from algo.algo import Algorithm
from algo.epsilon import EpsilonLinearDecay
from envs.env import Observation, Player


def create_q_table():
    return defaultdict(float)


class QLearning(Algorithm):
    def __init__(
        self,
        approx_steps: int,
        player: Player = Player(0),
        eps_init: float = 1.0,
        eps_final: float = 0.1,
        learning_rate: float = 0.2,
        gamma: float = 0.99,
    ) -> None:
        super().__init__()
        self.q = defaultdict(create_q_table)
        self.player = player
        self.eps_schedule = EpsilonLinearDecay(eps_init, eps_final, approx_steps)
        self.epsilon = self.eps_schedule.value
        self.learning_rate = learning_rate
        self.gamma = gamma
        self._prev_state = None
        self._prev_action = None

    def step(self, obs: Observation, eval: bool = False):
        action = None
        if not obs.terminate:
            action = self._act_e_greedy(obs.state, obs.valid_actions, eval)

        if self._prev_state:
            self._update(obs)

        if obs.terminate:
            self._prev_state = None
            return

        self._prev_state = obs.state
        self._prev_action = action
        return action

    def _update(self, obs: Observation):
        q_current = self.q[self._prev_state][self._prev_action]
        q_next = (
            max([self.q[obs.state][a] for a in obs.valid_actions])
            if not obs.terminate
            else 0
        )
        q_target = obs.reward[self.player] + self.gamma * q_next
        self.q[self._prev_state][self._prev_action] += self.learning_rate * (
            q_target - q_current
        )
        logging.debug(
            f"{self._prev_state} {self.q[self._prev_state]}, {self._prev_action}"
        )

    def _act_e_greedy(self, state: tuple, valid_actions: list, eval: bool):
        logging.debug(valid_actions)
        epsilon = 0.0 if eval else self.epsilon
        if np.random.rand() < 1 - epsilon:
            logging.debug("Acting greedy.")
            action = self._act_greedy(state, valid_actions)
        else:
            logging.debug("Acting random.")
            action = self._act_random(valid_actions)
        logging.debug(f"action={action}")
        self.epsilon = self.eps_schedule.step()
        return action

    def _act_greedy(self, state: tuple, valid_actions: list):
        q_values = self.q[state]
        logging.debug(q_values)
        return max(valid_actions, key=lambda x: q_values[x])

    def _act_random(self, valid_actions: list):
        return random.choice(valid_actions)

    def load(self, fname: str, eval: bool):
        with open(fname, "rb") as pickle_file:
            self.q = pickle.load(pickle_file)

    def __str__(self) -> str:
        return "Q-Learning"
