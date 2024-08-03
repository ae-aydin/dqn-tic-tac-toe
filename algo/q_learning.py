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


class QLearning(Algorithm):
    def __init__(
        self,
        approx_steps: int,  # for eps decay
        player: Player = Player(0),
        eps_init: float = 1.0,
        eps_final: float = 0.1,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
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

        if not obs.terminate:
            action = self._act_e_greedy(obs.state, obs.valid_actions, eval, eps)
        else:
            action = None

        if not eval:

            if self._prev_state[self.player]:
                self._update(obs)

            if obs.terminate:
                self._prev_state[self.player] = None
                self._prev_action[self.player] = None
                return
            else:
                self._prev_state[self.player] = obs.state
                self._prev_action[self.player] = action

        return action

    def _update(self, obs: Observation):
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
        epsilon = eps if eval else self.epsilon
        print(str(self), epsilon)
        if np.random.rand() < 1 - epsilon:
            action = self._act_greedy(state, valid_actions)
        else:
            action = self._act_random(valid_actions)
        self.epsilon = self.eps_schedule.step()
        return action

    def _act_greedy(self, state: tuple, valid_actions: list):
        q_values = self.q[state]
        return max(valid_actions, key=lambda x: q_values[x])

    def _act_random(self, valid_actions: list):
        return random.choice(valid_actions)

    def load(self, fname: str):
        with open(fname, "rb") as file:
            self.q = pickle.load(file)

    def save(self, fname: str):
        with open(f"{str(self)}_{fname}.pkl", "wb") as file:
            pickle.dump(self.q, file)

    def __str__(self) -> str:
        return QL_STR
