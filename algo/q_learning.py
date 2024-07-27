import logging
from collections import defaultdict

import numpy as np
from envs.env import Environment, Player


class QLearning:
    def __init__(
        self,
        action_space: int,
        player: Player = Player(0),
        eps_max: float = 0.9,
        eps_min: float = 0.05,
        eps_decay: float = 5e-6, 
        learning_rate: float = 0.1,
        gamma: float = 0.99,
    ) -> None:
        self.q = defaultdict(lambda: np.zeros(action_space))
        self.player = player
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.learning_rate = learning_rate
        self.gamma = gamma

    def update(self, state: tuple, action: int, reward, next_state, done):
        q_current = self.q[state][action]
        q_next = max(self.q[next_state]) if not done else 0
        q_target = reward + self.gamma * q_next
        self.q[state][action] += self.learning_rate * (q_target - q_current)
        logging.debug(f"{state} {self.q[state]}, {action}")

    def act_e_greedy(self, state: tuple, ep: int, env: Environment):
        epsilon = self._calculate_epsilon(ep)
        if np.random.rand() < 1 - epsilon:
            logging.debug("acting greedy.")
            action = self._act_greedy(state, env.valid_actions)
        else:
            logging.debug("acting random.")
            action = self._act_random(env)
        logging.debug(env.valid_actions)
        logging.debug(f"action={action}")
        return action

    def _calculate_epsilon(self, ep: int):
        return self.eps_min + (self.eps_max - self.eps_min) * np.exp(
            -self.eps_decay * ep
        )

    def _act_greedy(self, state: tuple, valid_actions: list):
        q_values = self.q[state]
        logging.debug(q_values) 
        return max(valid_actions, key=lambda x: q_values[x])

    def _act_random(self, env: Environment):
        return env._sample_action()

    def __str__(self) -> str:
        return "Q-Learning"