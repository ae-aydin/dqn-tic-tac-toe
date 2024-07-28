import logging
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from agent.agent import RandomAgent
from algo.algo import Algorithm
from algo.epsilon import EpsilonLinearDecay
from envs.env import Environment, Player

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class Memory:
    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self.memory = deque([], maxlen=max_size)

    def push(self, transition: Transition):
        self.memory.append(transition)

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_layers: list):
        super(MLP, self).__init__()
        layers = list()
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        layers.append(nn.ReLU())
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_layers[-1], output_size))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DQN(Algorithm):
    def __init__(
        self,
        state_space: int,
        action_space: int,
        hidden_layers: list,
        approx_steps: int,
        player: Player = Player(0),
        memory_size: int = 10_000,
        batch_size: int = 64,
        eps_init: float = 0.9,
        eps_final: float = 0.1,
        learning_rate: float = 0.0001,  # alpha
        gamma: float = 0.99,  # discount
        target_update_freq: int = 1000,  # target network update per this episode
        tau: float = 0.005,  # soft update rate
    ) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_layers = hidden_layers
        self.q_net = MLP(state_space, action_space, hidden_layers)
        self.target_net = MLP(state_space, action_space, hidden_layers)
        self.target_net.eval()
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.player = player
        self.memory = Memory(memory_size)
        self.batch_size = batch_size
        self.eps_schedule = EpsilonLinearDecay(eps_init, eps_final, approx_steps)
        self.epsilon = self.eps_schedule.value
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.tau = tau
        self._prev_state = None
        self._prev_action = None

    def _optimize(self):
        if len(self.memory) < self.memory.max_size:
            return

        experience_batch = self.memory.sample(self.batch_size)
        state_batch = torch.stack([torch.Tensor(obs.state) for obs in experience_batch])
        reward_batch = torch.Tensor(
            [obs.reward[self.player] for obs in experience_batch]
        )
        action_batch = (
            torch.Tensor([obs.action for obs in experience_batch])
            .to(torch.int64)
            .unsqueeze(1)
        )
        next_state_batch = torch.stack(
            [torch.Tensor(obs.next_state) for obs in experience_batch]
        )

        state_q = self.q_net(state_batch).gather(1, action_batch)
        with torch.no_grad():
            estimate_q = self.target_net(next_state_batch).max(1).values

        estimate_q = (self.gamma * estimate_q) + reward_batch
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_q, estimate_q.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.q_net.parameters(), 100)
        self.optimizer.step()
        return loss.item()

    def _act_e_greedy(self, state: tuple, valid_actions: list):
        if np.random.rand() < 1 - self.epsilon:
            logging.debug("Acting greedy.")
            action = self._act_greedy(state, valid_actions)
        else:
            logging.debug("Acting random.")
            action = self._act_random(valid_actions)
        logging.debug(f"action={action}")
        self.epsilon = self.eps_schedule.step()
        return action

    def _act_greedy(self, state: tuple, valid_actions: list):
        with torch.no_grad():
            q_values = self.q_net(torch.Tensor(state))
            # logging.debug(f"action_values={[f"({i}, {float(v):.5f})" for i, v in enumerate(q_values.numpy())]}")
            logging.debug(f"valid_actions={valid_actions}")
            mask = torch.full(q_values.shape, float("-inf"))
            mask[self.env.valid_actions] = q_values[valid_actions]
            action = torch.argmax(mask).item()
            return action

    def _act_random(self, valid_actions: list):
        return random.choice(valid_actions)

    def _soft_update_target(self):
        q_state_dict = self.q_net.state_dict()
        t_state_dict = self.target_net.state_dict()
        for key in q_state_dict:
            t_state_dict[key] = q_state_dict[key] * self.tau + t_state_dict[key] * (
                1 - self.tau
            )
        self.target_net.load_state_dict(t_state_dict)

    def save(self, fname):
        torch.save(self.q_net.state_dict(), fname)

    def load(self, fname, eval: bool):
        weights = torch.load(fname)
        self.q_net.load_state_dict(weights)
        self.target_net.load_state_dict(weights)
        if eval:
            self.q_net.eval()
        self.target_net.eval()
