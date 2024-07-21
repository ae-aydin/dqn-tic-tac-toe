import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn

from env.env import Environment

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
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DQN:
    def __init__(
        self,
        env: Environment,
        hidden_layers: list,
        memory_size: int = 2**10,
        total_steps: int = 10**5,
        batch_size: int = 2**5,
        eps_max: float = 0.9,
        eps_min: float = 0.05,
        eps_decay: float = 0.0005,
        learning_rate: float = 0.001,  # alpha
        gamma: float = 0.99,  # discount
        target_update_freq: int = 2**10,
        tau: float = 0.005,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.memory = Memory(memory_size)
        self.q_net = MLP(env.state_space, env.action_space, hidden_layers)
        self.target_net = MLP(env.state_space, env.action_space, hidden_layers)
        self.target_net.eval()
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.n_steps = total_steps
        self.batch_size = batch_size
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.tau = tau

    def learn(self, soft_update: bool = True):
        state, _ = self.env.reset()
        for step in range(self.n_steps):
            action = self._act_e_greedy(state, step)
            new_state, reward, term, trunc = self.env.step(int(action))
            transition = Transition(state, action, new_state, reward)
            done = term or trunc
            self.memory.push(transition)
            state = new_state
            self._optimize()

            if soft_update:
                self._soft_update_target()
            else:
                if ((step + 1) % self.target_update_freq) == 0:
                    self.target_net.load_state_dict(self.q_net.state_dict())

            if done:
                state, _ = self.env.reset()

    def _optimize(self):
        if len(self.memory) < self.memory.max_size // 2:
            return

        experience_batch = self.memory.sample(self.batch_size)
        state_batch = torch.stack([torch.Tensor(obs.state) for obs in experience_batch])
        reward_batch = torch.Tensor([obs.reward for obs in experience_batch])
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

    def _act_e_greedy(self, state: tuple, step: int):
        epsilon = self._calc_epsilon(step)
        if np.random.rand() < 1 - epsilon:
            action = self._act_greedy(state)
        else:
            action = self._act_random()
        return action

    def _calc_epsilon(self, step: int):
        return self.eps_min + (self.eps_max - self.eps_min) * np.exp(
            -self.eps_decay * step
        )

    def _act_greedy(self, state: tuple):
        with torch.no_grad():
            q_values = self.q_net(torch.Tensor(state))
            mask = torch.full(q_values.shape, float("-inf"))
            mask[self.env.valid_actions] = q_values[self.env.valid_actions]
            return torch.argmax(mask).item()

    def _act_random(self):
        return self.env._sample_action()

    def _soft_update_target(self):
        q_state_dict = self.q_net.state_dict()
        t_state_dict = self.target_net.state_dict()
        for key in q_state_dict:
            t_state_dict[key] = q_state_dict[key] * self.tau + t_state_dict[key] * (
                1 - self.tau
            )
        self.target_net.load_state_dict(t_state_dict)

    def load(self, agent):
        self.q_net.load_state_dict(agent.policy_net.state_dict())
        self.target_net.load_state_dict(agent.target_net.state_dict())

    def __str__(self) -> str:
        return "DQN"
