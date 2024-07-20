from collections import deque, namedtuple
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from env import Environment
import torch
import time


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class Memory:
    def __init__(self, size: int) -> None:
        self.memory = deque([], maxlen=size)

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


class DQNAgent:
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
        update_freq: int = 2**10,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.memory = Memory(memory_size)
        self.policy_net = MLP(env.state_space, env.action_space, hidden_layers)
        self.target_net = MLP(env.state_space, env.action_space, hidden_layers)
        self.target_net.eval()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=learning_rate
        )
        self.criterion = nn.SmoothL1Loss()
        self.n_steps = total_steps
        self.batch_size = batch_size
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.update_freq = update_freq

    def learn(self):
        state, player = self.env.reset()
        for step in range(self.n_steps):
            action = self._act_e_greedy(state, step)
            new_state, reward, term, trunc = self.env.step(int(action))
            transition = Transition(state, action, new_state, reward)
            done = term or trunc
            self.memory.push(transition)
            state = new_state
            self._optimize()
            if ((step + 1) % self.tau) == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            if done:
                state = self.env.reset()

    def _optimize(self):
        if len(self.memory) < self.batch_size:
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

        state_q = self.policy_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            estimate_q = self.target_net(next_state_batch).max(1).values

        estimate_q = (self.gamma * estimate_q) + reward_batch

        loss = self.criterion(state_q, estimate_q.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def load(self, agent):
        self.policy_net.load_state_dict(agent.policy_net.state_dict())
        self.target_net.load_state_dict(agent.target_net.state_dict())

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
            q_values = self.policy_net(torch.Tensor(state))
            mask = torch.full(q_values.shape, float("-inf"))
            mask[self.env.valid_actions] = q_values[self.env.valid_actions]
            return torch.argmax(mask).item()

    def _act_random(self):
        return self.env._sample_valid_action()


class DQNAgentSelfPlay:
    def __init__(self, env, hidden_layers: list, update_every: int = 2**13) -> None:
        self.main_agent = DQNAgent(env, hidden_layers)
        self.main_elo = 1000
        self.opp_elo = 1000
        self.opponent_agent = DQNAgent(env, hidden_layers)
        self.update_every = update_every

    def self_play(self):
        state, current_player = self.main_agent.env.reset()
        episode = 0
        rewards = 0
        for step in range(self.main_agent.n_steps):
            main_player_id = current_player
            opp_id = -main_player_id
            action = self.main_agent._act_e_greedy(state, step)
            new_state, reward, done, trunc = self.main_agent.env.step(
                action, main_player_id
            )
            transition = Transition(state, action, new_state, reward)
            self.main_agent.memory.push(transition)
            state = -new_state
            if done:
                state, current_player = self.main_agent.env.reset()
                main_player_id = current_player
                episode += 1
                if main_player_id == reward:
                    self.update_elo(self.main_agent)
                else:
                    self.update_elo(self.opponent_agent)
                if (episode + 1) % 100 == 0:
                    print("main-elo:", self.main_elo, "opp-elo:", self.opp_elo)
                continue

            opp_action = self.opponent_agent._act_e_greedy(state, step)
            new_state, reward, done, trunc = self.main_agent.env.step(
                opp_action, opp_id
            )
            opp_transition = Transition(state, opp_action, new_state, reward)
            self.main_agent.memory.push(opp_transition)
            state = -new_state
            if done:
                state, current_player = self.main_agent.env.reset()
                main_player_id = current_player
                episode += 1
                if main_player_id == reward:
                    self.update_elo(self.main_agent)
                else:
                    self.update_elo(self.opponent_agent)
                if (episode + 1) % 100 == 0:
                    print("main-elo:", self.main_elo, "opp-elo:", self.opp_elo)
                continue

            self.main_agent._optimize()

            if (step + 1) % self.main_agent.update_freq == 0:
                print("update-target-network")
                self.main_agent.target_net.load_state_dict(
                    self.main_agent.policy_net.state_dict()
                )

            if (step + 1) % self.update_every == 0:
                self.opponent_agent.load(self.main_agent)
                print("update-opponent")

            # time.sleep(1)

    def update_elo(self, winner):
        e_main = (10 ** ((self.opp_elo - self.main_elo) / 400) + 1) ** -1
        e_opp = (10 ** ((self.main_elo - self.opp_elo) / 400) + 1) ** -1
        if winner == self.main_agent:
            self.main_elo = self.main_elo + 16 * (1 - e_main)
            self.opp_elo = self.opp_elo + 16 * (0 - e_opp)
        else:
            self.main_elo = self.main_elo + 16 * (0 - e_main)
            self.opp_elo = self.opp_elo + 16 * (1 - e_opp)
