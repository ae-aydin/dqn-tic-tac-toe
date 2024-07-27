import logging
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn

from agent.agent import RandomAgent
from envs.env import Environment
from tqdm import tqdm

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


class DQN:
    def __init__(
        self,
        env: Environment,
        hidden_layers: list,
        memory_size: int = 2**14,
        n_episodes: int = 10**4,
        batch_size: int = 2**6,
        eps_max: float = 0.9,
        eps_min: float = 0.1,
        eps_decay: float = 0.0001,
        learning_rate: float = 0.0001,  # alpha
        gamma: float = 0.99,  # discount
        target_update_freq: int = 2**9,
        tau: float = 0.005,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.hidden_layers = hidden_layers
        self.memory = Memory(memory_size)
        self.q_net = MLP(env.state_space, env.action_space, hidden_layers)
        self.target_net = MLP(env.state_space, env.action_space, hidden_layers)
        self.target_net.eval()
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.tau = tau

    def self_play(self, load: str, soft_update: bool = True):
        if load is not None:
            self.load(load, False)
        n_steps = 0
        win, lose, draw = 0, 0, 0
        pbar = tqdm(range(self.n_episodes), desc=f"steps={n_steps+1} | {win}/{draw}/{lose}| episodes", ncols=150)
        for ep in pbar:
            logging.debug(f"episode={ep}")
            state = self.env.reset()
            done = False
            a2p, p2a = self.get_agent_map(("dqn", "self"), self.env.players)
            main_agent = "dqn"
            logging.debug(a2p)
            while not done:
                n_steps += 1
                current_player = self.env.current_player
                if current_player != a2p[main_agent]:
                    state = -1 * state
                action = self._act_e_greedy(state, ep)
                new_state, reward, term, trunc = self.env.step(
                    action, current_player
                )
                if reward != 0:
                    reward = 1 if reward == a2p[main_agent].value else -1
                transition = Transition(-1 * state, action, new_state, reward)
                logging.debug(transition)
                self.memory.push(transition)
                self._optimize()
                done = term or trunc
                if not done:
                    state = new_state

                    if soft_update:
                        self._soft_update_target()
                    else:
                        if ((ep + 1) % self.target_update_freq) == 0:
                            self.target_net.load_state_dict(self.q_net.state_dict())
                else:
                    # self.memory.push(Transition(state, None, None, reward))
                    if reward == 1:
                        logging.debug("WIN")
                        win += 1
                    elif reward == -1:
                        logging.debug("LOSE")
                        lose += 1
                    else:
                        logging.debug("DRAW")
                        draw += 1
                    logging.debug(f"performance={win}/{draw}/{lose}, episodes={ep}")
                    pbar.set_description(f"steps={n_steps+1} | {win}/{draw}/{lose} | episodes")
        self.save("dqn.pt")

    def agent_vs_random(self, load: str, soft_update: bool = True):
        if load is not None:
            self.load(load, False)
        random_agent = RandomAgent()
        n_steps = 0
        win, lose, draw = 0, 0, 0
        pbar = tqdm(range(self.n_episodes), desc=f"steps={n_steps+1} | {win}/{draw}/{lose}| episodes", ncols=150)
        for ep in pbar:
            logging.debug(f"episode={ep}")
            state = self.env.reset()
            done = False
            a2p, p2a = self.get_agent_map(("dqn", "random_agent"), self.env.players)
            logging.debug(a2p)
            while not done:
                n_steps += 1
                current_player = self.env.current_player
                if current_player == a2p["dqn"]:
                    action = self._act_e_greedy(state, ep)
                elif current_player == a2p["random_agent"]:
                    logging.debug("random_agent playing.")
                    action = random_agent.select_action(self.env)
                    logging.debug(f"action={action}")
                new_state, reward, term, trunc = self.env.step(
                    action, current_player
                )
                if reward != 0:
                    reward = 1 if reward == a2p["dqn"].value else -1
                transition = Transition(state, action, new_state, reward)
                logging.debug(transition)
                self.memory.push(transition)
                done = term or trunc
                if not done:
                    self._optimize()
                    state = new_state

                    if soft_update:
                        self._soft_update_target()
                    else:
                        if ((n_steps + 1) % self.target_update_freq) == 0:
                            self.target_net.load_state_dict(self.q_net.state_dict())
                else:
                    if reward == 1:
                        logging.debug("WIN")
                        win += 1
                    elif reward == -1:
                        logging.debug("LOSE")
                        lose += 1
                    else:
                        logging.debug("DRAW")
                        draw += 1
                    logging.debug(f"performance={win}/{draw}/{lose}, episodes={ep}")
                    pbar.set_description(f"steps={n_steps+1} | {win}/{draw}/{lose} | episodes")
        self.save("random.pt")

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
        logging.debug(f"epsilon={epsilon:.5f}")
        if np.random.rand() < 1 - epsilon:
            action = self._act_greedy(state)
        else:
            logging.debug("Acting random.")
            action = self._act_random()
            logging.debug(f"action={action}")
        return action

    def _calc_epsilon(self, step: int):
        return self.eps_min + (self.eps_max - self.eps_min) * np.exp(
            -self.eps_decay * step
        )

    def _act_greedy(self, state: tuple):
        logging.debug("Acting greedy.")
        with torch.no_grad():
            q_values = self.q_net(torch.Tensor(state))
            logging.debug(f"action_values={[f"({i}, {float(v):.5f})" for i, v in enumerate(q_values.numpy())]}")
            logging.debug(f"valid_actions={self.env.valid_actions}")
            mask = torch.full(q_values.shape, float("-inf"))
            mask[self.env.valid_actions] = q_values[self.env.valid_actions]
            action = torch.argmax(mask).item()
            logging.debug(f"action={action}")
            return action

    def _act_random(self):
        return self.env._sample_action()

    def _soft_update_target_old(self):
        q_state_dict = self.q_net.state_dict()
        t_state_dict = self.target_net.state_dict()
        for key in q_state_dict:
            t_state_dict[key] = q_state_dict[key] * self.tau + t_state_dict[key] * (
                1 - self.tau
            )
        self.target_net.load_state_dict(t_state_dict)

    def _soft_update_target(self):
        target_net_state_dict = self.target_net.state_dict()
        q_net_state_dict = self.q_net.state_dict()
        for key in q_net_state_dict:
            target_net_state_dict[key] = q_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def save(self, fname):
        torch.save(self.q_net.state_dict(), fname)

    def load(self, fname, eval: bool):
        weights = torch.load(fname)
        self.q_net.load_state_dict(weights)
        self.target_net.load_state_dict(weights)
        if eval:
            self.q_net.eval()
        self.target_net.eval()

    def load_from_agent(self, agent):
        self.q_net.load_state_dict(agent.policy_net.state_dict())
        self.target_net.load_state_dict(agent.target_net.state_dict())

    def get_agent_map(self, agents: tuple, players: tuple):
        agent_to_player = {agents[0]: random.choice(players), agents[1]: None}
        agent_to_player[agents[1]] = (
            players[1] if agent_to_player[agents[0]] == players[0] else players[0]
        )
        player_to_agent = {player: agent for agent, player in agent_to_player.items()}
        return agent_to_player, player_to_agent

