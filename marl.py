import logging
import pickle

import numpy as np
from tqdm import tqdm

from algo.dqn import DQN, Transition
from algo.q_learning import QLearning
from envs.env import Environment, Player
from envs.tic_tac_toe import TicTacToe
from agent.agent import RandomAgent
from eval import eval

# https://ai.stackexchange.com/questions/10032/why-isnt-my-q-learning-agent-able-to-play-tic-tac-toe?rq=1
# https://ai.stackexchange.com/questions/6573/how-can-both-agents-know-the-terminal-reward-in-self-play-reinforcement-learning


def duel_tabular_q_train(env: Environment, n_episodes: int):
    """Training two individual Tabular-Q-Learning agent.

    Args:
        env (Environment): Environment setting.
        n_episodes (int): Total training episodes.
    """

    agents = list()
    player_agent_dict = dict()
    total_steps = 100_000
    for idx in range(env.num_players):
        agent = QLearning(total_steps, player=env.players[idx])
        agents.append(agent)
        player_agent_dict[agent.player] = agent
    n_steps, win, lose, draw = 0, 0, 0, 0
    pbar = tqdm(
        range(n_episodes),
        desc=f"steps={n_steps+1} | {win}/{draw}/{lose} | episodes",
        ncols=150,
    )
    for ep in pbar:
        obs = env.reset()
        while not obs.terminate:
            n_steps += 1
            acting_agent = player_agent_dict[env.current_player]
            action = acting_agent.step(obs)
            obs = env.step(action, acting_agent.player)

        for agent in agents:
            agent.step(obs)

        if obs.reward[agents[0].player] == 1:
            logging.debug("AGENT X WIN")
            win += 1
        elif obs.reward[agents[0].player] == -1:
            logging.debug("AGENT O WIN")
            lose += 1
        else:
            logging.debug("DRAW")
            draw += 1
        logging.debug(f"performance={win}/{draw}/{lose}, episodes={ep}")
        pbar.set_description(f"steps={n_steps+1} | {win}/{draw}/{lose} | episodes")

    for p, a in player_agent_dict.items():
        with open(f"{str(a)}_{str(p)}.pkl", "wb") as file:
            pickle.dump(a.q, file)
        print(a.q[tuple(np.zeros(env.num_actions))])


def duel_dqn_train(env: Environment, n_episodes: int):
    """Training two individual DQN agent.

    Args:
        env (Environment): Environment setting.
        n_episodes (int): Total training episodes.
    """

    agents = list()
    player_agent_dict = dict()
    hidden_layers = [18, 36, 18]
    total_steps = 50_000
    for idx in range(env.num_players):
        agent = DQN(
            env.state_size,
            env.num_actions,
            hidden_layers,
            total_steps,
            player=env.players[idx],
        )
        agents.append(agent)
        player_agent_dict[agent.player] = agent
    n_steps, win, lose, draw = 0, 0, 0, 0
    pbar = tqdm(
        range(n_episodes),
        desc=f"steps={n_steps+1} | {win}/{draw}/{lose} | episodes",
        ncols=150,
    )
    for ep in pbar:
        obs = env.reset()
        while not obs.terminate:
            n_steps += 1
            acting_agent = player_agent_dict[env.current_player]
            action = acting_agent.step(obs)
            obs = env.step(action, acting_agent.player)

        for agent in agents:
            agent.step(obs)

        if obs.reward[agents[0].player] == 1:
            logging.debug("AGENT X WIN")
            win += 1
        elif obs.reward[agents[0].player] == -1:
            logging.debug("AGENT O WIN")
            lose += 1
        else:
            logging.debug("DRAW")
            draw += 1
        logging.debug(f"performance={win}/{draw}/{lose}, episodes={ep}")
        pbar.set_description(f"steps={n_steps+1} | {win}/{draw}/{lose} | episodes")

    for p, a in player_agent_dict.items():
        a.save(f"{str(a)}_{str(p)}.pt")


def self_play_dqn_train(env: Environment, n_episodes: int):
    """Training two individual DQN agent.

    Args:
        env (Environment): Environment setting.
        n_episodes (int): Total training episodes.
    """

    agents = list()
    player_agent_dict = dict()
    hidden_layers = [18, 36, 18]
    total_steps = 100_000
    agent = DQN(
        env.state_size,
        env.num_actions,
        hidden_layers,
        total_steps,
    )
    agents.append(agent)
    agents.append(agent)
    for idx in range(env.num_players):
        player_agent_dict[env.players[idx]] = agent
    n_steps, win, lose, draw = 0, 0, 0, 0
    pbar = tqdm(
        range(n_episodes),
        desc=f"steps={n_steps+1} | {win}/{draw}/{lose} | episodes",
        ncols=150,
    )
    for ep in pbar:
        obs = env.reset()
        while not obs.terminate:
            n_steps += 1
            acting_agent = player_agent_dict[env.current_player]
            acting_agent.player = env.current_player
            action = acting_agent.step(obs)
            obs = env.step(action, acting_agent.player)
            # print(acting_agent.eps_schedule.steps_taken)

        for agent in agents:
            agent.step(obs)

        if obs.reward[agents[0].player] == 1:
            if agent.player == env.players[0]:
                logging.debug("AGENT X WIN")
                win += 1
            else:
                logging.debug("AGENT O WIN")
                lose += 1
        else:
            logging.debug("DRAW")
            draw += 1
        logging.debug(f"performance={win}/{draw}/{lose}, episodes={ep}")
        pbar.set_description(f"steps={n_steps+1} | {win}/{draw}/{lose} | episodes")

    agent.save(f"{str(agent)}_SelfPlay.pt")


def train(agents: list, env: Environment, n_episodes: int):
    self_play = True if len(agents) == 1 else False
    player_agent_dict = dict()
    n_steps, win, draw, lose = 0, 0, 0, 0  # first player perspective

    if self_play:
        agents.append(agents[0])

    for idx in range(env.num_players):
        if not self_play:
            agents[idx].player = env.players[idx]
        player_agent_dict[env.players[idx]] = agents[idx]

    episodes = tqdm(
        range(n_episodes),
        desc=f"steps={n_steps+1} | {win}/{draw}/{lose} | episodes",
        ncols=125,
    )

    for _ in episodes:
        obs = env.reset()
        while not obs.terminate:
            n_steps += 1
            acting_agent = player_agent_dict[env.current_player]
            if self_play:
                acting_agent.player = env.current_player
                action = acting_agent.step(obs, self_play=False)
            else:
                action = acting_agent.step(obs)
            obs = env.step(action, acting_agent.player)

        for idx in range(len(agents)):
            agent = agents[idx]
            if self_play:
                agent.player = env.players[idx]
            agent.step(obs)

        if env.winner:
            if env.winner == env.players[0]:
                win += 1
            elif env.winner == env.players[1]:
                lose += 1
            else:
                draw += 1
        episodes.set_description(f"steps={n_steps+1} | {win}/{draw}/{lose} | episodes")

    if self_play:
        agents[0].save("SelfPlay")
    else:
        for p, a in player_agent_dict.items():
            a.save(str(p))


if __name__ == "__main__":
    env = TicTacToe()
    episodes = 50_000
    agents = list()
    agent_1 = DQN(env.state_size, env.num_actions, [18, 36, 18], episodes * 2)
    agents.append(agent_1)
    train(agents, env, episodes)
    agent_1.player = Player(1)
    eval(agent_1, RandomAgent(Player(-1)), env)
    agent_1.player = Player(-1)
    eval(RandomAgent(Player(1)), agent_1, env)
