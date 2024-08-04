import logging
import pickle

import numpy as np
from tqdm import tqdm

from agent.agent import RandomAgent
from algo.dqn import DQN, Transition
from algo.q_learning import QLearning
from envs.env import Environment, Player
from envs.tic_tac_toe import TicTacToe
from eval import eval

# https://ai.stackexchange.com/questions/10032/why-isnt-my-q-learning-agent-able-to-play-tic-tac-toe?rq=1
# https://ai.stackexchange.com/questions/6573/how-can-both-agents-know-the-terminal-reward-in-self-play-reinforcement-learning


def train(agents: list, env: Environment, n_episodes: int):
    """Train given agents with given environment.

    Args:
        agents (list): List of agents, singleton array if self play.
        env (Environment): Environment
        n_episodes (int): Total training episodes
    """
    self_play = True if len(agents) == 1 else False
    player_agent_dict = dict()
    n_steps, win, draw, lose = 0, 0, 0, 0  # first player perspective

    if self_play:  # two same agent
        agents.append(agents[0])

    for idx in range(env.num_players):  # set up players and agents
        if not self_play:
            agents[idx].player = env.players[idx]
        player_agent_dict[env.players[idx]] = agents[idx]

    episodes = tqdm(
        range(n_episodes),
        desc=f"steps={n_steps+1} | {win}/{draw}/{lose} | episodes",
        ncols=125,
    )

    for _ in episodes:
        obs = env.reset()  # reset environment and get initial observation
        while not obs.terminate:
            n_steps += 1
            acting_agent = player_agent_dict[env.current_player]
            if self_play:  # set agent to currently playing player
                acting_agent.player = env.current_player
            action = acting_agent.step(obs)  # get action
            obs = env.step(action, acting_agent.player)  # get next observation

        for idx in range(len(agents)):
            agent = agents[idx]
            if self_play:
                agent.player = env.players[idx]
            agent.step(obs)  # make all agents reach final observation

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
    agent_1 = DQN(env.state_size, env.num_actions, [18, 36, 18], episodes)
    agent_2 = DQN(env.state_size, env.num_actions, [18, 36, 18], episodes)
    # agent_1 = QLearning(episodes)
    # agent_2 = QLearning(episodes)
    agents.append(agent_1)
    agents.append(agent_2)
    train(agents, env, episodes)
    # agent_1.player = Player(1)
    eval(agent_1, RandomAgent(Player(-1)), env)
    # agent_1.player = Player(-1)
    eval(RandomAgent(Player(1)), agent_2, env)
