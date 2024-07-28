import logging
import pickle

import numpy as np
from tqdm import tqdm

from agent.agent import HumanAgent, RandomAgent
from algo.algo import Algorithm
from algo.dqn import DQN, Transition
from algo.q_learning import QLearning
from envs.env import Environment


# TODO
def agent_vs_human_play(env: Environment, agent: Algorithm, fname: str):
    agent.load(fname, True)
    agents = [agent, HumanAgent()]

    print("Welcome to Tic-Tac-Toe!")
    agent_id = fname.split(".")[-2]
    if agent_id == "X":
        print("You are 'O', the AI is 'X'.")
        player_agent_dict = dict(zip(env.players, agents))
    elif agent_id == "O":
        print("You are 'X', the AI is 'O'.")
        player_agent_dict = dict(zip(env.players, agents[::-1]))
    else:
        return

    for p, a in player_agent_dict.items():
        a.player = p

    print(
        "Enter your move as a number from 1-9, where 1 is the top-left corner and 9 is the bottom-right corner."
    )

    obs = env.reset()
    print(env._out())
    while True:
        acting_agent = player_agent_dict[env.current_player]
        if isinstance(player_agent_dict[env.current_player], HumanAgent):
            while True:
                action = acting_agent.select_action(env.action_space)
                if action is None:
                    break
                valid_move = env._is_valid_action(divmod(action, env.board_shape[0]))
                if valid_move:
                    break
                else:
                    print("Invalid move, try again.")
        else:
            print("AI's turn.")
            action = acting_agent.step(obs, eval=True)
            print(f"AI's move is {action}.")

        obs = env.step(action, acting_agent.player)
        print(env._out())

        if obs.terminate:
            if obs.reward[agents[1].player] == 1:
                print("You win!")
            elif obs.reward[agents[1].player] == -1:
                print("You lose!")
            elif obs.reward == 0:
                print("It's a draw!")
            break

    print("Game Over!")


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
    approx_steps = (n_episodes * env.action_space) // 2
    for idx in range(env.num_players):
        agent = QLearning(env.action_space, approx_steps, env.players[idx])
        agents.append(agent)
        player_agent_dict[agent.player] = agent
    n_steps, win, lose, draw = 0, 0, 0, 0
    pbar = tqdm(
        range(n_episodes),
        desc=f"steps={n_steps+1} | {win}/{draw}/{lose}| episodes",
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
        print(a.q[tuple(np.zeros(env.action_space))])
        print(a.eps_schedule.steps_taken, a.epsilon)


def duel_dqn_train(env: Environment, n_episodes: int):
    """Training two individual DQN agent.

    Args:
        env (Environment): Environment setting.
        n_episodes (int): Total training episodes.
    """
    pass
