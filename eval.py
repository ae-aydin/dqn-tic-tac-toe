import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import agent.agent as agent
import algo.dqn as dqn
import algo.q_learning as ql
from algo.algo import Algorithm
from envs.env import Environment, Player
from envs.tic_tac_toe import TicTacToe

# Model evaluation.

STR_AGENT_ENC = {dqn.DQN_STR: dqn.DQN, ql.QL_STR: ql.QLearning}
ARG_PLAYER_ENC = {"player_a": Player(1), "player_b": Player(-1)}


def extract_player(agent_str: str, arg_str: str, env: Environment):
    """Infer desired agent from command line argument.

    Args:
        agent_str (str): Value of the argument.
        arg_str (str): Name of the argument.
        env (Environment): Environment.

    Returns:
        Agent or Algorithm: Extracted Agent.
    """
    player = ARG_PLAYER_ENC[arg_str]
    if agent_str == "random":
        return agent.RandomAgent(player)
    elif agent_str == "human":
        return agent.HumanAgent(player)
    else:
        agent_str = Path(agent_str)
        agent_algo = agent_str.name.split("_")[0]
        algo = STR_AGENT_ENC[agent_algo]
        if algo == dqn.DQN:
            algo = algo(env.state_size, env.num_actions, [18, 36, 18], 100_000, player)
            algo.load(agent_str, True)
            return algo
        elif algo == ql.QLearning:
            algo = algo(100_000, player)
            algo.load(agent_str)
            return algo
        else:
            return


def arrange_players(args: argparse.Namespace, env: Environment):
    """Arrange players according to command line arguments.

    Args:
        args (argparse.Namespace): Command line arguments.
        env (Environment): Environment

    Returns:
        Agents or Algorithms: Extracted Agents.
    """
    player_a = extract_player(args.player_a, "player_a", env)
    player_b = extract_player(args.player_b, "player_b", env)
    return player_a, player_b


def eval(
    player_a, player_b, env: Environment, n_episodes: int = 1000, eps: float = 0.0
):
    """Evaluate given agents against each other. (Agent-Agent, Agent-Random, Agent-Human).

    Args:
        player_a (_type_): Agent to play player_a.
        player_b (_type_): Agent to play player_b.
        env (Environment): Environment
        n_episodes (int, optional): Total test episodes. Defaults to 1000.
        eps (float, optional): Exploration rate for algorithm agents. Defaults to 0.0.

    Returns:
        dict: Win/Draw/Lose rates.
    """
    agents = [player_a, player_b]
    human = next((a for a in agents if isinstance(a, agent.HumanAgent)), None)
    random = next((a for a in agents if isinstance(a, agent.RandomAgent)), None)
    algo = next((a for a in agents if isinstance(a, Algorithm)), None)
    results = {"win": 0, "draw": 0, "lose": 0, "eps": eps, "n_episodes": n_episodes}
    player_agent_dict = dict()
    for a in agents:
        player_agent_dict[a.player] = a

    if human:
        print("Enter your move as a number from 1-9.")
        print(env.render())

    episodes = (
        range(1) if human else tqdm(range(n_episodes), desc=f"epsilon={eps} | Episodes")
    )
    for _ in episodes:
        obs = env.reset()
        while not obs.terminate:
            acting_agent = player_agent_dict[env.current_player]
            if isinstance(acting_agent, Algorithm):
                action = acting_agent.step(obs, True, eps=eps)
            else:
                action = acting_agent.step(env)

            obs = env.step(action, acting_agent.player)

            if human:
                print(f"{acting_agent} turn, selected move is {action+1}.")
                print(env.render())

        if obs.reward[agents[0].player] == 1:
            results["win"] += 1
            if human:
                print(f"{agents[0]} wins")
        elif obs.reward[agents[0].player] == -1:
            results["lose"] += 1
            if human:
                print(f"{agents[1]} wins")
        else:
            results["draw"] += 1
            if human:
                print("Draw")

    if not human:
        print(f"{agents[0]}_{agents[0].player}={results['win']}")
        print(f"{agents[1]}_{agents[1].player}={results['lose']}")
        print(f"Draw={results['draw']}")
        if random:
            sum_reward = results["win"] - results["lose"]
            sum_reward = -sum_reward if algo.player == Player(-1) else sum_reward
            print(f"AverageReward={sum_reward / n_episodes}")
        return results


def measure_performance(player_a, player_b, env: Environment, n_episodes: int = 1000):
    """Plot win-draw-lose rate of given agents.

    Args:
        player_a: Agent to play player_a.
        player_b: Agent to play player_b.
        env (Environment): Environment
        n_episodes (int, optional): Total test episodes. Defaults to 1000.
    """
    epsilons = np.arange(0, 0.055, 0.005)
    eps, wins, draws, losses = list(), list(), list(), list()
    player_a_str = f"{str(player_a)}_{player_a.player}"
    player_b_str = f"{str(player_b)}_{player_b.player}"
    for epsilon in epsilons:
        result = eval(player_a, player_b, env, n_episodes, epsilon)
        eps.append(result["eps"])
        wins.append(result["win"])
        draws.append(result["draw"])
        losses.append(result["lose"])
    plt.figure(figsize=(12, 6))
    plt.plot(epsilons, wins, label=f"{player_a_str} Wins", marker="o")
    plt.plot(epsilons, draws, label="Draws", marker="o")
    plt.plot(epsilons, losses, label=f"{player_b_str} Wins", marker="o")
    plt.xlabel("Epsilon")
    plt.ylabel("Count over 1000 games")
    plt.title(f"{player_a_str} versus {player_b_str}")
    plt.legend()
    plt.grid()
    plt.savefig(f"{player_a_str} versus {player_b_str}.png", bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description="Evalute RL models.")
    parser.add_argument(
        "player_a", type=str, help="The first player (agent1, random, human, etc.)"
    )
    parser.add_argument(
        "player_b", type=str, help="The second player (agent2, random, human, etc.)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot performance across different epsilon values.",
    )
    args = parser.parse_args()
    env = TicTacToe()
    player_a, player_b = arrange_players(args, env)
    print(player_a, player_b)
    eval(player_a, player_b, env)
    if args.plot:
        measure_performance(player_a, player_b, env, 1000)


if __name__ == "__main__":
    main()
