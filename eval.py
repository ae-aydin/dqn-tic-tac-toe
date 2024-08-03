import argparse
import agent.agent as agent
from algo.algo import Algorithm
import algo.dqn as dqn
import algo.q_learning as ql
from envs.env import Player, Environment
from envs.tic_tac_toe import TicTacToe
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

STR_AGENT_ENC = {dqn.DQN_STR: dqn.DQN, ql.QL_STR: ql.QLearning}
ARG_PLAYER_ENC = {"player_a": Player(1), "player_b": Player(-1)}


def extract_player(agent_str: str, arg_str: str, env: Environment):
    player = ARG_PLAYER_ENC[arg_str]
    if agent_str == "random":
        return agent.RandomAgent(player)
    elif agent_str == "human":
        return agent.HumanAgent(player)
    else:
        agent_algo = agent_str.split("_")[0]
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
    player_a = extract_player(args.player_a, "player_a", env)
    player_b = extract_player(args.player_b, "player_b", env)
    return player_a, player_b


def eval(
    player_a, player_b, env: Environment, n_episodes: int = 1000, eps: float = 0.0
):
    agents = [player_a, player_b]
    results = {"win": 0, "draw": 0, "lose": 0, "eps": eps, "n_episodes": n_episodes}
    player_agent_dict = dict()
    for a in agents:
        player_agent_dict[a.player] = a

    human = any(isinstance(a, agent.HumanAgent) for a in agents)

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
        print(f"{agents[0]}_{agents[0].player}={results["win"]}")
        print(f"{agents[1]}_{agents[1].player}={results["lose"]}")
        print(f"Draw={results["draw"]}")
        return results


def measure_performance(player_a, player_b, env: Environment, n_episodes: int = 1000):
    epsilons = np.arange(0, 0.105, 0.005)
    results = dict()
    results["player_a"] = f"{str(player_a)}_{player_a.player}"
    results["player_b"] = f"{str(player_b)}_{player_b.player}"
    for eps in epsilons:
        results[eps] = eval(player_a, player_b, env, n_episodes, eps)
    print(results)


def main():
    parser = argparse.ArgumentParser(description="Evalute RL models.")
    parser.add_argument(
        "player_a", type=str, help="The first player (agent1, random, human, etc.)"
    )
    parser.add_argument(
        "player_b", type=str, help="The second player (agent2, random, human, etc.)"
    )
    parser.add_argument(
        "--env", type=Environment, default=TicTacToe, help="Environment setting."
    )
    args = parser.parse_args()
    env = args.env()
    player_a, player_b = arrange_players(args, env)
    print(player_a, player_b)
    # eval(player_a, player_b, env)
    measure_performance(player_a, player_b, env, 1000)


if __name__ == "__main__":
    main()
