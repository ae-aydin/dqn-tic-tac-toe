import argparse
import logging

from agent.agent import RandomAgent
from algo.dqn import DQN
from algo.q_learning import QLearning
from envs.tic_tac_toe import PLAYERS, TicTacToe
from marl import agent_vs_human, agent_vs_random, train


def main():
    parser = argparse.ArgumentParser(description="Tic-Tac-Toe Q-Learning")

    parser.add_argument(
        "mode",
        choices=["self", "random", "play", "ql"],
        help="Mode to run: 'self' for self-play, 'random' to play against a random player, 'human' to play against a human",
    )
    parser.add_argument("--load", type=str, help="File to load Q-values from")
    parser.add_argument(
        "--verbose", action="store_true", help="Increase output verbosity"
    )

    args = parser.parse_args()

    logging_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("debug.log", mode="w")],
    )

    agent = DQN(TicTacToe(), [18, 36, 18])

    if args.mode == "self":
        agent.self_play(args.load)
    elif args.mode == "random":
        agent.agent_vs_random(args.load)
    elif args.mode == "play":
        agent_vs_human(agent, args.load)
    elif args.mode == "ql":
        env = TicTacToe()
        agents = [QLearning(env.action_space) for _ in range(env.num_players)]
        train(agents, env, 200000)


if __name__ == "__main__":
    main()
