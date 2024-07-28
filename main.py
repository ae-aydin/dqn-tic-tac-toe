import argparse
import logging

from algo.dqn import DQN
from algo.q_learning import QLearning
from envs.tic_tac_toe import TicTacToe
from marl import agent_vs_human_play, duel_tabular_q_train


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

    if args.mode == "self":
        pass
    elif args.mode == "random":
        pass
    elif args.mode == "play":
        env = TicTacToe()
        agent_vs_human_play(env, QLearning(env.action_space), args.load)
    elif args.mode == "ql":
        env = TicTacToe()
        agents = [QLearning(env.action_space) for _ in range(env.num_players)]
        duel_tabular_q_train(agents, env, 100000)


if __name__ == "__main__":
    main()
