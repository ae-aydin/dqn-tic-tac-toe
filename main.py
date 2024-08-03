import argparse
import logging

from algo.dqn import DQN
from algo.q_learning import QLearning
from envs.tic_tac_toe import TicTacToe
from marl import (
    agent_vs_human_play,
    duel_dqn_train,
    duel_tabular_q_train,
    self_play_dqn_train,
)


def main():
    parser = argparse.ArgumentParser(description="Tic-Tac-Toe Q-Learning")

    parser.add_argument(
        "mode",
        choices=["dqn", "random", "play", "ql", "self"],
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

    if args.mode == "dqn":
        env = TicTacToe()
        duel_dqn_train(env, 50000)
    elif args.mode == "random":
        pass
    elif args.mode == "play":
        env = TicTacToe()
        if args.load.split(".")[-1] == "pkl":
            agent = QLearning(50_000)
        else:
            agent = DQN(env.state_size, env.num_actions, [18, 36, 18], 50_000)
        agent_vs_human_play(env, agent, args.load)
    elif args.mode == "ql":
        env = TicTacToe()
        duel_tabular_q_train(env, 100000)
    elif args.mode == "self":
        env = TicTacToe()
        self_play_dqn_train(env, 100000)


if __name__ == "__main__":
    main()
