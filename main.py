from agent.agent import RandomAgent
from algo.dqn import DQN
from env.tic_tac_toe import TicTacToe
from marl import agent_vs_random

if __name__ == "__main__":
    dqn_agent = DQN(TicTacToe(), [32, 64, 32])
    random_agent = RandomAgent()
    agent_vs_random(dqn_agent, random_agent, True)
