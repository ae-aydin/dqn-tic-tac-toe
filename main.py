from dqn import DQNAgent, DQNAgentSelfPlay
from tic_tac_toe import TicTacToe
import torch

if __name__ == "__main__":
    dqn_agent = DQNAgentSelfPlay(TicTacToe(), [32, 64, 32])
    dqn_agent.self_play()
