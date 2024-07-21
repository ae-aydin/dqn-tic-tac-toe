import random

from tqdm import tqdm

from agent.agent import RandomAgent
from algo.dqn import DQN, Transition
from env.tic_tac_toe import Player


def agent_vs_random(algo_agent: DQN, random_agent: RandomAgent, soft_update: bool):
    state, starting_player = algo_agent.env.reset()
    agent_player = assign(algo_agent, random_agent)
    player_agent = {player: agent for agent, player in agent_player.items()}
    algo_elo, opp_elo = 1200, 1200

    pbar = tqdm(
        range(algo_agent.n_steps),
        ncols=150,
        desc=f"Steps: [agent1_elo={algo_elo:.2f}, agent2_elo={opp_elo:.2f}]",
    )
    for step in pbar:
        if algo_agent.env.current_player == agent_player[algo_agent]:
            action = algo_agent._act_e_greedy(state, step)
            new_state, reward, term, trunc = algo_agent.env.step(
                int(action), agent_player[algo_agent]
            )
            transition = Transition(state, action, new_state, reward)
            done = term or trunc
            algo_agent.memory.push(transition)
            state = new_state
            algo_agent._optimize()
        else:
            action = random_agent.select_action(algo_agent.env)
            new_state, reward, term, trunc = algo_agent.env.step(
                int(action), agent_player[random_agent]
            )
            transition = Transition(state, action, new_state, reward)
            done = term or trunc
            algo_agent.memory.push(transition)
            state = new_state

        if soft_update:
            algo_agent._soft_update_target()
        else:
            if ((step + 1) % algo_agent.target_update_freq) == 0:
                algo_agent.target_net.load_state_dict(algo_agent.q_net.state_dict())

        if done:
            if reward != 0:
                winner = player_agent[Player(reward)]
                algo_elo, opp_elo = update_elo(algo_elo, opp_elo, winner, algo_agent)
            pbar.set_description(
                f"Steps: [agent1_elo={algo_elo:.2f}, agent2_elo={opp_elo:.2f}]"
            )
            state, _ = algo_agent.env.reset()


def self_play():
    pass


def agent_vs_human():
    pass


def human_vs_random():
    pass


def assign(agent1, agent2):
    player_agent = {agent1: random.choice((Player.X, Player.O)), agent2: None}
    player_agent[agent2] = Player.O if player_agent[agent1] == Player.X else Player.O
    return player_agent


def update_elo(a1_elo, a2_elo, winner, a1):
    e_a1 = (10 ** ((a2_elo - a1_elo) / 400) + 1) ** -1
    e_a2 = (10 ** ((a1_elo - a2_elo) / 400) + 1) ** -1
    if winner == a1:
        a1_elo = a1_elo + 16 * (1 - e_a1)
        a2_elo = a2_elo + 16 * (0 - e_a2)
    else:
        a1_elo = a1_elo + 16 * (0 - e_a1)
        a2_elo = a2_elo + 16 * (1 - e_a2)
    return a1_elo, a2_elo
