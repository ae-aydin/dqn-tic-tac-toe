import random

from tqdm import tqdm

from agent.agent import RandomAgent
from algo.dqn import DQN, Transition
from envs.env import Environment
from envs.tic_tac_toe import PLAYERS, Player, TicTacToe
import logging
import numpy as np

def agent_vs_random(env: Environment, soft_update: bool):
    dqn = DQN(env, [16, 32, 16])
    random_agent = RandomAgent()
    n_steps = 0
    win, lose, draw = 0, 0, 0
    for ep in range(dqn.n_episodes):
        state = dqn.env.reset()
        done = False
        a2p, p2a = dqn.get_agent_map((dqn, random_agent), PLAYERS)
        if dqn.env.current_player == a2p[dqn]:
            action = dqn._act_e_greedy(state, ep)
        elif dqn.env.current_player == a2p[random_agent]:
            action = random_agent.select_action(dqn.env)
        new_state, reward, term, trunc = dqn.env.step(action, dqn.env.current_player)
        if reward != 0:
            reward = 1 if reward == a2p[dqn].value else -1
        transition = Transition(state, action, new_state, reward)
        dqn.memory.push(transition)
        done = term or trunc

        if not done:
            dqn._optimize()
            state = new_state
            if soft_update:
                dqn._soft_update_target()
            else:
                if ((ep + 1) % dqn.target_update_freq) == 0:
                    dqn.target_net.load_state_dict(dqn.q_net.state_dict())
        else:
            print("MATCH ENDED:", reward)
            print("table:", dqn.env.board, sep="\n")
            if reward == 1:
                print("WIN")
                win += 1
            elif reward == -1:
                print("LOSE")
                lose += 1
            else:
                print("DRAW")
                draw += 1
            print(f"performance={win}/{draw}/{lose}, episodes={ep}")
            print()
    dqn.save("random.pt")


def agent_vs_human(agent, fname):
    agent.load(fname, True)
    state = agent.env.reset()
    print("Welcome to Tic-Tac-Toe!")
    print("You are 'O', the AI is 'X'.")
    print(
        "Enter your move as a number from 0-8, where 0 is the top-left corner and 8 is the bottom-right corner."
    )

    while True:
        # Human's turn
        while True:
            try:
                human_action = int(input("\nYour turn. Enter your move (0-8): "))
                if human_action not in agent.env.valid_actions:
                    raise ValueError
                break
            except ValueError:
                print("Invalid move. Please try again.")

        state, reward, done, _ = agent.env.step(human_action, agent.env.current_player)
        print(agent.env._out())

        if done:
            if reward == -1:
                print("You win!")
            elif reward == 0:
                print("It's a draw!")
            break

        # AI's turn
        print("\nAI's turn:")
        action = agent._act_greedy(state)
        state, reward, done, _ = agent.env.step(action, agent.env.current_player)
        print(agent.env._out())

        if done:
            if reward == 1:
                print("AI wins!")
            elif reward == 0:
                print("It's a draw!")
            break

    print("Game Over!")




def human_vs_random():
    pass

def train(agents: list, env: Environment, n_episodes: int):
    player_agent = dict(zip(env.players, agents))
    for p, a in player_agent.items():
        a.player = p
    n_steps, win, lose, draw = 0, 0, 0, 0
    pbar = tqdm(range(n_episodes), desc=f"steps={n_steps+1} | {win}/{draw}/{lose}| episodes", ncols=150)
    for ep in pbar:
        state = env.reset()
        done = False
        while not done:
            n_steps += 1
            acting_agent = player_agent[env.current_player]
            action = acting_agent.act_e_greedy(state, ep, env)
            new_state, reward, term, trunc = env.step(
                    action, acting_agent.player
                )
            done = term or trunc
            acting_agent.update(state, action, reward[acting_agent.player], new_state, done)
            if not done:
                state = new_state
            else:
                if reward[agents[0].player] == 1:
                    logging.debug("WIN")
                    win += 1
                elif reward[agents[0].player] == -1:
                    logging.debug("LOSE")
                    lose += 1
                else:
                    logging.debug("DRAW")
                    draw += 1
                logging.debug(f"performance={win}/{draw}/{lose}, episodes={ep}")
                pbar.set_description(f"steps={n_steps+1} | {win}/{draw}/{lose} | episodes")
    #for p, a in player_agent.items():
        #np.save(f"{str(a)}_{str(p)}.npy", a.q, allow_pickle=True)
    print(player_agent[agents[0].player].q[tuple(np.zeros(env.action_space))])
    print(len(player_agent[agents[0].player].q))
    non_empty_elements = {state: q_values for state, q_values in player_agent[agents[0].player].q.items() if np.any(q_values != 0)}
    print(len(non_empty_elements))

    # Print the total count of non-empty states
    total_non_empty_states = len(non_empty_elements)
    print(f"Total number of non-empty states: {total_non_empty_states}")
