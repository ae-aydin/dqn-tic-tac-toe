from enum import Enum

from open_spiel.python import rl_environment
from tqdm import tqdm

from algo.dqn import DQN
from envs.env import Observation

# Experimental Backgammon Training Using OpenSpiel Backgammon Environment


class BGPlayer(Enum):
    BG0 = 0
    BG1 = 1
    BGNone = -1


def time_step_to_observation(time_step: rl_environment.TimeStep, player_id: int):
    """Convert OpenSpiel TimeStep type to Observation type.

    Args:
        time_step (rl_environment.TimeStep): OpenSpiel TypeStep class.
        player_id (int): Player ID.

    Returns:
        Observation: TimeStep converted to Observation type.
    """
    state = tuple(time_step.observations["info_state"][player_id])
    valid_actions = time_step.observations["legal_actions"][player_id]
    reward = dict()
    if time_step.rewards:
        reward = {BGPlayer(0): time_step.rewards[0], BGPlayer(1): time_step.rewards[1]}
    else:
        reward = {BGPlayer(0): 0, BGPlayer(1): 0}
    terminate = True if time_step.step_type == rl_environment.StepType.LAST else False
    return Observation(state, valid_actions, reward, terminate)


def main():
    game = "backgammon"
    num_players = 2
    num_train_episodes = 50_000
    approx_steps = num_train_episodes * 50

    env = rl_environment.Environment(game)
    state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]
    hidden_layers = [256, 512, 768]

    agents = list()
    player_agent_dict = dict()
    for idx in range(num_players):
        agent = DQN(
            state_size,
            num_actions,
            hidden_layers,
            approx_steps,
            player=BGPlayer(idx),
            memory_size=40_000,
            batch_size=384,
            target_update_every=25_000,
            optimize_every=64,
        )
        agents.append(agent)
        player_agent_dict[agent.player] = agent

    n_steps, win, lose = 0, 0, 0  # first player perspective
    episodes = tqdm(
        range(num_train_episodes),
        desc=f"steps={n_steps+1} | {win}/{lose} | episodes",
        ncols=125,
    )

    for ep in episodes:
        time_step = env.reset()
        while not time_step.last():
            n_steps += 1
            player_id = time_step.observations["current_player"]
            if env.is_turn_based:
                obs = time_step_to_observation(time_step, player_id)
                action = player_agent_dict[BGPlayer(player_id)].step(obs)
                action_list = [action]
            time_step = env.step(action_list)

        for agent in agents:
            obs = time_step_to_observation(time_step, agent.player.value)
            agent.step(obs)

        if obs.reward[BGPlayer(0)] == 1:
            win += 1
        elif obs.reward[BGPlayer(0)] == -1:
            lose += 1
        else:
            pass
        episodes.set_description(f"steps={n_steps+1} | {win}/{lose} | episodes")

    for p, a in player_agent_dict.items():
        a.save(str(p))


if __name__ == "__main__":
    main()
