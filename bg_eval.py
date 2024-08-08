import random
from pathlib import Path

from open_spiel.python import rl_environment
from tqdm import tqdm

from agent.agent import HumanAgent, RandomAgent
from algo.algo import Algorithm
from algo.dqn import DQN
from bg import BGPlayer, time_step_to_observation


def test_against_random(player_a, player_b, env, n_episodes=1000):
    agents = [player_a, player_b]
    results = {"win": 0, "lose": 0, "n_episodes": n_episodes}
    player_agent_dict = dict()

    for a in agents:
        player_agent_dict[a.player] = a

    for _ in tqdm(range(n_episodes)):
        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            obs = time_step_to_observation(time_step, player_id)
            acting_agent = player_agent_dict[BGPlayer(player_id)]
            if isinstance(acting_agent, Algorithm):
                action = acting_agent.step(obs, eval=True)
            else:
                action = random.choice(obs.valid_actions)
            action_list = [action]
            time_step = env.step(action_list)

        if time_step.rewards[0] == 1:
            results["win"] += 1
        elif time_step.rewards[0] == -1:
            results["lose"] += 1
        else:
            results["draw"] += 1

    print(f"{agents[0]}_{agents[0].player}={results['win']}")
    print(f"{agents[1]}_{agents[1].player}={results['lose']}")


def main():
    game = "backgammon"
    env = rl_environment.Environment(game)
    state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]
    hidden_layers = [256, 512, 768]
    n_episodes = 1000
    player_a = DQN(state_size, num_actions, hidden_layers, n_episodes, BGPlayer(0))
    player_a.load(Path("models/DQN_BGPlayer.BG0.pt"), eval=True)
    player_b = DQN(state_size, num_actions, hidden_layers, n_episodes, BGPlayer(1))
    player_b.load(Path("models/DQN_BGPlayer.BG1.pt"), eval=True)
    test_against_random(
        RandomAgent(player=BGPlayer(0)),
        RandomAgent(player=BGPlayer(1)),
        env,
        n_episodes,
    )
    test_against_random(player_a, RandomAgent(player=BGPlayer(1)), env, n_episodes)
    test_against_random(RandomAgent(player=BGPlayer(0)), player_b, env, n_episodes)


if __name__ == "__main__":
    main()
