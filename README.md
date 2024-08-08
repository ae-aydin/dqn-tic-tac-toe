# Q-Learning and DQN for Tic-Tac-Toe Environment

Implementation of Q-Learning and Deep Q-Network (DQN) for training agents in a Tic-Tac-Toe environment. Supports self-play and individual training.

### Train via `train.py`

```
Usage: python3 train.py {dqn, qlearning} [--n-episodes N_EPISODES] [--n-decay-steps N_DECAY_STEPS] [--self-play]

Train RL agents.

Positional arguments:
  {dqn, qlearning}        Select the RL agent to use.

Optional arguments:
  --n-episodes N_EPISODES               Number of training episodes (default: 50000)
  --n-decay-steps N_DECAY_STEPS         Total epsilon decay steps (default: 50000)
  --self-play                           Enable self-play mode for training (default: False)
```

#### Examples

* Train DQN agents with default settings (self-play disabled, two individual agents will be trained):
```
python3 train.py dqn
```
* Train a Q-learning agent with self-play:
```
python3 train.py qlearning --self-play
```
* Train a DQN agent with 100,000 episodes and using self-play:
```
python3 train.py dqn --n-episodes 100000 --self-play
```

### Evaluate via `eval.py`

```
Usage: python3 eval.py player_a player_b [--env ENV] [--plot]

Evaluate RL models.

Positional arguments:
  player_a          The first player, X for Tic-Tac-Toe (agent1 (filepath), random, human)
  player_b          The second player, O for Tic-Tac-Toe (agent2 (filepath), random, human)

Optional arguments:
  --plot            Enable plotting performance across different epsilon values (default: False)
```

#### Examples

* Evaluate an agent vs random:
```
python3 eval.py models/DQN_Player.X.pt random
python3 eval.py models/DQN_SelfPlay.pt random
python3 eval.py random models/Q-Learning_Player.O.pkl
```
* Play against an agent:
```
python3 eval.py models/Q-Learning_Player.X.pkl human
python3 eval.py models/Q-Learning_SelfPlay.pkl human
python3 eval.py human models/DQN_Player.O.pt
```
* Evaluate two agent against each other:
```
python3 eval.py models/Q-Learning_Player.X.pkl models/DQN_Player.O.pt
python3 eval.py models/Q-Learning_SelfPlay.pkl models/DQN_SelfPlay.pt
python3 eval.py models/DQN_SelfPlay.pt models/DQN_Player.O.pt
```
* Plot performance graph across different epsilon values:
```
python3 eval.py models/DQN_SelfPlay.pt models/Q-Learning_Player.O.pkl --plot
```
Important Note: Filename of the agent should obey this rule: `{algo_name}_{player_id}.pkl/pt`

### Experimental: DQN for Backgammon

Train DQN agents using OpenSpiel Backgammon environment.

```
python3 bg.py
python3 bg_eval.py
```