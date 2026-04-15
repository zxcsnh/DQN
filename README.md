# DQN - Deep Q-Network for Atari Games

A PyTorch implementation of Deep Q-Network (DQN) for Atari games, with replay buffer, target network, Atari preprocessing, evaluation, and gameplay recording.

## Features

- **Core DQN Algorithm**: Experience replay, target network, epsilon-greedy exploration
- **Atari Preprocessing**: Frame stacking, reward clipping, frame skipping
- **Training Visualization**: Offline metrics export and matplotlib plotting
- **Model Management**: Save/load trained models
- **Gameplay Demo**: Play games with trained agents

## Installation

```bash
uv sync
```

or

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## Quick Start

### Train a DQN Agent

```bash
# Edit hyperparameters in config.py, then start training
python train.py
```

Typical fields to tune in [config.py](/e:/Programming_Language/MachineLearning/RL/DQN/config.py:8):
- `env_name`
- `num_episodes`
- `max_steps`
- `learning_rate`
- `batch_size`
- `use_per`
- `soft_update`
- `tau`

### Play with Trained Agent

```bash
# Edit inference options in config.py, then run evaluation/playback
python play.py
```

Playback-related fields in [config.py](/e:/Programming_Language/MachineLearning/RL/DQN/config.py:8):
- `render`
- `save_video`
- `video_dir`
- `eval_episodes`

### Training Outputs

After training finishes, offline artifacts are written to `runs/`:
- `config.json`
- `metrics.json`
- `training_curves.png`

## Project Structure

```
DQN/
├── dqn/
│   ├── __init__.py          # Package initialization
│   ├── agent.py             # DQN agent implementation
│   ├── env.py               # Atari environment preprocessing
│   ├── network.py           # CNN network architecture
│   ├── replay_buffer.py     # Replay buffer and PER
│   └── utils.py             # Training utilities and logging
├── config.py                 # Hyperparameters configuration
├── train.py                  # Training script
├── play.py                   # Gameplay demonstration
├── pyproject.toml            # Project dependencies
└── README.md                 # Documentation
```

## Hyperparameters

Default hyperparameters (tuned for Pong):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 2.5e-4 | Adam optimizer learning rate |
| `gamma` | 0.99 | Discount factor |
| `batch_size` | 32 | Training batch size |
| `buffer_size` | 100,000 | Replay buffer capacity |
| `epsilon_start` | 1.0 | Initial exploration rate |
| `epsilon_end` | 0.01 | Final exploration rate |
| `epsilon_decay` | 1,000,000 | Steps to decay epsilon |
| `target_update_freq` | 10,000 | Steps between target network updates |
| `frame_stack` | 4 | Number of frames to stack |
| `frame_skip` | 4 | Frame skip count |
| `eval_episodes` | 10 | Evaluation episodes per eval run |
| `use_per` | False | Enable prioritized replay |

## Network Architecture

```
Input: (batch, 4, 84, 84) - 4 stacked grayscale frames

Conv2d(4, 32, 8, stride=4) -> ReLU -> (batch, 32, 20, 20)
Conv2d(32, 64, 4, stride=2) -> ReLU -> (batch, 64, 9, 9)
Conv2d(64, 64, 3, stride=1) -> ReLU -> (batch, 64, 7, 7)

Flatten -> (batch, 3136)
Linear(3136, 512) -> ReLU
Linear(512, num_actions)

Output: Q-values for each action
```

## Training Tips

1. Warm-up: the agent waits for `min_buffer_size` transitions before learning starts.
2. Reproducibility: training saves the resolved config to `runs/config.json`.
3. Offline analysis: training also saves `runs/metrics.json` and `runs/training_curves.png`.
4. Evaluation: eval episodes use unclipped rewards and deterministic action selection.
5. GPU: training is much faster with CUDA-enabled GPU.

## Supported Games

Tested configuration target:
- `ALE/Pong-v5`

Should work on most Atari games from the ALE suite.

## Algorithm Details

### Experience Replay
- Stores transitions (state, action, reward, next_state, done)
- Random sampling breaks correlation between consecutive samples
- Improves sample efficiency

### Target Network
- Separate network for computing target Q-values
- Supports hard update or Polyak soft update
- Stabilizes training

### Epsilon-Greedy Exploration
- Linear decay from 1.0 to 0.01 over 1M steps
- Balances exploration and exploitation

### Atari Preprocessing
- **Frame Skip**: Repeat action for 4 frames
- **Max Pooling**: Max over the last two raw frames during frame skip
- **Frame Stack**: Stack 4 consecutive frames
- **Reward Clipping**: Clip rewards to [-1, 1]
- **No-op Starts**: Random no-op actions at episode start

## License

MIT License

## References

- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) - Mnih et al., 2015
- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) - Mnih et al., 2013
