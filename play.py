"""Playback helpers for trained low-dimensional DQN agents."""

from __future__ import annotations

import os
import sys
import time

import numpy as np
from gymnasium import spaces

from config import DQNConfig
from dqn.agent import DQNAgent
from dqn.env import make_env


def infer_model_metadata(model_path: str | None, default_hidden_sizes: tuple[int, ...]) -> dict[str, object]:
    """Infer vector-model configuration from a saved artifact when available."""
    metadata = {
        "network_type": "mlp",
        "obs_dim": None,
        "num_actions": None,
        "hidden_sizes": list(default_hidden_sizes),
    }
    if not model_path or not os.path.exists(model_path):
        return metadata

    try:
        import torch

        artifact = torch.load(model_path, map_location="cpu", weights_only=False)
        model_config = artifact.get("model_config", {})
        metadata["network_type"] = model_config.get("network_type", "mlp")
        metadata["obs_dim"] = model_config.get("obs_dim")
        metadata["num_actions"] = model_config.get("num_actions")
        metadata["hidden_sizes"] = model_config.get("hidden_sizes", list(default_hidden_sizes))
    except Exception as exc:
        print(f"Warning: failed to infer model metadata from saved model: {exc}")

    return metadata


def play(
    env_name: str = "CartPole-v1",
    env_family: str = "auto",
    obs_encoding: str = "auto",
    model_path: str | None = None,
    num_episodes: int = 5,
    render: bool = True,
    delay: float = 0.02,
    device: str = "auto",
    hidden_sizes: tuple[int, ...] = (128, 128),
):
    """运行若干回合，用于观察训练后策略的表现。"""
    if model_path is not None and not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    render_mode = "human" if render else None
    env = make_env(
        env_name,
        env_family=env_family,
        obs_encoding=obs_encoding,
        render_mode=render_mode,
    )

    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    metadata = infer_model_metadata(model_path, hidden_sizes)
    observation_space = env.observation_space
    if not isinstance(observation_space, spaces.Box):
        raise TypeError(f"Expected wrapped Box observation space, got {observation_space}")

    obs_dim = int(np.prod(observation_space.shape))
    num_actions = env.action_space.n
    print(f"Environment: {env_name}")
    print(f"Num actions: {num_actions}")
    print(f"Observation dim: {obs_dim}")

    agent = DQNAgent(
        num_actions=num_actions,
        obs_dim=obs_dim,
        hidden_sizes=tuple(metadata["hidden_sizes"]),
        device=device,
    )

    if model_path:
        agent.load(model_path)
        print(f"Loaded model: {model_path}")
    else:
        print("No model provided, using the untrained policy network.")

    agent.eval_mode()
    total_rewards = []

    print("\nStarting playback...")
    print("-" * 50)

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        episode_reward = 0.0
        episode_length = 0
        done = False

        while not done:
            if render:
                time.sleep(delay)

            action = agent.select_action(state, evaluate=True)
            state, reward, terminated, truncated, _ = env.step(action)
            state = np.asarray(state, dtype=np.float32).reshape(-1)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

        total_rewards.append(episode_reward)
        print(f"Episode {episode}: reward={episode_reward:.1f}, length={episode_length}")

    env.close()

    print("-" * 50)
    print(f"Average reward: {np.mean(total_rewards):.1f}")
    print(f"Std reward: {np.std(total_rewards):.1f}")
    print(f"Min reward: {np.min(total_rewards):.1f}")
    print(f"Max reward: {np.max(total_rewards):.1f}")


def resolve_default_model_path(config: DQNConfig) -> str | None:
    default_model_path = os.path.join(
        config.logging.save_dir,
        f"{config.model_name_prefix()}_ep{config.logging.save_freq}.pth",
    )
    if os.path.exists(default_model_path):
        print(f"Using saved model: {default_model_path}")
        return default_model_path

    print(
        "Warning: default saved model not found. Playback will use the untrained policy network.",
        file=sys.stderr,
    )
    return None


def main():
    config = DQNConfig()
    model_path = resolve_default_model_path(config)
    play(
        env_name=config.core.env_name,
        env_family=config.core.env_family,
        obs_encoding=config.replay.obs_encoding,
        model_path=model_path,
        num_episodes=config.evaluation.eval_episodes,
        render=config.playback.render,
        hidden_sizes=config.training.hidden_sizes,
    )


if __name__ == "__main__":
    main()
