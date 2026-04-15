"""Environment helpers for Atari DQN."""

from typing import Optional

import ale_py
import gymnasium as gym
import numpy as np


def make_env(
    env_name: str,
    frame_size: tuple[int, int] = (84, 84),
    frame_skip: int = 4,
    frame_stack: int = 4,
    noop_max: int = 30,
    clip_reward: bool = True,
    render_mode: Optional[str] = None,
    seed: Optional[int] = None,
    terminal_on_life_loss: bool = False,
) -> gym.Env:
    """Create an Atari environment with standard preprocessing."""
    if frame_size[0] != frame_size[1]:
        raise ValueError(
            "Gymnasium AtariPreprocessing only supports square screen_size values."
        )

    gym.register_envs(ale_py)

    # Disable ALE's internal frame skip so preprocessing owns the behavior.
    env = gym.make(env_name, render_mode=render_mode, frameskip=1)

    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=noop_max,
        frame_skip=frame_skip,
        screen_size=frame_size[0],
        grayscale_obs=True,
        scale_obs=False,
        terminal_on_life_loss=terminal_on_life_loss,
    )
    env = gym.wrappers.FrameStackObservation(env, frame_stack)

    if clip_reward:
        env = gym.wrappers.TransformReward(
            env, lambda reward: float(np.clip(reward, -1.0, 1.0))
        )

    if seed is not None:
        env.action_space.seed(seed)

    return env


if __name__ == "__main__":
    env = make_env("ALE/Pong-v5")

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    obs, info = env.reset(seed=42)
    print(f"Initial observation shape: {obs.shape}")

    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i + 1}: reward={reward:.2f}, shape={obs.shape}")

        if terminated or truncated:
            obs, info = env.reset(seed=42 + i + 1)

    env.close()
    print("\nEnvironment smoke test finished")
