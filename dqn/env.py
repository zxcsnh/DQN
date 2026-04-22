"""Environment construction helpers for low-dimensional Gym tasks."""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DiscreteObservationWrapper(gym.ObservationWrapper):
    """Convert discrete state ids into vectors usable by MLP policies."""

    def __init__(self, env: gym.Env, encoding: str = "one_hot"):
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Discrete):
            raise TypeError("DiscreteObservationWrapper requires a Discrete observation space.")
        if encoding != "one_hot":
            raise ValueError(f"Unsupported discrete observation encoding: {encoding}")

        self.encoding = encoding
        self.num_states = int(env.observation_space.n)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_states,),
            dtype=np.float32,
        )

    def observation(self, observation: int) -> np.ndarray:
        encoded = np.zeros(self.num_states, dtype=np.float32)
        encoded[int(observation)] = 1.0
        return encoded


class ObservationToFloat32Wrapper(gym.ObservationWrapper):
    """Normalize all vector observations to float32 arrays."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError("ObservationToFloat32Wrapper requires a Box observation space.")

        shape = tuple(int(v) for v in env.observation_space.shape)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=shape,
            dtype=np.float32,
        )

    def observation(self, observation: Any) -> np.ndarray:
        return np.asarray(observation, dtype=np.float32)


def infer_env_family(env: gym.Env) -> str:
    observation_space = env.observation_space
    if isinstance(observation_space, spaces.Discrete):
        return "discrete"
    if isinstance(observation_space, spaces.Box):
        return "vector"
    raise TypeError(f"Unsupported observation space: {observation_space}")


def make_env(
    env_name: str,
    env_family: str = "auto",
    obs_encoding: str = "auto",
    render_mode: Optional[str] = None,
    seed: Optional[int] = None,
    max_episode_steps_override: int | None = None,
) -> gym.Env:
    """Create a low-dimensional Gymnasium environment with vector observations."""
    env = gym.make(env_name, render_mode=render_mode)

    inferred_family = infer_env_family(env) if env_family == "auto" else env_family
    if inferred_family == "discrete":
        encoding = "one_hot" if obs_encoding == "auto" else obs_encoding
        env = DiscreteObservationWrapper(env, encoding=encoding)
    elif inferred_family in {"vector", "custom"}:
        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError(
                f"env_family={inferred_family} requires a Box observation space, got {env.observation_space}"
            )
        env = ObservationToFloat32Wrapper(env)
    else:
        raise ValueError(f"Unsupported env_family: {inferred_family}")

    if max_episode_steps_override is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps_override)

    if seed is not None:
        env.action_space.seed(seed)

    return env


if __name__ == "__main__":
    for env_name in ("CartPole-v1", "Taxi-v3", "MountainCar-v0"):
        env = make_env(env_name)
        print(f"Environment: {env_name}")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        obs, _ = env.reset(seed=42)
        print(f"Initial observation shape: {np.asarray(obs).shape}")
        env.close()
        print("-" * 40)

    print("Environment smoke test finished")
