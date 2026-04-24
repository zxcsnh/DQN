from __future__ import annotations

import gymnasium as gym

from config import get_env_config, supported_envs
from .dino.env import register_trex_envs


SUPPORTED_ENVS = supported_envs()


def make_env(env_name: str, render: bool = False):
    if env_name not in SUPPORTED_ENVS:
        raise ValueError(f"不支持的环境名称: {env_name}，可选值为 {SUPPORTED_ENVS}")

    render_mode = "human" if render else None
    env_config = get_env_config(env_name)

    if env_name in {"taxi", "mountaincar"}:
        return gym.make(env_config.env_id, render_mode=render_mode)

    if env_name == "dino":
        register_trex_envs()
        return gym.make(env_config.env_id, render_mode=render_mode, max_steps=env_config.max_steps_per_episode)

    raise ValueError(f"不支持的环境名称: {env_name}")
