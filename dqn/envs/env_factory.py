from __future__ import annotations

import gymnasium as gym

from config import ENV_CONFIGS
from .dino_env import DinoEnv


SUPPORTED_ENVS = tuple(ENV_CONFIGS.keys())


def make_env(env_name: str, render: bool = False):
    if env_name not in ENV_CONFIGS:
        raise ValueError(f"不支持的环境名称: {env_name}，可选值为 {SUPPORTED_ENVS}")

    render_mode = "human" if render else None
    env_config = ENV_CONFIGS[env_name]

    if env_name in {"taxi", "mountaincar"}:
        return gym.make(env_config["env_id"], render_mode=render_mode)

    if env_name == "dino":
        return DinoEnv(render_mode=render_mode, max_steps=env_config["max_steps"])

    raise ValueError(f"不支持的环境名称: {env_name}")
