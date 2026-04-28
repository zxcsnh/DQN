from __future__ import annotations

from config import PER_CONFIG, get_env_config, supported_envs
from .agents import DQNAgent, PERDQNAgent


def validate_names(env_name: str, algo_name: str) -> None:
    if env_name not in supported_envs():
        raise ValueError(f"不支持的环境名称: {env_name}，可选值为 {supported_envs()}")
    if algo_name not in {"dqn", "perdqn"}:
        raise ValueError(f"不支持的算法名称: {algo_name}")


def make_agent(env_name: str, algo_name: str, state_dim: int, action_dim: int):
    env_config = get_env_config(env_name)
    if algo_name == "dqn":
        return DQNAgent(state_dim, action_dim, env_config, env_name, algo_name="dqn")
    return PERDQNAgent(state_dim, action_dim, env_config, PER_CONFIG, env_name)


def compute_episode_metrics(env_name: str, episode_reward: float, info: dict, terminated: bool) -> tuple[int, float]:
    if env_name == "taxi":
        success = 1 if episode_reward > 0 and terminated else 0
        return success, float(success)

    if env_name == "mountaincar":
        max_position = float(info.get("max_position", -1.2))
        success = 1 if max_position >= 0.5 else 0
        return success, max_position

    obstacles_cleared = float(info.get("obstacles_cleared", 0))
    success = 1 if obstacles_cleared >= get_env_config(env_name).success_threshold else 0
    return success, obstacles_cleared
