from __future__ import annotations

from pathlib import Path

import numpy as np

from config import MODELS_DIR, get_env_config
from .envs import make_env
from .utils.seed_utils import seed_env, set_global_seed
from .utils.state_processor import get_state_dim, process_state


def _training_helpers():
    from .training import compute_episode_metrics, make_agent, validate_names

    return compute_episode_metrics, make_agent, validate_names


def _episode_metrics(env_name: str, total_reward: float, info: dict, terminated: bool):
    compute_episode_metrics, _, _ = _training_helpers()
    return compute_episode_metrics(env_name, total_reward, info, terminated)


def _make_agent(env_name: str, algo_name: str, state_dim: int, action_dim: int):
    _, make_agent, _ = _training_helpers()
    return make_agent(env_name, algo_name, state_dim, action_dim)


def _validate_names(env_name: str, algo_name: str) -> None:
    _, _, validate_names = _training_helpers()
    validate_names(env_name, algo_name)


def evaluate_agent(
    env_name: str,
    algo_name: str,
    agent,
    episodes: int,
    render: bool = False,
    seed: int | None = None,
) -> dict:
    if episodes < 1:
        raise ValueError("评估轮数必须大于等于 1")

    env_config = get_env_config(env_name)
    run_seed = env_config.seed if seed is None else seed
    env = make_env(env_name, render=render)
    seed_env(env, run_seed)

    rewards = []
    steps_list = []
    successes = []
    metrics = []

    for episode_idx in range(episodes):
        raw_state, _ = env.reset(seed=run_seed + episode_idx)
        state = process_state(env_name, raw_state, env)
        total_reward = 0.0
        terminated = False
        info = {}
        max_position = -1.2

        for step in range(1, env_config.max_steps_per_episode + 1):
            action = agent.select_action(state, training=False)
            next_raw_state, reward, terminated, truncated, info = env.step(action)
            state = process_state(env_name, next_raw_state, env)
            total_reward += reward

            if env_name == "mountaincar":
                max_position = max(max_position, float(next_raw_state[0]))
                info["max_position"] = max_position

            if terminated or truncated:
                break

        success, metric = _episode_metrics(env_name, total_reward, info, terminated)
        rewards.append(total_reward)
        steps_list.append(step)
        successes.append(success)
        metrics.append(metric)

    env.close()
    return {
        "env_name": env_name,
        "algo_name": algo_name,
        "avg_reward": float(np.mean(rewards)),
        "avg_steps": float(np.mean(steps_list)),
        "success_rate": float(np.mean(successes)),
        "avg_custom_metric": float(np.mean(metrics)),
    }


def evaluate(
    env_name: str,
    algo_name: str,
    render: bool = False,
    use_best_model: bool = True,
    seed: int | None = None,
    model_suffix_override: str = "",
    model_path_override: str | Path | None = None,
) -> dict:
    _validate_names(env_name, algo_name)
    env_config = get_env_config(env_name)
    run_seed = env_config.seed if seed is None else seed
    set_global_seed(run_seed)

    env = make_env(env_name, render=render)
    seed_env(env, run_seed)

    state_dim = get_state_dim(env_name, env)
    action_dim = int(env.action_space.n)
    agent = _make_agent(env_name, algo_name, state_dim, action_dim)
    env.close()

    if model_path_override is not None:
        model_path = Path(model_path_override)
    else:
        base_suffix = model_suffix_override if model_suffix_override else ("best" if use_best_model else "final")
        model_path = MODELS_DIR / f"{env_name}_{algo_name}_{base_suffix}.pth"

    if not model_path.exists():
        raise FileNotFoundError(f"未找到模型文件: {model_path}")

    agent.load(model_path)
    agent.epsilon = 0.0

    metrics = evaluate_agent(
        env_name=env_name,
        algo_name=algo_name,
        agent=agent,
        episodes=env_config.test_episodes,
        render=render,
        seed=run_seed,
    )
    metrics["model_path"] = str(model_path)
    return metrics


__all__ = ["evaluate", "evaluate_agent"]
