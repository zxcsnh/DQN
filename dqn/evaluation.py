from __future__ import annotations

from pathlib import Path

import numpy as np

from config import COMMON_CONFIG, ENV_CONFIGS, MODELS_DIR
from .envs import make_env
from .training import compute_episode_metrics, make_agent, validate_names
from .utils.seed_utils import seed_env, set_global_seed
from .utils.state_processor import get_state_dim, process_state


def evaluate(env_name: str, algo_name: str, render: bool = False, use_best_model: bool = True, seed: int | None = None, model_suffix_override: str = "") -> dict:
    # 评估入口：加载训练好的模型，并统计平均性能指标。
    validate_names(env_name, algo_name)
    run_seed = COMMON_CONFIG.seed if seed is None else seed
    set_global_seed(run_seed)

    env = make_env(env_name, render=render)
    seed_env(env, run_seed)

    state_dim = get_state_dim(env_name, env)
    action_dim = int(env.action_space.n)
    agent = make_agent(env_name, algo_name, state_dim, action_dim)

    base_suffix = model_suffix_override if model_suffix_override else ("best" if use_best_model else "final")
    model_path = MODELS_DIR / f"{env_name}_{algo_name}_{base_suffix}.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"未找到模型文件: {model_path}")

    agent.load(model_path)
    agent.epsilon = 0.0

    rewards = []
    steps_list = []
    successes = []
    metrics = []
    env_config = ENV_CONFIGS[env_name]

    for _ in range(COMMON_CONFIG.test_episodes):
        raw_state, _ = env.reset()
        state = process_state(env_name, raw_state, env)
        total_reward = 0.0
        terminated = False
        info = {}
        max_position = -1.2

        for step in range(1, env_config["max_steps"] + 1):
            action = agent.select_action(state, training=False)
            next_raw_state, reward, terminated, truncated, info = env.step(action)
            state = process_state(env_name, next_raw_state, env)
            total_reward += reward

            if env_name == "mountaincar":
                max_position = max(max_position, float(next_raw_state[0]))
                info["max_position"] = max_position

            if terminated or truncated:
                break

        success, metric = compute_episode_metrics(env_name, total_reward, info, terminated)
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
        "model_path": str(model_path),
    }
