from __future__ import annotations

from pathlib import Path

import numpy as np

from config import MODELS_DIR, get_env_config
from .envs import make_env
from .shared import compute_episode_metrics as _episode_metrics_raw, make_agent as _make_agent, validate_names as _validate_names
from .utils.seed_utils import seed_env, set_global_seed
from .utils.state_processor import get_state_dim, process_state


class RandomPolicy:
    def __init__(self, action_dim: int, seed: int | None = None) -> None:
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)

    def select_action(self, state, training: bool = False) -> int:
        del state, training
        return int(self.rng.integers(self.action_dim))


def _stats(values: list[float], prefix: str) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            f"avg_{prefix}": 0.0,
            f"std_{prefix}": 0.0,
            f"min_{prefix}": 0.0,
            f"max_{prefix}": 0.0,
            f"median_{prefix}": 0.0,
        }
    return {
        f"avg_{prefix}": float(np.mean(array)),
        f"std_{prefix}": float(np.std(array)),
        f"min_{prefix}": float(np.min(array)),
        f"max_{prefix}": float(np.max(array)),
        f"median_{prefix}": float(np.median(array)),
    }


def _resolve_model_path(
    env_name: str,
    algo_name: str,
    model_kind: str,
    model_suffix_override: str,
    model_path_override: str | Path | None,
) -> Path:
    if model_path_override is not None:
        return Path(model_path_override)
    suffix = model_suffix_override if model_suffix_override else model_kind
    return MODELS_DIR / f"{env_name}_{algo_name}_{suffix}.pth"


def evaluate_agent(
    env_name: str,
    algo_name: str,
    agent,
    episodes: int,
    render: bool = False,
    seed: int | None = None,
    return_episodes: bool = False,
) -> dict:
    if episodes < 1:
        raise ValueError("评估轮数必须大于等于 1")

    env_config = get_env_config(env_name)
    run_seed = env_config.seed if seed is None else seed
    env = make_env(env_name, render=render)
    seed_env(env, run_seed)

    rewards: list[float] = []
    steps_list: list[float] = []
    successes: list[float] = []
    metrics: list[float] = []
    reward_per_step: list[float] = []
    max_positions: list[float] = []
    scores: list[float] = []
    speeds: list[float] = []
    obstacles: list[float] = []
    episode_rows: list[dict] = []

    for episode_idx in range(episodes):
        raw_state, _ = env.reset(seed=run_seed + episode_idx)
        state = process_state(env_name, raw_state, env)
        total_reward = 0.0
        terminated = False
        truncated = False
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

        success, metric = _episode_metrics_raw(env_name, total_reward, info, terminated)
        rewards.append(float(total_reward))
        steps_list.append(float(step))
        successes.append(float(success))
        metrics.append(float(metric))
        reward_per_step.append(float(total_reward / step))

        row = {
            "episode": episode_idx + 1,
            "reward": float(total_reward),
            "steps": int(step),
            "success": int(success),
            "custom_metric": float(metric),
            "terminated": int(terminated),
            "truncated": int(truncated),
        }

        if env_name == "mountaincar":
            max_positions.append(max_position)
            row["max_position"] = max_position

        if env_name == "dino":
            score = float(info.get("score", 0.0))
            speed = float(info.get("speed", 0.0))
            cleared = float(info.get("obstacles_cleared", 0.0))
            scores.append(score)
            speeds.append(speed)
            obstacles.append(cleared)
            row.update({"score": score, "speed": speed, "obstacles_cleared": cleared})

        if return_episodes:
            episode_rows.append(row)

    env.close()

    result = {
        "env_name": env_name,
        "algo_name": algo_name,
        "episodes": episodes,
        "seed": run_seed,
        "success_count": int(np.sum(successes)),
        "success_rate": float(np.mean(successes)),
        "avg_custom_metric": float(np.mean(metrics)),
        "std_custom_metric": float(np.std(metrics)),
        "avg_reward_per_step": float(np.mean(reward_per_step)),
    }
    result.update(_stats(rewards, "reward"))
    result.update(_stats(steps_list, "steps"))

    if env_name == "mountaincar" and max_positions:
        result["avg_max_position"] = float(np.mean(max_positions))
        result["max_max_position"] = float(np.max(max_positions))

    if env_name == "dino" and scores:
        result.update(
            {
                "avg_score": float(np.mean(scores)),
                "max_score": float(np.max(scores)),
                "avg_speed": float(np.mean(speeds)),
                "avg_obstacles_cleared": float(np.mean(obstacles)),
                "max_obstacles_cleared": float(np.max(obstacles)),
            }
        )

    if return_episodes:
        result["episodes_detail"] = episode_rows
    return result



def final_test(
    env_name: str,
    algo_name: str,
    model_kind: str = "best",
    render: bool = False,
    seed: int | None = None,
    episodes: int | None = None,
    model_suffix_override: str = "",
    model_path_override: str | Path | None = None,
) -> dict:
    if model_kind not in {"best", "final"}:
        raise ValueError("model_kind 必须是 best 或 final")
    _validate_names(env_name, algo_name)
    env_config = get_env_config(env_name)
    run_seed = env_config.seed if seed is None else seed
    test_seed = run_seed + env_config.final_test_seed_offset
    set_global_seed(test_seed)

    env = make_env(env_name, render=render)
    seed_env(env, test_seed)
    state_dim = get_state_dim(env_name, env)
    action_dim = int(env.action_space.n)
    agent = _make_agent(env_name, algo_name, state_dim, action_dim)
    env.close()

    model_path = _resolve_model_path(env_name, algo_name, model_kind, model_suffix_override, model_path_override)
    if not model_path.exists():
        raise FileNotFoundError(f"未找到模型文件: {model_path}")

    agent.load(model_path)
    agent.epsilon = 0.0
    metrics = evaluate_agent(
        env_name=env_name,
        algo_name=algo_name,
        agent=agent,
        episodes=env_config.final_test_episodes if episodes is None else episodes,
        render=render,
        seed=test_seed,
    )
    metrics.update(
        {
            "phase": "final_test",
            "model_kind": model_kind,
            "model_path": str(model_path),
            "base_seed": run_seed,
            "test_seed_start": test_seed,
        }
    )
    return metrics


def evaluate_random_policy(
    env_name: str,
    episodes: int | None = None,
    render: bool = False,
    seed: int | None = None,
) -> dict:
    env_config = get_env_config(env_name)
    run_seed = env_config.seed if seed is None else seed
    test_seed = run_seed + env_config.final_test_seed_offset
    env = make_env(env_name, render=render)
    seed_env(env, test_seed)
    action_dim = int(env.action_space.n)
    env.close()
    agent = RandomPolicy(action_dim, seed=test_seed)

    metrics = evaluate_agent(
        env_name=env_name,
        algo_name="random",
        agent=agent,
        episodes=env_config.final_test_episodes if episodes is None else episodes,
        render=render,
        seed=test_seed,
    )
    metrics.update({"phase": "random_baseline", "model_path": "", "model_kind": "", "base_seed": run_seed, "test_seed_start": test_seed})
    return metrics


__all__ = ["evaluate_agent", "final_test", "evaluate_random_policy", "RandomPolicy"]
