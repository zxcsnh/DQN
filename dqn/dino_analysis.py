from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from config import MODELS_DIR, get_env_config
from .envs import make_env
from .shared import make_agent
from .utils.seed_utils import seed_env, set_global_seed
from .utils.state_processor import get_state_dim, process_state


ACTION_NAMES = {0: "noop", 1: "jump", 2: "duck"}
DISTANCE_BINS = [(0.0, 50.0, "0_50"), (50.0, 100.0, "50_100"), (100.0, 200.0, "100_200"), (200.0, float("inf"), "200_plus")]
SPEED_BINS = [(0.0, 5.0, "low"), (5.0, 8.0, "medium"), (8.0, float("inf"), "high")]


def _bin_name(value: float, bins: list[tuple[float, float, str]]) -> str:
    for low, high, name in bins:
        if low <= value < high:
            return name
    return bins[-1][2]


def _dino_features(raw_state) -> dict:
    state = np.asarray(raw_state, dtype=np.float32)
    obstacle_type = "ptera" if state[5] >= 0.5 else "cactus"
    distance = float(state[6])
    no_obstacle = distance >= 590.0 and state[7] == 0.0 and state[8] == 0.0
    return {
        "speed": float(state[4]),
        "obstacle_type": "none" if no_obstacle else obstacle_type,
        "distance": distance,
        "distance_bin": _bin_name(distance, DISTANCE_BINS),
        "speed_bin": _bin_name(float(state[4]), SPEED_BINS),
    }


def _rate_summary(counts: dict[str, int], prefix: str) -> dict:
    total = sum(counts.values())
    result = {f"{prefix}_{key}_count": value for key, value in counts.items()}
    for key, value in counts.items():
        result[f"{prefix}_{key}_rate"] = 0.0 if total == 0 else float(value / total)
    result[f"{prefix}_total"] = total
    return result


def _load_agent(env_name: str, algo_name: str, model_kind: str, model_path_override: str | Path | None, seed: int):
    env_config = get_env_config(env_name)
    env = make_env(env_name, render=False)
    seed_env(env, seed)
    state_dim = get_state_dim(env_name, env)
    action_dim = int(env.action_space.n)
    agent = make_agent(env_name, algo_name, state_dim, action_dim)
    env.close()

    model_path = Path(model_path_override) if model_path_override is not None else MODELS_DIR / f"{env_name}_{algo_name}_{model_kind}.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"未找到模型文件: {model_path}")
    agent.load(model_path)
    agent.epsilon = 0.0
    return agent, model_path, env_config


def analyze_dino_strategy(
    algo_name: str = "perdqn",
    model_kind: str = "best",
    episodes: int | None = None,
    seed: int | None = None,
    render: bool = False,
    model_path_override: str | Path | None = None,
) -> dict:
    env_name = "dino"
    env_config = get_env_config(env_name)
    run_seed = env_config.seed if seed is None else seed
    test_seed = run_seed + env_config.final_test_seed_offset
    set_global_seed(test_seed)
    agent, model_path, _ = _load_agent(env_name, algo_name, model_kind, model_path_override, test_seed)

    env = make_env(env_name, render=render)
    seed_env(env, test_seed)
    total_episodes = env_config.final_test_episodes if episodes is None else episodes

    action_counts = {name: 0 for name in ACTION_NAMES.values()}
    distance_action_counts = defaultdict(lambda: {name: 0 for name in ACTION_NAMES.values()})
    type_action_counts = defaultdict(lambda: {name: 0 for name in ACTION_NAMES.values()})
    speed_action_counts = defaultdict(lambda: {name: 0 for name in ACTION_NAMES.values()})
    rewards = []
    steps_list = []
    scores = []
    obstacles = []

    for episode_idx in range(total_episodes):
        raw_state, _ = env.reset(seed=test_seed + episode_idx)
        state = process_state(env_name, raw_state, env)
        total_reward = 0.0
        info = {}

        for step in range(1, env_config.max_steps_per_episode + 1):
            features = _dino_features(raw_state)
            action = agent.select_action(state, training=False)
            action_name = ACTION_NAMES.get(action, str(action))
            action_counts[action_name] += 1
            distance_action_counts[features["distance_bin"]][action_name] += 1
            type_action_counts[features["obstacle_type"]][action_name] += 1
            speed_action_counts[features["speed_bin"]][action_name] += 1

            next_raw_state, reward, terminated, truncated, info = env.step(action)
            raw_state = next_raw_state
            state = process_state(env_name, raw_state, env)
            total_reward += reward
            if terminated or truncated:
                break

        rewards.append(float(total_reward))
        steps_list.append(float(step))
        scores.append(float(info.get("score", 0.0)))
        obstacles.append(float(info.get("obstacles_cleared", 0.0)))

    env.close()

    result = {
        "env_name": env_name,
        "algo_name": algo_name,
        "model_kind": model_kind,
        "model_path": str(model_path),
        "episodes": total_episodes,
        "seed": run_seed,
        "test_seed_start": test_seed,
        "avg_reward": float(np.mean(rewards)),
        "avg_steps": float(np.mean(steps_list)),
        "avg_score": float(np.mean(scores)),
        "avg_obstacles_cleared": float(np.mean(obstacles)),
        "max_obstacles_cleared": float(np.max(obstacles)),
    }
    result.update(_rate_summary(action_counts, "action"))

    for bin_name, counts in distance_action_counts.items():
        result.update(_rate_summary(counts, f"distance_{bin_name}"))
    for obstacle_type, counts in type_action_counts.items():
        result.update(_rate_summary(counts, f"type_{obstacle_type}"))
    for speed_bin, counts in speed_action_counts.items():
        result.update(_rate_summary(counts, f"speed_{speed_bin}"))
    return result
