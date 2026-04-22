from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import trange

from config import COMMON_CONFIG, ENV_CONFIGS, LOGS_DIR, MODELS_DIR, PER_CONFIG, ensure_result_dirs
from .agents import DQNAgent, PERDQNAgent
from .envs import make_env
from .utils.logger import CSVLogger
from .utils.seed_utils import seed_env, set_global_seed
from .utils.state_processor import get_state_dim, process_state


def validate_names(env_name: str, algo_name: str) -> None:
    if env_name not in ENV_CONFIGS:
        raise ValueError(f"不支持的环境名称: {env_name}")
    if algo_name not in {"dqn", "perdqn"}:
        raise ValueError(f"不支持的算法名称: {algo_name}")


def make_agent(env_name: str, algo_name: str, state_dim: int, action_dim: int):
    if algo_name == "dqn":
        return DQNAgent(state_dim, action_dim, COMMON_CONFIG, ENV_CONFIGS[env_name], env_name, algo_name="dqn")
    return PERDQNAgent(state_dim, action_dim, COMMON_CONFIG, PER_CONFIG, ENV_CONFIGS[env_name], env_name)


def compute_episode_metrics(env_name: str, episode_reward: float, info: dict, terminated: bool) -> tuple[int, float]:
    if env_name == "taxi":
        success = 1 if episode_reward > 0 and terminated else 0
        return success, float(success)

    if env_name == "mountaincar":
        max_position = float(info.get("max_position", -1.2))
        success = 1 if max_position >= 0.5 else 0
        return success, max_position

    obstacles_cleared = float(info.get("obstacles_cleared", 0))
    success = 1 if obstacles_cleared >= ENV_CONFIGS[env_name]["success_threshold"] else 0
    return success, obstacles_cleared


def train(env_name: str, algo_name: str, render: bool = False, plot_after_train: bool = False, seed: int | None = None, log_name_suffix: str = "") -> dict:
    # 训练入口：负责环境创建、交互采样、网络更新、日志记录和模型保存。
    validate_names(env_name, algo_name)
    ensure_result_dirs()
    run_seed = COMMON_CONFIG.seed if seed is None else seed
    set_global_seed(run_seed)

    env = make_env(env_name, render=render)
    seed_env(env, run_seed)

    state_dim = get_state_dim(env_name, env)
    action_dim = int(env.action_space.n)
    agent = make_agent(env_name, algo_name, state_dim, action_dim)

    suffix = f"_{log_name_suffix}" if log_name_suffix else ""
    log_path = LOGS_DIR / f"{env_name}_{algo_name}{suffix}_train_log.csv"
    logger = CSVLogger(log_path)

    best_reward = float("-inf")
    env_config = ENV_CONFIGS[env_name]

    for episode in trange(env_config["episodes"], desc=f"Training {env_name}-{algo_name}"):
        raw_state, _ = env.reset()
        state = process_state(env_name, raw_state, env)
        episode_reward = 0.0
        episode_losses = []
        terminated = False
        info = {}
        max_position = -1.2

        # 每个回合都执行“动作选择 -> 环境交互 -> 经验回放 -> 网络更新”的标准流程。
        for step in range(1, env_config["max_steps"] + 1):
            action = agent.select_action(state, training=True)
            next_raw_state, reward, terminated, truncated, info = env.step(action)
            next_state = process_state(env_name, next_raw_state, env)
            done = terminated or truncated

            if env_name == "mountaincar":
                max_position = max(max_position, float(next_raw_state[0]))
                info["max_position"] = max_position

            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.update()
            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            episode_reward += reward
            if done:
                break

        success, custom_metric = compute_episode_metrics(env_name, episode_reward, info, terminated)
        average_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
        logger.log(
            {
                "episode": episode + 1,
                "total_reward": episode_reward,
                "steps": step,
                "epsilon": agent.epsilon,
                "loss": average_loss,
                "success": success,
                "custom_metric": custom_metric,
            }
        )

        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(MODELS_DIR / f"{env_name}_{algo_name}{suffix}_best.pth")

    final_model_path = MODELS_DIR / f"{env_name}_{algo_name}{suffix}_final.pth"
    agent.save(final_model_path)
    env.close()

    if plot_after_train:
        from .utils.plot_utils import plot_env_comparisons

        dqn_log = LOGS_DIR / f"{env_name}_dqn_train_log.csv"
        perdqn_log = LOGS_DIR / f"{env_name}_perdqn_train_log.csv"
        if dqn_log.exists() and perdqn_log.exists():
            plot_env_comparisons(env_name, dqn_log, perdqn_log, COMMON_CONFIG.moving_average_window)

    return {
        "env_name": env_name,
        "algo_name": algo_name,
        "seed": run_seed,
        "best_reward": best_reward,
        "final_model_path": str(final_model_path),
        "log_path": str(log_path),
    }
