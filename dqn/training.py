from __future__ import annotations

from collections import deque

import numpy as np
from tqdm import trange

from config import COMMON_CONFIG, LOGS_DIR, MODELS_DIR, PER_CONFIG, ensure_result_dirs, get_env_config, supported_envs
from .agents import DQNAgent, PERDQNAgent
from .evaluation import evaluate_agent
from .envs import make_env
from .utils.logger import CSVLogger
from .utils.seed_utils import seed_env, set_global_seed
from .utils.state_processor import get_state_dim, process_state


def validate_names(env_name: str, algo_name: str) -> None:
    if env_name not in supported_envs():
        raise ValueError(f"不支持的环境名称: {env_name}，可选值为 {supported_envs()}")
    if algo_name not in {"dqn", "perdqn"}:
        raise ValueError(f"不支持的算法名称: {algo_name}")


def make_agent(env_name: str, algo_name: str, state_dim: int, action_dim: int):
    env_config = get_env_config(env_name)
    if algo_name == "dqn":
        return DQNAgent(state_dim, action_dim, COMMON_CONFIG, env_config, env_name, algo_name="dqn")
    return PERDQNAgent(state_dim, action_dim, COMMON_CONFIG, PER_CONFIG, env_config, env_name)


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


def train(
    env_name: str,
    algo_name: str,
    render: bool = False,
    plot_after_train: bool = False,
    seed: int | None = None,
    log_name_suffix: str = "",
) -> dict:
    validate_names(env_name, algo_name)
    ensure_result_dirs()
    run_seed = COMMON_CONFIG.seed if seed is None else seed
    set_global_seed(run_seed)

    env_config = get_env_config(env_name)
    if env_config.eval_interval_episodes < 1:
        raise ValueError("eval_interval_episodes 必须大于等于 1")
    if env_config.eval_episodes < 1:
        raise ValueError("eval_episodes 必须大于等于 1")

    env = make_env(env_name, render=render)
    seed_env(env, run_seed)

    state_dim = get_state_dim(env_name, env)
    action_dim = int(env.action_space.n)
    agent = make_agent(env_name, algo_name, state_dim, action_dim)

    suffix = f"_{log_name_suffix}" if log_name_suffix else ""
    log_path = LOGS_DIR / f"{env_name}_{algo_name}{suffix}_train_log.csv"
    logger = CSVLogger(log_path)

    best_train_reward = float("-inf")
    best_eval_reward = float("-inf")
    best_model_episode: int | None = None
    best_model_path = MODELS_DIR / f"{env_name}_{algo_name}{suffix}_best.pth"
    last_eval_metrics: dict | None = None
    recent_rewards = deque(maxlen=5)
    recent_steps = deque(maxlen=5)
    recent_losses = deque(maxlen=5)
    recent_epsilon = deque(maxlen=5)
    progress = trange(env_config.episodes, desc=f"Training {env_name}-{algo_name}")

    for episode in progress:
        raw_state, _ = env.reset()
        state = process_state(env_name, raw_state, env)
        episode_reward = 0.0
        episode_losses = []
        terminated = False
        info = {}
        max_position = -1.2

        for step in range(1, env_config.max_steps_per_episode + 1):
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
        best_train_reward = max(best_train_reward, episode_reward)

        eval_metrics = None
        is_best_model = 0
        if (episode + 1) % env_config.eval_interval_episodes == 0:
            eval_metrics = evaluate_agent(
                env_name=env_name,
                algo_name=algo_name,
                agent=agent,
                episodes=env_config.eval_episodes,
                render=False,
                seed=run_seed + 10_000,
            )
            last_eval_metrics = eval_metrics
            if eval_metrics["avg_reward"] > best_eval_reward:
                best_eval_reward = eval_metrics["avg_reward"]
                best_model_episode = episode + 1
                agent.save(best_model_path)
                is_best_model = 1

        logger.log(
            {
                "episode": episode + 1,
                "total_reward": episode_reward,
                "steps": step,
                "epsilon": agent.epsilon,
                "loss": average_loss,
                "success": success,
                "custom_metric": custom_metric,
                "eval_avg_reward": "" if eval_metrics is None else eval_metrics["avg_reward"],
                "eval_avg_steps": "" if eval_metrics is None else eval_metrics["avg_steps"],
                "eval_success_rate": "" if eval_metrics is None else eval_metrics["success_rate"],
                "eval_avg_custom_metric": "" if eval_metrics is None else eval_metrics["avg_custom_metric"],
                "is_best_model": is_best_model,
            }
        )

        recent_rewards.append(episode_reward)
        recent_steps.append(step)
        recent_losses.append(average_loss)
        recent_epsilon.append(agent.epsilon)

        if (episode + 1) % 5 == 0:
            start_episode = episode + 2 - len(recent_rewards)
            progress.write(
                "\n".join(
                    [
                        f"Recent 5 episodes ({start_episode}-{episode + 1})",
                        f"  rewards: {[round(value, 2) for value in recent_rewards]}",
                        f"  steps:   {list(recent_steps)}",
                        f"  losses:  {[round(value, 4) for value in recent_losses]}",
                        f"  epsilon:  {[round(value, 4) for value in recent_epsilon]}",
                    ]
                )
            )

    if env_config.episodes % env_config.eval_interval_episodes != 0:
        final_eval_metrics = evaluate_agent(
            env_name=env_name,
            algo_name=algo_name,
            agent=agent,
            episodes=env_config.eval_episodes,
            render=False,
            seed=run_seed + 10_000,
        )
        last_eval_metrics = final_eval_metrics
        if final_eval_metrics["avg_reward"] > best_eval_reward:
            best_eval_reward = final_eval_metrics["avg_reward"]
            best_model_episode = env_config.episodes
            agent.save(best_model_path)

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
        "best_train_reward": best_train_reward,
        "best_eval_reward": best_eval_reward,
        "best_model_episode": best_model_episode,
        "best_model_path": str(best_model_path),
        "final_model_path": str(final_model_path),
        "last_eval_avg_reward": None if last_eval_metrics is None else last_eval_metrics["avg_reward"],
        "log_path": str(log_path),
    }
