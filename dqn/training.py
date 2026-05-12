from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from tqdm import trange

from config import RunDirs, create_run_dirs, ensure_result_dirs, get_env_config
from .evaluation import evaluate_agent
from .envs import make_env
from .shared import compute_episode_metrics, make_agent, validate_names
from .utils.logger import CSVLogger
from .utils.seed_utils import seed_env, set_global_seed
from .utils.state_processor import get_state_dim, process_state


def _episode_extra_metrics(env_name: str, info: dict, episode_reward: float, step: int) -> dict:
    values = {
        "reward_per_step": float(episode_reward / step),
        "max_position": "",
        "score": "",
        "high_score": "",
        "speed": "",
        "obstacles_cleared": "",
    }
    if env_name == "mountaincar":
        values["max_position"] = info.get("max_position", "")
    if env_name == "dino":
        values.update(
            {
                "score": info.get("score", ""),
                "high_score": info.get("high_score", ""),
                "speed": info.get("speed", ""),
                "obstacles_cleared": info.get("obstacles_cleared", ""),
            }
        )
    return values


def _prefixed_metrics(prefix: str, metrics: dict | None) -> dict:
    if metrics is None:
        return {}
    return {f"{prefix}_{key}": value for key, value in metrics.items() if key != "episodes_detail"}


def _run_model_test(agent, model_path: Path, env_name: str, algo_name: str, episodes: int, seed: int, model_kind: str) -> dict:
    agent.load(model_path)
    agent.epsilon = 0.0
    metrics = evaluate_agent(
        env_name=env_name,
        algo_name=algo_name,
        agent=agent,
        episodes=episodes,
        render=False,
        seed=seed,
    )
    metrics["model_kind"] = model_kind
    metrics["model_path"] = str(model_path)
    metrics["test_seed_start"] = seed
    return metrics


def train(
    env_name: str,
    algo_name: str,
    render: bool = False,
    plot_after_train: bool = False,
    seed: int | None = None,
    log_name_suffix: str = "",
    run_dir: str | Path | None = None,
    run_final_test: bool = True,
) -> dict:
    validate_names(env_name, algo_name)
    ensure_result_dirs()
    env_config = get_env_config(env_name)
    run_seed = env_config.seed if seed is None else seed
    set_global_seed(run_seed)

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

    if run_dir is None:
        run_dirs = create_run_dirs(env_name, algo_name, run_seed)
    else:
        resolved_run_dir = Path(run_dir)
        logs_dir = resolved_run_dir / "logs"
        models_dir = resolved_run_dir / "models"
        figures_dir = resolved_run_dir / "figures"
        for path in (resolved_run_dir, logs_dir, models_dir, figures_dir):
            path.mkdir(parents=True, exist_ok=True)
        run_dirs = RunDirs(
            run_dir=resolved_run_dir,
            logs_dir=logs_dir,
            models_dir=models_dir,
            figures_dir=figures_dir,
        )

    log_path = run_dirs.logs_dir / f"{env_name}_{algo_name}{suffix}_train_log.csv"
    logger = CSVLogger(log_path)

    best_train_reward = float("-inf")
    best_eval_reward = float("-inf")
    best_model_episode: int | None = None
    best_model_path = run_dirs.models_dir / f"{env_name}_{algo_name}{suffix}_best.pth"
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
            episode_done = terminated or truncated
            bootstrap_done = terminated

            if env_name == "mountaincar":
                max_position = max(max_position, float(next_raw_state[0]))
                info["max_position"] = max_position

            agent.store_transition(state, action, reward, next_state, bootstrap_done)
            loss = agent.update()
            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            episode_reward += reward
            if episode_done:
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

        log_row = {
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
            "per_beta": getattr(agent, "last_beta", ""),
            "per_mean_abs_td_error": getattr(agent, "last_mean_abs_td_error", ""),
            "per_max_abs_td_error": getattr(agent, "last_max_abs_td_error", ""),
            "per_mean_weight": getattr(agent, "last_mean_weight", ""),
            "per_max_weight": getattr(agent, "last_max_weight", ""),
        }
        log_row.update(_episode_extra_metrics(env_name, info, episode_reward, step))
        logger.log(log_row)

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

    final_model_path = run_dirs.models_dir / f"{env_name}_{algo_name}{suffix}_final.pth"
    agent.save(final_model_path)
    if best_model_episode is None:
        best_model_episode = env_config.episodes
        best_eval_reward = best_train_reward
        agent.save(best_model_path)
    env.close()

    best_final_test_metrics = None
    final_model_test_metrics = None
    if run_final_test:
        test_seed = run_seed + env_config.final_test_seed_offset
        best_final_test_metrics = _run_model_test(
            agent,
            best_model_path,
            env_name,
            algo_name,
            env_config.final_test_episodes,
            test_seed,
            "best",
        )
        final_model_test_metrics = _run_model_test(
            agent,
            final_model_path,
            env_name,
            algo_name,
            env_config.final_test_episodes,
            test_seed,
            "final",
        )

    if plot_after_train:
        from .utils.plot_utils import plot_single_run

        plot_single_run(log_path, env_name, algo_name, env_config.moving_average_window, figures_dir=run_dirs.figures_dir)

    result = {
        "env_name": env_name,
        "algo_name": algo_name,
        "seed": run_seed,
        "episodes": env_config.episodes,
        "eval_episodes": env_config.eval_episodes,
        "final_test_episodes": env_config.final_test_episodes,
        "best_train_reward": best_train_reward,
        "best_eval_reward": best_eval_reward,
        "best_model_episode": best_model_episode,
        "best_model_path": str(best_model_path),
        "final_model_path": str(final_model_path),
        "last_eval_avg_reward": None if last_eval_metrics is None else last_eval_metrics["avg_reward"],
        "log_path": str(log_path),
        "run_dir": str(run_dirs.run_dir),
        "logs_dir": str(run_dirs.logs_dir),
        "models_dir": str(run_dirs.models_dir),
        "figures_dir": str(run_dirs.figures_dir),
    }
    result.update(_prefixed_metrics("best_final_test", best_final_test_metrics))
    result.update(_prefixed_metrics("final_model_test", final_model_test_metrics))
    return result
