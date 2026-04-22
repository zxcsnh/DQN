"""Single-run DQN training entrypoint."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from config import DQNConfig, TrainingStats
from dqn.agent import DQNAgent
from dqn.env import make_env
from dqn.utils import TrainingLogger, print_episode_stats, set_seed


def train(config: DQNConfig) -> dict[str, Any]:
    """Train one DQN agent and persist the artifacts for this run."""
    core = config.core
    training = config.training
    replay = config.replay
    logging = config.logging
    evaluation = config.evaluation
    preprocess = config.preprocess
    device_config = config.device_config

    set_seed(
        core.seed,
        deterministic_torch=device_config.deterministic_torch,
        allow_tf32=device_config.allow_tf32,
    )
    model_prefix = config.model_name_prefix()
    training_start_step = replay.training_start_steps

    env = None
    eval_env = None
    agent = None
    logger = None

    try:
        # Training uses the optimization protocol: reward clipping may be enabled
        # and life loss can be treated as an episode boundary.
        env = make_env(
            core.env_name,
            frame_size=preprocess.frame_size,
            frame_skip=preprocess.frame_skip,
            frame_stack=preprocess.frame_stack,
            noop_max=preprocess.noop_max,
            clip_reward=preprocess.clip_reward,
            seed=core.seed,
            terminal_on_life_loss=preprocess.terminal_on_life_loss,
        )
        # Evaluation uses the reporting protocol: unclipped rewards and no
        # life-loss termination, matching playback semantics.
        eval_env = make_env(
            core.env_name,
            frame_size=preprocess.frame_size,
            frame_skip=preprocess.frame_skip,
            frame_stack=preprocess.frame_stack,
            noop_max=preprocess.noop_max,
            clip_reward=False,
            seed=core.seed,
            terminal_on_life_loss=False,
        )

        device = config.get_device()
        print(f"Training device: {device}")
        print(f"Environment: {core.env_name}")
        print(f"Num actions: {env.action_space.n}")
        print(f"Training starts at env step: {training_start_step}")

        agent = DQNAgent(
            num_actions=env.action_space.n,
            input_channels=preprocess.frame_stack,
            input_shape=preprocess.frame_size,
            device=device,
            learning_rate=training.learning_rate,
            gamma=training.gamma,
            initial_random_steps=replay.initial_random_steps,
            epsilon_start=replay.epsilon_start,
            epsilon_end=replay.epsilon_end,
            epsilon_decay=replay.epsilon_decay,
            buffer_size=replay.buffer_size,
            batch_size=training.batch_size,
            target_update_interval_updates=training.target_update_interval_updates,
            use_per=replay.use_per,
            per_alpha=replay.per_alpha,
            per_beta_start=replay.per_beta_start,
            per_beta_updates=replay.per_beta_updates,
            use_soft_target_update=training.use_soft_target_update,
            soft_target_update_tau=training.soft_target_update_tau,
            frame_stack=preprocess.frame_stack,
            optimizer_name=training.optimizer_name,
            rmsprop_alpha=training.rmsprop_alpha,
            rmsprop_eps=training.rmsprop_eps,
            rmsprop_centered=training.rmsprop_centered,
            use_double_dqn=training.use_double_dqn,
            replay_sample_torch_fastpath=replay.replay_sample_torch_fastpath,
            seed=core.seed,
        )

        logger = TrainingLogger(
            logging.log_dir,
            log_step_metrics=logging.log_step_metrics,
            step_log_interval=logging.step_log_interval,
            step_log_stream=logging.step_log_stream,
            step_log_file=logging.step_log_file,
        )
        stats = TrainingStats()

        os.makedirs(logging.save_dir, exist_ok=True)
        os.makedirs(logging.log_dir, exist_ok=True)

        with open(os.path.join(logging.log_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config.to_flat_dict(), f, indent=2, ensure_ascii=False)

        print("\nStart training...")
        print("-" * 80)

        stop_training = False
        last_episode_reward = logger.episode_rewards[-1] if logger.episode_rewards else 0.0
        last_avg_loss = logger.episode_losses[-1] if logger.episode_losses else 0.0
        next_eval_step = _next_eval_step(config, stats.total_steps, logger)

        def run_evaluation(step: int) -> None:
            eval_reward = evaluate(
                agent,
                eval_env,
                config,
                num_eval=evaluation.eval_episodes,
            )
            logger.log_evaluation(step, eval_reward)
            print(f"Evaluation reward @ step {step}: {eval_reward:.2f}")

            if eval_reward > stats.best_eval_reward:
                stats.best_eval_reward = eval_reward

        episode = agent.episodes_done
        while True:
            if core.num_episodes is not None and episode >= core.num_episodes:
                break
            if agent.env_steps_done >= core.max_steps:
                print(f"Reached max training steps: {core.max_steps}")
                break

            episode += 1
            reset_seed = core.seed if stats.total_steps == 0 and episode == 1 else None
            state, _ = env.reset(seed=reset_seed)
            agent.begin_episode(state)

            episode_reward = 0.0
            episode_loss: list[float] = []
            episode_length = 0
            done = False

            while not done:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                agent.store_transition(
                    state,
                    action,
                    reward,
                    next_state,
                    terminated,
                    truncated,
                )
                agent.on_env_step()

                current_step = agent.env_steps_done
                if _should_train(agent, config, training_start_step):
                    for _ in range(training.gradient_steps):
                        loss = agent.train_step()
                        if loss is None:
                            break
                        episode_loss.append(loss)
                        logger.log_step(current_step, agent.epsilon, loss)

                state = next_state
                episode_reward += reward
                episode_length += 1

                if next_eval_step is not None:
                    while current_step >= next_eval_step:
                        run_evaluation(next_eval_step)
                        next_eval_step += evaluation.eval_interval_env_steps

                if current_step >= core.max_steps:
                    stop_training = True
                    break

            avg_loss = float(np.mean(episode_loss)) if episode_loss else 0.0
            logger.log_episode(episode, episode_reward, episode_length, agent.epsilon)

            stats.episode = episode
            stats.total_steps = agent.env_steps_done
            stats.episode_reward = episode_reward
            stats.episode_length = episode_length
            stats.epsilon = agent.epsilon
            stats.loss = avg_loss
            stats.avg_reward_100 = logger.avg_rewards[-1]
            agent.episodes_done = episode

            last_episode_reward = episode_reward
            last_avg_loss = avg_loss

            print_episode_stats(
                episode,
                episode_reward,
                episode_length,
                agent.epsilon,
                stats.avg_reward_100,
                stats.best_eval_reward,
                avg_loss,
            )

            if episode % logging.save_freq == 0:
                periodic_model_path = os.path.join(
                    logging.save_dir, f"{model_prefix}_ep{episode}.pth"
                )
                agent.save(periodic_model_path)

            if next_eval_step is None and episode % evaluation.eval_interval_episodes == 0:
                run_evaluation(stats.total_steps)

            if stop_training:
                print(f"Reached max training steps: {core.max_steps}")
                break

        if evaluation.eval_episodes > 0:
            needs_final_eval = (
                len(logger.eval_steps) == 0 or logger.eval_steps[-1] < stats.total_steps
            )
            if needs_final_eval:
                run_evaluation(stats.total_steps)

        logger.save_metrics(os.path.join(logging.log_dir, "metrics.json"))
        logger.plot_training_curves(os.path.join(logging.log_dir, "training_curves.png"))

        summary: dict[str, Any] = {
            "env_name": core.env_name,
            "seed": core.seed,
            "use_per": replay.use_per,
            "use_double_dqn": training.use_double_dqn,
            "optimizer_name": training.optimizer_name,
            "total_steps": stats.total_steps,
            "episodes_completed": stats.episode,
            "best_eval_reward": stats.best_eval_reward,
            "best_eval_step": _best_eval_step(logger),
            "final_eval_reward": logger.eval_rewards[-1] if logger.eval_rewards else None,
            "final_train_reward": last_episode_reward,
            "final_train_avg_reward_100": stats.avg_reward_100,
            "final_avg_reward_100": stats.avg_reward_100,
            "final_loss": last_avg_loss,
            "eval_steps": logger.eval_steps,
            "eval_rewards": logger.eval_rewards,
            "train_clip_reward": preprocess.clip_reward,
            "train_terminal_on_life_loss": preprocess.terminal_on_life_loss,
            "eval_clip_reward": False,
            "eval_terminal_on_life_loss": False,
            "eval_epsilon": evaluation.eval_epsilon,
            "eval_max_episode_steps": evaluation.eval_max_episode_steps,
            "target_bootstrap_mask": "terminated_only",
            "protocol_note": "Train may use clipped rewards and life-loss termination; eval/play use unclipped rewards without life-loss termination.",
            "model_prefix": model_prefix,
        }
        with open(
            os.path.join(logging.log_dir, "run_summary.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print("-" * 80)
        print("Training finished!")
        print(f"Best evaluation reward: {stats.best_eval_reward:.2f}")
        print(f"Model saved to: {logging.save_dir}")
        return summary
    except Exception:
        raise
    finally:
        if logger is not None:
            logger.close()
        if eval_env is not None:
            eval_env.close()
        if env is not None:
            env.close()


def evaluate(
    agent: DQNAgent,
    eval_env,
    config: DQNConfig,
    num_eval: int = 10,
) -> float:
    """Evaluate the policy and return the mean reward."""
    if num_eval <= 0:
        raise ValueError("num_eval must be a positive integer.")

    agent.eval_mode()
    total_rewards = []

    try:
        for eval_idx in range(num_eval):
            eval_seed = config.core.seed + config.evaluation.eval_seed_offset + eval_idx
            state, _ = eval_env.reset(seed=eval_seed)
            episode_reward = 0.0
            episode_steps = 0
            done = False

            while not done:
                if (
                    config.evaluation.eval_max_episode_steps is not None
                    and episode_steps >= config.evaluation.eval_max_episode_steps
                ):
                    break

                action = agent.select_action(
                    state,
                    evaluate=True,
                    epsilon_override=config.evaluation.eval_epsilon,
                )
                state, reward, terminated, truncated, _ = eval_env.step(action)
                episode_reward += reward
                episode_steps += 1
                done = terminated or truncated

            total_rewards.append(episode_reward)
    finally:
        agent.train_mode()

    return float(np.mean(total_rewards))


def _should_train(agent: DQNAgent, config: DQNConfig, training_start_step: int) -> bool:
    return (
        len(agent.memory) >= training_start_step
        and agent.env_steps_done >= training_start_step
        and agent.env_steps_done % config.training.train_freq == 0
    )


def _next_eval_step(
    config: DQNConfig,
    current_total_steps: int,
    logger: TrainingLogger,
) -> int | None:
    if config.evaluation.eval_interval_env_steps <= 0:
        return None

    next_eval_step = config.evaluation.eval_interval_env_steps
    if logger.eval_steps:
        next_eval_step = logger.eval_steps[-1] + config.evaluation.eval_interval_env_steps

    while next_eval_step <= current_total_steps:
        next_eval_step += config.evaluation.eval_interval_env_steps
    return next_eval_step


def _best_eval_step(logger: TrainingLogger) -> int | None:
    if not logger.eval_rewards:
        return None
    best_index = int(np.argmax(np.asarray(logger.eval_rewards, dtype=np.float32)))
    return logger.eval_steps[best_index]


def main() -> None:
    config = DQNConfig()
    train(config)


if __name__ == "__main__":
    main()
