"""单次 DQN 训练入口。"""

import json
import os
from dataclasses import asdict
from typing import Any

import numpy as np

from config import DQNConfig, TrainingStats
from dqn.agent import DQNAgent
from dqn.env import make_env
from dqn.utils import TrainingLogger, print_episode_stats, set_seed


def train(config: DQNConfig) -> dict[str, Any]:
    """训练一个 DQN 智能体，并保存本次运行的全部产物。"""
    set_seed(config.seed)
    model_prefix = config.model_name_prefix()

    env = make_env(
        config.env_name,
        frame_size=config.frame_size,
        frame_skip=config.frame_skip,
        frame_stack=config.frame_stack,
        noop_max=config.noop_max,
        clip_reward=config.clip_reward,
        seed=config.seed,
        terminal_on_life_loss=config.terminal_on_life_loss,
    )

    device = config.get_device()
    print(f"Training device: {device}")

    num_actions = env.action_space.n
    print(f"Environment: {config.env_name}")
    print(f"Num actions: {num_actions}")

    agent = DQNAgent(
        num_actions=num_actions,
        input_channels=config.frame_stack,
        device=device,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        epsilon_start=config.epsilon_start,
        epsilon_end=config.epsilon_end,
        epsilon_decay=config.epsilon_decay,
        buffer_size=config.buffer_size,
        batch_size=config.batch_size,
        target_update_freq=config.target_update_freq,
        use_per=config.use_per,
        per_alpha=config.per_alpha,
        per_beta_start=config.per_beta_start,
        per_beta_frames=config.per_beta_frames,
        soft_update=config.soft_update,
        tau=config.tau,
    )

    logger = TrainingLogger(config.log_dir)
    stats = TrainingStats()

    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    with open(os.path.join(config.log_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2, ensure_ascii=False)

    print("\nStart training...")
    print("-" * 80)

    stop_training = False
    last_episode_reward = 0.0
    last_avg_loss = 0.0
    next_eval_step = (
        config.eval_interval_steps if config.eval_interval_steps > 0 else None
    )

    def run_evaluation(step: int):
        eval_reward = evaluate(
            agent,
            config,
            num_eval=config.eval_episodes,
        )
        logger.log_evaluation(step, eval_reward)
        print(f"Evaluation reward @ step {step}: {eval_reward:.2f}")

        # 最优模型以评估回报为准，而不是训练期回报。
        if eval_reward > stats.best_eval_reward:
            stats.best_eval_reward = eval_reward
            best_path = os.path.join(config.save_dir, f"{model_prefix}_best.pth")
            agent.save(best_path, save_replay_buffer=config.save_replay_buffer)

    for episode in range(1, config.num_episodes + 1):
        # 只在第一次 reset 时显式传 seed，后续让环境自然推进随机序列。
        reset_seed = config.seed if episode == 1 else None
        state, _ = env.reset(seed=reset_seed)
        episode_reward = 0.0
        episode_loss = []
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

            if len(agent.memory) >= config.min_buffer_size:
                loss = agent.train_step()
                if loss is not None:
                    episode_loss.append(loss)
                    logger.log_step(
                        stats.total_steps + episode_length + 1,
                        agent.epsilon,
                        loss,
                    )

            state = next_state
            episode_reward += reward
            episode_length += 1
            current_step = stats.total_steps + episode_length

            if next_eval_step is not None:
                while current_step >= next_eval_step:
                    run_evaluation(next_eval_step)
                    next_eval_step += config.eval_interval_steps

            if current_step >= config.max_steps:
                stop_training = True
                break

        avg_loss = float(np.mean(episode_loss)) if episode_loss else 0.0
        logger.log_episode(episode, episode_reward, episode_length, agent.epsilon)

        stats.episode = episode
        stats.total_steps += episode_length
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

        if episode % config.save_freq == 0:
            checkpoint_path = os.path.join(
                config.save_dir, f"{model_prefix}_ep{episode}.pth"
            )
            agent.save(checkpoint_path, save_replay_buffer=config.save_replay_buffer)

        if next_eval_step is None and episode % config.eval_freq == 0:
            run_evaluation(stats.total_steps)

        if stop_training:
            print(f"Reached max training steps: {config.max_steps}")
            break

    # 兜底最终评估：避免短训练时无评估记录，确保 summary 可用。
    if config.eval_episodes > 0:
        needs_final_eval = (
            len(logger.eval_steps) == 0 or logger.eval_steps[-1] < stats.total_steps
        )
        if needs_final_eval:
            run_evaluation(stats.total_steps)

    final_path = os.path.join(config.save_dir, f"{model_prefix}_final.pth")
    agent.save(final_path, save_replay_buffer=config.save_replay_buffer)

    logger.save_metrics(os.path.join(config.log_dir, "metrics.json"))
    logger.plot_training_curves(os.path.join(config.log_dir, "training_curves.png"))

    logger.close()
    env.close()

    summary: dict[str, Any] = {
        "env_name": config.env_name,
        "seed": config.seed,
        "use_per": config.use_per,
        "total_steps": stats.total_steps,
        "episodes_completed": stats.episode,
        "best_eval_reward": stats.best_eval_reward,
        "final_train_reward": last_episode_reward,
        "final_avg_reward_100": stats.avg_reward_100,
        "final_loss": last_avg_loss,
        "eval_steps": logger.eval_steps,
        "eval_rewards": logger.eval_rewards,
        "checkpoint_prefix": model_prefix,
    }
    with open(
        os.path.join(config.log_dir, "run_summary.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("-" * 80)
    print("Training finished!")
    print(f"Best evaluation reward: {stats.best_eval_reward:.2f}")
    print(f"Model saved to: {config.save_dir}")
    return summary


def evaluate(
    agent: DQNAgent,
    config: DQNConfig,
    num_eval: int = 10,
) -> float:
    """用贪心策略评估模型，并返回平均回报。"""
    if num_eval <= 0:
        raise ValueError("num_eval must be a positive integer.")

    eval_env = make_env(
        config.env_name,
        frame_size=config.frame_size,
        frame_skip=config.frame_skip,
        frame_stack=config.frame_stack,
        noop_max=config.noop_max,
        clip_reward=False,
        seed=config.seed,
        terminal_on_life_loss=False,
    )

    agent.eval_mode()
    total_rewards = []

    for eval_idx in range(num_eval):
        eval_seed = config.seed + config.eval_seed_offset + eval_idx
        state, _ = eval_env.reset(seed=eval_seed)
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state, evaluate=True)
            state, reward, terminated, truncated, _ = eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)

    agent.train_mode()
    eval_env.close()

    return float(np.mean(total_rewards))


def main():
    config = DQNConfig()
    train(config)


if __name__ == "__main__":
    main()
