import json
import os
from dataclasses import asdict

import numpy as np

from config import DQNConfig, TrainingStats
from dqn.agent import DQNAgent
from dqn.env import make_env
from dqn.utils import TrainingLogger, print_episode_stats, set_seed


def train(config: DQNConfig):
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
    for episode in range(1, config.num_episodes + 1):
        reset_seed = config.seed if episode == 1 else None
        state, _ = env.reset(seed=reset_seed)
        episode_reward = 0
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

            if stats.total_steps + episode_length >= config.max_steps:
                stop_training = True
                break

        avg_loss = np.mean(episode_loss) if episode_loss else 0.0
        logger.log_episode(episode, episode_reward, episode_length, agent.epsilon)

        stats.episode = episode
        stats.total_steps += episode_length
        stats.episode_reward = episode_reward
        stats.episode_length = episode_length
        stats.epsilon = agent.epsilon
        stats.loss = avg_loss
        stats.avg_reward_100 = logger.avg_rewards[-1]
        agent.episodes_done = episode

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

        if episode % config.eval_freq == 0:
            eval_seed_base = config.seed + stats.total_steps
            eval_reward = evaluate(
                agent,
                config,
                num_eval=config.eval_episodes,
                seed_base=eval_seed_base,
            )
            logger.log_evaluation(stats.total_steps, eval_reward)
            print(f"Evaluation reward: {eval_reward:.2f}")

            if eval_reward > stats.best_eval_reward:
                stats.best_eval_reward = eval_reward
                best_path = os.path.join(config.save_dir, f"{model_prefix}_best.pth")
                agent.save(best_path, save_replay_buffer=config.save_replay_buffer)

        if stop_training:
            print(f"Reached max training steps: {config.max_steps}")
            break

    final_path = os.path.join(config.save_dir, f"{model_prefix}_final.pth")
    agent.save(final_path, save_replay_buffer=config.save_replay_buffer)

    logger.save_metrics(os.path.join(config.log_dir, "metrics.json"))
    logger.plot_training_curves(os.path.join(config.log_dir, "training_curves.png"))

    logger.close()
    env.close()

    print("-" * 80)
    print("Training finished!")
    print(f"Best evaluation reward: {stats.best_eval_reward:.2f}")
    print(f"Model saved to: {config.save_dir}")


def evaluate(
    agent: DQNAgent,
    config: DQNConfig,
    num_eval: int = 10,
    seed_base: int | None = None,
) -> float:
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
        if seed_base is None:
            state, _ = eval_env.reset()
        else:
            state, _ = eval_env.reset(seed=seed_base + eval_idx)
        episode_reward = 0
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
