"""使用训练好的模型进行演示或录制视频。"""

from __future__ import annotations

import os
import time

import numpy as np

from config import DQNConfig
from dqn.agent import DQNAgent
from dqn.env import make_env


def infer_model_metadata(
    model_path: str | None,
    default_channels: int,
    default_frame_size: tuple[int, int],
) -> tuple[int, tuple[int, int]]:
    """优先从 checkpoint 推断输入配置，失败时回退到默认值。"""
    if not model_path or not os.path.exists(model_path):
        return default_channels, default_frame_size

    try:
        import torch

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        model_config = checkpoint.get("model_config", {})

        input_channels = int(model_config.get("input_channels", default_channels))
        input_shape = model_config.get("input_shape", default_frame_size)
        if isinstance(input_shape, list):
            input_shape = tuple(input_shape)
        if not (isinstance(input_shape, tuple) and len(input_shape) == 2):
            input_shape = default_frame_size

        policy_state = checkpoint.get("policy_net", {})
        conv1_weight = policy_state.get("conv1.weight")
        if conv1_weight is not None and conv1_weight.ndim == 4:
            input_channels = int(conv1_weight.shape[1])

        return input_channels, tuple(int(v) for v in input_shape)
    except Exception as exc:
        print(f"Warning: failed to infer model metadata from checkpoint: {exc}")

    return default_channels, default_frame_size


def play(
    env_name: str = "ALE/Pong-v5",
    model_path: str | None = None,
    num_episodes: int = 5,
    render: bool = True,
    delay: float = 0.02,
    device: str = "auto",
    frame_stack: int = 4,
    frame_size: tuple[int, int] = (84, 84),
):
    """运行若干回合，用于观察训练后策略的表现。"""
    input_channels, input_shape = infer_model_metadata(
        model_path, frame_stack, frame_size
    )
    render_mode = "human" if render else None
    env = make_env(
        env_name,
        render_mode=render_mode,
        frame_stack=input_channels,
        frame_size=input_shape,
        clip_reward=False,
        terminal_on_life_loss=False,
    )

    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    num_actions = env.action_space.n
    print(f"Environment: {env_name}")
    print(f"Num actions: {num_actions}")
    print(f"Input channels: {input_channels}")
    print(f"Input shape: {input_shape}")

    agent = DQNAgent(
        num_actions=num_actions,
        input_channels=input_channels,
        input_shape=input_shape,
        device=device,
    )

    if model_path:
        agent.load(model_path)
        print(f"Loaded model: {model_path}")
    else:
        print("No model provided, using the untrained policy network.")

    agent.eval_mode()
    total_rewards = []

    print("\nStarting playback...")
    print("-" * 50)

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            if render:
                time.sleep(delay)

            action = agent.select_action(state, evaluate=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

        total_rewards.append(episode_reward)
        print(
            f"Episode {episode}: reward={episode_reward:.1f}, length={episode_length}"
        )

    env.close()

    print("-" * 50)
    print(f"Average reward: {np.mean(total_rewards):.1f}")
    print(f"Std reward: {np.std(total_rewards):.1f}")
    print(f"Min reward: {np.min(total_rewards):.1f}")
    print(f"Max reward: {np.max(total_rewards):.1f}")


def record_video(
    env_name: str = "ALE/Pong-v5",
    model_path: str | None = None,
    num_episodes: int = 3,
    output_dir: str = "videos",
    fps: int = 30,
    frame_stack: int = 4,
    frame_size: tuple[int, int] = (84, 84),
):
    """把智能体的行为录制成 mp4 视频。"""
    import cv2

    os.makedirs(output_dir, exist_ok=True)
    input_channels, input_shape = infer_model_metadata(
        model_path, frame_stack, frame_size
    )

    env = make_env(
        env_name,
        render_mode="rgb_array",
        frame_stack=input_channels,
        frame_size=input_shape,
        clip_reward=False,
        terminal_on_life_loss=False,
    )

    agent = DQNAgent(
        num_actions=env.action_space.n,
        input_channels=input_channels,
        input_shape=input_shape,
        device="cpu",
    )

    if model_path:
        agent.load(model_path)

    agent.eval_mode()

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        frames = []
        done = False
        episode_reward = 0

        while not done:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

            action = agent.select_action(state, evaluate=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        if frames:
            safe_name = env_name.replace("/", "_").replace(":", "_")
            video_path = os.path.join(
                output_dir, f"{safe_name}_ep{episode}_reward{episode_reward:.0f}.mp4"
            )

            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

            for frame in frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            out.release()
            print(f"Saved {video_path} (reward: {episode_reward:.1f})")

    env.close()


def main():
    config = DQNConfig()
    model_path = os.path.join(config.save_dir, f"{config.model_name_prefix()}_best.pth")

    if config.save_video:
        record_video(
            env_name=config.env_name,
            model_path=model_path,
            num_episodes=config.eval_episodes,
            output_dir=config.video_dir,
            frame_stack=config.frame_stack,
            frame_size=config.frame_size,
        )
    else:
        play(
            env_name=config.env_name,
            model_path=model_path,
            num_episodes=config.eval_episodes,
            render=config.render,
            frame_stack=config.frame_stack,
            frame_size=config.frame_size,
        )


if __name__ == "__main__":
    main()
