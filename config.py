"""Atari DQN 项目的全局配置。"""

from dataclasses import dataclass


@dataclass
class DQNConfig:
    """DQN 训练与运行时使用的超参数配置。"""

    # 环境相关配置。
    env_name: str = "ALE/Pong-v5"
    seed: int = 42

    # 训练相关配置。
    num_episodes: int = 1000
    max_steps: int = 1_000_000
    batch_size: int = 32
    learning_rate: float = 2.5e-4
    gamma: float = 0.99

    # 经验回放相关配置。
    buffer_size: int = 100_000
    min_buffer_size: int = 10_000
    use_per: bool = False
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 100_000

    # 探索策略相关配置。
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 1_000_000

    # 目标网络更新配置。
    target_update_freq: int = 10_000
    soft_update: bool = False
    tau: float = 0.005

    # 模型保存与日志配置。
    save_dir: str = "models"
    save_freq: int = 100
    log_dir: str = "runs"
    save_replay_buffer: bool = False

    # 评估配置。
    eval_freq: int = 50
    eval_episodes: int = 10

    # 设备配置。
    device: str = "auto"

    # Atari 预处理配置。
    frame_stack: int = 4
    frame_size: tuple[int, int] = (84, 84)
    noop_max: int = 30
    clip_reward: bool = True
    frame_skip: int = 4
    terminal_on_life_loss: bool = False

    # 播放与视频录制配置。
    render: bool = False
    save_video: bool = False
    video_dir: str = "videos"

    def get_device(self) -> str:
        """自动解析训练设备。"""
        if self.device == "auto":
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def model_name_prefix(self) -> str:
        """把环境名转成适合保存到文件系统中的前缀。"""
        return self.env_name.replace("/", "_").replace(":", "_")


@dataclass
class TrainingStats:
    """记录训练过程中的高层统计信息。"""

    episode: int = 0
    total_steps: int = 0
    episode_reward: float = 0.0
    episode_length: int = 0
    epsilon: float = 1.0
    loss: float = 0.0
    avg_reward_100: float = 0.0
    best_eval_reward: float = -float("inf")

    def reset_episode(self):
        """重置按 episode 统计的字段。"""
        self.episode_reward = 0.0
        self.episode_length = 0


DEFAULT_CONFIG = DQNConfig()
