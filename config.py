"""Global configuration for Atari DQN experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DQNConfig:
    """Configuration used by training, evaluation, and playback."""

    # Environment configuration.
    env_name: str = "ALE/Pong-v5"
    seed: int = 42

    # Nature DQN is primarily constrained by frame budget, not episode count.
    num_episodes: int | None = None
    max_steps: int = 20_000_000
    batch_size: int = 32
    learning_rate: float = 2.5e-4
    gamma: float = 0.99

    # Optimizer settings. The original paper used a centered RMSProp variant.
    optimizer_name: str = "rmsprop"
    rmsprop_alpha: float = 0.95
    rmsprop_eps: float = 0.01
    rmsprop_centered: bool = True

    # Replay and update schedule.
    buffer_size: int = 1_000_000
    # Number of environment steps that must be collected before SGD updates start.
    # Set this equal to initial_random_steps to match the classic DQN warmup.
    training_start_steps: int = 100_000
    train_freq: int = 4
    gradient_steps: int = 1
    use_per: bool = False
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    # Annealed by replay-sampling updates, not by environment frames.
    per_beta_updates: int = 100_000
    use_double_dqn: bool = False

    # Exploration.
    # Collect uniformly random experience first, then start epsilon-greedy from epsilon_start.
    initial_random_steps: int = 100_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: int = 1_000_000

    # Target network updates.
    # Hard target sync cadence measured in optimizer updates, not env steps.
    target_update_interval_updates: int = 10_000
    use_soft_target_update: bool = False
    # Only used when use_soft_target_update=True.
    soft_target_update_tau: float = 0.005

    # Model saving and logging.
    save_dir: str = "models"
    save_freq: int = 100
    log_dir: str = "runs"
    log_step_metrics: bool = True
    step_log_interval: int = 1_000
    step_log_stream: bool = True
    step_log_file: str = "step_metrics.jsonl"

    # Evaluation. Nature DQN evaluates with epsilon=0.05 and a frame cap.
    # Preferred cadence measured in environment steps.
    eval_interval_env_steps: int = 250_000
    # Fallback cadence used only when eval_interval_env_steps <= 0.
    eval_interval_episodes: int = 50
    eval_episodes: int = 30
    eval_epsilon: float = 0.05
    eval_seed_offset: int = 100_000
    eval_max_episode_steps: int | None = 4_500

    # Device configuration.
    device: str = "auto"
    deterministic_torch: bool = False
    allow_tf32: bool = True
    replay_sample_torch_fastpath: bool = True

    # Atari preprocessing.
    frame_stack: int = 4
    frame_size: tuple[int, int] = (84, 84)
    noop_max: int = 30
    clip_reward: bool = True
    frame_skip: int = 4
    terminal_on_life_loss: bool = True

    # Playback and video recording.
    render: bool = False
    save_video: bool = True
    video_dir: str = "videos"

    def get_device(self) -> str:
        """Resolve the training device automatically when requested."""
        if self.device == "auto":
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def model_name_prefix(self) -> str:
        """Convert the environment name into a filesystem-safe model filename prefix."""
        return self.env_name.replace("/", "_").replace(":", "_")


@dataclass
class TrainingStats:
    """High-level statistics tracked during training."""

    episode: int = 0
    total_steps: int = 0
    episode_reward: float = 0.0
    episode_length: int = 0
    epsilon: float = 1.0
    loss: float = 0.0
    avg_reward_100: float = 0.0
    best_eval_reward: float = -float("inf")


DEFAULT_CONFIG = DQNConfig()
