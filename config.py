"""Global configuration for low-dimensional DQN experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass
class CoreConfig:
    """Run identity and global training budget."""

    env_name: str = "CartPole-v1"
    env_family: Literal["auto", "vector", "discrete", "custom"] = "auto"
    seed: int = 42
    num_episodes: int | None = None
    max_steps: int = 50_000


@dataclass
class TrainingConfig:
    """Optimization hyperparameters and target-network behavior."""

    batch_size: int = 64
    learning_rate: float = 1e-3
    gamma: float = 0.99
    optimizer_name: str = "adam"
    rmsprop_alpha: float = 0.95
    rmsprop_eps: float = 0.01
    rmsprop_centered: bool = True
    train_freq: int = 1
    gradient_steps: int = 1
    use_double_dqn: bool = False
    target_update_interval_updates: int = 500
    use_soft_target_update: bool = False
    soft_target_update_tau: float = 0.005
    hidden_sizes: tuple[int, ...] = (128, 128)


@dataclass
class ReplayConfig:
    """Replay buffer and exploration schedule."""

    buffer_size: int = 50_000
    training_start_steps: int = 1_000
    use_per: bool = False
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_updates: int = 50_000
    initial_random_steps: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 20_000
    replay_sample_torch_fastpath: bool = True
    obs_encoding: Literal["auto", "identity", "one_hot", "feature_vector"] = "auto"


@dataclass
class LoggingConfig:
    """Artifact output and training logging settings."""

    save_dir: str = "models"
    save_freq: int = 50
    log_dir: str = "runs"
    log_step_metrics: bool = True
    step_log_interval: int = 200
    step_log_stream: bool = True
    step_log_file: str = "step_metrics.jsonl"


@dataclass
class EvalConfig:
    """Evaluation protocol configuration."""

    eval_interval_env_steps: int = 5_000
    eval_interval_episodes: int = 20
    eval_episodes: int = 10
    eval_epsilon: float = 0.05
    eval_seed_offset: int = 100_000
    eval_max_episode_steps: int | None = 1_000
    success_threshold: float | None = None


@dataclass
class DeviceConfig:
    """Device and determinism knobs."""

    device: str = "auto"
    deterministic_torch: bool = False
    allow_tf32: bool = True


@dataclass
class PlaybackConfig:
    """Playback and recording defaults."""

    render: bool = False
    save_video: bool = False
    video_dir: str = "videos"


@dataclass
class DQNConfig:
    """Configuration used by training, evaluation, and playback."""

    core: CoreConfig = field(default_factory=CoreConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    device_config: DeviceConfig = field(default_factory=DeviceConfig)
    playback: PlaybackConfig = field(default_factory=PlaybackConfig)

    def clone(self) -> "DQNConfig":
        return DQNConfig(
            core=CoreConfig(**asdict(self.core)),
            training=TrainingConfig(**asdict(self.training)),
            replay=ReplayConfig(**asdict(self.replay)),
            logging=LoggingConfig(**asdict(self.logging)),
            evaluation=EvalConfig(**asdict(self.evaluation)),
            device_config=DeviceConfig(**asdict(self.device_config)),
            playback=PlaybackConfig(**asdict(self.playback)),
        )

    def to_flat_dict(self) -> dict[str, object]:
        flat: dict[str, object] = {}
        for section in (
            self.core,
            self.training,
            self.replay,
            self.logging,
            self.evaluation,
            self.device_config,
            self.playback,
        ):
            flat.update(asdict(section))
        return flat

    def get_device(self) -> str:
        """Resolve the training device automatically when requested."""
        if self.device_config.device == "auto":
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device_config.device

    def model_name_prefix(self) -> str:
        """Convert the environment name into a filesystem-safe model filename prefix."""
        return self.core.env_name.replace("/", "_").replace(":", "_")


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
    best_eval_success_rate: float = 0.0


DEFAULT_CONFIG = DQNConfig()
