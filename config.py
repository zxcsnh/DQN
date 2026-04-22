"""Global configuration for Atari DQN experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class CoreConfig:
    """Run identity and global training budget."""

    env_name: str = "ALE/Pong-v5"
    seed: int = 42
    num_episodes: int | None = None
    max_steps: int = 20_000_000


@dataclass
class TrainingConfig:
    """Optimization hyperparameters and target-network behavior."""

    batch_size: int = 32
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    optimizer_name: str = "rmsprop"
    rmsprop_alpha: float = 0.95
    rmsprop_eps: float = 0.01
    rmsprop_centered: bool = True
    train_freq: int = 4
    gradient_steps: int = 1
    use_double_dqn: bool = False
    target_update_interval_updates: int = 10_000
    use_soft_target_update: bool = False
    soft_target_update_tau: float = 0.005


@dataclass
class ReplayConfig:
    """Replay buffer and exploration schedule."""

    buffer_size: int = 1_000_000
    training_start_steps: int = 100_000
    use_per: bool = False
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_updates: int = 100_000
    initial_random_steps: int = 100_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: int = 1_000_000
    replay_sample_torch_fastpath: bool = True


@dataclass
class LoggingConfig:
    """Artifact output and training logging settings."""

    save_dir: str = "models"
    save_freq: int = 100
    log_dir: str = "runs"
    log_step_metrics: bool = True
    step_log_interval: int = 1_000
    step_log_stream: bool = True
    step_log_file: str = "step_metrics.jsonl"


@dataclass
class EvalConfig:
    """Evaluation protocol configuration."""

    eval_interval_env_steps: int = 250_000
    eval_interval_episodes: int = 50
    eval_episodes: int = 30
    eval_epsilon: float = 0.05
    eval_seed_offset: int = 100_000
    eval_max_episode_steps: int | None = 4_500


@dataclass
class DeviceConfig:
    """Device and determinism knobs."""

    device: str = "auto"
    deterministic_torch: bool = False
    allow_tf32: bool = True


@dataclass
class PreprocessConfig:
    """Atari preprocessing protocol shared by train/eval/play."""

    frame_stack: int = 4
    frame_size: tuple[int, int] = (84, 84)
    noop_max: int = 30
    clip_reward: bool = True
    frame_skip: int = 4
    terminal_on_life_loss: bool = True


@dataclass
class PlaybackConfig:
    """Playback and recording defaults."""

    render: bool = False
    save_video: bool = True
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
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    playback: PlaybackConfig = field(default_factory=PlaybackConfig)

    def clone(self) -> "DQNConfig":
        return DQNConfig(
            core=CoreConfig(**asdict(self.core)),
            training=TrainingConfig(**asdict(self.training)),
            replay=ReplayConfig(**asdict(self.replay)),
            logging=LoggingConfig(**asdict(self.logging)),
            evaluation=EvalConfig(**asdict(self.evaluation)),
            device_config=DeviceConfig(**asdict(self.device_config)),
            preprocess=PreprocessConfig(**asdict(self.preprocess)),
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
            self.preprocess,
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


DEFAULT_CONFIG = DQNConfig()
