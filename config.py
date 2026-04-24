from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = RESULTS_DIR / "models"
LOGS_DIR = RESULTS_DIR / "logs"
FIGURES_DIR = RESULTS_DIR / "figures"


@dataclass(frozen=True, slots=True)
class Config:
    env_id: str = "Taxi-v3"
    episodes: int = 1500
    test_episodes: int = 20
    eval_interval_episodes: int = 50
    eval_episodes: int = 10
    max_steps_per_episode: int = 200
    batch_size: int = 64
    seed: int = 42
    gamma: float = 0.99
    learning_rate: float = 5e-4
    hidden_dim: int = 128
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10000
    replay_buffer_size: int = 50_000
    min_replay_size: int = 500
    target_update_freq: int = 1000
    success_threshold: int = 1
    render_fps: int = 30
    moving_average_window: int = 50
    gradient_clip_norm: float = 10.0


@dataclass(frozen=True, slots=True)
class PerConfig:
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_increment: float = 0.00005
    priority_epsilon: float = 1e-5


ENV_CONFIGS: Mapping[str, Config] = MappingProxyType(
    {
        "taxi": Config(
            env_id="Taxi-v3",
            episodes=1800,
            eval_interval_episodes=50,
            eval_episodes=10,
            max_steps_per_episode=200,
            batch_size=64,
            gamma=0.99,
            learning_rate=3e-4,
            hidden_dim=128,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_steps=8000,
            replay_buffer_size=30_000,
            min_replay_size=500,
            target_update_freq=500,
            success_threshold=1,
            render_fps=8,
            moving_average_window=50,
            gradient_clip_norm=10.0,
        ),
        "mountaincar": Config(
            env_id="MountainCar-v0",
            episodes=3000,
            eval_interval_episodes=100,
            eval_episodes=8,
            max_steps_per_episode=200,
            batch_size=128,
            gamma=0.99,
            learning_rate=1e-3,
            hidden_dim=128,
            epsilon_start=1.0,
            epsilon_end=0.02,
            epsilon_decay_steps=20000,
            replay_buffer_size=100_000,
            min_replay_size=2000,
            target_update_freq=1000,
            success_threshold=1,
            render_fps=30,
            moving_average_window=100,
            gradient_clip_norm=10.0,
        ),
        "dino": Config(
            env_id="TrexEnv-v0",
            episodes=2000,
            eval_interval_episodes=30,
            eval_episodes=5,
            max_steps_per_episode=8000,
            batch_size=128,
            gamma=0.995,
            learning_rate=1e-4,
            hidden_dim=256,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_steps=30000,
            replay_buffer_size=100_000,
            min_replay_size=5000,
            target_update_freq=2000,
            success_threshold=15,
            render_fps=30,
            moving_average_window=30,
            gradient_clip_norm=5.0,
        ),
    }
)


COMMON_CONFIG = Config()
PER_CONFIG = PerConfig()


def supported_envs() -> tuple[str, ...]:
    return tuple(ENV_CONFIGS.keys())


def get_env_config(env_name: str) -> Config:
    env_config = ENV_CONFIGS.get(env_name)
    if env_config is None:
        raise ValueError(f"不支持的环境名称: {env_name}，可选值为 {supported_envs()}")
    return env_config


def ensure_result_dirs() -> None:
    for path in (RESULTS_DIR, MODELS_DIR, LOGS_DIR, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)
