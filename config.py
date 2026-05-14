from dataclasses import dataclass
from datetime import datetime
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
    env_id: str = "Taxi-v4"
    episodes: int = 3000
    max_steps_per_episode: int = 5000
    final_test_episodes: int = 30
    final_test_seed_offset: int = 20_000
    eval_interval_episodes: int = 50
    eval_episodes: int = 10
    batch_size: int = 256
    seed: int = 37
    gamma: float = 0.99
    learning_rate: float = 5e-4
    hidden_dim: int = 256
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 200000
    replay_buffer_size: int = 50_000
    min_replay_size: int = 2000
    target_update_freq: int = 1000
    soft_target_update: bool = False
    target_update_tau: float = 0.005
    success_threshold: int = 1
    render_fps: int = 30
    moving_average_window: int = 50
    gradient_clip_norm: float = 10.0
    use_double_dqn: bool = False
    warmup_steps: int = 0


@dataclass(frozen=True, slots=True)
class PerConfig:
    alpha: float = 0.5
    beta_start: float = 0.4
    beta_anneal_steps: int | None = 180000
    priority_epsilon: float = 1e-4


ENV_CONFIGS: Mapping[str, Config] = MappingProxyType(
    {
        "taxi": Config(
            env_id="Taxi-v4",
            episodes=2000,
            max_steps_per_episode=200,
            final_test_episodes=50,
            eval_interval_episodes=50,
            eval_episodes=10,
            batch_size=64,
            gamma=0.99,
            learning_rate=5e-4,
            hidden_dim=128,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_steps=20000,
            replay_buffer_size=30000,
            min_replay_size=500,
            target_update_freq=500,
            success_threshold=1,
            render_fps=8,
        ),
        "mountaincar": Config(
            env_id="MountainCar-v0",
            episodes=2000,
            max_steps_per_episode=500,
            final_test_episodes=30,
            eval_interval_episodes=50,
            eval_episodes=8,
            batch_size=128,
            gamma=0.99,
            learning_rate=5e-4,
            hidden_dim=128,
            epsilon_start=1.0,
            epsilon_end=0.02,
            epsilon_decay_steps=30000,
            replay_buffer_size=100000,
            min_replay_size=5000,
            target_update_freq=1000,
            success_threshold=1,
            render_fps=30,
        ),
        "dino": Config(
            env_id="TrexEnv-v0",
            episodes=3000,
            max_steps_per_episode=8000,
            final_test_episodes=30,
            eval_interval_episodes=50,
            eval_episodes=15,
            batch_size=256,
            gamma=0.99,
            learning_rate=1e-5,
            hidden_dim=256,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_steps=500000,
            replay_buffer_size=200000,
            min_replay_size=5000,
            target_update_freq=2000,
            soft_target_update=True,
            target_update_tau=0.001,
            success_threshold=20,
            render_fps=30,
            moving_average_window=30,
            gradient_clip_norm=5.0,
            warmup_steps=10000,
            use_double_dqn=False
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


@dataclass(frozen=True, slots=True)
class RunDirs:
    run_dir: Path
    logs_dir: Path
    models_dir: Path
    figures_dir: Path


def build_run_name(env_name: str, algo_name: str, seed: int | None = None) -> str:
    timestamp = datetime.now().strftime("%m%d-%H%M")
    seed_part = "" if seed is None else f"-seed{seed}"
    return f"{timestamp}-{env_name}-{algo_name}{seed_part}"


def create_run_dirs(env_name: str, algo_name: str, seed: int | None = None, run_name: str | None = None) -> RunDirs:
    ensure_result_dirs()
    resolved_run_name = run_name or build_run_name(env_name, algo_name, seed)
    run_dir = RESULTS_DIR / resolved_run_name
    logs_dir = run_dir / "logs"
    models_dir = run_dir / "models"
    figures_dir = run_dir / "figures"

    for path in (run_dir, logs_dir, models_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    return RunDirs(run_dir=run_dir, logs_dir=logs_dir, models_dir=models_dir, figures_dir=figures_dir)
