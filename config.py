from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = RESULTS_DIR / "models"
LOGS_DIR = RESULTS_DIR / "logs"
FIGURES_DIR = RESULTS_DIR / "figures"


@dataclass(frozen=True)
class CommonConfig:
    # 强化学习通用超参数，统一作用于三个环境。
    gamma: float = 0.99
    learning_rate: float = 1e-3
    batch_size: int = 64
    replay_buffer_size: int = 50_000
    target_update_freq: int = 800
    hidden_dim: int = 128
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    seed: int = 42
    test_episodes: int = 20
    moving_average_window: int = 50
    gradient_clip_norm: float = 10.0


@dataclass(frozen=True)
class PerConfig:
    # PER-DQN 的优先采样控制参数。
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_increment: float = 0.0005
    priority_epsilon: float = 1e-5


ENV_CONFIGS = {
    "taxi": {
        "env_id": "Taxi-v3",
        "episodes": 1500,
        "max_steps": 200,
        "min_replay_size": 500,
        "epsilon_decay": 0.997,
        "success_threshold": 1,
        "render_fps": 8,
    },
    "mountaincar": {
        "env_id": "MountainCar-v0",
        "episodes": 2000,
        "max_steps": 200,
        "min_replay_size": 1000,
        "epsilon_decay": 0.998,
        "success_threshold": 1,
        "render_fps": 30,
    },
    "dino": {
        "env_id": "DinoEnv",
        "episodes": 1800,
        "max_steps": 800,
        "min_replay_size": 1000,
        "epsilon_decay": 0.997,
        "success_threshold": 15,
        "render_fps": 30,
    },
}


COMMON_CONFIG = CommonConfig()
PER_CONFIG = PerConfig()


def ensure_result_dirs() -> None:
    for path in (RESULTS_DIR, MODELS_DIR, LOGS_DIR, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)
