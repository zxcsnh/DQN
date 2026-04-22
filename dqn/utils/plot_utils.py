from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import FIGURES_DIR


REQUIRED_COLUMNS = {"episode", "total_reward", "steps", "epsilon", "loss", "success", "custom_metric"}


def _validate_log_file(csv_path: Path) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到日志文件: {csv_path}")
    with csv_path.open("r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"日志文件为空或表头缺失: {csv_path}")
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"日志文件缺少字段 {missing}: {csv_path}")


def _read_column(csv_path: Path, column: str) -> np.ndarray:
    _validate_log_file(csv_path)
    values = []
    with csv_path.open("r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            values.append(float(row[column]))
    return np.asarray(values, dtype=np.float32)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0 or len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float32) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    prefix = values[: window - 1]
    return np.concatenate([prefix, smoothed])


def plot_comparison(log_path_a: Path, log_path_b: Path, env_name: str, metric: str, output_name: str, window: int = 50) -> None:
    # 使用滑动平均让论文中的训练曲线更平滑，更便于观察趋势。
    values_a = moving_average(_read_column(log_path_a, metric), window)
    values_b = moving_average(_read_column(log_path_b, metric), window)

    plt.figure(figsize=(8, 5))
    plt.plot(values_a, label="DQN")
    plt.plot(values_b, label="PER-DQN")
    plt.xlabel("Episode")
    plt.ylabel(metric)
    plt.title(f"{env_name} {metric} comparison")
    plt.legend()
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / output_name)
    plt.close()


def plot_env_comparisons(env_name: str, dqn_log: Path, perdqn_log: Path, window: int = 50) -> None:
    plot_comparison(dqn_log, perdqn_log, env_name, "total_reward", f"{env_name}_reward_comparison.png", window)
    plot_comparison(dqn_log, perdqn_log, env_name, "steps", f"{env_name}_steps_comparison.png", window)
    plot_comparison(dqn_log, perdqn_log, env_name, "loss", f"{env_name}_loss_comparison.png", window)
    metric_name = "custom_metric" if env_name == "dino" else "success"
    suffix = "obstacles" if env_name == "dino" else "success_rate"
    plot_comparison(dqn_log, perdqn_log, env_name, metric_name, f"{env_name}_{suffix}_comparison.png", window)
