from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import FIGURES_DIR


REQUIRED_COLUMNS = {"episode", "total_reward", "steps", "epsilon", "loss", "success", "custom_metric"}

def comparison_metrics_for_env(env_name: str) -> list[tuple[str, str]]:
    metrics = [
        ("total_reward", "reward"),
        ("reward_per_step", "reward_per_step"),
        ("steps", "steps"),
        ("loss", "loss"),
        ("success", "success"),
        ("eval_avg_reward", "eval_reward"),
        ("eval_success_rate", "eval_success_rate"),
    ]
    if env_name == "dino":
        metrics.extend(
            [
                ("custom_metric", "obstacles"),
                ("score", "score"),
                ("speed", "speed"),
                ("obstacles_cleared", "obstacles_cleared"),
            ]
        )
    else:
        metrics.append(("custom_metric", "custom_metric"))
    return metrics


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


def _has_column(csv_path: Path, column: str) -> bool:
    with csv_path.open("r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return reader.fieldnames is not None and column in reader.fieldnames


def _read_column(csv_path: Path, column: str, skip_blank: bool = False) -> np.ndarray:
    values = []
    with csv_path.open("r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or column not in reader.fieldnames:
            return np.asarray(values, dtype=np.float32)
        for row in reader:
            value = row.get(column, "")
            if value == "" or value is None:
                if skip_blank:
                    continue
                value = 0.0
            values.append(float(value))
    return np.asarray(values, dtype=np.float32)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0 or len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float32) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    prefix = values[: window - 1]
    return np.concatenate([prefix, smoothed])


def _save_figure(name: str, figures_dir: Path | None = None) -> None:
    target_dir = FIGURES_DIR if figures_dir is None else figures_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(target_dir / name)
    plt.close()


def plot_single_run(csv_path: Path, env_name: str, algo_name: str, window: int = 50, figures_dir: Path | None = None) -> None:
    _validate_log_file(csv_path)

    metrics = ["total_reward", "reward_per_step", "steps", "loss"]
    if env_name == "dino":
        metrics.extend(["custom_metric", "score", "speed", "obstacles_cleared"])
    else:
        metrics.append("success")
    metrics.extend(["eval_avg_reward", "eval_success_rate"])

    for metric in metrics:
        if not _has_column(csv_path, metric):
            continue
        values = moving_average(_read_column(csv_path, metric, skip_blank=metric.startswith("eval_")), window)
        if len(values) == 0:
            continue
        plt.figure(figsize=(8, 5))
        plt.plot(values, label=f"{algo_name.upper()}")
        plt.xlabel("Episode")
        plt.ylabel(metric)
        plt.title(f"{env_name} — {algo_name.upper()} — {metric}")
        plt.legend()
        plt.tight_layout()
        suffix = "obstacles" if metric == "custom_metric" else metric
        _save_figure(f"{env_name}_{algo_name}_{suffix}.png", figures_dir=figures_dir)


def _plot_comparison_impl(
    log_path_a: Path,
    log_path_b: Path,
    env_name: str,
    metric: str,
    output_name: str,
    window: int,
    figures_dir: Path | None = None,
) -> None:
    plot_metric_comparison(
        env_name=env_name,
        log_paths={"DQN": log_path_a, "PER-DQN": log_path_b},
        metric=metric,
        output_name=output_name,
        window=window,
        figures_dir=figures_dir,
    )


def plot_metric_comparison(
    env_name: str,
    log_paths: dict[str, Path],
    metric: str,
    output_name: str,
    window: int = 50,
    figures_dir: Path | None = None,
) -> None:
    plotted = False
    plt.figure(figsize=(8, 5))
    for label, path in log_paths.items():
        if not path.exists() or not _has_column(path, metric):
            continue
        values = _read_column(path, metric, skip_blank=metric.startswith("eval_"))
        if len(values) == 0:
            continue
        values = moving_average(values, window)
        plt.plot(values, label=label)
        plotted = True

    if not plotted:
        plt.close()
        return
    plt.xlabel("Episode" if not metric.startswith("eval_") else "Evaluation point")
    plt.ylabel(metric)
    plt.title(f"{env_name} {metric} comparison")
    plt.legend()
    plt.tight_layout()
    _save_figure(output_name, figures_dir=figures_dir)

def _aligned_series(series: list[np.ndarray]) -> np.ndarray:
    non_empty = [values for values in series if len(values) > 0]
    if not non_empty:
        return np.empty((0, 0), dtype=np.float32)
    min_len = min(len(values) for values in non_empty)
    if min_len == 0:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack([values[:min_len] for values in non_empty]).astype(np.float32)

def plot_multiseed_metric_comparison(
    env_name: str,
    log_paths: dict[str, list[Path]],
    metric: str,
    output_name: str,
    window: int = 50,
    figures_dir: Path | None = None,
) -> None:
    plotted = False
    plt.figure(figsize=(8, 5))
    for label, paths in log_paths.items():
        seed_series = []
        for path in paths:
            if not path.exists() or not _has_column(path, metric):
                continue
            values = _read_column(path, metric, skip_blank=metric.startswith("eval_"))
            if len(values) == 0:
                continue
            seed_series.append(moving_average(values, window))

        stacked = _aligned_series(seed_series)
        if stacked.size == 0:
            continue

        mean_values = stacked.mean(axis=0)
        std_values = stacked.std(axis=0)
        x_values = np.arange(1, len(mean_values) + 1)
        plt.plot(x_values, mean_values, label=f"{label} (n={stacked.shape[0]})")
        plt.fill_between(x_values, mean_values - std_values, mean_values + std_values, alpha=0.18)
        plotted = True

    if not plotted:
        plt.close()
        return
    plt.xlabel("Episode" if not metric.startswith("eval_") else "Evaluation point")
    plt.ylabel(metric)
    plt.title(f"{env_name} {metric} multi-seed comparison")
    plt.legend()
    plt.tight_layout()
    _save_figure(output_name, figures_dir=figures_dir)

def plot_env_multiseed_comparisons(
    env_name: str,
    dqn_logs: list[Path],
    perdqn_logs: list[Path],
    window: int = 50,
    figures_dir: Path | None = None,
) -> None:
    for log_path in [*dqn_logs, *perdqn_logs]:
        _validate_log_file(log_path)

    log_paths = {"DQN": dqn_logs, "PER-DQN": perdqn_logs}
    for metric, suffix in comparison_metrics_for_env(env_name):
        plot_multiseed_metric_comparison(
            env_name=env_name,
            log_paths=log_paths,
            metric=metric,
            output_name=f"{env_name}_{suffix}_multiseed_comparison.png",
            window=window,
            figures_dir=figures_dir,
        )


def plot_multiseed_single_algo_metric(
    env_name: str,
    algo_name: str,
    log_paths: list[Path],
    metric: str,
    output_name: str,
    window: int = 50,
    figures_dir: Path | None = None,
) -> None:
    seed_series = []
    for path in log_paths:
        if not path.exists() or not _has_column(path, metric):
            continue
        values = _read_column(path, metric, skip_blank=metric.startswith("eval_"))
        if len(values) == 0:
            continue
        seed_series.append(moving_average(values, window))

    stacked = _aligned_series(seed_series)
    if stacked.size == 0:
        return

    mean_values = stacked.mean(axis=0)
    std_values = stacked.std(axis=0)
    x_values = np.arange(1, len(mean_values) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(x_values, mean_values, label=f"{algo_name.upper()} (n={stacked.shape[0]})")
    plt.fill_between(x_values, mean_values - std_values, mean_values + std_values, alpha=0.18)
    plt.xlabel("Episode" if not metric.startswith("eval_") else "Evaluation point")
    plt.ylabel(metric)
    plt.title(f"{env_name} {algo_name.upper()} {metric} multi-seed")
    plt.legend()
    plt.tight_layout()
    _save_figure(output_name, figures_dir=figures_dir)

def plot_env_multiseed_single_algo(
    env_name: str,
    algo_name: str,
    logs: list[Path],
    window: int = 50,
    figures_dir: Path | None = None,
) -> None:
    for log_path in logs:
        _validate_log_file(log_path)

    for metric, suffix in comparison_metrics_for_env(env_name):
        plot_multiseed_single_algo_metric(
            env_name=env_name,
            algo_name=algo_name,
            log_paths=logs,
            metric=metric,
            output_name=f"{env_name}_{algo_name}_{suffix}_multiseed.png",
            window=window,
            figures_dir=figures_dir,
        )

def plot_env_comparisons(
    env_name: str,
    dqn_log: Path,
    perdqn_log: Path,
    window: int = 50,
    figures_dir: Path | None = None,
) -> None:
    _validate_log_file(dqn_log)
    _validate_log_file(perdqn_log)

    log_paths = {"DQN": dqn_log, "PER-DQN": perdqn_log}
    for metric, suffix in comparison_metrics_for_env(env_name):
        plot_metric_comparison(
            env_name=env_name,
            log_paths=log_paths,
            metric=metric,
            output_name=f"{env_name}_{suffix}_comparison.png",
            window=window,
            figures_dir=figures_dir,
        )
