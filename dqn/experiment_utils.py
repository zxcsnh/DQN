from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from config import LOGS_DIR
from .evaluation import evaluate_random_policy
from .training import train


SUMMARY_FIELDNAMES = [
    "env_name",
    "algo_name",
    "seed",
    "episodes",
    "eval_episodes",
    "final_test_episodes",
    "best_train_reward",
    "best_eval_reward",
    "best_model_episode",
    "last_eval_avg_reward",
    "best_final_test_avg_reward",
    "best_final_test_std_reward",
    "best_final_test_success_rate",
    "best_final_test_avg_steps",
    "best_final_test_std_steps",
    "best_final_test_avg_custom_metric",
    "best_final_test_avg_score",
    "best_final_test_max_score",
    "best_final_test_avg_obstacles_cleared",
    "best_final_test_max_obstacles_cleared",
    "best_final_test_model_path",
    "final_model_test_avg_reward",
    "final_model_test_std_reward",
    "final_model_test_success_rate",
    "final_model_test_avg_steps",
    "final_model_test_std_steps",
    "final_model_test_avg_custom_metric",
    "final_model_test_avg_score",
    "final_model_test_max_score",
    "final_model_test_avg_obstacles_cleared",
    "final_model_test_max_obstacles_cleared",
    "final_model_test_model_path",
    "final_test_avg_reward",
    "final_test_std_reward",
    "final_test_min_reward",
    "final_test_max_reward",
    "final_test_median_reward",
    "final_test_avg_steps",
    "final_test_std_steps",
    "final_test_success_count",
    "final_test_success_rate",
    "final_test_avg_custom_metric",
    "final_test_std_custom_metric",
    "final_test_avg_reward_per_step",
    "final_test_avg_score",
    "final_test_max_score",
    "final_test_avg_speed",
    "final_test_avg_obstacles_cleared",
    "final_test_max_obstacles_cleared",
    "final_test_avg_max_position",
    "final_test_max_max_position",
    "log_final_reward",
    "log_final_steps",
    "log_final_success",
    "log_final_custom_metric",
    "log_window_avg_reward",
    "log_window_avg_steps",
    "log_window_success_rate",
    "log_window_avg_custom_metric",
    "best_model_path",
    "final_model_path",
    "log_path",
    "run_dir",
    "logs_dir",
    "models_dir",
    "figures_dir",
]


def _run_one_experiment(task: tuple[str, str, int], render: bool, plot_after_each_env: bool, run_dir: str | Path | None, run_final_test: bool) -> dict:
    env_name, algo_name, seed = task
    return train(
        env_name=env_name,
        algo_name=algo_name,
        render=render,
        plot_after_train=plot_after_each_env,
        seed=seed,
        log_name_suffix=f"seed{seed}",
        run_dir=run_dir,
        run_final_test=run_final_test,
    )


def run_batch_experiments(
    env_names: list[str],
    algo_names: list[str],
    seeds: list[int],
    render: bool = False,
    plot_after_each_env: bool = False,
    run_dir: str | Path | None = None,
    run_final_test: bool = True,
) -> list[dict]:
    tasks = [(env_name, algo_name, seed) for env_name in env_names for algo_name in algo_names for seed in seeds]
    return [_run_one_experiment(task, render, plot_after_each_env, run_dir, run_final_test) for task in tasks]


def _mean(rows: list[dict], column: str) -> float | str:
    values = []
    for row in rows:
        value = row.get(column, "")
        if value == "" or value is None:
            continue
        values.append(float(value))
    return "" if not values else float(np.mean(values))


def summarize_training_log(log_path: Path, window: int | None = None) -> dict:
    with log_path.open("r", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        return {"log_path": str(log_path)}

    window_size = window or min(100, len(rows))
    tail_rows = rows[-window_size:]
    final_row = rows[-1]
    eval_rows = [row for row in rows if row.get("eval_avg_reward", "") not in {"", None}]
    best_rows = [row for row in rows if row.get("is_best_model") == "1"]

    summary = {
        "log_path": str(log_path),
        "episodes": len(rows),
        "log_final_reward": final_row.get("total_reward", ""),
        "log_final_steps": final_row.get("steps", ""),
        "log_final_success": final_row.get("success", ""),
        "log_final_custom_metric": final_row.get("custom_metric", ""),
        "log_window_avg_reward": _mean(tail_rows, "total_reward"),
        "log_window_avg_steps": _mean(tail_rows, "steps"),
        "log_window_success_rate": _mean(tail_rows, "success"),
        "log_window_avg_custom_metric": _mean(tail_rows, "custom_metric"),
        "best_train_reward": max(float(row["total_reward"]) for row in rows),
    }
    if eval_rows:
        best_eval_row = max(eval_rows, key=lambda row: float(row["eval_avg_reward"]))
        summary["best_eval_reward"] = best_eval_row["eval_avg_reward"]
        summary["last_eval_avg_reward"] = eval_rows[-1]["eval_avg_reward"]
    if best_rows:
        summary["best_model_episode"] = best_rows[-1].get("episode", "")
    return summary


def _parse_log_name(log_path: Path) -> tuple[str, str, int | str]:
    name = log_path.name.removesuffix("_train_log.csv")
    parts = name.split("_")
    env_name = parts[0]
    algo_name = parts[1] if len(parts) > 1 else ""
    seed = ""
    for part in parts[2:]:
        if part.startswith("seed"):
            try:
                seed = int(part.removeprefix("seed"))
            except ValueError:
                seed = part.removeprefix("seed")
    return env_name, algo_name, seed


def generate_experiment_summary(
    logs_dir: str | Path = LOGS_DIR,
    output_path: str | Path | None = None,
    include_random_baseline: bool = False,
) -> Path:
    resolved_logs_dir = Path(logs_dir)
    rows = []
    for log_path in sorted(resolved_logs_dir.glob("*_train_log.csv")):
        if log_path.name == "experiment_summary.csv":
            continue
        env_name, algo_name, seed = _parse_log_name(log_path)
        row = summarize_training_log(log_path)
        row.update({"env_name": env_name, "algo_name": algo_name, "seed": seed})
        run_dir = resolved_logs_dir.parent
        row.update(
            {
                "run_dir": str(run_dir),
                "logs_dir": str(resolved_logs_dir),
                "models_dir": str(run_dir / "models"),
                "figures_dir": str(run_dir / "figures"),
            }
        )
        seed_suffix = f"_seed{seed}" if seed != "" else ""
        row["best_model_path"] = str(run_dir / "models" / f"{env_name}_{algo_name}{seed_suffix}_best.pth")
        row["final_model_path"] = str(run_dir / "models" / f"{env_name}_{algo_name}{seed_suffix}_final.pth")
        rows.append(row)

    if include_random_baseline:
        env_seed_pairs = sorted({(row["env_name"], row.get("seed", "")) for row in rows})
        for env_name, seed in env_seed_pairs:
            baseline = evaluate_random_policy(env_name, seed=None if seed == "" else int(seed))
            rows.append({f"final_test_{key}": value for key, value in baseline.items()})
            rows[-1].update({"env_name": env_name, "algo_name": "random", "seed": seed})

    target = Path(output_path) if output_path is not None else resolved_logs_dir / "generated_summary.csv"
    return save_experiment_summary(rows, target)


def save_experiment_summary(results: list[dict], file_path: Path | None = None) -> Path:
    if file_path is not None:
        output_path = file_path
    elif results and results[0].get("logs_dir"):
        output_path = Path(results[0]["logs_dir"]) / "experiment_summary.csv"
    else:
        output_path = LOGS_DIR / "experiment_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    extra_fields = sorted({key for row in results for key in row if key not in SUMMARY_FIELDNAMES})
    fieldnames = SUMMARY_FIELDNAMES + extra_fields
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return output_path
