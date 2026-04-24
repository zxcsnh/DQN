from __future__ import annotations

import csv
from pathlib import Path

from config import LOGS_DIR
from .training import train


def run_batch_experiments(env_names: list[str], algo_names: list[str], seeds: list[int], render: bool = False, plot_after_each_env: bool = False) -> list[dict]:
    results = []
    for env_name in env_names:
        for algo_name in algo_names:
            for seed in seeds:
                suffix = f"seed{seed}"
                result = train(
                    env_name=env_name,
                    algo_name=algo_name,
                    render=render,
                    plot_after_train=plot_after_each_env,
                    seed=seed,
                    log_name_suffix=suffix,
                )
                results.append(result)
    return results


def save_experiment_summary(results: list[dict], file_path: Path | None = None) -> Path:
    output_path = file_path or (LOGS_DIR / "experiment_summary.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "env_name",
        "algo_name",
        "seed",
        "best_train_reward",
        "best_eval_reward",
        "best_model_episode",
        "best_model_path",
        "final_model_path",
        "last_eval_avg_reward",
        "log_path",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    return output_path
