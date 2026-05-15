from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from config import get_env_config, supported_envs
from DQN.envs import make_env
from DQN.evaluation import evaluate_agent
from DQN.shared import make_agent, validate_names
from DQN.utils.seed_utils import seed_env, set_global_seed
from DQN.utils.state_processor import get_state_dim

DEFAULT_RUN_DIR = Path("results") / "0514-2319-experiment"
DEFAULT_EPISODES = 100
DEFAULT_TEST_SEED_START = 90_000
ALGO_NAMES = ("dqn", "perdqn")
MODEL_RE = re.compile(r"^(?P<env>.+)_(?P<algo>dqn|perdqn)_seed(?P<seed>\d+)_best\.pth$")

BASE_DETAIL_FIELDS = [
    "env_name",
    "algo_name",
    "train_seed",
    "model_path",
    "episode",
    "episode_seed",
    "reward",
    "steps",
    "success",
    "custom_metric",
    "terminated",
    "truncated",
]
OPTIONAL_DETAIL_FIELDS = ["max_position", "score", "speed", "obstacles_cleared"]
SUMMARY_METRICS = [
    "avg_reward",
    "std_reward",
    "avg_steps",
    "std_steps",
    "success_rate",
    "avg_custom_metric",
    "std_custom_metric",
    "avg_max_position",
    "avg_score",
    "avg_speed",
    "avg_obstacles_cleared",
]


@dataclass(frozen=True)
class BestModel:
    env_name: str
    algo_name: str
    train_seed: int
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate best models with a shared episode seed set and aggregate "
            "results across training seeds."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="Experiment run directory.")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES, help="Evaluation episodes per model.")
    parser.add_argument(
        "--test-seed-start",
        type=int,
        default=DEFAULT_TEST_SEED_START,
        help="First episode seed. Episode i uses test_seed_start + i.",
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        default=list(supported_envs()),
        choices=list(supported_envs()),
        help="Environments to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <run-dir>/best_model_eval_<timestamp>.",
    )
    parser.add_argument("--render", action="store_true", help="Render evaluation episodes.")
    return parser.parse_args()


def discover_best_models(run_dir: Path, env_names: Iterable[str]) -> list[BestModel]:
    models_dir = run_dir / "models"
    if not models_dir.exists():
        raise FileNotFoundError(f"未找到 models 目录: {models_dir}")

    env_set = set(env_names)
    models: list[BestModel] = []
    for path in models_dir.glob("*_best.pth"):
        match = MODEL_RE.match(path.name)
        if match is None:
            continue
        env_name = match.group("env")
        algo_name = match.group("algo")
        if env_name not in env_set:
            continue
        train_seed = int(match.group("seed"))
        validate_names(env_name, algo_name)
        models.append(BestModel(env_name, algo_name, train_seed, path))

    return sorted(models, key=lambda item: (item.env_name, item.algo_name, item.train_seed))


def build_agent(env_name: str, algo_name: str, model_path: Path, render: bool = False):
    env = make_env(env_name, render=render)
    seed_env(env, get_env_config(env_name).seed)
    state_dim = get_state_dim(env_name, env)
    action_dim = int(env.action_space.n)
    env.close()

    agent = make_agent(env_name, algo_name, state_dim, action_dim)
    agent.load(model_path)
    agent.epsilon = 0.0
    return agent


def evaluate_best_model(model: BestModel, episodes: int, test_seed_start: int, render: bool) -> tuple[dict, list[dict]]:
    set_global_seed(test_seed_start)
    agent = build_agent(model.env_name, model.algo_name, model.path, render=render)
    metrics = evaluate_agent(
        env_name=model.env_name,
        algo_name=model.algo_name,
        agent=agent,
        episodes=episodes,
        render=render,
        seed=test_seed_start,
        return_episodes=True,
    )
    detail_rows = metrics.pop("episodes_detail")
    metrics.update(
        {
            "env_name": model.env_name,
            "algo_name": model.algo_name,
            "train_seed": model.train_seed,
            "model_kind": "best",
            "model_path": str(model.path),
            "test_seed_start": test_seed_start,
            "episodes": episodes,
        }
    )

    for row in detail_rows:
        episode_idx = int(row["episode"]) - 1
        row.update(
            {
                "env_name": model.env_name,
                "algo_name": model.algo_name,
                "train_seed": model.train_seed,
                "model_path": str(model.path),
                "episode_seed": test_seed_start + episode_idx,
            }
        )
    return metrics, detail_rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys

    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std())


def aggregate_model_summaries(model_summaries: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in model_summaries:
        grouped[(row["env_name"], row["algo_name"])].append(row)

    aggregate_rows: list[dict] = []
    for (env_name, algo_name), rows in sorted(grouped.items()):
        aggregate = {
            "env_name": env_name,
            "algo_name": algo_name,
            "model_kind": "best",
            "num_train_seeds": len(rows),
            "train_seeds": ",".join(str(row["train_seed"]) for row in rows),
            "episodes_per_model": rows[0]["episodes"],
            "test_seed_start": rows[0]["test_seed_start"],
        }
        for metric in SUMMARY_METRICS:
            values = [float(row[metric]) for row in rows if row.get(metric, "") not in {"", None}]
            if not values:
                continue
            mean_value, std_value = mean_std(values)
            aggregate[f"{metric}_mean"] = mean_value
            aggregate[f"{metric}_std"] = std_value
        aggregate_rows.append(aggregate)
    return aggregate_rows


def aggregate_episode_rows(detail_rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in detail_rows:
        grouped[(row["env_name"], row["algo_name"], int(row["episode"]))].append(row)

    by_env_algo: dict[tuple[str, str], list[dict]] = defaultdict(list)
    metrics = ["reward", "steps", "success", "custom_metric", *OPTIONAL_DETAIL_FIELDS]
    for (env_name, algo_name, episode), rows in sorted(grouped.items()):
        aggregate = {
            "env_name": env_name,
            "algo_name": algo_name,
            "episode": episode,
            "episode_seed": rows[0]["episode_seed"],
            "num_train_seeds": len(rows),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in rows if row.get(metric, "") not in {"", None}]
            if not values:
                continue
            mean_value, std_value = mean_std(values)
            aggregate[f"{metric}_mean"] = mean_value
            aggregate[f"{metric}_std"] = std_value
        by_env_algo[(env_name, algo_name)].append(aggregate)
    return by_env_algo


def plot_metric(
    env_name: str,
    by_env_algo: dict[tuple[str, str], list[dict]],
    metric: str,
    ylabel: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    plotted = False
    for algo_name, label in [("dqn", "DQN"), ("perdqn", "PER-DQN")]:
        rows = by_env_algo.get((env_name, algo_name), [])
        if not rows:
            continue
        mean_key = f"{metric}_mean"
        std_key = f"{metric}_std"
        rows = [row for row in rows if mean_key in row]
        if not rows:
            continue
        x_values = np.asarray([int(row["episode"]) for row in rows], dtype=np.int32)
        mean_values = np.asarray([float(row[mean_key]) for row in rows], dtype=np.float64)
        std_values = np.asarray([float(row.get(std_key, 0.0)) for row in rows], dtype=np.float64)
        plt.plot(x_values, mean_values, label=f"{label} (n={rows[0]['num_train_seeds']})")
        plt.fill_between(x_values, mean_values - std_values, mean_values + std_values, alpha=0.18)
        plotted = True

    if not plotted:
        plt.close()
        return
    plt.xlabel("Evaluation episode")
    plt.ylabel(ylabel)
    plt.title(f"{env_name} best-model evaluation {metric}")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_all(by_env_algo: dict[tuple[str, str], list[dict]], env_names: Iterable[str], figures_dir: Path) -> None:
    common_metrics = [
        ("reward", "Reward"),
        ("steps", "Steps"),
        ("success", "Success"),
        ("custom_metric", "Custom metric"),
    ]
    special_metrics = {
        "mountaincar": [("max_position", "Max position")],
        "dino": [
            ("score", "Score"),
            ("speed", "Speed"),
            ("obstacles_cleared", "Obstacles cleared"),
        ],
    }

    for env_name in env_names:
        metrics = [*common_metrics, *special_metrics.get(env_name, [])]
        for metric, ylabel in metrics:
            plot_metric(
                env_name=env_name,
                by_env_algo=by_env_algo,
                metric=metric,
                ylabel=ylabel,
                output_path=figures_dir / f"{env_name}_{metric}_best_eval_multiseed.png",
            )


def validate_model_coverage(models: list[BestModel], env_names: Iterable[str]) -> None:
    by_key = defaultdict(list)
    for model in models:
        by_key[(model.env_name, model.algo_name)].append(model.train_seed)

    for env_name in env_names:
        for algo_name in ALGO_NAMES:
            seeds = sorted(by_key.get((env_name, algo_name), []))
            if not seeds:
                print(f"跳过提示：未找到 {env_name}-{algo_name} 的 best 模型。")
            else:
                print(f"{env_name}-{algo_name}: best 模型 seeds={seeds}")


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes 必须大于等于 1")

    run_dir = args.run_dir
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%m%d-%H%M")
        output_dir = run_dir / f"best_model_eval_{timestamp}"

    models = discover_best_models(run_dir, args.envs)
    if not models:
        raise FileNotFoundError(f"未在 {run_dir / 'models'} 中找到匹配的 best 模型。")

    validate_model_coverage(models, args.envs)

    model_summaries: list[dict] = []
    detail_rows: list[dict] = []
    for model in models:
        print(
            f"评估 {model.env_name}-{model.algo_name}-seed{model.train_seed} "
            f"best 模型，共 {args.episodes} 回合..."
        )
        summary, rows = evaluate_best_model(model, args.episodes, args.test_seed_start, args.render)
        model_summaries.append(summary)
        detail_rows.extend(rows)

    detail_fields = [*BASE_DETAIL_FIELDS, *OPTIONAL_DETAIL_FIELDS]
    write_csv(output_dir / "best_model_eval_episode_detail.csv", detail_rows, detail_fields)
    write_csv(output_dir / "best_model_eval_model_summary.csv", model_summaries)

    aggregate_summary = aggregate_model_summaries(model_summaries)
    write_csv(output_dir / "best_model_eval_aggregate_summary.csv", aggregate_summary)

    episode_aggregates = aggregate_episode_rows(detail_rows)
    aggregate_episode_rows_flat = [row for rows in episode_aggregates.values() for row in rows]
    write_csv(output_dir / "best_model_eval_episode_multiseed_summary.csv", aggregate_episode_rows_flat)
    plot_all(episode_aggregates, args.envs, output_dir / "figures")

    print(f"评估完成，结果已保存到: {output_dir}")
    print(f"聚合汇总: {output_dir / 'best_model_eval_aggregate_summary.csv'}")
    print(f"图表目录: {output_dir / 'figures'}")


if __name__ == "__main__":
    main()
