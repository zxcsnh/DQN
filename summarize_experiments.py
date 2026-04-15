"""汇总批量实验结果，输出论文可用表格与曲线图。"""

import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class SummarySettings:
    """在这里直接修改汇总脚本的输入输出路径。"""

    manifest_path: str = os.path.join("experiments", "experiment_manifest.json")
    output_dir: str | None = None


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_group(runs: list[dict]) -> dict:
    """对同一环境、同一方法的多 seed 结果做统计。"""
    best_eval_rewards = [run["summary"]["best_eval_reward"] for run in runs]
    final_avg_rewards = [run["summary"]["final_avg_reward_100"] for run in runs]
    total_steps = [run["summary"]["total_steps"] for run in runs]

    return {
        "num_runs": len(runs),
        "best_eval_reward_mean": float(np.mean(best_eval_rewards)),
        "best_eval_reward_std": float(np.std(best_eval_rewards)),
        "final_avg_reward_100_mean": float(np.mean(final_avg_rewards)),
        "final_avg_reward_100_std": float(np.std(final_avg_rewards)),
        "total_steps_mean": float(np.mean(total_steps)),
        "total_steps_std": float(np.std(total_steps)),
    }


def align_eval_curves(runs: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按 step 对齐评估曲线，减少不同 run 评估时刻不一致带来的偏差。"""
    valid_runs: list[tuple[np.ndarray, np.ndarray]] = []
    for run in runs:
        steps = np.array(run["summary"]["eval_steps"], dtype=np.float64)
        rewards = np.array(run["summary"]["eval_rewards"], dtype=np.float64)
        usable_len = min(len(steps), len(rewards))
        if usable_len == 0:
            continue

        steps = steps[:usable_len]
        rewards = rewards[:usable_len]

        sort_idx = np.argsort(steps)
        steps = steps[sort_idx]
        rewards = rewards[sort_idx]

        # np.interp 要求 x 递增，重复 step 只保留首个点。
        unique_steps, unique_idx = np.unique(steps, return_index=True)
        unique_rewards = rewards[unique_idx]
        if len(unique_steps) == 0:
            continue

        valid_runs.append((unique_steps, unique_rewards))

    if not valid_runs:
        return np.array([]), np.array([]), np.array([])

    min_len = min(len(steps) for steps, _ in valid_runs)
    overlap_start = max(steps[0] for steps, _ in valid_runs)
    overlap_end = min(steps[-1] for steps, _ in valid_runs)

    # 正常情况：在重叠 step 区间上插值后再求均值/方差。
    if overlap_start < overlap_end:
        common_steps = np.linspace(overlap_start, overlap_end, num=min_len)
        aligned_rewards = np.stack(
            [
                np.interp(common_steps, run_steps, run_rewards)
                for run_steps, run_rewards in valid_runs
            ]
        )
        return common_steps, aligned_rewards.mean(axis=0), aligned_rewards.std(axis=0)

    # 兜底：若重叠区间退化，回退到按序号截断聚合。
    steps = np.mean(
        np.stack([run_steps[:min_len] for run_steps, _ in valid_runs]),
        axis=0,
    )
    rewards = np.stack([run_rewards[:min_len] for _, run_rewards in valid_runs])
    return steps, rewards.mean(axis=0), rewards.std(axis=0)


def save_csv(path: str, rows: list[dict], fieldnames: list[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_markdown_table(path: str, rows: list[dict]):
    headers = [
        "环境",
        "方法",
        "运行次数",
        "最佳评估均值",
        "最佳评估标准差",
        "最终 Avg100 均值",
        "最终 Avg100 标准差",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["env_name"],
                    row["variant"],
                    str(row["num_runs"]),
                    f"{row['best_eval_reward_mean']:.3f}",
                    f"{row['best_eval_reward_std']:.3f}",
                    f"{row['final_avg_reward_100_mean']:.3f}",
                    f"{row['final_avg_reward_100_std']:.3f}",
                ]
            )
            + " |"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def plot_group_curves(groups: dict, output_dir: str):
    """为每个环境绘制 DQN 与 PER-DQN 的对比曲线。"""
    env_to_variants: dict[str, list[tuple[str, list[dict]]]] = defaultdict(list)
    for (env_name, variant), runs in groups.items():
        env_to_variants[env_name].append((variant, runs))

    for env_name, variant_runs in env_to_variants.items():
        plt.figure(figsize=(8, 5))

        for variant, runs in sorted(variant_runs):
            steps, reward_mean, reward_std = align_eval_curves(runs)
            if len(steps) == 0:
                continue

            plt.plot(steps, reward_mean, label=variant)
            plt.fill_between(
                steps,
                reward_mean - reward_std,
                reward_mean + reward_std,
                alpha=0.2,
            )

        plt.xlabel("Environment Steps")
        plt.ylabel("Evaluation Reward")
        plt.title(f"{env_name}: DQN vs PER-DQN")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plot_name = env_name.replace("/", "_").replace(":", "_") + "_comparison.png"
        plt.savefig(os.path.join(output_dir, plot_name), dpi=150, bbox_inches="tight")
        plt.close()


def main():
    settings = SummarySettings()
    manifest = load_json(settings.manifest_path)

    manifest_dir = os.path.dirname(settings.manifest_path)
    output_dir = settings.output_dir or os.path.join(manifest_dir, "summary")
    os.makedirs(output_dir, exist_ok=True)

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for run in manifest["runs"]:
        summary = run.get("summary")
        if summary is None:
            summary = load_json(run["summary_path"])
            run["summary"] = summary
        groups[(run["env_name"], run["variant"])].append(run)

    aggregate_rows = []
    aggregate_json = {}
    for (env_name, variant), runs in sorted(groups.items()):
        stats = summarize_group(runs)
        row = {
            "env_name": env_name,
            "variant": variant,
            **stats,
        }
        aggregate_rows.append(row)
        aggregate_json.setdefault(env_name, {})[variant] = row

    save_csv(
        os.path.join(output_dir, "aggregate_results.csv"),
        aggregate_rows,
        [
            "env_name",
            "variant",
            "num_runs",
            "best_eval_reward_mean",
            "best_eval_reward_std",
            "final_avg_reward_100_mean",
            "final_avg_reward_100_std",
            "total_steps_mean",
            "total_steps_std",
        ],
    )
    save_markdown_table(os.path.join(output_dir, "aggregate_results.md"), aggregate_rows)
    with open(
        os.path.join(output_dir, "aggregate_results.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(aggregate_json, f, indent=2, ensure_ascii=False)

    plot_group_curves(groups, output_dir)

    print("Saved aggregate summaries to:")
    print(f"- {os.path.join(output_dir, 'aggregate_results.csv')}")
    print(f"- {os.path.join(output_dir, 'aggregate_results.md')}")
    print(f"- {os.path.join(output_dir, 'aggregate_results.json')}")


if __name__ == "__main__":
    main()
