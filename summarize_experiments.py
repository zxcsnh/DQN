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
    """把不同 seed 的评估曲线裁到相同长度，便于求均值和方差。"""
    eval_steps_list = [
        np.array(run["summary"]["eval_steps"], dtype=np.float64) for run in runs
    ]
    eval_rewards_list = [
        np.array(run["summary"]["eval_rewards"], dtype=np.float64) for run in runs
    ]

    valid_lengths = [
        min(len(steps), len(rewards))
        for steps, rewards in zip(eval_steps_list, eval_rewards_list, strict=False)
        if len(steps) > 0 and len(rewards) > 0
    ]
    if not valid_lengths:
        return np.array([]), np.array([]), np.array([])

    min_len = min(valid_lengths)
    steps = np.mean(
        np.stack(
            [steps[:min_len] for steps in eval_steps_list if len(steps) >= min_len]
        ),
        axis=0,
    )
    rewards = np.stack(
        [rewards[:min_len] for rewards in eval_rewards_list if len(rewards) >= min_len]
    )
    reward_mean = rewards.mean(axis=0)
    reward_std = rewards.std(axis=0)
    return steps, reward_mean, reward_std


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
