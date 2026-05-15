from pathlib import Path

from DQN.utils.plot_utils import plot_env_comparisons, plot_env_multiseed_comparisons, plot_env_multiseed_single_algo


COMPARE_RUN_DIR = Path("results\\0514-2319-experiment")
env_names = ["dino", "taxi", "mountaincar"]
window = 50
BY_SEED_FIGURES_DIR = "by_seed"
MULTISEED_FIGURES_DIR = "multiseed"
SINGLE_ALGO_MULTISEED_FIGURES_DIR = "multiseed_single"


def _collect_suffix_logs(logs_dir: Path, env_name: str, algo_name: str) -> dict[str, Path]:
    logs: dict[str, Path] = {}
    prefix = f"{env_name}_{algo_name}_"
    postfix = "_train_log.csv"

    for path in logs_dir.glob(f"{env_name}_{algo_name}_*_train_log.csv"):
        name = path.name
        if not name.startswith(prefix) or not name.endswith(postfix):
            continue
        suffix = name[len(prefix) : -len(postfix)]
        if not suffix:
            continue
        existing = logs.get(suffix)
        if existing is None or path.stat().st_mtime > existing.stat().st_mtime:
            logs[suffix] = path

    return logs


def _resolve_log_pair(logs_dir: Path, env_name: str) -> tuple[Path | None, Path | None]:
    dqn_plain = logs_dir / f"{env_name}_dqn_train_log.csv"
    perdqn_plain = logs_dir / f"{env_name}_perdqn_train_log.csv"
    if dqn_plain.exists() and perdqn_plain.exists():
        return dqn_plain, perdqn_plain

    dqn_by_suffix = _collect_suffix_logs(logs_dir, env_name, "dqn")
    perdqn_by_suffix = _collect_suffix_logs(logs_dir, env_name, "perdqn")
    common_suffixes = set(dqn_by_suffix).intersection(perdqn_by_suffix)
    if not common_suffixes:
        return None, None

    selected_suffix = max(
        common_suffixes,
        key=lambda suffix: max(
            dqn_by_suffix[suffix].stat().st_mtime,
            perdqn_by_suffix[suffix].stat().st_mtime,
        ),
    )
    return dqn_by_suffix[selected_suffix], perdqn_by_suffix[selected_suffix]


def _resolve_seed_log_pairs(logs_dir: Path, env_name: str) -> dict[str, tuple[Path, Path]]:
    dqn_by_suffix = _collect_suffix_logs(logs_dir, env_name, "dqn")
    perdqn_by_suffix = _collect_suffix_logs(logs_dir, env_name, "perdqn")
    common_suffixes = sorted(set(dqn_by_suffix).intersection(perdqn_by_suffix))
    return {suffix: (dqn_by_suffix[suffix], perdqn_by_suffix[suffix]) for suffix in common_suffixes}


def plot_comparisons_by_seed(logs_dir: Path, figures_dir: Path, env_name: str) -> int:
    seed_pairs = _resolve_seed_log_pairs(logs_dir, env_name)
    for suffix, (dqn_log, perdqn_log) in seed_pairs.items():
        seed_figures_dir = figures_dir / BY_SEED_FIGURES_DIR / suffix
        plot_env_comparisons(env_name, dqn_log, perdqn_log, window, figures_dir=seed_figures_dir)
    return len(seed_pairs)


def plot_multiseed_comparisons(logs_dir: Path, figures_dir: Path, env_name: str) -> int:
    seed_pairs = _resolve_seed_log_pairs(logs_dir, env_name)
    if not seed_pairs:
        return 0

    dqn_logs = [pair[0] for pair in seed_pairs.values()]
    perdqn_logs = [pair[1] for pair in seed_pairs.values()]
    plot_env_multiseed_comparisons(
        env_name,
        dqn_logs,
        perdqn_logs,
        window,
        figures_dir=figures_dir / MULTISEED_FIGURES_DIR,
    )
    return len(seed_pairs)


def plot_single_algo_multiseed(logs_dir: Path, figures_dir: Path, env_name: str, algo_name: str) -> int:
    logs_by_suffix = _collect_suffix_logs(logs_dir, env_name, algo_name)
    if not logs_by_suffix:
        return 0

    logs = [logs_by_suffix[suffix] for suffix in sorted(logs_by_suffix)]
    plot_env_multiseed_single_algo(
        env_name,
        algo_name,
        logs,
        window,
        figures_dir=figures_dir / SINGLE_ALGO_MULTISEED_FIGURES_DIR / algo_name,
    )
    return len(logs)

def main() -> None:
    run_dir = COMPARE_RUN_DIR
    logs_dir = run_dir / "logs"
    figures_dir = run_dir / "figures"

    if not run_dir.exists():
        print(f"指定结果目录不存在：{run_dir}")
        return
    if not logs_dir.exists():
        print(f"指定结果目录缺少 logs 子目录：{logs_dir}")
        return

    for env_name in env_names:
        dqn_log, perdqn_log = _resolve_log_pair(logs_dir, env_name)
        if dqn_log is None or perdqn_log is None:
            print(f"跳过 {env_name}：缺少 DQN 或 PER-DQN 日志文件。")
            continue

        plot_env_comparisons(env_name, dqn_log, perdqn_log, window, figures_dir=figures_dir)
        print(f"已生成 {env_name} 的对比图。")

        seed_count = plot_comparisons_by_seed(logs_dir, figures_dir, env_name)
        if seed_count == 0:
            print(f"跳过 {env_name} 的逐 seed 对比图：没有共同 seed 日志。")
        else:
            print(f"已生成 {env_name} 的逐 seed 对比图，共 {seed_count} 组。")

        multiseed_count = plot_multiseed_comparisons(logs_dir, figures_dir, env_name)
        if multiseed_count == 0:
            print(f"跳过 {env_name} 的多 seed 聚合图：没有共同 seed 日志。")
        else:
            print(f"已生成 {env_name} 的多 seed 聚合图，共 {multiseed_count} 个 seed。")


        for algo_name in ("dqn", "perdqn"):
            single_algo_count = plot_single_algo_multiseed(logs_dir, figures_dir, env_name, algo_name)
            if single_algo_count == 0:
                print(f"Skip {env_name} {algo_name} single-algo multi-seed plots: no logs found.")
            else:
                print(f"Generated {env_name} {algo_name} single-algo multi-seed plots from {single_algo_count} seeds.")

if __name__ == "__main__":
    main()
