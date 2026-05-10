from pathlib import Path

from DQN.utils.plot_utils import plot_env_comparisons, plot_summary_bars


COMPARE_RUN_DIR = Path("results\\0509-2254-experiment")
env_names = ["taxi", "mountaincar", "dino"]
window = 50


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

    plot_summary_bars(logs_dir / "experiment_summary.csv", figures_dir=figures_dir)


if __name__ == "__main__":
    main()
