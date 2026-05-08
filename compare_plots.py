from pathlib import Path

from config import LOGS_DIR
from DQN.utils.plot_utils import plot_env_comparisons


env_names = ["taxi", "mountaincar", "dino"]
window = 50


def _collect_suffix_logs(env_name: str, algo_name: str) -> dict[str, Path]:
    logs: dict[str, Path] = {}
    prefix = f"{env_name}_{algo_name}_"
    postfix = "_train_log.csv"

    for path in LOGS_DIR.glob(f"{env_name}_{algo_name}_*_train_log.csv"):
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


def _resolve_log_pair(env_name: str) -> tuple[Path | None, Path | None]:
    dqn_plain = LOGS_DIR / f"{env_name}_dqn_train_log.csv"
    perdqn_plain = LOGS_DIR / f"{env_name}_perdqn_train_log.csv"
    if dqn_plain.exists() and perdqn_plain.exists():
        return dqn_plain, perdqn_plain

    dqn_by_suffix = _collect_suffix_logs(env_name, "dqn")
    perdqn_by_suffix = _collect_suffix_logs(env_name, "perdqn")
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
    for env_name in env_names:
        dqn_log, perdqn_log = _resolve_log_pair(env_name)
        if dqn_log is None or perdqn_log is None:
            print(f"跳过 {env_name}：缺少 DQN 或 PER-DQN 日志文件。")
            continue

        plot_env_comparisons(env_name, dqn_log, perdqn_log, window)
        print(f"已生成 {env_name} 的对比图。")


if __name__ == "__main__":
    main()
