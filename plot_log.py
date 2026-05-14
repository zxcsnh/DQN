from __future__ import annotations

import argparse
from pathlib import Path

from config import get_env_config
from DQN.utils.plot_utils import plot_single_run


DEFAULT_LOG_PATH = Path("results\\0514-1035-dino-dqn-seed37\logs\dino_dqn_train_log.csv")


def _infer_env_algo(log_path: Path) -> tuple[str, str]:
    name = log_path.name.removesuffix("_train_log.csv")
    parts = name.split("_")
    if len(parts) < 2:
        raise ValueError("无法从日志文件名推断环境和算法，请使用 --env-name 和 --algo-name 指定。")
    return parts[0], parts[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curves for a single log file.")
    parser.add_argument("log_path", nargs="?", type=Path, default=DEFAULT_LOG_PATH, help="训练日志 CSV 路径")
    parser.add_argument("--env-name", default="", help="环境名称；默认从文件名推断")
    parser.add_argument("--algo-name", default="", help="算法名称；默认从文件名推断")
    parser.add_argument("--window", type=int, default=0, help="移动平均窗口；默认使用环境配置")
    parser.add_argument("--figures-dir", type=Path, default=None, help="图片输出目录；默认使用全局 figures 目录")
    args = parser.parse_args()

    log_path = args.log_path
    env_name, algo_name = _infer_env_algo(log_path)
    env_name = args.env_name or env_name
    algo_name = args.algo_name or algo_name
    window = args.window if args.window > 0 else get_env_config(env_name).moving_average_window

    plot_single_run(log_path, env_name, algo_name, window=window, figures_dir=args.figures_dir)
    print({"log_path": str(log_path), "env_name": env_name, "algo_name": algo_name, "figures_dir": str(args.figures_dir or "default")})


if __name__ == "__main__":
    main()
