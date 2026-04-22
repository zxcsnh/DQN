from pathlib import Path

from config import LOGS_DIR
from DQN.utils.plot_utils import plot_env_comparisons



env_names = ["taxi", "mountaincar", "dino"],
window = 50,

def main() -> None:
    for env_name in env_names:
        dqn_log = LOGS_DIR / f"{env_name}_dqn_train_log.csv"
        perdqn_log = LOGS_DIR / f"{env_name}_perdqn_train_log.csv"
        if not dqn_log.exists() or not perdqn_log.exists():
            print(f"跳过 {env_name}：缺少 DQN 或 PER-DQN 日志文件。")
            continue
        plot_env_comparisons(env_name, dqn_log, perdqn_log, window)
        print(f"已生成 {env_name} 的对比图。")


if __name__ == "__main__":
    main()
