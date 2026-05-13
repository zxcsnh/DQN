from datetime import datetime
from pathlib import Path

from config import RESULTS_DIR
from DQN.experiment_utils import run_batch_experiments, save_experiment_summary


def _make_experiment_run_dir() -> Path:
    base_name = f"{datetime.now().strftime('%m%d-%H%M')}-experiment"
    run_dir = RESULTS_DIR / base_name
    suffix = 1
    while run_dir.exists():
        run_dir = RESULTS_DIR / f"{base_name}-{suffix}"
        suffix += 1
    return run_dir


def main() -> None:
    run_dir = _make_experiment_run_dir()
    results = run_batch_experiments(
        # env_names=["taxi", "mountaincar", "dino"],
        env_names=["dino"],
        algo_names=["dqn"],
        render=False,
        plot_after_each_env=False,
        run_dir=run_dir,
    )
    summary_path = save_experiment_summary(results)
    print({"runs": len(results), "run_dir": str(run_dir), "summary_path": str(summary_path)})


if __name__ == "__main__":
    main()
