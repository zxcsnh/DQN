from pathlib import Path

from DQN.experiment_utils import generate_experiment_summary


SUMMARY_RUN_DIR = Path("results/0509-2254-experiment")
INCLUDE_RANDOM_BASELINE = True


def main() -> None:
    summary_path = generate_experiment_summary(
        logs_dir=SUMMARY_RUN_DIR / "logs",
        include_random_baseline=INCLUDE_RANDOM_BASELINE,
    )
    print({"summary_path": str(summary_path)})


if __name__ == "__main__":
    main()
