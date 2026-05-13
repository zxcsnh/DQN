import argparse
from pathlib import Path

from DQN.experiment_utils import generate_experiment_summary


DEFAULT_RUN_DIR = Path("results\\0513-1746-experiment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an experiment summary from train logs.")
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Experiment run directory containing a logs/ folder.",
    )
    parser.add_argument(
        "--include-random-baseline",
        action="store_true",
        help="Append random-policy final test rows for each environment/seed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path. Defaults to <run_dir>/logs/generated_summary.csv.",
    )
    parser.add_argument(
        "--evaluate-models",
        action="store_true",
        help="Run final tests for both best and final model checkpoints before writing summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = generate_experiment_summary(
        logs_dir=args.run_dir / "logs",
        output_path=args.output,
        include_random_baseline=args.include_random_baseline,
        evaluate_models=args.evaluate_models,
    )
    print({"summary_path": str(summary_path)})


if __name__ == "__main__":
    main()
