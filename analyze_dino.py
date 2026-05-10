from pathlib import Path
from pprint import pprint

from DQN.dino_analysis import analyze_dino_strategy


MODEL_PATH_OVERRIDE = Path("results/0509-2254-experiment/models/dino_perdqn_seed47_best.pth")


def main() -> None:
    result = analyze_dino_strategy(
        algo_name="perdqn",
        model_kind="best",
        episodes=10,
        model_path_override=MODEL_PATH_OVERRIDE,
    )
    pprint(result)


if __name__ == "__main__":
    main()
