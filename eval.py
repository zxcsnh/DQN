from DQN.evaluation import final_test


MODEL_PATH_OVERRIDE = "results/0509-2254-experiment/models/dino_perdqn_seed47_best.pth"


def main() -> None:
    result = final_test(
        env_name="dino",
        algo_name="perdqn",
        model_kind="best",
        render=False,
        model_path_override=MODEL_PATH_OVERRIDE,
    )
    print(result)


if __name__ == "__main__":
    main()
