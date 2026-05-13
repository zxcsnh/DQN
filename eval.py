from DQN.evaluation import final_test


MODEL_PATH_OVERRIDE = "results\\0512-1913-experiment\\models\\dino_dqn_seed42_best.pth"


def main() -> None:
    result = final_test(
        env_name="dino",
        algo_name="dqn",
        model_kind="best",
        render=True,
        model_path_override=MODEL_PATH_OVERRIDE,
    )
    print(result)


if __name__ == "__main__":
    main()
