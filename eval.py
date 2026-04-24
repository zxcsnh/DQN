from DQN.evaluation import evaluate


def main() -> None:
    result = evaluate(
        env_name="taxi",
        algo_name="dqn",
        render=False,
        # seed=42,
        model_path_override="results\\models\\taxi_dqn_best.pth",
    )
    print(result)


if __name__ == "__main__":
    main()
