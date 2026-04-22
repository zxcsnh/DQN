from DQN.evaluation import evaluate


def main() -> None:
    result = evaluate(
        env_name="taxi",
        algo_name="dqn",
        render=False,
        use_best_model=True,
    )
    print(result)


if __name__ == "__main__":
    main()
