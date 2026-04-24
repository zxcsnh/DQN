from DQN.evaluation import evaluate


def main() -> None:
    result = evaluate(
        env_name="dino",
        algo_name="perdqn",
        render=True,
        # seed=42,
        model_path_override="results\\models\\dino_perdqn_best.pth",
    )
    print(result)


if __name__ == "__main__":
    main()
