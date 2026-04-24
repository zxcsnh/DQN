from DQN.training import train

def main() -> None:
    result = train(
        env_name="dino",
        algo_name="perdqn",
        render=False,
        plot_after_train=True,
    )
    print(result)


if __name__ == "__main__":
    main()
