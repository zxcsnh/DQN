from DQN.experiment_utils import run_batch_experiments, save_experiment_summary

def main() -> None:
    results = run_batch_experiments(
        env_names=["taxi", "mountaincar", "dino"],
        algo_names=["dqn", "perdqn"],
        seeds=[42, 52, 62],
        render=False,
        plot_after_each_env=False,
    )
    summary_path = save_experiment_summary(results)
    print({"runs": len(results), "summary_path": str(summary_path)})


if __name__ == "__main__":
    main()
