from pathlib import Path

from DQN.evaluation import final_test


MODEL_PATH = Path("results\\0513-1746-experiment\models\dino_dqn_seed57_best.pth")


def parse_model_name(model_path: Path) -> tuple[str, str, int | None, str]:
    parts = model_path.stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"模型文件名格式不正确: {model_path.name}")

    env_name = parts[0]
    algo_name = parts[1]
    model_kind = parts[-1]
    seed = None
    for part in parts[2:-1]:
        if part.startswith("seed"):
            seed = int(part.removeprefix("seed"))
            break

    return env_name, algo_name, seed, model_kind


def main() -> None:
    env_name, algo_name, seed, model_kind = parse_model_name(MODEL_PATH)
    result = final_test(
        env_name=env_name,
        algo_name=algo_name,
        model_kind=model_kind,
        seed=seed,
        render=False,
        model_path_override=MODEL_PATH,
    )
    print(result)


if __name__ == "__main__":
    main()
