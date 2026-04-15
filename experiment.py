"""批量运行 DQN 与 PER-DQN 对比实验。"""

import json
import os
from dataclasses import asdict, dataclass, replace

from config import DQNConfig
from train import train


@dataclass
class ExperimentSettings:
    """在这里直接修改实验设置，再运行脚本。"""

    envs: list[str] = None
    seeds: list[int] = None
    variants: list[str] = None
    output_root: str = "experiments"
    num_episodes: int | None = None
    max_steps: int | None = 300_000
    eval_freq: int | None = 20
    eval_episodes: int | None = 5
    save_freq: int | None = 100
    device: str | None = None
    save_replay_buffer: bool = False

    def __post_init__(self):
        if self.envs is None:
            self.envs = ["ALE/Pong-v5", "ALE/Breakout-v5"]
        if self.seeds is None:
            self.seeds = [42, 43, 44]
        if self.variants is None:
            # dqn 表示普通经验回放，per 表示优先经验回放。
            self.variants = ["dqn", "per"]


def build_run_config(
    base_config: DQNConfig,
    env_name: str,
    seed: int,
    variant: str,
    settings: ExperimentSettings,
) -> DQNConfig:
    """根据实验设置，为每一次 run 生成独立配置。"""
    variant_name = "per_dqn" if variant == "per" else "dqn"
    env_slug = env_name.replace("/", "_").replace(":", "_")
    run_dir = os.path.join(settings.output_root, env_slug, variant_name, f"seed_{seed}")

    config = replace(
        base_config,
        env_name=env_name,
        seed=seed,
        use_per=(variant == "per"),
        save_dir=os.path.join(run_dir, "models"),
        log_dir=os.path.join(run_dir, "logs"),
        save_replay_buffer=settings.save_replay_buffer,
    )

    if settings.num_episodes is not None:
        config.num_episodes = settings.num_episodes
    if settings.max_steps is not None:
        config.max_steps = settings.max_steps
    if settings.eval_freq is not None:
        config.eval_freq = settings.eval_freq
    if settings.eval_episodes is not None:
        config.eval_episodes = settings.eval_episodes
    if settings.save_freq is not None:
        config.save_freq = settings.save_freq
    if settings.device is not None:
        config.device = settings.device

    return config


def main():
    settings = ExperimentSettings()
    base_config = DQNConfig()

    os.makedirs(settings.output_root, exist_ok=True)

    manifest = {
        "envs": settings.envs,
        "seeds": settings.seeds,
        "variants": settings.variants,
        "base_config": asdict(base_config),
        "overrides": asdict(settings),
        "runs": [],
    }

    total_runs = len(settings.envs) * len(settings.variants) * len(settings.seeds)
    run_index = 0

    for env_name in settings.envs:
        for variant in settings.variants:
            for seed in settings.seeds:
                run_index += 1
                config = build_run_config(
                    base_config=base_config,
                    env_name=env_name,
                    seed=seed,
                    variant=variant,
                    settings=settings,
                )
                variant_name = "PER-DQN" if config.use_per else "DQN"
                print(
                    f"\n[{run_index}/{total_runs}] Running {variant_name} on "
                    f"{env_name} with seed {seed}"
                )
                summary = train(config)

                run_record = {
                    "env_name": env_name,
                    "variant": variant_name,
                    "seed": seed,
                    "run_dir": os.path.dirname(config.save_dir),
                    "config_path": os.path.join(config.log_dir, "config.json"),
                    "summary_path": os.path.join(config.log_dir, "run_summary.json"),
                    "metrics_path": os.path.join(config.log_dir, "metrics.json"),
                    "summary": summary,
                }
                manifest["runs"].append(run_record)

                # 每完成一次 run 就落盘 manifest，便于中途中断后追踪结果。
                with open(
                    os.path.join(settings.output_root, "experiment_manifest.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\nAll experiment runs completed.")
    print(
        "Manifest saved to "
        f"{os.path.join(settings.output_root, 'experiment_manifest.json')}"
    )


if __name__ == "__main__":
    main()
