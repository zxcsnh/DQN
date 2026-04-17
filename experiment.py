"""批量运行 DQN 与 PER-DQN 对比实验。"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from itertools import product

from config import DQNConfig
from train import train


VARIANT_LABELS = {
    "dqn": "DQN",
    "per": "PER-DQN",
}


@dataclass
class ExperimentSettings:
    """批量实验调度配置，与单次训练配置解耦。"""

    base_config: DQNConfig = field(default_factory=DQNConfig)
    envs: list[str] = field(default_factory=lambda: ["ALE/Pong-v5"])
    seeds: list[int] = field(default_factory=lambda: [42])
    variants: list[str] = field(default_factory=lambda: ["dqn", "per"])
    output_root: str = "experiments"

    def __post_init__(self):
        invalid_variants = [variant for variant in self.variants if variant not in VARIANT_LABELS]
        if invalid_variants:
            raise ValueError(
                "Unsupported variants: "
                f"{invalid_variants}. Expected one of {list(VARIANT_LABELS)}."
            )

    def manifest_path(self) -> str:
        return os.path.join(self.output_root, "experiment_manifest.json")

    def experiment_metadata(self) -> dict:
        return {
            "envs": self.envs,
            "seeds": self.seeds,
            "variants": self.variants,
            "output_root": self.output_root,
        }

    def iter_runs(self):
        for env_name, variant, seed in product(self.envs, self.variants, self.seeds):
            yield env_name, variant, seed


def build_run_config(
    base_config: DQNConfig,
    env_name: str,
    seed: int,
    variant: str,
    settings: ExperimentSettings,
) -> DQNConfig:
    """根据实验设置，为每一次 run 生成独立训练配置。"""
    variant_name = "per_dqn" if variant == "per" else "dqn"
    env_slug = env_name.replace("/", "_").replace(":", "_")
    run_dir = os.path.join(settings.output_root, env_slug, variant_name, f"seed_{seed}")

    return replace(
        base_config,
        env_name=env_name,
        seed=seed,
        use_per=(variant == "per"),
        save_dir=os.path.join(run_dir, "models"),
        log_dir=os.path.join(run_dir, "logs"),
    )


def create_manifest(settings: ExperimentSettings, base_config: DQNConfig) -> dict:
    return {
        "envs": settings.envs,
        "seeds": settings.seeds,
        "variants": settings.variants,
        "base_config": asdict(base_config),
        "experiment_settings": settings.experiment_metadata(),
        "runs": [],
    }


def append_run_record(manifest: dict, config: DQNConfig, summary: dict) -> None:
    variant_label = VARIANT_LABELS["per"] if config.use_per else VARIANT_LABELS["dqn"]
    manifest["runs"].append(
        {
            "env_name": config.env_name,
            "variant": variant_label,
            "seed": config.seed,
            "run_dir": os.path.dirname(config.save_dir),
            "config_path": os.path.join(config.log_dir, "config.json"),
            "summary_path": os.path.join(config.log_dir, "run_summary.json"),
            "metrics_path": os.path.join(config.log_dir, "metrics.json"),
            "summary": summary,
        }
    )


def save_manifest(manifest: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def main():
    base_config = DQNConfig()
    settings = ExperimentSettings(
        base_config=base_config,
        envs=["ALE/Pong-v5"],
        seeds=[42],
        variants=["dqn", "per"],
    )

    os.makedirs(settings.output_root, exist_ok=True)
    manifest = create_manifest(settings, settings.base_config)

    total_runs = len(settings.envs) * len(settings.variants) * len(settings.seeds)
    run_index = 0

    for env_name, variant, seed in settings.iter_runs():
        run_index += 1
        config = build_run_config(
            base_config=settings.base_config,
            env_name=env_name,
            seed=seed,
            variant=variant,
            settings=settings,
        )
        variant_label = VARIANT_LABELS[variant]
        print(
            f"\n[{run_index}/{total_runs}] Running {variant_label} on "
            f"{env_name} with seed {seed}"
        )
        summary = train(config)

        append_run_record(manifest, config, summary)
        save_manifest(manifest, settings.manifest_path())

    print("\nAll experiment runs completed.")
    print(f"Manifest saved to {settings.manifest_path()}")


if __name__ == "__main__":
    main()
