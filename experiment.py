"""批量运行 DQN 与 PER-DQN 对比实验。"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import multiprocessing as mp

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
    envs: list[str] = field(default_factory=lambda: ["CartPole-v1", "Taxi-v3", "MountainCar-v0"])
    seeds: list[int] = field(default_factory=lambda: [42])
    variants: list[str] = field(default_factory=lambda: ["dqn", "per"])
    output_root: str = "experiments"
    max_workers: int = 1

    def __post_init__(self):
        self.max_workers = max(1, int(self.max_workers))
        invalid_variants = [variant for variant in self.variants if variant not in VARIANT_LABELS]
        if invalid_variants:
            raise ValueError(
                "Unsupported variants: "
                f"{invalid_variants}. Expected one of {list(VARIANT_LABELS)}."
            )
        run_keys = list(self.iter_runs())
        if len(run_keys) != len(set(run_keys)):
            raise ValueError("Duplicate (env, variant, seed) runs are not allowed.")

    def manifest_path(self) -> str:
        return os.path.join(self.output_root, "experiment_manifest.json")

    def experiment_metadata(self) -> dict:
        return {
            "envs": self.envs,
            "seeds": self.seeds,
            "variants": self.variants,
            "output_root": self.output_root,
            "max_workers": self.max_workers,
        }

    def iter_runs(self):
        for env_name, variant, seed in product(self.envs, self.variants, self.seeds):
            yield env_name, variant, seed


@dataclass(frozen=True)
class RunSpec:
    env_name: str
    variant: str
    seed: int


def _run_single_experiment(
    base_config: DQNConfig,
    settings: ExperimentSettings,
    run_spec: RunSpec,
) -> tuple[DQNConfig, dict[str, Any]]:
    config = build_run_config(
        base_config=base_config,
        env_name=run_spec.env_name,
        seed=run_spec.seed,
        variant=run_spec.variant,
        settings=settings,
    )
    summary = train(config)
    return config, summary


def _run_label(run_spec: RunSpec) -> str:
    return (
        f"{VARIANT_LABELS[run_spec.variant]} on {run_spec.env_name} "
        f"with seed {run_spec.seed}"
    )


def _run_specs(settings: ExperimentSettings) -> list[RunSpec]:
    return [RunSpec(env_name, variant, seed) for env_name, variant, seed in settings.iter_runs()]


def _print_run_header(index: int, total_runs: int, run_spec: RunSpec) -> None:
    print(f"\n[{index}/{total_runs}] Running {_run_label(run_spec)}")


def _execute_runs(settings: ExperimentSettings) -> list[tuple[DQNConfig, dict[str, Any]]]:
    run_specs = _run_specs(settings)
    total_runs = len(run_specs)
    if settings.max_workers == 1:
        results = []
        for run_index, run_spec in enumerate(run_specs, start=1):
            _print_run_header(run_index, total_runs, run_spec)
            results.append(_run_single_experiment(settings.base_config, settings, run_spec))
        return results

    results_by_spec: dict[RunSpec, tuple[DQNConfig, dict[str, Any]]] = {}
    with ProcessPoolExecutor(
        max_workers=settings.max_workers,
        mp_context=mp.get_context("spawn"),
    ) as executor:
        future_to_spec = {
            executor.submit(_run_single_experiment, settings.base_config, settings, run_spec): run_spec
            for run_spec in run_specs
        }
        completed = 0
        for future in as_completed(future_to_spec):
            run_spec = future_to_spec[future]
            completed += 1
            print(f"\n[{completed}/{total_runs}] Finished {_run_label(run_spec)}")
            results_by_spec[run_spec] = future.result()

    return [results_by_spec[run_spec] for run_spec in run_specs]


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

    config = base_config.clone()
    config.core.env_name = env_name
    config.core.seed = seed
    config.replay.use_per = variant == "per"
    config.logging.save_dir = os.path.join(run_dir, "models")
    config.logging.log_dir = os.path.join(run_dir, "logs")
    _apply_environment_defaults(config, env_name)
    return config


def _apply_environment_defaults(config: DQNConfig, env_name: str) -> None:
    if env_name == "CartPole-v1":
        config.core.env_family = "vector"
        config.core.max_steps = 20_000
        config.training.hidden_sizes = (128, 128)
        config.replay.obs_encoding = "identity"
        config.replay.buffer_size = 20_000
        config.replay.training_start_steps = 1_000
        config.replay.initial_random_steps = 1_000
        config.replay.epsilon_decay = 10_000
        config.evaluation.eval_interval_env_steps = 2_000
        config.evaluation.eval_episodes = 10
        config.evaluation.eval_max_episode_steps = 500
        config.evaluation.success_threshold = 475.0
        return
    if env_name == "Taxi-v3":
        config.core.env_family = "discrete"
        config.core.max_steps = 50_000
        config.training.hidden_sizes = (256, 256)
        config.replay.obs_encoding = "one_hot"
        config.replay.buffer_size = 50_000
        config.replay.training_start_steps = 2_000
        config.replay.initial_random_steps = 2_000
        config.replay.epsilon_decay = 25_000
        config.evaluation.eval_interval_env_steps = 5_000
        config.evaluation.eval_episodes = 20
        config.evaluation.eval_max_episode_steps = 200
        config.evaluation.success_threshold = 8.0
        return
    if env_name == "MountainCar-v0":
        config.core.env_family = "vector"
        config.core.max_steps = 80_000
        config.training.hidden_sizes = (128, 128)
        config.replay.obs_encoding = "identity"
        config.replay.buffer_size = 50_000
        config.replay.training_start_steps = 2_000
        config.replay.initial_random_steps = 2_000
        config.replay.epsilon_decay = 40_000
        config.evaluation.eval_interval_env_steps = 5_000
        config.evaluation.eval_episodes = 10
        config.evaluation.eval_max_episode_steps = 200
        config.evaluation.success_threshold = -120.0
        return


def create_manifest(settings: ExperimentSettings, base_config: DQNConfig) -> dict:
    return {
        "envs": settings.envs,
        "seeds": settings.seeds,
        "variants": settings.variants,
        "base_config": base_config.to_flat_dict(),
        "experiment_settings": settings.experiment_metadata(),
        "runs": [],
    }


def append_run_record(manifest: dict, config: DQNConfig, summary: dict) -> None:
    variant_label = VARIANT_LABELS["per"] if config.replay.use_per else VARIANT_LABELS["dqn"]
    manifest["runs"].append(
        {
            "env_name": config.core.env_name,
            "variant": variant_label,
            "seed": config.core.seed,
            "run_dir": os.path.dirname(config.logging.save_dir),
            "config_path": os.path.join(config.logging.log_dir, "config.json"),
            "summary_path": os.path.join(config.logging.log_dir, "run_summary.json"),
            "metrics_path": os.path.join(config.logging.log_dir, "metrics.json"),
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
        envs=["CartPole-v1", "Taxi-v3", "MountainCar-v0"],
        seeds=[42],
        variants=["dqn", "per"],
        max_workers=1,
    )

    os.makedirs(settings.output_root, exist_ok=True)
    manifest = create_manifest(settings, settings.base_config)

    for config, summary in _execute_runs(settings):
        append_run_record(manifest, config, summary)
        save_manifest(manifest, settings.manifest_path())

    print("\nAll experiment runs completed.")
    print(f"Manifest saved to {settings.manifest_path()}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
