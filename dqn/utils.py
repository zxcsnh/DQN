"""Training logging, plotting, and random seed utilities."""

from __future__ import annotations

import json
import os
from typing import List, Optional, TextIO

import matplotlib.pyplot as plt
import numpy as np


class TrainingLogger:
    """Track training metrics with optional sampled step-level logging."""

    def __init__(
        self,
        log_dir: str = "runs",
        log_step_metrics: bool = True,
        step_log_interval: int = 1000,
        step_log_stream: bool = True,
        step_log_file: str = "step_metrics.jsonl",
    ):
        self.log_dir = log_dir
        self.log_step_metrics = bool(log_step_metrics)
        self.step_log_interval = max(1, int(step_log_interval))
        self.step_log_stream = bool(step_log_stream)
        self.step_log_file = step_log_file

        os.makedirs(log_dir, exist_ok=True)

        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_losses: List[float] = []
        self.epsilons: List[float] = []
        self.avg_rewards: List[float] = []
        self.eval_rewards: List[float] = []
        self.eval_steps: List[int] = []
        self.global_steps: List[int] = []
        self.step_losses: List[float] = []
        self.step_epsilons: List[float] = []

        # Keep per-update losses only within one episode for episode-level loss stats.
        self.current_episode_loss: List[float] = []

        self._step_accum_count = 0
        self._step_accum_loss_sum = 0.0
        self._step_accum_epsilon_sum = 0.0
        self._step_accum_last_step = 0

        self.step_log_path: str | None = (
            os.path.join(self.log_dir, self.step_log_file)
            if self.log_step_metrics and self.step_log_stream
            else None
        )
        self._step_stream_fp: TextIO | None = None
        if self.step_log_path is not None:
            self._step_stream_fp = open(self.step_log_path, "w", encoding="utf-8")

    def log_step(self, step: int, epsilon: float, loss: Optional[float] = None):
        if loss is None:
            return

        loss_value = float(loss)
        epsilon_value = float(epsilon)
        self.current_episode_loss.append(loss_value)

        if not self.log_step_metrics:
            return

        self._step_accum_count += 1
        self._step_accum_loss_sum += loss_value
        self._step_accum_epsilon_sum += epsilon_value
        self._step_accum_last_step = int(step)
        if self._step_accum_count >= self.step_log_interval:
            self._flush_step_aggregate()

    def log_episode(self, episode: int, reward: float, length: int, epsilon: float):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.epsilons.append(epsilon)

        avg_loss = (
            np.mean(self.current_episode_loss) if self.current_episode_loss else 0.0
        )
        self.episode_losses.append(float(avg_loss))
        self.current_episode_loss = []

        avg_reward = float(np.mean(self.episode_rewards[-100:]))
        self.avg_rewards.append(avg_reward)

    def log_evaluation(self, step: int, reward: float):
        self.eval_steps.append(step)
        self.eval_rewards.append(float(reward))

    def plot_training_curves(self, save_path: Optional[str] = None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax1 = axes[0, 0]
        ax1.plot(self.episode_rewards, alpha=0.6, label="reward")
        ax1.plot(self.avg_rewards, color="red", label="avg (100 eps)")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax1.set_title("Episode Reward")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        ax2.plot(self.episode_lengths, alpha=0.6)
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Steps")
        ax2.set_title("Episode Length")
        ax2.grid(True, alpha=0.3)

        ax3 = axes[1, 0]
        ax3.plot(self.episode_losses, alpha=0.6)
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Loss")
        ax3.set_title("Training Loss")
        ax3.grid(True, alpha=0.3)

        ax4 = axes[1, 1]
        ax4.plot(self.epsilons, alpha=0.6)
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Epsilon")
        ax4.set_title("Epsilon Decay")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Training curves saved to {save_path}")

        plt.close(fig)

    def save_metrics(self, path: str):
        self._flush_step_aggregate()
        metrics = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "episode_losses": self.episode_losses,
            "epsilons": self.epsilons,
            "avg_rewards": self.avg_rewards,
            "eval_steps": self.eval_steps,
            "eval_rewards": self.eval_rewards,
            "global_steps": self.global_steps,
            "step_losses": self.step_losses,
            "step_epsilons": self.step_epsilons,
            "step_log_interval": self.step_log_interval,
            "step_log_path": self.step_log_path,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    def load_metrics(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        self.episode_rewards = metrics["episode_rewards"]
        self.episode_lengths = metrics["episode_lengths"]
        self.episode_losses = metrics["episode_losses"]
        self.epsilons = metrics["epsilons"]
        self.avg_rewards = metrics["avg_rewards"]
        self.eval_steps = metrics.get("eval_steps", [])
        self.eval_rewards = metrics.get("eval_rewards", [])
        self.global_steps = metrics.get("global_steps", [])
        self.step_losses = metrics.get("step_losses", [])
        self.step_epsilons = metrics.get("step_epsilons", [])
        self.step_log_interval = int(
            metrics.get("step_log_interval", self.step_log_interval)
        )
        self.step_log_path = metrics.get("step_log_path", self.step_log_path)

    def close(self):
        self._flush_step_aggregate()
        if self._step_stream_fp is not None:
            self._step_stream_fp.close()
            self._step_stream_fp = None

    def _flush_step_aggregate(self) -> None:
        if self._step_accum_count <= 0:
            return

        avg_loss = self._step_accum_loss_sum / self._step_accum_count
        avg_epsilon = self._step_accum_epsilon_sum / self._step_accum_count

        self.global_steps.append(self._step_accum_last_step)
        self.step_losses.append(float(avg_loss))
        self.step_epsilons.append(float(avg_epsilon))

        if self._step_stream_fp is not None:
            record = {
                "step": self._step_accum_last_step,
                "avg_loss": float(avg_loss),
                "avg_epsilon": float(avg_epsilon),
                "num_updates": self._step_accum_count,
            }
            self._step_stream_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._step_stream_fp.flush()

        self._step_accum_count = 0
        self._step_accum_loss_sum = 0.0
        self._step_accum_epsilon_sum = 0.0


def print_episode_stats(
    episode: int,
    reward: float,
    length: int,
    epsilon: float,
    avg_reward: float,
    best_reward: float,
    loss: float = 0.0,
):
    print(
        f"Episode {episode:4d} | "
        f"Reward: {reward:7.2f} | "
        f"Length: {length:5d} | "
        f"Avg100: {avg_reward:7.2f} | "
        f"Best: {best_reward:7.2f} | "
        f"Epsilon: {epsilon:.4f} | "
        f"Loss: {loss:.4f}"
    )


def set_seed(
    seed: int,
    deterministic_torch: bool = False,
    allow_tf32: bool = True,
):
    """Try to fix random seeds in Python, NumPy and PyTorch."""
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic_torch
        torch.backends.cudnn.benchmark = not deterministic_torch
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = allow_tf32
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
