"""训练日志、可视化与随机种子工具。"""

import json
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


class TrainingLogger:
    """记录训练过程中常用的标量指标。"""

    def __init__(self, log_dir: str = "runs"):
        self.log_dir = log_dir

        os.makedirs(log_dir, exist_ok=True)

        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_losses: List[float] = []
        self.epsilons: List[float] = []
        self.avg_rewards: List[float] = []
        self.eval_rewards: List[float] = []
        self.eval_steps: List[int] = []

        # 先按 step 暂存 loss，等 episode 结束后再聚合成单个值。
        self.current_episode_loss: List[float] = []

    def log_step(self, step: int, epsilon: float, loss: Optional[float] = None):
        if loss is not None:
            self.current_episode_loss.append(loss)

    def log_episode(self, episode: int, reward: float, length: int, epsilon: float):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.epsilons.append(epsilon)

        avg_loss = (
            np.mean(self.current_episode_loss) if self.current_episode_loss else 0.0
        )
        self.episode_losses.append(float(avg_loss))
        self.current_episode_loss = []

        # 使用最近 100 个 episode 的均值作为平滑指标。
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
        metrics = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "episode_losses": self.episode_losses,
            "epsilons": self.epsilons,
            "avg_rewards": self.avg_rewards,
            "eval_steps": self.eval_steps,
            "eval_rewards": self.eval_rewards,
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

    def close(self):
        pass


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


def set_seed(seed: int):
    """尽量固定 Python、NumPy 与 PyTorch 的随机性。"""
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, episode: int, reward: float, path: str):
    import torch

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "episode": episode,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_reward": reward,
        },
        path,
    )


def load_checkpoint(model, optimizer, path: str, device: str = "cpu"):
    import torch

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {
        "episode": checkpoint.get("episode", 0),
        "best_reward": checkpoint.get("best_reward", 0),
    }
