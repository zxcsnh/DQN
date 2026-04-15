"""DQN agent implementation."""

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .network import DQNCNN
from .replay_buffer import PrioritizedReplayBuffer, ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        num_actions: int,
        input_channels: int = 4,
        device: str = "cpu",
        learning_rate: float = 2.5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 1000000,
        buffer_size: int = 100000,
        batch_size: int = 32,
        target_update_freq: int = 10000,
        use_per: bool = False,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 100000,
        soft_update: bool = False,
        tau: float = 0.005,
    ):
        self.num_actions = num_actions
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_per = use_per
        self.soft_update = soft_update
        self.tau = tau

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.policy_net = DQNCNN(input_channels, num_actions).to(device)
        self.target_net = DQNCNN(input_channels, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        if use_per:
            self.memory = PrioritizedReplayBuffer(
                buffer_size,
                alpha=per_alpha,
                beta_start=per_beta_start,
                beta_frames=per_beta_frames,
            )
        else:
            self.memory = ReplayBuffer(buffer_size)

        # optimization steps for target-network schedule
        self.steps_done = 0
        # environment interaction steps for epsilon schedule
        self.env_steps_done = 0
        self.episodes_done = 0

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        if not evaluate and np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            if len(state_tensor.shape) == 3:
                state_tensor = state_tensor.unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def on_env_step(self):
        """Track one environment interaction step and update epsilon."""
        self.env_steps_done += 1
        self.update_epsilon()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
    ):
        self.memory.push(state, action, reward, next_state, terminated, truncated)

    def compute_loss(self, batch: Tuple) -> Dict[str, torch.Tensor]:
        states, actions, rewards, next_states, terminated, _truncated = batch

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        terminated = torch.tensor(terminated, dtype=torch.float32, device=self.device)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1)[0]
            target_q = rewards + (1 - terminated) * self.gamma * next_q

        td_errors = target_q - current_q
        per_sample_loss = nn.functional.smooth_l1_loss(
            current_q, target_q, reduction="none"
        )

        return {
            "loss": per_sample_loss.mean(),
            "per_sample_loss": per_sample_loss,
            "td_errors": td_errors,
        }

    def train_step(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None

        if self.use_per:
            (
                states,
                actions,
                rewards,
                next_states,
                terminated,
                truncated,
                indices,
                weights,
            ) = self.memory.sample(self.batch_size)
            batch = (states, actions, rewards, next_states, terminated, truncated)
            weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        else:
            batch = self.memory.sample(self.batch_size)

        loss_info = self.compute_loss(batch)
        if self.use_per:
            loss = (loss_info["per_sample_loss"] * weights).mean()
        else:
            loss = loss_info["loss"]

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        self.steps_done += 1

        if self.use_per:
            priorities = loss_info["td_errors"].detach().abs().cpu().numpy() + 1e-6
            self.memory.update_priorities(indices, priorities)

        if self.soft_update:
            self.soft_update_target_network()
        elif self.steps_done % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    def update_epsilon(self):
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * max(
            0, (self.epsilon_decay - self.env_steps_done) / self.epsilon_decay
        )

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(f"Target network updated at optimization step {self.steps_done}")

    def soft_update_target_network(self):
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                (1.0 - self.tau) * target_param.data + self.tau * policy_param.data
            )

    def save(self, path: str, save_replay_buffer: bool = False):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps_done": self.steps_done,
            "env_steps_done": self.env_steps_done,
            "episodes_done": self.episodes_done,
            "epsilon": self.epsilon,
            "use_per": self.use_per,
        }
        if save_replay_buffer:
            checkpoint["memory_state"] = self.memory.state_dict()

        torch.save(
            checkpoint,
            path,
        )
        print(f"Model saved to {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.steps_done = checkpoint.get("steps_done", 0)
        self.env_steps_done = checkpoint.get("env_steps_done", self.steps_done)
        self.episodes_done = checkpoint.get("episodes_done", 0)
        self.epsilon = checkpoint.get("epsilon", self.epsilon)

        checkpoint_use_per = checkpoint.get("use_per", self.use_per)
        if checkpoint_use_per != self.use_per:
            print(
                "Warning: checkpoint replay mode does not match current agent. "
                "Replay buffer state was not restored."
            )
        elif "memory_state" in checkpoint:
            self.memory.load_state_dict(checkpoint["memory_state"])

        print(f"Model loaded from {path}")

    def eval_mode(self):
        self.policy_net.eval()
        self.target_net.eval()

    def train_mode(self):
        self.policy_net.train()
        self.target_net.eval()


if __name__ == "__main__":
    agent = DQNAgent(
        num_actions=6,
        input_channels=4,
        device="cpu",
    )

    dummy_state = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
    action = agent.select_action(dummy_state)
    print(f"Selected action: {action}")

    dummy_next_state = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
    agent.store_transition(dummy_state, action, 1.0, dummy_next_state, False)

    print(f"Buffer size: {len(agent.memory)}")
    print(f"Epsilon: {agent.epsilon}")
