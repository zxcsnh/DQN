from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from ..buffers.replay_buffer import ReplayBuffer
from ..q_network import QNetwork


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        common_config,
        env_config,
        env_name: str,
        algo_name: str = "dqn",
        device: str | None = None,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = common_config.gamma
        self.batch_size = common_config.batch_size
        self.min_replay_size = int(env_config.min_replay_size)
        self.epsilon_start = common_config.epsilon_start
        self.epsilon = self.epsilon_start
        self.epsilon_end = common_config.epsilon_end
        self.epsilon_decay_steps = max(1, int(env_config.epsilon_decay_steps))
        self.target_update_freq = common_config.target_update_freq
        self.gradient_clip_norm = common_config.gradient_clip_norm
        self.env_name = env_name
        self.algo_name = algo_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.q_network = QNetwork(state_dim, action_dim, common_config.hidden_dim).to(self.device)
        self.target_q_network = QNetwork(state_dim, action_dim, common_config.hidden_dim).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()

        self.optimizer = Adam(self.q_network.parameters(), lr=common_config.learning_rate)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")
        self.replay_buffer = ReplayBuffer(common_config.replay_buffer_size)
        self.train_steps = 0

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def store_transition(self, state, action, reward, next_state, done) -> None:
        self.replay_buffer.push(state, action, reward, next_state, done)

    def _update_epsilon(self) -> None:
        progress = min(1.0, self.train_steps / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress

    def update(self):
        if len(self.replay_buffer) < max(self.min_replay_size, self.batch_size):
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        current_q = self.q_network(states_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            next_q = self.target_q_network(next_states_t).max(dim=1).values
            target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = self.loss_fn(current_q, target_q).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.gradient_clip_norm)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        self._update_epsilon()
        return float(loss.item())

    def save(self, path) -> None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.q_network.state_dict(),
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "env_name": self.env_name,
                "algo_name": self.algo_name,
            },
            save_path,
        )

    def load(self, path) -> None:
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"未找到模型文件: {load_path}")
        checkpoint = torch.load(load_path, map_location=self.device)
        if "model_state_dict" not in checkpoint:
            raise KeyError("模型文件缺少 model_state_dict 字段。")
        self.q_network.load_state_dict(checkpoint["model_state_dict"])
        self.target_q_network.load_state_dict(checkpoint["model_state_dict"])
