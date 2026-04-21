"""DQN agent implementation."""

from __future__ import annotations

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
        initial_random_steps: int = 100_000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: int = 1_000_000,
        buffer_size: int = 1_000_000,
        batch_size: int = 32,
        target_update_interval_updates: int = 10_000,
        use_per: bool = False,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_updates: int = 100_000,
        use_soft_target_update: bool = False,
        soft_target_update_tau: float = 0.005,
        frame_stack: int = 4,
        input_shape: tuple[int, int] = (84, 84),
        optimizer_name: str = "rmsprop",
        rmsprop_alpha: float = 0.95,
        rmsprop_eps: float = 0.01,
        rmsprop_centered: bool = True,
        use_double_dqn: bool = False,
        replay_sample_torch_fastpath: bool = True,
        seed: Optional[int] = None,
    ):
        self.num_actions = num_actions
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_interval_updates = target_update_interval_updates
        self.use_per = use_per
        self.use_soft_target_update = use_soft_target_update
        self.soft_target_update_tau = soft_target_update_tau
        self.input_channels = input_channels
        self.input_shape = input_shape
        self.optimizer_name = optimizer_name
        self.use_double_dqn = use_double_dqn
        self.replay_sample_torch_fastpath = replay_sample_torch_fastpath
        self.initial_random_steps = max(0, initial_random_steps)

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.policy_net = DQNCNN(
            input_channels,
            num_actions,
            input_shape=input_shape,
        ).to(device)
        self.target_net = DQNCNN(
            input_channels,
            num_actions,
            input_shape=input_shape,
        ).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = self._build_optimizer(
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            rmsprop_alpha=rmsprop_alpha,
            rmsprop_eps=rmsprop_eps,
            rmsprop_centered=rmsprop_centered,
        )

        if use_per:
            self.memory = PrioritizedReplayBuffer(
                buffer_size,
                frame_stack=frame_stack,
                alpha=per_alpha,
                beta_start=per_beta_start,
                beta_updates=per_beta_updates,
                seed=seed,
            )
        else:
            self.memory = ReplayBuffer(
                buffer_size,
                frame_stack=frame_stack,
                seed=seed,
            )

        # Optimization steps drive target-network updates.
        self.steps_done = 0
        # Environment steps drive epsilon decay and training schedule.
        self.env_steps_done = 0
        self.episodes_done = 0

    def _build_optimizer(
        self,
        optimizer_name: str,
        learning_rate: float,
        rmsprop_alpha: float,
        rmsprop_eps: float,
        rmsprop_centered: bool,
    ) -> optim.Optimizer:
        optimizer_name = optimizer_name.lower()
        if optimizer_name == "rmsprop":
            # This is the closest PyTorch equivalent to the centered RMSProp
            # variant used by Nature DQN.
            return optim.RMSprop(
                self.policy_net.parameters(),
                lr=learning_rate,
                alpha=rmsprop_alpha,
                eps=rmsprop_eps,
                centered=rmsprop_centered,
                momentum=0.0,
            )
        if optimizer_name == "adam":
            return optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        raise ValueError(f"Unsupported optimizer_name: {optimizer_name}")

    def select_action(
        self,
        state: np.ndarray,
        evaluate: bool = False,
        epsilon_override: float | None = None,
    ) -> int:
        """Choose an action with epsilon-greedy exploration."""
        if epsilon_override is not None:
            epsilon = epsilon_override
        elif evaluate:
            epsilon = 0.0
        else:
            if self.env_steps_done < self.initial_random_steps:
                return int(np.random.randint(self.num_actions))
            epsilon = self.epsilon

        if np.random.random() < epsilon:
            return int(np.random.randint(self.num_actions))

        with torch.no_grad():
            state_tensor = torch.as_tensor(state, device=self.device)
            if len(state_tensor.shape) == 3:
                state_tensor = state_tensor.unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return int(q_values.argmax(dim=1).item())

    def begin_episode(self, initial_state: np.ndarray) -> None:
        """Notify the replay buffer that a new episode has started."""
        self.memory.start_episode(initial_state)

    def on_env_step(self) -> None:
        """Record an environment interaction and update epsilon."""
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
    ) -> None:
        self.memory.push(state, action, reward, next_state, terminated, truncated)

    def compute_loss(self, batch: Tuple) -> Dict[str, torch.Tensor]:
        states, actions, rewards, next_states, terminated, truncated = batch

        # Only true MDP termination should stop bootstrapping. Time-limit or wrapper
        # truncation still ends the sampled episode boundary, but does not zero the
        # Bellman target in the current training protocol.
        _ = truncated

        states = self._to_device_tensor(states)
        actions = self._to_device_tensor(actions, dtype=torch.long)
        rewards = self._to_device_tensor(rewards, dtype=torch.float32)
        next_states = self._to_device_tensor(next_states)
        terminated = self._to_device_tensor(terminated, dtype=torch.float32)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.use_double_dqn:
                next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_net(next_states).max(dim=1).values

            target_q = rewards + (1.0 - terminated) * self.gamma * next_q

        td_errors = target_q - current_q
        # Nature DQN clips the TD error to [-1, 1]. Smooth L1 is equivalent.
        per_sample_loss = nn.functional.smooth_l1_loss(
            current_q,
            target_q,
            reduction="none",
        )

        return {
            "loss": per_sample_loss.mean(),
            "per_sample_loss": per_sample_loss,
            "td_errors": td_errors,
        }

    def _to_device_tensor(
        self,
        value: np.ndarray | torch.Tensor,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        target_device = torch.device(self.device)
        if isinstance(value, torch.Tensor):
            if value.device != target_device or (
                dtype is not None and value.dtype != dtype
            ):
                value = value.to(
                    device=target_device,
                    dtype=dtype if dtype is not None else value.dtype,
                    non_blocking=True,
                )
            return value
        return torch.as_tensor(value, dtype=dtype, device=target_device)

    def train_step(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None

        sample_fn = None
        if self.replay_sample_torch_fastpath:
            sample_fn = getattr(self.memory, "sample_tensors", None)

        if self.use_per and sample_fn is not None:
            (
                states,
                actions,
                rewards,
                next_states,
                terminated,
                truncated,
                indices,
                weights,
            ) = sample_fn(self.batch_size, device=self.device)
            batch = (states, actions, rewards, next_states, terminated, truncated)
            priority_indices = indices.detach().cpu().numpy()
        elif self.use_per:
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
            weights = self._to_device_tensor(weights, dtype=torch.float32)
            priority_indices = indices
        elif sample_fn is not None:
            batch = sample_fn(self.batch_size, device=self.device)
        else:
            batch = self.memory.sample(self.batch_size)

        loss_info = self.compute_loss(batch)
        if self.use_per:
            loss = (loss_info["per_sample_loss"] * weights).mean()
        else:
            loss = loss_info["loss"]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1

        if self.use_per:
            priorities = loss_info["td_errors"].detach().abs().cpu().numpy() + 1e-6
            self.memory.update_priorities(priority_indices, priorities)

        if self.use_soft_target_update:
            self.soft_update_target_network()
        elif self.steps_done % self.target_update_interval_updates == 0:
            self.update_target_network()

        return float(loss.item())

    def update_epsilon(self) -> None:
        if self.env_steps_done <= self.initial_random_steps:
            self.epsilon = self.epsilon_start
            return
        if self.epsilon_decay <= 0:
            self.epsilon = self.epsilon_end
            return

        epsilon_step = self.env_steps_done - self.initial_random_steps
        remaining = max(0, self.epsilon_decay - epsilon_step)
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * (
            remaining / self.epsilon_decay
        )

    def update_target_network(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(f"Target network updated at optimization step {self.steps_done}")

    def soft_update_target_network(self) -> None:
        for target_param, policy_param in zip(
            self.target_net.parameters(),
            self.policy_net.parameters(),
        ):
            target_param.data.copy_(
                (1.0 - self.soft_target_update_tau) * target_param.data
                + self.soft_target_update_tau * policy_param.data
            )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_artifact = {
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "use_per": self.use_per,
            "use_double_dqn": self.use_double_dqn,
            "optimizer_name": self.optimizer_name,
            "model_config": {
                "input_channels": self.input_channels,
                "input_shape": self.input_shape,
                "num_actions": self.num_actions,
            },
        }
        torch.save(model_artifact, path)
        print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        artifact = torch.load(path, map_location=self.device, weights_only=False)
        policy_state = artifact.get("policy_net")
        if policy_state is None:
            policy_state = artifact.get("model_state_dict")
        if policy_state is None:
            raise ValueError(f"File does not contain model weights: {path}")

        self.policy_net.load_state_dict(policy_state)
        target_state = artifact.get("target_net")
        if target_state is None:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            self.target_net.load_state_dict(target_state)
        print(f"Model loaded from {path}")

    def eval_mode(self) -> None:
        self.policy_net.eval()
        self.target_net.eval()

    def train_mode(self) -> None:
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
    agent.begin_episode(dummy_state)
    agent.store_transition(dummy_state, action, 1.0, dummy_next_state, False, False)

    print(f"Buffer size: {len(agent.memory)}")
    print(f"Epsilon: {agent.epsilon}")
