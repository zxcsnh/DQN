"""Replay buffer implementations for vector-observation DQN and PER-DQN."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class ReplayBatch:
    states: np.ndarray | torch.Tensor
    actions: np.ndarray | torch.Tensor
    rewards: np.ndarray | torch.Tensor
    next_states: np.ndarray | torch.Tensor
    terminated: np.ndarray | torch.Tensor
    truncated: np.ndarray | torch.Tensor
    indices: np.ndarray | torch.Tensor | None = None
    weights: np.ndarray | torch.Tensor | None = None


class SumTree:
    """Prefix-sum tree used for prioritized replay sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float32)

    def total(self) -> float:
        return float(self.tree[1])

    def update(self, index: int, value: float) -> None:
        tree_index = index + self.capacity
        delta = value - self.tree[tree_index]
        self.tree[tree_index] = value

        tree_index //= 2
        while tree_index >= 1:
            self.tree[tree_index] += delta
            tree_index //= 2

    def get(self, mass: float) -> int:
        index = 1
        while index < self.capacity:
            left = index * 2
            if mass <= self.tree[left]:
                index = left
            else:
                mass -= self.tree[left]
                index = left + 1
        return index - self.capacity


class ReplayBuffer:
    """Uniform replay buffer storing full vector states."""

    def __init__(
        self,
        capacity: int = 100000,
        obs_shape: tuple[int, ...] = (4,),
        seed: Optional[int] = None,
    ):
        self.capacity = int(capacity)
        self.obs_shape = tuple(int(v) for v in obs_shape)
        self.rng = random.Random(seed)

        self.states = np.empty((self.capacity, *self.obs_shape), dtype=np.float32)
        self.next_states = np.empty((self.capacity, *self.obs_shape), dtype=np.float32)
        self.actions = np.empty(self.capacity, dtype=np.int64)
        self.rewards = np.empty(self.capacity, dtype=np.float32)
        self.terminated = np.empty(self.capacity, dtype=np.bool_)
        self.truncated = np.empty(self.capacity, dtype=np.bool_)

        self.position = 0
        self.size = 0

    def start_episode(self, initial_state: np.ndarray, stream_id: int = 0) -> None:
        _ = initial_state
        _ = stream_id

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
        stream_id: int = 0,
    ) -> None:
        _ = stream_id
        self.states[self.position] = np.asarray(state, dtype=np.float32)
        self.next_states[self.position] = np.asarray(next_state, dtype=np.float32)
        self.actions[self.position] = int(action)
        self.rewards[self.position] = float(reward)
        self.terminated[self.position] = bool(terminated)
        self.truncated[self.position] = bool(truncated)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> ReplayBatch:
        slots = self._sample_uniform_slots(batch_size)
        return self._build_batch(slots)

    def sample_tensors(
        self,
        batch_size: int,
        device: str | torch.device,
    ) -> ReplayBatch:
        slots = self._sample_uniform_slots(batch_size)
        return self._build_batch_tensors(slots, device=device)

    def _sample_uniform_slots(self, batch_size: int) -> np.ndarray:
        if batch_size > self.size:
            raise ValueError("batch_size cannot exceed replay buffer size.")
        return np.array(self.rng.sample(range(self.size), batch_size), dtype=np.int64)

    def _build_batch(self, slots: np.ndarray) -> ReplayBatch:
        return ReplayBatch(
            states=self.states[slots].copy(),
            actions=self.actions[slots].copy(),
            rewards=self.rewards[slots].copy(),
            next_states=self.next_states[slots].copy(),
            terminated=self.terminated[slots].astype(np.float32),
            truncated=self.truncated[slots].astype(np.float32),
        )

    def _build_batch_tensors(
        self,
        slots: np.ndarray,
        device: str | torch.device,
    ) -> ReplayBatch:
        batch = self._build_batch(slots)
        return ReplayBatch(
            states=self._array_to_tensor(batch.states, device=device, dtype=torch.float32),
            actions=self._array_to_tensor(batch.actions, device=device, dtype=torch.long),
            rewards=self._array_to_tensor(batch.rewards, device=device, dtype=torch.float32),
            next_states=self._array_to_tensor(batch.next_states, device=device, dtype=torch.float32),
            terminated=self._array_to_tensor(batch.terminated, device=device, dtype=torch.float32),
            truncated=self._array_to_tensor(batch.truncated, device=device, dtype=torch.float32),
        )

    @staticmethod
    def _array_to_tensor(
        array: np.ndarray,
        device: str | torch.device,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        device = torch.device(device)
        tensor = torch.from_numpy(np.ascontiguousarray(array))
        if device.type == "cuda":
            tensor = tensor.to(device=device, non_blocking=True)
        else:
            tensor = tensor.to(device=device)
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        return tensor

    def __len__(self) -> int:
        return self.size

    def is_ready(self, min_size: int) -> bool:
        return self.size >= min_size


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized replay buffer for vector states."""

    def __init__(
        self,
        capacity: int = 100000,
        obs_shape: tuple[int, ...] = (4,),
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_updates: int = 100000,
        seed: Optional[int] = None,
    ):
        super().__init__(capacity=capacity, obs_shape=obs_shape, seed=seed)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_updates = max(1, beta_updates)
        self.sample_updates_done = 0
        self.max_priority = 1.0
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.tree = SumTree(self.capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
        stream_id: int = 0,
    ) -> None:
        slot = self.position
        super().push(state, action, reward, next_state, terminated, truncated, stream_id=stream_id)
        self.priorities[slot] = self.max_priority
        self.tree.update(slot, self.max_priority)

    def sample(self, batch_size: int) -> ReplayBatch:
        slots, weights = self._sample_prioritized_slots(batch_size)
        batch = self._build_batch(slots)
        batch.indices = slots
        batch.weights = weights
        return batch

    def sample_tensors(
        self,
        batch_size: int,
        device: str | torch.device,
    ) -> ReplayBatch:
        slots, weights = self._sample_prioritized_slots(batch_size)
        batch = self._build_batch_tensors(slots, device=device)
        batch.indices = self._array_to_tensor(slots, device=device, dtype=torch.long)
        batch.weights = self._array_to_tensor(weights, device=device, dtype=torch.float32)
        return batch

    def _sample_prioritized_slots(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        if batch_size > self.size:
            raise ValueError("batch_size cannot exceed replay buffer size.")
        if self.tree.total() <= 0.0:
            raise ValueError("Cannot sample from prioritized replay before any priorities exist.")

        self.sample_updates_done += 1
        beta = min(
            1.0,
            self.beta_start + (1.0 - self.beta_start) * (self.sample_updates_done / self.beta_updates),
        )

        slots = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float32)
        segment = self.tree.total() / batch_size
        for i in range(batch_size):
            mass = self.rng.uniform(segment * i, segment * (i + 1))
            slot = self.tree.get(mass)
            slots[i] = slot
            priorities[i] = self.priorities[slot]

        probs = priorities / self.tree.total()
        weights = np.power(self.size * np.maximum(probs, 1e-12), -beta).astype(np.float32)
        weights /= weights.max()
        return slots, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        for index, priority in zip(indices, priorities, strict=False):
            value = float(np.power(max(float(priority), 1e-6), self.alpha))
            self.priorities[int(index)] = value
            self.tree.update(int(index), value)
            self.max_priority = max(self.max_priority, value)
