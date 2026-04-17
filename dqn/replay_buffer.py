"""DQN 与 PER 使用的经验回放缓冲区实现。"""

from __future__ import annotations

import random
from collections import deque
from typing import Optional, Tuple

import numpy as np
import torch


class SumTree:
    """用于 PER 的前缀和树。"""

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
    """按帧去重存储的均匀采样经验回放。"""

    def __init__(
        self,
        capacity: int = 100000,
        frame_stack: int = 4,
        seed: Optional[int] = None,
    ):
        self.capacity = capacity
        self.frame_stack = frame_stack

        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.bool_)
        self.truncated = np.empty(capacity, dtype=np.bool_)
        self.state_indices = np.empty(capacity, dtype=np.int64)
        self.episode_start_indices = np.empty(capacity, dtype=np.int64)

        self.position = 0
        self.size = 0

        # 只保存每个环境状态的“最新一帧”，按需重建 frame stack。
        self.frames: dict[int, np.ndarray] = {}
        self.frame_queue: deque[int] = deque()
        self.next_state_index = 0
        self.current_state_idx: Optional[int] = None
        self.current_episode_start_idx: Optional[int] = None

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def start_episode(self, initial_state: np.ndarray) -> None:
        """在每个 episode reset 后注册初始观测。"""
        state_idx = self._store_frame(initial_state)
        self.current_state_idx = state_idx
        self.current_episode_start_idx = state_idx

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
    ):
        if self.current_state_idx is None or self.current_episode_start_idx is None:
            self.start_episode(state)

        next_state_idx = self._register_next_state(next_state)
        self._store_transition(
            slot=self.position,
            state_idx=self.current_state_idx,
            episode_start_idx=self.current_episode_start_idx,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        if terminated or truncated:
            self.current_state_idx = None
            self.current_episode_start_idx = None
        else:
            self.current_state_idx = next_state_idx

        self._prune_frames()

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        if batch_size > self.size:
            raise ValueError("batch_size cannot exceed replay buffer size.")

        indices = np.array(random.sample(range(self.size), batch_size), dtype=np.int64)
        return self._build_batch(indices)

    def sample_tensors(
        self,
        batch_size: int,
        device: str | torch.device,
    ) -> Tuple[torch.Tensor, ...]:
        if batch_size > self.size:
            raise ValueError("batch_size cannot exceed replay buffer size.")

        indices = np.array(random.sample(range(self.size), batch_size), dtype=np.int64)
        return self._build_batch_tensors(indices, device=device)

    def __len__(self) -> int:
        return self.size

    def is_ready(self, min_size: int) -> bool:
        return self.size >= min_size

    def _store_transition(
        self,
        slot: int,
        state_idx: int,
        episode_start_idx: int,
        action: int,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        self.actions[slot] = action
        self.rewards[slot] = reward
        self.terminated[slot] = terminated
        self.truncated[slot] = truncated
        self.state_indices[slot] = state_idx
        self.episode_start_indices[slot] = episode_start_idx

    def _register_next_state(self, next_state: np.ndarray) -> int:
        return self._store_frame(next_state)

    def _store_frame(self, state: np.ndarray) -> int:
        latest_frame = self._extract_latest_frame(state)
        state_idx = self.next_state_index
        self.frames[state_idx] = latest_frame
        self.frame_queue.append(state_idx)
        self.next_state_index += 1
        return state_idx

    def _build_batch(self, slots: np.ndarray) -> Tuple[np.ndarray, ...]:
        return self._build_batch_arrays(slots)

    def _build_batch_arrays(
        self,
        slots: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch_size = int(len(slots))
        frame_shape = self._frame_shape()
        states = np.empty((batch_size, self.frame_stack, *frame_shape), dtype=np.uint8)
        next_states = np.empty_like(states)

        for batch_idx, slot in enumerate(slots):
            state_idx = int(self.state_indices[slot])
            episode_start_idx = int(self.episode_start_indices[slot])
            self._fill_state_array(states[batch_idx], state_idx, episode_start_idx)
            self._fill_state_array(next_states[batch_idx], state_idx + 1, episode_start_idx)

        return (
            states,
            self.actions[slots].copy(),
            self.rewards[slots].copy(),
            next_states,
            self.terminated[slots].astype(np.float32),
            self.truncated[slots].astype(np.float32),
        )

    def _build_batch_tensors(
        self,
        slots: np.ndarray,
        device: str | torch.device,
    ) -> Tuple[torch.Tensor, ...]:
        states, actions, rewards, next_states, terminated, truncated = (
            self._build_batch_arrays(slots)
        )
        return (
            self._array_to_tensor(states, device=device),
            self._array_to_tensor(actions, device=device, dtype=torch.long),
            self._array_to_tensor(rewards, device=device, dtype=torch.float32),
            self._array_to_tensor(next_states, device=device),
            self._array_to_tensor(terminated, device=device, dtype=torch.float32),
            self._array_to_tensor(truncated, device=device, dtype=torch.float32),
        )

    def _build_state(self, state_idx: int, episode_start_idx: int) -> np.ndarray:
        state = np.empty((self.frame_stack, *self._frame_shape()), dtype=np.uint8)
        self._fill_state_array(state, state_idx, episode_start_idx)
        return state

    def _fill_state_array(
        self,
        output: np.ndarray,
        state_idx: int,
        episode_start_idx: int,
    ) -> None:
        start_idx = max(episode_start_idx, state_idx - self.frame_stack + 1)
        if start_idx > state_idx:
            raise ValueError("Failed to reconstruct stacked observation from replay buffer.")

        first_frame = self.frames[start_idx]
        frame_count = state_idx - start_idx + 1
        pad_count = self.frame_stack - frame_count
        if pad_count > 0:
            output[:pad_count] = first_frame

        output_idx = pad_count
        for idx in range(start_idx, state_idx + 1):
            output[output_idx] = self.frames[idx]
            output_idx += 1

    def _frame_shape(self) -> tuple[int, ...]:
        if not self.frames:
            raise ValueError("Replay buffer has no frames to infer observation shape from.")
        return next(iter(self.frames.values())).shape

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

    def _prune_frames(self) -> None:
        if self.size == 0:
            return

        oldest_slot = self.position if self.size == self.capacity else 0
        oldest_state_idx = int(self.state_indices[oldest_slot])
        oldest_episode_start_idx = int(self.episode_start_indices[oldest_slot])
        min_required_idx = max(
            oldest_episode_start_idx, oldest_state_idx - self.frame_stack + 1
        )

        while self.frame_queue and self.frame_queue[0] < min_required_idx:
            idx = self.frame_queue.popleft()
            del self.frames[idx]

    @staticmethod
    def _extract_latest_frame(state: np.ndarray) -> np.ndarray:
        if state.ndim != 3:
            raise ValueError(
                f"Expected stacked observation with 3 dimensions, got shape {state.shape}."
            )
        return np.asarray(state[-1], dtype=np.uint8).copy()


class PrioritizedReplayBuffer(ReplayBuffer):
    """按 TD 误差优先级采样的经验回放。"""

    def __init__(
        self,
        capacity: int = 100000,
        frame_stack: int = 4,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        seed: Optional[int] = None,
    ):
        super().__init__(capacity=capacity, frame_stack=frame_stack, seed=seed)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0
        self.max_priority = 1.0
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.tree = SumTree(capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
    ):
        slot = self.position
        super().push(state, action, reward, next_state, terminated, truncated)
        self.priorities[slot] = self.max_priority
        self.tree.update(slot, self.max_priority)

    def sample(self, batch_size: int) -> Tuple:
        if batch_size > self.size:
            raise ValueError("batch_size cannot exceed replay buffer size.")

        self.frame += 1
        beta = min(
            1.0,
            self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames,
        )

        total_priority = self.tree.total()
        if total_priority <= 0:
            slots = np.array(random.sample(range(self.size), batch_size), dtype=np.int64)
            probs = np.full(batch_size, 1.0 / self.size, dtype=np.float32)
        else:
            segment = total_priority / batch_size
            sampled_slots = []
            probs = []
            for idx in range(batch_size):
                left = segment * idx
                right = segment * (idx + 1)
                mass = random.uniform(left, right)
                slot = self.tree.get(mass)
                while slot >= self.size:
                    mass = random.uniform(0.0, total_priority)
                    slot = self.tree.get(mass)
                sampled_slots.append(slot)
                probs.append(self.priorities[slot] / total_priority)

            slots = np.array(sampled_slots, dtype=np.int64)
            probs = np.asarray(probs, dtype=np.float32)

        weights = (self.size * probs) ** (-beta)
        weights = weights / weights.max()

        batch = self._build_batch(slots)
        return (*batch, slots, weights.astype(np.float32))

    def sample_tensors(
        self,
        batch_size: int,
        device: str | torch.device,
    ) -> Tuple[torch.Tensor, ...]:
        if batch_size > self.size:
            raise ValueError("batch_size cannot exceed replay buffer size.")

        self.frame += 1
        beta = min(
            1.0,
            self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames,
        )

        total_priority = self.tree.total()
        if total_priority <= 0:
            slots = np.array(random.sample(range(self.size), batch_size), dtype=np.int64)
            probs = np.full(batch_size, 1.0 / self.size, dtype=np.float32)
        else:
            segment = total_priority / batch_size
            sampled_slots = []
            probs = []
            for idx in range(batch_size):
                left = segment * idx
                right = segment * (idx + 1)
                mass = random.uniform(left, right)
                slot = self.tree.get(mass)
                while slot >= self.size:
                    mass = random.uniform(0.0, total_priority)
                    slot = self.tree.get(mass)
                sampled_slots.append(slot)
                probs.append(self.priorities[slot] / total_priority)

            slots = np.array(sampled_slots, dtype=np.int64)
            probs = np.asarray(probs, dtype=np.float32)

        weights = (self.size * probs) ** (-beta)
        weights = weights / weights.max()

        batch = self._build_batch_tensors(slots, device=device)
        return (
            *batch,
            self._array_to_tensor(slots, device=device, dtype=torch.long),
            self._array_to_tensor(weights.astype(np.float32), device=device, dtype=torch.float32),
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        for index, priority in zip(indices, priorities):
            adjusted = float(max(priority, 1e-6) ** self.alpha)
            self.priorities[index] = adjusted
            self.tree.update(int(index), adjusted)
            self.max_priority = max(self.max_priority, adjusted)
