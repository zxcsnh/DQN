"""DQN 与 PER 使用的经验回放缓冲区实现。"""

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
        self.frame_capacity = capacity + frame_stack + 1
        self.frames: np.ndarray | None = None
        self.frame_ids = np.full(self.frame_capacity, -1, dtype=np.int64)
        self.next_state_index = 0
        self.stream_state: dict[int, tuple[int, int] | None] = {0: None}
        self.rng = random.Random(seed)

    def start_episode(self, initial_state: np.ndarray, stream_id: int = 0) -> None:
        """在每个 episode reset 后注册初始观测。"""
        state_idx = self._store_frame(initial_state)
        self.stream_state[stream_id] = (state_idx, state_idx)

    def _get_stream_state(self, stream_id: int) -> tuple[int, int] | None:
        return self.stream_state.get(stream_id)

    def _set_stream_state(self, stream_id: int, state: tuple[int, int] | None) -> None:
        self.stream_state[stream_id] = state

    def _clear_stream_state(self, stream_id: int) -> None:
        self.stream_state[stream_id] = None

    def _touch_stream(self, stream_id: int) -> None:
        self.stream_state.setdefault(stream_id, None)

    def _get_or_start_stream(self, state: np.ndarray, stream_id: int) -> tuple[int, int]:
        self._touch_stream(stream_id)
        stream_state = self._get_stream_state(stream_id)
        if stream_state is None:
            self.start_episode(state, stream_id=stream_id)
            stream_state = self._get_stream_state(stream_id)
            if stream_state is None:
                raise RuntimeError("Failed to initialize replay stream state.")
        return stream_state

    def _advance_stream(
        self,
        stream_id: int,
        next_state_idx: int,
        terminated: bool,
        truncated: bool,
        episode_start_idx: int,
    ) -> None:
        if terminated or truncated:
            self._clear_stream_state(stream_id)
            return
        self._set_stream_state(stream_id, (next_state_idx, episode_start_idx))

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
        stream_id: int = 0,
    ):
        current_state_idx, episode_start_idx = self._get_or_start_stream(
            state,
            stream_id=stream_id,
        )

        next_state_idx = self._register_next_state(next_state)
        self._store_transition(
            slot=self.position,
            state_idx=current_state_idx,
            episode_start_idx=episode_start_idx,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self._advance_stream(
            stream_id,
            next_state_idx=next_state_idx,
            terminated=terminated,
            truncated=truncated,
            episode_start_idx=episode_start_idx,
        )

        self._prune_frames()

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
        if self.frames is None:
            self.frames = np.empty(
                (self.frame_capacity, *latest_frame.shape),
                dtype=np.uint8,
            )

        frame_slot = state_idx % self.frame_capacity
        self.frames[frame_slot] = latest_frame
        self.frame_ids[frame_slot] = state_idx
        self.next_state_index += 1
        return state_idx

    def _build_batch(self, slots: np.ndarray) -> ReplayBatch:
        return self._build_batch_arrays(slots)

    def _build_batch_arrays(
        self,
        slots: np.ndarray,
    ) -> ReplayBatch:
        state_indices = self.state_indices[slots].astype(np.int64, copy=False)
        episode_start_indices = self.episode_start_indices[slots].astype(
            np.int64, copy=False
        )

        states = self._gather_frames(
            self._stack_frame_indices(state_indices, episode_start_indices)
        )
        next_states = self._gather_frames(
            self._stack_frame_indices(state_indices + 1, episode_start_indices)
        )

        return ReplayBatch(
            states=states,
            actions=self.actions[slots].copy(),
            rewards=self.rewards[slots].copy(),
            next_states=next_states,
            terminated=self.terminated[slots].astype(np.float32),
            truncated=self.truncated[slots].astype(np.float32),
        )

    def _build_batch_tensors(
        self,
        slots: np.ndarray,
        device: str | torch.device,
    ) -> ReplayBatch:
        batch = self._build_batch_arrays(slots)
        return ReplayBatch(
            states=self._array_to_tensor(batch.states, device=device),
            actions=self._array_to_tensor(batch.actions, device=device, dtype=torch.long),
            rewards=self._array_to_tensor(batch.rewards, device=device, dtype=torch.float32),
            next_states=self._array_to_tensor(batch.next_states, device=device),
            terminated=self._array_to_tensor(batch.terminated, device=device, dtype=torch.float32),
            truncated=self._array_to_tensor(batch.truncated, device=device, dtype=torch.float32),
            indices=(
                None
                if batch.indices is None
                else self._array_to_tensor(batch.indices, device=device, dtype=torch.long)
            ),
            weights=(
                None
                if batch.weights is None
                else self._array_to_tensor(batch.weights, device=device, dtype=torch.float32)
            ),
        )

    def _build_state(self, state_idx: int, episode_start_idx: int) -> np.ndarray:
        frame_indices = self._stack_frame_indices(
            np.array([state_idx], dtype=np.int64),
            np.array([episode_start_idx], dtype=np.int64),
        )
        return self._gather_frames(frame_indices)[0]

    def _fill_state_array(
        self,
        output: np.ndarray,
        state_idx: int,
        episode_start_idx: int,
    ) -> None:
        output[...] = self._build_state(state_idx, episode_start_idx)

    def _frame_shape(self) -> tuple[int, ...]:
        if self.frames is None:
            raise ValueError("Replay buffer has no frames to infer observation shape from.")
        return tuple(self.frames.shape[1:])

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
        # The frame ring buffer is sized to retain all history needed by the
        # oldest transition still present in replay, so no explicit pruning is
        # required here.
        return

    def _stack_frame_indices(
        self,
        state_indices: np.ndarray,
        episode_start_indices: np.ndarray,
    ) -> np.ndarray:
        offsets = np.arange(self.frame_stack, dtype=np.int64) - (self.frame_stack - 1)
        frame_indices = state_indices[:, None] + offsets[None, :]
        np.maximum(frame_indices, episode_start_indices[:, None], out=frame_indices)
        return frame_indices

    def _gather_frames(self, frame_indices: np.ndarray) -> np.ndarray:
        if self.frames is None:
            raise ValueError("Replay buffer has no frames to gather.")

        ring_indices = np.mod(frame_indices, self.frame_capacity)
        available_frame_ids = self.frame_ids[ring_indices]
        if not np.all(available_frame_ids == frame_indices):
            raise RuntimeError(
                "Replay buffer frame history was overwritten before batch reconstruction."
            )

        gathered = self.frames[ring_indices]
        return np.ascontiguousarray(gathered)

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
        beta_updates: int = 100000,
        seed: Optional[int] = None,
    ):
        super().__init__(capacity=capacity, frame_stack=frame_stack, seed=seed)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_updates = max(1, beta_updates)
        self.sample_updates_done = 0
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
        stream_id: int = 0,
    ):
        slot = self.position
        super().push(
            state,
            action,
            reward,
            next_state,
            terminated,
            truncated,
            stream_id=stream_id,
        )
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

        self.sample_updates_done += 1
        beta = min(
            1.0,
            self.beta_start
            + (1.0 - self.beta_start)
            * self.sample_updates_done
            / self.beta_updates,
        )

        total_priority = self.tree.total()
        if total_priority <= 0:
            slots = self._sample_uniform_slots(batch_size)
            probs = np.full(batch_size, 1.0 / self.size, dtype=np.float32)
        else:
            segment = total_priority / batch_size
            slots = np.empty(batch_size, dtype=np.int64)
            probs = np.empty(batch_size, dtype=np.float32)
            for idx in range(batch_size):
                left = segment * idx
                right = segment * (idx + 1)
                mass = self.rng.uniform(left, right)
                slot = self.tree.get(mass)
                while slot >= self.size:
                    mass = self.rng.uniform(0.0, total_priority)
                    slot = self.tree.get(mass)
                slots[idx] = slot
                probs[idx] = self.priorities[slot] / total_priority

        weights = (self.size * probs) ** (-beta)
        weights = weights / weights.max()
        return slots, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        adjusted = np.maximum(priorities, 1e-6).astype(np.float32) ** self.alpha
        self.priorities[indices] = adjusted
        self.max_priority = max(self.max_priority, float(np.max(adjusted)))
        for index, priority in zip(indices, adjusted, strict=False):
            self.tree.update(int(index), float(priority))
