"""DQN 与 PER 使用的经验回放缓冲区实现。"""

from __future__ import annotations

import random
from typing import Any, Dict, Optional, Tuple

import numpy as np


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

    def state_dict(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "tree": self.tree.copy(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.capacity = state["capacity"]
        self.tree = state["tree"]


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
        self.next_state_index = 0
        self.current_state_idx: Optional[int] = None
        self.current_episode_start_idx: Optional[int] = None

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def start_episode(self, initial_state: np.ndarray) -> None:
        """在每个 episode reset 后注册初始观测。"""
        latest_frame = self._extract_latest_frame(initial_state)
        state_idx = self.next_state_index
        self.frames[state_idx] = latest_frame
        self.next_state_index += 1
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

    def __len__(self) -> int:
        return self.size

    def is_ready(self, min_size: int) -> bool:
        return self.size >= min_size

    def state_dict(self) -> Dict[str, Any]:
        """保存完整缓冲区状态，便于中断后继续训练。"""
        return {
            "capacity": self.capacity,
            "frame_stack": self.frame_stack,
            "actions": self.actions.copy(),
            "rewards": self.rewards.copy(),
            "terminated": self.terminated.copy(),
            "truncated": self.truncated.copy(),
            "state_indices": self.state_indices.copy(),
            "episode_start_indices": self.episode_start_indices.copy(),
            "position": self.position,
            "size": self.size,
            "frames": {idx: frame.copy() for idx, frame in self.frames.items()},
            "next_state_index": self.next_state_index,
            "current_state_idx": self.current_state_idx,
            "current_episode_start_idx": self.current_episode_start_idx,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """从 checkpoint 中恢复缓冲区状态。"""
        if "frames" not in state:
            raise ValueError(
                "Legacy replay buffer checkpoints are not compatible with the "
                "frame-deduplicated buffer. Resume from a newer checkpoint instead."
            )

        self.capacity = state["capacity"]
        self.frame_stack = state["frame_stack"]
        self.actions = state["actions"]
        self.rewards = state["rewards"]
        self.terminated = state["terminated"]
        self.truncated = state["truncated"]
        self.state_indices = state["state_indices"]
        self.episode_start_indices = state["episode_start_indices"]
        self.position = state["position"]
        self.size = state["size"]
        self.frames = {
            int(idx): np.asarray(frame, dtype=np.uint8)
            for idx, frame in state["frames"].items()
        }
        self.next_state_index = state["next_state_index"]
        self.current_state_idx = state["current_state_idx"]
        self.current_episode_start_idx = state["current_episode_start_idx"]

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
        latest_frame = self._extract_latest_frame(next_state)
        state_idx = self.next_state_index
        self.frames[state_idx] = latest_frame
        self.next_state_index += 1
        return state_idx

    def _build_batch(self, slots: np.ndarray) -> Tuple[np.ndarray, ...]:
        states = []
        next_states = []

        for slot in slots:
            state_idx = int(self.state_indices[slot])
            episode_start_idx = int(self.episode_start_indices[slot])
            states.append(self._build_state(state_idx, episode_start_idx))
            next_states.append(self._build_state(state_idx + 1, episode_start_idx))

        return (
            np.stack(states, axis=0),
            self.actions[slots].copy(),
            self.rewards[slots].copy(),
            np.stack(next_states, axis=0),
            self.terminated[slots].astype(np.float32),
            self.truncated[slots].astype(np.float32),
        )

    def _build_state(self, state_idx: int, episode_start_idx: int) -> np.ndarray:
        start_idx = max(episode_start_idx, state_idx - self.frame_stack + 1)
        frames = [self.frames[idx] for idx in range(start_idx, state_idx + 1)]
        if not frames:
            raise ValueError("Failed to reconstruct stacked observation from replay buffer.")

        pad_count = self.frame_stack - len(frames)
        if pad_count > 0:
            frames = [frames[0]] * pad_count + frames

        return np.stack(frames, axis=0)

    def _prune_frames(self) -> None:
        if self.size == 0:
            return

        oldest_slot = self.position if self.size == self.capacity else 0
        oldest_state_idx = int(self.state_indices[oldest_slot])
        oldest_episode_start_idx = int(self.episode_start_indices[oldest_slot])
        min_required_idx = max(
            oldest_episode_start_idx, oldest_state_idx - self.frame_stack + 1
        )

        stale_indices = [idx for idx in self.frames if idx < min_required_idx]
        for idx in stale_indices:
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

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        for index, priority in zip(indices, priorities):
            adjusted = float(max(priority, 1e-6) ** self.alpha)
            self.priorities[index] = adjusted
            self.tree.update(int(index), adjusted)
            self.max_priority = max(self.max_priority, adjusted)

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state.update(
            {
                "alpha": self.alpha,
                "beta_start": self.beta_start,
                "beta_frames": self.beta_frames,
                "frame": self.frame,
                "max_priority": self.max_priority,
                "priorities": self.priorities.copy(),
                "tree": self.tree.state_dict(),
            }
        )
        return state

    def load_state_dict(self, state: Dict[str, Any]):
        if "tree" not in state:
            raise ValueError(
                "Legacy prioritized replay checkpoints are not compatible with the "
                "SumTree-based buffer. Resume from a newer checkpoint instead."
            )

        super().load_state_dict(state)
        self.alpha = state["alpha"]
        self.beta_start = state["beta_start"]
        self.beta_frames = state["beta_frames"]
        self.frame = state["frame"]
        self.max_priority = state["max_priority"]
        self.priorities = state["priorities"]
        self.tree = SumTree(self.capacity)
        self.tree.load_state_dict(state["tree"])
