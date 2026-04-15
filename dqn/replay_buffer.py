"""Replay buffer implementations for DQN and PER."""

import random
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _upgrade_transition(transition):
    """Upgrade legacy (s, a, r, s', done) tuples to the new format."""
    if transition is None or len(transition) == 6:
        return transition
    if len(transition) == 5:
        state, action, reward, next_state, done = transition
        return (state, action, reward, next_state, done, False)
    raise ValueError(f"Unexpected transition format with length {len(transition)}")


class ReplayBuffer:
    """Uniform replay buffer."""

    def __init__(self, capacity: int = 100000, seed: Optional[int] = None):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
    ):
        transition = (state, action, reward, next_state, terminated, truncated)

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        batch = random.sample(self.buffer, batch_size)

        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        terminated = np.array([t[4] for t in batch], dtype=np.float32)
        truncated = np.array([t[5] for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, terminated, truncated

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, min_size: int) -> bool:
        return len(self.buffer) >= min_size

    def state_dict(self) -> Dict[str, Any]:
        """Return full replay buffer state for checkpoint resume."""
        return {
            "capacity": self.capacity,
            "buffer": self.buffer,
            "position": self.position,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """Restore replay buffer state from checkpoint."""
        self.capacity = state["capacity"]
        self.buffer = [_upgrade_transition(transition) for transition in state["buffer"]]
        self.position = state["position"]


class PrioritizedReplayBuffer:
    """Prioritized replay buffer (proportional variant)."""

    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        seed: Optional[int] = None,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0

        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.buffer = [None] * capacity
        self.position = 0
        self.size = 0

        if seed is not None:
            np.random.seed(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
    ):
        # priorities are stored as p^alpha already
        max_priority = self.priorities[: self.size].max() if self.size > 0 else 1.0

        self.buffer[self.position] = (
            state,
            action,
            reward,
            next_state,
            terminated,
            truncated,
        )
        self.priorities[self.position] = max_priority

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple:
        self.frame += 1

        beta = min(
            1.0,
            self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames,
        )

        total_priority = self.priorities[: self.size].sum()
        if total_priority <= 0:
            probs = np.ones(self.size, dtype=np.float32) / self.size
        else:
            probs = self.priorities[: self.size] / total_priority

        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)

        samples = [self.buffer[i] for i in indices]
        states = np.array([s[0] for s in samples])
        actions = np.array([s[1] for s in samples])
        rewards = np.array([s[2] for s in samples])
        next_states = np.array([s[3] for s in samples])
        terminated = np.array([s[4] for s in samples], dtype=np.float32)
        truncated = np.array([s[5] for s in samples], dtype=np.float32)

        weights = (self.size * probs[indices]) ** (-beta)
        weights = weights / weights.max()

        return (
            states,
            actions,
            rewards,
            next_states,
            terminated,
            truncated,
            indices,
            weights,
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        self.priorities[indices] = priorities ** self.alpha

    def __len__(self) -> int:
        return self.size

    def is_ready(self, min_size: int) -> bool:
        return self.size >= min_size

    def state_dict(self) -> Dict[str, Any]:
        """Return full PER state for checkpoint resume."""
        return {
            "capacity": self.capacity,
            "alpha": self.alpha,
            "beta_start": self.beta_start,
            "beta_frames": self.beta_frames,
            "frame": self.frame,
            "priorities": self.priorities,
            "buffer": self.buffer,
            "position": self.position,
            "size": self.size,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """Restore PER state from checkpoint."""
        self.capacity = state["capacity"]
        self.alpha = state["alpha"]
        self.beta_start = state["beta_start"]
        self.beta_frames = state["beta_frames"]
        self.frame = state["frame"]
        self.priorities = state["priorities"]
        self.buffer = [_upgrade_transition(transition) for transition in state["buffer"]]
        self.position = state["position"]
        self.size = state["size"]
