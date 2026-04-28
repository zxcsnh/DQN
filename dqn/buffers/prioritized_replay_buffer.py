from __future__ import annotations

from collections import deque

import numpy as np


class SumTree:
    """Binary segment tree for O(log n) priority sampling and updates.

    Tree is stored as a flat array.  Leaves (priorities) occupy the second half;
    internal nodes store the sum of their two children.  Root at index 0 holds
    the total priority sum.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self._leaf_offset = capacity - 1  # first leaf index

    def total(self) -> float:
        return float(self.tree[0])

    def update(self, data_idx: int, priority: float) -> None:
        """Set the priority for *data_idx* (0-indexed position in the ring)."""
        leaf = self._leaf_offset + data_idx
        delta = priority - self.tree[leaf]
        self.tree[leaf] = priority
        # propagate delta up to root
        while leaf > 0:
            leaf = (leaf - 1) // 2
            self.tree[leaf] += delta

    def sample(self, value: float) -> tuple[int, float]:
        """Return (data_idx, priority) for the leaf that *value* falls into.

        *value* should be in [0, total_priority).
        """
        idx = 0
        while idx < self._leaf_offset:
            left = 2 * idx + 1
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        data_idx = idx - self._leaf_offset
        return data_idx, float(self.tree[idx])


class PrioritizedReplayBuffer:
    """Proportional-priority experience replay with SumTree.

    Sampling and priority updates are both O(log n), making this practical for
    large buffers (100k+).  Importance-sampling weights correct for the
    distributional bias introduced by non-uniform sampling.
    """

    def __init__(self, capacity: int, alpha: float = 0.6, priority_epsilon: float = 1e-5) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.priority_epsilon = priority_epsilon
        self.buffer = deque(maxlen=capacity)
        self.tree = SumTree(capacity)
        self._write_pos = 0
        self._size = 0

    def push(self, state, action, reward, next_state, done) -> None:
        state_array = np.asarray(state, dtype=np.float32)
        next_state_array = np.asarray(next_state, dtype=np.float32)
        self.buffer.append((state_array, int(action), float(reward), next_state_array, float(done)))

        max_priority = max(self.tree.tree[self.tree._leaf_offset:].max(), 1.0)
        self.tree.update(self._write_pos, float(max_priority))
        self._write_pos = (self._write_pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float):
        if len(self.buffer) < batch_size:
            raise ValueError("经验数量不足，无法进行优先采样。")

        total = self.tree.total()
        if total <= 0.0:
            # uniform fallback when no priorities have been set
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            samples = [self.buffer[idx] for idx in indices]
            states, actions, rewards, next_states, dones = zip(*samples)
            weights = np.ones(batch_size, dtype=np.float32)
            return (
                np.stack(states),
                np.asarray(actions, dtype=np.int64),
                np.asarray(rewards, dtype=np.float32),
                np.stack(next_states),
                np.asarray(dones, dtype=np.float32),
                indices,
                weights,
            )

        # segment the [0, total) range so each segment gets one sample
        segment = total / batch_size
        data_indices = []
        weights = np.empty(batch_size, dtype=np.float32)

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            value = np.random.uniform(lo, hi)
            data_idx, priority = self.tree.sample(value)
            data_indices.append(data_idx)

            # IS weight: w = (N * P(i)) ^ (-beta), then normalise by 1 / max(w)
            # so that weights only scale the update downwards (standard PER).
            prob = priority / total
            weights[i] = (len(self.buffer) * prob) ** (-beta)

        weights = weights / weights.max()

        samples = [self.buffer[idx] for idx in data_indices]
        states, actions, rewards, next_states, dones = zip(*samples)

        return (
            np.stack(states),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_states),
            np.asarray(dones, dtype=np.float32),
            np.asarray(data_indices, dtype=np.int64),
            weights.astype(np.float32),
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        for idx, td_error in zip(indices, td_errors):
            priority = float((abs(td_error) + self.priority_epsilon) ** self.alpha)
            self.tree.update(int(idx), priority)

    def __len__(self) -> int:
        return len(self.buffer)
