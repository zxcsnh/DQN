from __future__ import annotations

from collections import deque

import numpy as np


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, priority_epsilon: float = 1e-5) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.priority_epsilon = priority_epsilon
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        state_array = np.asarray(state, dtype=np.float32)
        next_state_array = np.asarray(next_state, dtype=np.float32)
        max_priority = max(self.priorities, default=1.0)
        self.buffer.append((state_array, int(action), float(reward), next_state_array, float(done)))
        self.priorities.append(float(max_priority))

    def sample(self, batch_size: int, beta: float):
        if len(self.buffer) < batch_size:
            raise ValueError("经验数量不足，无法进行优先采样。")

        priorities = np.asarray(self.priorities, dtype=np.float32)
        scaled_priorities = priorities ** self.alpha
        total_priority = float(scaled_priorities.sum())
        if total_priority <= 0.0:
            probabilities = np.full(len(self.buffer), 1.0 / len(self.buffer), dtype=np.float32)
        else:
            probabilities = scaled_priorities / total_priority

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)

        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights = weights / weights.max()

        return (
            np.stack(states),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_states),
            np.asarray(dones, dtype=np.float32),
            indices,
            weights.astype(np.float32),
        )

    def update_priorities(self, indices, td_errors) -> None:
        updated = list(self.priorities)
        for idx, td_error in zip(indices, td_errors):
            updated[int(idx)] = float(abs(td_error) + self.priority_epsilon)
        self.priorities = deque(updated, maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.buffer)
