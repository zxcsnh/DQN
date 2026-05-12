from __future__ import annotations

import torch

from ..buffers.prioritized_replay_buffer import PrioritizedReplayBuffer
from .dqn_agent import DQNAgent


class PERDQNAgent(DQNAgent):
    def __init__(self, state_dim: int, action_dim: int, env_config, per_config, env_name: str) -> None:
        super().__init__(state_dim, action_dim, env_config, env_name, algo_name="perdqn")
        self.beta_start = per_config.beta_start
        self.beta = self.beta_start
        self.beta_anneal_steps = max(
            1,
            int(per_config.beta_anneal_steps)
            if per_config.beta_anneal_steps is not None
            else int(env_config.epsilon_decay_steps),
        )
        self.replay_buffer = PrioritizedReplayBuffer(
            env_config.replay_buffer_size,
            alpha=per_config.alpha,
            priority_epsilon=per_config.priority_epsilon,
            max_weight=per_config.max_weight,
        )

    def update(self):
        if self.env_steps < self.warmup_steps:
            return None
        if len(self.replay_buffer) < max(self.min_replay_size, self.batch_size):
            return None

        states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(
            self.batch_size, beta=self.beta
        )
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        weights_t = torch.as_tensor(weights, dtype=torch.float32, device=self.device)

        current_q = self.q_network(states_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            if self.use_double_dqn:
                next_actions = self.q_network(next_states_t).argmax(dim=1, keepdim=True)
                next_q = self.target_q_network(next_states_t).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_q_network(next_states_t).max(dim=1).values
            target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        # PER samples by priority and uses IS weights to reduce bias.
        td_errors = target_q - current_q
        loss = (self.loss_fn(current_q, target_q) * weights_t).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.gradient_clip_norm)
        self.optimizer.step()

        # 裁剪TD误差用于优先级更新，防止极端优先级（范围[-10, 10]）
        clipped_td_errors = torch.clamp(td_errors, -10.0, 10.0)
        self.replay_buffer.update_priorities(indices, clipped_td_errors.detach().cpu().numpy())

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        self._update_epsilon()
        progress = min(1.0, self.train_steps / self.beta_anneal_steps)
        self.beta = self.beta_start + (1.0 - self.beta_start) * progress
        return float(loss.item())
