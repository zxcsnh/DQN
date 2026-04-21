import unittest
from unittest import mock

import torch

from dqn.agent import DQNAgent


class ComputeLossSemanticsTest(unittest.TestCase):
    def _make_agent(self) -> DQNAgent:
        return DQNAgent(
            num_actions=2,
            input_channels=4,
            input_shape=(84, 84),
            device="cpu",
            gamma=0.99,
            replay_sample_torch_fastpath=False,
            seed=123,
        )

    def test_terminated_masks_bootstrap(self) -> None:
        agent = self._make_agent()
        states = torch.zeros((1, 4, 84, 84), dtype=torch.uint8)
        next_states = torch.zeros((1, 4, 84, 84), dtype=torch.uint8)
        actions = torch.tensor([1], dtype=torch.long)
        rewards = torch.tensor([2.0], dtype=torch.float32)
        terminated = torch.tensor([1.0], dtype=torch.float32)
        truncated = torch.tensor([0.0], dtype=torch.float32)

        with (
            mock.patch.object(agent.policy_net, "forward", return_value=torch.tensor([[1.0, 5.0]])),
            mock.patch.object(agent.target_net, "forward", return_value=torch.tensor([[10.0, 20.0]])),
        ):
            loss_info = agent.compute_loss(
                (states, actions, rewards, next_states, terminated, truncated)
            )

        expected_target = torch.tensor([2.0])
        expected_td_error = expected_target - torch.tensor([5.0])
        self.assertTrue(torch.allclose(loss_info["td_errors"], expected_td_error))

    def test_truncated_does_not_mask_bootstrap(self) -> None:
        agent = self._make_agent()
        states = torch.zeros((1, 4, 84, 84), dtype=torch.uint8)
        next_states = torch.zeros((1, 4, 84, 84), dtype=torch.uint8)
        actions = torch.tensor([1], dtype=torch.long)
        rewards = torch.tensor([2.0], dtype=torch.float32)
        terminated = torch.tensor([0.0], dtype=torch.float32)
        truncated = torch.tensor([1.0], dtype=torch.float32)

        with (
            mock.patch.object(agent.policy_net, "forward", return_value=torch.tensor([[1.0, 5.0]])),
            mock.patch.object(agent.target_net, "forward", return_value=torch.tensor([[10.0, 20.0]])),
        ):
            loss_info = agent.compute_loss(
                (states, actions, rewards, next_states, terminated, truncated)
            )

        expected_target = torch.tensor([2.0 + 0.99 * 20.0])
        expected_td_error = expected_target - torch.tensor([5.0])
        self.assertTrue(torch.allclose(loss_info["td_errors"], expected_td_error))


if __name__ == "__main__":
    unittest.main()
