import unittest

import numpy as np

from dqn.replay_buffer import ReplayBuffer


class ReplayBufferSemanticsTest(unittest.TestCase):
    def _make_state(self, value: int) -> np.ndarray:
        return np.full((4, 2, 2), value, dtype=np.uint8)

    def test_stores_terminated_and_truncated_flags(self) -> None:
        buffer = ReplayBuffer(capacity=8, frame_stack=4, seed=7)
        state0 = self._make_state(0)
        state1 = self._make_state(1)
        state2 = self._make_state(2)

        buffer.start_episode(state0)
        buffer.push(state0, 0, 1.0, state1, False, True)
        buffer.start_episode(state2)
        buffer.push(state2, 1, 2.0, self._make_state(3), True, False)

        self.assertEqual(buffer.size, 2)
        self.assertEqual(buffer.truncated[0], True)
        self.assertEqual(buffer.terminated[0], False)
        self.assertEqual(buffer.terminated[1], True)
        self.assertEqual(buffer.truncated[1], False)

    def test_truncated_cuts_episode_boundary_for_frame_stack(self) -> None:
        buffer = ReplayBuffer(capacity=8, frame_stack=4, seed=11)
        state0 = self._make_state(0)
        state1 = self._make_state(1)
        state2 = self._make_state(2)
        state3 = self._make_state(3)

        buffer.start_episode(state0)
        buffer.push(state0, 0, 0.0, state1, False, True)
        buffer.start_episode(state2)
        buffer.push(state2, 1, 0.0, state3, False, False)

        sampled_state = buffer._build_state(
            int(buffer.state_indices[1]),
            int(buffer.episode_start_indices[1]),
        )
        self.assertTrue(np.all(sampled_state[0] == 2))
        self.assertTrue(np.all(sampled_state[1] == 2))
        self.assertTrue(np.all(sampled_state[2] == 2))
        self.assertTrue(np.all(sampled_state[3] == 2))


if __name__ == "__main__":
    unittest.main()
