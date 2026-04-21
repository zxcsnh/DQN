import unittest
from unittest import mock

from config import DQNConfig
from train import train


class ProtocolConfigurationTest(unittest.TestCase):
    def test_train_and_eval_env_use_different_protocol_flags(self) -> None:
        config = DQNConfig(
            max_steps=0,
            eval_episodes=0,
            clip_reward=True,
            terminal_on_life_loss=True,
            log_dir="runs_test_protocol",
            save_dir="models_test_protocol",
        )

        env_mock = mock.MagicMock()
        env_mock.action_space.n = 6
        env_mock.reset.return_value = (None, {})
        eval_env_mock = mock.MagicMock()
        eval_env_mock.action_space.n = 6
        eval_env_mock.reset.return_value = (None, {})
        logger_mock = mock.MagicMock()
        logger_mock.episode_rewards = []
        logger_mock.episode_losses = []
        logger_mock.eval_steps = []
        logger_mock.eval_rewards = []
        logger_mock.avg_rewards = [0.0]

        with (
            mock.patch("train.make_env", side_effect=[env_mock, eval_env_mock]) as make_env,
            mock.patch("train.DQNAgent") as agent_cls,
            mock.patch("train.TrainingLogger", return_value=logger_mock),
            mock.patch("train.os.makedirs"),
            mock.patch("train.open", mock.mock_open()),
            mock.patch("train.json.dump"),
        ):
            agent = mock.MagicMock()
            agent.env_steps_done = 0
            agent.episodes_done = 0
            agent.epsilon = 1.0
            agent_cls.return_value = agent
            train(config)

        self.assertEqual(make_env.call_count, 2)
        train_call = make_env.call_args_list[0].kwargs
        eval_call = make_env.call_args_list[1].kwargs
        self.assertEqual(train_call["clip_reward"], True)
        self.assertEqual(train_call["terminal_on_life_loss"], True)
        self.assertEqual(eval_call["clip_reward"], False)
        self.assertEqual(eval_call["terminal_on_life_loss"], False)


if __name__ == "__main__":
    unittest.main()
