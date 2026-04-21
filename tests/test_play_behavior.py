import io
import unittest
from contextlib import redirect_stderr
from unittest import mock

from config import DQNConfig
import play


class PlayBehaviorTest(unittest.TestCase):
    def test_resolve_default_model_path_warns_when_missing(self) -> None:
        config = DQNConfig(save_dir="missing_models", save_freq=5)
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            model_path = play.resolve_default_model_path(config)
        self.assertIsNone(model_path)
        self.assertIn("untrained policy network", stderr.getvalue())

    def test_play_raises_for_missing_explicit_model_path(self) -> None:
        with self.assertRaises(FileNotFoundError):
            play.play(model_path="definitely_missing_model.pth", render=False, num_episodes=1)

    def test_record_video_raises_for_missing_explicit_model_path(self) -> None:
        with self.assertRaises(FileNotFoundError):
            play.record_video(model_path="definitely_missing_model.pth", num_episodes=1)


if __name__ == "__main__":
    unittest.main()
