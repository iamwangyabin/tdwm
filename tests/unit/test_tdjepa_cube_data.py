import unittest

import numpy as np

from tdwm.adapters.td_jepa import build_tdjepa_episode

try:
    import torch  # noqa: F401
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for TD-JEPA pixel resizing")
class TDJEPACubeDataTest(unittest.TestCase):
    def setUp(self):
        self.pixels = np.arange(3 * 3 * 4 * 4, dtype=np.uint8).reshape(3, 3, 4, 4)
        self.action = np.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32
        )
        self.qpos = np.arange(3 * 5, dtype=np.float32).reshape(3, 5)

    def test_matches_official_episode_field_and_action_alignment(self):
        episode = build_tdjepa_episode(
            {"pixels": self.pixels, "action": self.action, "qpos": self.qpos},
            image_size=4,
        )

        self.assertEqual(
            set(episode),
            {"observation", "pixels", "action", "physics", "reward", "discount"},
        )
        np.testing.assert_array_equal(episode["pixels"], self.pixels)
        np.testing.assert_array_equal(
            episode["action"],
            np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(episode["physics"], self.qpos)
        self.assertEqual(episode["observation"].shape, (3, 1))
        self.assertEqual(episode["reward"].shape, (3, 1))
        np.testing.assert_array_equal(
            episode["discount"], np.ones((3, 1), dtype=np.float32)
        )

    def test_resizes_channel_first_pixels(self):
        episode = build_tdjepa_episode(
            {"pixels": self.pixels, "action": self.action, "qpos": self.qpos},
            image_size=2,
        )

        self.assertEqual(episode["pixels"].shape, (3, 3, 2, 2))
        self.assertEqual(episode["pixels"].dtype, np.uint8)

    def test_rejects_mismatched_lengths(self):
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            build_tdjepa_episode(
                {
                    "pixels": self.pixels,
                    "action": self.action[:2],
                    "qpos": self.qpos,
                },
                image_size=4,
            )


if __name__ == "__main__":
    unittest.main()
