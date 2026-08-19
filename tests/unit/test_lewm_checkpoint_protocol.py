import unittest
from pathlib import Path

import numpy as np

from tdwm.evaluation.lewm_checkpoint import (
    load_protocol,
    sample_start_goal_pairs,
)


ROOT = Path(__file__).resolve().parents[2]


class LeWMCheckpointProtocolTest(unittest.TestCase):
    def test_locked_protocol_is_valid(self):
        protocol = load_protocol(
            ROOT / "configs/experiment/lewm_cube_checkpoint_o25.yaml"
        )
        self.assertEqual(protocol["checkpoint"]["parameters"], 18034628)
        self.assertEqual(
            protocol["planning"]["executed_environment_steps_before_replanning"],
            25,
        )
        self.assertEqual(protocol["planning"]["iterations"], 30)
        self.assertEqual(protocol["dataset"]["lance"]["image_codec"], "jpeg")
        self.assertEqual(protocol["dataset"]["lance"]["jpeg_quality"], 100)

    def test_trained_lance_protocol_is_valid(self):
        protocol = load_protocol(
            ROOT / "configs/experiment/lewm_cube_seed0_o25.yaml"
        )
        self.assertEqual(protocol["dataset"]["format"], "lance")
        self.assertEqual(protocol["checkpoint"]["epoch"], 10)
        self.assertEqual(protocol["planning"]["iterations"], 30)

    def test_seed3072_lance_protocol_is_valid(self):
        protocol = load_protocol(
            ROOT / "configs/experiment/lewm_cube_seed3072_o25.yaml"
        )
        self.assertEqual(protocol["checkpoint"]["seed"], 3072)
        self.assertEqual(protocol["checkpoint"]["epoch"], 10)
        self.assertEqual(protocol["planning"]["iterations"], 30)

    def test_gt_lewm_protocol_is_valid(self):
        protocol = load_protocol(
            ROOT / "configs/experiment/gt_lewm_cube_checkpoint_o25.yaml"
        )
        self.assertEqual(protocol["method"], "gt_lewm")
        self.assertEqual(protocol["tail_value"]["horizon"], 8)
        self.assertEqual(
            protocol["planning"]["objective"],
            "discounted_latent_path_cost_plus_terminal_tail_value",
        )

    def test_sampler_is_deterministic_and_respects_goal_offset(self):
        lengths = np.array([10, 12, 8])
        first = sample_start_goal_pairs(
            lengths, goal_offset=4, episodes=5, seed=42
        )
        second = sample_start_goal_pairs(
            lengths, goal_offset=4, episodes=5, seed=42
        )
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
        episodes, starts, _ = first
        self.assertTrue(np.all(starts + 4 < lengths[episodes]))


if __name__ == "__main__":
    unittest.main()
