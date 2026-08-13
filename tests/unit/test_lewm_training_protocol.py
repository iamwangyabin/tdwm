import unittest
from pathlib import Path

from tdwm.training.lewm import load_training_protocol, validate_training_protocol


ROOT = Path(__file__).resolve().parents[2]


class LeWMTrainingProtocolTest(unittest.TestCase):
    def test_locked_training_protocol_is_valid(self):
        protocol = load_training_protocol(
            ROOT / "configs/experiment/lewm_cube_train.yaml"
        )
        self.assertEqual(protocol["seeds"], [0, 42, 3072])
        self.assertEqual(protocol["split"]["unit"], "sequence_clip")
        self.assertEqual(protocol["training"]["epochs"], 10)
        self.assertEqual(protocol["training"]["scheduler_epochs"], 10)

    def test_scheduler_must_match_training_horizon(self):
        protocol = load_training_protocol(
            ROOT / "configs/experiment/lewm_cube_train.yaml"
        )
        protocol["training"]["scheduler_epochs"] = 100
        with self.assertRaisesRegex(ValueError, "remain locked together"):
            validate_training_protocol(protocol)


if __name__ == "__main__":
    unittest.main()
