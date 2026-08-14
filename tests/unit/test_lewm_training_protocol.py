import unittest
from pathlib import Path

from tdwm.training.lewm import (
    load_training_protocol,
    _resolve_loader_runtime,
    validate_training_protocol,
)


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
        self.assertIn(74104077358, protocol["dataset"]["accepted_size_bytes"])
        self.assertEqual(protocol["dataset"]["lance"]["jpeg_quality"], 100)
        self.assertEqual(protocol["dataset"]["lance"]["suffix"], ".lance")
        self.assertEqual(protocol["logging"]["type"], "csv")
        self.assertEqual(protocol["logging"]["flush_every_n_steps"], 50)

    def test_scheduler_must_match_training_horizon(self):
        protocol = load_training_protocol(
            ROOT / "configs/experiment/lewm_cube_train.yaml"
        )
        protocol["training"]["scheduler_epochs"] = 100
        with self.assertRaisesRegex(ValueError, "remain locked together"):
            validate_training_protocol(protocol)

    def test_smoke_disables_multiprocess_loader_teardown(self):
        loader_config = {"workers": 6, "prefetch_factor": 3}

        smoke = _resolve_loader_runtime(loader_config, smoke=True)
        formal = _resolve_loader_runtime(loader_config, smoke=False)

        self.assertEqual(smoke["configured_workers"], 6)
        self.assertEqual(smoke["workers"], 0)
        self.assertFalse(smoke["persistent_workers"])
        self.assertIsNone(smoke["prefetch_factor"])
        self.assertEqual(formal["workers"], 6)
        self.assertTrue(formal["persistent_workers"])
        self.assertEqual(formal["prefetch_factor"], 3)

    def test_csv_metrics_logging_is_required(self):
        protocol = load_training_protocol(
            ROOT / "configs/experiment/lewm_cube_train.yaml"
        )
        protocol["logging"]["flush_every_n_steps"] = 0

        with self.assertRaisesRegex(ValueError, "flush after a positive"):
            validate_training_protocol(protocol)


if __name__ == "__main__":
    unittest.main()
