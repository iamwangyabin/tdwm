import unittest
from pathlib import Path

from tdwm.training.lewm import (
    load_training_protocol,
    _resolve_train_batch_limit,
    _resolve_device_image_preprocessing,
    _resolve_loader_runtime,
    _resolve_stride_aware_lance,
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
        self.assertEqual(protocol["loader"]["workers"], 6)
        self.assertEqual(protocol["loader"]["prefetch_factor"], 1)
        self.assertEqual(protocol["loader"]["validation_workers"], 0)
        self.assertTrue(protocol["loader"]["stride_aware_lance"])
        self.assertTrue(protocol["loader"]["device_image_preprocessing"])
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
        loader_config = {
            "workers": 6,
            "prefetch_factor": 3,
            "validation_workers": 0,
        }

        smoke = _resolve_loader_runtime(loader_config, smoke=True)
        formal = _resolve_loader_runtime(loader_config, smoke=False)

        self.assertEqual(smoke["train"]["configured_workers"], 6)
        self.assertEqual(smoke["train"]["workers"], 0)
        self.assertFalse(smoke["train"]["persistent_workers"])
        self.assertIsNone(smoke["train"]["prefetch_factor"])
        self.assertEqual(formal["train"]["workers"], 6)
        self.assertTrue(formal["train"]["persistent_workers"])
        self.assertEqual(formal["train"]["prefetch_factor"], 3)
        self.assertEqual(formal["validation"]["workers"], 0)

    def test_loader_overrides_are_recorded_without_changing_data(self):
        runtime = _resolve_loader_runtime(
            {"workers": 6, "prefetch_factor": 3, "validation_workers": 0},
            smoke=False,
            workers=12,
            prefetch_factor=2,
            validation_workers=1,
        )

        self.assertEqual(runtime["train"]["configured_workers"], 6)
        self.assertEqual(runtime["train"]["workers"], 12)
        self.assertEqual(runtime["train"]["prefetch_factor"], 2)
        self.assertEqual(runtime["validation"]["workers"], 1)

    def test_throughput_run_ends_at_a_loader_epoch_boundary(self):
        self.assertEqual(
            _resolve_train_batch_limit(
                smoke=False, max_steps=100, train_loader_length=12796
            ),
            100,
        )
        self.assertEqual(
            _resolve_train_batch_limit(
                smoke=False, max_steps=20000, train_loader_length=12796
            ),
            12796,
        )
        self.assertEqual(
            _resolve_train_batch_limit(
                smoke=False, max_steps=None, train_loader_length=12796
            ),
            1.0,
        )

    def test_csv_metrics_logging_is_required(self):
        protocol = load_training_protocol(
            ROOT / "configs/experiment/lewm_cube_train.yaml"
        )
        protocol["logging"]["flush_every_n_steps"] = 0

        with self.assertRaisesRegex(ValueError, "flush after a positive"):
            validate_training_protocol(protocol)

    def test_stride_aware_loader_only_activates_for_lance(self):
        loader_config = {"stride_aware_lance": True}

        lance = _resolve_stride_aware_lance(
            loader_config, dataset_format="lance"
        )
        hdf5 = _resolve_stride_aware_lance(
            loader_config, dataset_format="hdf5"
        )
        disabled = _resolve_stride_aware_lance(
            loader_config, dataset_format="lance", override=False
        )

        self.assertTrue(lance["effective"])
        self.assertFalse(hdf5["effective"])
        self.assertFalse(disabled["effective"])

    def test_stride_aware_loader_flag_must_be_boolean(self):
        protocol = load_training_protocol(
            ROOT / "configs/experiment/lewm_cube_train.yaml"
        )
        protocol["loader"]["stride_aware_lance"] = "yes"

        with self.assertRaisesRegex(ValueError, "must be true or false"):
            validate_training_protocol(protocol)

    def test_device_image_preprocessing_can_be_disabled_for_ab_comparison(self):
        loader_config = {"device_image_preprocessing": True}

        configured = _resolve_device_image_preprocessing(loader_config)
        disabled = _resolve_device_image_preprocessing(
            loader_config, override=False
        )

        self.assertTrue(configured["effective"])
        self.assertFalse(disabled["effective"])


if __name__ == "__main__":
    unittest.main()
