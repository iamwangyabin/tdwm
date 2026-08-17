import unittest
from pathlib import Path
from unittest.mock import patch

from tdwm.training.lewm import (
    load_training_protocol,
    _resolve_train_batch_limit,
    _resolve_device_image_preprocessing,
    _resolve_block_prefetch,
    _resolve_block_shuffle,
    _resolve_loader_runtime,
    _resolve_model_compile,
    _resolve_stride_aware_lance,
    _compile_world_model,
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
        self.assertEqual(protocol["scheduler"]["interval"], "optimizer_step")
        self.assertIn(74104077358, protocol["dataset"]["accepted_size_bytes"])
        self.assertEqual(protocol["dataset"]["lance"]["jpeg_quality"], 100)
        self.assertEqual(protocol["dataset"]["lance"]["suffix"], ".lance")
        self.assertEqual(protocol["loader"]["workers"], 6)
        self.assertEqual(protocol["loader"]["prefetch_factor"], 2)
        self.assertEqual(protocol["loader"]["validation_workers"], 2)
        self.assertTrue(protocol["loader"]["validation_locality"])
        self.assertTrue(protocol["loader"]["stride_aware_lance"])
        self.assertTrue(protocol["loader"]["device_image_preprocessing"])
        self.assertFalse(protocol["loader"]["block_shuffle"])
        self.assertEqual(protocol["loader"]["block_size"], 2048)
        self.assertFalse(protocol["loader"]["block_prefetch"])
        self.assertEqual(protocol["loader"]["block_prefetch_size"], 512)
        self.assertFalse(protocol["training"]["model_compile"])
        self.assertEqual(
            protocol["training"]["model_compile_mode"], "reduce-overhead"
        )
        self.assertEqual(protocol["logging"]["type"], "csv")
        self.assertEqual(protocol["logging"]["flush_every_n_steps"], 50)

    def test_scheduler_must_match_training_horizon(self):
        protocol = load_training_protocol(
            ROOT / "configs/experiment/lewm_cube_train.yaml"
        )
        protocol["training"]["scheduler_epochs"] = 100
        with self.assertRaisesRegex(ValueError, "remain locked together"):
            validate_training_protocol(protocol)

    def test_scheduler_interval_must_be_supported(self):
        protocol = load_training_protocol(
            ROOT / "configs/experiment/lewm_cube_train.yaml"
        )
        protocol["scheduler"]["interval"] = "epoch"

        with self.assertRaisesRegex(ValueError, "scheduler.interval"):
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

    def test_block_shuffle_can_be_enabled_for_a_controlled_comparison(self):
        loader_config = {
            "batch_size": 128,
            "block_shuffle": False,
            "block_size": 2048,
            "shuffle_batches_within_block": True,
        }

        configured = _resolve_block_shuffle(loader_config)
        enabled = _resolve_block_shuffle(
            loader_config, override=True, block_size=4096
        )

        self.assertFalse(configured["effective"])
        self.assertTrue(enabled["effective"])
        self.assertEqual(enabled["block_size"], 4096)

    def test_block_prefetch_can_be_enabled_for_a_controlled_comparison(self):
        loader_config = {
            "batch_size": 128,
            "block_prefetch": False,
            "block_prefetch_size": 512,
        }

        configured = _resolve_block_prefetch(loader_config)
        enabled = _resolve_block_prefetch(
            loader_config, override=True, block_size=1024
        )

        self.assertFalse(configured["effective"])
        self.assertTrue(enabled["effective"])
        self.assertEqual(enabled["block_size"], 1024)

    def test_validation_locality_flag_must_be_boolean(self):
        protocol = load_training_protocol(
            ROOT / "configs/experiment/lewm_cube_train.yaml"
        )
        protocol["loader"]["validation_locality"] = "yes"

        with self.assertRaisesRegex(ValueError, "validation_locality"):
            validate_training_protocol(protocol)

    def test_model_compile_can_be_enabled_for_a_controlled_comparison(self):
        training_config = {
            "model_compile": True,
            "model_compile_mode": "reduce-overhead",
        }

        enabled = _resolve_model_compile(training_config)
        disabled = _resolve_model_compile(training_config, override=False)

        self.assertTrue(enabled["effective"])
        self.assertFalse(disabled["effective"])
        self.assertEqual(enabled["mode"], "reduce-overhead")

    def test_compile_wraps_only_lewm_public_compute_methods(self):
        class Model:
            def encode(self, batch):
                return batch

            def predict(self, embedding, action_embedding):
                return embedding + action_embedding

        model = Model()
        compiled = []

        def fake_compile(method, *, mode):
            compiled.append((method.__name__, mode))
            return method

        with patch("torch.compile", side_effect=fake_compile):
            _compile_world_model(model, mode="reduce-overhead")

        self.assertEqual(
            compiled,
            [("encode", "reduce-overhead"), ("predict", "reduce-overhead")],
        )


if __name__ == "__main__":
    unittest.main()
