import json
import tempfile
import unittest
from pathlib import Path

from tdwm.training.cube_data import validate_cube_training_dataset
from tdwm.training.lewm import load_training_protocol


ROOT = Path(__file__).resolve().parents[2]


class CubeTrainingDataTest(unittest.TestCase):
    def setUp(self):
        protocol = load_training_protocol(
            ROOT / "configs/experiment/lewm_cube_train.yaml"
        )
        self.dataset_config = protocol["dataset"]

    def _manifest(self, dataset_path: Path) -> dict:
        lance = self.dataset_config["lance"]
        return {
            "schema_version": 1,
            "source": {
                "name": lance["source"]["file"],
                "size_bytes": lance["source"]["size_bytes"],
                "sha256": lance["source"]["sha256"],
            },
            "destination": {
                "name": dataset_path.name,
                "format": "lance",
                "size_bytes": 123,
                "image_codec": lance["image_codec"],
                "jpeg_quality": lance["jpeg_quality"],
            },
            "conversion": {
                "api": "swm.data.convert",
                "stable_worldmodel_version": lance["stable_worldmodel_version"],
                "mode": "error",
            },
            "verification": {
                "episodes": self.dataset_config["expected_episodes"],
                "transitions": self.dataset_config["expected_transitions"],
                "numeric_columns": {
                    "action": {"exact": True},
                    "observation": {"exact": True},
                },
                "pixels": {
                    "samples": lance["minimum_pixel_verification_samples"],
                    "mae": 0.25,
                },
            },
        }

    def _write_lance_fixture(self, root: Path) -> Path:
        dataset_path = root / "cube_single_expert_jpeg100.lance"
        dataset_path.mkdir()
        manifest_path = Path(
            f"{dataset_path}{self.dataset_config['lance']['manifest_suffix']}"
        )
        manifest_path.write_text(json.dumps(self._manifest(dataset_path)))
        return dataset_path

    def test_accepts_locked_hdf5_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset_path = Path(temporary) / "cube.h5"
            with dataset_path.open("wb") as stream:
                stream.truncate(self.dataset_config["accepted_size_bytes"][1])

            result = validate_cube_training_dataset(
                dataset_path, self.dataset_config
            )

        self.assertEqual(result["format"], "hdf5")
        self.assertIsNone(result["conversion_manifest"])

    def test_accepts_audited_lance_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset_path = self._write_lance_fixture(Path(temporary))

            result = validate_cube_training_dataset(
                dataset_path, self.dataset_config
            )

        self.assertEqual(result["format"], "lance")
        self.assertEqual(result["conversion_manifest"]["schema_version"], 1)

    def test_rejects_lance_without_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset_path = Path(temporary) / "cube.lance"
            dataset_path.mkdir()

            with self.assertRaisesRegex(FileNotFoundError, "manifest"):
                validate_cube_training_dataset(dataset_path, self.dataset_config)

    def test_rejects_different_jpeg_quality(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset_path = self._write_lance_fixture(Path(temporary))
            manifest_path = Path(
                f"{dataset_path}{self.dataset_config['lance']['manifest_suffix']}"
            )
            manifest = json.loads(manifest_path.read_text())
            manifest["destination"]["jpeg_quality"] = 95
            manifest_path.write_text(json.dumps(manifest))

            with self.assertRaisesRegex(ValueError, "jpeg_quality"):
                validate_cube_training_dataset(dataset_path, self.dataset_config)


if __name__ == "__main__":
    unittest.main()
