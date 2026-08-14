from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def lance_manifest_path(dataset_path: str | Path, suffix: str) -> Path:
    return Path(f"{Path(dataset_path)}{suffix}")


def _expect_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(
            f"Lance manifest {label} is {actual!r}, expected {expected!r}."
        )


def validate_lance_manifest(
    dataset_path: Path,
    manifest: dict[str, Any],
    dataset_config: dict[str, Any],
) -> None:
    lance = dataset_config["lance"]
    source = manifest.get("source", {})
    destination = manifest.get("destination", {})
    conversion = manifest.get("conversion", {})
    verification = manifest.get("verification", {})

    _expect_equal("schema_version", manifest.get("schema_version"), 1)
    _expect_equal(
        "source.size_bytes",
        source.get("size_bytes"),
        lance["source"]["size_bytes"],
    )
    _expect_equal("source.sha256", source.get("sha256"), lance["source"]["sha256"])
    _expect_equal("destination.name", destination.get("name"), dataset_path.name)
    _expect_equal("destination.format", destination.get("format"), "lance")
    _expect_equal(
        "destination.image_codec",
        destination.get("image_codec"),
        lance["image_codec"],
    )
    _expect_equal(
        "destination.jpeg_quality",
        destination.get("jpeg_quality"),
        lance["jpeg_quality"],
    )
    _expect_equal("conversion.api", conversion.get("api"), "swm.data.convert")
    _expect_equal(
        "conversion.stable_worldmodel_version",
        conversion.get("stable_worldmodel_version"),
        lance["stable_worldmodel_version"],
    )
    _expect_equal("conversion.mode", conversion.get("mode"), "error")
    _expect_equal(
        "verification.episodes",
        verification.get("episodes"),
        dataset_config["expected_episodes"],
    )
    _expect_equal(
        "verification.transitions",
        verification.get("transitions"),
        dataset_config["expected_transitions"],
    )

    numeric_columns = verification.get("numeric_columns", {})
    for column in dataset_config["keys_to_cache"]:
        if numeric_columns.get(column, {}).get("exact") is not True:
            raise ValueError(
                f"Lance manifest must record an exact {column!r} verification."
            )

    pixel_check = verification.get("pixels", {})
    if pixel_check.get("samples", 0) < lance["minimum_pixel_verification_samples"]:
        raise ValueError("Lance manifest has too few verified pixel samples.")
    if not isinstance(pixel_check.get("mae"), (int, float)):
        raise ValueError("Lance manifest must record pixel MAE.")


def validate_cube_training_dataset(
    dataset_path: str | Path,
    dataset_config: dict[str, Any],
) -> dict[str, Any]:
    path = Path(dataset_path).expanduser().resolve()
    if path.is_file():
        actual_size = path.stat().st_size
        expected_sizes = dataset_config.get(
            "accepted_size_bytes",
            [dataset_config.get("expected_size_bytes")],
        )
        if actual_size not in expected_sizes:
            raise ValueError(
                f"Cube dataset size {actual_size} is not one of the locked "
                f"HDF5 layouts {expected_sizes}."
            )
        return {
            "path": str(path),
            "format": "hdf5",
            "size_bytes": actual_size,
            "conversion_manifest": None,
        }

    if not path.exists():
        raise FileNotFoundError(f"Cube dataset not found: {path}")

    lance = dataset_config["lance"]
    if not path.is_dir() or path.suffix.lower() != lance["suffix"]:
        raise ValueError(
            "Cube training data must be a locked HDF5 file or a .lance directory."
        )

    manifest_path = lance_manifest_path(path, lance["manifest_suffix"])
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Audited Lance conversion manifest not found: {manifest_path}"
        )
    with manifest_path.open() as stream:
        manifest = json.load(stream)
    validate_lance_manifest(path, manifest, dataset_config)

    return {
        "path": str(path),
        "format": "lance",
        "size_bytes": manifest["destination"].get("size_bytes"),
        "conversion_manifest_path": str(manifest_path),
        "conversion_manifest": manifest,
    }
