#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from tdwm.training import load_training_protocol
from tdwm.training.cube_data import lance_manifest_path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/experiment/lewm_cube_train.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the locked Cube HDF5 dataset to an audited JPEG-100 Lance table."
        )
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--verify-samples", type=int, default=128)
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    processed = 0
    report_at = 8 * 1024**3
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            processed += len(chunk)
            if processed >= report_at:
                print(f"Hashed {processed / 1024**3:.1f} GiB...", flush=True)
                report_at += 8 * 1024**3
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def _sample_indices(length: int, samples: int) -> np.ndarray:
    count = min(length, samples)
    return np.unique(np.linspace(0, length - 1, count, dtype=np.int64))


def _verify_conversion(
    source_path: Path,
    output_path: Path,
    *,
    expected_episodes: int,
    expected_transitions: int,
    pixel_samples: int,
) -> dict[str, Any]:
    import stable_worldmodel as swm

    load_kwargs = {
        "transform": None,
        "num_steps": 1,
        "frameskip": 1,
        "keys_to_load": ["pixels", "action", "observation"],
    }
    source = swm.data.load_dataset(
        str(source_path), format="hdf5", **load_kwargs
    )
    output = swm.data.load_dataset(
        str(output_path), format="lance", **load_kwargs
    )

    source_lengths = np.asarray(source.lengths)
    output_lengths = np.asarray(output.lengths)
    if not np.array_equal(source_lengths, output_lengths):
        raise RuntimeError("Lance episode boundaries differ from the source HDF5.")
    episodes = len(output_lengths)
    transitions = int(output_lengths.sum())
    if episodes != expected_episodes or transitions != expected_transitions:
        raise RuntimeError(
            f"Converted dataset has {episodes} episodes and {transitions} transitions."
        )

    numeric_checks: dict[str, Any] = {}
    for column in ("action", "observation"):
        source_values = np.asarray(source.get_col_data(column))
        output_values = np.asarray(output.get_col_data(column))
        exact = source_values.dtype == output_values.dtype and np.array_equal(
            source_values, output_values, equal_nan=True
        )
        numeric_checks[column] = {
            "exact": bool(exact),
            "dtype": str(output_values.dtype),
            "shape": list(output_values.shape),
            "sha256": _array_sha256(output_values),
        }
        if not exact:
            raise RuntimeError(f"Lance conversion changed the {column!r} column.")

    absolute_error = 0.0
    squared_error = 0.0
    maximum_error = 0.0
    compared_values = 0
    indices = _sample_indices(len(source), pixel_samples)
    for index in indices:
        source_pixels = _as_numpy(source[int(index)]["pixels"]).astype(np.float64)
        output_pixels = _as_numpy(output[int(index)]["pixels"]).astype(np.float64)
        if source_pixels.shape != output_pixels.shape:
            raise RuntimeError("Lance conversion changed the pixel tensor shape.")
        difference = np.abs(source_pixels - output_pixels)
        absolute_error += float(difference.sum())
        squared_error += float(np.square(difference).sum())
        maximum_error = max(maximum_error, float(difference.max()))
        compared_values += difference.size

    mae = absolute_error / compared_values
    mse = squared_error / compared_values
    return {
        "episodes": episodes,
        "transitions": transitions,
        "episode_lengths_sha256": _array_sha256(output_lengths),
        "numeric_columns": numeric_checks,
        "pixels": {
            "samples": int(indices.size),
            "mae": mae,
            "mse": mse,
            "psnr_db": 20 * math.log10(255.0 / math.sqrt(mse)) if mse else None,
            "maximum_absolute_error": maximum_error,
            "lossless": mse == 0,
        },
    }


def convert_cube_to_lance(
    source_path: Path,
    output_path: Path,
    *,
    protocol_path: Path = DEFAULT_CONFIG,
    verify_samples: int = 128,
) -> dict[str, Any]:
    source_path = source_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    protocol = load_training_protocol(protocol_path)
    dataset_config = protocol["dataset"]
    lance_config = dataset_config["lance"]
    manifest_path = lance_manifest_path(output_path, lance_config["manifest_suffix"])

    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if output_path.suffix.lower() != lance_config["suffix"]:
        raise ValueError(f"Lance output must end in {lance_config['suffix']!r}.")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(output_path if output_path.exists() else manifest_path)
    if verify_samples < lance_config["minimum_pixel_verification_samples"]:
        raise ValueError(
            "verify_samples is below the protocol's minimum pixel audit size."
        )

    expected_source = lance_config["source"]
    actual_size = source_path.stat().st_size
    if actual_size != expected_source["size_bytes"]:
        raise ValueError(
            f"Lance source size is {actual_size}, expected "
            f"{expected_source['size_bytes']}."
        )
    source_sha256 = _file_sha256(source_path)
    if source_sha256 != expected_source["sha256"]:
        raise ValueError("Lance source SHA-256 does not match the locked HDF5 layout.")

    package_version = importlib.metadata.version("stable-worldmodel")
    if package_version != lance_config["stable_worldmodel_version"]:
        raise RuntimeError(
            f"Expected stable-worldmodel {lance_config['stable_worldmodel_version']}, "
            f"found {package_version}."
        )

    import stable_worldmodel as swm

    started = time.time()
    swm.data.convert(
        str(source_path),
        str(output_path),
        source_format="hdf5",
        dest_format="lance",
        jpeg_quality=lance_config["jpeg_quality"],
        mode="error",
    )
    verification = _verify_conversion(
        source_path,
        output_path,
        expected_episodes=dataset_config["expected_episodes"],
        expected_transitions=dataset_config["expected_transitions"],
        pixel_samples=verify_samples,
    )
    output_size = sum(
        path.stat().st_size for path in output_path.rglob("*") if path.is_file()
    )
    result = {
        "schema_version": 1,
        "source": {
            "name": source_path.name,
            "size_bytes": actual_size,
            "sha256": source_sha256,
        },
        "destination": {
            "name": output_path.name,
            "format": "lance",
            "size_bytes": output_size,
            "image_codec": lance_config["image_codec"],
            "jpeg_quality": lance_config["jpeg_quality"],
        },
        "conversion": {
            "api": "swm.data.convert",
            "stable_worldmodel_version": package_version,
            "mode": "error",
            "elapsed_seconds": time.time() - started,
        },
        "verification": verification,
    }
    _write_json(manifest_path, result)
    return result


def main() -> None:
    args = parse_args()
    result = convert_cube_to_lance(
        args.source,
        args.output,
        protocol_path=args.config,
        verify_samples=args.verify_samples,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
