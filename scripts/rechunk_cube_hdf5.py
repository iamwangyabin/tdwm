#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin
import numpy as np


PIXEL_COLUMN = "pixels"
REQUIRED_TRAINING_COLUMNS = {
    "pixels",
    "action",
    "observation",
    "ep_len",
    "ep_offset",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Losslessly rechunk the Cube HDF5 pixels for random clip training."
        )
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pixel-chunk", type=int, default=1)
    parser.add_argument("--copy-rows", type=int, default=100)
    parser.add_argument("--verify-samples", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(_jsonable(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_attributes(source: Any, output: Any) -> None:
    for key, value in source.attrs.items():
        output.attrs[key] = value


def _create_dataset(
    output: h5py.File,
    name: str,
    source: h5py.Dataset,
    *,
    pixel_chunk: int,
) -> h5py.Dataset:
    kwargs: dict[str, Any] = {
        "shape": source.shape,
        "dtype": source.dtype,
    }
    if source.maxshape is not None:
        kwargs["maxshape"] = source.maxshape
    if name == PIXEL_COLUMN:
        kwargs["chunks"] = (pixel_chunk, *source.shape[1:])
        kwargs.update(
            hdf5plugin.Blosc(
                cname="lz4",
                clevel=5,
                shuffle=hdf5plugin.Blosc.SHUFFLE,
            )
        )
    elif source.chunks is not None:
        kwargs["chunks"] = source.chunks

    dataset = output.create_dataset(name, **kwargs)
    _copy_attributes(source, dataset)
    return dataset


def _sample_indices(length: int, samples: int) -> np.ndarray:
    count = min(length, samples)
    if count == 0:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.linspace(0, length - 1, count, dtype=np.int64))


def _equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.dtype.kind == "O" or right.dtype.kind == "O":
        return np.array_equal(left.astype(str), right.astype(str))
    return np.array_equal(left, right, equal_nan=True)


def _verify(
    source_path: Path,
    output_path: Path,
    *,
    pixel_chunk: int,
    samples: int,
) -> dict[str, Any]:
    checked: dict[str, int] = {}
    with (
        h5py.File(source_path, "r") as source,
        h5py.File(output_path, "r") as output,
    ):
        if set(source.keys()) != set(output.keys()):
            raise RuntimeError("Output HDF5 columns differ from the source.")
        for name in source:
            source_dataset = source[name]
            output_dataset = output[name]
            if source_dataset.shape != output_dataset.shape:
                raise RuntimeError(f"Shape mismatch for {name}.")
            if source_dataset.dtype != output_dataset.dtype:
                raise RuntimeError(f"Dtype mismatch for {name}.")
            indices = _sample_indices(len(source_dataset), samples)
            if indices.size and not _equal(
                source_dataset[indices], output_dataset[indices]
            ):
                raise RuntimeError(f"Sample value mismatch for {name}.")
            checked[name] = int(indices.size)

        expected_chunk = (pixel_chunk, *source[PIXEL_COLUMN].shape[1:])
        if output[PIXEL_COLUMN].chunks != expected_chunk:
            raise RuntimeError(
                f"Expected pixel chunks {expected_chunk}, "
                f"found {output[PIXEL_COLUMN].chunks}."
            )
        if not REQUIRED_TRAINING_COLUMNS.issubset(output.keys()):
            raise RuntimeError("Output is missing a required LeWM training column.")
        episodes = len(output["ep_len"])
        transitions = int(np.asarray(output["ep_len"]).sum())

    return {
        "columns": len(checked),
        "samples_checked_per_column": checked,
        "episodes": episodes,
        "transitions": transitions,
        "pixel_chunk": pixel_chunk,
    }


def rechunk_cube_hdf5(
    source_path: Path,
    output_path: Path,
    *,
    pixel_chunk: int = 1,
    copy_rows: int = 100,
    verify_samples: int = 128,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_path = source_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if source_path == output_path:
        raise ValueError("Source and output paths must differ.")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if pixel_chunk <= 0 or copy_rows <= 0 or verify_samples <= 0:
        raise ValueError("Chunk, copy, and verification sizes must be positive.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".partial")
    progress_path = output_path.with_suffix(output_path.suffix + ".progress.json")
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    if temporary_path.exists():
        if not overwrite:
            raise FileExistsError(temporary_path)
        temporary_path.unlink()
    if output_path.exists():
        output_path.unlink()

    started = time.time()
    with h5py.File(source_path, "r") as source:
        missing = REQUIRED_TRAINING_COLUMNS - source.keys()
        if missing:
            raise ValueError(f"Source is missing columns: {sorted(missing)}")
        total_rows = sum(len(source[name]) for name in source)
        copied_rows = 0
        with h5py.File(temporary_path, "w", libver="latest") as output:
            _copy_attributes(source, output)
            for name in source:
                source_dataset = source[name]
                output_dataset = _create_dataset(
                    output,
                    name,
                    source_dataset,
                    pixel_chunk=pixel_chunk,
                )
                block_rows = (
                    copy_rows if name == PIXEL_COLUMN else max(copy_rows, 1000)
                )
                for start in range(0, len(source_dataset), block_rows):
                    stop = min(start + block_rows, len(source_dataset))
                    output_dataset[start:stop] = source_dataset[start:stop]
                    copied_rows += stop - start
                    if name == PIXEL_COLUMN and (start // block_rows) % 100 == 0:
                        elapsed = time.time() - started
                        fraction = copied_rows / total_rows
                        progress = {
                            "source": source_path,
                            "output": output_path,
                            "current_column": name,
                            "current_row": stop,
                            "current_column_rows": len(source_dataset),
                            "overall_fraction": fraction,
                            "elapsed_seconds": elapsed,
                            "eta_seconds": (
                                elapsed * (1 - fraction) / fraction
                                if fraction > 0
                                else None
                            ),
                            "temporary_size_bytes": temporary_path.stat().st_size,
                        }
                        _write_json(progress_path, progress)
                        print(
                            f"{name}: {stop}/{len(source_dataset)} "
                            f"({stop / len(source_dataset):.1%}), "
                            f"output={temporary_path.stat().st_size / 1e9:.1f} GB",
                            flush=True,
                        )
                output.flush()

    os.replace(temporary_path, output_path)
    verification = _verify(
        source_path,
        output_path,
        pixel_chunk=pixel_chunk,
        samples=verify_samples,
    )
    result = {
        "source": source_path,
        "output": output_path,
        "source_size_bytes": source_path.stat().st_size,
        "output_size_bytes": output_path.stat().st_size,
        "output_sha256": _sha256(output_path),
        "elapsed_seconds": time.time() - started,
        "compression": {
            "filter": "blosc",
            "codec": "lz4",
            "level": 5,
            "shuffle": True,
            "lossless": True,
        },
        "verification": verification,
    }
    _write_json(manifest_path, result)
    progress_path.unlink(missing_ok=True)
    return _jsonable(result)


def main() -> None:
    args = parse_args()
    result = rechunk_cube_hdf5(
        args.source,
        args.output,
        pixel_chunk=args.pixel_chunk,
        copy_rows=args.copy_rows,
        verify_samples=args.verify_samples,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
