"""Adapters for running the released TD-JEPA implementation on Cube data."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


TD_JEPA_DOMAIN = "visual-cube-single-play-v0"
TD_JEPA_BUFFER_VERSION = 1


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_tdjepa_episode(
    source_episode: Mapping[str, Any],
    *,
    image_size: int,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Build the public TD-JEPA OGBench episode-buffer representation.

    This mirrors TD-JEPA's ``extract_all.py`` action alignment and field names.
    Cube Lance pixels are already channel-first, whereas the official extractor
    converts OGBench's channel-last images into the same layout.
    """

    if image_size <= 0:
        raise ValueError("image_size must be positive.")
    required = {"pixels", "action", "qpos"}
    missing = required - source_episode.keys()
    if missing:
        raise ValueError(f"Cube episode is missing columns: {sorted(missing)}")

    import torch
    import torch.nn.functional as F

    pixels = torch.as_tensor(source_episode["pixels"], device=device)
    action = _as_numpy(source_episode["action"]).astype(np.float32, copy=True)
    qpos = _as_numpy(source_episode["qpos"]).astype(np.float32, copy=False)

    if pixels.ndim != 4 or pixels.shape[1] != 3:
        raise ValueError(
            "Expected channel-first Cube pixels with shape (steps, 3, height, width)."
        )
    if action.ndim != 2 or qpos.ndim != 2:
        raise ValueError("Cube actions and qpos values must be rank-two arrays.")
    steps = int(pixels.shape[0])
    if steps < 2:
        raise ValueError("TD-JEPA episode buffers require at least two frames.")
    if action.shape[0] != steps or qpos.shape[0] != steps:
        raise ValueError("Cube pixels, actions, and qpos must have equal lengths.")

    pixels = pixels.to(torch.float32)
    if tuple(pixels.shape[-2:]) != (image_size, image_size):
        pixels = F.interpolate(
            pixels,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
    pixels = pixels.round().clamp_(0, 255).to(torch.uint8).cpu().numpy()

    # The released extractor stores the previous action at each observation.
    action[1:] = action[:-1]
    action[0] = 0.0
    return {
        "observation": np.zeros((steps, 1), dtype=np.float32),
        "pixels": pixels,
        "action": action,
        "physics": qpos,
        "reward": np.zeros((steps, 1), dtype=np.float32),
        "discount": np.ones((steps, 1), dtype=np.float32),
    }


def _save_episode(path: Path, episode: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        # Uncompressed NPZ is intentional: local NVMe plus pinned workers is the
        # fast path for the released TD-JEPA ParallelBuffer.
        np.savez(stream, **episode)
    temporary.replace(path)


def convert_cube_lance_to_tdjepa_buffer(
    source_path: str | Path,
    output_root: str | Path,
    *,
    image_size: int = 64,
    device: str = "cpu",
    episodes: int | None = None,
    resume: bool = False,
    report_interval: int = 100,
    implementation_revision: str,
) -> dict[str, Any]:
    """Convert an audited Cube Lance table to TD-JEPA's documented buffer API."""

    source_path = Path(source_path).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if not source_path.is_dir() or source_path.suffix != ".lance":
        raise ValueError("source_path must be an existing Cube .lance directory.")
    if image_size <= 0 or report_interval <= 0:
        raise ValueError("image_size and report_interval must be positive.")
    if not implementation_revision:
        raise ValueError("implementation_revision is required for the audit record.")

    stable_worldmodel_version = importlib.metadata.version("stable-worldmodel")
    if stable_worldmodel_version != "0.1.1":
        raise RuntimeError(
            "TD-JEPA Cube conversion requires stable-worldmodel==0.1.1, found "
            f"{stable_worldmodel_version}."
        )

    import stable_worldmodel as swm

    dataset = swm.data.load_dataset(
        str(source_path),
        format="lance",
        keys_to_load=["pixels", "action", "qpos"],
    )
    lengths = np.asarray(dataset.lengths, dtype=np.int64)
    total_episodes = int(lengths.size)
    requested_episodes = total_episodes if episodes is None else int(episodes)
    if requested_episodes <= 0 or requested_episodes > total_episodes:
        raise ValueError(
            f"episodes must be in [1, {total_episodes}], found {requested_episodes}."
        )

    buffer_dir = output_root / TD_JEPA_DOMAIN / "buffer"
    manifest_path = output_root / "td_jepa_cube_lance_manifest.json"
    progress_path = output_root / "td_jepa_cube_lance_progress.json"
    existing = list(buffer_dir.glob("episode_*.npz")) if buffer_dir.exists() else []
    if existing and not resume:
        raise FileExistsError(
            f"TD-JEPA buffer already contains {len(existing)} episode files; use resume=True."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    buffer_dir.mkdir(parents=True, exist_ok=True)

    source_manifest_path = Path(f"{source_path}.manifest.json")
    source_manifest_sha256 = (
        _sha256(source_manifest_path) if source_manifest_path.is_file() else None
    )
    started = time.time()
    converted = 0
    skipped = 0
    for episode_index in range(requested_episodes):
        steps = int(lengths[episode_index])
        destination = buffer_dir / f"episode_{episode_index:06}_{steps - 1}.npz"
        if destination.exists():
            if not resume:
                raise FileExistsError(destination)
            skipped += 1
            continue
        source_episode = dataset.load_episode(episode_index)
        episode = build_tdjepa_episode(
            source_episode,
            image_size=image_size,
            device=device,
        )
        _save_episode(destination, episode)
        converted += 1

        completed = converted + skipped
        if completed % report_interval == 0 or completed == requested_episodes:
            progress = {
                "schema_version": TD_JEPA_BUFFER_VERSION,
                "requested_episodes": requested_episodes,
                "completed_episodes": completed,
                "converted_episodes": converted,
                "skipped_episodes": skipped,
                "elapsed_seconds": time.time() - started,
                "updated_at_unix": time.time(),
            }
            _write_json(progress_path, progress)
            print(
                f"TD-JEPA buffer: {completed}/{requested_episodes} episodes "
                f"({progress['elapsed_seconds']:.1f}s)",
                flush=True,
            )

    output_size = sum(path.stat().st_size for path in buffer_dir.glob("*.npz"))
    result = {
        "schema_version": TD_JEPA_BUFFER_VERSION,
        "source": {
            "path": str(source_path),
            "format": "lance",
            "manifest_path": str(source_manifest_path),
            "manifest_sha256": source_manifest_sha256,
            "episodes": total_episodes,
            "transitions": int(lengths.sum()),
        },
        "destination": {
            "root": str(output_root),
            "domain": TD_JEPA_DOMAIN,
            "buffer": str(buffer_dir),
            "format": "npz",
            "compression": "stored",
            "episodes": requested_episodes,
            "transitions": int(lengths[:requested_episodes].sum()),
            "size_bytes": output_size,
        },
        "preprocessing": {
            "input_layout": "CHW uint8",
            "output_layout": "CHW uint8",
            "image_size": image_size,
            "resize": "torch.interpolate(bilinear, align_corners=False, round)",
            "action_alignment": "previous-action; action[0]=0",
            "physics": "qpos",
        },
        "implementation": {
            "repository": "facebookresearch/td_jepa",
            "revision": implementation_revision,
            "stable_worldmodel_version": stable_worldmodel_version,
        },
        "conversion": {
            "resume": resume,
            "converted_episodes": converted,
            "skipped_episodes": skipped,
            "elapsed_seconds": time.time() - started,
        },
    }
    _write_json(manifest_path, result)
    return result
