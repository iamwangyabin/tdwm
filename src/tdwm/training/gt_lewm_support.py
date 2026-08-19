"""Small, standalone data and model assembly helpers for GT-LeWM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(_jsonable(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def fit_column_stats(dataset: Any, columns: list[str], path: Path) -> dict[str, Any]:
    from sklearn.preprocessing import StandardScaler

    if path.is_file():
        with path.open() as stream:
            return json.load(stream)

    statistics: dict[str, Any] = {}
    for column in columns:
        print(f"Fitting {column} normalization on the complete dataset...", flush=True)
        scaler = StandardScaler().fit(dataset.get_col_data(column))
        statistics[column] = {
            "mean": scaler.mean_,
            "scale": scaler.scale_,
            "variance": scaler.var_,
            "samples": int(np.min(np.asarray(scaler.n_samples_seen_))),
        }
    write_json(path, statistics)
    return statistics


class LeWMTransform:
    """Apply the locked image and numeric preprocessing to GT-LeWM samples."""

    def __init__(
        self,
        *,
        image: dict[str, Any],
        columns: dict[str, Any],
        preprocess_images: bool = True,
    ) -> None:
        self.image_transform = None
        if preprocess_images:
            from torchvision.transforms import v2 as transforms

            self.image_transform = transforms.Compose(
                [
                    transforms.ToImage(),
                    transforms.ToDtype(torch.float32, scale=True),
                    transforms.Normalize(mean=image["mean"], std=image["std"]),
                    transforms.Resize(size=image["size"]),
                ]
            )
        self.column_stats = {
            name: (
                torch.as_tensor(stats["mean"], dtype=torch.float32),
                torch.as_tensor(stats["scale"], dtype=torch.float32),
            )
            for name, stats in columns.items()
        }

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        if self.image_transform is not None:
            sample["pixels"] = self.image_transform(sample["pixels"])
        for name, (mean, scale) in self.column_stats.items():
            sample[name] = (sample[name] - mean) / scale
        return sample


def preprocess_image_batch(
    pixels: Any,
    *,
    mean: Any,
    std: Any,
    size: int,
):
    import torch.nn.functional as functional

    if pixels.dtype != torch.uint8:
        raise TypeError("Device-side image preprocessing expects uint8 dataset pixels.")
    if pixels.ndim != 5 or pixels.shape[2] != 3:
        raise ValueError("Expected pixels with shape (batch, time, 3, height, width).")
    normalized = pixels.to(dtype=torch.float32).div(255.0)
    normalized = (normalized - mean) / std
    batch, time, channels, height, width = normalized.shape
    if height == size and width == size:
        return normalized
    resized = functional.interpolate(
        normalized.reshape(batch * time, channels, height, width),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return resized.reshape(batch, time, channels, size, size)


def build_model_config(protocol: dict[str, Any], action_dim: int) -> dict[str, Any]:
    model = protocol["model"]
    embed_dim = model["embed_dim"]
    return {
        "_target_": "stable_worldmodel.wm.lewm.LeWM",
        "encoder": {
            "_target_": "stable_pretraining.backbone.utils.vit_hf",
            "size": model["encoder_size"],
            "patch_size": model["patch_size"],
            "image_size": protocol["image_preprocessing"]["size"],
            "pretrained": False,
            "use_mask_token": False,
        },
        "predictor": {
            "_target_": "stable_worldmodel.wm.lewm.module.Predictor",
            "num_frames": protocol["sequence"]["history_frames"],
            "input_dim": embed_dim,
            "hidden_dim": embed_dim,
            "output_dim": embed_dim,
            "depth": model["predictor_depth"],
            "heads": model["predictor_heads"],
            "mlp_dim": model["predictor_mlp_dim"],
            "dim_head": model["predictor_dim_head"],
            "dropout": model["predictor_dropout"],
            "emb_dropout": model["predictor_embedding_dropout"],
        },
        "action_encoder": {
            "_target_": "stable_worldmodel.wm.lewm.module.Embedder",
            "input_dim": protocol["sequence"]["frame_skip"] * action_dim,
            "emb_dim": embed_dim,
        },
        "projector": {
            "_target_": "stable_worldmodel.wm.lewm.module.MLP",
            "input_dim": embed_dim,
            "output_dim": embed_dim,
            "hidden_dim": model["projector_hidden_dim"],
            "norm_fn": {
                "_target_": "torch.nn.BatchNorm1d",
                "_partial_": True,
            },
        },
        "pred_proj": {
            "_target_": "stable_worldmodel.wm.lewm.module.MLP",
            "input_dim": embed_dim,
            "output_dim": embed_dim,
            "hidden_dim": model["projector_hidden_dim"],
            "norm_fn": {
                "_target_": "torch.nn.BatchNorm1d",
                "_partial_": True,
            },
        },
    }


def compile_world_model(world_model: Any, *, mode: str) -> None:
    world_model.encode = torch.compile(world_model.encode, mode=mode)
    world_model.predict = torch.compile(world_model.predict, mode=mode)


def build_metrics_logger(run_dir: Path, logging_config: dict[str, Any]):
    from lightning.pytorch.loggers import CSVLogger

    return CSVLogger(
        save_dir=str(run_dir),
        name="metrics",
        flush_logs_every_n_steps=logging_config["flush_every_n_steps"],
    )


def resolve_train_batch_limit(
    *, smoke: bool, max_steps: int | None, train_loader_length: int
) -> int | float:
    if smoke:
        return min(2, train_loader_length)
    if max_steps is not None:
        return min(max_steps, train_loader_length)
    return 1.0


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
