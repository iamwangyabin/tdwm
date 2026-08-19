from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import yaml

from tdwm.adapters import prepare_cloud_runtime
from tdwm.methods.goal_tail import (
    GoalTailValue,
    discounted_goal_tail_target,
    goal_tail_loss,
    soft_update,
)
from tdwm.training.block_sampler import BlockShuffleBatchSampler
from tdwm.training.cube_data import validate_cube_training_dataset
from tdwm.training.lance_batch import (
    BlockPrefetchBatchDataset,
    EpisodeStreamingBatchDataset,
    StrideAwareLanceDataset,
)


def load_training_protocol(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as stream:
        protocol = yaml.safe_load(stream)
    validate_training_protocol(protocol)
    return protocol


def validate_training_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema_version") != 1:
        raise ValueError("The training protocol must use schema_version 1.")
    method = protocol.get("method")
    if method not in {"lewm", "gt_lewm"} or protocol.get("environment") != "cube":
        raise ValueError("This trainer only accepts LeWM variants on OGBench-Cube.")
    if protocol.get("runtime", {}).get("stable_worldmodel_version") != "0.1.1":
        raise ValueError("LeWM reproduction is locked to stable-worldmodel 0.1.1.")

    sequence = protocol.get("sequence", {})
    history_frames = sequence.get("history_frames", 0)
    prediction_frames = sequence.get("prediction_frames", 0)
    if method == "lewm":
        if sequence.get("num_steps") != history_frames + prediction_frames:
            raise ValueError("num_steps must equal history_frames + prediction_frames.")
    else:
        tail = protocol.get("tail_value", {})
        horizon = tail.get("horizon", 0)
        if history_frames <= 0 or prediction_frames != 1:
            raise ValueError(
                "GT-LeWM requires a positive history and one-step local prediction."
            )
        if horizon <= 0 or sequence.get("num_steps") != history_frames + horizon:
            raise ValueError(
                "GT-LeWM num_steps must equal history_frames + tail_value.horizon."
            )
        if not 0.0 <= tail.get("gamma", -1.0) < 1.0:
            raise ValueError("GT-LeWM tail_value.gamma must lie in [0, 1).")
        if tail.get("loss_weight", 0) < 0:
            raise ValueError("GT-LeWM tail_value.loss_weight cannot be negative.")
        if not 0.0 < tail.get("target_tau", 0.0) <= 1.0:
            raise ValueError("GT-LeWM tail_value.target_tau must lie in (0, 1].")
        if tail.get("hidden_dim", 0) <= 0:
            raise ValueError("GT-LeWM tail_value.hidden_dim must be positive.")
        if tail.get("goal_source") != "n_step_episode_endpoint":
            raise ValueError(
                "GT-LeWM currently trains against the N-step episode endpoint."
            )
    if sequence.get("frame_skip", 0) <= 0:
        raise ValueError("frame_skip must be positive.")

    split = protocol.get("split", {})
    if split.get("unit") != "sequence_clip":
        raise ValueError("The released LeWM trainer splits sequence clips.")
    if not 0 < split.get("train_fraction", 0) < 1:
        raise ValueError("train_fraction must lie strictly between zero and one.")
    if not math.isclose(
        split["train_fraction"] + split.get("validation_fraction", 0), 1.0
    ):
        raise ValueError("Training and validation fractions must sum to one.")

    training = protocol.get("training", {})
    if training.get("epochs") != training.get("scheduler_epochs"):
        raise ValueError("Scheduler and trainer epochs must remain locked together.")
    if training.get("epochs", 0) <= 0:
        raise ValueError("Training epochs must be positive.")
    if not isinstance(training.get("model_compile"), bool):
        raise ValueError("training.model_compile must be true or false.")
    if training.get("model_compile_mode") not in {
        "default",
        "reduce-overhead",
        "max-autotune",
    }:
        raise ValueError("training.model_compile_mode is not supported.")
    if not protocol.get("seeds"):
        raise ValueError("At least one training seed is required.")

    scheduler = protocol.get("scheduler", {})
    if scheduler.get("interval") != "optimizer_step":
        raise ValueError(
            "scheduler.interval must be optimizer_step because warmup and "
            "cosine horizons are measured in optimizer steps."
        )

    logging = protocol.get("logging", {})
    if logging.get("type") != "csv":
        raise ValueError("LeWM training must persist metrics with a CSV logger.")
    if logging.get("flush_every_n_steps", 0) <= 0:
        raise ValueError("CSV metrics must flush after a positive number of steps.")

    loader = protocol.get("loader", {})
    if loader.get("workers", -1) < 0:
        raise ValueError("Training loader workers cannot be negative.")
    if loader.get("prefetch_factor", 0) <= 0:
        raise ValueError("Training loader prefetch_factor must be positive.")
    if loader.get("validation_workers", loader.get("workers", -1)) < 0:
        raise ValueError("Validation loader workers cannot be negative.")
    if not isinstance(loader.get("stride_aware_lance"), bool):
        raise ValueError("loader.stride_aware_lance must be true or false.")
    if not isinstance(loader.get("device_image_preprocessing"), bool):
        raise ValueError("loader.device_image_preprocessing must be true or false.")
    if not isinstance(loader.get("block_shuffle"), bool):
        raise ValueError("loader.block_shuffle must be true or false.")
    if not isinstance(loader.get("block_prefetch"), bool):
        raise ValueError("loader.block_prefetch must be true or false.")
    if not isinstance(loader.get("episode_streaming"), bool):
        raise ValueError("loader.episode_streaming must be true or false.")
    if not isinstance(loader.get("validation_locality"), bool):
        raise ValueError("loader.validation_locality must be true or false.")
    if loader.get("block_size", 0) <= 0:
        raise ValueError("loader.block_size must be positive.")
    if loader["block_size"] % loader["batch_size"]:
        raise ValueError("loader.block_size must be divisible by loader.batch_size.")
    if loader.get("block_prefetch_size", 0) <= 0:
        raise ValueError("loader.block_prefetch_size must be positive.")
    if loader["block_prefetch_size"] % loader["batch_size"]:
        raise ValueError(
            "loader.block_prefetch_size must be divisible by loader.batch_size."
        )
    if not isinstance(loader.get("shuffle_batches_within_block"), bool):
        raise ValueError(
            "loader.shuffle_batches_within_block must be true or false."
        )
    if loader.get("episode_pool_size", 0) < loader["batch_size"]:
        raise ValueError("loader.episode_pool_size must be at least batch_size.")
    if loader.get("episode_read_size", 0) <= 0:
        raise ValueError("loader.episode_read_size must be positive.")
    if loader.get("episode_cache_bytes", 0) <= 0:
        raise ValueError("loader.episode_cache_bytes must be positive.")
    if loader.get("episode_prefetch_blocks", 0) <= 0:
        raise ValueError("loader.episode_prefetch_blocks must be positive.")
    if not 1 <= loader.get("minimum_unique_episodes_per_batch", 0) <= loader[
        "batch_size"
    ]:
        raise ValueError(
            "loader.minimum_unique_episodes_per_batch must lie in [1, batch_size]."
        )
    if loader["episode_streaming"] and (
        loader["block_shuffle"] or loader["block_prefetch"]
    ):
        raise ValueError(
            "episode_streaming cannot be combined with block shuffle or prefetch."
        )

    dataset = protocol.get("dataset", {})
    lance = dataset.get("lance", {})
    optimized = dataset.get("optimized_layout", {})
    if lance.get("suffix") != ".lance":
        raise ValueError("The fast Cube layout must use a .lance directory.")
    if lance.get("manifest_suffix") != ".manifest.json":
        raise ValueError("The Lance conversion manifest suffix must remain locked.")
    if lance.get("image_codec") != "jpeg" or lance.get("jpeg_quality") != 100:
        raise ValueError("The fast Cube layout must use JPEG quality 100.")
    if lance.get("minimum_pixel_verification_samples", 0) <= 0:
        raise ValueError("Lance pixel verification must use at least one sample.")
    numeric_verification = lance.get("numeric_verification", {})
    if set(numeric_verification) != set(dataset.get("keys_to_cache", [])):
        raise ValueError("Lance numeric verification must cover cached columns.")
    if set(numeric_verification.values()) - {"exact", "float32_cast_exact"}:
        raise ValueError("Lance numeric verification contains an unknown policy.")
    if lance.get("stable_worldmodel_version") != "0.1.1":
        raise ValueError("Lance conversion is locked to stable-worldmodel 0.1.1.")
    if lance.get("source", {}).get("size_bytes") not in dataset.get(
        "accepted_size_bytes", []
    ):
        raise ValueError("The Lance source must be an accepted HDF5 layout.")
    if lance.get("source", {}).get("sha256") != optimized.get("sha256"):
        raise ValueError("The Lance source must match the audited optimized HDF5.")


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(_jsonable(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _save_split(
    run_dir: Path, train_indices: np.ndarray, val_indices: np.ndarray
) -> dict[str, Any]:
    split_path = run_dir / "split_indices.npz"
    np.savez_compressed(
        split_path,
        train_indices=np.asarray(train_indices, dtype=np.int64),
        val_indices=np.asarray(val_indices, dtype=np.int64),
    )
    return {
        "path": str(split_path),
        "train_samples": int(train_indices.size),
        "validation_samples": int(val_indices.size),
        "train_indices_sha256": _array_sha256(train_indices),
        "validation_indices_sha256": _array_sha256(val_indices),
    }


def _fit_column_stats(dataset: Any, columns: list[str], path: Path) -> dict[str, Any]:
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
    _write_json(path, statistics)
    return statistics


class LeWMTransform:
    """Apply the released image and column preprocessing without spt.data imports."""

    def __init__(
        self,
        *,
        image: dict[str, Any],
        columns: dict[str, Any],
        preprocess_images: bool = True,
    ) -> None:
        import torch

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


def _preprocess_image_batch(
    pixels: Any,
    *,
    mean: Any,
    std: Any,
    size: int,
):
    """Apply the released LeWM image transform to a batch on its device."""

    import torch
    import torch.nn.functional as functional

    if pixels.dtype != torch.uint8:
        raise TypeError(
            "Device-side image preprocessing expects uint8 dataset pixels."
        )
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


def _model_config(protocol: dict[str, Any], action_dim: int) -> dict[str, Any]:
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


def _build_training_module(
    world_model: Any,
    protocol: dict[str, Any],
    total_steps: int,
    *,
    device_image_preprocessing: bool,
):
    import lightning as pl
    import stable_worldmodel as swm
    import torch

    class LeWMTrainingModule(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.model = world_model
            self.device_image_preprocessing = device_image_preprocessing
            if device_image_preprocessing:
                image = protocol["image_preprocessing"]
                self.register_buffer(
                    "image_mean",
                    torch.tensor(image["mean"], dtype=torch.float32).reshape(
                        1, 1, 3, 1, 1
                    ),
                    persistent=False,
                )
                self.register_buffer(
                    "image_std",
                    torch.tensor(image["std"], dtype=torch.float32).reshape(
                        1, 1, 3, 1, 1
                    ),
                    persistent=False,
                )
            sigreg = protocol["loss"]["sigreg"]
            self.sigreg = swm.wm.SIGReg(
                knots=sigreg["knots"], num_proj=sigreg["num_projections"]
            )

        def _forward_loss(self, batch: dict[str, Any], stage: str):
            episode_ids = batch.pop("_tdwm_episode_id", None)
            cache_bytes = batch.pop("_tdwm_cache_bytes", None)
            if episode_ids is not None:
                self.log(
                    f"{stage}/unique_episodes_per_batch",
                    torch.unique(episode_ids).numel(),
                    on_step=stage == "train",
                    on_epoch=True,
                    sync_dist=False,
                )
            if cache_bytes is not None:
                self.log(
                    f"{stage}/compressed_cache_gib",
                    float(cache_bytes) / 1024**3,
                    on_step=stage == "train",
                    on_epoch=True,
                    sync_dist=False,
                )
            if self.device_image_preprocessing:
                batch["pixels"] = _preprocess_image_batch(
                    batch["pixels"],
                    mean=self.image_mean,
                    std=self.image_std,
                    size=protocol["image_preprocessing"]["size"],
                )
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)
            output = self.model.encode(batch)
            history = protocol["sequence"]["history_frames"]
            predictions = protocol["sequence"]["prediction_frames"]
            predicted = self.model.predict(
                output["emb"][:, :history], output["act_emb"][:, :history]
            )
            target = output["emb"][:, predictions:]
            prediction_loss = (predicted - target).pow(2).mean()
            sigreg_loss = self.sigreg(output["emb"].transpose(0, 1))
            loss = prediction_loss + protocol["loss"]["sigreg"]["weight"] * sigreg_loss
            self.log_dict(
                {
                    f"{stage}/loss": loss,
                    f"{stage}/prediction_loss": prediction_loss,
                    f"{stage}/sigreg_loss": sigreg_loss,
                },
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=stage != "train",
                sync_dist=False,
            )
            return loss

        def training_step(self, batch: dict[str, Any], batch_idx: int):
            return self._forward_loss(batch, "train")

        def validation_step(self, batch: dict[str, Any], batch_idx: int):
            return self._forward_loss(batch, "validation")

        def configure_optimizers(self):
            optimizer_cfg = protocol["optimizer"]
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=optimizer_cfg["learning_rate"],
                weight_decay=optimizer_cfg["weight_decay"],
            )
            warmup_steps = max(
                1, int(protocol["scheduler"]["warmup_fraction"] * total_steps)
            )

            def learning_rate_scale(step: int) -> float:
                if step < warmup_steps:
                    return float(step + 1) / float(warmup_steps)
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=learning_rate_scale
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

    return LeWMTrainingModule()


def _build_gt_training_module(
    world_model: Any,
    protocol: dict[str, Any],
    total_steps: int,
    *,
    device_image_preprocessing: bool,
):
    """Build LeWM plus the goal-conditioned N-step tail-value objective."""

    import lightning as pl
    import stable_worldmodel as swm
    import torch

    class GTLeWMTrainingModule(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.model = world_model
            self.device_image_preprocessing = device_image_preprocessing
            if device_image_preprocessing:
                image = protocol["image_preprocessing"]
                self.register_buffer(
                    "image_mean",
                    torch.tensor(image["mean"], dtype=torch.float32).reshape(
                        1, 1, 3, 1, 1
                    ),
                    persistent=False,
                )
                self.register_buffer(
                    "image_std",
                    torch.tensor(image["std"], dtype=torch.float32).reshape(
                        1, 1, 3, 1, 1
                    ),
                    persistent=False,
                )
            sigreg = protocol["loss"]["sigreg"]
            self.sigreg = swm.wm.SIGReg(
                knots=sigreg["knots"], num_proj=sigreg["num_projections"]
            )
            tail_config = protocol["tail_value"]
            embed_dim = protocol["model"]["embed_dim"]
            self.value = GoalTailValue(
                embed_dim=embed_dim,
                hidden_dim=tail_config["hidden_dim"],
            )
            self.target_value = copy.deepcopy(self.value)
            self.target_value.requires_grad_(False)
            self.gamma = float(tail_config["gamma"])
            self.target_tau = float(tail_config["target_tau"])
            self.value_horizon = int(tail_config["horizon"])

        def _log_batch_metadata(self, batch: dict[str, Any], stage: str) -> None:
            episode_ids = batch.pop("_tdwm_episode_id", None)
            cache_bytes = batch.pop("_tdwm_cache_bytes", None)
            if episode_ids is not None:
                self.log(
                    f"{stage}/unique_episodes_per_batch",
                    torch.unique(episode_ids).numel(),
                    on_step=stage == "train",
                    on_epoch=True,
                    sync_dist=False,
                )
            if cache_bytes is not None:
                self.log(
                    f"{stage}/compressed_cache_gib",
                    float(cache_bytes) / 1024**3,
                    on_step=stage == "train",
                    on_epoch=True,
                    sync_dist=False,
                )

        def _forward_loss(self, batch: dict[str, Any], stage: str):
            self._log_batch_metadata(batch, stage)
            if self.device_image_preprocessing:
                batch["pixels"] = _preprocess_image_batch(
                    batch["pixels"],
                    mean=self.image_mean,
                    std=self.image_std,
                    size=protocol["image_preprocessing"]["size"],
                )
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)
            output = self.model.encode(batch)
            embeddings = output["emb"]
            history = protocol["sequence"]["history_frames"]
            prediction_start = protocol["sequence"]["prediction_frames"]
            if embeddings.shape[1] < history + self.value_horizon:
                raise RuntimeError(
                    "GT-LeWM requires each batch clip to include the full "
                    "history and tail-value horizon."
                )

            predicted = self.model.predict(
                embeddings[:, :history], output["act_emb"][:, :history]
            )
            target = embeddings[:, prediction_start : prediction_start + history]
            prediction_loss = (predicted - target).pow(2).mean()
            sigreg_loss = self.sigreg(embeddings.transpose(0, 1))

            current = embeddings[:, history - 1]
            future = embeddings[:, history : history + self.value_horizon]
            goal = future[:, -1]
            value_prediction = self.value(current, goal).squeeze(-1)
            with torch.no_grad():
                bootstrap = self.target_value(
                    future[:, -1].detach(), goal.detach()
                ).squeeze(-1)
                value_target = discounted_goal_tail_target(
                    future.detach(),
                    goal.detach(),
                    bootstrap,
                    gamma=self.gamma,
                )
            tail_loss = goal_tail_loss(value_prediction, value_target)
            loss = (
                prediction_loss
                + protocol["loss"]["sigreg"]["weight"] * sigreg_loss
                + protocol["tail_value"]["loss_weight"] * tail_loss
            )
            self.log_dict(
                {
                    f"{stage}/loss": loss,
                    f"{stage}/prediction_loss": prediction_loss,
                    f"{stage}/sigreg_loss": sigreg_loss,
                    f"{stage}/tail_value_loss": tail_loss,
                    f"{stage}/tail_value_target": value_target.mean(),
                },
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=stage != "train",
                sync_dist=False,
            )
            return loss

        def training_step(self, batch: dict[str, Any], batch_idx: int):
            return self._forward_loss(batch, "train")

        def validation_step(self, batch: dict[str, Any], batch_idx: int):
            return self._forward_loss(batch, "validation")

        def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
            soft_update(self.target_value, self.value, tau=self.target_tau)

        def configure_optimizers(self):
            optimizer_cfg = protocol["optimizer"]
            optimizer = torch.optim.AdamW(
                [*self.model.parameters(), *self.value.parameters()],
                lr=optimizer_cfg["learning_rate"],
                weight_decay=optimizer_cfg["weight_decay"],
            )
            warmup_steps = max(
                1, int(protocol["scheduler"]["warmup_fraction"] * total_steps)
            )

            def learning_rate_scale(step: int) -> float:
                if step < warmup_steps:
                    return float(step + 1) / float(warmup_steps)
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=learning_rate_scale
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

    return GTLeWMTrainingModule()


def _build_export_callback(run_dir: Path, model_config: dict[str, Any]):
    import lightning as pl
    from omegaconf import OmegaConf

    export_config = OmegaConf.create(model_config)

    class ExportPretrainedCallback(pl.Callback):
        def on_train_epoch_end(self, trainer, pl_module) -> None:
            if not trainer.is_global_zero:
                return
            import stable_worldmodel as swm

            swm.wm.save_pretrained(
                pl_module.model,
                run_name=f"epoch_{trainer.current_epoch + 1:02d}",
                config=export_config,
                cache_dir=str(run_dir / "checkpoints" / "exports"),
            )

    return ExportPretrainedCallback()


def _build_gt_export_callback(
    run_dir: Path,
    model_config: dict[str, Any],
    protocol: dict[str, Any],
):
    """Export the standard LeWM weights and the project-owned value head."""

    import lightning as pl

    base_export = _build_export_callback(run_dir, model_config)

    class GTExportPretrainedCallback(pl.Callback):
        def on_train_epoch_end(self, trainer, pl_module) -> None:
            if not trainer.is_global_zero:
                return
            base_export.on_train_epoch_end(trainer, pl_module)
            import torch

            export_dir = run_dir / "checkpoints" / "gt_lewm"
            export_dir.mkdir(parents=True, exist_ok=True)
            export_path = export_dir / f"epoch_{trainer.current_epoch + 1:02d}.pt"
            torch.save(
                {
                    "value_state_dict": pl_module.value.state_dict(),
                    "target_value_state_dict": pl_module.target_value.state_dict(),
                    "value_config": {
                        "embed_dim": protocol["model"]["embed_dim"],
                        "hidden_dim": protocol["tail_value"]["hidden_dim"],
                        "gamma": protocol["tail_value"]["gamma"],
                        "horizon": protocol["tail_value"]["horizon"],
                        "target_tau": protocol["tail_value"]["target_tau"],
                    },
                    "world_model_config": model_config,
                },
                export_path,
            )

    return GTExportPretrainedCallback()


def _build_generator_callback(generator: Any):
    import lightning as pl

    class DataLoaderGeneratorCallback(pl.Callback):
        @property
        def state_key(self) -> str:
            return "tdwm_train_dataloader_generator"

        def state_dict(self) -> dict[str, Any]:
            return {"generator_state": generator.get_state()}

        def load_state_dict(self, state_dict: dict[str, Any]) -> None:
            generator.set_state(state_dict["generator_state"])

    return DataLoaderGeneratorCallback()


def _build_episode_epoch_callback(dataset: Any):
    import lightning as pl

    class EpisodeStreamingEpochCallback(pl.Callback):
        def on_train_epoch_start(self, trainer, pl_module) -> None:
            dataset.set_epoch(int(trainer.current_epoch))

    return EpisodeStreamingEpochCallback()


def _build_metrics_logger(run_dir: Path, logging_config: dict[str, Any]):
    from lightning.pytorch.loggers import CSVLogger

    return CSVLogger(
        save_dir=str(run_dir),
        name="metrics",
        flush_logs_every_n_steps=logging_config["flush_every_n_steps"],
    )


def _resolve_loader_runtime(
    loader_config: dict[str, Any],
    *,
    smoke: bool,
    workers: int | None = None,
    prefetch_factor: int | None = None,
    validation_workers: int | None = None,
) -> dict[str, dict[str, Any]]:
    configured_train_workers = int(loader_config["workers"])
    configured_prefetch_factor = int(loader_config["prefetch_factor"])
    configured_validation_workers = int(
        loader_config.get("validation_workers", configured_train_workers)
    )
    train_workers = configured_train_workers if workers is None else workers
    train_prefetch_factor = (
        configured_prefetch_factor if prefetch_factor is None else prefetch_factor
    )
    validation_workers = (
        configured_validation_workers
        if validation_workers is None
        else validation_workers
    )
    if min(train_workers, validation_workers) < 0:
        raise ValueError("Loader workers cannot be negative.")
    if train_prefetch_factor <= 0:
        raise ValueError("Loader prefetch_factor must be positive.")

    def runtime(*, configured_workers: int, active_workers: int, prefetch: int):
        effective_workers = 0 if smoke else active_workers
        return {
            "configured_workers": configured_workers,
            "workers": effective_workers,
            "persistent_workers": effective_workers > 0,
            "prefetch_factor": prefetch if effective_workers else None,
        }

    return {
        "train": runtime(
            configured_workers=configured_train_workers,
            active_workers=train_workers,
            prefetch=train_prefetch_factor,
        ),
        "validation": runtime(
            configured_workers=configured_validation_workers,
            active_workers=validation_workers,
            prefetch=configured_prefetch_factor,
        ),
    }


def _resolve_block_shuffle(
    loader_config: dict[str, Any],
    *,
    override: bool | None = None,
    block_size: int | None = None,
) -> dict[str, Any]:
    configured = bool(loader_config["block_shuffle"])
    effective_block_size = (
        int(loader_config["block_size"]) if block_size is None else block_size
    )
    if effective_block_size < int(loader_config["batch_size"]):
        raise ValueError("block_size must be at least loader.batch_size.")
    if effective_block_size % int(loader_config["batch_size"]):
        raise ValueError("block_size must be divisible by loader.batch_size.")
    return {
        "configured": configured,
        "effective": configured if override is None else override,
        "block_size": effective_block_size,
        "shuffle_batches_within_block": bool(
            loader_config["shuffle_batches_within_block"]
        ),
    }


def _resolve_block_prefetch(
    loader_config: dict[str, Any],
    *,
    override: bool | None = None,
    block_size: int | None = None,
) -> dict[str, Any]:
    configured = bool(loader_config["block_prefetch"])
    effective_block_size = (
        int(loader_config["block_prefetch_size"])
        if block_size is None
        else block_size
    )
    if effective_block_size < int(loader_config["batch_size"]):
        raise ValueError("block_prefetch_size must be at least loader.batch_size.")
    if effective_block_size % int(loader_config["batch_size"]):
        raise ValueError(
            "block_prefetch_size must be divisible by loader.batch_size."
        )
    return {
        "configured": configured,
        "effective": configured if override is None else override,
        "block_size": effective_block_size,
    }


def _resolve_episode_streaming(
    loader_config: dict[str, Any],
    *,
    dataset_format: str,
    smoke: bool,
    override: bool | None = None,
) -> dict[str, Any]:
    configured = bool(loader_config["episode_streaming"])
    requested = configured if override is None else override
    batch_size = int(loader_config["batch_size"])
    pool_size = int(loader_config["episode_pool_size"])
    if smoke:
        pool_size = batch_size
    return {
        "configured": configured,
        "requested": requested,
        "effective": requested and dataset_format == "lance",
        "dataset_format": dataset_format,
        "pool_size": pool_size,
        "read_size": int(loader_config["episode_read_size"]),
        "cache_bytes": int(loader_config["episode_cache_bytes"]),
        "prefetch_blocks": int(loader_config["episode_prefetch_blocks"]),
        "minimum_unique_episodes_per_batch": int(
            loader_config["minimum_unique_episodes_per_batch"]
        ),
    }


def _resolve_stride_aware_lance(
    loader_config: dict[str, Any],
    *,
    dataset_format: str,
    override: bool | None = None,
) -> dict[str, Any]:
    configured = bool(loader_config["stride_aware_lance"])
    requested = configured if override is None else override
    return {
        "configured": configured,
        "requested": requested,
        "effective": requested and dataset_format == "lance",
        "dataset_format": dataset_format,
    }


def _resolve_device_image_preprocessing(
    loader_config: dict[str, Any], *, override: bool | None = None
) -> dict[str, bool]:
    configured = bool(loader_config["device_image_preprocessing"])
    return {
        "configured": configured,
        "effective": configured if override is None else override,
    }


def _resolve_model_compile(
    training_config: dict[str, Any], *, override: bool | None = None
) -> dict[str, Any]:
    configured = bool(training_config["model_compile"])
    return {
        "configured": configured,
        "effective": configured if override is None else override,
        "mode": training_config["model_compile_mode"],
    }


def _compile_world_model(world_model: Any, *, mode: str) -> None:
    """Compile only LeWM's two public compute paths, preserving checkpoints."""

    import torch

    world_model.encode = torch.compile(world_model.encode, mode=mode)
    world_model.predict = torch.compile(world_model.predict, mode=mode)


def _resolve_train_batch_limit(
    *, smoke: bool, max_steps: int | None, train_loader_length: int
) -> int | float:
    if smoke:
        return min(2, train_loader_length)
    if max_steps is not None:
        # Finish a short profiling run at an epoch boundary. Lightning's
        # mid-epoch max_steps teardown can abort persistent DataLoader workers.
        return min(max_steps, train_loader_length)
    return 1.0


def train_lewm(
    *,
    protocol_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    seed: int,
    smoke: bool = False,
    resume: str = "auto",
    max_steps: int | None = None,
    skip_validation: bool = False,
    loader_workers: int | None = None,
    loader_prefetch_factor: int | None = None,
    validation_loader_workers: int | None = None,
    block_shuffle: bool | None = None,
    block_size: int | None = None,
    block_prefetch: bool | None = None,
    block_prefetch_size: int | None = None,
    episode_streaming: bool | None = None,
    stride_aware_lance: bool | None = None,
    device_image_preprocessing: bool | None = None,
    compile_model: bool | None = None,
    run_label: str | None = None,
) -> dict[str, Any]:
    protocol = load_training_protocol(protocol_path)
    if seed not in protocol["seeds"]:
        raise ValueError(f"Seed {seed} is not in the locked seeds {protocol['seeds']}.")
    if resume not in {"auto", "never", "required"}:
        raise ValueError("resume must be one of: auto, never, required.")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive when provided.")
    if run_label and Path(run_label).name != run_label:
        raise ValueError("run_label must be a single directory name.")

    dataset_source = validate_cube_training_dataset(
        dataset_path, protocol["dataset"]
    )
    dataset_path = Path(dataset_source["path"])

    output_dir = Path(output_dir).expanduser().resolve()
    run_name = f"seed_{seed}_smoke" if smoke else f"seed_{seed}"
    if run_label:
        run_name = f"{run_name}_{run_label}"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    compatibility = prepare_cloud_runtime() or {}
    compatibility["lightning_external_callbacks"] = {
        "reason": (
            "the cloud's stable-pretraining callback registry imports torchvision "
            "APIs absent from its installed version and writes unrelated artifacts"
        ),
        "disabled": True,
    }

    import hydra
    import lightning as pl
    import stable_worldmodel as swm
    import torch
    from lightning.pytorch.callbacks import ModelCheckpoint

    package_version = importlib.metadata.version("stable-worldmodel")
    expected_version = protocol["runtime"]["stable_worldmodel_version"]
    if package_version != expected_version:
        raise RuntimeError(
            f"Expected stable-worldmodel {expected_version}, found {package_version}."
        )

    pl.seed_everything(seed, workers=True)
    sequence = protocol["sequence"]
    dataset_cfg = protocol["dataset"]
    loader_cfg = protocol["loader"]
    block_sampler = _resolve_block_shuffle(
        loader_cfg, override=block_shuffle, block_size=block_size
    )
    validation_locality = bool(loader_cfg["validation_locality"])
    prefetched_blocks = _resolve_block_prefetch(
        loader_cfg,
        override=block_prefetch,
        block_size=block_prefetch_size,
    )
    episode_stream = _resolve_episode_streaming(
        loader_cfg,
        dataset_format=dataset_source["format"],
        smoke=smoke,
        override=episode_streaming,
    )
    stride_loader = _resolve_stride_aware_lance(
        loader_cfg,
        dataset_format=dataset_source["format"],
        override=stride_aware_lance,
    )
    device_preprocessing = _resolve_device_image_preprocessing(
        loader_cfg, override=device_image_preprocessing
    )
    model_compile = _resolve_model_compile(
        protocol["training"], override=compile_model
    )
    dataset = swm.data.load_dataset(
        str(dataset_path),
        format=dataset_source["format"],
        transform=None,
        num_steps=sequence["num_steps"],
        frameskip=sequence["frame_skip"],
        keys_to_load=list(dataset_cfg["keys_to_load"]),
        keys_to_cache=list(dataset_cfg["keys_to_cache"]),
        keys_to_merge=dict(dataset_cfg["keys_to_merge"]),
    )
    actual_episodes = len(dataset.lengths)
    actual_transitions = int(np.asarray(dataset.lengths).sum())
    if actual_episodes != dataset_cfg["expected_episodes"]:
        raise ValueError(
            f"Expected {dataset_cfg['expected_episodes']} episodes, "
            f"found {actual_episodes}."
        )
    if actual_transitions != dataset_cfg["expected_transitions"]:
        raise ValueError(
            f"Expected {dataset_cfg['expected_transitions']} transitions, "
            f"found {actual_transitions}."
        )
    statistics = _fit_column_stats(
        dataset,
        list(protocol["normalization"]["columns"]),
        output_dir / "column_normalization.json",
    )
    dataset.transform = LeWMTransform(
        image=protocol["image_preprocessing"],
        columns=statistics,
        preprocess_images=not device_preprocessing["effective"],
    )
    if stride_loader["effective"]:
        dataset = StrideAwareLanceDataset(dataset)

    generator = torch.Generator().manual_seed(seed)
    train_set, validation_set = torch.utils.data.random_split(
        dataset,
        [
            protocol["split"]["train_fraction"],
            protocol["split"]["validation_fraction"],
        ],
        generator=generator,
    )
    split_manifest = _save_split(
        run_dir,
        np.asarray(train_set.indices, dtype=np.int64),
        np.asarray(validation_set.indices, dtype=np.int64),
    )

    loader_runtime = _resolve_loader_runtime(
        loader_cfg,
        smoke=smoke,
        workers=loader_workers,
        prefetch_factor=loader_prefetch_factor,
        validation_workers=validation_loader_workers,
    )
    if prefetched_blocks["effective"]:
        if not block_sampler["effective"]:
            raise ValueError("block_prefetch requires block_shuffle to be enabled.")
        if not stride_loader["effective"]:
            raise ValueError("block_prefetch requires a Lance stride-aware loader.")
        if loader_runtime["train"]["workers"] == 0:
            raise ValueError("block_prefetch requires at least one loader worker.")
        # Each newly seeded worker must build the same global block schedule
        # before taking its disjoint share.  Keeping workers alive would reuse
        # their initial seed and repeat the schedule on every epoch.
        loader_runtime["train"]["persistent_workers"] = False
        loader_runtime["train"]["persistent_workers_disabled_reason"] = (
            "block_prefetch_reseeds_workers_each_epoch"
        )
    if episode_stream["effective"]:
        if block_sampler["effective"] or prefetched_blocks["effective"]:
            raise ValueError(
                "episode streaming cannot be combined with block shuffle or prefetch."
            )
        if not stride_loader["effective"]:
            raise ValueError(
                "episode streaming requires the Lance stride-aware dataset."
            )
        loader_runtime["train"] = {
            **loader_runtime["train"],
            "workers": 0,
            "persistent_workers": False,
            "prefetch_factor": None,
            "forced_single_process_reason": (
                "one bounded cache and one sequential background reader"
            ),
        }
    train_loader_kwargs = {
        "num_workers": loader_runtime["train"]["workers"],
        "persistent_workers": loader_runtime["train"]["persistent_workers"],
        "prefetch_factor": loader_runtime["train"]["prefetch_factor"],
        "pin_memory": loader_cfg["pin_memory"],
        "generator": generator,
    }
    episode_train_dataset = None
    if episode_stream["effective"]:
        episode_train_dataset = EpisodeStreamingBatchDataset(
            dataset,
            train_set.indices,
            batch_size=loader_cfg["batch_size"],
            active_episodes=episode_stream["pool_size"],
            read_episodes=episode_stream["read_size"],
            cache_bytes=episode_stream["cache_bytes"],
            prefetch_blocks=episode_stream["prefetch_blocks"],
            seed=seed,
            drop_last=loader_cfg["train_drop_last"],
            min_unique_episodes=episode_stream[
                "minimum_unique_episodes_per_batch"
            ],
        )
        train_loader = torch.utils.data.DataLoader(
            episode_train_dataset,
            batch_size=None,
            num_workers=0,
            pin_memory=loader_cfg["pin_memory"],
        )
    elif prefetched_blocks["effective"]:
        train_loader = torch.utils.data.DataLoader(
            BlockPrefetchBatchDataset(
                dataset,
                train_set.indices,
                batch_size=loader_cfg["batch_size"],
                block_size=prefetched_blocks["block_size"],
                drop_last=loader_cfg["train_drop_last"],
                seed=seed,
                shuffle_batches_within_block=block_sampler[
                    "shuffle_batches_within_block"
                ],
            ),
            batch_size=None,
            **train_loader_kwargs,
        )
    elif block_sampler["effective"]:
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_sampler=BlockShuffleBatchSampler(
                train_set.indices,
                batch_size=loader_cfg["batch_size"],
                block_size=block_sampler["block_size"],
                drop_last=loader_cfg["train_drop_last"],
                generator=generator,
                shuffle_batches_within_block=block_sampler[
                    "shuffle_batches_within_block"
                ],
            ),
            **train_loader_kwargs,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=loader_cfg["batch_size"],
            drop_last=loader_cfg["train_drop_last"],
            shuffle=loader_cfg["train_shuffle"],
            **train_loader_kwargs,
        )
    validation_loader_kwargs = {
        "num_workers": loader_runtime["validation"]["workers"],
        "persistent_workers": loader_runtime["validation"]["persistent_workers"],
        "prefetch_factor": loader_runtime["validation"]["prefetch_factor"],
        "pin_memory": loader_cfg["pin_memory"],
    }
    if validation_locality:
        validation_loader = torch.utils.data.DataLoader(
            validation_set,
            batch_sampler=BlockShuffleBatchSampler(
                validation_set.indices,
                batch_size=loader_cfg["batch_size"],
                block_size=block_sampler["block_size"],
                drop_last=loader_cfg["validation_drop_last"],
                shuffle_batches_within_block=False,
                shuffle_blocks=False,
            ),
            **validation_loader_kwargs,
        )
    else:
        validation_loader = torch.utils.data.DataLoader(
            validation_set,
            batch_size=loader_cfg["batch_size"],
            drop_last=loader_cfg["validation_drop_last"],
            shuffle=loader_cfg["validation_shuffle"],
            **validation_loader_kwargs,
        )

    action_dim = int(dataset.get_dim("action"))
    model_config = _model_config(protocol, action_dim)
    world_model = hydra.utils.instantiate(model_config)
    parameter_count = sum(parameter.numel() for parameter in world_model.parameters())
    expected_parameters = protocol["model"].get("parameters")
    if expected_parameters and parameter_count != expected_parameters:
        raise ValueError(
            f"Expected {expected_parameters} parameters, found {parameter_count}."
        )
    if model_compile["effective"]:
        _compile_world_model(world_model, mode=model_compile["mode"])

    formal_steps = protocol["training"]["scheduler_epochs"] * len(train_loader)
    train_batch_limit = _resolve_train_batch_limit(
        smoke=smoke,
        max_steps=max_steps,
        train_loader_length=len(train_loader),
    )
    total_steps = min(2, len(train_loader)) if smoke else formal_steps
    if max_steps is not None:
        total_steps = int(train_batch_limit)
    is_gt_lewm = protocol["method"] == "gt_lewm"
    module_builder = _build_gt_training_module if is_gt_lewm else _build_training_module
    module = module_builder(
        world_model,
        protocol,
        total_steps,
        device_image_preprocessing=device_preprocessing["effective"],
    )
    checkpoint_dir = run_dir / "checkpoints" / "lightning"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch-{epoch:02d}",
        every_n_epochs=protocol["training"]["checkpoint_every_epochs"],
        save_last=True,
        save_top_k=-1,
    )
    metrics_logger = _build_metrics_logger(run_dir, protocol["logging"])
    epochs = 1 if smoke or max_steps is not None else protocol["training"]["epochs"]
    export_callback = (
        _build_gt_export_callback(run_dir, model_config, protocol)
        if is_gt_lewm
        else _build_export_callback(run_dir, model_config)
    )
    with patch(
        "lightning.pytorch.trainer.connectors.callback_connector._load_external_callbacks",
        return_value=[],
    ):
        trainer = pl.Trainer(
            default_root_dir=run_dir,
            accelerator="gpu",
            devices=1,
            precision=protocol["training"]["precision"],
            max_epochs=epochs,
            max_steps=-1,
            gradient_clip_val=protocol["training"]["gradient_clip_norm"],
            limit_train_batches=train_batch_limit,
            limit_val_batches=0.0 if skip_validation else (1 if smoke else 1.0),
            num_sanity_val_steps=0 if skip_validation else 1,
            logger=metrics_logger,
            callbacks=[
                checkpoint_callback,
                export_callback,
                _build_generator_callback(generator),
            ]
            + (
                [_build_episode_epoch_callback(episode_train_dataset)]
                if episode_train_dataset is not None
                else []
            ),
            log_every_n_steps=1 if smoke else 50,
        )

    last_checkpoint = checkpoint_dir / "last.ckpt"
    if resume == "required" and not last_checkpoint.is_file():
        raise FileNotFoundError(
            f"Required resume checkpoint not found: {last_checkpoint}"
        )
    checkpoint_path = (
        None
        if resume == "never"
        else (str(last_checkpoint) if last_checkpoint.is_file() else None)
    )
    manifest = {
        "protocol": protocol,
        "protocol_path": str(Path(protocol_path).resolve()),
        "seed": seed,
        "smoke": smoke,
        "dataset": {
            **dataset_source,
            "episodes": actual_episodes,
            "transitions": actual_transitions,
            "sequence_samples": len(dataset),
        },
        "split": split_manifest,
        "model": {"config": model_config, "parameters": parameter_count},
        "training": {
            "formal_optimizer_steps": formal_steps,
            "configured_optimizer_steps": total_steps,
            "loader_runtime": loader_runtime,
            "block_shuffle": block_sampler,
            "block_prefetch": prefetched_blocks,
            "episode_streaming": episode_stream,
            "validation_locality": {
                "enabled": validation_locality,
                "order": "source_clip_index" if validation_locality else "subset_order",
            },
            "stride_aware_lance": {
                **stride_loader,
                "upstream_fetched_rows_per_clip": sequence["num_steps"]
                * sequence["frame_skip"],
                "stride_fetched_image_rows_per_clip": sequence["num_steps"],
            },
            "device_image_preprocessing": device_preprocessing,
            "model_compile": model_compile,
            "max_steps": max_steps,
            "skip_validation": skip_validation,
            "resume_mode": resume,
            "resumed_from": checkpoint_path,
        },
        "logging": {
            **protocol["logging"],
            "path": metrics_logger.log_dir,
        },
        "runtime": {
            "stable_worldmodel": package_version,
            "stable_pretraining": importlib.metadata.version("stable-pretraining"),
            "lightning": importlib.metadata.version("lightning"),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "tdwm_git_revision": os.environ.get("TDWM_CODE_REVISION")
            or _git_revision(),
            "tdwm_worktree_revision": _git_revision(),
            "cuda_device": torch.cuda.get_device_name(0),
            "compatibility_adapter": compatibility,
        },
    }
    _write_json(run_dir / "training_manifest.json", manifest)
    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
        ckpt_path=checkpoint_path,
    )
    result = {
        "run_dir": str(run_dir),
        "seed": seed,
        "smoke": smoke,
        "resumed_from": checkpoint_path,
        "last_checkpoint": str(last_checkpoint),
        "final_epoch": trainer.current_epoch,
        "global_step": trainer.global_step,
        "callback_metrics": {
            key: value.detach().cpu().item()
            for key, value in trainer.callback_metrics.items()
            if value.numel() == 1
        },
    }
    result_name = (
        "training_result_resume_check.json"
        if smoke and checkpoint_path
        else "training_result.json"
    )
    _write_json(run_dir / result_name, result)
    return _jsonable(result)
