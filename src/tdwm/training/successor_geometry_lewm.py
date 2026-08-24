"""End-to-end training for directed Successor-Geometry LeWM."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
import yaml

from tdwm.adapters import prepare_cloud_runtime
from tdwm.methods.successor_geometry_lewm import (
    DirectedSuccessorGeometry,
    successor_geometry_objective,
)
from tdwm.training.cube_data import validate_cube_training_dataset
from tdwm.training.gt_lewm_support import (
    LeWMTransform,
    build_metrics_logger,
    build_model_config,
    compile_world_model,
    fit_column_stats,
    preprocess_image_batch,
    resolve_train_batch_limit,
    save_split,
    write_json,
)
from tdwm.training.joint_td_gt_lewm import rollout_from_latents
from tdwm.training.lance_batch import (
    EpisodeStreamingBatchDataset,
    StrideAwareLanceDataset,
)
from tdwm.training.lewm import _git_revision


METHOD = "successor_geometry_lewm"
OBJECTIVE_VERSION = 1


class EpisodeTaggedSubset(torch.utils.data.Subset):
    """Preserve source episode ids for contrastive negative masking."""

    def __getitem__(self, position: int) -> dict[str, Any]:
        return self.__getitems__([position])[0]

    def __getitems__(self, positions: list[int]) -> list[dict[str, Any]]:
        source_indices = [int(self.indices[int(position)]) for position in positions]
        batched_get = getattr(self.dataset, "__getitems__", None)
        if batched_get is None:
            samples = [self.dataset[index] for index in source_indices]
        else:
            samples = batched_get(source_indices)
        tagged = []
        for sample, source_index in zip(samples, source_indices, strict=True):
            episode = int(self.dataset.clip_indices[source_index][0])
            tagged.append(
                {
                    **sample,
                    "_tdwm_episode_id": torch.tensor(episode, dtype=torch.int32),
                }
            )
        return tagged


class EpisodeDiverseBatchSampler:
    """Build fixed validation batches with at most one clip per episode."""

    def __init__(
        self,
        source_indices: list[int] | np.ndarray,
        clip_indices: list[tuple[int, int]],
        *,
        batch_size: int,
        seed: int,
    ) -> None:
        if batch_size < 2:
            raise ValueError("Episode-diverse batches require batch_size >= 2.")
        positions_by_episode: dict[int, deque[int]] = {}
        for position, raw_source_index in enumerate(source_indices):
            source_index = int(raw_source_index)
            episode = int(clip_indices[source_index][0])
            positions_by_episode.setdefault(episode, deque()).append(position)
        if len(positions_by_episode) < batch_size:
            raise ValueError("The validation split has too few distinct episodes.")

        rng = np.random.default_rng(seed)
        episode_order = list(positions_by_episode)
        rng.shuffle(episode_order)
        active = deque(episode_order)
        batches: list[tuple[int, ...]] = []
        while len(active) >= batch_size:
            batch = []
            for _ in range(batch_size):
                episode = active.popleft()
                batch.append(positions_by_episode[episode].popleft())
                if positions_by_episode[episode]:
                    active.append(episode)
            batches.append(tuple(batch))
        if not batches:
            raise ValueError("The validation split produced no complete batch.")
        self._batches = tuple(batches)

    def __iter__(self):
        return (list(batch) for batch in self._batches)

    def __len__(self) -> int:
        return len(self._batches)


def load_successor_geometry_training_protocol(
    path: str | Path,
) -> dict[str, Any]:
    with Path(path).open() as stream:
        protocol = yaml.safe_load(stream)
    validate_successor_geometry_training_protocol(protocol)
    return protocol


def validate_successor_geometry_training_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema_version") != 1 or protocol.get("method") != METHOD:
        raise ValueError("Successor-Geometry LeWM requires its schema 1 protocol.")
    if protocol.get("environment") != "cube" or protocol.get("stage") != "full_training":
        raise ValueError("Successor-Geometry LeWM training is locked to Cube.")
    if protocol.get("initialization") != "random_from_scratch":
        raise ValueError("The primary method is jointly trained from scratch.")
    if protocol.get("runtime", {}).get("stable_worldmodel_version") != "0.1.1":
        raise ValueError("Successor-Geometry LeWM requires stable-worldmodel 0.1.1.")
    if not protocol.get("seeds"):
        raise ValueError("At least one training seed is required.")

    sequence = protocol.get("sequence", {})
    history = int(sequence.get("history_frames", 0))
    rollout = int(sequence.get("rollout_horizon", 0))
    offset = int(sequence.get("max_future_offset", 0))
    if min(history, rollout, offset) <= 0:
        raise ValueError("History, rollout, and future offset must be positive.")
    if int(sequence.get("num_steps", 0)) < history + rollout + offset:
        raise ValueError("The clip must cover history, rollout, and future goals.")
    if sequence.get("prediction_frames") != 1:
        raise ValueError("The public LeWM configuration predicts one frame at a time.")
    if int(sequence.get("frame_skip", 0)) <= 0:
        raise ValueError("sequence.frame_skip must be positive.")

    objective = protocol.get("objective", {})
    expected_objective = {
        "dynamics": "open_loop_latent_mse_h1_to_hK",
        "successor_geometry": "discounted_future_pair_infonce",
        "query_sources": ["real_terminal", "predicted_terminal"],
        "negative_sampling": "cross_episode_in_batch",
        "same_episode_negatives": "masked",
        "goal_conditioning": "none",
        "reward": "none",
        "policy": "none",
        "td_bootstrap": False,
        "dynamics_weight": 1.0,
        "geometry_weight": 1.0,
    }
    for key, value in expected_objective.items():
        if objective.get(key) != value:
            raise ValueError(f"objective.{key} must be {value!r}.")

    geometry = protocol.get("geometry", {})
    expected_geometry = {
        "objective_version": OBJECTIVE_VERSION,
        "architecture": "dual_mlp_directed_cosine",
        "max_future_offset": offset,
        "rollout_horizon": rollout,
        "query_mix": "equal_real_and_predicted",
        "loss_normalization": "cross_episode_random_baseline",
    }
    for key, value in expected_geometry.items():
        if geometry.get(key) != value:
            raise ValueError(f"geometry.{key} must be {value!r}.")
    if min(
        int(geometry.get("projection_dim", 0)),
        int(geometry.get("hidden_dim", 0)),
    ) <= 0:
        raise ValueError("Successor geometry dimensions must be positive.")
    if float(geometry.get("temperature", 0.0)) <= 0.0:
        raise ValueError("geometry.temperature must be positive.")
    if not 0.0 < float(geometry.get("gamma", 0.0)) <= 1.0:
        raise ValueError("geometry.gamma must lie in (0, 1].")

    loader = protocol.get("loader", {})
    batch_size = int(loader.get("batch_size", 0))
    if batch_size < 2:
        raise ValueError("Contrastive training requires a batch of at least two clips.")
    if loader.get("episode_streaming") is not True:
        raise ValueError("Formal training requires episode-diverse streaming batches.")
    if int(loader.get("minimum_unique_episodes_per_batch", 0)) != batch_size:
        raise ValueError("Every formal batch clip must come from a distinct episode.")
    if loader.get("train_drop_last") is not True:
        raise ValueError("Formal contrastive batches must remain full-sized.")
    if loader.get("validation_drop_last") is not True:
        raise ValueError("Validation contrastive batches must remain full-sized.")
    if loader.get("validation_episode_diverse") is not True:
        raise ValueError("Validation must guarantee one clip per episode in each batch.")
    if loader.get("validation_locality") is not False:
        raise ValueError("Locality-sorted validation conflicts with episode diversity.")

    training = protocol.get("training", {})
    if int(training.get("epochs", 0)) <= 0:
        raise ValueError("training.epochs must be positive.")
    if int(training.get("scheduler_epochs", 0)) != int(training["epochs"]):
        raise ValueError("Scheduler and training epochs must match.")
    if int(training.get("optimizer_steps_per_epoch", 0)) <= 0:
        raise ValueError("optimizer_steps_per_epoch must be positive.")
    if int(training.get("validation_batches", 0)) <= 0:
        raise ValueError("training.validation_batches must be positive.")


@dataclass(frozen=True)
class SuccessorGeometryWindows:
    history: torch.Tensor
    rollout_actions: torch.Tensor
    target_future: torch.Tensor
    real_terminal: torch.Tensor
    future_goals: torch.Tensor
    group_ids: torch.Tensor
    dynamics_count_per_clip: int
    geometry_count_per_clip: int


def build_successor_geometry_windows(
    latents: torch.Tensor,
    actions: torch.Tensor,
    *,
    history_size: int,
    rollout_horizon: int,
    max_future_offset: int,
    episode_ids: torch.Tensor | None = None,
) -> SuccessorGeometryWindows:
    """Align open-loop dynamics and future-pair geometry supervision."""

    if latents.ndim != 3:
        raise ValueError("latents must have shape (batch, time, dim).")
    if actions.ndim != 3 or actions.shape[:2] != latents.shape[:2]:
        raise ValueError("actions must share the latent batch and time axes.")
    if min(history_size, rollout_horizon, max_future_offset) <= 0:
        raise ValueError("Window sizes must be positive.")
    batch, time = latents.shape[:2]
    dynamics_count = time - history_size - rollout_horizon + 1
    geometry_count = dynamics_count - max_future_offset
    if geometry_count <= 0:
        raise ValueError("The clip contains no complete successor-geometry window.")
    if episode_ids is None:
        episode_ids = torch.arange(batch, device=latents.device)
    if episode_ids.ndim != 1 or episode_ids.shape[0] != batch:
        raise ValueError("episode_ids must contain one identifier per clip.")

    history = torch.cat(
        [latents[:, start : start + history_size] for start in range(dynamics_count)],
        dim=0,
    )
    rollout_actions = torch.cat(
        [
            actions[:, start : start + history_size + rollout_horizon - 1]
            for start in range(dynamics_count)
        ],
        dim=0,
    )
    target_future = torch.cat(
        [
            latents[
                :, start + history_size : start + history_size + rollout_horizon
            ]
            for start in range(dynamics_count)
        ],
        dim=0,
    )
    real_terminal = torch.cat(
        [
            latents[:, start + history_size + rollout_horizon - 1]
            for start in range(geometry_count)
        ],
        dim=0,
    )
    future_goals = torch.cat(
        [
            latents[
                :,
                start
                + history_size
                + rollout_horizon : start
                + history_size
                + rollout_horizon
                + max_future_offset,
            ]
            for start in range(geometry_count)
        ],
        dim=0,
    )
    return SuccessorGeometryWindows(
        history=history,
        rollout_actions=rollout_actions,
        target_future=target_future,
        real_terminal=real_terminal,
        future_goals=future_goals,
        group_ids=episode_ids.repeat(geometry_count),
        dynamics_count_per_clip=dynamics_count,
        geometry_count_per_clip=geometry_count,
    )


def _build_training_module(
    world_model: Any,
    protocol: dict[str, Any],
    total_steps: int,
    *,
    device_image_preprocessing: bool,
):
    import lightning as pl
    import stable_worldmodel as swm

    class SuccessorGeometryTrainingModule(pl.LightningModule):
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
            geometry = protocol["geometry"]
            self.geometry = DirectedSuccessorGeometry(
                embed_dim=int(protocol["model"]["embed_dim"]),
                projection_dim=int(geometry["projection_dim"]),
                hidden_dim=int(geometry["hidden_dim"]),
                temperature=float(geometry["temperature"]),
            )
            self.history_size = int(protocol["sequence"]["history_frames"])
            self.rollout_horizon = int(protocol["sequence"]["rollout_horizon"])
            self.max_future_offset = int(
                protocol["sequence"]["max_future_offset"]
            )
            self.gamma = float(geometry["gamma"])

        def _preprocess(self, pixels: torch.Tensor) -> torch.Tensor:
            if not self.device_image_preprocessing:
                return pixels
            return preprocess_image_batch(
                pixels,
                mean=self.image_mean,
                std=self.image_std,
                size=protocol["image_preprocessing"]["size"],
            )

        def _forward_loss(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
            batch_size = int(batch["pixels"].shape[0])
            episode_ids = batch.pop("_tdwm_episode_id", None)
            cache_bytes = batch.pop("_tdwm_cache_bytes", None)
            pixels = self._preprocess(batch["pixels"])
            actions = torch.nan_to_num(batch["action"], 0.0)
            encoded = self.model.encode({**batch, "pixels": pixels, "action": actions})
            latents = encoded["emb"]
            expected_steps = int(protocol["sequence"]["num_steps"])
            if latents.shape[1] != expected_steps:
                raise RuntimeError("The encoded clip has an unexpected length.")

            windows = build_successor_geometry_windows(
                latents,
                actions,
                history_size=self.history_size,
                rollout_horizon=self.rollout_horizon,
                max_future_offset=self.max_future_offset,
                episode_ids=episode_ids,
            )
            pred_proj = getattr(self.model, "pred_proj", None)
            pred_proj_was_training = (
                bool(pred_proj.training) if pred_proj is not None else False
            )
            if pred_proj is not None:
                pred_proj.eval()
            try:
                rollout = rollout_from_latents(
                    self.model,
                    windows.history,
                    windows.rollout_actions,
                    history_size=self.history_size,
                )
            finally:
                if pred_proj is not None:
                    pred_proj.train(pred_proj_was_training)
            predicted_future = rollout[
                ..., self.history_size : self.history_size + self.rollout_horizon, :
            ]
            if predicted_future.shape != windows.target_future.shape:
                raise RuntimeError("LeWM rollout and multi-horizon targets differ.")

            dynamics_mse_by_horizon = (
                predicted_future - windows.target_future
            ).square().mean(dim=(0, 2))
            dynamics_loss = dynamics_mse_by_horizon.mean()
            geometry_pairs = windows.geometry_count_per_clip * batch_size
            predicted_terminal = predicted_future[:geometry_pairs, -1]
            geometry_output = successor_geometry_objective(
                self.geometry,
                windows.real_terminal,
                predicted_terminal,
                windows.future_goals,
                windows.group_ids,
                gamma=self.gamma,
            )

            local_count = latents.shape[1] - self.history_size
            local_sequences = torch.cat(
                [
                    latents[:, start : start + self.history_size + 1]
                    for start in range(local_count)
                ],
                dim=0,
            )
            sigreg_loss = self.sigreg(local_sequences.transpose(0, 1))
            objective = protocol["objective"]
            loss = (
                float(objective["dynamics_weight"]) * dynamics_loss
                + float(protocol["loss"]["sigreg"]["weight"]) * sigreg_loss
                + float(objective["geometry_weight"]) * geometry_output.loss
            )
            metrics = {
                f"{stage}/loss": loss.detach(),
                f"{stage}/dynamics_loss": dynamics_loss.detach(),
                f"{stage}/dynamics_mse_h1": dynamics_mse_by_horizon[0].detach(),
                f"{stage}/dynamics_mse_hK": dynamics_mse_by_horizon[-1].detach(),
                f"{stage}/sigreg_loss": sigreg_loss.detach(),
                f"{stage}/successor_geometry_loss": geometry_output.loss.detach(),
                f"{stage}/successor_geometry_raw_loss": (
                    geometry_output.raw_loss.detach()
                ),
                f"{stage}/real_query_loss": geometry_output.real_query_loss.detach(),
                f"{stage}/predicted_query_loss": (
                    geometry_output.predicted_query_loss.detach()
                ),
                f"{stage}/successor_top1": geometry_output.top1.detach(),
                f"{stage}/successor_margin": geometry_output.positive_margin.detach(),
                f"{stage}/successor_loss_offset1": (
                    geometry_output.loss_by_offset[0].detach()
                ),
                f"{stage}/successor_loss_offsetK": (
                    geometry_output.loss_by_offset[-1].detach()
                ),
                f"{stage}/successor_top1_offset1": (
                    geometry_output.top1_by_offset[0].detach()
                ),
                f"{stage}/successor_top1_offsetK": (
                    geometry_output.top1_by_offset[-1].detach()
                ),
                f"{stage}/dynamics_windows": loss.new_tensor(
                    float(windows.dynamics_count_per_clip * batch_size)
                ),
                f"{stage}/geometry_pairs": loss.new_tensor(float(geometry_pairs)),
            }
            if episode_ids is not None:
                metrics[f"{stage}/unique_episodes_per_batch"] = loss.new_tensor(
                    float(torch.unique(episode_ids).numel())
                )
            if cache_bytes is not None:
                metrics[f"{stage}/compressed_cache_gib"] = loss.new_tensor(
                    float(cache_bytes) / 1024**3
                )
            self.log_dict(
                metrics,
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=stage == "validation",
                sync_dist=False,
                batch_size=batch_size,
            )
            return loss

        def training_step(self, batch: dict[str, Any], batch_idx: int):
            del batch_idx
            return self._forward_loss(batch, "train")

        def validation_step(self, batch: dict[str, Any], batch_idx: int):
            del batch_idx
            return self._forward_loss(batch, "validation")

        def configure_optimizers(self):
            optimizer_cfg = protocol["optimizer"]
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": list(self.model.parameters()),
                        "lr": optimizer_cfg["world_model_learning_rate"],
                    },
                    {
                        "params": list(self.geometry.parameters()),
                        "lr": optimizer_cfg["geometry_learning_rate"],
                    },
                ],
                weight_decay=optimizer_cfg["weight_decay"],
            )
            warmup_steps = max(
                1, int(float(protocol["scheduler"]["warmup_fraction"]) * total_steps)
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

    return SuccessorGeometryTrainingModule()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _geometry_config(
    protocol: dict[str, Any], *, base_run_name: str, base_sha256: str
) -> dict[str, Any]:
    geometry = protocol["geometry"]
    return {
        "objective_version": OBJECTIVE_VERSION,
        "architecture": geometry["architecture"],
        "embed_dim": protocol["model"]["embed_dim"],
        "projection_dim": geometry["projection_dim"],
        "hidden_dim": geometry["hidden_dim"],
        "temperature": geometry["temperature"],
        "gamma": geometry["gamma"],
        "history_size": protocol["sequence"]["history_frames"],
        "rollout_horizon": protocol["sequence"]["rollout_horizon"],
        "max_future_offset": protocol["sequence"]["max_future_offset"],
        "query_sources": ["real_terminal", "predicted_terminal"],
        "goal_conditioning": "future_pairs_only",
        "negative_sampling": "cross_episode_in_batch",
        "same_episode_negatives": "masked",
        "reward": "none",
        "policy": "none",
        "td_bootstrap": False,
        "base_export_run_name": base_run_name,
        "base_checkpoint_sha256": base_sha256,
    }


def _build_export_callback(
    run_dir: Path,
    model_config: dict[str, Any],
    protocol: dict[str, Any],
):
    import lightning as pl
    import stable_worldmodel as swm
    from omegaconf import OmegaConf

    export_config = OmegaConf.create(model_config)

    class SuccessorGeometryExportCallback(pl.Callback):
        def on_train_epoch_end(self, trainer, pl_module) -> None:
            if not trainer.is_global_zero:
                return
            epoch = trainer.current_epoch + 1
            if epoch % int(protocol["training"]["checkpoint_every_epochs"]):
                return
            base_run_name = f"epoch_{epoch:02d}"
            export_root = run_dir / "checkpoints" / "exports"
            swm.wm.save_pretrained(
                pl_module.model,
                run_name=base_run_name,
                config=export_config,
                cache_dir=str(export_root),
            )
            base_dir = export_root / "checkpoints" / base_run_name
            base_weights = sorted(base_dir.glob("*.pt"))
            if len(base_weights) != 1:
                raise RuntimeError(
                    "Stable World Model export did not contain exactly one weight file."
                )
            base_hash = _file_sha256(base_weights[0])
            deployment_dir = run_dir / "checkpoints" / METHOD
            deployment_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "method": METHOD,
                    "objective_version": OBJECTIVE_VERSION,
                    "deployment_checkpoint_version": 1,
                    "epoch": epoch,
                    "global_step": int(trainer.global_step),
                    "world_model_state_dict": pl_module.model.state_dict(),
                    "geometry_state_dict": pl_module.geometry.state_dict(),
                    "world_model_config": model_config,
                    "geometry_config": _geometry_config(
                        protocol,
                        base_run_name=base_run_name,
                        base_sha256=base_hash,
                    ),
                },
                deployment_dir / f"epoch_{epoch:02d}.pt",
            )

    return SuccessorGeometryExportCallback()


def _build_generator_callback(generator: torch.Generator):
    import lightning as pl

    class DataLoaderGeneratorCallback(pl.Callback):
        @property
        def state_key(self) -> str:
            return "tdwm_successor_geometry_dataloader_generator"

        def state_dict(self) -> dict[str, Any]:
            return {"generator_state": generator.get_state()}

        def load_state_dict(self, state_dict: dict[str, Any]) -> None:
            generator.set_state(state_dict["generator_state"])

    return DataLoaderGeneratorCallback()


def _build_episode_epoch_callback(dataset: EpisodeStreamingBatchDataset):
    import lightning as pl

    class EpisodeStreamingEpochCallback(pl.Callback):
        def on_train_epoch_start(self, trainer, pl_module) -> None:
            del pl_module
            dataset.set_epoch(int(trainer.current_epoch))

    return EpisodeStreamingEpochCallback()


def train_successor_geometry_lewm(
    *,
    protocol_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    seed: int,
    smoke: bool = False,
    resume: str = "auto",
    max_steps: int | None = None,
    skip_validation: bool = False,
) -> dict[str, Any]:
    """Jointly train LeWM dynamics and reward-free successor geometry."""

    protocol = load_successor_geometry_training_protocol(protocol_path)
    if seed not in protocol["seeds"]:
        raise ValueError(f"Seed {seed} is not in the locked seeds {protocol['seeds']}.")
    if resume not in {"auto", "never", "required"}:
        raise ValueError("resume must be one of: auto, never, required.")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive when provided.")

    dataset_path = Path(dataset_path).expanduser().resolve()
    dataset_source = validate_cube_training_dataset(dataset_path, protocol["dataset"])
    output_dir = Path(output_dir).expanduser().resolve()
    run_dir = output_dir / (f"seed_{seed}_smoke" if smoke else f"seed_{seed}")
    run_dir.mkdir(parents=True, exist_ok=True)

    compatibility = prepare_cloud_runtime() or {}
    import hydra
    import lightning as pl
    import stable_worldmodel as swm
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
    device_preprocessing = bool(loader_cfg["device_image_preprocessing"])
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
    if len(dataset.lengths) != dataset_cfg["expected_episodes"]:
        raise ValueError("Dataset episode count differs from the protocol.")
    if int(np.asarray(dataset.lengths).sum()) != dataset_cfg["expected_transitions"]:
        raise ValueError("Dataset transition count differs from the protocol.")

    statistics = fit_column_stats(
        dataset,
        list(protocol["normalization"]["columns"]),
        output_dir / "column_normalization.json",
    )
    dataset.transform = LeWMTransform(
        image=protocol["image_preprocessing"],
        columns=statistics,
        preprocess_images=not device_preprocessing,
    )
    if dataset_source["format"] == "lance":
        dataset = StrideAwareLanceDataset(dataset)

    generator = torch.Generator().manual_seed(seed)
    raw_train_set, raw_validation_set = torch.utils.data.random_split(
        dataset,
        [protocol["split"]["train_fraction"], protocol["split"]["validation_fraction"]],
        generator=generator,
    )
    train_set = EpisodeTaggedSubset(dataset, raw_train_set.indices)
    validation_set = EpisodeTaggedSubset(dataset, raw_validation_set.indices)
    split_manifest = save_split(
        run_dir,
        np.asarray(train_set.indices, dtype=np.int64),
        np.asarray(validation_set.indices, dtype=np.int64),
    )

    episode_train_dataset = None
    use_episode_streaming = bool(loader_cfg["episode_streaming"]) and not smoke
    if use_episode_streaming:
        if not isinstance(dataset, StrideAwareLanceDataset):
            raise ValueError("Episode streaming requires the audited Lance dataset.")
        episode_train_dataset = EpisodeStreamingBatchDataset(
            dataset,
            train_set.indices,
            batch_size=loader_cfg["batch_size"],
            active_episodes=loader_cfg["episode_pool_size"],
            read_episodes=loader_cfg["episode_read_size"],
            cache_bytes=loader_cfg["episode_cache_bytes"],
            prefetch_blocks=loader_cfg["episode_prefetch_blocks"],
            seed=seed,
            drop_last=loader_cfg["train_drop_last"],
            min_unique_episodes=loader_cfg["minimum_unique_episodes_per_batch"],
        )
        train_loader = torch.utils.data.DataLoader(
            episode_train_dataset,
            batch_size=None,
            num_workers=0,
            pin_memory=loader_cfg["pin_memory"],
        )
    else:
        workers = 0 if smoke else int(loader_cfg["workers"])
        train_kwargs: dict[str, Any] = {
            "num_workers": workers,
            "pin_memory": loader_cfg["pin_memory"],
        }
        if workers:
            train_kwargs.update(
                {
                    "persistent_workers": True,
                    "prefetch_factor": loader_cfg["prefetch_factor"],
                }
            )
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=loader_cfg["batch_size"],
            shuffle=loader_cfg["train_shuffle"],
            drop_last=loader_cfg["train_drop_last"],
            generator=generator,
            **train_kwargs,
        )

    validation_workers = 0 if smoke else int(loader_cfg["validation_workers"])
    validation_kwargs: dict[str, Any] = {
        "num_workers": validation_workers,
        "pin_memory": loader_cfg["pin_memory"],
    }
    if validation_workers:
        validation_kwargs.update(
            {
                "persistent_workers": True,
                "prefetch_factor": loader_cfg["prefetch_factor"],
            }
        )
    validation_loader = torch.utils.data.DataLoader(
        validation_set,
        batch_sampler=EpisodeDiverseBatchSampler(
            validation_set.indices,
            dataset.clip_indices,
            batch_size=loader_cfg["batch_size"],
            seed=seed,
        ),
        **validation_kwargs,
    )

    action_dim = int(dataset.get_dim("action"))
    model_config = build_model_config(protocol, action_dim)
    world_model = hydra.utils.instantiate(model_config)
    parameter_count = sum(parameter.numel() for parameter in world_model.parameters())
    expected_parameters = protocol["model"].get("parameters")
    if expected_parameters and parameter_count != expected_parameters:
        raise ValueError(
            f"Expected {expected_parameters} LeWM parameters, found {parameter_count}."
        )
    if protocol["training"]["model_compile"]:
        compile_world_model(
            world_model, mode=protocol["training"]["model_compile_mode"]
        )

    available_epoch_steps = len(train_loader)
    formal_epoch_steps = int(protocol["training"]["optimizer_steps_per_epoch"])
    if formal_epoch_steps > available_epoch_steps:
        raise ValueError("optimizer_steps_per_epoch exceeds available batches.")
    formal_steps = int(protocol["training"]["scheduler_epochs"]) * formal_epoch_steps
    train_limit = resolve_train_batch_limit(
        smoke=smoke,
        max_steps=max_steps,
        train_loader_length=available_epoch_steps,
    )
    if not smoke and max_steps is None:
        train_limit = formal_epoch_steps
    total_steps = int(train_limit) if (smoke or max_steps is not None) else formal_steps
    module = _build_training_module(
        world_model,
        protocol,
        total_steps,
        device_image_preprocessing=device_preprocessing,
    )

    checkpoint_dir = run_dir / "checkpoints" / "lightning"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch-{epoch:02d}",
        every_n_epochs=protocol["training"]["checkpoint_every_epochs"],
        save_last=True,
        save_top_k=-1,
    )
    callbacks = [
        checkpoint_callback,
        _build_export_callback(run_dir, model_config, protocol),
        _build_generator_callback(generator),
    ]
    if episode_train_dataset is not None:
        callbacks.append(_build_episode_epoch_callback(episode_train_dataset))
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    precision = (
        protocol["training"]["precision"] if accelerator == "gpu" else "32-true"
    )
    epochs = 1 if (smoke or max_steps is not None) else int(protocol["training"]["epochs"])
    validation_limit: int | float = 0.0
    if not smoke and not skip_validation:
        validation_limit = min(
            int(protocol["training"]["validation_batches"]), len(validation_loader)
        )
    with patch(
        "lightning.pytorch.trainer.connectors.callback_connector."
        "_load_external_callbacks",
        return_value=[],
    ):
        trainer = pl.Trainer(
            default_root_dir=run_dir,
            accelerator=accelerator,
            devices=1,
            precision=precision,
            max_epochs=epochs,
            gradient_clip_val=protocol["training"]["gradient_clip_norm"],
            limit_train_batches=train_limit,
            limit_val_batches=validation_limit,
            num_sanity_val_steps=0,
            logger=build_metrics_logger(run_dir, protocol["logging"]),
            callbacks=callbacks,
            log_every_n_steps=1 if smoke else 50,
        )

    last_checkpoint = checkpoint_dir / "last.ckpt"
    if resume == "required" and not last_checkpoint.is_file():
        raise FileNotFoundError(f"Required checkpoint not found: {last_checkpoint}")
    checkpoint_path = None
    if resume != "never" and last_checkpoint.is_file():
        manifest_path = run_dir / "training_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("Cannot verify the objective version for resume.")
        with manifest_path.open() as stream:
            previous = json.load(stream)
        previous_protocol = previous.get("protocol", {})
        if previous_protocol.get("method") != METHOD or previous_protocol.get(
            "geometry", {}
        ).get("objective_version") != OBJECTIVE_VERSION:
            raise RuntimeError("Refusing to resume an incompatible objective.")
        checkpoint_path = str(last_checkpoint)

    runtime = {
        "stable_worldmodel": package_version,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tdwm_git_revision": _git_revision(),
        "compatibility_adapter": compatibility,
    }
    if torch.cuda.is_available():
        runtime["cuda_device"] = torch.cuda.get_device_name(0)
    write_json(
        run_dir / "training_manifest.json",
        {
            "method": METHOD,
            "protocol": protocol,
            "protocol_path": str(Path(protocol_path).resolve()),
            "seed": seed,
            "dataset": {
                **dataset_source,
                "sequence_samples": len(dataset),
                "split": split_manifest,
            },
            "model": {
                "config": model_config,
                "initialization": "random_from_scratch_joint_end_to_end",
                "lewm_parameters": parameter_count,
                "geometry_parameters": sum(
                    parameter.numel() for parameter in module.geometry.parameters()
                ),
            },
            "training": {
                "formal_optimizer_steps": formal_steps,
                "optimizer_steps_per_epoch": formal_epoch_steps,
                "available_batches_per_epoch": available_epoch_steps,
                "configured_optimizer_steps": total_steps,
                "resume_mode": resume,
                "resumed_from": checkpoint_path,
                "episode_streaming": use_episode_streaming,
                "validation_batches": validation_limit,
                "validation_skipped": smoke or skip_validation,
            },
            "runtime": runtime,
        },
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
        ckpt_path=checkpoint_path,
    )
    result = {
        "run_dir": str(run_dir),
        "seed": seed,
        "last_checkpoint": str(last_checkpoint),
        "final_epoch": trainer.current_epoch,
        "global_step": trainer.global_step,
    }
    if torch.cuda.is_available():
        result["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated())
    write_json(run_dir / "training_result.json", result)
    return result


__all__ = [
    "METHOD",
    "OBJECTIVE_VERSION",
    "EpisodeDiverseBatchSampler",
    "SuccessorGeometryWindows",
    "_build_training_module",
    "build_successor_geometry_windows",
    "load_successor_geometry_training_protocol",
    "train_successor_geometry_lewm",
    "validate_successor_geometry_training_protocol",
]
