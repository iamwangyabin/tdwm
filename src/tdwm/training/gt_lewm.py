"""Standalone GT-LeWM training entry point.

This module intentionally does not modify or dispatch through the LeWM
baseline trainer. It owns its protocol, data/model assembly, Lightning module,
and exports as a separate method entry point.
"""

from __future__ import annotations

import copy
import importlib.metadata
import json
import math
import platform
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
import yaml

from tdwm.adapters import prepare_cloud_runtime
from tdwm.methods.goal_tail import (
    GoalTailValue,
    ema_update,
    future_goal_td_objective,
)
from tdwm.training.block_sampler import BlockShuffleBatchSampler
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
from tdwm.training.lance_batch import (
    EpisodeStreamingBatchDataset,
    StrideAwareLanceDataset,
)
from tdwm.training.lewm import _git_revision


def load_gt_training_protocol(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as stream:
        protocol = yaml.safe_load(stream)
    validate_gt_training_protocol(protocol)
    return protocol


def validate_gt_training_protocol(protocol: dict[str, Any]) -> None:
    """Validate the standalone GT-LeWM training protocol."""

    if protocol.get("method") != "gt_lewm":
        raise ValueError("This entry point only accepts the standalone GT-LeWM method.")
    sequence = protocol.get("sequence", {})
    tail = protocol.get("tail_value", {})
    if tail.get("objective_version") != 2:
        raise ValueError("GT-LeWM training requires objective_version 2.")
    history = sequence.get("history_frames", 0)
    max_goal_offset = tail.get("max_goal_offset", 0)
    if sequence.get("prediction_frames") != 1:
        raise ValueError("GT-LeWM requires exactly one local prediction frame.")
    if max_goal_offset <= 0 or sequence.get("num_steps") != history + max_goal_offset:
        raise ValueError(
            "GT-LeWM num_steps must equal history_frames + max_goal_offset."
        )
    if not 0 < tail.get("td_horizon", 0) <= max_goal_offset:
        raise ValueError("GT-LeWM td_horizon must lie in [1, max_goal_offset].")
    if not 0.0 <= tail.get("gamma", -1.0) < 1.0:
        raise ValueError("GT-LeWM gamma must lie in [0, 1).")
    if tail.get("loss_weight", -1.0) < 0.0:
        raise ValueError("GT-LeWM loss_weight cannot be negative.")
    if tail.get("boundary_loss_weight", -1.0) < 0.0:
        raise ValueError("GT-LeWM boundary_loss_weight cannot be negative.")
    if not 0.0 <= tail.get("loss_warmup_fraction", -1.0) < 1.0:
        raise ValueError("GT-LeWM loss_warmup_fraction must lie in [0, 1).")
    if not 0.0 <= tail.get("target_ema_decay", -1.0) < 1.0:
        raise ValueError("GT-LeWM target_ema_decay must lie in [0, 1).")
    if tail.get("goal_source") != "all_future_states_in_clip":
        raise ValueError("GT-LeWM must supervise every future goal in each clip.")
    if tail.get("continuation_policy") != "offline_dataset_behavior":
        raise ValueError("GT-LeWM tail targets require the offline dataset behavior.")

    if protocol.get("schema_version") != 1 or protocol.get("environment") != "cube":
        raise ValueError("GT-LeWM requires schema_version 1 and the Cube environment.")
    if protocol.get("runtime", {}).get("stable_worldmodel_version") != "0.1.1":
        raise ValueError("GT-LeWM is locked to stable-worldmodel 0.1.1.")
    split = protocol.get("split", {})
    if split.get("unit") != "sequence_clip":
        raise ValueError("GT-LeWM splits sequence clips.")
    if not 0.0 < split.get("train_fraction", 0.0) < 1.0:
        raise ValueError("train_fraction must lie strictly between zero and one.")
    if not math.isclose(
        split["train_fraction"] + split.get("validation_fraction", 0.0), 1.0
    ):
        raise ValueError("Training and validation fractions must sum to one.")
    training = protocol.get("training", {})
    if training.get("epochs") != training.get("scheduler_epochs"):
        raise ValueError("Scheduler and trainer epochs must remain locked together.")
    if training.get("epochs", 0) <= 0:
        raise ValueError("Training epochs must be positive.")
    if training.get("optimizer_steps_per_epoch", 0) <= 0:
        raise ValueError("optimizer_steps_per_epoch must be positive.")
    if not protocol.get("seeds"):
        raise ValueError("At least one training seed is required.")
    if protocol.get("scheduler", {}).get("interval") != "optimizer_step":
        raise ValueError("GT-LeWM scheduler must step per optimizer step.")
    loader = protocol.get("loader", {})
    if loader.get("batch_size", 0) <= 0 or loader.get("workers", -1) < 0:
        raise ValueError("GT-LeWM loader settings are invalid.")
    if loader.get("prefetch_factor", 0) <= 0:
        raise ValueError("GT-LeWM loader prefetch_factor must be positive.")
    if not isinstance(loader.get("device_image_preprocessing"), bool):
        raise ValueError("loader.device_image_preprocessing must be true or false.")
    if not isinstance(loader.get("episode_streaming"), bool):
        raise ValueError("loader.episode_streaming must be true or false.")
    if not 1 <= loader.get("minimum_unique_episodes_per_batch", 0) <= loader[
        "batch_size"
    ]:
        raise ValueError(
            "minimum_unique_episodes_per_batch must lie in [1, batch_size]."
        )
    dataset = protocol.get("dataset", {})
    lance = dataset.get("lance", {})
    if (
        lance.get("suffix") != ".lance"
        or lance.get("manifest_suffix") != ".manifest.json"
    ):
        raise ValueError("GT-LeWM Lance layout is not locked to the audited format.")
    if lance.get("image_codec") != "jpeg" or lance.get("jpeg_quality") != 100:
        raise ValueError("GT-LeWM Lance images must use JPEG quality 100.")
    if lance.get("stable_worldmodel_version") != "0.1.1":
        raise ValueError("GT-LeWM Lance conversion is locked to 0.1.1.")
    if lance.get("source", {}).get("sha256") != dataset.get(
        "optimized_layout", {}
    ).get("sha256"):
        raise ValueError(
            "GT-LeWM Lance source must match the audited optimized layout."
        )

    local_windows = sequence["num_steps"] - history
    effective_sigreg_batch = loader["batch_size"] * local_windows
    configured_sigreg_batch = protocol.get("loss", {}).get("sigreg", {}).get(
        "effective_batch_size"
    )
    if effective_sigreg_batch != configured_sigreg_batch:
        raise ValueError(
            "GT-LeWM must preserve the configured effective SIGReg batch size."
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
            tail = protocol["tail_value"]
            self.value = GoalTailValue(
                embed_dim=protocol["model"]["embed_dim"],
                hidden_dim=tail["hidden_dim"],
            )
            self.target_value = copy.deepcopy(self.value)
            self.target_value.requires_grad_(False)
            self.gamma = float(tail["gamma"])
            self.target_ema_decay = float(tail["target_ema_decay"])
            self.max_goal_offset = int(tail["max_goal_offset"])
            self.td_horizon = int(tail["td_horizon"])
            self.tail_warmup_steps = int(
                float(tail["loss_warmup_fraction"]) * total_steps
            )

        def _tail_weight_scale(self) -> float:
            if self.tail_warmup_steps <= 0:
                return 1.0
            return min(1.0, float(self.global_step + 1) / self.tail_warmup_steps)

        def _forward_loss(self, batch: dict[str, Any], stage: str):
            batch_size = int(batch["pixels"].shape[0])
            episode_ids = batch.pop("_tdwm_episode_id", None)
            if episode_ids is not None:
                self.log(
                    f"{stage}/unique_episodes_per_batch",
                    torch.unique(episode_ids).numel(),
                    on_step=stage == "train",
                    on_epoch=True,
                    sync_dist=False,
                    batch_size=batch_size,
                )
            if self.device_image_preprocessing:
                batch["pixels"] = preprocess_image_batch(
                    batch["pixels"],
                    mean=self.image_mean,
                    std=self.image_std,
                    size=protocol["image_preprocessing"]["size"],
                )
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)
            output = self.model.encode(batch)
            embeddings = output["emb"]
            history = protocol["sequence"]["history_frames"]
            if embeddings.shape[1] != history + self.max_goal_offset:
                raise RuntimeError("GT-LeWM received a clip with an unexpected length.")

            local_window_count = embeddings.shape[1] - history
            local_histories = torch.cat(
                [
                    embeddings[:, start : start + history]
                    for start in range(local_window_count)
                ],
                dim=0,
            )
            local_actions = torch.cat(
                [
                    output["act_emb"][:, start : start + history]
                    for start in range(local_window_count)
                ],
                dim=0,
            )
            local_target = torch.cat(
                [
                    embeddings[:, start + 1 : start + history + 1]
                    for start in range(local_window_count)
                ],
                dim=0,
            )
            predicted = self.model.predict(local_histories, local_actions)
            prediction_loss = (predicted - local_target).pow(2).mean()
            local_sequences = torch.cat(
                [
                    embeddings[:, start : start + history + 1]
                    for start in range(local_window_count)
                ],
                dim=0,
            )
            sigreg_loss = self.sigreg(local_sequences.transpose(0, 1))

            tail_output = future_goal_td_objective(
                self.value,
                self.target_value,
                embeddings,
                first_current_index=history - 1,
                max_goal_offset=self.max_goal_offset,
                td_horizon=self.td_horizon,
                gamma=self.gamma,
            )
            tail_scale = self._tail_weight_scale()
            weighted_tail_loss = tail_scale * (
                protocol["tail_value"]["loss_weight"] * tail_output.td_loss
                + protocol["tail_value"]["boundary_loss_weight"]
                * tail_output.boundary_loss
            )
            loss = (
                prediction_loss
                + protocol["loss"]["sigreg"]["weight"] * sigreg_loss
                + weighted_tail_loss
            )
            self.log_dict(
                {
                    f"{stage}/loss": loss,
                    f"{stage}/prediction_loss": prediction_loss,
                    f"{stage}/sigreg_loss": sigreg_loss,
                    f"{stage}/tail_value_loss": tail_output.td_loss,
                    f"{stage}/tail_boundary_loss": tail_output.boundary_loss,
                    f"{stage}/tail_value_prediction": tail_output.prediction_mean,
                    f"{stage}/tail_value_target": tail_output.target_mean,
                    f"{stage}/tail_weight_scale": tail_scale,
                    f"{stage}/tail_pairs": float(tail_output.pair_count),
                },
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=stage != "train",
                sync_dist=False,
                batch_size=batch_size,
            )
            return loss

        def training_step(self, batch: dict[str, Any], batch_idx: int):
            return self._forward_loss(batch, "train")

        def validation_step(self, batch: dict[str, Any], batch_idx: int):
            return self._forward_loss(batch, "validation")

        def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
            ema_update(
                self.target_value,
                self.value,
                decay=self.target_ema_decay,
            )

        def configure_optimizers(self):
            cfg = protocol["optimizer"]
            optimizer = torch.optim.AdamW(
                [*self.model.parameters(), *self.value.parameters()],
                lr=cfg["learning_rate"],
                weight_decay=cfg["weight_decay"],
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


def _build_export_callback(
    run_dir: Path, model_config: dict[str, Any], protocol: dict[str, Any]
):
    import lightning as pl
    from omegaconf import OmegaConf

    base_config = OmegaConf.create(model_config)

    class GTExportCallback(pl.Callback):
        def on_train_epoch_end(self, trainer, pl_module) -> None:
            if not trainer.is_global_zero:
                return
            import stable_worldmodel as swm

            swm.wm.save_pretrained(
                pl_module.model,
                run_name=f"epoch_{trainer.current_epoch + 1:02d}",
                config=base_config,
                cache_dir=str(run_dir / "checkpoints" / "exports"),
            )
            value_dir = run_dir / "checkpoints" / "gt_lewm"
            value_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "value_state_dict": pl_module.value.state_dict(),
                    "target_value_state_dict": pl_module.target_value.state_dict(),
                    "value_config": {
                        "objective_version": protocol["tail_value"][
                            "objective_version"
                        ],
                        "embed_dim": protocol["model"]["embed_dim"],
                        "hidden_dim": protocol["tail_value"]["hidden_dim"],
                        "gamma": protocol["tail_value"]["gamma"],
                        "max_goal_offset": protocol["tail_value"][
                            "max_goal_offset"
                        ],
                        "td_horizon": protocol["tail_value"]["td_horizon"],
                        "target_ema_decay": protocol["tail_value"][
                            "target_ema_decay"
                        ],
                        "continuation_policy": protocol["tail_value"][
                            "continuation_policy"
                        ],
                    },
                    "world_model_config": model_config,
                },
                value_dir / f"epoch_{trainer.current_epoch + 1:02d}.pt",
            )

    return GTExportCallback()


def _build_generator_callback(generator: torch.Generator):
    import lightning as pl

    class DataLoaderGeneratorCallback(pl.Callback):
        @property
        def state_key(self) -> str:
            return "tdwm_gt_train_dataloader_generator"

        def state_dict(self) -> dict[str, Any]:
            return {"generator_state": generator.get_state()}

        def load_state_dict(self, state_dict: dict[str, Any]) -> None:
            generator.set_state(state_dict["generator_state"])

    return DataLoaderGeneratorCallback()


def _build_episode_epoch_callback(dataset: EpisodeStreamingBatchDataset):
    import lightning as pl

    class EpisodeStreamingEpochCallback(pl.Callback):
        def on_train_epoch_start(self, trainer, pl_module) -> None:
            dataset.set_epoch(int(trainer.current_epoch))

    return EpisodeStreamingEpochCallback()


def train_gt_lewm(
    *,
    protocol_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    seed: int,
    smoke: bool = False,
    resume: str = "auto",
    max_steps: int | None = None,
    skip_validation: bool = False,
    _protocol_loader=None,
    _module_builder=None,
    _export_callback_builder=None,
    _method: str = "gt_lewm",
) -> dict[str, Any]:
    """Train GT-LeWM without entering the baseline LeWM training entry point."""

    protocol_loader = _protocol_loader or load_gt_training_protocol
    module_builder = _module_builder or _build_training_module
    export_callback_builder = _export_callback_builder or _build_export_callback
    protocol = protocol_loader(protocol_path)
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

    expected_version = protocol["runtime"]["stable_worldmodel_version"]
    package_version = importlib.metadata.version("stable-worldmodel")
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
        raise ValueError("Dataset episode count does not match the GT-LeWM protocol.")
    if (
        int(torch.as_tensor(dataset.lengths).sum())
        != dataset_cfg["expected_transitions"]
    ):
        raise ValueError(
            "Dataset transition count does not match the GT-LeWM protocol."
        )

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
    train_set, validation_set = torch.utils.data.random_split(
        dataset,
        [
            protocol["split"]["train_fraction"],
            protocol["split"]["validation_fraction"],
        ],
        generator=generator,
    )
    split_manifest = save_split(
        run_dir,
        np.asarray(train_set.indices, dtype=np.int64),
        np.asarray(validation_set.indices, dtype=np.int64),
    )

    episode_train_dataset = None
    use_episode_streaming = bool(loader_cfg["episode_streaming"]) and not smoke
    if use_episode_streaming:
        if not isinstance(dataset, StrideAwareLanceDataset):
            raise ValueError(
                "GT-LeWM episode streaming requires the stride-aware Lance dataset."
            )
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
        loader_workers = 0 if smoke else int(loader_cfg["workers"])
        loader_kwargs: dict[str, Any] = {
            "num_workers": loader_workers,
            "pin_memory": loader_cfg["pin_memory"],
        }
        if loader_workers:
            loader_kwargs.update(
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
            **loader_kwargs,
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
    if loader_cfg["validation_locality"]:
        validation_loader = torch.utils.data.DataLoader(
            validation_set,
            batch_sampler=BlockShuffleBatchSampler(
                validation_set.indices,
                batch_size=loader_cfg["batch_size"],
                block_size=loader_cfg["block_size"],
                drop_last=loader_cfg["validation_drop_last"],
                shuffle_batches_within_block=False,
                shuffle_blocks=False,
            ),
            **validation_kwargs,
        )
    else:
        validation_loader = torch.utils.data.DataLoader(
            validation_set,
            batch_size=loader_cfg["batch_size"],
            shuffle=loader_cfg["validation_shuffle"],
            drop_last=loader_cfg["validation_drop_last"],
            **validation_kwargs,
        )

    action_dim = int(dataset.get_dim("action"))
    model_config = build_model_config(protocol, action_dim)
    world_model = hydra.utils.instantiate(model_config)
    parameter_count = sum(parameter.numel() for parameter in world_model.parameters())
    expected_parameters = protocol["model"].get("parameters")
    if expected_parameters and parameter_count != expected_parameters:
        raise ValueError(
            f"Expected {expected_parameters} parameters, found {parameter_count}."
        )
    if protocol["training"]["model_compile"]:
        compile_world_model(
            world_model, mode=protocol["training"]["model_compile_mode"]
        )

    available_epoch_steps = len(train_loader)
    formal_epoch_steps = int(protocol["training"]["optimizer_steps_per_epoch"])
    if formal_epoch_steps > available_epoch_steps:
        raise ValueError(
            "optimizer_steps_per_epoch exceeds the available training batches."
        )
    formal_steps = protocol["training"]["scheduler_epochs"] * formal_epoch_steps
    train_limit = resolve_train_batch_limit(
        smoke=smoke, max_steps=max_steps, train_loader_length=len(train_loader)
    )
    if not smoke and max_steps is None:
        train_limit = formal_epoch_steps
    total_steps = min(2, len(train_loader)) if smoke else formal_steps
    if max_steps is not None:
        total_steps = int(train_limit)
    module = module_builder(
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
    logger = build_metrics_logger(run_dir, protocol["logging"])
    epochs = 1 if smoke or max_steps is not None else protocol["training"]["epochs"]
    callbacks = [
        checkpoint_callback,
        export_callback_builder(run_dir, model_config, protocol),
        _build_generator_callback(generator),
    ]
    if episode_train_dataset is not None:
        callbacks.append(_build_episode_epoch_callback(episode_train_dataset))
    with patch(
        "lightning.pytorch.trainer.connectors.callback_connector."
        "_load_external_callbacks",
        return_value=[],
    ):
        trainer = pl.Trainer(
            default_root_dir=run_dir,
            accelerator="gpu",
            devices=1,
            precision=protocol["training"]["precision"],
            max_epochs=epochs,
            gradient_clip_val=protocol["training"]["gradient_clip_norm"],
            limit_train_batches=train_limit,
            limit_val_batches=0.0 if smoke or skip_validation else 1.0,
            num_sanity_val_steps=0,
            logger=logger,
            callbacks=callbacks,
            log_every_n_steps=1 if smoke else 50,
        )

    last_checkpoint = checkpoint_dir / "last.ckpt"
    if resume == "required" and not last_checkpoint.is_file():
        raise FileNotFoundError(
            f"Required resume checkpoint not found: {last_checkpoint}"
        )
    if resume != "never" and last_checkpoint.is_file():
        manifest_path = run_dir / "training_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(
                "Cannot verify the objective version of this checkpoint."
            )
        with manifest_path.open() as stream:
            previous_manifest = json.load(stream)
        previous_version = (
            previous_manifest.get("protocol", {})
            .get("tail_value", {})
            .get("objective_version")
        )
        if previous_version != protocol["tail_value"]["objective_version"]:
            raise RuntimeError(
                "Refusing to resume a checkpoint from a different GT-LeWM objective."
            )
    checkpoint_path = None
    if resume != "never" and last_checkpoint.is_file():
        checkpoint_path = str(last_checkpoint)
    write_json(
        run_dir / "training_manifest.json",
        {
            "method": _method,
            "protocol": protocol,
            "protocol_path": str(Path(protocol_path).resolve()),
            "seed": seed,
            "dataset": {
                **dataset_source,
                "sequence_samples": len(dataset),
                "split": split_manifest,
            },
            "model": {"config": model_config, "parameters": parameter_count},
            "training": {
                "formal_optimizer_steps": formal_steps,
                "optimizer_steps_per_epoch": formal_epoch_steps,
                "available_batches_per_epoch": available_epoch_steps,
                "configured_optimizer_steps": total_steps,
                "resume_mode": resume,
                "resumed_from": checkpoint_path,
                "episode_streaming": use_episode_streaming,
                "validation_batches": len(validation_loader),
                "validation_skipped": smoke or skip_validation,
            },
            "runtime": {
                "stable_worldmodel": package_version,
                "torch": torch.__version__,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "tdwm_git_revision": _git_revision(),
                "cuda_device": torch.cuda.get_device_name(0),
                "compatibility_adapter": compatibility,
            },
        },
    )
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
    write_json(run_dir / "training_result.json", result)
    return result
