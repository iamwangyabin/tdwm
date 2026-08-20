"""Standalone LS-LeWM training without modifying the LeWM baseline."""

from __future__ import annotations

import hashlib
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
from tdwm.methods.local_successor import (
    LocalSuccessorHeads,
    ema_update,
    future_goal_successor_objective,
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


def load_ls_training_protocol(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as stream:
        protocol = yaml.safe_load(stream)
    validate_ls_training_protocol(protocol)
    return protocol


def validate_ls_training_protocol(protocol: dict[str, Any]) -> None:
    """Validate the independent LS-LeWM Cube training protocol."""

    if protocol.get("schema_version") != 1:
        raise ValueError("LS-LeWM requires schema_version 1.")
    if protocol.get("method") != "ls_lewm" or protocol.get("environment") != "cube":
        raise ValueError("This trainer only accepts LS-LeWM on Cube.")
    if protocol.get("runtime", {}).get("stable_worldmodel_version") != "0.1.1":
        raise ValueError("LS-LeWM is locked to stable-worldmodel 0.1.1.")

    sequence = protocol.get("sequence", {})
    successor = protocol.get("successor", {})
    history = int(sequence.get("history_frames", 0))
    goal_offset = int(successor.get("max_goal_offset", 0))
    if history <= 0 or sequence.get("prediction_frames") != 1:
        raise ValueError("LS-LeWM requires positive history and one-step prediction.")
    if goal_offset <= 0 or sequence.get("num_steps") != history + goal_offset:
        raise ValueError(
            "LS-LeWM num_steps must equal history_frames + max_goal_offset."
        )
    if successor.get("objective_version") != 1:
        raise ValueError("LS-LeWM requires successor objective_version 1.")
    if successor.get("feature_basis") != "augmented_latent_squared_distance":
        raise ValueError("LS-LeWM requires the exact squared-distance feature basis.")
    if successor.get("continuation_policy") != "hindsight_gcbc":
        raise ValueError("LS-LeWM requires an executable hindsight GCBC policy.")
    if successor.get("goal_source") != "all_future_states_in_clip":
        raise ValueError("LS-LeWM requires all future states as hindsight goals.")
    if successor.get("goal_offset_weighting") != "uniform_offsets":
        raise ValueError("LS-LeWM requires uniform weighting over goal offsets.")
    if successor.get("terminal_condition") != "next_state_is_hindsight_goal":
        raise ValueError("LS-LeWM requires the locked hindsight terminal condition.")
    if successor.get("td_steps") != 1:
        raise ValueError("LS-LeWM uses one-step off-policy TD targets.")
    if not 0.0 <= successor.get("gamma", -1.0) < 1.0:
        raise ValueError("successor.gamma must lie in [0, 1).")
    for key in (
        "loss_weight",
        "boundary_loss_weight",
        "policy_loss_weight",
        "imagined_context_weight",
    ):
        if successor.get(key, -1.0) < 0.0:
            raise ValueError(f"successor.{key} cannot be negative.")
    if successor.get("hidden_dim", 0) <= 0:
        raise ValueError("successor.hidden_dim must be positive.")
    if successor.get("policy_learning_rate", 0.0) <= 0.0:
        raise ValueError("successor.policy_learning_rate must be positive.")
    if not 0.0 <= successor.get("loss_warmup_fraction", -1.0) < 1.0:
        raise ValueError("successor.loss_warmup_fraction must lie in [0, 1).")
    if not 0.0 <= successor.get("target_ema_decay", -1.0) < 1.0:
        raise ValueError("successor.target_ema_decay must lie in [0, 1).")

    split = protocol.get("split", {})
    if split.get("unit") != "sequence_clip":
        raise ValueError("LS-LeWM splits sequence clips.")
    if not 0.0 < split.get("train_fraction", 0.0) < 1.0:
        raise ValueError("train_fraction must lie strictly between zero and one.")
    if not math.isclose(
        split["train_fraction"] + split.get("validation_fraction", 0.0), 1.0
    ):
        raise ValueError("Training and validation fractions must sum to one.")

    training = protocol.get("training", {})
    if training.get("epochs") != training.get("scheduler_epochs"):
        raise ValueError("Scheduler and trainer epochs must remain locked together.")
    if min(
        training.get("epochs", 0), training.get("optimizer_steps_per_epoch", 0)
    ) <= 0:
        raise ValueError("Training epochs and optimizer steps must be positive.")
    if protocol.get("scheduler", {}).get("interval") != "optimizer_step":
        raise ValueError("LS-LeWM scheduler must step per optimizer step.")
    if not protocol.get("seeds"):
        raise ValueError("At least one training seed is required.")

    loader = protocol.get("loader", {})
    if loader.get("batch_size", 0) <= 0 or loader.get("workers", -1) < 0:
        raise ValueError("LS-LeWM loader settings are invalid.")
    if loader.get("prefetch_factor", 0) <= 0:
        raise ValueError("loader.prefetch_factor must be positive.")
    if not isinstance(loader.get("device_image_preprocessing"), bool):
        raise ValueError("loader.device_image_preprocessing must be boolean.")
    if not isinstance(loader.get("episode_streaming"), bool):
        raise ValueError("loader.episode_streaming must be boolean.")
    if not 1 <= loader.get("minimum_unique_episodes_per_batch", 0) <= loader[
        "batch_size"
    ]:
        raise ValueError("minimum_unique_episodes_per_batch is invalid.")

    dataset = protocol.get("dataset", {})
    lance = dataset.get("lance", {})
    if lance.get("stable_worldmodel_version") != "0.1.1":
        raise ValueError("The LS-LeWM Lance conversion must use version 0.1.1.")
    if lance.get("image_codec") != "jpeg" or lance.get("jpeg_quality") != 100:
        raise ValueError("LS-LeWM requires the audited JPEG quality 100 data.")
    if lance.get("source", {}).get("sha256") != dataset.get(
        "optimized_layout", {}
    ).get("sha256"):
        raise ValueError("The Lance source does not match the audited Cube data.")

    local_windows = sequence["num_steps"] - history
    effective_batch = loader["batch_size"] * local_windows
    configured_batch = protocol.get("loss", {}).get("sigreg", {}).get(
        "effective_batch_size"
    )
    if effective_batch != configured_batch:
        raise ValueError("LS-LeWM must preserve the effective SIGReg batch size.")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_training_module(
    world_model: Any,
    protocol: dict[str, Any],
    total_steps: int,
    *,
    action_block_dim: int,
    device_image_preprocessing: bool,
):
    import lightning as pl
    import stable_worldmodel as swm

    class LSLeWMTrainingModule(pl.LightningModule):
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
            successor = protocol["successor"]
            self.heads = LocalSuccessorHeads(
                embed_dim=protocol["model"]["embed_dim"],
                action_dim=action_block_dim,
                history_size=protocol["sequence"]["history_frames"],
                hidden_dim=successor["hidden_dim"],
            )
            self.target_heads = self.heads.make_target()
            self.gamma = float(successor["gamma"])
            self.target_ema_decay = float(successor["target_ema_decay"])
            self.successor_warmup_steps = int(
                float(successor["loss_warmup_fraction"]) * total_steps
            )

        def _successor_weight_scale(self) -> float:
            if self.successor_warmup_steps <= 0:
                return 1.0
            return min(
                1.0, float(self.global_step + 1) / self.successor_warmup_steps
            )

        def _forward_loss(self, batch: dict[str, Any], stage: str):
            batch_size = int(batch["pixels"].shape[0])
            episode_ids = batch.pop("_tdwm_episode_id", None)
            cache_bytes = batch.pop("_tdwm_cache_bytes", None)
            if episode_ids is not None:
                self.log(
                    f"{stage}/unique_episodes_per_batch",
                    torch.unique(episode_ids).numel(),
                    on_step=stage == "train",
                    on_epoch=True,
                    sync_dist=False,
                    batch_size=batch_size,
                )
            if cache_bytes is not None:
                self.log(
                    f"{stage}/compressed_cache_gib",
                    float(cache_bytes) / 1024**3,
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
            actions = batch["action"]
            history = protocol["sequence"]["history_frames"]
            expected_steps = protocol["sequence"]["num_steps"]
            if embeddings.shape[1] != expected_steps:
                raise RuntimeError("LS-LeWM received a clip with unexpected length.")
            if actions.ndim != 3:
                raise RuntimeError("LS-LeWM actions must be flattened action blocks.")

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
            prediction_loss = (predicted - local_target).square().mean()
            local_sequences = torch.cat(
                [
                    embeddings[:, start : start + history + 1]
                    for start in range(local_window_count)
                ],
                dim=0,
            )
            sigreg_loss = self.sigreg(local_sequences.transpose(0, 1))

            successor_output = future_goal_successor_objective(
                self.heads,
                self.target_heads,
                embeddings,
                actions,
                gamma=self.gamma,
            )
            imagined_weight = float(
                protocol["successor"]["imagined_context_weight"]
            )
            imagined_td = embeddings.new_zeros(())
            imagined_boundary = embeddings.new_zeros(())
            if imagined_weight > 0.0:
                one_step = predicted.reshape(
                    local_window_count,
                    batch_size,
                    history,
                    embeddings.shape[-1],
                )[:, :, -1].transpose(0, 1)
                imagined_context = embeddings.detach().clone()
                imagined_context[:, history:] = one_step.detach()
                imagined_output = future_goal_successor_objective(
                    self.heads,
                    self.target_heads,
                    embeddings,
                    actions,
                    gamma=self.gamma,
                    context_latents=imagined_context,
                    train_policy=False,
                )
                imagined_td = imagined_output.td_loss
                imagined_boundary = imagined_output.boundary_loss

            successor_scale = self._successor_weight_scale()
            successor_cfg = protocol["successor"]
            td_loss = successor_output.td_loss + imagined_weight * imagined_td
            boundary_loss = (
                successor_output.boundary_loss
                + imagined_weight * imagined_boundary
            )
            weighted_successor = successor_scale * (
                successor_cfg["loss_weight"] * td_loss
                + successor_cfg["boundary_loss_weight"] * boundary_loss
            )
            weighted_policy = (
                successor_cfg["policy_loss_weight"]
                * successor_output.policy_loss
            )
            loss = (
                prediction_loss
                + protocol["loss"]["sigreg"]["weight"] * sigreg_loss
                + weighted_successor
                + weighted_policy
            )
            self.log_dict(
                {
                    f"{stage}/loss": loss,
                    f"{stage}/prediction_loss": prediction_loss,
                    f"{stage}/sigreg_loss": sigreg_loss,
                    f"{stage}/successor_td_loss": successor_output.td_loss,
                    f"{stage}/successor_boundary_loss": successor_output.boundary_loss,
                    f"{stage}/imagined_successor_td_loss": imagined_td,
                    f"{stage}/policy_bc_loss": successor_output.policy_loss,
                    f"{stage}/successor_scalar_prediction": (
                        successor_output.scalar_prediction_mean
                    ),
                    f"{stage}/successor_scalar_target": (
                        successor_output.scalar_target_mean
                    ),
                    f"{stage}/successor_weight_scale": successor_scale,
                    f"{stage}/successor_pairs": float(successor_output.pair_count),
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
                self.target_heads,
                self.heads,
                decay=self.target_ema_decay,
            )

        def configure_optimizers(self):
            optimizer_cfg = protocol["optimizer"]
            policy_lr = protocol["successor"].get(
                "policy_learning_rate", optimizer_cfg["learning_rate"]
            )
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": [
                            *self.model.parameters(),
                            *self.heads.successor.parameters(),
                        ],
                        "lr": optimizer_cfg["learning_rate"],
                    },
                    {
                        "params": self.heads.policy.parameters(),
                        "lr": policy_lr,
                    },
                ],
                weight_decay=optimizer_cfg["weight_decay"],
            )
            warmup_steps = max(
                1, int(protocol["scheduler"]["warmup_fraction"] * total_steps)
            )

            def learning_rate_scale(step: int) -> float:
                if step < warmup_steps:
                    return float(step + 1) / float(warmup_steps)
                progress = (step - warmup_steps) / max(
                    1, total_steps - warmup_steps
                )
                return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=learning_rate_scale
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

    return LSLeWMTrainingModule()


def _build_export_callback(
    run_dir: Path,
    model_config: dict[str, Any],
    protocol: dict[str, Any],
    action_block_dim: int,
):
    import lightning as pl
    from omegaconf import OmegaConf

    base_config = OmegaConf.create(model_config)

    class LSExportCallback(pl.Callback):
        def on_train_epoch_end(self, trainer, pl_module) -> None:
            if not trainer.is_global_zero:
                return
            if (
                trainer.current_epoch + 1
            ) % protocol["training"]["checkpoint_every_epochs"]:
                return
            import stable_worldmodel as swm

            epoch = trainer.current_epoch + 1
            swm.wm.save_pretrained(
                pl_module.model,
                run_name=f"epoch_{epoch:02d}",
                config=base_config,
                cache_dir=str(run_dir / "checkpoints" / "exports"),
            )
            base_run_name = f"epoch_{epoch:02d}"
            base_dir = (
                run_dir
                / "checkpoints"
                / "exports"
                / "checkpoints"
                / base_run_name
            )
            base_weights = sorted(base_dir.glob("*.pt"))
            if len(base_weights) != 1:
                raise RuntimeError(
                    "Stable World Model export did not contain exactly one weight file."
                )
            heads_dir = run_dir / "checkpoints" / "ls_lewm"
            heads_dir.mkdir(parents=True, exist_ok=True)
            successor = protocol["successor"]
            torch.save(
                {
                    "heads_state_dict": pl_module.heads.state_dict(),
                    "target_heads_state_dict": pl_module.target_heads.state_dict(),
                    "heads_config": {
                        "objective_version": successor["objective_version"],
                        "embed_dim": protocol["model"]["embed_dim"],
                        "action_dim": action_block_dim,
                        "history_size": protocol["sequence"]["history_frames"],
                        "hidden_dim": successor["hidden_dim"],
                        "gamma": successor["gamma"],
                        "feature_basis": successor["feature_basis"],
                        "continuation_policy": successor["continuation_policy"],
                        "goal_offset_weighting": successor[
                            "goal_offset_weighting"
                        ],
                        "terminal_condition": successor["terminal_condition"],
                        "target_ema_decay": successor["target_ema_decay"],
                        "base_export_run_name": base_run_name,
                        "base_checkpoint_sha256": _file_sha256(base_weights[0]),
                    },
                    "world_model_config": model_config,
                },
                heads_dir / f"epoch_{epoch:02d}.pt",
            )

    return LSExportCallback()


def _build_generator_callback(generator: torch.Generator):
    import lightning as pl

    class DataLoaderGeneratorCallback(pl.Callback):
        @property
        def state_key(self) -> str:
            return "tdwm_ls_train_dataloader_generator"

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


def train_ls_lewm(
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
    """Train LS-LeWM as a method separate from the LeWM baseline."""

    protocol = load_ls_training_protocol(protocol_path)
    if seed not in protocol["seeds"]:
        raise ValueError(f"Seed {seed} is not in {protocol['seeds']}.")
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
    train_set, validation_set = torch.utils.data.random_split(
        dataset,
        [protocol["split"]["train_fraction"], protocol["split"]["validation_fraction"]],
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
        loader_kwargs: dict[str, Any] = {
            "num_workers": workers,
            "pin_memory": loader_cfg["pin_memory"],
        }
        if workers:
            loader_kwargs.update(
                {"persistent_workers": True, "prefetch_factor": loader_cfg["prefetch_factor"]}
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
            {"persistent_workers": True, "prefetch_factor": loader_cfg["prefetch_factor"]}
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
    action_block_dim = sequence["frame_skip"] * action_dim
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
    formal_steps = protocol["training"]["scheduler_epochs"] * formal_epoch_steps
    train_limit = resolve_train_batch_limit(
        smoke=smoke, max_steps=max_steps, train_loader_length=len(train_loader)
    )
    if not smoke and max_steps is None:
        train_limit = formal_epoch_steps
    total_steps = min(2, len(train_loader)) if smoke else formal_steps
    if max_steps is not None:
        total_steps = int(train_limit)
    module = _build_training_module(
        world_model,
        protocol,
        total_steps,
        action_block_dim=action_block_dim,
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
        _build_export_callback(run_dir, model_config, protocol, action_block_dim),
        _build_generator_callback(generator),
    ]
    if episode_train_dataset is not None:
        callbacks.append(_build_episode_epoch_callback(episode_train_dataset))
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    precision = protocol["training"]["precision"] if accelerator == "gpu" else "32-true"
    epochs = 1 if smoke or max_steps is not None else protocol["training"]["epochs"]
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
            limit_val_batches=0.0 if smoke or skip_validation else 1.0,
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
        previous_version = (
            previous.get("protocol", {}).get("successor", {}).get("objective_version")
        )
        if previous_version != protocol["successor"]["objective_version"]:
            raise RuntimeError("Refusing to resume a different successor objective.")
        checkpoint_path = str(last_checkpoint)

    write_json(
        run_dir / "training_manifest.json",
        {
            "method": "ls_lewm",
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
                "lewm_parameters": parameter_count,
                "heads_parameters": sum(
                    parameter.numel() for parameter in module.heads.parameters()
                ),
                "action_block_dim": action_block_dim,
            },
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
