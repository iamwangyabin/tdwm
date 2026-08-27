"""Protocol-aligned end-to-end MC GoalTail training on OGBench Cube."""

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
from tdwm.methods.goal_tail_value import (
    BoundaryAnchoredGoalTailValue,
    monte_carlo_goal_tail_targets,
)
from tdwm.training.cube_data import validate_cube_training_dataset
from tdwm.training.gt_lewm_support import (
    LeWMTransform,
    build_metrics_logger,
    build_model_config,
    compile_world_model,
    fit_column_stats,
    preprocess_image_batch,
    save_split,
    write_json,
)
from tdwm.training.joint_td_gt_lewm import build_history_at, rollout_from_latents
from tdwm.training.lance_batch import (
    EpisodeStreamingBatchDataset,
    PairedEpisodeStreamingBatchDataset,
    StrideAwareLanceDataset,
)
from tdwm.training.lewm import _git_revision


METHOD = "aligned_e2e_mc_gt_lewm"


def load_aligned_e2e_mc_gt_protocol(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as stream:
        protocol = yaml.safe_load(stream)
    validate_aligned_e2e_mc_gt_protocol(protocol)
    return protocol


def validate_aligned_e2e_mc_gt_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema_version") != 1 or protocol.get("method") != METHOD:
        raise ValueError("This trainer only accepts aligned E2E MC-GT-LeWM schema 1.")
    if protocol.get("environment") != "cube" or protocol.get("stage") != "full_training":
        raise ValueError("Aligned E2E MC-GT-LeWM is locked to full Cube training.")
    if protocol.get("initialization") != "random_from_scratch":
        raise ValueError("Aligned E2E MC-GT-LeWM must start from random parameters.")
    if protocol.get("runtime", {}).get("stable_worldmodel_version") != "0.1.1":
        raise ValueError("Aligned E2E MC-GT-LeWM requires stable-worldmodel 0.1.1.")

    sequence = protocol.get("sequence", {})
    history = sequence.get("history_frames")
    rollout = sequence.get("model_rollout_horizon")
    max_offset = sequence.get("max_goal_offset")
    if (history, rollout, max_offset) != (3, 5, 16):
        raise ValueError("The aligned method locks history 3, rollout 5, goals 1..16.")
    if sequence.get("world_num_steps") != history + 1:
        raise ValueError("The world view must exactly match LeWM's 3+1 clip.")
    if sequence.get("tail_num_steps") != history + rollout + max_offset:
        raise ValueError("The tail view must cover history, rollout, and MC future.")
    if sequence.get("frame_skip") != 5 or sequence.get("prediction_frames") != 1:
        raise ValueError("Cube training requires frame skip 5 and one-step prediction.")

    objective = protocol.get("joint_objective", {})
    expected_objective = {
        "prediction": "original_lewm_mse_on_independent_short_clips",
        "regularization": "sigreg_on_independent_short_clips",
        "tail": "supervised_mc_on_predicted_terminal_history",
        "tail_target_encoder": "ema_world_model",
        "backpropagate_tail_through_rollout": True,
    }
    for key, expected in expected_objective.items():
        if objective.get(key) != expected:
            raise ValueError(f"joint_objective.{key} must be {expected!r}.")
    if objective.get("prediction_weight") != 1.0:
        raise ValueError("The original LeWM prediction weight remains one.")
    if objective.get("tail_weight", -1.0) <= 0.0:
        raise ValueError("The MC tail weight must be positive.")

    tail = protocol.get("tail_value", {})
    if tail.get("objective_version") != 2 or tail.get("objective") != "supervised_mc":
        raise ValueError("The aligned value requires objective-version-two MC targets.")
    if tail.get("architecture") != "squared_shared_potential_anchor":
        raise ValueError("The value must enforce its zero boundary structurally.")
    if tail.get("boundary_condition") != "exact_current_goal_zero":
        raise ValueError("The exact current-goal boundary must remain enabled.")
    if tail.get("goal_sampling") != "all_future_offsets_uniform":
        raise ValueError("Every future offset must receive equal MC supervision.")
    if tail.get("continuation_policy") != "offline_dataset_behavior":
        raise ValueError("The MC target is defined under the offline behavior.")
    if not 0.0 <= tail.get("gamma", -1.0) < 1.0:
        raise ValueError("tail_value.gamma must lie in [0, 1).")
    if not 0.0 <= tail.get("target_world_ema_decay", -1.0) < 1.0:
        raise ValueError("target_world_ema_decay must lie in [0, 1).")
    if not 0.0 < tail.get("world_gradient_scale", 0.0) <= 1.0:
        raise ValueError("world_gradient_scale must lie in (0, 1].")
    if not 0.0 <= tail.get("loss_warmup_fraction", -1.0) < 1.0:
        raise ValueError("loss_warmup_fraction must lie in [0, 1).")

    loader = protocol.get("loader", {})
    if loader.get("world_batch_size") != 128:
        raise ValueError("The LeWM view must retain 128 independent clips.")
    if loader.get("world_minimum_unique_episodes_per_batch") != 128:
        raise ValueError("Every LeWM batch item must come from a distinct episode.")
    if loader.get("tail_batch_size", 0) <= 0:
        raise ValueError("The long-sequence tail batch must be positive.")
    if loader.get("tail_minimum_unique_episodes_per_batch") != loader.get(
        "tail_batch_size"
    ):
        raise ValueError("Every tail batch item must come from a distinct episode.")
    if loader.get("episode_pool_size", 0) < loader["world_batch_size"]:
        raise ValueError("episode_pool_size must cover the world batch.")
    if min(
        loader.get("world_episode_cache_bytes", 0),
        loader.get("tail_episode_cache_bytes", 0),
        loader.get("episode_read_size", 0),
        loader.get("episode_prefetch_blocks", 0),
    ) <= 0:
        raise ValueError("Episode-streaming cache settings must be positive.")
    if loader.get("episode_streaming") is not True:
        raise ValueError("The aligned dual-view protocol requires episode streaming.")
    if loader.get("device_image_preprocessing") is not True:
        raise ValueError("The aligned raw-image run preprocesses uint8 images on GPU.")
    if protocol.get("loss", {}).get("sigreg", {}).get("effective_batch_size") != 128:
        raise ValueError("SIGReg must receive the true independent batch of 128.")

    training = protocol.get("training", {})
    if training.get("epochs") != training.get("scheduler_epochs"):
        raise ValueError("Scheduler and trainer epochs must match.")
    if training.get("epochs", 0) <= 0 or training.get("optimizer_steps_per_epoch", 0) <= 0:
        raise ValueError("The formal optimizer budget must be positive.")
    if training.get("checkpoint_selection") != "locked_final_epoch":
        raise ValueError("The aligned first run selects the locked final epoch.")
    if protocol.get("scheduler", {}).get("interval") != "optimizer_step":
        raise ValueError("The scheduler must step once per optimizer update.")
    if not protocol.get("seeds"):
        raise ValueError("At least one formal seed is required.")

    dataset = protocol.get("dataset", {})
    lance = dataset.get("lance", {})
    if lance.get("image_codec") != "jpeg" or lance.get("jpeg_quality") != 100:
        raise ValueError("The audited Cube input must be JPEG-100 Lance.")
    if lance.get("stable_worldmodel_version") != "0.1.1":
        raise ValueError("The Lance dataset must use stable-worldmodel 0.1.1.")
    if lance.get("source", {}).get("sha256") != dataset.get("optimized_layout", {}).get(
        "sha256"
    ):
        raise ValueError("The Lance source does not match the audited Cube layout.")


def _scale_gradient(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    return tensor.detach() + scale * (tensor - tensor.detach())


@torch.no_grad()
def ema_update_module(target: torch.nn.Module, source: torch.nn.Module, *, decay: float) -> None:
    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must lie in [0, 1).")
    for target_parameter, source_parameter in zip(
        target.parameters(), source.parameters(), strict=True
    ):
        target_parameter.mul_(decay).add_(source_parameter, alpha=1.0 - decay)
    for target_buffer, source_buffer in zip(
        target.buffers(), source.buffers(), strict=True
    ):
        if target_buffer.is_floating_point():
            target_buffer.mul_(decay).add_(source_buffer, alpha=1.0 - decay)
        else:
            target_buffer.copy_(source_buffer)


def _build_training_module(
    world_model: Any,
    protocol: dict[str, Any],
    total_steps: int,
):
    import lightning as pl
    import stable_worldmodel as swm

    class AlignedE2EMCGTLeWMModule(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.automatic_optimization = False
            self.model = world_model
            self.target_model = copy.deepcopy(world_model).requires_grad_(False)
            self.target_model.eval()
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
            self.value = BoundaryAnchoredGoalTailValue(
                history_dim=int(tail["history_dim"]),
                goal_dim=int(protocol["model"]["embed_dim"]),
                history_size=int(protocol["sequence"]["history_frames"]),
                hidden_dim=int(tail["hidden_dim"]),
            )
            self.history_size = int(protocol["sequence"]["history_frames"])
            self.rollout_horizon = int(protocol["sequence"]["model_rollout_horizon"])
            self.max_goal_offset = int(protocol["sequence"]["max_goal_offset"])
            self.gamma = float(tail["gamma"])
            self.target_world_ema_decay = float(tail["target_world_ema_decay"])
            self.world_gradient_scale = float(tail["world_gradient_scale"])
            self.gradient_clip_norm = float(protocol["training"]["gradient_clip_norm"])
            self.tail_warmup_steps = int(
                float(tail["loss_warmup_fraction"]) * total_steps
            )

        def train(self, mode: bool = True):
            super().train(mode)
            self.target_model.eval()
            return self

        def _tail_scale(self) -> float:
            if self.tail_warmup_steps <= 0:
                return 1.0
            return min(1.0, float(self.global_step + 1) / self.tail_warmup_steps)

        def _preprocess(self, pixels: torch.Tensor) -> torch.Tensor:
            return preprocess_image_batch(
                pixels,
                mean=self.image_mean,
                std=self.image_std,
                size=protocol["image_preprocessing"]["size"],
            )

        def _world_loss(
            self, batch: dict[str, Any]
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            episode_ids = batch.pop("_tdwm_episode_id", None)
            batch.pop("_tdwm_cache_bytes", None)
            pixels = self._preprocess(batch["pixels"])
            actions = torch.nan_to_num(batch["action"], 0.0)
            output = self.model.encode({"pixels": pixels, "action": actions})
            history = self.history_size
            predicted = self.model.predict(
                output["emb"][:, :history], output["act_emb"][:, :history]
            )
            target = output["emb"][:, 1:]
            prediction_loss = (predicted - target).pow(2).mean()
            sigreg_loss = self.sigreg(output["emb"].transpose(0, 1))
            loss = prediction_loss + protocol["loss"]["sigreg"]["weight"] * sigreg_loss
            unique = prediction_loss.new_tensor(
                float(torch.unique(episode_ids).numel())
                if episode_ids is not None
                else float(pixels.shape[0])
            )
            return loss, {
                "prediction_loss": prediction_loss.detach(),
                "sigreg_loss": sigreg_loss.detach(),
                "world_unique_episodes": unique,
            }

        def _tail_loss(
            self, batch: dict[str, Any]
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            episode_ids = batch.pop("_tdwm_episode_id", None)
            batch.pop("_tdwm_cache_bytes", None)
            actions = torch.nan_to_num(batch["action"], 0.0)
            terminal_index = self.history_size + self.rollout_horizon - 1

            initial_pixels = self._preprocess(batch["pixels"][:, : self.history_size])
            initial = self.model.encode(
                {
                    "pixels": initial_pixels,
                    "action": actions[:, : self.history_size],
                }
            )["emb"]
            pred_proj_was_training = self.model.pred_proj.training
            self.model.pred_proj.eval()
            try:
                predicted_rollout = rollout_from_latents(
                    self.model,
                    initial,
                    actions[:, :terminal_index],
                    history_size=self.history_size,
                )
            finally:
                self.model.pred_proj.train(pred_proj_was_training)
            predicted_history = build_history_at(
                predicted_rollout,
                actions,
                current_index=terminal_index,
                history_size=self.history_size,
            )

            target_end = terminal_index + self.max_goal_offset + 1
            target_pixels = self._preprocess(
                batch["pixels"][:, terminal_index:target_end]
            )
            with torch.no_grad():
                target_latents = self.target_model.encode(
                    {
                        "pixels": target_pixels,
                        "action": actions[:, terminal_index:target_end],
                    }
                )["emb"]
                targets = monte_carlo_goal_tail_targets(
                    target_latents,
                    current_index=0,
                    max_goal_offset=self.max_goal_offset,
                    gamma=self.gamma,
                )
                goals = target_latents[:, 1 : self.max_goal_offset + 1]

            history_for_value = _scale_gradient(
                predicted_history, self.world_gradient_scale
            )
            expanded_history = history_for_value.unsqueeze(1).expand(
                -1, self.max_goal_offset, -1
            )
            predictions = self.value(expanded_history, goals)
            squared_error = (predictions - targets).pow(2)
            loss = squared_error.mean()
            current = self.value.current_latent(predicted_history)
            boundary = self.value(predicted_history, current)
            terminal_mse = (
                predicted_rollout[:, terminal_index] - target_latents[:, 0]
            ).pow(2).mean()
            unique = loss.new_tensor(
                float(torch.unique(episode_ids).numel())
                if episode_ids is not None
                else float(actions.shape[0])
            )
            return loss, {
                "tail_value_loss": loss.detach(),
                "tail_value_prediction": predictions.detach().mean(),
                "tail_value_target": targets.detach().mean(),
                "tail_boundary_max": boundary.detach().abs().max(),
                "terminal_rollout_mse": terminal_mse.detach(),
                "tail_mse_offset_1": squared_error[:, 0].detach().mean(),
                "tail_mse_offset_5": squared_error[:, 4].detach().mean(),
                "tail_mse_offset_16": squared_error[:, 15].detach().mean(),
                "tail_unique_episodes": unique,
            }

        def _log_metrics(
            self,
            stage: str,
            total: torch.Tensor,
            world_metrics: dict[str, torch.Tensor],
            tail_metrics: dict[str, torch.Tensor],
            tail_scale: float,
        ) -> None:
            metrics = {f"{stage}/loss": total.detach()}
            metrics.update(
                {f"{stage}/{key}": value for key, value in world_metrics.items()}
            )
            metrics.update(
                {f"{stage}/{key}": value for key, value in tail_metrics.items()}
            )
            metrics[f"{stage}/tail_weight_scale"] = total.new_tensor(tail_scale)
            self.log_dict(
                metrics,
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=stage == "validation",
                sync_dist=False,
                batch_size=protocol["loader"]["world_batch_size"],
            )

        def training_step(self, batch: dict[str, dict[str, Any]], batch_idx: int):
            del batch_idx
            optimizer = self.optimizers()
            optimizer.zero_grad()

            world_loss, world_metrics = self._world_loss(batch["world"])
            self.manual_backward(world_loss)
            tail_loss, tail_metrics = self._tail_loss(batch["tail"])
            tail_scale = self._tail_scale()
            weighted_tail = (
                tail_scale * protocol["joint_objective"]["tail_weight"] * tail_loss
            )
            self.manual_backward(weighted_tail)
            self.clip_gradients(
                optimizer,
                gradient_clip_val=self.gradient_clip_norm,
                gradient_clip_algorithm="norm",
            )
            optimizer.step()
            scheduler = self.lr_schedulers()
            scheduler.step()
            ema_update_module(
                self.target_model,
                self.model,
                decay=self.target_world_ema_decay,
            )
            total = world_loss.detach() + weighted_tail.detach()
            self._log_metrics(
                "train", total, world_metrics, tail_metrics, tail_scale
            )
            return total

        def validation_step(
            self, batch: dict[str, dict[str, Any]], batch_idx: int
        ) -> torch.Tensor:
            del batch_idx
            world_loss, world_metrics = self._world_loss(batch["world"])
            tail_loss, tail_metrics = self._tail_loss(batch["tail"])
            tail_scale = self._tail_scale()
            total = (
                world_loss
                + tail_scale
                * protocol["joint_objective"]["tail_weight"]
                * tail_loss
            )
            self._log_metrics(
                "validation", total, world_metrics, tail_metrics, tail_scale
            )
            return total

        def configure_optimizers(self):
            optimizer_cfg = protocol["optimizer"]
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": list(self.model.parameters()),
                        "lr": optimizer_cfg["world_model_learning_rate"],
                    },
                    {
                        "params": list(self.value.parameters()),
                        "lr": optimizer_cfg["value_learning_rate"],
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
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=learning_rate_scale
            )
            return [optimizer], [scheduler]

    return AlignedE2EMCGTLeWMModule()


def _value_config(protocol: dict[str, Any]) -> dict[str, Any]:
    tail = protocol["tail_value"]
    return {
        "objective_version": tail["objective_version"],
        "objective": tail["objective"],
        "architecture": tail["architecture"],
        "boundary_condition": tail["boundary_condition"],
        "input_distribution": "lewm_predicted_terminal_history",
        "continuation_policy": tail["continuation_policy"],
        "history_size": protocol["sequence"]["history_frames"],
        "history_dim": tail["history_dim"],
        "goal_dim": protocol["model"]["embed_dim"],
        "action_block_dim": tail["action_block_dim"],
        "hidden_dim": tail["hidden_dim"],
        "gamma": tail["gamma"],
        "max_goal_offset": protocol["sequence"]["max_goal_offset"],
        "model_rollout_horizon": protocol["sequence"]["model_rollout_horizon"],
        "target_world_ema_decay": tail["target_world_ema_decay"],
        "world_gradient_scale": tail["world_gradient_scale"],
    }


def _build_export_callback(
    run_dir: Path, model_config: dict[str, Any], protocol: dict[str, Any]
):
    import lightning as pl
    import stable_worldmodel as swm
    from omegaconf import OmegaConf

    export_config = OmegaConf.create(model_config)

    class AlignedE2EExportCallback(pl.Callback):
        def on_train_epoch_end(self, trainer, pl_module) -> None:
            if not trainer.is_global_zero:
                return
            epoch = trainer.current_epoch + 1
            swm.wm.save_pretrained(
                pl_module.model,
                run_name=f"epoch_{epoch:02d}",
                config=export_config,
                cache_dir=str(run_dir / "checkpoints" / "exports"),
            )
            deployment_dir = run_dir / "checkpoints" / METHOD
            deployment_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "method": METHOD,
                    "objective_version": 2,
                    "deployment_checkpoint_version": 1,
                    "epoch": epoch,
                    "global_step": int(trainer.global_step),
                    "world_model_state_dict": pl_module.model.state_dict(),
                    "value_state_dict": pl_module.value.state_dict(),
                    "world_model_config": model_config,
                    "value_config": _value_config(protocol),
                },
                deployment_dir / f"epoch_{epoch:02d}.pt",
            )

    return AlignedE2EExportCallback()


def _build_stream(
    dataset: StrideAwareLanceDataset,
    indices: list[int] | np.ndarray,
    *,
    batch_size: int,
    minimum_unique_episodes: int,
    cache_bytes: int,
    protocol: dict[str, Any],
    seed: int,
) -> EpisodeStreamingBatchDataset:
    loader = protocol["loader"]
    return EpisodeStreamingBatchDataset(
        dataset,
        indices,
        batch_size=batch_size,
        active_episodes=loader["episode_pool_size"],
        read_episodes=loader["episode_read_size"],
        cache_bytes=cache_bytes,
        prefetch_blocks=loader["episode_prefetch_blocks"],
        seed=seed,
        drop_last=True,
        min_unique_episodes=minimum_unique_episodes,
    )


def _build_epoch_callback(dataset: PairedEpisodeStreamingBatchDataset):
    import lightning as pl

    class PairedEpochCallback(pl.Callback):
        def on_train_epoch_start(self, trainer, pl_module) -> None:
            del pl_module
            dataset.set_epoch(int(trainer.current_epoch))

    return PairedEpochCallback()


def train_aligned_e2e_mc_gt_lewm(
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
    """Train aligned LeWM and boundary-anchored MC GoalTail from raw images."""

    protocol = load_aligned_e2e_mc_gt_protocol(protocol_path)
    if seed not in protocol["seeds"]:
        raise ValueError(f"Seed {seed} is not in the locked seeds {protocol['seeds']}.")
    if resume not in {"auto", "never", "required"}:
        raise ValueError("resume must be one of: auto, never, required.")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive when provided.")

    dataset_path = Path(dataset_path).expanduser().resolve()
    dataset_source = validate_cube_training_dataset(dataset_path, protocol["dataset"])
    if dataset_source["format"] != "lance":
        raise ValueError("Aligned dual-view training requires the audited Lance data.")
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

    dataset_cfg = protocol["dataset"]
    sequence = protocol["sequence"]

    def load_view(num_steps: int):
        return swm.data.load_dataset(
            str(dataset_path),
            format="lance",
            transform=None,
            num_steps=num_steps,
            frameskip=sequence["frame_skip"],
            keys_to_load=list(dataset_cfg["keys_to_load"]),
            keys_to_cache=list(dataset_cfg["keys_to_cache"]),
            keys_to_merge=dict(dataset_cfg["keys_to_merge"]),
        )

    world_dataset = load_view(sequence["world_num_steps"])
    tail_dataset = load_view(sequence["tail_num_steps"])
    for view in (world_dataset, tail_dataset):
        if len(view.lengths) != dataset_cfg["expected_episodes"]:
            raise ValueError("Dataset episode count differs from the protocol.")
        if int(torch.as_tensor(view.lengths).sum()) != dataset_cfg["expected_transitions"]:
            raise ValueError("Dataset transition count differs from the protocol.")

    statistics = fit_column_stats(
        world_dataset,
        list(protocol["normalization"]["columns"]),
        output_dir / "column_normalization.json",
    )
    transform = LeWMTransform(
        image=protocol["image_preprocessing"],
        columns=statistics,
        preprocess_images=False,
    )
    world_dataset.transform = transform
    tail_dataset.transform = transform
    world_dataset = StrideAwareLanceDataset(world_dataset)
    tail_dataset = StrideAwareLanceDataset(tail_dataset)

    split = protocol["split"]
    world_generator = torch.Generator().manual_seed(seed)
    tail_generator = torch.Generator().manual_seed(seed)
    world_train, world_validation = torch.utils.data.random_split(
        world_dataset,
        [split["train_fraction"], split["validation_fraction"]],
        generator=world_generator,
    )
    tail_train, tail_validation = torch.utils.data.random_split(
        tail_dataset,
        [split["train_fraction"], split["validation_fraction"]],
        generator=tail_generator,
    )
    world_split_dir = run_dir / "splits" / "world"
    tail_split_dir = run_dir / "splits" / "tail"
    world_split_dir.mkdir(parents=True, exist_ok=True)
    tail_split_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = {
        "world": save_split(
            world_split_dir,
            np.asarray(world_train.indices, dtype=np.int64),
            np.asarray(world_validation.indices, dtype=np.int64),
        ),
        "tail": save_split(
            tail_split_dir,
            np.asarray(tail_train.indices, dtype=np.int64),
            np.asarray(tail_validation.indices, dtype=np.int64),
        ),
    }

    loader = protocol["loader"]
    world_stream = _build_stream(
        world_dataset,
        world_train.indices,
        batch_size=loader["world_batch_size"],
        minimum_unique_episodes=loader["world_minimum_unique_episodes_per_batch"],
        cache_bytes=loader["world_episode_cache_bytes"],
        protocol=protocol,
        seed=seed,
    )
    tail_stream = _build_stream(
        tail_dataset,
        tail_train.indices,
        batch_size=loader["tail_batch_size"],
        minimum_unique_episodes=loader["tail_minimum_unique_episodes_per_batch"],
        cache_bytes=loader["tail_episode_cache_bytes"],
        protocol=protocol,
        seed=seed + 1_000_003,
    )
    paired_train = PairedEpisodeStreamingBatchDataset(world_stream, tail_stream)
    train_loader = torch.utils.data.DataLoader(
        paired_train,
        batch_size=None,
        num_workers=0,
        pin_memory=loader["pin_memory"],
    )

    validation_seed = seed + 2_000_003
    validation_world_stream = _build_stream(
        world_dataset,
        world_validation.indices,
        batch_size=loader["world_batch_size"],
        minimum_unique_episodes=loader["world_minimum_unique_episodes_per_batch"],
        cache_bytes=loader["world_episode_cache_bytes"],
        protocol=protocol,
        seed=validation_seed,
    )
    validation_tail_stream = _build_stream(
        tail_dataset,
        tail_validation.indices,
        batch_size=loader["tail_batch_size"],
        minimum_unique_episodes=loader["tail_minimum_unique_episodes_per_batch"],
        cache_bytes=loader["tail_episode_cache_bytes"],
        protocol=protocol,
        seed=validation_seed + 1,
    )
    paired_validation = PairedEpisodeStreamingBatchDataset(
        validation_world_stream, validation_tail_stream
    )
    validation_loader = torch.utils.data.DataLoader(
        paired_validation,
        batch_size=None,
        num_workers=0,
        pin_memory=loader["pin_memory"],
    )

    action_dim = int(world_dataset.get_dim("action"))
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
        raise ValueError("The formal optimizer budget exceeds the paired loader.")
    formal_steps = protocol["training"]["scheduler_epochs"] * formal_epoch_steps
    train_limit = min(2, available_epoch_steps) if smoke else formal_epoch_steps
    if max_steps is not None:
        train_limit = min(int(max_steps), available_epoch_steps)
    if smoke:
        # The paired smoke is intentionally two-stage: the first invocation
        # writes a checkpoint after two updates and ``resume=required`` advances
        # it through a second two-update epoch.
        total_steps = 2 * int(train_limit)
    elif max_steps is not None:
        total_steps = int(train_limit)
    else:
        total_steps = formal_steps
    module = _build_training_module(world_model, protocol, total_steps)

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
        _build_epoch_callback(paired_train),
    ]
    logger = build_metrics_logger(run_dir, protocol["logging"])
    if smoke:
        epochs = 2 if resume == "required" else 1
    elif max_steps is not None:
        epochs = 1
    else:
        epochs = protocol["training"]["epochs"]
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
            limit_train_batches=train_limit,
            limit_val_batches=0.0 if smoke or skip_validation else 1.0,
            num_sanity_val_steps=0,
            logger=logger,
            callbacks=callbacks,
            log_every_n_steps=1 if smoke else 50,
        )

    last_checkpoint = checkpoint_dir / "last.ckpt"
    if resume == "required" and not last_checkpoint.is_file():
        raise FileNotFoundError(f"Required resume checkpoint not found: {last_checkpoint}")
    if resume != "never" and last_checkpoint.is_file():
        manifest_path = run_dir / "training_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("Cannot verify the objective version of this checkpoint.")
        with manifest_path.open() as stream:
            previous_manifest = json.load(stream)
        previous_protocol = previous_manifest.get("protocol", {})
        if previous_protocol.get("method") != METHOD or previous_protocol.get(
            "tail_value", {}
        ).get("objective_version") != 2:
            raise RuntimeError("Refusing to resume an incompatible objective.")
    checkpoint_path = None
    if resume != "never" and last_checkpoint.is_file():
        checkpoint_path = str(last_checkpoint)

    write_json(
        run_dir / "training_manifest.json",
        {
            "method": METHOD,
            "protocol": protocol,
            "protocol_path": str(Path(protocol_path).resolve()),
            "seed": seed,
            "dataset": {
                **dataset_source,
                "world_sequence_samples": len(world_dataset),
                "tail_sequence_samples": len(tail_dataset),
                "split": split_manifest,
            },
            "model": {"config": model_config, "parameters": parameter_count},
            "training": {
                "formal_optimizer_steps": formal_steps,
                "optimizer_steps_per_epoch": formal_epoch_steps,
                "available_paired_batches_per_epoch": available_epoch_steps,
                "configured_optimizer_steps": total_steps,
                "resume_mode": resume,
                "resumed_from": checkpoint_path,
                "world_batch_size": loader["world_batch_size"],
                "tail_batch_size": loader["tail_batch_size"],
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
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    write_json(run_dir / "training_result.json", result)
    return result
