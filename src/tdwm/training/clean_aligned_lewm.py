"""Causally controlled LeWM training with optional long-horizon auxiliaries.

The clean trainer keeps the short-clip LeWM update identical across variants.
Auxiliary forwards cannot update BatchNorm running statistics or advance the RNG
stream used by the common world-model branch, and world/value gradients are
clipped independently.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence
from unittest.mock import patch

import numpy as np
import torch
import yaml

from tdwm.adapters import prepare_cloud_runtime
from tdwm.methods.goal_tail_value import (
    BoundaryAnchoredGoalTailValue,
    monte_carlo_goal_tail_targets,
)
from tdwm.training.aligned_e2e_mc_gt_lewm import (
    _build_epoch_callback,
    _build_stream,
    _scale_gradient,
    ema_update_module,
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
    PairedEpisodeStreamingBatchDataset,
    StrideAwareLanceDataset,
)
from tdwm.training.lewm import _git_revision


METHOD = "clean_aligned_lewm"
VARIANTS = (
    "r0_common_lewm",
    "r1_head_only",
    "r2_anchored_mc",
    "r_data_extra_long_one_step",
    "r3_open_loop_multistep",
)


def load_clean_aligned_protocol(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as stream:
        protocol = yaml.safe_load(stream)
    validate_clean_aligned_protocol(protocol)
    return protocol


def validate_clean_aligned_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema_version") != 1 or protocol.get("method") != METHOD:
        raise ValueError("This trainer only accepts clean-aligned LeWM schema 1.")
    if protocol.get("environment") != "cube" or protocol.get("stage") != "full_training":
        raise ValueError("Clean-aligned LeWM is locked to full Cube training.")
    if protocol.get("initialization") != "paired_random_from_scratch":
        raise ValueError("All clean variants must use paired random initialization.")
    if protocol.get("runtime", {}).get("stable_worldmodel_version") != "0.1.1":
        raise ValueError("Clean-aligned LeWM requires stable-worldmodel 0.1.1.")

    sequence = protocol.get("sequence", {})
    history = sequence.get("history_frames")
    rollout = sequence.get("model_rollout_horizon")
    max_offset = sequence.get("max_goal_offset")
    if (history, rollout, max_offset) != (3, 5, 16):
        raise ValueError("The clean protocol locks history 3, rollout 5, goals 1..16.")
    if sequence.get("world_num_steps") != history + 1:
        raise ValueError("The common world view must be exactly one 3+1 LeWM clip.")
    if sequence.get("tail_num_steps") != history + rollout + max_offset:
        raise ValueError("The long view must cover history, rollout, and MC future.")
    if sequence.get("frame_skip") != 5 or sequence.get("prediction_frames") != 1:
        raise ValueError("Cube training requires frame skip 5 and one-step prediction.")

    controls = protocol.get("causal_controls", {})
    expected_controls = {
        "same_initial_world_state": True,
        "same_world_split": True,
        "same_tail_split": True,
        "same_world_batch_order": True,
        "same_tail_batch_order": True,
        "freeze_auxiliary_batchnorm_running_stats": True,
        "preserve_rng_around_auxiliary": True,
        "separate_world_value_gradient_clipping": True,
        "deterministic_algorithms": True,
    }
    for key, expected in expected_controls.items():
        if controls.get(key) is not expected:
            raise ValueError(f"causal_controls.{key} must be {expected!r}.")

    variants = protocol.get("variants", {})
    if tuple(variants) != VARIANTS:
        raise ValueError(f"variants must be declared in locked order: {VARIANTS!r}.")
    expected_variants = {
        "r0_common_lewm": ("none", 0.0, 0.0, False),
        "r1_head_only": ("anchored_mc_tail", 1.0, 0.0, True),
        "r2_anchored_mc": ("anchored_mc_tail", 1.0, 0.1, True),
        "r_data_extra_long_one_step": (
            "extra_long_view_one_step_lewm",
            0.1,
            1.0,
            False,
        ),
        "r3_open_loop_multistep": ("open_loop_multistep_mse", 1.0, 0.1, False),
    }
    for name, expected in expected_variants.items():
        variant = variants[name]
        actual = (
            variant.get("auxiliary"),
            variant.get("auxiliary_weight"),
            variant.get("world_gradient_scale"),
            variant.get("train_value_head"),
        )
        if actual != expected:
            raise ValueError(f"variants.{name} must equal {expected!r}, found {actual!r}.")
        if variant.get("inference") != "terminal_only":
            raise ValueError(f"variants.{name} must use terminal-only inference.")

    tail = protocol.get("tail_value", {})
    if tail.get("objective_version") != 2 or tail.get("objective") != "supervised_mc":
        raise ValueError("The clean tail head requires objective-version-two MC targets.")
    if tail.get("architecture") != "squared_shared_potential_anchor":
        raise ValueError("The tail head must enforce its zero boundary structurally.")
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
    if not 0.0 <= tail.get("auxiliary_warmup_fraction", -1.0) < 1.0:
        raise ValueError("auxiliary_warmup_fraction must lie in [0, 1).")

    loader = protocol.get("loader", {})
    if loader.get("world_batch_size") != 128:
        raise ValueError("The common LeWM view must retain 128 independent clips.")
    if loader.get("world_minimum_unique_episodes_per_batch") != 128:
        raise ValueError("Every common LeWM item must come from a distinct episode.")
    if loader.get("tail_batch_size", 0) <= 0:
        raise ValueError("The paired long-view batch must be positive.")
    if loader.get("tail_minimum_unique_episodes_per_batch") != loader.get(
        "tail_batch_size"
    ):
        raise ValueError("Every long-view item must come from a distinct episode.")
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
        raise ValueError("The clean dual-view protocol requires episode streaming.")
    if loader.get("device_image_preprocessing") is not True:
        raise ValueError("The clean raw-image run preprocesses uint8 images on GPU.")
    if protocol.get("loss", {}).get("sigreg", {}).get("effective_batch_size") != 128:
        raise ValueError("The common SIGReg branch must use 128 independent clips.")

    training = protocol.get("training", {})
    if training.get("epochs") != training.get("scheduler_epochs"):
        raise ValueError("Scheduler and trainer epochs must match.")
    if training.get("epochs", 0) <= 0 or training.get("optimizer_steps_per_epoch", 0) <= 0:
        raise ValueError("The formal optimizer budget must be positive.")
    if min(
        training.get("world_gradient_clip_norm", 0.0),
        training.get("value_gradient_clip_norm", 0.0),
    ) <= 0.0:
        raise ValueError("World and value clipping norms must both be positive.")
    if training.get("checkpoint_selection") != "locked_final_epoch":
        raise ValueError("The clean first run selects the locked final epoch.")
    if protocol.get("scheduler", {}).get("interval") != "optimizer_step":
        raise ValueError("The scheduler must step once per optimizer update.")
    if not protocol.get("seeds"):
        raise ValueError("At least one formal seed is required.")

    reproducibility = protocol.get("reproducibility", {})
    expected_seed_sources = {
        "initialization_seed": "training_seed",
        "split_seed": "training_seed",
        "world_stream_seed": "training_seed",
        "tail_stream_seed_offset": 1_000_003,
        "validation_stream_seed_offset": 2_000_003,
    }
    for key, expected in expected_seed_sources.items():
        if reproducibility.get(key) != expected:
            raise ValueError(f"reproducibility.{key} must be {expected!r}.")

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


@contextmanager
def freeze_batchnorm_running_stats(module: torch.nn.Module) -> Iterator[None]:
    """Freeze all BatchNorm buffers while retaining gradients through the module."""

    batchnorm_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
    )
    batchnorms = [
        child for child in module.modules() if isinstance(child, batchnorm_types)
    ]
    training_states = [child.training for child in batchnorms]
    for child in batchnorms:
        child.eval()
    try:
        yield
    finally:
        for child, was_training in zip(batchnorms, training_states, strict=True):
            child.train(was_training)


@contextmanager
def preserve_torch_rng_state() -> Iterator[None]:
    """Prevent an auxiliary forward from perturbing the common branch RNG."""

    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    """Hash tensor names, metadata, and exact bytes in a state dict."""

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(value.dtype).encode())
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode())
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def split_sha256(split_manifest: dict[str, Any]) -> str:
    """Hash both train and validation index hashes as one split identity."""

    digest = hashlib.sha256()
    for key in ("train_indices_sha256", "validation_indices_sha256"):
        digest.update(key.encode())
        digest.update(b"\0")
        digest.update(str(split_manifest[key]).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def compare_state_dicts(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> dict[str, Any]:
    """Return an exact, auditable tensor-by-tensor state comparison."""

    left_keys = set(left)
    right_keys = set(right)
    mismatches: list[dict[str, Any]] = []
    maximum_absolute_difference = 0.0
    for name in sorted(left_keys & right_keys):
        left_value = left[name].detach().cpu()
        right_value = right[name].detach().cpu()
        if left_value.shape != right_value.shape or left_value.dtype != right_value.dtype:
            mismatches.append(
                {
                    "name": name,
                    "left_shape": list(left_value.shape),
                    "right_shape": list(right_value.shape),
                    "left_dtype": str(left_value.dtype),
                    "right_dtype": str(right_value.dtype),
                }
            )
            continue
        if not torch.equal(left_value, right_value):
            if left_value.is_floating_point():
                difference = float(
                    (left_value.float() - right_value.float()).abs().max().item()
                )
                maximum_absolute_difference = max(
                    maximum_absolute_difference, difference
                )
            mismatches.append({"name": name})
    return {
        "exact_match": not mismatches and left_keys == right_keys,
        "left_sha256": state_dict_sha256(left),
        "right_sha256": state_dict_sha256(right),
        "left_only": sorted(left_keys - right_keys),
        "right_only": sorted(right_keys - left_keys),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "maximum_absolute_difference": maximum_absolute_difference,
    }


def clip_parameter_groups(
    world_parameters: Sequence[torch.nn.Parameter],
    value_parameters: Sequence[torch.nn.Parameter],
    *,
    world_max_norm: float,
    value_max_norm: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clip world and value gradients independently and return pre-clip norms."""

    world_with_grad = [parameter for parameter in world_parameters if parameter.grad is not None]
    value_with_grad = [parameter for parameter in value_parameters if parameter.grad is not None]
    if world_with_grad:
        world_norm = torch.nn.utils.clip_grad_norm_(world_with_grad, world_max_norm)
    else:
        world_norm = torch.tensor(0.0)
    if value_with_grad:
        value_norm = torch.nn.utils.clip_grad_norm_(value_with_grad, value_max_norm)
    else:
        device = world_norm.device if isinstance(world_norm, torch.Tensor) else None
        value_norm = torch.tensor(0.0, device=device)
    return world_norm, value_norm


def _build_training_module(
    world_model: Any,
    protocol: dict[str, Any],
    variant_name: str,
    total_steps: int,
):
    import lightning as pl
    import stable_worldmodel as swm

    variant = protocol["variants"][variant_name]

    class CleanAlignedLeWMModule(pl.LightningModule):
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
            self.variant_name = variant_name
            self.auxiliary = str(variant["auxiliary"])
            self.auxiliary_weight = float(variant["auxiliary_weight"])
            self.world_gradient_scale = float(variant["world_gradient_scale"])
            self.history_size = int(protocol["sequence"]["history_frames"])
            self.rollout_horizon = int(protocol["sequence"]["model_rollout_horizon"])
            self.max_goal_offset = int(protocol["sequence"]["max_goal_offset"])
            self.gamma = float(tail["gamma"])
            self.target_world_ema_decay = float(tail["target_world_ema_decay"])
            self.world_gradient_clip_norm = float(
                protocol["training"]["world_gradient_clip_norm"]
            )
            self.value_gradient_clip_norm = float(
                protocol["training"]["value_gradient_clip_norm"]
            )
            self.auxiliary_warmup_steps = int(
                float(tail["auxiliary_warmup_fraction"]) * total_steps
            )

        def train(self, mode: bool = True):
            super().train(mode)
            self.target_model.eval()
            return self

        def _auxiliary_scale(self) -> float:
            if self.auxiliary == "none":
                return 0.0
            if self.auxiliary_warmup_steps <= 0:
                return 1.0
            return min(
                1.0,
                float(self.global_step + 1) / self.auxiliary_warmup_steps,
            )

        def _preprocess(self, pixels: torch.Tensor) -> torch.Tensor:
            return preprocess_image_batch(
                pixels,
                mean=self.image_mean,
                std=self.image_std,
                size=protocol["image_preprocessing"]["size"],
            )

        def _lewm_loss(
            self,
            batch: dict[str, Any],
            *,
            num_steps: int | None = None,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            episode_ids = batch.get("_tdwm_episode_id")
            pixels = batch["pixels"]
            actions = batch["action"]
            if num_steps is not None:
                pixels = pixels[:, :num_steps]
                actions = actions[:, :num_steps]
            pixels = self._preprocess(pixels)
            actions = torch.nan_to_num(actions, 0.0)
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

        def _predicted_rollout(
            self, batch: dict[str, Any]
        ) -> tuple[torch.Tensor, torch.Tensor]:
            actions = torch.nan_to_num(batch["action"], 0.0)
            terminal_index = self.history_size + self.rollout_horizon - 1
            initial_pixels = self._preprocess(batch["pixels"][:, : self.history_size])
            initial = self.model.encode(
                {
                    "pixels": initial_pixels,
                    "action": actions[:, : self.history_size],
                }
            )["emb"]
            predicted = rollout_from_latents(
                self.model,
                initial,
                actions[:, :terminal_index],
                history_size=self.history_size,
            )
            return predicted, actions

        def _mc_tail_loss(
            self, batch: dict[str, Any]
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            episode_ids = batch.get("_tdwm_episode_id")
            terminal_index = self.history_size + self.rollout_horizon - 1
            if self.world_gradient_scale == 0.0:
                with (
                    preserve_torch_rng_state(),
                    freeze_batchnorm_running_stats(self.model),
                    torch.no_grad(),
                ):
                    predicted_rollout, actions = self._predicted_rollout(batch)
            else:
                with (
                    preserve_torch_rng_state(),
                    freeze_batchnorm_running_stats(self.model),
                ):
                    predicted_rollout, actions = self._predicted_rollout(batch)
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

            if self.world_gradient_scale == 0.0:
                history_for_value = predicted_history.detach()
            else:
                history_for_value = _scale_gradient(
                    predicted_history, self.world_gradient_scale
                )
            expanded_history = history_for_value.unsqueeze(1).expand(
                -1, self.max_goal_offset, -1
            )
            predictions = self.value(expanded_history, goals)
            squared_error = (predictions - targets).pow(2)
            loss = squared_error.mean()
            with torch.no_grad():
                current = self.value.current_latent(predicted_history.detach())
                boundary = self.value(predicted_history.detach(), current)
                terminal_mse = (
                    predicted_rollout[:, terminal_index].detach() - target_latents[:, 0]
                ).pow(2).mean()
            unique = loss.new_tensor(
                float(torch.unique(episode_ids).numel())
                if episode_ids is not None
                else float(actions.shape[0])
            )
            return loss, {
                "auxiliary_loss": loss.detach(),
                "tail_value_prediction": predictions.detach().mean(),
                "tail_value_target": targets.detach().mean(),
                "tail_boundary_max": boundary.abs().max(),
                "terminal_rollout_mse": terminal_mse,
                "tail_mse_offset_1": squared_error[:, 0].detach().mean(),
                "tail_mse_offset_5": squared_error[:, 4].detach().mean(),
                "tail_mse_offset_16": squared_error[:, 15].detach().mean(),
                "tail_unique_episodes": unique,
            }

        def _extra_long_one_step_loss(
            self, batch: dict[str, Any]
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            with preserve_torch_rng_state(), freeze_batchnorm_running_stats(self.model):
                loss, metrics = self._lewm_loss(
                    batch,
                    num_steps=int(protocol["sequence"]["world_num_steps"]),
                )
            return loss, {
                "auxiliary_loss": loss.detach(),
                "extra_prediction_loss": metrics["prediction_loss"],
                "extra_sigreg_loss": metrics["sigreg_loss"],
                "tail_unique_episodes": metrics["world_unique_episodes"],
            }

        def _multistep_loss(
            self, batch: dict[str, Any]
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            terminal_index = self.history_size + self.rollout_horizon - 1
            with preserve_torch_rng_state(), freeze_batchnorm_running_stats(self.model):
                predicted_rollout, actions = self._predicted_rollout(batch)
            predicted_future = predicted_rollout[
                :, self.history_size : terminal_index + 1
            ]
            target_pixels = self._preprocess(
                batch["pixels"][:, self.history_size : terminal_index + 1]
            )
            with torch.no_grad():
                target_future = self.target_model.encode(
                    {
                        "pixels": target_pixels,
                        "action": actions[:, self.history_size : terminal_index + 1],
                    }
                )["emb"]
            predicted_for_loss = _scale_gradient(
                predicted_future, self.world_gradient_scale
            )
            per_horizon = (predicted_for_loss - target_future).pow(2).mean(dim=-1)
            loss = per_horizon.mean()
            metrics = {
                "auxiliary_loss": loss.detach(),
                "multistep_mse_h1": per_horizon[:, 0].detach().mean(),
                "multistep_mse_h5": per_horizon[:, -1].detach().mean(),
            }
            return loss, metrics

        def _auxiliary_loss(
            self, batch: dict[str, Any]
        ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
            if self.auxiliary == "none":
                return None, {}
            if self.auxiliary == "anchored_mc_tail":
                return self._mc_tail_loss(batch)
            if self.auxiliary == "extra_long_view_one_step_lewm":
                return self._extra_long_one_step_loss(batch)
            if self.auxiliary == "open_loop_multistep_mse":
                return self._multistep_loss(batch)
            raise RuntimeError(f"Unknown auxiliary objective: {self.auxiliary}")

        def _log_metrics(
            self,
            stage: str,
            total: torch.Tensor,
            world_metrics: dict[str, torch.Tensor],
            auxiliary_metrics: dict[str, torch.Tensor],
            auxiliary_scale: float,
            gradient_metrics: dict[str, torch.Tensor] | None = None,
        ) -> None:
            metrics = {f"{stage}/loss": total.detach()}
            metrics.update(
                {f"{stage}/{key}": value for key, value in world_metrics.items()}
            )
            metrics.update(
                {f"{stage}/{key}": value for key, value in auxiliary_metrics.items()}
            )
            if gradient_metrics:
                metrics.update(
                    {f"{stage}/{key}": value for key, value in gradient_metrics.items()}
                )
            metrics[f"{stage}/auxiliary_weight_scale"] = total.new_tensor(
                auxiliary_scale
            )
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

            world_loss, world_metrics = self._lewm_loss(batch["world"])
            self.manual_backward(world_loss)
            auxiliary_loss, auxiliary_metrics = self._auxiliary_loss(batch["tail"])
            auxiliary_scale = self._auxiliary_scale()
            weighted_auxiliary = None
            if auxiliary_loss is not None:
                weighted_auxiliary = (
                    auxiliary_scale * self.auxiliary_weight * auxiliary_loss
                )
                self.manual_backward(weighted_auxiliary)

            world_parameters = list(self.model.parameters())
            value_parameters = list(self.value.parameters())
            world_norm, value_norm = clip_parameter_groups(
                world_parameters,
                value_parameters,
                world_max_norm=self.world_gradient_clip_norm,
                value_max_norm=self.value_gradient_clip_norm,
            )
            optimizer.step()
            scheduler = self.lr_schedulers()
            scheduler.step()
            ema_update_module(
                self.target_model,
                self.model,
                decay=self.target_world_ema_decay,
            )
            total = world_loss.detach()
            if weighted_auxiliary is not None:
                total = total + weighted_auxiliary.detach()
            self._log_metrics(
                "train",
                total,
                world_metrics,
                auxiliary_metrics,
                auxiliary_scale,
                {
                    "world_gradient_norm_before_clip": world_norm.detach(),
                    "value_gradient_norm_before_clip": value_norm.detach(),
                },
            )
            return total

        def validation_step(
            self, batch: dict[str, dict[str, Any]], batch_idx: int
        ) -> torch.Tensor:
            del batch_idx
            world_loss, world_metrics = self._lewm_loss(batch["world"])
            auxiliary_loss, auxiliary_metrics = self._auxiliary_loss(batch["tail"])
            auxiliary_scale = self._auxiliary_scale()
            total = world_loss
            if auxiliary_loss is not None:
                total = total + auxiliary_scale * self.auxiliary_weight * auxiliary_loss
            self._log_metrics(
                "validation",
                total,
                world_metrics,
                auxiliary_metrics,
                auxiliary_scale,
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

    return CleanAlignedLeWMModule()


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
    }


def _build_export_callback(
    run_dir: Path,
    model_config: dict[str, Any],
    protocol: dict[str, Any],
    variant_name: str,
):
    import lightning as pl
    import stable_worldmodel as swm
    from omegaconf import OmegaConf

    export_config = OmegaConf.create(model_config)

    class CleanAlignedExportCallback(pl.Callback):
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
                    "variant": variant_name,
                    "deployment_checkpoint_version": 1,
                    "epoch": epoch,
                    "global_step": int(trainer.global_step),
                    "inference": "terminal_only",
                    "world_model_state_sha256": state_dict_sha256(
                        pl_module.model.state_dict()
                    ),
                    "world_model_state_dict": pl_module.model.state_dict(),
                    "value_state_dict": pl_module.value.state_dict(),
                    "world_model_config": model_config,
                    "value_config": _value_config(protocol),
                },
                deployment_dir / f"epoch_{epoch:02d}.pt",
            )

    return CleanAlignedExportCallback()


def train_clean_aligned_lewm(
    *,
    protocol_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    variant: str,
    seed: int,
    smoke: bool = False,
    resume: str = "auto",
    max_steps: int | None = None,
    skip_validation: bool = False,
) -> dict[str, Any]:
    """Train one causally controlled LeWM variant from raw Cube observations."""

    protocol = load_clean_aligned_protocol(protocol_path)
    if variant not in protocol["variants"]:
        raise ValueError(f"Unknown variant {variant!r}; choose from {VARIANTS!r}.")
    if seed not in protocol["seeds"]:
        raise ValueError(f"Seed {seed} is not in the locked seeds {protocol['seeds']}.")
    if resume not in {"auto", "never", "required"}:
        raise ValueError("resume must be one of: auto, never, required.")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive when provided.")

    dataset_path = Path(dataset_path).expanduser().resolve()
    dataset_source = validate_cube_training_dataset(dataset_path, protocol["dataset"])
    if dataset_source["format"] != "lance":
        raise ValueError("Clean dual-view training requires the audited Lance data.")
    output_dir = Path(output_dir).expanduser().resolve()
    run_dir = output_dir / variant / (
        f"seed_{seed}_smoke" if smoke else f"seed_{seed}"
    )
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
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    initialization_seed = seed
    split_seed = seed
    world_stream_seed = seed
    tail_stream_seed = seed + int(
        protocol["reproducibility"]["tail_stream_seed_offset"]
    )
    validation_seed = seed + int(
        protocol["reproducibility"]["validation_stream_seed_offset"]
    )
    pl.seed_everything(initialization_seed, workers=True)

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
    world_generator = torch.Generator().manual_seed(split_seed)
    tail_generator = torch.Generator().manual_seed(split_seed)
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
        seed=world_stream_seed,
    )
    tail_stream = _build_stream(
        tail_dataset,
        tail_train.indices,
        batch_size=loader["tail_batch_size"],
        minimum_unique_episodes=loader["tail_minimum_unique_episodes_per_batch"],
        cache_bytes=loader["tail_episode_cache_bytes"],
        protocol=protocol,
        seed=tail_stream_seed,
    )
    paired_train = PairedEpisodeStreamingBatchDataset(world_stream, tail_stream)
    train_loader = torch.utils.data.DataLoader(
        paired_train,
        batch_size=None,
        num_workers=0,
        pin_memory=loader["pin_memory"],
    )

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
    initial_world_state_sha256 = state_dict_sha256(world_model.state_dict())
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
        total_steps = 2 * int(train_limit)
    elif max_steps is not None:
        total_steps = int(train_limit)
    else:
        total_steps = formal_steps
    module = _build_training_module(world_model, protocol, variant, total_steps)

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
        _build_export_callback(run_dir, model_config, protocol, variant),
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
            deterministic=True,
            benchmark=False,
        )

    last_checkpoint = checkpoint_dir / "last.ckpt"
    if resume == "required" and not last_checkpoint.is_file():
        raise FileNotFoundError(f"Required resume checkpoint not found: {last_checkpoint}")
    if resume != "never" and last_checkpoint.is_file():
        manifest_path = run_dir / "training_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("Cannot verify this clean checkpoint without its manifest.")
        with manifest_path.open() as stream:
            previous_manifest = json.load(stream)
        if previous_manifest.get("method") != METHOD or previous_manifest.get(
            "variant"
        ) != variant:
            raise RuntimeError("Refusing to resume an incompatible clean variant.")
        if previous_manifest.get("initial_world_state_sha256") != initial_world_state_sha256:
            raise RuntimeError("Fresh initialization hash changed since the prior run.")
    checkpoint_path = None
    if resume != "never" and last_checkpoint.is_file():
        checkpoint_path = str(last_checkpoint)

    world_split_sha256 = split_sha256(split_manifest["world"])
    tail_split_sha256 = split_sha256(split_manifest["tail"])
    write_json(
        run_dir / "training_manifest.json",
        {
            "method": METHOD,
            "variant": variant,
            "variant_config": protocol["variants"][variant],
            "protocol": protocol,
            "protocol_path": str(Path(protocol_path).resolve()),
            "seed": seed,
            "initialization_seed": initialization_seed,
            "split_seed": split_seed,
            "initial_world_state_sha256": initial_world_state_sha256,
            "world_split_sha256": world_split_sha256,
            "tail_split_sha256": tail_split_sha256,
            "world_stream_seed": world_stream_seed,
            "tail_stream_seed": tail_stream_seed,
            "validation_world_stream_seed": validation_seed,
            "validation_tail_stream_seed": validation_seed + 1,
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
    final_world_state_sha256 = state_dict_sha256(module.model.state_dict())
    result = {
        "run_dir": str(run_dir),
        "variant": variant,
        "seed": seed,
        "initial_world_state_sha256": initial_world_state_sha256,
        "final_world_state_sha256": final_world_state_sha256,
        "world_split_sha256": world_split_sha256,
        "tail_split_sha256": tail_split_sha256,
        "world_stream_seed": world_stream_seed,
        "tail_stream_seed": tail_stream_seed,
        "last_checkpoint": str(last_checkpoint),
        "final_epoch": trainer.current_epoch,
        "global_step": trainer.global_step,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    write_json(run_dir / "training_result.json", result)
    return result
