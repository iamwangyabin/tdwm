"""End-to-end Joint TD-GT-LeWM training from raw Cube observations."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import torch
import yaml

from tdwm.methods.goal_tail_value import GoalTailValue
from tdwm.training.gt_lewm import train_gt_lewm
from tdwm.training.gt_lewm_support import preprocess_image_batch
from tdwm.training.joint_td_gt_lewm import (
    build_joint_td_batch,
    teacher_forced_windows,
)
from tdwm.training.td_gt_lewm import ema_update_target


METHOD = "e2e_joint_td_gt_lewm"


def load_e2e_joint_td_gt_protocol(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as stream:
        protocol = yaml.safe_load(stream)
    validate_e2e_joint_td_gt_protocol(protocol)
    return protocol


def validate_e2e_joint_td_gt_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema_version") != 1 or protocol.get("method") != METHOD:
        raise ValueError("This trainer only accepts E2E Joint TD-GT-LeWM schema 1.")
    if protocol.get("environment") != "cube" or protocol.get("stage") != "full_training":
        raise ValueError("E2E Joint TD-GT-LeWM is locked to full Cube training.")
    if protocol.get("runtime", {}).get("stable_worldmodel_version") != "0.1.1":
        raise ValueError("E2E Joint TD-GT-LeWM requires stable-worldmodel 0.1.1.")
    if protocol.get("initialization") != "random_from_scratch":
        raise ValueError("The end-to-end method must start from random parameters.")

    sequence = protocol.get("sequence", {})
    history = sequence.get("history_frames")
    rollout = sequence.get("model_rollout_horizon")
    max_offset = sequence.get("max_goal_offset")
    if (history, rollout, max_offset) != (3, 5, 16):
        raise ValueError("The method locks history 3, rollout 5, and goals 1..16.")
    if sequence.get("num_steps") != history + rollout + max_offset:
        raise ValueError("num_steps must cover history, rollout, and all future goals.")
    prediction_windows = sequence.get("prediction_windows", 0)
    if prediction_windows <= 0 or prediction_windows > sequence["num_steps"] - history:
        raise ValueError("prediction_windows does not fit inside the raw clip.")
    if sequence.get("prediction_frames") != 1 or sequence.get("frame_skip", 0) <= 0:
        raise ValueError("LeWM requires one-step local targets and positive frame skip.")

    objective = protocol.get("joint_objective", {})
    required_objective = {
        "prediction": "teacher_forced_lewm_mse",
        "regularization": "sigreg",
        "tail": "one_step_td_on_predicted_terminal_history",
        "tail_input": "five_step_differentiable_lewm_rollout",
        "backpropagate_tail_through_rollout": True,
    }
    for key, expected in required_objective.items():
        if objective.get(key) != expected:
            raise ValueError(f"joint_objective.{key} must be {expected!r}.")
    if objective.get("prediction_weight") != 1.0:
        raise ValueError("The original LeWM prediction weight remains one.")
    if objective.get("tail_weight", -1.0) <= 0.0:
        raise ValueError("The TD tail weight must be positive.")

    tail = protocol.get("tail_value", {})
    if tail.get("objective_version") != 1 or tail.get("objective") != "one_step_td":
        raise ValueError("The value objective must be version-one one-step TD.")
    if tail.get("goal_sampling") != "uniform_future_offset":
        raise ValueError("Hindsight goals must be sampled uniformly from offsets 1..16.")
    if tail.get("target_network") is not True:
        raise ValueError("The TD objective requires an EMA target value.")
    if not 0.0 <= tail.get("gamma", -1.0) < 1.0:
        raise ValueError("tail_value.gamma must lie in [0, 1).")
    if not 0.0 <= tail.get("target_ema_decay", -1.0) < 1.0:
        raise ValueError("target_ema_decay must lie in [0, 1).")
    if tail.get("terminate_bootstrap_at_goal") is not True:
        raise ValueError("Hindsight goals must terminate TD bootstrapping.")

    split = protocol.get("split", {})
    if split.get("unit") != "sequence_clip" or not math.isclose(
        split.get("train_fraction", 0.0) + split.get("validation_fraction", 0.0),
        1.0,
    ):
        raise ValueError("Training requires a complete clip-level split.")
    training = protocol.get("training", {})
    if training.get("epochs") != training.get("scheduler_epochs"):
        raise ValueError("Scheduler and trainer epochs must match.")
    if training.get("epochs", 0) <= 0 or training.get("optimizer_steps_per_epoch", 0) <= 0:
        raise ValueError("The formal optimizer budget must be positive.")
    if protocol.get("scheduler", {}).get("interval") != "optimizer_step":
        raise ValueError("The scheduler must step once per optimizer update.")
    if not protocol.get("seeds"):
        raise ValueError("At least one formal seed is required.")

    loader = protocol.get("loader", {})
    if loader.get("batch_size", 0) <= 0 or loader.get("workers", -1) < 0:
        raise ValueError("Loader settings are invalid.")
    if not isinstance(loader.get("device_image_preprocessing"), bool):
        raise ValueError("device_image_preprocessing must be boolean.")
    if not isinstance(loader.get("episode_streaming"), bool):
        raise ValueError("episode_streaming must be boolean.")
    if not 1 <= loader.get("minimum_unique_episodes_per_batch", 0) <= loader["batch_size"]:
        raise ValueError("The batch episode-diversity constraint is invalid.")
    effective_batch = loader["batch_size"] * prediction_windows
    if effective_batch != protocol.get("loss", {}).get("sigreg", {}).get(
        "effective_batch_size"
    ):
        raise ValueError("The local windows must preserve SIGReg batch size 128.")

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


def _build_training_module(
    world_model: Any,
    protocol: dict[str, Any],
    total_steps: int,
    *,
    device_image_preprocessing: bool,
):
    import lightning as pl
    import stable_worldmodel as swm

    class E2EJointTDGTLeWMModule(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.model = world_model
            self.device_image_preprocessing = device_image_preprocessing
            if device_image_preprocessing:
                image = protocol["image_preprocessing"]
                self.register_buffer(
                    "image_mean",
                    torch.tensor(image["mean"], dtype=torch.float32).reshape(1, 1, 3, 1, 1),
                    persistent=False,
                )
                self.register_buffer(
                    "image_std",
                    torch.tensor(image["std"], dtype=torch.float32).reshape(1, 1, 3, 1, 1),
                    persistent=False,
                )
            sigreg = protocol["loss"]["sigreg"]
            self.sigreg = swm.wm.SIGReg(
                knots=sigreg["knots"], num_proj=sigreg["num_projections"]
            )
            tail = protocol["tail_value"]
            self.value = GoalTailValue(
                history_dim=int(tail["history_dim"]),
                goal_dim=int(protocol["model"]["embed_dim"]),
                hidden_dim=int(tail["hidden_dim"]),
            )
            self.target_value = copy.deepcopy(self.value).requires_grad_(False)
            self.gamma = float(tail["gamma"])
            self.ema_decay = float(tail["target_ema_decay"])
            self.history_size = int(protocol["sequence"]["history_frames"])
            self.rollout_horizon = int(protocol["sequence"]["model_rollout_horizon"])
            self.max_goal_offset = int(protocol["sequence"]["max_goal_offset"])
            self.prediction_windows = int(protocol["sequence"]["prediction_windows"])
            self.tail_warmup_steps = int(
                float(tail["loss_warmup_fraction"]) * total_steps
            )

        def _tail_scale(self) -> float:
            if self.tail_warmup_steps <= 0:
                return 1.0
            return min(1.0, float(self.global_step + 1) / self.tail_warmup_steps)

        def _goal_offsets(
            self, batch_size: int, batch_idx: int, stage: str, device: torch.device
        ) -> torch.Tensor:
            if stage == "train":
                return torch.randint(
                    1,
                    self.max_goal_offset + 1,
                    (batch_size,),
                    device=device,
                )
            first = batch_idx * batch_size
            return (
                torch.arange(first, first + batch_size, device=device)
                % self.max_goal_offset
            ) + 1

        def _forward_loss(
            self, batch: dict[str, Any], batch_idx: int, stage: str
        ) -> torch.Tensor:
            batch_size = int(batch["pixels"].shape[0])
            episode_ids = batch.pop("_tdwm_episode_id", None)
            if episode_ids is not None:
                self.log(
                    f"{stage}/unique_episodes_per_batch",
                    torch.unique(episode_ids).numel(),
                    on_step=stage == "train",
                    on_epoch=True,
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
            encoded = self.model.encode(batch)
            embeddings = encoded["emb"]
            if embeddings.shape[1] != protocol["sequence"]["num_steps"]:
                raise RuntimeError("The raw Cube clip has an unexpected length.")

            local_histories, local_actions, local_targets = teacher_forced_windows(
                embeddings,
                encoded["act_emb"],
                history_size=self.history_size,
                rollout_horizon=self.prediction_windows,
            )
            local_predictions = self.model.predict(local_histories, local_actions)
            prediction_loss = (local_predictions - local_targets).pow(2).mean()
            local_sequences = torch.cat(
                [
                    embeddings[:, start : start + self.history_size + 1]
                    for start in range(self.prediction_windows)
                ],
                dim=0,
            )
            sigreg_loss = self.sigreg(local_sequences.transpose(0, 1))

            goal_offsets = self._goal_offsets(
                batch_size, batch_idx, stage, embeddings.device
            )
            pred_proj_was_training = self.model.pred_proj.training
            self.model.pred_proj.eval()
            try:
                td_batch = build_joint_td_batch(
                    self.model,
                    self.target_value,
                    embeddings,
                    batch["action"],
                    goal_offsets,
                    history_size=self.history_size,
                    rollout_horizon=self.rollout_horizon,
                    gamma=self.gamma,
                )
            finally:
                self.model.pred_proj.train(pred_proj_was_training)
            tail_prediction = self.value(td_batch.predicted_history, td_batch.goals)
            tail_loss = (tail_prediction - td_batch.targets).pow(2).mean()
            tail_scale = self._tail_scale()
            loss = (
                protocol["joint_objective"]["prediction_weight"] * prediction_loss
                + protocol["loss"]["sigreg"]["weight"] * sigreg_loss
                + tail_scale
                * protocol["joint_objective"]["tail_weight"]
                * tail_loss
            )
            terminal_index = self.history_size + self.rollout_horizon - 1
            terminal_mse = (
                td_batch.predicted_rollout[:, terminal_index]
                - embeddings[:, terminal_index]
            ).pow(2).mean()
            self.log_dict(
                {
                    f"{stage}/loss": loss,
                    f"{stage}/prediction_loss": prediction_loss,
                    f"{stage}/sigreg_loss": sigreg_loss,
                    f"{stage}/tail_value_loss": tail_loss,
                    f"{stage}/tail_value_prediction": tail_prediction.detach().mean(),
                    f"{stage}/tail_value_target": td_batch.targets.mean(),
                    f"{stage}/terminal_rollout_mse": terminal_mse,
                    f"{stage}/tail_weight_scale": tail_scale,
                    f"{stage}/goal_offset_mean": goal_offsets.float().mean(),
                },
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=stage != "train",
                batch_size=batch_size,
            )
            return loss

        def training_step(self, batch: dict[str, Any], batch_idx: int):
            return self._forward_loss(batch, batch_idx, "train")

        def validation_step(self, batch: dict[str, Any], batch_idx: int):
            return self._forward_loss(batch, batch_idx, "validation")

        def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
            ema_update_target(self.target_value, self.value, decay=self.ema_decay)

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
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

    return E2EJointTDGTLeWMModule()


def _value_config(protocol: dict[str, Any]) -> dict[str, Any]:
    tail = protocol["tail_value"]
    return {
        "objective_version": tail["objective_version"],
        "objective": tail["objective"],
        "input_distribution": "lewm_predicted_terminal_history",
        "history_size": protocol["sequence"]["history_frames"],
        "history_dim": tail["history_dim"],
        "goal_dim": protocol["model"]["embed_dim"],
        "action_block_dim": tail["action_block_dim"],
        "hidden_dim": tail["hidden_dim"],
        "gamma": tail["gamma"],
        "max_goal_offset": protocol["sequence"]["max_goal_offset"],
        "model_rollout_horizon": protocol["sequence"]["model_rollout_horizon"],
        "target_ema_decay": tail["target_ema_decay"],
    }


def _build_export_callback(
    run_dir: Path, model_config: dict[str, Any], protocol: dict[str, Any]
):
    import lightning as pl
    import stable_worldmodel as swm
    from omegaconf import OmegaConf

    export_config = OmegaConf.create(model_config)

    class E2EExportCallback(pl.Callback):
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
                    "objective_version": 1,
                    "deployment_checkpoint_version": 1,
                    "epoch": epoch,
                    "global_step": int(trainer.global_step),
                    "world_model_state_dict": pl_module.model.state_dict(),
                    "value_state_dict": pl_module.value.state_dict(),
                    "target_value_state_dict": pl_module.target_value.state_dict(),
                    "world_model_config": model_config,
                    "value_config": _value_config(protocol),
                },
                deployment_dir / f"epoch_{epoch:02d}.pt",
            )

    return E2EExportCallback()


def train_e2e_joint_td_gt_lewm(
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
    """Train all LeWM and goal-tail parameters together from raw observations."""

    return train_gt_lewm(
        protocol_path=protocol_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        seed=seed,
        smoke=smoke,
        resume=resume,
        max_steps=max_steps,
        skip_validation=skip_validation,
        _protocol_loader=load_e2e_joint_td_gt_protocol,
        _module_builder=_build_training_module,
        _export_callback_builder=_build_export_callback,
        _method=METHOD,
    )
