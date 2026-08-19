"""Standalone GT-LeWM training entry point.

This module intentionally does not modify or dispatch through the LeWM
baseline trainer. It owns its protocol, data/model assembly, Lightning module,
and exports as a separate method entry point.
"""

from __future__ import annotations

import copy
import importlib.metadata
import math
import platform
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
import yaml

from tdwm.adapters import prepare_cloud_runtime
from tdwm.methods.goal_tail import (
    GoalTailValue,
    discounted_goal_tail_target,
    goal_tail_loss,
    soft_update,
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
    write_json,
)
from tdwm.training.lance_batch import StrideAwareLanceDataset


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
    history = sequence.get("history_frames", 0)
    horizon = tail.get("horizon", 0)
    if sequence.get("prediction_frames") != 1:
        raise ValueError("GT-LeWM requires exactly one local prediction frame.")
    if horizon <= 0 or sequence.get("num_steps") != history + horizon:
        raise ValueError("GT-LeWM num_steps must equal history_frames + tail horizon.")
    if not 0.0 <= tail.get("gamma", -1.0) < 1.0:
        raise ValueError("GT-LeWM gamma must lie in [0, 1).")
    if tail.get("loss_weight", -1.0) < 0.0:
        raise ValueError("GT-LeWM loss_weight cannot be negative.")
    if not 0.0 < tail.get("target_tau", 0.0) <= 1.0:
        raise ValueError("GT-LeWM target_tau must lie in (0, 1].")
    if tail.get("goal_source") != "n_step_episode_endpoint":
        raise ValueError("GT-LeWM uses the N-step episode endpoint as its training goal.")

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
    dataset = protocol.get("dataset", {})
    lance = dataset.get("lance", {})
    if lance.get("suffix") != ".lance" or lance.get("manifest_suffix") != ".manifest.json":
        raise ValueError("GT-LeWM Lance layout is not locked to the audited format.")
    if lance.get("image_codec") != "jpeg" or lance.get("jpeg_quality") != 100:
        raise ValueError("GT-LeWM Lance images must use JPEG quality 100.")
    if lance.get("stable_worldmodel_version") != "0.1.1":
        raise ValueError("GT-LeWM Lance conversion is locked to 0.1.1.")
    if lance.get("source", {}).get("sha256") != dataset.get("optimized_layout", {}).get("sha256"):
        raise ValueError("GT-LeWM Lance source must match the audited optimized layout.")


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
            self.target_tau = float(tail["target_tau"])
            self.horizon = int(tail["horizon"])

        def _forward_loss(self, batch: dict[str, Any], stage: str):
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
            if embeddings.shape[1] < history + self.horizon:
                raise RuntimeError("GT-LeWM received a clip shorter than its tail horizon.")

            predicted = self.model.predict(
                embeddings[:, :history], output["act_emb"][:, :history]
            )
            local_target = embeddings[:, 1 : history + 1]
            prediction_loss = (predicted - local_target).pow(2).mean()
            sigreg_loss = self.sigreg(embeddings.transpose(0, 1))

            current = embeddings[:, history - 1]
            future = embeddings[:, history : history + self.horizon]
            goal = future[:, -1]
            value_prediction = self.value(current, goal).squeeze(-1)
            with torch.no_grad():
                bootstrap = self.target_value(
                    future[:, -1].detach(), goal.detach()
                ).squeeze(-1)
                value_target = discounted_goal_tail_target(
                    future.detach(), goal.detach(), bootstrap, gamma=self.gamma
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
                        "embed_dim": protocol["model"]["embed_dim"],
                        "hidden_dim": protocol["tail_value"]["hidden_dim"],
                        "gamma": protocol["tail_value"]["gamma"],
                        "horizon": protocol["tail_value"]["horizon"],
                        "target_tau": protocol["tail_value"]["target_tau"],
                    },
                    "world_model_config": model_config,
                },
                value_dir / f"epoch_{trainer.current_epoch + 1:02d}.pt",
            )

    return GTExportCallback()


def train_gt_lewm(
    *,
    protocol_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    seed: int,
    smoke: bool = False,
    resume: str = "auto",
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Train GT-LeWM without entering the baseline LeWM training entry point."""

    protocol = load_gt_training_protocol(protocol_path)
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
        raise RuntimeError(f"Expected stable-worldmodel {expected_version}, found {package_version}.")
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
    if int(torch.as_tensor(dataset.lengths).sum()) != dataset_cfg["expected_transitions"]:
        raise ValueError("Dataset transition count does not match the GT-LeWM protocol.")

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
    validation_loader = torch.utils.data.DataLoader(
        validation_set,
        batch_size=loader_cfg["batch_size"],
        shuffle=False,
        drop_last=loader_cfg["validation_drop_last"],
        num_workers=0 if smoke else int(loader_cfg["validation_workers"]),
        pin_memory=loader_cfg["pin_memory"],
    )

    action_dim = int(dataset.get_dim("action"))
    model_config = build_model_config(protocol, action_dim)
    world_model = hydra.utils.instantiate(model_config)
    parameter_count = sum(parameter.numel() for parameter in world_model.parameters())
    expected_parameters = protocol["model"].get("parameters")
    if expected_parameters and parameter_count != expected_parameters:
        raise ValueError(f"Expected {expected_parameters} parameters, found {parameter_count}.")
    if protocol["training"]["model_compile"]:
        compile_world_model(
            world_model, mode=protocol["training"]["model_compile_mode"]
        )

    formal_steps = protocol["training"]["scheduler_epochs"] * len(train_loader)
    train_limit = resolve_train_batch_limit(
        smoke=smoke, max_steps=max_steps, train_loader_length=len(train_loader)
    )
    total_steps = min(2, len(train_loader)) if smoke else formal_steps
    if max_steps is not None:
        total_steps = int(train_limit)
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
    logger = build_metrics_logger(run_dir, protocol["logging"])
    epochs = 1 if smoke or max_steps is not None else protocol["training"]["epochs"]
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
            gradient_clip_val=protocol["training"]["gradient_clip_norm"],
            limit_train_batches=train_limit,
            limit_val_batches=0.0 if smoke else 1,
            num_sanity_val_steps=0,
            logger=logger,
            callbacks=[
                checkpoint_callback,
                _build_export_callback(run_dir, model_config, protocol),
            ],
            log_every_n_steps=1 if smoke else 50,
        )

    last_checkpoint = checkpoint_dir / "last.ckpt"
    if resume == "required" and not last_checkpoint.is_file():
        raise FileNotFoundError(f"Required resume checkpoint not found: {last_checkpoint}")
    checkpoint_path = (
        None if resume == "never" else str(last_checkpoint) if last_checkpoint.is_file() else None
    )
    write_json(
        run_dir / "training_manifest.json",
        {
            "method": "gt_lewm",
            "protocol": protocol,
            "protocol_path": str(Path(protocol_path).resolve()),
            "seed": seed,
            "dataset": {
                **dataset_source,
                "sequence_samples": len(dataset),
            },
            "model": {"config": model_config, "parameters": parameter_count},
            "training": {
                "formal_optimizer_steps": formal_steps,
                "configured_optimizer_steps": total_steps,
                "resume_mode": resume,
                "resumed_from": checkpoint_path,
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
