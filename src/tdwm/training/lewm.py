from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
import platform
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import lightning as pl
import numpy as np
import stable_worldmodel as swm
import torch
import yaml
from hydra.utils import instantiate
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, random_split


PINNED_STABLE_WORLDMODEL_VERSION = "0.1.1"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _install_torchvision_v2_compatibility() -> None:
    """Bridge APIs missing from the torchvision version bundled on Gemini."""
    from torchvision.transforms import v2

    if not hasattr(v2.Transform, "transform"):

        def transform(instance: Any, value: Any, params: dict[str, Any]) -> Any:
            return instance._transform(value, params)

        v2.Transform.transform = transform

    if not hasattr(v2.Transform, "make_params"):

        def make_params(instance: Any, flat_inputs: list[Any]) -> dict[str, Any]:
            return instance._get_params(flat_inputs)

        v2.Transform.make_params = make_params

    for name in ("GaussianNoise", "RGB"):
        if not hasattr(v2, name):
            setattr(v2, name, type(f"_{name}Compatibility", (), {}))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def _verify_installed_platform() -> None:
    version = importlib.metadata.version("stable-worldmodel")
    if version != PINNED_STABLE_WORLDMODEL_VERSION:
        raise RuntimeError(
            "TDWM requires stable-worldmodel=="
            f"{PINNED_STABLE_WORLDMODEL_VERSION}, found {version}."
        )

    imported_path = Path(swm.__file__).resolve()
    if "site-packages" not in imported_path.parts:
        raise RuntimeError(
            "stable_worldmodel must be imported from the installed package; "
            f"found {imported_path}. Remove source-checkout PYTHONPATH entries."
        )


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_repo_root(),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


class SequenceTransform:
    """Normalize image and state-like columns without changing the dataset."""

    def __init__(self, statistics: dict[str, dict[str, list[float]]]) -> None:
        self.statistics = statistics
        self.image_mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        self.image_std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    @staticmethod
    def _expanded_statistic(value: torch.Tensor, width: int) -> torch.Tensor:
        if width % value.numel() != 0:
            raise ValueError(
                f"Cannot broadcast statistic of width {value.numel()} to {width}."
            )
        return value.repeat(width // value.numel())

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        output = dict(sample)
        pixels = output["pixels"].to(torch.float32).div(255.0)
        output["pixels"] = (pixels - self.image_mean) / self.image_std

        for key, values in self.statistics.items():
            tensor = torch.nan_to_num(output[key].to(torch.float32), nan=0.0)
            mean = self._expanded_statistic(
                torch.tensor(values["mean"], dtype=tensor.dtype), tensor.shape[-1]
            )
            std = self._expanded_statistic(
                torch.tensor(values["std"], dtype=tensor.dtype), tensor.shape[-1]
            )
            output[key] = (tensor - mean) / std
        return output


def _column_statistics(dataset: Any, keys: list[str]) -> dict[str, Any]:
    statistics: dict[str, Any] = {}
    for key in keys:
        values = np.asarray(dataset.get_col_data(key), dtype=np.float32)
        values = values.reshape(-1, values.shape[-1])
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        statistics[key] = {
            "mean": mean.astype(float).tolist(),
            "std": std.astype(float).tolist(),
        }
    return statistics


class LeWMTrainingModule(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        sigreg_weight: float,
        sigreg_knots: int,
        sigreg_projections: int,
        history_size: int,
        num_predictions: int,
        learning_rate: float,
        weight_decay: float,
    ) -> None:
        super().__init__()
        self.model = model
        self.sigreg = swm.wm.SIGReg(
            knots=sigreg_knots,
            num_proj=sigreg_projections,
        )
        self.sigreg_weight = sigreg_weight
        self.history_size = history_size
        self.num_predictions = num_predictions
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        batch["action"] = torch.nan_to_num(batch["action"], nan=0.0)
        output = self.model.encode(batch)
        embeddings = output["emb"]
        action_embeddings = output["act_emb"]

        context = embeddings[:, : self.history_size]
        context_actions = action_embeddings[:, : self.history_size]
        target = embeddings[:, self.num_predictions :]
        prediction = self.model.predict(context, context_actions)

        prediction_loss = (prediction - target).pow(2).mean()
        sigreg_loss = self.sigreg(embeddings.transpose(0, 1))
        loss = prediction_loss + self.sigreg_weight * sigreg_loss
        self.log_dict(
            {
                f"{stage}/loss": loss,
                f"{stage}/prediction_loss": prediction_loss,
                f"{stage}/sigreg_loss": sigreg_loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=stage == "train",
            sync_dist=False,
        )
        return loss

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._shared_step(batch, "validation")

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        warmup_steps = max(1, int(total_steps * 0.01))

        def schedule(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class ExperimentCheckpoint(pl.Callback):
    def __init__(
        self,
        *,
        run_dir: Path,
        run_root: Path,
        run_id: str,
        model_config: dict[str, Any],
        every_n_steps: int,
    ) -> None:
        super().__init__()
        self.run_dir = run_dir
        self.run_root = run_root
        self.run_id = run_id
        self.model_config = model_config
        self.every_n_steps = every_n_steps

    def _save_resume_checkpoint(self, trainer: pl.Trainer) -> None:
        destination = self.run_dir / "last.ckpt"
        temporary = self.run_dir / "last.ckpt.tmp"
        trainer.save_checkpoint(temporary, weights_only=False)
        os.replace(temporary, destination)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: LeWMTrainingModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if (
            trainer.is_global_zero
            and trainer.global_step > 0
            and trainer.global_step % self.every_n_steps == 0
        ):
            self._save_resume_checkpoint(trainer)

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: LeWMTrainingModule
    ) -> None:
        if not trainer.is_global_zero:
            return
        self._save_resume_checkpoint(trainer)
        swm.wm.save_pretrained(
            pl_module.model,
            run_name=self.run_id,
            config=self.model_config,
            filename=f"weights_epoch_{trainer.current_epoch + 1}.pt",
            cache_dir=str(self.run_root),
        )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> None:
    mp.set_start_method("forkserver", force=True)
    _install_torchvision_v2_compatibility()
    _verify_installed_platform()
    if args.env != "pusht" or args.method != "lewm":
        raise NotImplementedError(
            "The first executable TDWM entry supports only --env pusht "
            "--method lewm."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("The configured LeWM experiment requires CUDA.")

    root = _repo_root()
    environment = _load_yaml(root / "configs" / "envs" / f"{args.env}.yaml")
    method = _load_yaml(root / "configs" / "methods" / f"{args.method}.yaml")
    dataset_path = Path(args.dataset).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    torch.set_float32_matmul_precision("high")
    pl.seed_everything(args.seed, workers=True)

    sequence = method["sequence"]
    training = method["training"]
    loader_config = environment["dataset"]["training_loader"]
    dataset = swm.data.load_dataset(
        str(dataset_path),
        frameskip=int(sequence["frameskip"]),
        num_steps=int(sequence["history_size"] + sequence["num_predictions"]),
        keys_to_load=list(loader_config["keys_to_load"]),
        keys_to_cache=list(loader_config.get("keys_to_cache", [])),
        transform=None,
    )
    normalizer_keys = [
        key for key in loader_config["keys_to_load"] if key != "pixels"
    ]
    statistics = _column_statistics(dataset, normalizer_keys)
    dataset.transform = SequenceTransform(statistics)

    split = loader_config["split"]
    train_length = int(len(dataset) * float(split["train"]))
    validation_length = len(dataset) - train_length
    split_generator = torch.Generator().manual_seed(int(split["seed"]))
    train_dataset, validation_dataset = random_split(
        dataset,
        [train_length, validation_length],
        generator=split_generator,
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    worker_count = args.workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=worker_count,
        persistent_workers=worker_count > 0,
        prefetch_factor=2 if worker_count > 0 else None,
        pin_memory=True,
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )

    model_config = deepcopy(method["factory"])
    model_config["action_encoder"]["input_dim"] = int(
        sequence["frameskip"] * dataset.get_dim("action")
    )
    model = instantiate(model_config)
    optimizer = training["optimizer"]
    sigreg = method["loss"]["sigreg"]
    module = LeWMTrainingModule(
        model,
        sigreg_weight=float(sigreg["weight"]),
        sigreg_knots=int(sigreg["integration_knots"]),
        sigreg_projections=int(sigreg["projections"]),
        history_size=int(sequence["history_size"]),
        num_predictions=int(sequence["num_predictions"]),
        learning_rate=float(optimizer["learning_rate"]),
        weight_decay=float(optimizer["weight_decay"]),
    )

    commit = _git_commit()
    run_id = args.run_id or (
        f"{args.method}_{args.env}_seed{args.seed}_"
        f"{training['epochs']}ep_{commit[:8]}"
    )
    run_root = Path(args.run_root).expanduser().resolve()
    run_dir = run_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "run_id": run_id,
        "git_commit": commit,
        "stable_worldmodel_version": importlib.metadata.version(
            "stable-worldmodel"
        ),
        "stable_worldmodel_import": str(Path(swm.__file__).resolve()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "seed": args.seed,
        "dataset": str(dataset_path),
        "dataset_length": len(dataset),
        "train_length": train_length,
        "validation_length": validation_length,
        "normalization": statistics,
        "environment_config": environment,
        "method_config": method,
        "resolved_model_config": model_config,
    }
    _write_json(run_dir / "metadata.json", metadata)

    checkpoint = ExperimentCheckpoint(
        run_dir=run_dir,
        run_root=run_root,
        run_id=run_id,
        model_config=model_config,
        every_n_steps=args.checkpoint_steps,
    )
    logger = CSVLogger(save_dir=str(run_dir), name="metrics")
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=int(training["epochs"]),
        precision=str(training["precision"]),
        gradient_clip_val=float(training["gradient_clip"]),
        callbacks=[checkpoint],
        logger=logger,
        enable_checkpointing=False,
        num_sanity_val_steps=1,
        log_every_n_steps=50,
    )
    resume_checkpoint = run_dir / "last.ckpt"
    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
        ckpt_path=str(resume_checkpoint)
        if args.resume and resume_checkpoint.exists()
        else None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a TDWM baseline experiment.")
    parser.add_argument("--env", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--checkpoint-steps", type=int, default=1000)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())
