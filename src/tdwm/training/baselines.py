from __future__ import annotations

import gc
import importlib.metadata
import math
import os
import platform
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import lightning as pl
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from tdwm.training.data import (
    column_statistics,
    make_episode_split,
    make_episode_view,
)
from tdwm.training.experiment import (
    dataset_signature,
    git_state,
    prepare_run_directory,
)
from tdwm.training.lewm import (
    SequenceTransform,
    _install_torchvision_v2_compatibility,
    _load_yaml,
    _repo_root,
    _verify_installed_platform,
    _write_json,
)

SUPPORTED_METHODS = frozenset({"pldm", "dino_wm", "gcbc", "gcivl", "gciql"})


class GoalSequenceTransform(SequenceTransform):
    """Apply the same image normalization to observations and sampled goals."""

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        output = super().__call__(sample)
        goal = sample["goal_pixels"].to(torch.float32).div(255.0)
        output["goal_pixels"] = (goal - self.image_mean) / self.image_std
        return output


class MappedDataset(Dataset):
    def __init__(self, dataset: Dataset, transform: Any) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.transform(self.dataset[index])

    @property
    def clip_indices(self) -> Any:
        return self.dataset.clip_indices


class BaselineCheckpoint(pl.Callback):
    """Save resumable Lightning state and an upstream-compatible weight export."""

    def __init__(
        self,
        *,
        run_dir: Path,
        run_root: Path,
        run_id: str,
        stage: str,
        model_config: dict[str, Any],
        every_n_steps: int,
    ) -> None:
        super().__init__()
        self.run_dir = run_dir
        self.run_root = run_root
        self.run_id = run_id
        self.stage = stage
        self.model_config = model_config
        self.every_n_steps = every_n_steps

    @property
    def resume_path(self) -> Path:
        return self.run_dir / f"{self.stage}.ckpt"

    def _save_resume_checkpoint(self, trainer: pl.Trainer) -> None:
        temporary = self.resume_path.with_suffix(".ckpt.tmp")
        trainer.save_checkpoint(temporary, weights_only=False)
        os.replace(temporary, self.resume_path)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
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
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if not trainer.is_global_zero:
            return
        self._save_resume_checkpoint(trainer)
        swm.wm.save_pretrained(
            pl_module.model,
            run_name=f"{self.run_id}_{self.stage}",
            config=OmegaConf.create(self.model_config),
            filename=f"weights_epoch_{trainer.current_epoch + 1}.pt",
            cache_dir=str(self.run_root),
        )


class OptimizedModule(pl.LightningModule):
    def __init__(
        self,
        *,
        learning_rate: float,
        weight_decay: float,
        cosine_schedule: bool,
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.cosine_schedule = cosine_schedule

    def optimized_parameters(self) -> Iterable[nn.Parameter]:
        raise NotImplementedError

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.optimized_parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        if not self.cosine_schedule:
            return optimizer

        total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        warmup_steps = max(1, int(total_steps * 0.01))

        def schedule(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = (step - warmup_steps) / max(
                1, total_steps - warmup_steps
            )
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class PLDMTrainingModule(OptimizedModule):
    def __init__(
        self,
        model: nn.Module,
        *,
        loss_config: dict[str, Any],
        history_size: int,
        num_predictions: int,
        learning_rate: float,
        weight_decay: float,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            cosine_schedule=True,
        )
        self.model = model
        self.loss_config = loss_config
        self.history_size = history_size
        self.num_predictions = num_predictions
        self.pldm_loss = swm.wm.PLDMLoss()
        self.temporal_straightening = swm.wm.TemporalStraighteningLoss()
        self.sigreg = swm.wm.SIGReg()

    def optimized_parameters(self) -> Iterable[nn.Parameter]:
        return self.model.parameters()

    def _weight(self, key: str) -> float:
        config = self.loss_config[key]
        return float(config["weight"]) if config.get("enabled", True) else 0.0

    def _shared_step(
        self, batch: dict[str, torch.Tensor], stage: str
    ) -> torch.Tensor:
        batch["action"] = torch.nan_to_num(batch["action"], nan=0.0)
        output = self.model.encode(batch)
        embeddings = output["emb"]
        actions = output["act_emb"]
        prediction = self.model.predict(
            embeddings[:, : self.history_size],
            actions[:, : self.history_size],
        )
        target = embeddings[:, self.num_predictions :]

        losses = self.pldm_loss(embeddings)
        losses["prediction_loss"] = F.mse_loss(prediction, target)
        if self._weight("temporal_straightening"):
            losses["temporal_straightening_loss"] = (
                self.temporal_straightening(embeddings)
            )
        if self._weight("sigreg"):
            losses["sigreg_loss"] = self.sigreg(embeddings.transpose(0, 1))

        mapping = {
            "prediction": "prediction_loss",
            "sigreg": "sigreg_loss",
            "temporal_straightening": "temporal_straightening_loss",
            "variance": "std_loss",
            "temporal_variance": "std_t_loss",
            "covariance": "cov_loss",
            "temporal_covariance": "cov_t_loss",
            "temporal_alignment": "temp_align_loss",
            "inverse_dynamics": "idm_loss",
        }
        total = embeddings.new_zeros(())
        for config_key, loss_key in mapping.items():
            weight = self._weight(config_key)
            if weight and loss_key in losses:
                total = total + weight * losses[loss_key]

        logs = {f"{stage}/{key}": value for key, value in losses.items()}
        logs[f"{stage}/loss"] = total
        self.log_dict(
            logs,
            on_step=True,
            on_epoch=True,
            prog_bar=stage == "train",
            sync_dist=False,
        )
        return total

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._shared_step(batch, "validation")


class PreJEPATrainingModule(OptimizedModule):
    def __init__(
        self,
        model: nn.Module,
        *,
        history_size: int,
        num_predictions: int,
        pixel_embedding_dimension: int,
        learning_rate: float,
        weight_decay: float,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            cosine_schedule=False,
        )
        self.model = model
        self.history_size = history_size
        self.num_predictions = num_predictions
        self.pixel_embedding_dimension = pixel_embedding_dimension

    def optimized_parameters(self) -> Iterable[nn.Parameter]:
        return (parameter for parameter in self.model.parameters() if parameter.requires_grad)

    def _shared_step(
        self, batch: dict[str, torch.Tensor], stage: str
    ) -> torch.Tensor:
        batch["action"] = torch.nan_to_num(batch["action"], nan=0.0)
        output = self.model.encode(batch)
        embeddings = output["emb"]
        prediction = self.model.predict(embeddings[:, : self.history_size])
        target = embeddings[:, self.num_predictions :].detach()
        loss = F.mse_loss(
            prediction[..., : self.pixel_embedding_dimension],
            target[..., : self.pixel_embedding_dimension],
        )
        self.log(
            f"{stage}/loss",
            loss,
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


class GoalPolicyTrainingModule(OptimizedModule):
    def __init__(
        self,
        model: nn.Module,
        *,
        mode: str,
        history_size: int,
        td_offset: int,
        discount: float,
        expectile: float,
        advantage_temperature: float,
        learning_rate: float,
        weight_decay: float,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            cosine_schedule=False,
        )
        self.model = model
        self.mode = mode
        self.history_size = history_size
        self.td_offset = td_offset
        self.discount = discount
        self.advantage_temperature = advantage_temperature

        from stable_worldmodel.wm.gcrl.module import ExpectileLoss

        self.expectile_loss = ExpectileLoss(tau=expectile)

    def optimized_parameters(self) -> Iterable[nn.Parameter]:
        if self.mode == "gcbc":
            return self.model.action_predictor.parameters()
        if self.mode == "gcivl_value":
            return self.model.value_predictor.student.parameters()
        if self.mode == "gciql_critics":
            return list(self.model.value_predictor.student.parameters()) + list(
                self.model.critic_predictor.student.parameters()
            )
        if self.mode in {"gcivl_actor", "gciql_actor"}:
            return list(self.model.action_predictor.parameters()) + [
                self.model.log_stds
            ]
        raise ValueError(f"Unknown training mode: {self.mode}")

    def _encode(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch["action"] = torch.nan_to_num(batch["action"], nan=0.0)
        output = self.model.encode(batch, pixels_key="pixels", target="embed")
        output = self.model.encode(
            output,
            pixels_key="goal_pixels",
            emb_keys=[],
            prefix="goal_",
            target="goal_embed",
        )
        embedding = output["embed"][:, : self.history_size]
        target_embedding = output["embed"][:, self.td_offset :]
        goal_embedding = output["goal_embed"]
        return embedding, target_embedding, goal_embedding

    @staticmethod
    def _flatten(embedding: torch.Tensor) -> torch.Tensor:
        return embedding.flatten(1, 2)

    def _goal_reward(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observations = batch["pixels"][:, : self.history_size]
        goal = batch["goal_pixels"].expand(-1, observations.shape[1], -1, -1, -1)
        matches = (observations == goal).all(dim=(2, 3, 4))
        masks = (~matches).to(torch.float32).unsqueeze(-1)
        return -masks, masks

    def _gcbc_step(
        self,
        batch: dict[str, torch.Tensor],
        embedding: torch.Tensor,
        goal_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        action_prediction, _ = self.model.predict_actions(
            embedding, goal_embedding
        )
        target = batch["action"][:, : self.history_size]
        loss = F.mse_loss(action_prediction, target)
        return loss, {"behavior_cloning_loss": loss}

    def _critic_step(
        self,
        batch: dict[str, torch.Tensor],
        embedding: torch.Tensor,
        target_embedding: torch.Tensor,
        goal_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        state = self._flatten(embedding.detach())
        next_state = self._flatten(target_embedding.detach())
        goal = self._flatten(goal_embedding.detach())
        reward, masks = self._goal_reward(batch)

        if self.mode == "gcivl_value":
            value = self.model.value_predictor.forward_student(state, goal)
            with torch.no_grad():
                next_value = self.model.value_predictor.forward_teacher(
                    next_state, goal
                )
                target = reward + self.discount * masks * next_value
            value_loss = self.expectile_loss(value, target)
            return value_loss, {"value_loss": value_loss}

        actions = batch["action"][:, : self.history_size]
        with torch.no_grad():
            q_value = self.model.critic_predictor.forward_teacher(
                state, actions, goal
            )
            next_value = self.model.value_predictor.forward_teacher(
                next_state, goal
            )
            q_target = reward + self.discount * masks * next_value
        value = self.model.value_predictor.forward_student(state, goal)
        q_prediction = self.model.critic_predictor.forward_student(
            state, actions, goal
        )
        value_loss = self.expectile_loss(value, q_value)
        critic_loss = F.mse_loss(q_prediction, q_target)
        return value_loss + critic_loss, {
            "value_loss": value_loss,
            "critic_loss": critic_loss,
        }

    def _actor_step(
        self,
        batch: dict[str, torch.Tensor],
        embedding: torch.Tensor,
        target_embedding: torch.Tensor,
        goal_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with torch.no_grad():
            state = self._flatten(embedding)
            next_state = self._flatten(target_embedding)
            goal = self._flatten(goal_embedding)
            value = self.model.value_predictor(state, goal)
            next_value = self.model.value_predictor(next_state, goal)
            advantage = next_value - value

        means, _ = self.model.predict_actions(
            embedding.detach(), goal_embedding.detach()
        )
        target = batch["action"][:, : self.history_size]
        log_stds = torch.clamp(
            self.model.log_stds,
            self.model.log_std_min,
            self.model.log_std_max,
        )
        variance = torch.exp(2 * log_stds)
        negative_log_likelihood = log_stds + 0.5 * (target - means).pow(2) / variance
        advantage_weight = torch.exp(
            advantage.detach() * self.advantage_temperature
        ).clamp(max=100.0)
        loss = (advantage_weight * negative_log_likelihood).mean()
        return loss, {
            "awr_loss": loss,
            "advantage_mean": advantage.mean(),
            "advantage_weight_mean": advantage_weight.mean(),
        }

    def _shared_step(
        self, batch: dict[str, torch.Tensor], stage: str
    ) -> torch.Tensor:
        if self.mode in {"gcivl_actor", "gciql_actor"}:
            with torch.no_grad():
                embedding, target_embedding, goal_embedding = self._encode(batch)
        else:
            embedding, target_embedding, goal_embedding = self._encode(batch)

        if self.mode == "gcbc":
            loss, metrics = self._gcbc_step(batch, embedding, goal_embedding)
        elif self.mode in {"gcivl_value", "gciql_critics"}:
            loss, metrics = self._critic_step(
                batch, embedding, target_embedding, goal_embedding
            )
        else:
            loss, metrics = self._actor_step(
                batch, embedding, target_embedding, goal_embedding
            )

        self.log_dict(
            {f"{stage}/{key}": value for key, value in {"loss": loss, **metrics}.items()},
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

    def on_train_epoch_start(self) -> None:
        if self.mode in {"gcivl_value", "gciql_critics"}:
            total_epochs = int(self.trainer.max_epochs)
            self.model.value_predictor.update_ema_coefficient(
                self.current_epoch, total_epochs
            )
            if self.mode == "gciql_critics":
                self.model.critic_predictor.update_ema_coefficient(
                    self.current_epoch, total_epochs
                )

    def on_train_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        if self.mode in {"gcivl_value", "gciql_critics"}:
            self.model.value_predictor.update_teacher()
            if self.mode == "gciql_critics":
                self.model.critic_predictor.update_teacher()


def _make_loaders(
    args: Any,
    environment: dict[str, Any],
    method: dict[str, Any],
    *,
    goal_probabilities: tuple[float, float, float, float] | None = None,
) -> tuple[Any, DataLoader, DataLoader, dict[str, Any], int, dict[str, Any]]:
    sequence = method["sequence"]
    loader_config = environment["dataset"]["training_loader"]
    dataset = swm.data.load_dataset(
        str(Path(args.dataset).expanduser().resolve()),
        frameskip=int(sequence["frameskip"]),
        num_steps=int(sequence["num_steps"]),
        keys_to_load=["pixels", "action"],
        keys_to_cache=["action"],
        transform=None,
    )
    split_config = loader_config["split"]
    episode_split = make_episode_split(
        dataset.clip_indices,
        train_fraction=float(split_config["train"]),
        seed=int(split_config["seed"]),
    )
    statistics = column_statistics(
        dataset, ["action"], episode_split.train_episodes
    )
    effective_action_dimension = int(
        int(sequence["frameskip"]) * dataset.get_dim("action")
    )

    if goal_probabilities is None:
        dataset.transform = SequenceTransform(statistics)
        transformed: Dataset = dataset
        train_indices, validation_indices = episode_split.indices_for(
            transformed.clip_indices
        )
        train_dataset = Subset(transformed, train_indices)
        validation_dataset = Subset(transformed, validation_indices)
    else:
        def goal_dataset_for(
            episodes: tuple[int, ...], seed: int
        ) -> MappedDataset:
            episode_dataset = make_episode_view(dataset, episodes)
            goal_dataset = swm.data.GoalDataset(
                dataset=episode_dataset,
                goal_probabilities=goal_probabilities,
                gamma=float(method.get("rl", {}).get("goal_gamma", 0.99)),
                current_goal_offset=int(sequence["history_size"]),
                goal_keys={"pixels": "goal_pixels"},
                seed=seed,
            )
            return MappedDataset(
                goal_dataset,
                GoalSequenceTransform(statistics),
            )

        train_dataset = goal_dataset_for(episode_split.train_episodes, args.seed)
        validation_dataset = goal_dataset_for(
            episode_split.validation_episodes,
            args.seed + 1,
        )
        transformed = dataset
    train_length = len(train_dataset)
    validation_length = len(validation_dataset)
    loader_generator = torch.Generator().manual_seed(args.seed)
    worker_count = int(args.workers)
    batch_size = int(method["training"]["batch_size"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
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
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )
    dataset_info = {
        "dataset_length": len(transformed),
        "train_length": train_length,
        "validation_length": validation_length,
    }
    return (
        dataset_info,
        train_loader,
        validation_loader,
        statistics,
        effective_action_dimension,
        episode_split.as_dict(),
    )


def _build_dino_encoder(
    backbone_source: str | None = None,
) -> tuple[nn.Module, int]:
    import stable_pretraining as spt
    from stable_worldmodel.wm.prejepa.module import create_backbone

    backbone_source = backbone_source or os.environ.get(
        "TDWM_DINO_BACKBONE", "dinov2_small"
    )
    encoder = create_backbone(backbone_source)
    embedding_dimension = int(encoder.config.hidden_size)
    encoder.requires_grad_(False)
    return spt.backbone.EvalOnly(encoder), embedding_dimension


def _build_prejepa(
    method: dict[str, Any],
    effective_action_dimension: int,
    backbone_source: str | None = None,
) -> tuple[nn.Module, int]:
    from stable_worldmodel.wm.prejepa.module import CausalPredictor, Embedder

    encoder, pixel_dimension = _build_dino_encoder(backbone_source)
    sequence = method["sequence"]
    predictor_config = method["model"]["predictor"]
    action_embedding_dimension = int(
        method["model"]["action_encoder"]["embedding_dimension"]
    )
    predictor = CausalPredictor(
        num_patches=(int(method["input"]["image_size"]) // 14) ** 2,
        num_frames=int(sequence["history_size"]),
        dim=pixel_dimension + action_embedding_dimension,
        depth=int(predictor_config["depth"]),
        heads=int(predictor_config["heads"]),
        mlp_dim=int(predictor_config["mlp_dimension"]),
        dim_head=int(predictor_config["head_dimension"]),
        dropout=float(predictor_config["dropout"]),
    )
    action_encoder = Embedder(
        in_chans=effective_action_dimension,
        emb_dim=action_embedding_dimension,
    )
    model = swm.wm.PreJEPA(
        encoder=encoder,
        predictor=predictor,
        extra_encoders=nn.ModuleDict({"action": action_encoder}),
        history_size=int(sequence["history_size"]),
        num_pred=int(sequence["num_predictions"]),
        interpolate_pos_encoding=False,
    )
    return model, pixel_dimension


def _build_gcrl(
    method: dict[str, Any],
    effective_action_dimension: int,
    backbone_source: str | None = None,
) -> nn.Module:
    import stable_pretraining as spt
    from stable_worldmodel.wm.gcrl.module import Predictor, QPredictor

    encoder, embedding_dimension = _build_dino_encoder(backbone_source)
    sequence = method["sequence"]
    predictor_config = method["model"].get(
        "predictor", method["model"].get("action_predictor")
    )
    common = {
        "num_patches": (int(method["input"]["image_size"]) // 14) ** 2,
        "num_frames": int(sequence["history_size"]),
        "dim": embedding_dimension,
        "depth": int(predictor_config["depth"]),
        "heads": int(predictor_config["heads"]),
        "mlp_dim": int(predictor_config["mlp_dimension"]),
        "dim_head": int(predictor_config["head_dimension"]),
        "dropout": float(predictor_config["dropout"]),
    }
    action_predictor = Predictor(
        **common,
        out_dim=effective_action_dimension,
    )
    value_predictor = None
    critic_predictor = None
    if method["id"] in {"gcivl", "gciql"}:
        value = Predictor(**common, out_dim=1, pool_type="mean")
        tau = float(method["rl"]["value_ema_tau"])
        value_predictor = spt.TeacherStudentWrapper(
            value,
            warm_init=True,
            base_ema_coefficient=tau,
            final_ema_coefficient=tau,
        )
    if method["id"] == "gciql":
        critic = QPredictor(
            **common,
            action_dim=effective_action_dimension,
            pool_type="mean",
        )
        tau = float(method["rl"]["value_ema_tau"])
        critic_predictor = spt.TeacherStudentWrapper(
            critic,
            warm_init=True,
            base_ema_coefficient=tau,
            final_ema_coefficient=tau,
        )
    return swm.wm.GCRL(
        encoder=encoder,
        action_predictor=action_predictor,
        value_predictor=value_predictor,
        critic_predictor=critic_predictor,
        history_size=int(sequence["history_size"]),
        interpolate_pos_encoding=False,
    )


def build_baseline_model(
    method_config: dict[str, Any],
    effective_action_dimension: int,
    backbone_source: str | None = None,
) -> nn.Module:
    """Build an exported baseline model from a self-contained public config."""
    method_id = str(method_config["id"])
    if method_id == "pldm":
        model_config = deepcopy(method_config["factory"])
        model_config["action_encoder"]["input_dim"] = int(
            effective_action_dimension
        )
        return instantiate(model_config)
    if method_id == "dino_wm":
        model, _ = _build_prejepa(
            method_config,
            int(effective_action_dimension),
            backbone_source,
        )
        return model
    if method_id in {"gcbc", "gcivl", "gciql"}:
        return _build_gcrl(
            method_config,
            int(effective_action_dimension),
            backbone_source,
        )
    raise ValueError(f"Unsupported baseline model: {method_id}")


def _export_model_config(
    method: dict[str, Any],
    effective_action_dimension: int,
    backbone_source: str | None,
) -> dict[str, Any]:
    return {
        "_target_": "tdwm.training.baselines.build_baseline_model",
        "_recursive_": False,
        "method_config": method,
        "effective_action_dimension": int(effective_action_dimension),
        "backbone_source": backbone_source,
    }


def _goal_probabilities(config: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(config["random"]),
        float(config["geometric_future"]),
        float(config["uniform_future"]),
        float(config["current"]),
    )


def _checkpoint_is_complete(
    module: pl.LightningModule, path: Path, max_epochs: int
) -> bool:
    if not path.exists():
        return False
    checkpoint = torch.load(path, map_location="cpu")
    if int(checkpoint.get("epoch", -1)) + 1 < max_epochs:
        return False
    module.load_state_dict(checkpoint["state_dict"])
    return True


def _fit_stage(
    *,
    module: pl.LightningModule,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    method: dict[str, Any],
    model_config: dict[str, Any],
    run_dir: Path,
    run_root: Path,
    run_id: str,
    stage: str,
    checkpoint_steps: int,
    resume: bool,
) -> None:
    training = method["training"]
    max_epochs = int(training["epochs"])
    checkpoint = BaselineCheckpoint(
        run_dir=run_dir,
        run_root=run_root,
        run_id=run_id,
        stage=stage,
        model_config=model_config,
        every_n_steps=checkpoint_steps,
    )
    if resume and _checkpoint_is_complete(
        module, checkpoint.resume_path, max_epochs
    ):
        return

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=max_epochs,
        precision=str(training["precision"]),
        gradient_clip_val=float(training.get("gradient_clip", 0.0)),
        callbacks=[checkpoint],
        logger=CSVLogger(save_dir=str(run_dir), name=f"metrics_{stage}"),
        enable_checkpointing=False,
        num_sanity_val_steps=1,
        log_every_n_steps=50,
    )
    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
        ckpt_path=(
            str(checkpoint.resume_path)
            if resume and checkpoint.resume_path.exists()
            else None
        ),
    )


def _write_metadata(
    *,
    args: Any,
    run_dir: Path,
    run_id: str,
    environment: dict[str, Any],
    method: dict[str, Any],
    dataset_info: dict[str, Any],
    statistics: dict[str, Any],
    effective_action_dimension: int,
    experiment_fingerprint: str,
    repository_state: dict[str, Any],
    data_signature: dict[str, Any],
    split: dict[str, Any],
    model_config: dict[str, Any],
    backbone_source: str | None,
) -> None:
    metadata = {
        "run_id": run_id,
        "experiment_fingerprint": experiment_fingerprint,
        "git": repository_state,
        "git_commit": repository_state["commit"],
        "stable_worldmodel_version": importlib.metadata.version(
            "stable-worldmodel"
        ),
        "stable_worldmodel_import": str(Path(swm.__file__).resolve()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "seed": args.seed,
        "dataset": str(Path(args.dataset).expanduser().resolve()),
        "dataset_signature": data_signature,
        **dataset_info,
        "split": split,
        "normalization": statistics,
        "effective_action_dimension": effective_action_dimension,
        "dino_backbone_source": backbone_source,
        "environment_config": environment,
        "method_config": method,
        "resolved_model_config": model_config,
    }
    _write_json(run_dir / "metadata.json", metadata)


def run(args: Any) -> None:
    _install_torchvision_v2_compatibility()
    _verify_installed_platform()
    import stable_pretraining as spt

    if args.env != "pusht" or args.method not in SUPPORTED_METHODS:
        raise NotImplementedError(
            "The offline baseline adapter supports PushT with one of: "
            + ", ".join(sorted(SUPPORTED_METHODS))
        )
    if not torch.cuda.is_available():
        raise RuntimeError("The configured PushT baseline requires CUDA.")

    run_root = Path(args.run_root).expanduser().resolve()
    spt.set(
        cache_dir=str(run_root / "spt"),
        default_callbacks={"unused_params": False},
    )
    dataset_path = Path(args.dataset).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    root = _repo_root()
    environment = _load_yaml(root / "configs" / "envs" / "pusht.yaml")
    method = _load_yaml(root / "configs" / "methods" / f"{args.method}.yaml")
    training = method["training"]
    optimizer = training["optimizer"]
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(args.seed, workers=True)

    if args.method in {"pldm", "dino_wm"}:
        dataset_info, train, validation, statistics, action_dimension, split = (
            _make_loaders(args, environment, method)
        )
    else:
        sampling = (
            method["goal_sampling"]
            if args.method == "gcbc"
            else method["rl"]["value_goal_sampling"]
        )
        dataset_info, train, validation, statistics, action_dimension, split = (
            _make_loaders(
                args,
                environment,
                method,
                goal_probabilities=_goal_probabilities(sampling),
            )
        )

    backbone_source = (
        None
        if args.method == "pldm"
        else os.environ.get("TDWM_DINO_BACKBONE", "dinov2_small")
    )
    model_config = _export_model_config(
        method, action_dimension, backbone_source
    )
    repository_state = git_state(root)
    data_signature = dataset_signature(dataset_path, args.dataset_sha256)
    identity = {
        "git": repository_state,
        "environment_config": environment,
        "method_config": method,
        "model_config": model_config,
        "seed": args.seed,
        "dataset": data_signature,
        "split": split,
    }
    run_id, run_dir, experiment_fingerprint = prepare_run_directory(
        run_root=run_root,
        requested_run_id=args.run_id,
        method=args.method,
        environment=args.env,
        seed=args.seed,
        identity=identity,
    )
    _write_metadata(
        args=args,
        run_dir=run_dir,
        run_id=run_id,
        environment=environment,
        method=method,
        dataset_info=dataset_info,
        statistics=statistics,
        effective_action_dimension=action_dimension,
        experiment_fingerprint=experiment_fingerprint,
        repository_state=repository_state,
        data_signature=data_signature,
        split=split,
        model_config=model_config,
        backbone_source=backbone_source,
    )

    if args.method == "pldm":
        model = build_baseline_model(method, action_dimension)
        module: pl.LightningModule = PLDMTrainingModule(
            model,
            loss_config=method["loss"],
            history_size=int(method["sequence"]["history_size"]),
            num_predictions=int(method["sequence"]["num_predictions"]),
            learning_rate=float(optimizer["learning_rate"]),
            weight_decay=float(optimizer["weight_decay"]),
        )
        stages = [("train", module, train, validation)]
    elif args.method == "dino_wm":
        model, pixel_dimension = _build_prejepa(
            method, action_dimension, backbone_source
        )
        module = PreJEPATrainingModule(
            model,
            history_size=int(method["sequence"]["history_size"]),
            num_predictions=int(method["sequence"]["num_predictions"]),
            pixel_embedding_dimension=pixel_dimension,
            learning_rate=float(optimizer["learning_rate"]),
            weight_decay=float(optimizer["weight_decay"]),
        )
        stages = [("train", module, train, validation)]
    else:
        model = _build_gcrl(method, action_dimension, backbone_source)
        if args.method == "gcbc":
            module = GoalPolicyTrainingModule(
                model,
                mode="gcbc",
                history_size=int(method["sequence"]["history_size"]),
                td_offset=1,
                discount=0.99,
                expectile=0.9,
                advantage_temperature=1.0,
                learning_rate=float(optimizer["predictor_learning_rate"]),
                weight_decay=0.0,
            )
            stages = [("policy", module, train, validation)]
        else:
            first_mode = (
                "gcivl_value" if args.method == "gcivl" else "gciql_critics"
            )
            first_stage = "value" if args.method == "gcivl" else "value_and_q"
            module = GoalPolicyTrainingModule(
                model,
                mode=first_mode,
                history_size=int(method["sequence"]["history_size"]),
                td_offset=int(method["sequence"]["td_offset"]),
                discount=float(method["rl"]["discount"]),
                expectile=float(method["rl"]["expectile"]),
                advantage_temperature=float(
                    method["rl"]["advantage_weight_temperature"]
                ),
                learning_rate=float(optimizer["predictor_learning_rate"]),
                weight_decay=0.0,
            )
            stages = [(first_stage, module, train, validation)]

    for stage, stage_module, stage_train, stage_validation in stages:
        _fit_stage(
            module=stage_module,
            train_loader=stage_train,
            validation_loader=stage_validation,
            method=method,
            model_config=model_config,
            run_dir=run_dir,
            run_root=run_root,
            run_id=run_id,
            stage=stage,
            checkpoint_steps=int(args.checkpoint_steps),
            resume=bool(args.resume),
        )

    if args.method not in {"gcivl", "gciql"}:
        return

    # Lightning keeps the completed trainer (and its optimizer state) on the
    # module. The policy stage reuses only the trained model, so release that
    # stage-local state before allocating the actor optimizer.
    for _, completed_module, _, _ in stages:
        completed_module._trainer = None
    del stages, module, stage_module, stage_train, stage_validation
    del train, validation
    gc.collect()
    torch.cuda.empty_cache()
    sampling = method["rl"]["actor_goal_sampling"]
    _, actor_train, actor_validation, _, _, _ = _make_loaders(
        args,
        environment,
        method,
        goal_probabilities=_goal_probabilities(sampling),
    )
    actor = GoalPolicyTrainingModule(
        model,
        mode=f"{args.method}_actor",
        history_size=int(method["sequence"]["history_size"]),
        td_offset=int(method["sequence"]["td_offset"]),
        discount=float(method["rl"]["discount"]),
        expectile=float(method["rl"]["expectile"]),
        advantage_temperature=float(
            method["rl"]["advantage_weight_temperature"]
        ),
        learning_rate=float(optimizer["predictor_learning_rate"]),
        weight_decay=0.0,
    )
    _fit_stage(
        module=actor,
        train_loader=actor_train,
        validation_loader=actor_validation,
        method=method,
        model_config=model_config,
        run_dir=run_dir,
        run_root=run_root,
        run_id=run_id,
        stage="policy",
        checkpoint_steps=int(args.checkpoint_steps),
        resume=bool(args.resume),
    )
