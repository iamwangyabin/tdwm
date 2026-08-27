"""Joint model-aware TD goal-tail training for LeWM on OGBench Cube."""

from __future__ import annotations

import copy
import importlib.metadata
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml

from tdwm.adapters import prepare_cloud_runtime
from tdwm.methods.goal_tail_value import (
    GoalTailValue,
    latent_goal_cost,
    monte_carlo_goal_tail_targets,
)
from tdwm.training.cube_data import validate_cube_training_dataset
from tdwm.training.gt_lewm_support import write_json
from tdwm.training.mc_gt_lewm import (
    _append_json_line,
    _git_revision,
    _resolve_base_checkpoint,
    _save_checkpoint,
    _sha256,
    ensure_cube_latent_cache,
)
from tdwm.training.td_gt_lewm import ema_update_target
from tdwm.training.train_goal_tail_cube import (
    spearman_correlation,
    split_clip_indices_by_episode,
)


class CachedCubeJointTDDataset(torch.utils.data.Dataset):
    """Read the latent and action span used by model-aware terminal TD."""

    def __init__(
        self,
        *,
        latent_cache_path: str | Path,
        normalized_actions: np.ndarray,
        clip_indices: Sequence[tuple[int, int]],
        episode_offsets: Sequence[int],
        source_indices: Sequence[int],
        frame_skip: int,
        num_steps: int,
        action_blocks: int,
    ) -> None:
        if frame_skip <= 0 or num_steps <= 0 or action_blocks <= 0:
            raise ValueError("frame_skip, num_steps, and action_blocks must be positive.")
        if normalized_actions.ndim != 2:
            raise ValueError("normalized_actions must have shape (time, action_dim).")
        self.latent_cache_path = str(Path(latent_cache_path).resolve())
        self.normalized_actions = np.asarray(normalized_actions, dtype=np.float32)
        offsets = np.asarray(episode_offsets, dtype=np.int64)
        selected_indices = np.asarray(source_indices, dtype=np.int64)
        self.global_starts = np.fromiter(
            (
                int(offsets[int(clip_indices[int(index)][0])])
                + int(clip_indices[int(index)][1])
                for index in selected_indices
            ),
            dtype=np.int64,
            count=int(selected_indices.size),
        )
        self.frame_offsets = np.arange(num_steps, dtype=np.int64) * frame_skip
        self.action_offsets = np.arange(action_blocks * frame_skip, dtype=np.int64)
        self.action_blocks = int(action_blocks)
        self._latents: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.global_starts.size)

    def __getitem__(self, position: int) -> dict[str, torch.Tensor]:
        return self.__getitems__([position])[0]

    def __getitems__(self, positions: list[int]) -> list[dict[str, torch.Tensor]]:
        starts = self.global_starts[np.asarray(positions, dtype=np.int64)]
        latent_rows = starts[:, None] + self.frame_offsets[None, :]
        latents = np.array(self._latent_array[latent_rows], copy=True)
        action_rows = starts[:, None] + self.action_offsets[None, :]
        actions = np.array(self.normalized_actions[action_rows], copy=True).reshape(
            len(positions), self.action_blocks, -1
        )
        return [
            {
                "latents": torch.from_numpy(latents[index]),
                "action_blocks": torch.from_numpy(actions[index]),
            }
            for index in range(len(positions))
        ]

    @property
    def _latent_array(self) -> np.ndarray:
        if self._latents is None:
            self._latents = np.load(self.latent_cache_path, mmap_mode="r")
        return self._latents

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_latents"] = None
        return state


def load_joint_td_gt_protocol(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as stream:
        protocol = yaml.safe_load(stream)
    validate_joint_td_gt_protocol(protocol)
    return protocol


def validate_joint_td_gt_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema_version") != 1:
        raise ValueError("Joint TD-GT-LeWM requires schema_version 1.")
    if protocol.get("method") != "joint_td_gt_lewm":
        raise ValueError("This trainer only accepts method joint_td_gt_lewm.")
    if protocol.get("environment") != "cube" or protocol.get("stage") != "full_training":
        raise ValueError("Joint TD-GT-LeWM full training is locked to OGBench Cube.")
    if protocol.get("runtime", {}).get("stable_worldmodel_version") != "0.1.1":
        raise ValueError("Joint TD-GT-LeWM is locked to stable-worldmodel 0.1.1.")
    if not protocol.get("seeds"):
        raise ValueError("At least one training seed is required.")

    base = protocol.get("base_model", {})
    if base.get("method") != "lewm" or base.get("initialization") != "checkpoint":
        raise ValueError("Joint training must initialize from the reproduced LeWM.")
    if base.get("history_size") != 3 or base.get("latent_dim") != 192:
        raise ValueError("Joint training requires LeWM history 3 and latent dim 192.")
    if base.get("frozen_components") != ["encoder", "projector"]:
        raise ValueError("V0 freezes exactly the LeWM encoder and projector.")
    if base.get("trainable_components") != [
        "predictor",
        "action_encoder",
        "pred_proj",
    ]:
        raise ValueError("V0 must update exactly the LeWM dynamics components.")
    if base.get("frozen_running_statistics") != ["pred_proj"]:
        raise ValueError("V0 must preserve the reproduced pred_proj statistics.")

    sequence = protocol.get("sequence", {})
    history = sequence.get("history_frames")
    rollout = sequence.get("model_rollout_horizon")
    max_offset = sequence.get("max_goal_offset")
    if (history, rollout, max_offset) != (3, 5, 16):
        raise ValueError("V0 locks history 3, CEM-matched rollout 5, and goals 1..16.")
    if sequence.get("num_steps") != history + rollout + max_offset:
        raise ValueError("num_steps must cover history, rollout, and future goals.")
    if sequence.get("frame_skip", 0) <= 0:
        raise ValueError("sequence.frame_skip must be positive.")

    objective = protocol.get("joint_objective", {})
    if objective.get("prediction") != "teacher_forced_lewm_mse":
        raise ValueError("The original LeWM prediction MSE must be retained.")
    if objective.get("tail_input") != "predicted_terminal_history":
        raise ValueError("The tail must train on a model-predicted terminal history.")
    if objective.get("backpropagate_tail_through_rollout") is not True:
        raise ValueError("The tail loss must update LeWM through its rollout.")
    if objective.get("prediction_weight") != 1.0:
        raise ValueError("V0 keeps the LeWM prediction loss weight at one.")
    if objective.get("tail_weight", -1.0) <= 0.0:
        raise ValueError("joint_objective.tail_weight must be positive.")

    tail = protocol.get("tail_value", {})
    if tail.get("objective") != "one_step_td" or tail.get("hidden_dim", 0) <= 0:
        raise ValueError("Joint TD-GT-LeWM requires a positive-capacity TD value.")
    if not 0.0 <= tail.get("gamma", -1.0) < 1.0:
        raise ValueError("tail_value.gamma must lie in [0, 1).")
    if tail.get("target_network") is not True:
        raise ValueError("Joint TD-GT-LeWM requires an EMA target value.")
    if not 0.0 <= tail.get("target_ema_decay", -1.0) < 1.0:
        raise ValueError("target_ema_decay must lie in [0, 1).")
    if tail.get("terminate_bootstrap_at_goal") is not True:
        raise ValueError("Hindsight goals must terminate TD bootstrapping.")

    split = protocol.get("split", {})
    if split.get("unit") != "episode" or not np.isclose(
        split.get("train_fraction", 0.0) + split.get("validation_fraction", 0.0),
        1.0,
    ):
        raise ValueError("Joint training requires a complete episode-level split.")
    loader = protocol.get("loader", {})
    if loader.get("batch_size", 0) <= 0 or loader.get("validation_batch_size", 0) <= 0:
        raise ValueError("Joint loader batch sizes must be positive.")
    if loader.get("workers", -1) < 0:
        raise ValueError("Joint loader workers cannot be negative.")
    training = protocol.get("training", {})
    if training.get("epochs", 0) <= 0:
        raise ValueError("training.epochs must be positive.")
    if training.get("coverage") != "all_training_clips_each_epoch":
        raise ValueError("Formal joint training must cover all training clips.")
    if training.get("checkpoint_selection") != "minimum_validation_joint_loss":
        raise ValueError("Joint checkpoints must be selected by validation joint loss.")


def build_history_at(
    latents: torch.Tensor,
    action_blocks: torch.Tensor,
    *,
    current_index: int,
    history_size: int,
) -> torch.Tensor:
    """Flatten the LeWM latent/action history ending at ``current_index``."""

    if latents.ndim != 3 or action_blocks.ndim != 3:
        raise ValueError("latents and action_blocks must be rank-three tensors.")
    if latents.shape[0] != action_blocks.shape[0]:
        raise ValueError("latents and action_blocks must share their batch size.")
    first = current_index - history_size + 1
    if first < 0 or current_index >= latents.shape[1]:
        raise ValueError("current_index does not contain the requested history.")
    if current_index > action_blocks.shape[1]:
        raise ValueError("action_blocks do not contain the requested history.")
    return torch.cat(
        (
            latents[:, first : current_index + 1].flatten(1),
            action_blocks[:, first:current_index].flatten(1),
        ),
        dim=-1,
    )


def teacher_forced_windows(
    latents: torch.Tensor,
    action_blocks: torch.Tensor,
    *,
    history_size: int,
    rollout_horizon: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build all local LeWM windows along the planned rollout horizon."""

    if latents.shape[1] < history_size + rollout_horizon:
        raise ValueError("latents do not cover every teacher-forced window.")
    if action_blocks.shape[1] < history_size + rollout_horizon - 1:
        raise ValueError("action_blocks do not cover every teacher-forced window.")
    histories = torch.cat(
        [latents[:, start : start + history_size] for start in range(rollout_horizon)],
        dim=0,
    )
    actions = torch.cat(
        [
            action_blocks[:, start : start + history_size]
            for start in range(rollout_horizon)
        ],
        dim=0,
    )
    targets = torch.cat(
        [
            latents[:, start + 1 : start + history_size + 1]
            for start in range(rollout_horizon)
        ],
        dim=0,
    )
    return histories, actions, targets


def rollout_from_latents(
    world_model: torch.nn.Module,
    initial_latents: torch.Tensor,
    action_blocks: torch.Tensor,
    *,
    history_size: int,
) -> torch.Tensor:
    """Call LeWM's public rollout API while reusing cached initial embeddings."""

    if initial_latents.ndim != 3 or initial_latents.shape[1] != history_size:
        raise ValueError("initial_latents must contain exactly one LeWM history.")
    if action_blocks.ndim != 3 or action_blocks.shape[0] != initial_latents.shape[0]:
        raise ValueError("action_blocks must be batched with initial_latents.")
    if action_blocks.shape[1] < history_size:
        raise ValueError("action_blocks must include the initial LeWM action window.")
    batch = initial_latents.shape[0]
    info = {
        # stable-worldmodel 0.1.1 uses this tensor only to read the history length
        # when an embedding is already cached.
        "pixels": initial_latents.new_empty(batch, 1, history_size, 1),
        "emb": initial_latents.unsqueeze(1),
    }
    output = world_model.rollout(
        info,
        action_blocks.unsqueeze(1),
        history_size=history_size,
    )
    return output["predicted_emb"][:, 0]


@dataclass(frozen=True)
class JointTDBatch:
    predicted_rollout: torch.Tensor
    predicted_history: torch.Tensor
    goals: torch.Tensor
    targets: torch.Tensor
    continuation: torch.Tensor


def build_joint_td_batch(
    world_model: torch.nn.Module,
    target_value: GoalTailValue,
    latents: torch.Tensor,
    action_blocks: torch.Tensor,
    goal_offsets: torch.Tensor,
    *,
    history_size: int,
    rollout_horizon: int,
    gamma: float,
) -> JointTDBatch:
    """Roll out LeWM, then build a real-continuation TD target at its terminal."""

    if goal_offsets.ndim != 1 or goal_offsets.shape[0] != latents.shape[0]:
        raise ValueError("goal_offsets must contain one offset per batch element.")
    if goal_offsets.dtype not in (torch.int32, torch.int64):
        raise TypeError("goal_offsets must use an integer dtype.")
    if torch.any(goal_offsets < 1):
        raise ValueError("goal_offsets must be positive.")
    terminal_index = history_size + rollout_horizon - 1
    if torch.any(terminal_index + goal_offsets >= latents.shape[1]):
        raise ValueError("latents do not contain every requested future goal.")
    rollout_actions = action_blocks[:, :terminal_index]
    predicted_rollout = rollout_from_latents(
        world_model,
        latents[:, :history_size],
        rollout_actions,
        history_size=history_size,
    )
    predicted_history = build_history_at(
        predicted_rollout,
        action_blocks,
        current_index=terminal_index,
        history_size=history_size,
    )
    rows = torch.arange(latents.shape[0], device=latents.device)
    goals = latents[rows, terminal_index + goal_offsets]
    continuation = (goal_offsets > 1).to(latents.dtype)
    with torch.no_grad():
        next_history = build_history_at(
            latents,
            action_blocks,
            current_index=terminal_index + 1,
            history_size=history_size,
        )
        immediate = (1.0 - gamma) * latent_goal_cost(
            latents[:, terminal_index + 1], goals
        )
        targets = immediate + gamma * continuation * target_value(next_history, goals)
    return JointTDBatch(
        predicted_rollout=predicted_rollout,
        predicted_history=predicted_history,
        goals=goals,
        targets=targets,
        continuation=continuation,
    )


def configure_joint_trainability(
    world_model: torch.nn.Module,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Freeze the representation and return the two joint optimizer groups."""

    world_model.requires_grad_(False)
    frozen_modules = [world_model.encoder, world_model.projector]
    trainable_modules = [
        world_model.predictor,
        world_model.action_encoder,
        world_model.pred_proj,
    ]
    for module in trainable_modules:
        module.requires_grad_(True)
    frozen_parameters = [
        parameter for module in frozen_modules for parameter in module.parameters()
    ]
    dynamics_parameters = [
        parameter for module in trainable_modules for parameter in module.parameters()
    ]
    return frozen_parameters, dynamics_parameters


def _set_model_mode(world_model: torch.nn.Module, *, training: bool) -> None:
    world_model.train(training)
    world_model.encoder.eval()
    world_model.projector.eval()
    # The same pred_proj is called repeatedly during an imagined rollout.
    # Keep the reproduced LeWM BatchNorm statistics fixed while still training
    # its affine parameters; otherwise imagined states overwrite those stats.
    world_model.pred_proj.eval()


def _make_optimizer(
    dynamics_parameters: list[torch.nn.Parameter],
    value: GoalTailValue,
    config: dict[str, Any],
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [
            {
                "params": dynamics_parameters,
                "lr": float(config["dynamics_learning_rate"]),
            },
            {
                "params": list(value.parameters()),
                "lr": float(config["value_learning_rate"]),
            },
        ],
        weight_decay=float(config["weight_decay"]),
    )


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_fraction: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, int(total_steps * warmup_fraction))

    def scale(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scale)


def _checkpoint_payload(
    *,
    world_model: torch.nn.Module,
    value: GoalTailValue,
    target_value: GoalTailValue,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loader_generator: torch.Generator,
    goal_generator: torch.Generator,
    base_checkpoint_sha256: str,
    value_config: dict[str, Any],
    seed: int,
    epoch: int,
    global_step: int,
    best_validation_joint_loss: float,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "objective_version": 1,
        "training_state_version": 1,
        "method": "joint_td_gt_lewm",
        "seed": seed,
        "world_model_state_dict": world_model.state_dict(),
        "value_state_dict": value.state_dict(),
        "target_value_state_dict": target_value.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loader_generator_state": loader_generator.get_state(),
        "goal_generator_state": goal_generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        "numpy_random_state": np.random.get_state(),
        "value_config": value_config,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "epoch": epoch,
        "global_step": global_step,
        "best_validation_joint_loss": best_validation_joint_loss,
        "metrics": metrics,
    }


def restore_joint_training_state(
    checkpoint_path: str | Path,
    *,
    world_model: torch.nn.Module,
    value: GoalTailValue,
    target_value: GoalTailValue,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loader_generator: torch.Generator,
    goal_generator: torch.Generator,
    base_checkpoint_sha256: str,
    seed: int,
) -> dict[str, Any]:
    """Restore model, optimizer, scheduler, and every sampling RNG."""

    payload = torch.load(
        Path(checkpoint_path).expanduser().resolve(),
        map_location=next(value.parameters()).device,
        weights_only=False,
    )
    if payload.get("method") != "joint_td_gt_lewm":
        raise ValueError("The resume checkpoint is not Joint TD-GT-LeWM.")
    if payload.get("training_state_version") != 1:
        raise ValueError("The joint checkpoint is not resumable state version 1.")
    if payload.get("base_checkpoint_sha256") != base_checkpoint_sha256:
        raise ValueError("The joint checkpoint uses a different LeWM initialization.")
    if int(payload.get("seed", -1)) != seed:
        raise ValueError("The joint checkpoint seed differs from this run.")
    world_model.load_state_dict(payload["world_model_state_dict"])
    value.load_state_dict(payload["value_state_dict"])
    target_value.load_state_dict(payload["target_value_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    loader_generator.set_state(payload["loader_generator_state"].cpu())
    goal_generator.set_state(payload["goal_generator_state"].cpu())
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    torch.cuda.set_rng_state_all(
        [state.cpu() for state in payload["cuda_rng_state_all"]]
    )
    np.random.set_state(payload["numpy_random_state"])
    return payload


def train_joint_td_gt_lewm(
    *,
    protocol_path: str | Path,
    dataset_path: str | Path,
    base_checkpoint_path: str | Path,
    normalization_stats_path: str | Path,
    output_dir: str | Path,
    seed: int,
    latent_cache_dir: str | Path | None = None,
    smoke: bool = False,
    smoke_epochs: int = 1,
    resume_checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Jointly fine-tune LeWM dynamics and a rollout-conditioned TD tail."""

    protocol = load_joint_td_gt_protocol(protocol_path)
    if smoke:
        if smoke_epochs <= 0:
            raise ValueError("smoke_epochs must be positive.")
        protocol["id"] = f"{protocol['id']}_smoke"
        protocol["training"]["epochs"] = smoke_epochs
        protocol["loader"].update(
            {"batch_size": 256, "validation_batch_size": 128, "workers": 0}
        )
        protocol["smoke"] = True
    if seed not in protocol["seeds"]:
        raise ValueError(f"Seed {seed} is not in the configured training seeds.")
    dataset_path = Path(dataset_path).expanduser().resolve()
    dataset_source = validate_cube_training_dataset(dataset_path, protocol["dataset"])
    base_name, base_weights, base_cache = _resolve_base_checkpoint(base_checkpoint_path)
    base_sha256 = _sha256(base_weights)
    normalization_stats_path = Path(normalization_stats_path).expanduser().resolve()
    with normalization_stats_path.open() as stream:
        normalization = json.load(stream)
    if "action" not in normalization:
        raise ValueError("Normalization statistics do not contain action.")

    output_dir = Path(output_dir).expanduser().resolve()
    run_dir = output_dir / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    compatibility = prepare_cloud_runtime() or {}
    import stable_worldmodel as swm

    package_version = importlib.metadata.version("stable-worldmodel")
    expected_version = protocol["runtime"]["stable_worldmodel_version"]
    if package_version != expected_version:
        raise RuntimeError(
            f"Expected stable-worldmodel {expected_version}, found {package_version}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Joint TD-GT-LeWM training requires one CUDA device.")

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    world_model = swm.wm.load_pretrained(base_name, cache_dir=str(base_cache)).to(device)
    parameter_count = sum(parameter.numel() for parameter in world_model.parameters())
    if parameter_count != protocol["base_model"]["parameters"]:
        raise ValueError("The LeWM parameter count differs from protocol.")
    if int(world_model.predictor.num_frames) != protocol["base_model"]["history_size"]:
        raise ValueError("The base checkpoint has a different LeWM history size.")

    frozen_parameters, dynamics_parameters = configure_joint_trainability(world_model)
    if any(parameter.requires_grad for parameter in frozen_parameters):
        raise RuntimeError("The representation parameters must remain frozen.")
    if not all(parameter.requires_grad for parameter in dynamics_parameters):
        raise RuntimeError("Every configured dynamics parameter must be trainable.")
    _set_model_mode(world_model, training=False)

    cache_dir = (
        Path(latent_cache_dir).expanduser().resolve()
        if latent_cache_dir is not None
        else output_dir / "latent_cache"
    )
    latent_cache_path, latent_cache_manifest = ensure_cube_latent_cache(
        protocol=protocol,
        dataset_path=dataset_path,
        dataset_source=dataset_source,
        world_model=world_model,
        base_checkpoint_sha256=base_sha256,
        cache_dir=cache_dir,
        device=device,
    )

    sequence = protocol["sequence"]
    dataset = swm.data.load_dataset(
        str(dataset_path),
        format=dataset_source["format"],
        transform=None,
        num_steps=int(sequence["num_steps"]),
        frameskip=int(sequence["frame_skip"]),
        keys_to_load=["action"],
    )
    if len(dataset.lengths) != protocol["dataset"]["expected_episodes"]:
        raise ValueError("Cube episode count differs from protocol.")
    if int(np.asarray(dataset.lengths).sum()) != protocol["dataset"]["expected_transitions"]:
        raise ValueError("Cube transition count differs from protocol.")

    action_stats = normalization["action"]
    raw_actions = np.asarray(dataset.get_col_data("action"), dtype=np.float32)
    normalized_actions = (
        raw_actions - np.asarray(action_stats["mean"], dtype=np.float32)
    ) / np.asarray(action_stats["scale"], dtype=np.float32)
    split = protocol["split"]
    train_indices, validation_indices, train_episodes, validation_episodes = (
        split_clip_indices_by_episode(
            dataset.clip_indices,
            episode_count=len(dataset.lengths),
            train_fraction=float(split["train_fraction"]),
            seed=int(split["seed"]),
        )
    )
    if smoke:
        train_indices = train_indices[:512]
        validation_indices = validation_indices[:128]
    split_path = run_dir / "episode_split.npz"
    np.savez_compressed(
        split_path,
        train_indices=train_indices,
        validation_indices=validation_indices,
        train_episodes=train_episodes,
        validation_episodes=validation_episodes,
    )
    write_json(run_dir / "column_normalization.json", {"action": action_stats})

    history_size = int(sequence["history_frames"])
    rollout_horizon = int(sequence["model_rollout_horizon"])
    terminal_index = history_size + rollout_horizon - 1
    dataset_kwargs = {
        "latent_cache_path": latent_cache_path,
        "normalized_actions": normalized_actions,
        "clip_indices": dataset.clip_indices,
        "episode_offsets": dataset.offsets,
        "frame_skip": int(sequence["frame_skip"]),
        "num_steps": int(sequence["num_steps"]),
        "action_blocks": terminal_index + 1,
    }
    train_dataset = CachedCubeJointTDDataset(
        **dataset_kwargs, source_indices=train_indices
    )
    validation_dataset = CachedCubeJointTDDataset(
        **dataset_kwargs, source_indices=validation_indices
    )
    loader_cfg = protocol["loader"]
    loader_workers = int(loader_cfg["workers"])
    loader_common: dict[str, Any] = {
        "num_workers": loader_workers,
        "pin_memory": bool(loader_cfg["pin_memory"]),
    }
    if loader_workers:
        loader_common.update(
            {
                "persistent_workers": True,
                "prefetch_factor": int(loader_cfg["prefetch_factor"]),
            }
        )
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(loader_cfg["batch_size"]),
        shuffle=True,
        drop_last=bool(loader_cfg["train_drop_last"]),
        generator=loader_generator,
        **loader_common,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=int(loader_cfg["validation_batch_size"]),
        shuffle=False,
        drop_last=False,
        **loader_common,
    )

    latent_dim = int(protocol["base_model"]["latent_dim"])
    action_block_dim = int(dataset.get_dim("action")) * int(sequence["frame_skip"])
    history_dim = history_size * latent_dim + (history_size - 1) * action_block_dim
    tail_cfg = protocol["tail_value"]
    value = GoalTailValue(
        history_dim=history_dim,
        goal_dim=latent_dim,
        hidden_dim=int(tail_cfg["hidden_dim"]),
    ).to(device)
    target_value = copy.deepcopy(value).requires_grad_(False)
    optimizer = _make_optimizer(dynamics_parameters, value, protocol["optimizer"])
    training_cfg = protocol["training"]
    epochs = int(training_cfg["epochs"])
    total_steps = epochs * len(train_loader)
    scheduler = _make_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_fraction=float(protocol["scheduler"]["warmup_fraction"]),
    )
    value_config = {
        "history_dim": history_dim,
        "goal_dim": latent_dim,
        "hidden_dim": int(tail_cfg["hidden_dim"]),
        "history_size": history_size,
        "action_block_dim": action_block_dim,
        "max_goal_offset": int(sequence["max_goal_offset"]),
        "gamma": float(tail_cfg["gamma"]),
        "objective": "one_step_td",
        "input_distribution": "lewm_predicted_terminal_history",
        "model_rollout_horizon": rollout_horizon,
    }
    goal_generator = torch.Generator().manual_seed(seed + 1)
    start_epoch = 0
    global_step = 0
    best_validation_joint_loss = float("inf")
    last_metrics: dict[str, Any] | None = None
    if resume_checkpoint_path is not None:
        payload = restore_joint_training_state(
            resume_checkpoint_path,
            world_model=world_model,
            value=value,
            target_value=target_value,
            optimizer=optimizer,
            scheduler=scheduler,
            loader_generator=loader_generator,
            goal_generator=goal_generator,
            base_checkpoint_sha256=base_sha256,
            seed=seed,
        )
        start_epoch = int(payload["epoch"])
        global_step = int(payload["global_step"])
        best_validation_joint_loss = float(payload["best_validation_joint_loss"])
        last_metrics = dict(payload["metrics"])
        if start_epoch >= epochs:
            raise ValueError("The resume checkpoint already reached configured epochs.")

    optimized_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    expected_ids = {id(parameter) for parameter in dynamics_parameters} | {
        id(parameter) for parameter in value.parameters()
    }
    if optimized_ids != expected_ids:
        raise RuntimeError("The joint optimizer parameter boundary is incorrect.")
    manifest = {
        "method": "joint_td_gt_lewm",
        "display_name": "Joint TD-GT-LeWM",
        "stage": "full_training",
        "protocol": protocol,
        "protocol_path": str(Path(protocol_path).resolve()),
        "seed": seed,
        "smoke": smoke,
        "resume_checkpoint": (
            str(Path(resume_checkpoint_path).expanduser().resolve())
            if resume_checkpoint_path is not None
            else None
        ),
        "dataset": {
            **dataset_source,
            "sequence_samples": len(dataset),
            "train_clips": int(train_indices.size),
            "validation_clips": int(validation_indices.size),
            "train_episodes": int(train_episodes.size),
            "validation_episodes": int(validation_episodes.size),
            "split_path": str(split_path),
        },
        "base_checkpoint": {
            "name": base_name,
            "weights": str(base_weights),
            "sha256": base_sha256,
            "parameters": parameter_count,
            "initialization_only": True,
        },
        "latent_cache": {**latent_cache_manifest, "reuse_directory": str(cache_dir)},
        "joint_model": {
            "frozen_parameters": sum(p.numel() for p in frozen_parameters),
            "trainable_dynamics_parameters": sum(p.numel() for p in dynamics_parameters),
            "value_parameters": sum(p.numel() for p in value.parameters()),
            "tail_backpropagates_through_rollout": True,
            "value_config": value_config,
        },
        "training": {
            "epochs": epochs,
            "steps_per_epoch": len(train_loader),
            "total_optimizer_steps": total_steps,
            "training_clips_seen_per_epoch": int(train_indices.size),
        },
        "normalization_source": str(normalization_stats_path),
        "runtime": {
            "stable_worldmodel": package_version,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "cuda_device": torch.cuda.get_device_name(0),
            "tdwm_git_revision": _git_revision(),
            "compatibility_adapter": compatibility,
        },
    }
    write_json(run_dir / "training_manifest.json", manifest)

    gamma = float(tail_cfg["gamma"])
    ema_decay = float(tail_cfg["target_ema_decay"])
    max_goal_offset = int(sequence["max_goal_offset"])
    prediction_weight = float(protocol["joint_objective"]["prediction_weight"])
    tail_weight = float(protocol["joint_objective"]["tail_weight"])
    metrics_path = run_dir / "metrics.jsonl"
    initial_validation = None
    if start_epoch == 0 and bool(training_cfg["validate_before_training"]):
        initial_validation = _validate_joint_model(
            world_model,
            value,
            target_value,
            validation_loader,
            device=device,
            history_size=history_size,
            rollout_horizon=rollout_horizon,
            max_goal_offset=max_goal_offset,
            gamma=gamma,
            prediction_weight=prediction_weight,
            tail_weight=tail_weight,
        )
        initial_metrics = {"epoch": 0, "global_step": 0, **initial_validation}
        print(json.dumps(initial_metrics, sort_keys=True), flush=True)
        _append_json_line(metrics_path, initial_metrics)

    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    for epoch in range(start_epoch, epochs):
        _set_model_mode(world_model, training=True)
        value.train()
        target_value.eval()
        prediction_error = 0.0
        tail_error = 0.0
        sample_count = 0
        terminal_count = 0
        for batch in train_loader:
            latents = batch["latents"].to(device, dtype=torch.float32, non_blocking=True)
            action_blocks = batch["action_blocks"].to(
                device, dtype=torch.float32, non_blocking=True
            )
            offsets = torch.randint(
                1,
                max_goal_offset + 1,
                (latents.shape[0],),
                generator=goal_generator,
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                local_latents, local_actions, local_targets = teacher_forced_windows(
                    latents,
                    action_blocks,
                    history_size=history_size,
                    rollout_horizon=rollout_horizon,
                )
                local_action_embeddings = world_model.action_encoder(local_actions)
                local_predictions = world_model.predict(
                    local_latents, local_action_embeddings
                )
                prediction_loss = (
                    local_predictions.float() - local_targets.float()
                ).pow(2).mean()
                td_batch = build_joint_td_batch(
                    world_model,
                    target_value,
                    latents,
                    action_blocks,
                    offsets,
                    history_size=history_size,
                    rollout_horizon=rollout_horizon,
                    gamma=gamma,
                )
                tail_predictions = value(td_batch.predicted_history, td_batch.goals)
                tail_loss = (
                    tail_predictions.float() - td_batch.targets.float()
                ).pow(2).mean()
                loss = prediction_weight * prediction_loss + tail_weight * tail_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [*dynamics_parameters, *value.parameters()],
                float(training_cfg["gradient_clip_norm"]),
            )
            optimizer.step()
            scheduler.step()
            ema_update_target(target_value, value, decay=ema_decay)

            batch_size = int(latents.shape[0])
            prediction_error += float(prediction_loss.detach().cpu()) * batch_size
            tail_error += float(tail_loss.detach().cpu()) * batch_size
            sample_count += batch_size
            terminal_count += int((td_batch.continuation == 0).sum().cpu())
            global_step += 1

        validation = _validate_joint_model(
            world_model,
            value,
            target_value,
            validation_loader,
            device=device,
            history_size=history_size,
            rollout_horizon=rollout_horizon,
            max_goal_offset=max_goal_offset,
            gamma=gamma,
            prediction_weight=prediction_weight,
            tail_weight=tail_weight,
        )
        last_metrics = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_prediction_mse": prediction_error / sample_count,
            "train_tail_td_mse": tail_error / sample_count,
            "train_joint_loss": (
                prediction_weight * prediction_error + tail_weight * tail_error
            )
            / sample_count,
            "train_terminal_fraction": terminal_count / sample_count,
            "dynamics_learning_rate": optimizer.param_groups[0]["lr"],
            "value_learning_rate": optimizer.param_groups[1]["lr"],
            **validation,
        }
        print(json.dumps(last_metrics, sort_keys=True), flush=True)
        _append_json_line(metrics_path, last_metrics)
        is_best = validation["validation_joint_loss"] < best_validation_joint_loss
        if is_best:
            best_validation_joint_loss = float(validation["validation_joint_loss"])
        checkpoint = _checkpoint_payload(
            world_model=world_model,
            value=value,
            target_value=target_value,
            optimizer=optimizer,
            scheduler=scheduler,
            loader_generator=loader_generator,
            goal_generator=goal_generator,
            base_checkpoint_sha256=base_sha256,
            value_config=value_config,
            seed=seed,
            epoch=epoch + 1,
            global_step=global_step,
            best_validation_joint_loss=best_validation_joint_loss,
            metrics=last_metrics,
        )
        checkpoint_dir = run_dir / "checkpoints"
        _save_checkpoint(checkpoint_dir / f"epoch_{epoch + 1:02d}.pt", checkpoint)
        _save_checkpoint(checkpoint_dir / "last.pt", checkpoint)
        if is_best:
            _save_checkpoint(checkpoint_dir / "best.pt", checkpoint)

    result = {
        "method": "joint_td_gt_lewm",
        "display_name": "Joint TD-GT-LeWM",
        "run_dir": str(run_dir),
        "global_step": global_step,
        "epochs": epochs,
        "start_epoch": start_epoch,
        "elapsed_seconds": time.time() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "best_validation_joint_loss": best_validation_joint_loss,
        "initial_validation": initial_validation,
        "last_metrics": last_metrics,
        "best_checkpoint": str(run_dir / "checkpoints" / "best.pt"),
        "planner_connected": True,
        "smoke": smoke,
    }
    write_json(run_dir / "training_result.json", result)
    return result


@torch.inference_mode()
def _validate_joint_model(
    world_model: torch.nn.Module,
    value: GoalTailValue,
    target_value: GoalTailValue,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    history_size: int,
    rollout_horizon: int,
    max_goal_offset: int,
    gamma: float,
    prediction_weight: float,
    tail_weight: float,
) -> dict[str, float | int]:
    _set_model_mode(world_model, training=False)
    value.eval()
    target_value.eval()
    prediction_error = 0.0
    rollout_error = torch.zeros(rollout_horizon, dtype=torch.float64)
    td_error = 0.0
    mc_error = 0.0
    mc_absolute_error = 0.0
    pair_count = 0
    clip_count = 0
    predictions_all: list[np.ndarray] = []
    mc_targets_all: list[np.ndarray] = []
    terminal_index = history_size + rollout_horizon - 1
    offsets = torch.arange(1, max_goal_offset + 1, device=device)

    for batch in loader:
        latents = batch["latents"].to(device, dtype=torch.float32, non_blocking=True)
        action_blocks = batch["action_blocks"].to(
            device, dtype=torch.float32, non_blocking=True
        )
        local_latents, local_actions, local_targets = teacher_forced_windows(
            latents,
            action_blocks,
            history_size=history_size,
            rollout_horizon=rollout_horizon,
        )
        local_predictions = world_model.predict(
            local_latents, world_model.action_encoder(local_actions)
        )
        batch_prediction_mse = (
            local_predictions.float() - local_targets.float()
        ).pow(2).mean()
        predicted_rollout = rollout_from_latents(
            world_model,
            latents[:, :history_size],
            action_blocks[:, :terminal_index],
            history_size=history_size,
        )
        predicted_future = predicted_rollout[:, history_size:]
        true_future = latents[:, history_size : history_size + rollout_horizon]
        step_mse = (predicted_future.float() - true_future.float()).pow(2).mean((0, 2))
        predicted_history = build_history_at(
            predicted_rollout,
            action_blocks,
            current_index=terminal_index,
            history_size=history_size,
        )
        goals = latents[
            :, terminal_index + 1 : terminal_index + max_goal_offset + 1
        ]
        count = goals.shape[1]
        predictions = value(
            predicted_history.unsqueeze(1).expand(-1, count, -1), goals
        )
        next_history = build_history_at(
            latents,
            action_blocks,
            current_index=terminal_index + 1,
            history_size=history_size,
        )
        bootstrap = target_value(
            next_history.unsqueeze(1).expand(-1, count, -1), goals
        )
        immediate = (1.0 - gamma) * latent_goal_cost(
            latents[:, terminal_index + 1].unsqueeze(1), goals
        )
        continuation = (offsets > 1).to(latents.dtype).unsqueeze(0)
        td_targets = immediate + gamma * continuation * bootstrap
        mc_targets = monte_carlo_goal_tail_targets(
            latents,
            current_index=terminal_index,
            max_goal_offset=max_goal_offset,
            gamma=gamma,
        )

        batch_size = int(latents.shape[0])
        prediction_error += float(batch_prediction_mse.cpu()) * batch_size
        rollout_error += step_mse.double().cpu() * batch_size
        td_error += float((predictions - td_targets).pow(2).sum().cpu())
        mc_error += float((predictions - mc_targets).pow(2).sum().cpu())
        mc_absolute_error += float((predictions - mc_targets).abs().sum().cpu())
        clip_count += batch_size
        pair_count += int(predictions.numel())
        predictions_all.append(predictions.float().cpu().numpy().reshape(-1))
        mc_targets_all.append(mc_targets.float().cpu().numpy().reshape(-1))

    prediction_mse = prediction_error / clip_count
    td_mse = td_error / pair_count
    prediction_array = np.concatenate(predictions_all)
    mc_target_array = np.concatenate(mc_targets_all)
    metrics: dict[str, float | int] = {
        "validation_clips": clip_count,
        "validation_pairs": pair_count,
        "validation_prediction_mse": prediction_mse,
        "validation_rollout_mse": float(rollout_error.mean().item() / clip_count),
        "validation_terminal_rollout_mse": float(rollout_error[-1].item() / clip_count),
        "validation_tail_td_mse": td_mse,
        "validation_tail_mc_mse": mc_error / pair_count,
        "validation_tail_mc_mae": mc_absolute_error / pair_count,
        "validation_tail_mc_spearman": spearman_correlation(
            prediction_array, mc_target_array
        ),
        "validation_joint_loss": prediction_weight * prediction_mse + tail_weight * td_mse,
    }
    for step, error in enumerate(rollout_error.tolist(), start=1):
        metrics[f"validation_rollout_step_{step}_mse"] = error / clip_count
    return metrics
