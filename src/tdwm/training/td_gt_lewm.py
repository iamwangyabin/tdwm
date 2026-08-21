"""Full TD-GT-LeWM training from frozen LeWM Cube latents."""

from __future__ import annotations

import copy
import importlib.metadata
import json
import platform
import time
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
from tdwm.training.train_goal_tail_cube import (
    build_value_optimizer,
    freeze_world_model,
    spearman_correlation,
    split_clip_indices_by_episode,
)


class CachedCubeTDGoalTailDataset(torch.utils.data.Dataset):
    """Read latent clips and the three action blocks needed by one-step TD."""

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
        history_size: int,
    ) -> None:
        if frame_skip <= 0 or num_steps <= 0 or history_size <= 0:
            raise ValueError("frame_skip, num_steps, and history_size must be positive.")
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
        self.action_context_steps = history_size * frame_skip
        self.action_offsets = np.arange(self.action_context_steps, dtype=np.int64)
        self.history_size = int(history_size)
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
        action_blocks = np.array(
            self.normalized_actions[action_rows], copy=True
        ).reshape(len(positions), self.history_size, -1)
        return [
            {
                "latents": torch.from_numpy(latents[index]),
                "action_blocks": torch.from_numpy(action_blocks[index]),
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


def load_td_gt_protocol(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as stream:
        protocol = yaml.safe_load(stream)
    validate_td_gt_protocol(protocol)
    return protocol


def validate_td_gt_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema_version") != 1:
        raise ValueError("TD-GT-LeWM requires schema_version 1.")
    if protocol.get("method") != "td_gt_lewm":
        raise ValueError("This trainer only accepts method td_gt_lewm.")
    if protocol.get("environment") != "cube":
        raise ValueError("TD-GT-LeWM V0.2 is locked to OGBench Cube.")
    if protocol.get("stage") != "full_training":
        raise ValueError("TD-GT-LeWM requires full_training stage.")
    if protocol.get("runtime", {}).get("stable_worldmodel_version") != "0.1.1":
        raise ValueError("TD-GT-LeWM is locked to stable-worldmodel 0.1.1.")
    if not protocol.get("seeds"):
        raise ValueError("At least one TD-GT-LeWM seed is required.")

    base = protocol.get("base_model", {})
    if base.get("method") != "lewm" or base.get("frozen") is not True:
        raise ValueError("TD-GT-LeWM requires a frozen LeWM base model.")
    if base.get("history_size") != 3 or base.get("latent_dim", 0) <= 0:
        raise ValueError("TD-GT-LeWM requires the shipped LeWM history size 3.")

    sequence = protocol.get("sequence", {})
    if sequence.get("history_frames") != 3:
        raise ValueError("TD-GT-LeWM history_frames must remain 3.")
    if sequence.get("max_goal_offset") != 16:
        raise ValueError("TD-GT-LeWM goal offsets must remain 1 through 16.")
    if sequence.get("num_steps") != 19:
        raise ValueError("TD-GT-LeWM requires 3 history plus 16 future frames.")
    if sequence.get("frame_skip", 0) <= 0:
        raise ValueError("sequence.frame_skip must be positive.")

    tail = protocol.get("tail_value", {})
    if tail.get("objective") != "one_step_td":
        raise ValueError("TD-GT-LeWM only supports the one-step TD objective.")
    if tail.get("goal_sampling") != "uniform_future_offset":
        raise ValueError("TD-GT-LeWM requires uniform future-goal sampling.")
    if not 0.0 <= tail.get("gamma", -1.0) < 1.0:
        raise ValueError("tail_value.gamma must lie in [0, 1).")
    if tail.get("hidden_dim", 0) <= 0:
        raise ValueError("tail_value.hidden_dim must be positive.")
    if tail.get("target_network") is not True:
        raise ValueError("TD-GT-LeWM requires a target network.")
    if not 0.0 <= tail.get("target_ema_decay", -1.0) < 1.0:
        raise ValueError("target_ema_decay must lie in [0, 1).")
    if tail.get("terminate_bootstrap_at_goal") is not True:
        raise ValueError("TD-GT-LeWM must stop bootstrapping at the hindsight goal.")

    initialization = protocol.get("initialization", {})
    if initialization.get("value") != "random":
        raise ValueError("The controlled TD head must start from random initialization.")
    if initialization.get("target") != "copy_of_initial_value":
        raise ValueError("The target network must start as an exact value copy.")
    if initialization.get("mc_gt_warm_start") is not False:
        raise ValueError("TD-GT-LeWM cannot warm-start from MC-GT-LeWM.")

    split = protocol.get("split", {})
    if split.get("unit") != "episode":
        raise ValueError("TD-GT-LeWM requires a whole-episode split.")
    if not np.isclose(
        split.get("train_fraction", 0.0)
        + split.get("validation_fraction", 0.0),
        1.0,
    ):
        raise ValueError("Training and validation fractions must sum to one.")

    loader = protocol.get("loader", {})
    if loader.get("batch_size", 0) <= 0 or loader.get("workers", -1) < 0:
        raise ValueError("TD-GT-LeWM loader settings are invalid.")
    if loader.get("validation_batch_size", 0) <= 0:
        raise ValueError("loader.validation_batch_size must be positive.")

    training = protocol.get("training", {})
    if training.get("epochs", 0) <= 0:
        raise ValueError("training.epochs must be positive.")
    if training.get("coverage") != "all_training_clips_each_epoch":
        raise ValueError("Formal TD training must cover every training clip.")
    if training.get("checkpoint_selection") != "minimum_validation_mc_mse":
        raise ValueError("TD checkpoint selection must use the fixed MC validation metric.")


def build_td_histories(
    latents: torch.Tensor,
    action_blocks: torch.Tensor,
    *,
    history_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build h_t and h_(t+1) from aligned latent and action blocks."""

    if latents.ndim != 3 or action_blocks.ndim != 3:
        raise ValueError("latents and action_blocks must be rank-three tensors.")
    if latents.shape[0] != action_blocks.shape[0]:
        raise ValueError("latents and action_blocks must share their batch size.")
    if latents.shape[1] < history_size + 1:
        raise ValueError("latents do not contain the next-state history.")
    if action_blocks.shape[1] < history_size:
        raise ValueError("action_blocks do not contain the next-state history.")
    current = torch.cat(
        (
            latents[:, :history_size].flatten(1),
            action_blocks[:, : history_size - 1].flatten(1),
        ),
        dim=-1,
    )
    following = torch.cat(
        (
            latents[:, 1 : history_size + 1].flatten(1),
            action_blocks[:, 1:history_size].flatten(1),
        ),
        dim=-1,
    )
    return current, following


@torch.no_grad()
def one_step_td_goal_tail_targets(
    target_value: GoalTailValue,
    latents: torch.Tensor,
    action_blocks: torch.Tensor,
    goal_offsets: torch.Tensor,
    *,
    history_size: int,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build goals and one-step TD targets, terminating at hindsight goals."""

    if goal_offsets.ndim != 1 or goal_offsets.shape[0] != latents.shape[0]:
        raise ValueError("goal_offsets must contain one offset per batch element.")
    if goal_offsets.dtype not in (torch.int32, torch.int64):
        raise TypeError("goal_offsets must use an integer dtype.")
    if torch.any(goal_offsets < 1):
        raise ValueError("goal_offsets must be positive.")
    current_index = history_size - 1
    if torch.any(current_index + goal_offsets >= latents.shape[1]):
        raise ValueError("latents do not contain every requested goal.")
    current_history, next_history = build_td_histories(
        latents, action_blocks, history_size=history_size
    )
    rows = torch.arange(latents.shape[0], device=latents.device)
    goals = latents[rows, current_index + goal_offsets]
    immediate = (1.0 - gamma) * latent_goal_cost(
        latents[:, current_index + 1], goals
    )
    continuation = (goal_offsets > 1).to(latents.dtype)
    bootstrap = target_value(next_history, goals)
    targets = immediate + gamma * continuation * bootstrap
    return current_history, goals, targets, continuation


@torch.no_grad()
def ema_update_target(
    target_value: GoalTailValue,
    value: GoalTailValue,
    *,
    decay: float,
) -> None:
    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must lie in [0, 1).")
    for target_parameter, parameter in zip(
        target_value.parameters(), value.parameters(), strict=True
    ):
        target_parameter.mul_(decay).add_(parameter, alpha=1.0 - decay)


def restore_td_training_state(
    checkpoint_path: str | Path,
    *,
    value: GoalTailValue,
    target_value: GoalTailValue,
    optimizer: torch.optim.Optimizer,
    loader_generator: torch.Generator,
    goal_generator: torch.Generator,
    base_checkpoint_sha256: str,
    seed: int,
) -> dict[str, Any]:
    """Restore every state that affects subsequent TD optimizer updates."""

    device = next(value.parameters()).device
    payload = torch.load(
        Path(checkpoint_path).expanduser().resolve(),
        map_location=device,
        weights_only=False,
    )
    if payload.get("method") != "td_gt_lewm":
        raise ValueError("The resume checkpoint is not TD-GT-LeWM.")
    if payload.get("training_state_version") != 1:
        raise ValueError("The TD checkpoint does not contain resumable state version 1.")
    if payload.get("base_checkpoint_sha256") != base_checkpoint_sha256:
        raise ValueError("The resume checkpoint uses a different LeWM base.")
    if int(payload.get("seed", -1)) != seed:
        raise ValueError("The resume checkpoint seed differs from this run.")
    config = payload["value_config"]
    expected = {
        "history_dim": value.history_dim,
        "goal_dim": value.goal_dim,
        "hidden_dim": value.hidden_dim,
    }
    for key, expected_value in expected.items():
        if int(config.get(key, -1)) != expected_value:
            raise ValueError(f"The resume checkpoint {key} differs from this run.")
    value.load_state_dict(payload["value_state_dict"])
    target_value.load_state_dict(payload["target_value_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    loader_generator.set_state(payload["loader_generator_state"].cpu())
    goal_generator.set_state(payload["goal_generator_state"].cpu())
    return payload


def train_td_gt_lewm(
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
    """Train a one-step TD goal-tail value on the full frozen Cube dataset."""

    protocol = load_td_gt_protocol(protocol_path)
    if smoke:
        if smoke_epochs <= 0:
            raise ValueError("smoke_epochs must be positive.")
        protocol["id"] = f"{protocol['id']}_smoke"
        protocol["training"]["epochs"] = smoke_epochs
        protocol["loader"].update(
            {"batch_size": 1024, "validation_batch_size": 512, "workers": 0}
        )
        protocol["smoke"] = True
    if seed not in protocol["seeds"]:
        raise ValueError(f"Seed {seed} is not in the configured TD head seeds.")
    dataset_path = Path(dataset_path).expanduser().resolve()
    dataset_source = validate_cube_training_dataset(dataset_path, protocol["dataset"])
    base_name, base_weights, base_cache = _resolve_base_checkpoint(
        base_checkpoint_path
    )
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
        raise RuntimeError("Full TD-GT-LeWM training requires one CUDA device.")

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    base_sha256 = _sha256(base_weights)
    world_model = freeze_world_model(
        swm.wm.load_pretrained(base_name, cache_dir=str(base_cache)).to(device)
    )
    parameter_count = sum(parameter.numel() for parameter in world_model.parameters())
    if parameter_count != protocol["base_model"]["parameters"]:
        raise ValueError("The LeWM parameter count differs from protocol.")
    if int(world_model.predictor.num_frames) != 3:
        raise ValueError("The base checkpoint does not use LeWM history size 3.")

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
    del world_model
    torch.cuda.empty_cache()

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
    action_mean = np.asarray(action_stats["mean"], dtype=np.float32)
    action_scale = np.asarray(action_stats["scale"], dtype=np.float32)
    normalized_actions = (raw_actions - action_mean) / action_scale

    split_cfg = protocol["split"]
    train_indices, validation_indices, train_episodes, validation_episodes = (
        split_clip_indices_by_episode(
            dataset.clip_indices,
            episode_count=len(dataset.lengths),
            train_fraction=float(split_cfg["train_fraction"]),
            seed=int(split_cfg["seed"]),
        )
    )
    if smoke:
        train_indices = train_indices[:2048]
        validation_indices = validation_indices[:1024]
    split_path = run_dir / "episode_split.npz"
    np.savez_compressed(
        split_path,
        train_indices=train_indices,
        validation_indices=validation_indices,
        train_episodes=train_episodes,
        validation_episodes=validation_episodes,
    )
    write_json(run_dir / "column_normalization.json", {"action": action_stats})

    dataset_kwargs = {
        "latent_cache_path": latent_cache_path,
        "normalized_actions": normalized_actions,
        "clip_indices": dataset.clip_indices,
        "episode_offsets": dataset.offsets,
        "frame_skip": int(sequence["frame_skip"]),
        "num_steps": int(sequence["num_steps"]),
        "history_size": int(sequence["history_frames"]),
    }
    train_dataset = CachedCubeTDGoalTailDataset(
        **dataset_kwargs, source_indices=train_indices
    )
    validation_dataset = CachedCubeTDGoalTailDataset(
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
    history_size = int(sequence["history_frames"])
    action_block_dim = int(dataset.get_dim("action")) * int(sequence["frame_skip"])
    history_dim = history_size * latent_dim + (history_size - 1) * action_block_dim
    value = GoalTailValue(
        history_dim=history_dim,
        goal_dim=latent_dim,
        hidden_dim=int(protocol["tail_value"]["hidden_dim"]),
    ).to(device)
    target_value = copy.deepcopy(value).to(device).eval()
    target_value.requires_grad_(False)
    optimizer = build_value_optimizer(value, protocol["optimizer"])
    optimized_parameters = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimized_parameters != {id(parameter) for parameter in value.parameters()}:
        raise RuntimeError("The optimizer must contain exactly value.parameters().")

    training_cfg = protocol["training"]
    epochs = int(training_cfg["epochs"])
    goal_generator = torch.Generator().manual_seed(seed + 1)
    start_epoch = 0
    global_step = 0
    best_validation_mc_mse = float("inf")
    last_metrics: dict[str, Any] | None = None
    resume_payload = None
    if resume_checkpoint_path is not None:
        resume_payload = restore_td_training_state(
            resume_checkpoint_path,
            value=value,
            target_value=target_value,
            optimizer=optimizer,
            loader_generator=loader_generator,
            goal_generator=goal_generator,
            base_checkpoint_sha256=base_sha256,
            seed=seed,
        )
        start_epoch = int(resume_payload["epoch"])
        global_step = int(resume_payload["global_step"])
        best_validation_mc_mse = float(
            resume_payload["best_validation_mc_mse"]
        )
        last_metrics = dict(resume_payload["metrics"])
        if start_epoch >= epochs:
            raise ValueError("The resume checkpoint already reached configured epochs.")
    manifest = {
        "method": "td_gt_lewm",
        "display_name": "TD-GT-LeWM",
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
        "start_epoch": start_epoch,
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
            "frozen": True,
        },
        "latent_cache": {
            **latent_cache_manifest,
            "reuse_directory": str(cache_dir),
        },
        "value": {
            "history_dim": history_dim,
            "goal_dim": latent_dim,
            "action_block_dim": action_block_dim,
            "parameters": sum(parameter.numel() for parameter in value.parameters()),
            "optimizer_parameter_source": "value.parameters()",
            "target_parameters_optimized": False,
        },
        "training": {
            "epochs": epochs,
            "steps_per_epoch": len(train_loader),
            "total_optimizer_steps": epochs * len(train_loader),
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

    tail_cfg = protocol["tail_value"]
    gamma = float(tail_cfg["gamma"])
    ema_decay = float(tail_cfg["target_ema_decay"])
    current_index = history_size - 1
    max_goal_offset = int(sequence["max_goal_offset"])
    metrics_path = run_dir / "metrics.jsonl"
    initial_validation = None
    if start_epoch == 0 and bool(training_cfg["validate_before_training"]):
        initial_validation = _validate_td_value(
            value,
            target_value,
            validation_loader,
            device=device,
            history_size=history_size,
            max_goal_offset=max_goal_offset,
            gamma=gamma,
        )
        initial_metrics = {"epoch": 0, "global_step": 0, **initial_validation}
        print(json.dumps(initial_metrics, sort_keys=True), flush=True)
        _append_json_line(metrics_path, initial_metrics)

    started = time.time()
    for epoch in range(start_epoch, epochs):
        value.train()
        squared_error = 0.0
        sample_count = 0
        terminal_samples = 0
        for batch in train_loader:
            latents = batch["latents"].to(
                device, dtype=torch.float32, non_blocking=True
            )
            action_blocks = batch["action_blocks"].to(
                device, dtype=torch.float32, non_blocking=True
            )
            offsets = torch.randint(
                1,
                max_goal_offset + 1,
                (latents.shape[0],),
                generator=goal_generator,
            ).to(device)
            history, goals, targets, continuation = one_step_td_goal_tail_targets(
                target_value,
                latents,
                action_blocks,
                offsets,
                history_size=history_size,
                gamma=gamma,
            )
            predictions = value(history, goals)
            loss = torch.nn.functional.mse_loss(predictions, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                value.parameters(), float(training_cfg["gradient_clip_norm"])
            )
            optimizer.step()
            ema_update_target(target_value, value, decay=ema_decay)

            squared_error += float(
                (predictions.detach() - targets).pow(2).sum().cpu()
            )
            sample_count += int(targets.numel())
            terminal_samples += int((continuation == 0).sum().cpu())
            global_step += 1

        validation = _validate_td_value(
            value,
            target_value,
            validation_loader,
            device=device,
            history_size=history_size,
            max_goal_offset=max_goal_offset,
            gamma=gamma,
        )
        last_metrics = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_td_mse": squared_error / sample_count,
            "train_terminal_fraction": terminal_samples / sample_count,
            **validation,
        }
        print(json.dumps(last_metrics, sort_keys=True), flush=True)
        _append_json_line(metrics_path, last_metrics)
        best_validation_mc_mse = min(
            best_validation_mc_mse, float(validation["validation_mc_mse"])
        )
        checkpoint = {
            "objective_version": 1,
            "training_state_version": 1,
            "method": "td_gt_lewm",
            "seed": seed,
            "value_state_dict": value.state_dict(),
            "target_value_state_dict": target_value.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loader_generator_state": loader_generator.get_state(),
            "goal_generator_state": goal_generator.get_state(),
            "value_config": {
                "history_dim": history_dim,
                "goal_dim": latent_dim,
                "hidden_dim": int(tail_cfg["hidden_dim"]),
                "history_size": history_size,
                "action_block_dim": action_block_dim,
                "max_goal_offset": max_goal_offset,
                "gamma": gamma,
                "objective": "one_step_td",
                "target_ema_decay": ema_decay,
                "terminate_bootstrap_at_goal": True,
            },
            "base_checkpoint_sha256": base_sha256,
            "epoch": epoch + 1,
            "global_step": global_step,
            "best_validation_mc_mse": best_validation_mc_mse,
            "metrics": last_metrics,
        }
        checkpoint_dir = run_dir / "checkpoints"
        _save_checkpoint(checkpoint_dir / f"epoch_{epoch + 1:02d}.pt", checkpoint)
        _save_checkpoint(checkpoint_dir / "last.pt", checkpoint)
        if validation["validation_mc_mse"] == best_validation_mc_mse:
            _save_checkpoint(checkpoint_dir / "best.pt", checkpoint)

    result = {
        "method": "td_gt_lewm",
        "display_name": "TD-GT-LeWM",
        "run_dir": str(run_dir),
        "global_step": global_step,
        "epochs": epochs,
        "start_epoch": start_epoch,
        "elapsed_seconds": time.time() - started,
        "best_validation_mc_mse": best_validation_mc_mse,
        "initial_validation": initial_validation,
        "last_metrics": last_metrics,
        "best_checkpoint": str(run_dir / "checkpoints" / "best.pt"),
        "planner_connected": False,
        "smoke": smoke,
    }
    write_json(run_dir / "training_result.json", result)
    return result


@torch.inference_mode()
def _validate_td_value(
    value: GoalTailValue,
    target_value: GoalTailValue,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    history_size: int,
    max_goal_offset: int,
    gamma: float,
) -> dict[str, float | int]:
    value.eval()
    target_value.eval()
    td_squared_error = 0.0
    mc_squared_error = 0.0
    mc_absolute_error = 0.0
    prediction_sum = 0.0
    td_target_sum = 0.0
    mc_target_sum = 0.0
    pair_count = 0
    predictions_all: list[np.ndarray] = []
    mc_targets_all: list[np.ndarray] = []
    offsets = torch.arange(1, max_goal_offset + 1, device=device)
    current_index = history_size - 1

    for batch in loader:
        latents = batch["latents"].to(
            device, dtype=torch.float32, non_blocking=True
        )
        action_blocks = batch["action_blocks"].to(
            device, dtype=torch.float32, non_blocking=True
        )
        current_history, next_history = build_td_histories(
            latents, action_blocks, history_size=history_size
        )
        goals = latents[
            :, current_index + 1 : current_index + max_goal_offset + 1
        ]
        count = goals.shape[1]
        predictions = value(
            current_history.unsqueeze(1).expand(-1, count, -1), goals
        )
        bootstrap = target_value(
            next_history.unsqueeze(1).expand(-1, count, -1), goals
        )
        immediate = (1.0 - gamma) * latent_goal_cost(
            latents[:, current_index + 1].unsqueeze(1), goals
        )
        continuation = (offsets > 1).to(latents.dtype).unsqueeze(0)
        td_targets = immediate + gamma * continuation * bootstrap
        mc_targets = monte_carlo_goal_tail_targets(
            latents,
            current_index=current_index,
            max_goal_offset=max_goal_offset,
            gamma=gamma,
        )

        td_squared_error += float((predictions - td_targets).pow(2).sum().cpu())
        mc_squared_error += float((predictions - mc_targets).pow(2).sum().cpu())
        mc_absolute_error += float((predictions - mc_targets).abs().sum().cpu())
        prediction_sum += float(predictions.sum().cpu())
        td_target_sum += float(td_targets.sum().cpu())
        mc_target_sum += float(mc_targets.sum().cpu())
        pair_count += int(predictions.numel())
        predictions_all.append(predictions.float().cpu().numpy().reshape(-1))
        mc_targets_all.append(mc_targets.float().cpu().numpy().reshape(-1))

    prediction_array = np.concatenate(predictions_all)
    mc_target_array = np.concatenate(mc_targets_all)
    return {
        "validation_pairs": pair_count,
        "validation_td_mse": td_squared_error / pair_count,
        "validation_mc_mse": mc_squared_error / pair_count,
        "validation_mc_mae": mc_absolute_error / pair_count,
        "validation_mc_spearman": spearman_correlation(
            prediction_array, mc_target_array
        ),
        "validation_prediction_mean": prediction_sum / pair_count,
        "validation_td_target_mean": td_target_sum / pair_count,
        "validation_mc_target_mean": mc_target_sum / pair_count,
    }
