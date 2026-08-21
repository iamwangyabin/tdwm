"""Full MC-GT-LeWM training from a frozen, cached LeWM Cube encoder."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml

from tdwm.adapters import prepare_cloud_runtime
from tdwm.methods.goal_tail_value import (
    GoalTailValue,
    monte_carlo_goal_tail_targets,
    monte_carlo_goal_tail_targets_for_offsets,
)
from tdwm.training.cube_data import validate_cube_training_dataset
from tdwm.training.gt_lewm_support import preprocess_image_batch, write_json
from tdwm.training.train_goal_tail_cube import (
    build_value_optimizer,
    freeze_world_model,
    spearman_correlation,
    split_clip_indices_by_episode,
)


class CachedCubeGoalTailDataset(torch.utils.data.Dataset):
    """Read strided latent clips and preceding action blocks from local arrays."""

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
        self.action_history_steps = (history_size - 1) * frame_skip
        self.action_offsets = np.arange(self.action_history_steps, dtype=np.int64)
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
        action_history = np.array(
            self.normalized_actions[action_rows], copy=True
        ).reshape(len(positions), self.history_size - 1, -1)
        return [
            {
                "latents": torch.from_numpy(latents[index]),
                "action_history": torch.from_numpy(action_history[index]),
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


def load_mc_gt_protocol(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as stream:
        protocol = yaml.safe_load(stream)
    validate_mc_gt_protocol(protocol)
    return protocol


def validate_mc_gt_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema_version") != 1:
        raise ValueError("MC-GT-LeWM requires schema_version 1.")
    if protocol.get("method") != "mc_gt_lewm":
        raise ValueError("This trainer only accepts method mc_gt_lewm.")
    if protocol.get("environment") != "cube":
        raise ValueError("MC-GT-LeWM V0.1 is locked to OGBench Cube.")
    if protocol.get("stage") != "full_training":
        raise ValueError("The formal MC-GT-LeWM config must use full_training.")
    if protocol.get("runtime", {}).get("stable_worldmodel_version") != "0.1.1":
        raise ValueError("MC-GT-LeWM is locked to stable-worldmodel 0.1.1.")
    if not protocol.get("seeds"):
        raise ValueError("At least one MC-GT-LeWM head seed is required.")

    base = protocol.get("base_model", {})
    if base.get("method") != "lewm" or base.get("frozen") is not True:
        raise ValueError("MC-GT-LeWM requires a frozen LeWM base model.")
    if base.get("history_size") != 3 or base.get("latent_dim", 0) <= 0:
        raise ValueError("MC-GT-LeWM requires the shipped LeWM history size 3.")

    sequence = protocol.get("sequence", {})
    if sequence.get("history_frames") != 3:
        raise ValueError("MC-GT-LeWM history_frames must remain 3.")
    if sequence.get("max_goal_offset") != 16:
        raise ValueError("MC-GT-LeWM goal offsets must remain 1 through 16.")
    if sequence.get("num_steps") != 19:
        raise ValueError("MC-GT-LeWM requires 3 history plus 16 future frames.")
    if sequence.get("frame_skip", 0) <= 0:
        raise ValueError("sequence.frame_skip must be positive.")

    tail = protocol.get("tail_value", {})
    if tail.get("objective") != "supervised_mc":
        raise ValueError("MC-GT-LeWM only supports supervised MC targets.")
    if tail.get("goal_sampling") != "uniform_future_offset":
        raise ValueError("MC-GT-LeWM requires uniform future-goal sampling.")
    if not 0.0 <= tail.get("gamma", -1.0) < 1.0:
        raise ValueError("tail_value.gamma must lie in [0, 1).")
    if tail.get("hidden_dim", 0) <= 0:
        raise ValueError("tail_value.hidden_dim must be positive.")
    forbidden = {"td_horizon", "target_ema_decay", "target_network", "policy"}
    if forbidden.intersection(tail):
        raise ValueError("MC-GT-LeWM V0.1 cannot configure TD, EMA, or policy.")

    split = protocol.get("split", {})
    train_fraction = split.get("train_fraction", 0.0)
    if split.get("unit") != "episode" or not 0.0 < train_fraction < 1.0:
        raise ValueError("MC-GT-LeWM requires a whole-episode train split.")
    if not np.isclose(
        train_fraction + split.get("validation_fraction", 0.0), 1.0
    ):
        raise ValueError("Training and validation fractions must sum to one.")

    cache = protocol.get("latent_cache", {})
    if cache.get("dtype") != "float32":
        raise ValueError("The formal latent cache must preserve float32 values.")
    if cache.get("batch_size", 0) <= 0 or cache.get("workers", -1) < 0:
        raise ValueError("latent_cache loader settings are invalid.")

    loader = protocol.get("loader", {})
    if loader.get("batch_size", 0) <= 0 or loader.get("workers", -1) < 0:
        raise ValueError("MC-GT-LeWM loader settings are invalid.")
    if loader.get("validation_batch_size", 0) <= 0:
        raise ValueError("loader.validation_batch_size must be positive.")
    if loader.get("prefetch_factor", 0) <= 0:
        raise ValueError("loader.prefetch_factor must be positive.")

    training = protocol.get("training", {})
    if training.get("epochs", 0) <= 0:
        raise ValueError("training.epochs must be positive.")
    if training.get("coverage") != "all_training_clips_each_epoch":
        raise ValueError("Formal training must cover every training clip each epoch.")
    if not isinstance(training.get("validate_before_training"), bool):
        raise ValueError("training.validate_before_training must be true or false.")
    if training.get("checkpoint_selection") != "minimum_validation_mse":
        raise ValueError("Formal checkpoint selection must use validation MSE.")


def train_mc_gt_lewm(
    *,
    protocol_path: str | Path,
    dataset_path: str | Path,
    base_checkpoint_path: str | Path,
    normalization_stats_path: str | Path,
    output_dir: str | Path,
    seed: int,
) -> dict[str, Any]:
    """Cache frozen Cube latents once, then train MC-GT-LeWM on full clips."""

    protocol = load_mc_gt_protocol(protocol_path)
    if seed not in protocol["seeds"]:
        raise ValueError(f"Seed {seed} is not in the configured head seeds.")
    dataset_path = Path(dataset_path).expanduser().resolve()
    dataset_source = validate_cube_training_dataset(
        dataset_path, protocol["dataset"]
    )
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
        raise RuntimeError("Full MC-GT-LeWM training requires one CUDA device.")

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
    expected_parameters = protocol["base_model"].get("parameters")
    if expected_parameters and parameter_count != expected_parameters:
        raise ValueError(
            f"Expected {expected_parameters} LeWM parameters, found {parameter_count}."
        )
    if int(world_model.predictor.num_frames) != 3:
        raise ValueError("The base checkpoint does not use LeWM history size 3.")

    latent_cache_path, latent_cache_manifest = ensure_cube_latent_cache(
        protocol=protocol,
        dataset_path=dataset_path,
        dataset_source=dataset_source,
        world_model=world_model,
        base_checkpoint_sha256=base_sha256,
        cache_dir=output_dir / "latent_cache",
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
    dataset_cfg = protocol["dataset"]
    if len(dataset.lengths) != dataset_cfg["expected_episodes"]:
        raise ValueError("Cube episode count differs from the locked protocol.")
    if int(np.asarray(dataset.lengths).sum()) != dataset_cfg["expected_transitions"]:
        raise ValueError("Cube transition count differs from the locked protocol.")

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
    train_dataset = CachedCubeGoalTailDataset(
        **dataset_kwargs, source_indices=train_indices
    )
    validation_dataset = CachedCubeGoalTailDataset(
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
    steps_per_epoch = len(train_loader)
    manifest = {
        "method": "mc_gt_lewm",
        "display_name": "MC-GT-LeWM",
        "stage": "full_training",
        "protocol": protocol,
        "protocol_path": str(Path(protocol_path).resolve()),
        "seed": seed,
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
        "latent_cache": latent_cache_manifest,
        "value": {
            "history_dim": history_dim,
            "goal_dim": latent_dim,
            "action_block_dim": action_block_dim,
            "parameters": sum(parameter.numel() for parameter in value.parameters()),
            "optimizer_parameter_source": "value.parameters()",
        },
        "training": {
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "total_optimizer_steps": epochs * steps_per_epoch,
            "training_clips_seen_per_epoch": int(train_indices.size),
            "full_training_clip_count": int(train_indices.size),
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

    gamma = float(protocol["tail_value"]["gamma"])
    current_index = history_size - 1
    max_goal_offset = int(sequence["max_goal_offset"])
    goal_generator = torch.Generator().manual_seed(seed + 1)
    metrics_path = run_dir / "metrics.jsonl"
    best_validation_mse = float("inf")
    global_step = 0
    initial_validation = None
    if bool(training_cfg["validate_before_training"]):
        initial_validation = _validate_cached_value(
            value,
            validation_loader,
            device=device,
            history_size=history_size,
            current_index=current_index,
            max_goal_offset=max_goal_offset,
            gamma=gamma,
        )
        initial_metrics = {"epoch": 0, "global_step": 0, **initial_validation}
        print(json.dumps(initial_metrics, sort_keys=True), flush=True)
        _append_json_line(metrics_path, initial_metrics)

    started = time.time()
    last_metrics: dict[str, Any] | None = None
    for epoch in range(epochs):
        value.train()
        squared_error = 0.0
        sample_count = 0
        for batch in train_loader:
            latents = batch["latents"].to(
                device, dtype=torch.float32, non_blocking=True
            )
            action_history = batch["action_history"].to(
                device, dtype=torch.float32, non_blocking=True
            )
            history = torch.cat(
                (
                    latents[:, :history_size].flatten(1),
                    action_history.flatten(1),
                ),
                dim=-1,
            )
            offsets = torch.randint(
                1,
                max_goal_offset + 1,
                (latents.shape[0],),
                generator=goal_generator,
            ).to(device)
            rows = torch.arange(latents.shape[0], device=device)
            goals = latents[rows, current_index + offsets]
            targets = monte_carlo_goal_tail_targets_for_offsets(
                latents,
                offsets,
                current_index=current_index,
                max_goal_offset=max_goal_offset,
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

            squared_error += float(
                (predictions.detach() - targets).pow(2).sum().cpu()
            )
            sample_count += int(targets.numel())
            global_step += 1

        validation = _validate_cached_value(
            value,
            validation_loader,
            device=device,
            history_size=history_size,
            current_index=current_index,
            max_goal_offset=max_goal_offset,
            gamma=gamma,
        )
        last_metrics = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_mse": squared_error / sample_count,
            **validation,
        }
        print(json.dumps(last_metrics, sort_keys=True), flush=True)
        _append_json_line(metrics_path, last_metrics)
        checkpoint = {
            "objective_version": 1,
            "method": "mc_gt_lewm",
            "value_state_dict": value.state_dict(),
            "value_config": {
                "history_dim": history_dim,
                "goal_dim": latent_dim,
                "hidden_dim": int(protocol["tail_value"]["hidden_dim"]),
                "history_size": history_size,
                "action_block_dim": action_block_dim,
                "max_goal_offset": max_goal_offset,
                "gamma": gamma,
                "objective": "supervised_mc",
            },
            "base_checkpoint_sha256": base_sha256,
            "epoch": epoch + 1,
            "global_step": global_step,
            "metrics": last_metrics,
        }
        checkpoint_dir = run_dir / "checkpoints"
        _save_checkpoint(checkpoint_dir / f"epoch_{epoch + 1:02d}.pt", checkpoint)
        _save_checkpoint(checkpoint_dir / "last.pt", checkpoint)
        if validation["validation_mse"] < best_validation_mse:
            best_validation_mse = float(validation["validation_mse"])
            _save_checkpoint(checkpoint_dir / "best.pt", checkpoint)

    result = {
        "method": "mc_gt_lewm",
        "display_name": "MC-GT-LeWM",
        "run_dir": str(run_dir),
        "global_step": global_step,
        "epochs": epochs,
        "elapsed_seconds": time.time() - started,
        "best_validation_mse": best_validation_mse,
        "initial_validation": initial_validation,
        "last_metrics": last_metrics,
        "best_checkpoint": str(run_dir / "checkpoints" / "best.pt"),
        "planner_connected": False,
    }
    write_json(run_dir / "training_result.json", result)
    return result


def ensure_cube_latent_cache(
    *,
    protocol: dict[str, Any],
    dataset_path: Path,
    dataset_source: dict[str, Any],
    world_model: torch.nn.Module,
    base_checkpoint_sha256: str,
    cache_dir: Path,
    device: torch.device,
) -> tuple[Path, dict[str, Any]]:
    """Encode each Cube transition once and persist an audited float32 array."""

    import stable_worldmodel as swm

    cache_cfg = protocol["latent_cache"]
    latent_dim = int(protocol["base_model"]["latent_dim"])
    transitions = int(protocol["dataset"]["expected_transitions"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = f"cube_lewm_{base_checkpoint_sha256[:12]}_float32"
    cache_path = cache_dir / f"{stem}.npy"
    manifest_path = cache_dir / f"{stem}.manifest.json"
    if cache_path.is_file() and manifest_path.is_file():
        with manifest_path.open() as stream:
            manifest = json.load(stream)
        _validate_latent_cache_manifest(
            manifest,
            cache_path=cache_path,
            transitions=transitions,
            latent_dim=latent_dim,
            base_checkpoint_sha256=base_checkpoint_sha256,
        )
        return cache_path, manifest
    if cache_path.exists() or manifest_path.exists():
        raise RuntimeError("Latent cache data and manifest must either both exist or neither.")

    frame_dataset = swm.data.load_dataset(
        str(dataset_path),
        format=dataset_source["format"],
        transform=None,
        num_steps=1,
        frameskip=1,
        keys_to_load=["pixels"],
    )
    if len(frame_dataset) != transitions:
        raise ValueError("Single-frame Cube dataset does not cover every transition.")
    for index in (0, transitions // 2, transitions - 1):
        episode, start = frame_dataset.clip_indices[index]
        if int(frame_dataset.offsets[episode]) + int(start) != index:
            raise ValueError("Single-frame Cube indexing is not globally contiguous.")

    workers = int(cache_cfg["workers"])
    loader_kwargs: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": bool(cache_cfg["pin_memory"]),
    }
    if workers:
        loader_kwargs.update(
            {
                "persistent_workers": True,
                "prefetch_factor": int(cache_cfg["prefetch_factor"]),
            }
        )
    loader = torch.utils.data.DataLoader(
        frame_dataset,
        batch_size=int(cache_cfg["batch_size"]),
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    image_cfg = protocol["image_preprocessing"]
    image_mean = torch.tensor(
        image_cfg["mean"], device=device, dtype=torch.float32
    ).reshape(1, 1, 3, 1, 1)
    image_std = torch.tensor(
        image_cfg["std"], device=device, dtype=torch.float32
    ).reshape(1, 1, 3, 1, 1)
    started = time.time()
    encode_latent_cache_file(
        world_model=world_model,
        loader=loader,
        destination=cache_path,
        transitions=transitions,
        latent_dim=latent_dim,
        image_mean=image_mean,
        image_std=image_std,
        image_size=int(image_cfg["size"]),
        log_every_batches=int(cache_cfg["log_every_batches"]),
        device=device,
    )
    manifest = {
        "schema_version": 1,
        "path": str(cache_path),
        "shape": [transitions, latent_dim],
        "dtype": "float32",
        "sha256": _sha256(cache_path),
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "dataset_path": str(dataset_path),
        "dataset_format": dataset_source["format"],
        "stable_worldmodel_version": protocol["runtime"][
            "stable_worldmodel_version"
        ],
        "image_preprocessing": image_cfg,
        "elapsed_seconds": time.time() - started,
    }
    write_json(manifest_path, manifest)
    return cache_path, manifest


def encode_latent_cache_file(
    *,
    world_model: torch.nn.Module,
    loader,
    destination: str | Path,
    transitions: int,
    latent_dim: int,
    image_mean: torch.Tensor,
    image_std: torch.Tensor,
    image_size: int,
    log_every_batches: int,
    device: torch.device,
) -> None:
    """Encode a sequential frame loader into one atomic float32 NPY cache."""

    if transitions <= 0 or latent_dim <= 0 or log_every_batches <= 0:
        raise ValueError("Cache dimensions and log interval must be positive.")
    destination = Path(destination)
    temporary_path = destination.parent / (
        f"{destination.stem}.tmp.{os.getpid()}{destination.suffix}"
    )
    cache = np.lib.format.open_memmap(
        temporary_path,
        mode="w+",
        dtype=np.float32,
        shape=(transitions, latent_dim),
    )
    cursor = 0
    started = time.time()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            pixels = batch["pixels"].to(device, non_blocking=True)
            pixels = preprocess_image_batch(
                pixels,
                mean=image_mean,
                std=image_std,
                size=image_size,
            )
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=True
            ):
                latents = world_model.encode({"pixels": pixels})["emb"]
            latents = latents[:, 0].float().cpu().numpy()
            if cursor + latents.shape[0] > transitions:
                raise RuntimeError("Latent loader produced more frames than expected.")
            cache[cursor : cursor + latents.shape[0]] = latents
            cursor += latents.shape[0]
            if (batch_index + 1) % log_every_batches == 0:
                print(
                    f"latent_cache frames={cursor}/{transitions} "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )
    if cursor != transitions:
        raise RuntimeError("Latent cache did not encode every expected transition.")
    cache.flush()
    del cache
    temporary_path.replace(destination)


@torch.inference_mode()
def _validate_cached_value(
    value: GoalTailValue,
    validation_loader,
    *,
    device: torch.device,
    history_size: int,
    current_index: int,
    max_goal_offset: int,
    gamma: float,
) -> dict[str, float | int]:
    value.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for batch in validation_loader:
        latents = batch["latents"].to(
            device, dtype=torch.float32, non_blocking=True
        )
        action_history = batch["action_history"].to(
            device, dtype=torch.float32, non_blocking=True
        )
        history = torch.cat(
            (
                latents[:, :history_size].flatten(1),
                action_history.flatten(1),
            ),
            dim=-1,
        )
        goals = latents[
            :, current_index + 1 : current_index + max_goal_offset + 1
        ]
        target = monte_carlo_goal_tail_targets(
            latents,
            current_index=current_index,
            max_goal_offset=max_goal_offset,
            gamma=gamma,
        )
        expanded_history = history.unsqueeze(1).expand(
            -1, max_goal_offset, -1
        )
        prediction = value(
            expanded_history.reshape(-1, history.shape[-1]),
            goals.reshape(-1, goals.shape[-1]),
        ).reshape_as(target)
        predictions.append(prediction.cpu().numpy().reshape(-1))
        targets.append(target.cpu().numpy().reshape(-1))
    prediction_array = np.concatenate(predictions)
    target_array = np.concatenate(targets)
    residual = prediction_array - target_array
    return {
        "validation_mse": float(np.mean(np.square(residual))),
        "validation_mae": float(np.mean(np.abs(residual))),
        "validation_spearman": spearman_correlation(
            prediction_array, target_array
        ),
        "validation_samples": int(target_array.size),
        "validation_target_mean": float(target_array.mean()),
        "validation_prediction_mean": float(prediction_array.mean()),
    }


def _validate_latent_cache_manifest(
    manifest: dict[str, Any],
    *,
    cache_path: Path,
    transitions: int,
    latent_dim: int,
    base_checkpoint_sha256: str,
) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("Latent cache manifest schema differs from version 1.")
    if manifest.get("shape") != [transitions, latent_dim]:
        raise ValueError("Latent cache shape differs from the formal protocol.")
    if manifest.get("dtype") != "float32":
        raise ValueError("Latent cache dtype differs from float32.")
    if manifest.get("base_checkpoint_sha256") != base_checkpoint_sha256:
        raise ValueError("Latent cache was produced by a different LeWM checkpoint.")
    array = np.load(cache_path, mmap_mode="r")
    if list(array.shape) != [transitions, latent_dim] or array.dtype != np.float32:
        raise ValueError("Latent cache file does not match its manifest.")


def _resolve_base_checkpoint(path: str | Path) -> tuple[str, Path, Path]:
    requested = Path(path).expanduser().resolve()
    checkpoint_dir = requested if requested.is_dir() else requested.parent
    weights = sorted(checkpoint_dir.glob("*.pt"))
    if len(weights) != 1:
        raise FileNotFoundError(
            "A local LeWM export must contain exactly one .pt weights file."
        )
    if checkpoint_dir.parent.name != "checkpoints":
        raise ValueError(
            "Expected Stable World Model's <cache>/checkpoints/<run> layout."
        )
    return checkpoint_dir.name, weights[0], checkpoint_dir.parent.parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _append_json_line(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(payload, sort_keys=True))
        stream.write("\n")


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
