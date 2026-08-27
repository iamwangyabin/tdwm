"""Offline supervised GoalTailValue training on frozen LeWM Cube latents."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml

from tdwm.adapters import prepare_cloud_runtime
from tdwm.methods.goal_tail_value import (
    GoalTailValue,
    build_goal_tail_history,
    monte_carlo_goal_tail_targets,
    select_goal_tail_samples,
)
from tdwm.training.block_sampler import BlockShuffleBatchSampler
from tdwm.training.cube_data import validate_cube_training_dataset
from tdwm.training.gt_lewm_support import (
    LeWMTransform,
    fit_column_stats,
    preprocess_image_batch,
    write_json,
)
from tdwm.training.lance_batch import (
    EpisodeStreamingBatchDataset,
    StrideAwareLanceDataset,
)


def load_goal_tail_protocol(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as stream:
        protocol = yaml.safe_load(stream)
    validate_goal_tail_protocol(protocol)
    return protocol


def validate_goal_tail_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema_version") != 1:
        raise ValueError("GoalTailValue requires schema_version 1.")
    if protocol.get("method") != "goal_tail_value":
        raise ValueError("This trainer only accepts method goal_tail_value.")
    if protocol.get("environment") != "cube":
        raise ValueError("GoalTailValue V0.1 is locked to OGBench Cube.")
    if protocol.get("runtime", {}).get("stable_worldmodel_version") != "0.1.1":
        raise ValueError("GoalTailValue is locked to stable-worldmodel 0.1.1.")

    base = protocol.get("base_model", {})
    if base.get("method") != "lewm" or base.get("frozen") is not True:
        raise ValueError("GoalTailValue requires a frozen LeWM base model.")
    if base.get("history_size") != 3:
        raise ValueError("The shipped LeWM Cube history size must remain 3.")

    sequence = protocol.get("sequence", {})
    history_size = sequence.get("history_frames", 0)
    max_goal_offset = sequence.get("max_goal_offset", 0)
    if history_size != 3 or max_goal_offset != 16:
        raise ValueError("V0.1 requires history 3 and future offsets 1 through 16.")
    if sequence.get("num_steps") != history_size + max_goal_offset:
        raise ValueError("num_steps must equal history_frames + max_goal_offset.")
    if sequence.get("frame_skip", 0) <= 0:
        raise ValueError("frame_skip must be positive.")

    tail = protocol.get("tail_value", {})
    if tail.get("objective") != "supervised_mc":
        raise ValueError("V0.1 only supports the supervised MC objective.")
    if tail.get("goal_sampling") != "uniform_future_offset":
        raise ValueError("V0.1 samples hindsight goals uniformly from future offsets.")
    if not 0.0 <= tail.get("gamma", -1.0) < 1.0:
        raise ValueError("tail_value.gamma must lie in [0, 1).")
    if tail.get("hidden_dim", 0) <= 0:
        raise ValueError("tail_value.hidden_dim must be positive.")
    forbidden = {"td_horizon", "target_ema_decay", "target_network", "policy"}
    if forbidden.intersection(tail):
        raise ValueError("V0.1 cannot configure TD, EMA, a target network, or policy.")

    split = protocol.get("split", {})
    if split.get("unit") != "episode":
        raise ValueError("Offline GoalTail validation must split whole episodes.")
    train_fraction = split.get("train_fraction", 0.0)
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("split.train_fraction must lie strictly between 0 and 1.")
    if not np.isclose(
        train_fraction + split.get("validation_fraction", 0.0), 1.0
    ):
        raise ValueError("Training and validation fractions must sum to one.")

    loader = protocol.get("loader", {})
    if loader.get("batch_size", 0) <= 0:
        raise ValueError("loader.batch_size must be positive.")
    if loader.get("episode_streaming") is not True:
        raise ValueError("The Cube V0.1 trainer requires episode streaming.")
    if loader.get("validation_block_size", 0) < loader["batch_size"]:
        raise ValueError("validation_block_size must be at least batch_size.")
    if loader["validation_block_size"] % loader["batch_size"]:
        raise ValueError("validation_block_size must be divisible by batch_size.")

    training = protocol.get("training", {})
    if training.get("epochs", 0) <= 0:
        raise ValueError("training.epochs must be positive.")
    if training.get("steps_per_epoch", 0) <= 0:
        raise ValueError("training.steps_per_epoch must be positive.")
    if training.get("validation_batches", 0) <= 0:
        raise ValueError("training.validation_batches must be positive.")
    if training.get("precision") not in {"32", "bf16"}:
        raise ValueError("training.precision must be 32 or bf16.")
    if not isinstance(training.get("validate_before_training"), bool):
        raise ValueError("training.validate_before_training must be true or false.")


def split_clip_indices_by_episode(
    clip_indices: Sequence[tuple[int, int]],
    *,
    episode_count: int,
    train_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split clips by episode so overlapping windows cannot cross the split."""

    if episode_count <= 1:
        raise ValueError("episode_count must be greater than one.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie strictly between 0 and 1.")
    rng = np.random.default_rng(seed)
    episode_order = rng.permutation(episode_count)
    train_count = int(np.floor(train_fraction * episode_count))
    train_episodes = np.sort(episode_order[:train_count])
    validation_episodes = np.sort(episode_order[train_count:])
    is_train = np.zeros(episode_count, dtype=bool)
    is_train[train_episodes] = True

    train_indices: list[int] = []
    validation_indices: list[int] = []
    for clip_index, (episode, _) in enumerate(clip_indices):
        destination = train_indices if is_train[int(episode)] else validation_indices
        destination.append(clip_index)
    return (
        np.asarray(train_indices, dtype=np.int64),
        np.asarray(validation_indices, dtype=np.int64),
        train_episodes.astype(np.int64),
        validation_episodes.astype(np.int64),
    )


def spearman_correlation(prediction: np.ndarray, target: np.ndarray) -> float:
    """Compute tie-aware Spearman correlation without an extra dependency."""

    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if prediction.shape != target.shape or prediction.size < 2:
        raise ValueError("prediction and target must contain matching samples.")
    prediction_rank = _average_ranks(prediction)
    target_rank = _average_ranks(target)
    if prediction_rank.std() == 0.0 or target_rank.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(prediction_rank, target_rank)[0, 1])


def freeze_world_model(world_model: torch.nn.Module) -> torch.nn.Module:
    world_model.eval()
    world_model.requires_grad_(False)
    return world_model


def build_value_optimizer(
    value: GoalTailValue, optimizer_config: dict[str, Any]
) -> torch.optim.Optimizer:
    """Build the only optimizer in V0.1; it owns value parameters and nothing else."""

    if optimizer_config.get("type") != "AdamW":
        raise ValueError("V0.1 currently requires AdamW.")
    return torch.optim.AdamW(
        value.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )


def train_goal_tail_cube(
    *,
    protocol_path: str | Path,
    dataset_path: str | Path,
    base_checkpoint_path: str | Path,
    output_dir: str | Path,
    seed: int,
    normalization_stats_path: str | Path | None = None,
    smoke: bool = False,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Train and validate a scalar MC tail head while LeWM stays frozen."""

    protocol = load_goal_tail_protocol(protocol_path)
    if seed not in protocol["seeds"]:
        raise ValueError(f"Seed {seed} is not in the configured seeds.")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive when provided.")

    dataset_path = Path(dataset_path).expanduser().resolve()
    dataset_source = validate_cube_training_dataset(
        dataset_path, protocol["dataset"]
    )
    base_name, base_weights, base_cache = _resolve_base_checkpoint(
        base_checkpoint_path
    )
    output_dir = Path(output_dir).expanduser().resolve()
    run_dir = output_dir / (f"seed_{seed}_smoke" if smoke else f"seed_{seed}")
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
        raise RuntimeError("GoalTailValue training requires one CUDA device.")

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    sequence = protocol["sequence"]
    dataset_cfg = protocol["dataset"]
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
        raise ValueError("Cube episode count differs from the locked protocol.")
    if int(np.asarray(dataset.lengths).sum()) != dataset_cfg["expected_transitions"]:
        raise ValueError("Cube transition count differs from the locked protocol.")

    if normalization_stats_path is None:
        statistics = fit_column_stats(
            dataset,
            ["action"],
            run_dir / "column_normalization.json",
        )
        normalization_source = str(run_dir / "column_normalization.json")
    else:
        normalization_source = str(
            Path(normalization_stats_path).expanduser().resolve()
        )
        with Path(normalization_source).open() as stream:
            loaded_statistics = json.load(stream)
        if "action" not in loaded_statistics:
            raise ValueError("Normalization statistics do not contain action.")
        statistics = {"action": loaded_statistics["action"]}
        write_json(run_dir / "column_normalization.json", statistics)
    dataset.transform = LeWMTransform(
        image=protocol["image_preprocessing"],
        columns=statistics,
        preprocess_images=False,
    )
    if dataset_source["format"] == "lance":
        dataset = StrideAwareLanceDataset(dataset)

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

    loader_cfg = protocol["loader"]
    train_batches = EpisodeStreamingBatchDataset(
        dataset,
        train_indices,
        batch_size=int(loader_cfg["batch_size"]),
        active_episodes=int(loader_cfg["episode_pool_size"]),
        read_episodes=int(loader_cfg["episode_read_size"]),
        cache_bytes=int(loader_cfg["episode_cache_bytes"]),
        prefetch_blocks=int(loader_cfg["episode_prefetch_blocks"]),
        seed=seed,
        drop_last=True,
        min_unique_episodes=int(loader_cfg["minimum_unique_episodes_per_batch"]),
    )
    train_loader = torch.utils.data.DataLoader(
        train_batches,
        batch_size=None,
        num_workers=0,
        pin_memory=bool(loader_cfg["pin_memory"]),
    )
    validation_set = torch.utils.data.Subset(
        dataset, validation_indices.tolist()
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_set,
        batch_sampler=BlockShuffleBatchSampler(
            validation_indices,
            batch_size=int(loader_cfg["batch_size"]),
            block_size=int(loader_cfg["validation_block_size"]),
            drop_last=False,
            shuffle_batches_within_block=False,
            shuffle_blocks=False,
        ),
        num_workers=int(loader_cfg["validation_workers"]),
        pin_memory=bool(loader_cfg["pin_memory"]),
        persistent_workers=int(loader_cfg["validation_workers"]) > 0,
        prefetch_factor=(
            int(loader_cfg["prefetch_factor"])
            if int(loader_cfg["validation_workers"]) > 0
            else None
        ),
    )

    device = torch.device("cuda")
    world_model = freeze_world_model(
        swm.wm.load_pretrained(base_name, cache_dir=str(base_cache)).to(device)
    )
    parameter_count = sum(parameter.numel() for parameter in world_model.parameters())
    expected_parameters = protocol["base_model"].get("parameters")
    if expected_parameters and parameter_count != expected_parameters:
        raise ValueError(
            f"Expected {expected_parameters} LeWM parameters, found {parameter_count}."
        )
    if int(world_model.predictor.num_frames) != sequence["history_frames"]:
        raise ValueError("The base checkpoint does not use LeWM history size 3.")

    latent_dim = int(protocol["base_model"]["latent_dim"])
    action_block_dim = int(dataset.get_dim("action")) * int(sequence["frame_skip"])
    history_dim = (
        int(sequence["history_frames"]) * latent_dim
        + (int(sequence["history_frames"]) - 1) * action_block_dim
    )
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

    image_cfg = protocol["image_preprocessing"]
    image_mean = torch.tensor(
        image_cfg["mean"], device=device, dtype=torch.float32
    ).reshape(1, 1, 3, 1, 1)
    image_std = torch.tensor(
        image_cfg["std"], device=device, dtype=torch.float32
    ).reshape(1, 1, 3, 1, 1)
    precision = protocol["training"]["precision"]
    use_bf16 = precision == "bf16"
    gamma = float(protocol["tail_value"]["gamma"])
    history_size = int(sequence["history_frames"])
    current_index = history_size - 1
    max_goal_offset = int(sequence["max_goal_offset"])
    goal_generator = torch.Generator().manual_seed(seed + 1)

    manifest = {
        "method": "goal_tail_value",
        "stage": "v0.1_offline_debug",
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
            "sha256": _sha256(base_weights),
            "parameters": parameter_count,
            "frozen": True,
        },
        "value": {
            "history_dim": history_dim,
            "goal_dim": latent_dim,
            "action_block_dim": action_block_dim,
            "parameters": sum(parameter.numel() for parameter in value.parameters()),
            "optimizer_parameter_source": "value.parameters()",
        },
        "normalization_source": normalization_source,
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

    training_cfg = protocol["training"]
    epochs = 1 if smoke else int(training_cfg["epochs"])
    steps_per_epoch = 2 if smoke else int(training_cfg["steps_per_epoch"])
    validation_batches = (
        2 if smoke else int(training_cfg["validation_batches"])
    )
    total_step_limit = max_steps
    global_step = 0
    best_validation_mse = float("inf")
    epoch_metrics: list[dict[str, Any]] = []
    initial_validation = None

    if bool(training_cfg["validate_before_training"]):
        initial_validation = _validate_value(
            world_model,
            value,
            validation_loader,
            device=device,
            image_mean=image_mean,
            image_std=image_std,
            image_size=int(image_cfg["size"]),
            use_bf16=use_bf16,
            history_size=history_size,
            current_index=current_index,
            max_goal_offset=max_goal_offset,
            gamma=gamma,
            max_batches=validation_batches,
        )
        initial_metrics = {
            "epoch": 0,
            "global_step": 0,
            **initial_validation,
        }
        epoch_metrics.append(initial_metrics)
        print(json.dumps(initial_metrics, sort_keys=True), flush=True)
        _append_json_line(run_dir / "metrics.jsonl", initial_metrics)

    for epoch in range(epochs):
        train_batches.set_epoch(epoch)
        value.train()
        train_squared_error = 0.0
        train_samples = 0
        for batch_index, batch in enumerate(train_loader):
            if batch_index >= steps_per_epoch:
                break
            latents, actions = _encode_batch(
                world_model,
                batch,
                device=device,
                image_mean=image_mean,
                image_std=image_std,
                image_size=int(image_cfg["size"]),
                use_bf16=use_bf16,
            )
            offsets = torch.randint(
                1,
                max_goal_offset + 1,
                (latents.shape[0],),
                generator=goal_generator,
            ).to(device)
            history, goal, target = select_goal_tail_samples(
                latents,
                actions,
                offsets,
                current_index=current_index,
                history_size=history_size,
                max_goal_offset=max_goal_offset,
                gamma=gamma,
            )
            prediction = value(history, goal)
            loss = torch.nn.functional.mse_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                value.parameters(), float(training_cfg["gradient_clip_norm"])
            )
            optimizer.step()

            sample_count = int(target.numel())
            train_squared_error += float(
                (prediction.detach() - target).pow(2).sum().cpu()
            )
            train_samples += sample_count
            global_step += 1
            if global_step % int(training_cfg["log_every_steps"]) == 0:
                print(
                    f"step={global_step} train_mse="
                    f"{train_squared_error / train_samples:.8f}",
                    flush=True,
                )
            if total_step_limit is not None and global_step >= total_step_limit:
                break

        validation = _validate_value(
            world_model,
            value,
            validation_loader,
            device=device,
            image_mean=image_mean,
            image_std=image_std,
            image_size=int(image_cfg["size"]),
            use_bf16=use_bf16,
            history_size=history_size,
            current_index=current_index,
            max_goal_offset=max_goal_offset,
            gamma=gamma,
            max_batches=validation_batches,
        )
        metrics = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_mse": train_squared_error / max(1, train_samples),
            **validation,
        }
        epoch_metrics.append(metrics)
        print(json.dumps(metrics, sort_keys=True), flush=True)
        _append_json_line(run_dir / "metrics.jsonl", metrics)

        checkpoint = {
            "objective_version": 1,
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
            "base_checkpoint_sha256": manifest["base_checkpoint"]["sha256"],
            "epoch": epoch + 1,
            "global_step": global_step,
            "metrics": metrics,
        }
        _save_checkpoint(run_dir / "checkpoints" / "last.pt", checkpoint)
        if validation["validation_mse"] < best_validation_mse:
            best_validation_mse = validation["validation_mse"]
            _save_checkpoint(run_dir / "checkpoints" / "best.pt", checkpoint)
        if total_step_limit is not None and global_step >= total_step_limit:
            break

    result = {
        "run_dir": str(run_dir),
        "global_step": global_step,
        "best_validation_mse": best_validation_mse,
        "initial_validation": initial_validation,
        "last_metrics": epoch_metrics[-1],
        "best_checkpoint": str(run_dir / "checkpoints" / "best.pt"),
        "planner_connected": False,
    }
    write_json(run_dir / "training_result.json", result)
    return result


@torch.inference_mode()
def _encode_batch(
    world_model: torch.nn.Module,
    batch: dict[str, Any],
    *,
    device: torch.device,
    image_mean: torch.Tensor,
    image_std: torch.Tensor,
    image_size: int,
    use_bf16: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    pixels = batch["pixels"].to(device, non_blocking=True)
    pixels = preprocess_image_batch(
        pixels,
        mean=image_mean,
        std=image_std,
        size=image_size,
    )
    actions = torch.nan_to_num(
        batch["action"].to(device, dtype=torch.float32, non_blocking=True),
        0.0,
    )
    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16
    ):
        latents = world_model.encode({"pixels": pixels})["emb"]
    return latents.float(), actions


@torch.inference_mode()
def _validate_value(
    world_model: torch.nn.Module,
    value: GoalTailValue,
    validation_loader,
    *,
    device: torch.device,
    image_mean: torch.Tensor,
    image_std: torch.Tensor,
    image_size: int,
    use_bf16: bool,
    history_size: int,
    current_index: int,
    max_goal_offset: int,
    gamma: float,
    max_batches: int,
) -> dict[str, float | int]:
    value.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for batch_index, batch in enumerate(validation_loader):
        if batch_index >= max_batches:
            break
        latents, actions = _encode_batch(
            world_model,
            batch,
            device=device,
            image_mean=image_mean,
            image_std=image_std,
            image_size=image_size,
            use_bf16=use_bf16,
        )
        history = build_goal_tail_history(
            latents,
            actions,
            current_index=current_index,
            history_size=history_size,
        )
        goal = latents[
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
            goal.reshape(-1, goal.shape[-1]),
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


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


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
