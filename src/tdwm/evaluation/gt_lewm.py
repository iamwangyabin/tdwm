"""Standalone GT-LeWM evaluation through Stable World Model's public API."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tdwm.adapters import prepare_cloud_runtime
from tdwm.adapters.gt_lewm import load_goal_tail_value, make_goal_tail_policy


REQUIRED_PLANNING_KEYS = {
    "horizon",
    "candidates",
    "iterations",
    "elites",
    "action_block",
    "receding_horizon",
    "executed_environment_steps_before_replanning",
    "episode_budget",
    "planning_seed",
    "solver_batch_size",
    "initial_variance",
    "warm_start",
}


def load_protocol(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as stream:
        protocol = yaml.safe_load(stream)
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema_version") != 1:
        raise ValueError("The experiment protocol must use schema_version 1.")
    if protocol.get("method") != "gt_lewm":
        raise ValueError("This evaluator only accepts standalone GT-LeWM.")
    if protocol.get("environment") != "cube":
        raise ValueError("This evaluator only accepts the OGBench-Cube environment.")

    tail = protocol.get("tail_value", {})
    if tail.get("objective_version") != 2:
        raise ValueError("GT-LeWM evaluation requires objective_version 2.")
    if tail.get("max_goal_offset", 0) <= 0:
        raise ValueError("GT-LeWM max_goal_offset must be positive.")
    if not 0 < tail.get("td_horizon", 0) <= tail["max_goal_offset"]:
        raise ValueError("GT-LeWM td_horizon must lie in [1, max_goal_offset].")
    if not 0.0 <= tail.get("gamma", -1.0) < 1.0:
        raise ValueError("GT-LeWM gamma must lie in [0, 1).")
    if tail.get("goal_source") != "all_future_states_in_clip":
        raise ValueError("GT-LeWM must use all future goals from each training clip.")
    if tail.get("continuation_policy") != "offline_dataset_behavior":
        raise ValueError("GT-LeWM requires the offline dataset behavior tail value.")

    planning = protocol.get("planning", {})
    missing = REQUIRED_PLANNING_KEYS - planning.keys()
    if missing:
        raise ValueError(f"Missing planning protocol keys: {sorted(missing)}")
    if planning["elites"] > planning["candidates"]:
        raise ValueError("CEM elites cannot exceed candidates.")
    if planning["receding_horizon"] > planning["horizon"]:
        raise ValueError("Receding horizon cannot exceed the CEM horizon.")
    if planning["horizon"] * planning["action_block"] > planning["episode_budget"]:
        raise ValueError("The planned action sequence exceeds the episode budget.")
    if planning["action_block"] != planning.get("frame_skip"):
        raise ValueError(
            "Each planned action block must match the training frame skip."
        )

    evaluation = protocol.get("evaluation", {})
    if evaluation.get("episodes", 0) <= 0:
        raise ValueError("Evaluation episodes must be positive.")
    if evaluation.get("goal_offset", 0) <= 0:
        raise ValueError("Goal offset must be positive.")
    planned_environment_steps = planning["horizon"] * planning["action_block"]
    if (
        evaluation.get("requires_tail_beyond_planning_horizon", False)
        and evaluation["goal_offset"] <= planned_environment_steps
    ):
        raise ValueError("The long-horizon protocol must place goals beyond the plan.")
    remaining_goal_steps = max(
        0, evaluation["goal_offset"] - planned_environment_steps
    )
    learned_tail_steps = tail["max_goal_offset"] * planning["frame_skip"]
    if remaining_goal_steps > learned_tail_steps:
        raise ValueError("The terminal tail query exceeds its training support.")
    expected_replan_steps = planning["receding_horizon"] * planning["action_block"]
    if (
        planning.get("executed_environment_steps_before_replanning")
        != expected_replan_steps
    ):
        raise ValueError(
            "The documented replanning interval does not match PlanConfig."
        )
    context = protocol.get("context", {})
    if context.get("predictor_recurrent_window") != context.get(
        "training_history_frames"
    ):
        raise ValueError("Training and recurrent prediction histories must match.")


def sample_start_goal_pairs(
    episode_lengths: np.ndarray,
    *,
    goal_offset: int,
    episodes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lengths = np.asarray(episode_lengths, dtype=np.int64)
    valid_per_episode = np.maximum(lengths - goal_offset, 0)
    cumulative = np.cumsum(valid_per_episode)
    total = int(cumulative[-1]) if cumulative.size else 0
    if total <= 0:
        raise ValueError("Dataset has no valid start/goal pairs.")
    if episodes > total:
        raise ValueError(
            f"Requested {episodes} evaluations but only {total} are sampleable."
        )
    rng = np.random.default_rng(seed)
    ranks = np.sort(rng.choice(total, size=episodes, replace=False))
    episode_indices = np.searchsorted(cumulative, ranks, side="right")
    previous = np.where(episode_indices == 0, 0, cumulative[episode_indices - 1])
    start_steps = ranks - previous
    return episode_indices, start_steps, ranks


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(_jsonable(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def _resolve_dataset_source(
    dataset_path: Path, dataset_config: dict[str, Any]
) -> dict[str, Any]:
    if dataset_path.is_file():
        expected_sizes = dataset_config.get(
            "accepted_size_bytes", [dataset_config.get("expected_size_bytes")]
        )
        actual_size = dataset_path.stat().st_size
        if actual_size not in expected_sizes:
            raise ValueError(
                f"Dataset size mismatch: expected one of {expected_sizes}, "
                f"found {actual_size}."
            )
        return {
            "path": str(dataset_path),
            "format": "hdf5",
            "size_bytes": actual_size,
            "conversion_manifest_path": None,
        }

    lance = dataset_config.get("lance")
    if (
        not dataset_path.is_dir()
        or dataset_path.suffix.lower() != ".lance"
        or lance is None
    ):
        raise FileNotFoundError(f"Cube dataset not found: {dataset_path}")

    manifest_path = Path(f"{dataset_path}{lance['manifest_suffix']}")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Audited Lance conversion manifest not found: {manifest_path}"
        )
    with manifest_path.open() as stream:
        manifest = json.load(stream)
    destination = manifest.get("destination", {})
    conversion = manifest.get("conversion", {})
    if destination.get("format") != "lance":
        raise ValueError("The Cube Lance manifest does not describe a Lance table.")
    if destination.get("image_codec") != lance["image_codec"]:
        raise ValueError("The Cube Lance manifest image codec differs from protocol.")
    if destination.get("jpeg_quality") != lance["jpeg_quality"]:
        raise ValueError("The Cube Lance manifest JPEG quality differs from protocol.")
    if conversion.get("stable_worldmodel_version") != "0.1.1":
        raise ValueError(
            "The Cube Lance manifest was not created by stable-worldmodel 0.1.1."
        )
    return {
        "path": str(dataset_path),
        "format": "lance",
        "size_bytes": destination.get("size_bytes"),
        "conversion_manifest_path": str(manifest_path),
    }


def _resolve_local_checkpoint(checkpoint_path: str | Path) -> tuple[str, Path, Path]:
    requested = Path(checkpoint_path).expanduser().resolve()
    checkpoint_dir = requested if requested.is_dir() else requested.parent
    weights = sorted(checkpoint_dir.glob("*.pt"))
    if len(weights) != 1:
        raise FileNotFoundError(
            "A local Stable World Model export must contain exactly one .pt file."
        )
    if checkpoint_dir.parent.name != "checkpoints":
        raise ValueError(
            "A local export must use the <cache_dir>/checkpoints/<run_name> layout."
        )
    return checkpoint_dir.name, weights[0], checkpoint_dir.parent.parent


def _resolve_value_checkpoint(checkpoint_path: str | Path) -> Path:
    requested = Path(checkpoint_path).expanduser().resolve()
    if requested.is_file():
        return requested
    if requested.is_dir():
        weights = sorted(requested.glob("*.pt"))
        if len(weights) == 1:
            return weights[0]
    raise FileNotFoundError(
        "The GT-LeWM value checkpoint must be a .pt file or a directory containing one."
    )


def _load_action_processor(dataset: Any, path: Path):
    from sklearn.preprocessing import StandardScaler

    if path.is_file():
        with path.open() as stream:
            stats = json.load(stream)
        processor = StandardScaler()
        processor.mean_ = np.asarray(stats["mean"], dtype=np.float64)
        processor.scale_ = np.asarray(stats["scale"], dtype=np.float64)
        processor.var_ = np.asarray(stats["variance"], dtype=np.float64)
        processor.n_features_in_ = len(processor.mean_)
        processor.n_samples_seen_ = int(stats["samples"])
        return processor, stats

    processor = StandardScaler().fit(dataset.get_col_data("action"))
    stats = {
        "mean": processor.mean_,
        "scale": processor.scale_,
        "variance": processor.var_,
        "samples": int(processor.n_samples_seen_),
    }
    _write_json(path, stats)
    return processor, stats


def _validate_value_config(
    value_config: dict[str, Any], protocol: dict[str, Any]
) -> None:
    expected = protocol["tail_value"]
    if int(value_config["objective_version"]) != int(expected["objective_version"]):
        raise ValueError(
            "The value checkpoint objective version differs from protocol."
        )
    if int(value_config["embed_dim"]) != int(protocol["model"]["embed_dim"]):
        raise ValueError(
            "The value checkpoint embedding dimension differs from protocol."
        )
    if int(value_config["hidden_dim"]) != int(expected["hidden_dim"]):
        raise ValueError("The value checkpoint hidden dimension differs from protocol.")
    if not np.isclose(float(value_config["gamma"]), float(expected["gamma"])):
        raise ValueError("The value checkpoint gamma differs from protocol.")
    if int(value_config["max_goal_offset"]) != int(expected["max_goal_offset"]):
        raise ValueError("The value checkpoint goal offset differs from protocol.")
    if int(value_config["td_horizon"]) != int(expected["td_horizon"]):
        raise ValueError("The value checkpoint TD horizon differs from protocol.")
    if value_config.get("continuation_policy") != expected["continuation_policy"]:
        raise ValueError(
            "The value checkpoint continuation policy differs from protocol."
        )


def evaluate_gt_lewm(
    *,
    protocol_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    base_checkpoint_path: str | Path,
    value_checkpoint_path: str | Path,
    video: bool = False,
    smoke: bool = False,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    if smoke:
        protocol["id"] = f"{protocol['id']}_smoke"
        protocol["evaluation"]["episodes"] = 1
        protocol["planning"].update(
            {"candidates": 8, "iterations": 1, "elites": 2, "episode_budget": 25}
        )
        validate_protocol(protocol)

    dataset_path = Path(dataset_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_source = _resolve_dataset_source(dataset_path, protocol["dataset"])

    compatibility = prepare_cloud_runtime()
    import stable_worldmodel as swm
    import torch
    from torchvision.transforms import v2 as transforms

    package_version = importlib.metadata.version("stable-worldmodel")
    expected_version = protocol["runtime"]["stable_worldmodel_version"]
    if package_version != expected_version:
        raise RuntimeError(
            f"Expected stable-worldmodel {expected_version}, found {package_version}."
        )

    cache_root = Path(
        os.environ.get("STABLEWM_HOME", str(Path.home() / ".stable_worldmodel"))
    ).expanduser()
    base_name, base_file, base_cache = _resolve_local_checkpoint(base_checkpoint_path)
    value_file = _resolve_value_checkpoint(value_checkpoint_path)
    value, value_config = load_goal_tail_value(value_file, map_location="cuda")
    _validate_value_config(value_config, protocol)
    base_hash = _sha256(base_file)
    value_hash = _sha256(value_file)

    dataset_cfg = protocol["dataset"]
    dataset = swm.data.load_dataset(
        str(dataset_path),
        format=dataset_source["format"],
        keys_to_load=list(dataset_cfg["keys_to_load"]),
    )
    actual_episodes = len(dataset.lengths)
    actual_transitions = int(np.asarray(dataset.lengths).sum())
    if actual_episodes != dataset_cfg["expected_episodes"]:
        raise ValueError(
            f"Expected {dataset_cfg['expected_episodes']} episodes, "
            f"found {actual_episodes}."
        )
    if actual_transitions != dataset_cfg["expected_transitions"]:
        raise ValueError(
            f"Expected {dataset_cfg['expected_transitions']} transitions, "
            f"found {actual_transitions}."
        )

    evaluation_cfg = protocol["evaluation"]
    planning_cfg = protocol["planning"]
    episode_indices, start_steps, valid_ranks = sample_start_goal_pairs(
        np.asarray(dataset.lengths),
        goal_offset=evaluation_cfg["goal_offset"],
        episodes=evaluation_cfg["episodes"],
        seed=planning_cfg["planning_seed"],
    )
    selection = {
        "episode_indices": episode_indices,
        "start_steps": start_steps,
        "goal_steps": start_steps + evaluation_cfg["goal_offset"],
        "valid_row_ranks": valid_ranks,
    }
    _write_json(output_dir / "episode_selection.json", selection)

    action_processor, action_stats = _load_action_processor(
        dataset, output_dir / "action_normalization.json"
    )
    model = (
        swm.wm.load_pretrained(base_name, cache_dir=str(base_cache))
        .to("cuda")
        .eval()
    )
    model.requires_grad_(False)
    expected_parameters = protocol["checkpoint"].get("parameters")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if expected_parameters and parameter_count != expected_parameters:
        raise ValueError(
            f"Expected {expected_parameters} model parameters, found {parameter_count}."
        )

    image_stats = protocol["image_preprocessing"]
    image_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=image_stats["mean"], std=image_stats["std"]),
            transforms.Resize(size=protocol["world"]["image_size"]),
        ]
    )
    policy = make_goal_tail_policy(
        world_model=model,
        value=value,
        planning=planning_cfg,
        tail_value=protocol["tail_value"],
        history_size=protocol["context"]["plan_config_history_len"],
        process={"action": action_processor},
        transform={"pixels": image_transform, "goal": image_transform},
        device="cuda",
    )

    runtime_manifest = {
        "protocol": protocol,
        "protocol_path": str(Path(protocol_path).resolve()),
        "dataset": {
            **dataset_source,
            "episodes": actual_episodes,
            "transitions": actual_transitions,
        },
        "checkpoints": {
            "base_name": base_name,
            "base_path": str(base_file),
            "base_cache_dir": str(base_cache),
            "base_sha256": base_hash,
            "value_path": str(value_file),
            "value_sha256": value_hash,
            "value_config": value_config,
        },
        "selection": selection,
        "normalization": {"action": action_stats},
        "runtime": {
            "stable_worldmodel": package_version,
            "stable_pretraining": importlib.metadata.version("stable-pretraining"),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "tdwm_git_revision": _git_revision(),
            "cuda_device": torch.cuda.get_device_name(0),
            "stablewm_home": str(cache_root),
            "compatibility_adapter": compatibility,
        },
    }
    _write_json(output_dir / "protocol_manifest.json", runtime_manifest)

    world_cfg = protocol["world"]
    world = swm.World(
        world_cfg["env_name"],
        num_envs=evaluation_cfg["episodes"],
        image_shape=(world_cfg["image_size"], world_cfg["image_size"]),
        max_episode_steps=planning_cfg["episode_budget"],
        env_type=world_cfg["env_type"],
        ob_type=world_cfg["ob_type"],
        multiview=world_cfg["multiview"],
        width=world_cfg["image_size"],
        height=world_cfg["image_size"],
        visualize_info=world_cfg["visualize_info"],
        terminate_at_goal=world_cfg["terminate_at_goal"],
    )
    world.set_policy(policy)

    callables = [
        {
            "method": "set_state",
            "args": {"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}},
        },
        {
            "method": "set_target_pos",
            "args": {
                "cube_id": {"value": 0, "in_dataset": False},
                "target_pos": {"value": "goal_privileged_block_0_pos"},
                "target_quat": {"value": "goal_privileged_block_0_quat"},
            },
        },
    ]

    video_path = output_dir / "videos" if video else None
    started = time.time()
    try:
        with torch.inference_mode():
            metrics = world.evaluate(
                dataset=dataset,
                episodes_idx=episode_indices.tolist(),
                start_steps=start_steps.tolist(),
                goal_offset=evaluation_cfg["goal_offset"],
                eval_budget=planning_cfg["episode_budget"],
                callables=callables,
                video=video_path,
            )
    finally:
        world.close()
    result = {
        "metrics": metrics,
        "elapsed_seconds": time.time() - started,
        "parameter_count": parameter_count,
        "method": "gt_lewm",
        "smoke": smoke,
        "protocol_manifest": str(output_dir / "protocol_manifest.json"),
    }
    _write_json(output_dir / "results.json", result)
    return _jsonable(result)
