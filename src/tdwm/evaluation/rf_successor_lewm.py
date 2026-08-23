"""Controlled Cube evaluation for reward-free Successor-LeWM."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tdwm.adapters import (
    load_rf_successor_checkpoint,
    make_rf_successor_policy,
    prepare_cloud_runtime,
)
from tdwm.evaluation.lewm_checkpoint import (
    REQUIRED_PLANNING_KEYS,
    _git_revision,
    _jsonable,
    _resolve_dataset_source,
    _resolve_local_export_checkpoint,
    _sha256,
    _write_json,
    sample_start_goal_pairs,
)
from tdwm.evaluation.mc_gt_lewm import _load_action_processor

METHOD = "rf_successor_lewm"


def load_rf_successor_evaluation_protocol(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as stream:
        protocol = yaml.safe_load(stream)
    validate_rf_successor_evaluation_protocol(protocol)
    return protocol


def validate_rf_successor_evaluation_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema_version") != 1 or protocol.get("method") != METHOD:
        raise ValueError("RF-Successor-LeWM evaluation requires schema 1.")
    if protocol.get("environment") != "cube" or protocol.get("stage") != "planner_evaluation":
        raise ValueError("RF-Successor-LeWM evaluation is locked to Cube planning.")
    if protocol.get("runtime", {}).get("stable_worldmodel_version") != "0.1.1":
        raise ValueError("Evaluation requires stable-worldmodel 0.1.1.")

    successor = protocol.get("successor", {})
    expected = {
        "objective_version": 1,
        "architecture": "causal_gru_action_prefix",
        "feature_basis": "augmented_latent_squared_distance",
        "horizon_normalization": "discounted_prefix_mean",
        "target": "direct_monte_carlo",
        "action_conditioning": "causal_prefix",
        "goal_conditioning": "none",
        "continuation_policy": "none",
        "td_bootstrap": False,
    }
    for key, value in expected.items():
        if successor.get(key) != value:
            raise ValueError(f"successor.{key} must be {value!r}.")
    if int(successor.get("history_size", 0)) <= 0:
        raise ValueError("successor.history_size must be positive.")
    if int(successor.get("max_horizon", 0)) <= 0:
        raise ValueError("successor.max_horizon must be positive.")
    if not 0.0 <= float(successor.get("gamma", -1.0)) <= 1.0:
        raise ValueError("successor.gamma must lie in [0, 1].")
    if min(
        float(successor.get("planning_weight", -1.0)),
        float(successor.get("terminal_weight", -1.0)),
    ) < 0.0:
        raise ValueError("Planning cost weights cannot be negative.")

    planning = protocol.get("planning", {})
    missing = REQUIRED_PLANNING_KEYS - planning.keys()
    if missing:
        raise ValueError(f"Missing planning keys: {sorted(missing)}")
    if planning["elites"] > planning["candidates"]:
        raise ValueError("CEM elites cannot exceed candidates.")
    if planning["receding_horizon"] > planning["horizon"]:
        raise ValueError("Receding horizon cannot exceed planning horizon.")
    if planning["horizon"] > successor["max_horizon"]:
        raise ValueError("Planning exceeds the trained successor horizon.")
    if planning["action_block"] != planning.get("frame_skip"):
        raise ValueError("Planning action blocks must match training frame skip.")
    if planning["horizon"] * planning["action_block"] > planning["episode_budget"]:
        raise ValueError("The explicit plan exceeds the episode budget.")
    if planning.get("initial_distribution") != "cem_gaussian_no_actor":
        raise ValueError("RF-Successor-LeWM must not use a learned actor warm start.")
    evaluation = protocol.get("evaluation", {})
    if min(int(evaluation.get("episodes", 0)), int(evaluation.get("goal_offset", 0))) <= 0:
        raise ValueError("Evaluation episodes and goal offset must be positive.")


def _resolve_successor_checkpoint(path: str | Path) -> Path:
    requested = Path(path).expanduser().resolve()
    if requested.is_file():
        return requested
    if requested.is_dir():
        files = sorted(requested.glob("*.pt"))
        if len(files) == 1:
            return files[0]
    raise FileNotFoundError(
        "An RF-Successor-LeWM deployment checkpoint must be a .pt file or a "
        "directory containing exactly one .pt file."
    )


def _validate_successor_config(
    config: dict[str, Any], protocol: dict[str, Any]
) -> None:
    successor = protocol["successor"]
    expected = {
        "objective_version": successor["objective_version"],
        "architecture": successor["architecture"],
        "embed_dim": protocol["model"]["embed_dim"],
        "history_size": successor["history_size"],
        "hidden_dim": successor["hidden_dim"],
        "max_horizon": successor["max_horizon"],
        "feature_basis": successor["feature_basis"],
        "horizon_normalization": successor["horizon_normalization"],
        "target": successor["target"],
        "action_conditioning": successor["action_conditioning"],
        "goal_conditioning": successor["goal_conditioning"],
        "continuation_policy": successor["continuation_policy"],
        "td_bootstrap": successor["td_bootstrap"],
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Successor checkpoint {key} differs from protocol.")
    for key in ("gamma", "planning_weight", "terminal_weight"):
        if not np.isclose(float(config[key]), float(successor[key])):
            raise ValueError(f"Successor checkpoint {key} differs from protocol.")


def _validate_checkpoint_pair(
    *,
    base_name: str,
    base_file: Path,
    successor_config: dict[str, Any],
) -> None:
    if successor_config.get("base_export_run_name") != base_name:
        raise ValueError("The successor and LeWM exports came from different epochs.")
    expected_hash = successor_config.get("base_checkpoint_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("The successor checkpoint is missing its paired LeWM hash.")
    if _sha256(base_file) != expected_hash:
        raise ValueError("The successor checkpoint does not match the LeWM weights.")


def evaluate_rf_successor_lewm(
    *,
    protocol_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    base_checkpoint_path: str | Path,
    successor_checkpoint_path: str | Path,
    video: bool = False,
    smoke: bool = False,
) -> dict[str, Any]:
    protocol = load_rf_successor_evaluation_protocol(protocol_path)
    if smoke:
        protocol["id"] = f"{protocol['id']}_smoke"
        protocol["evaluation"]["episodes"] = 1
        protocol["planning"].update(
            {"candidates": 8, "iterations": 1, "elites": 2, "episode_budget": 25}
        )
        validate_rf_successor_evaluation_protocol(protocol)

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_name, base_file, base_cache = _resolve_local_export_checkpoint(
        base_checkpoint_path
    )
    successor_file = _resolve_successor_checkpoint(successor_checkpoint_path)
    head, head_config, payload = load_rf_successor_checkpoint(
        successor_file, map_location=device
    )
    _validate_successor_config(head_config, protocol)
    _validate_checkpoint_pair(
        base_name=base_name,
        base_file=base_file,
        successor_config=head_config,
    )

    dataset_cfg = protocol["dataset"]
    dataset = swm.data.load_dataset(
        str(dataset_path),
        format=dataset_source["format"],
        keys_to_load=list(dataset_cfg["keys_to_load"]),
    )
    actual_episodes = len(dataset.lengths)
    actual_transitions = int(np.asarray(dataset.lengths).sum())
    if actual_episodes != dataset_cfg["expected_episodes"]:
        raise ValueError("Dataset episode count differs from protocol.")
    if actual_transitions != dataset_cfg["expected_transitions"]:
        raise ValueError("Dataset transition count differs from protocol.")
    expected_action_dim = int(dataset.get_dim("action")) * int(
        protocol["planning"]["action_block"]
    )
    if int(head_config["action_dim"]) != expected_action_dim:
        raise ValueError("The successor action-block dimension is incompatible.")

    evaluation = protocol["evaluation"]
    planning = protocol["planning"]
    episode_indices, start_steps, valid_ranks = sample_start_goal_pairs(
        np.asarray(dataset.lengths),
        goal_offset=evaluation["goal_offset"],
        episodes=evaluation["episodes"],
        seed=planning["planning_seed"],
    )
    selection = {
        "episode_indices": episode_indices,
        "start_steps": start_steps,
        "goal_steps": start_steps + evaluation["goal_offset"],
        "valid_row_ranks": valid_ranks,
    }
    _write_json(output_dir / "episode_selection.json", selection)
    action_processor, action_stats = _load_action_processor(
        dataset, output_dir / "action_normalization.json"
    )

    model = swm.wm.load_pretrained(base_name, cache_dir=str(base_cache)).to(device)
    model.load_state_dict(payload["world_model_state_dict"])
    model.eval()
    model.requires_grad_(False)
    head = head.to(device).eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    successor_parameter_count = sum(
        parameter.numel() for parameter in head.parameters()
    )

    image = protocol["image_preprocessing"]
    image_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=image["mean"], std=image["std"]),
            transforms.Resize(size=protocol["world"]["image_size"]),
        ]
    )
    policy = make_rf_successor_policy(
        world_model=model,
        successor=head,
        planning=planning,
        successor_config=protocol["successor"],
        process={"action": action_processor},
        transform={"pixels": image_transform, "goal": image_transform},
        device=device,
    )

    runtime = {
        "stable_worldmodel": package_version,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tdwm_git_revision": _git_revision(),
        "device": device,
        "stablewm_home": os.environ.get("STABLEWM_HOME"),
        "compatibility_adapter": compatibility,
    }
    if torch.cuda.is_available():
        runtime["cuda_device"] = torch.cuda.get_device_name(0)
    manifest = {
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
            "base_sha256": _sha256(base_file),
            "successor_path": str(successor_file),
            "successor_sha256": _sha256(successor_file),
            "successor_config": head_config,
        },
        "selection": selection,
        "normalization": {"action": action_stats},
        "runtime": runtime,
    }
    _write_json(output_dir / "protocol_manifest.json", manifest)

    world_cfg = protocol["world"]
    world = swm.World(
        world_cfg["env_name"],
        num_envs=evaluation["episodes"],
        image_shape=(world_cfg["image_size"], world_cfg["image_size"]),
        max_episode_steps=planning["episode_budget"],
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
    started = time.time()
    try:
        with torch.inference_mode():
            metrics = world.evaluate(
                dataset=dataset,
                episodes_idx=episode_indices.tolist(),
                start_steps=start_steps.tolist(),
                goal_offset=evaluation["goal_offset"],
                eval_budget=planning["episode_budget"],
                callables=callables,
                video=output_dir / "videos" if video else None,
            )
    finally:
        world.close()
    result = {
        "metrics": metrics,
        "elapsed_seconds": time.time() - started,
        "world_model_parameter_count": parameter_count,
        "successor_parameter_count": successor_parameter_count,
        "method": METHOD,
        "smoke": smoke,
        "protocol_manifest": str(output_dir / "protocol_manifest.json"),
    }
    _write_json(output_dir / "results.json", result)
    return _jsonable(result)


__all__ = [
    "evaluate_rf_successor_lewm",
    "load_rf_successor_evaluation_protocol",
    "validate_rf_successor_evaluation_protocol",
]
