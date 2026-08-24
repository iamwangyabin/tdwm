"""Controlled Cube evaluation for directed Successor-Geometry LeWM."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tdwm.adapters import prepare_cloud_runtime
from tdwm.adapters.successor_geometry_lewm import (
    METHOD,
    POLICY_AUXILIARY_METHODS,
    RESIDUAL_POLICY_METHOD,
    SUPPORTED_METHODS,
    load_successor_geometry_checkpoint,
    make_successor_geometry_policy,
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


def load_successor_geometry_evaluation_protocol(
    path: str | Path,
) -> dict[str, Any]:
    with Path(path).open() as stream:
        protocol = yaml.safe_load(stream)
    validate_successor_geometry_evaluation_protocol(protocol)
    return protocol


def validate_successor_geometry_evaluation_protocol(
    protocol: dict[str, Any],
) -> None:
    method = protocol.get("method")
    if protocol.get("schema_version") != 1 or method not in SUPPORTED_METHODS:
        raise ValueError("Successor geometry evaluation requires its schema 1 method.")
    if protocol.get("environment") != "cube" or protocol.get("stage") != "planner_evaluation":
        raise ValueError("Successor geometry evaluation is locked to Cube planning.")
    if protocol.get("runtime", {}).get("stable_worldmodel_version") != "0.1.1":
        raise ValueError("Evaluation requires stable-worldmodel 0.1.1.")

    geometry = protocol.get("geometry", {})
    expected = {
        "objective_version": 1,
        "architecture": "dual_mlp_directed_cosine",
        "query_sources": ["real_terminal", "predicted_terminal"],
        "goal_conditioning": "future_pairs_only",
        "negative_sampling": "cross_episode_in_batch",
        "same_episode_negatives": "masked",
        "reward": "none",
        "policy": (
            "expert_action_auxiliary_training_only"
            if method in POLICY_AUXILIARY_METHODS
            else "none"
        ),
        "td_bootstrap": False,
        "planning_cost": "one_minus_directed_cosine",
    }
    for key, value in expected.items():
        if geometry.get(key) != value:
            raise ValueError(f"geometry.{key} must be {value!r}.")
    if method in POLICY_AUXILIARY_METHODS:
        expected_transition = (
            "current_latent_plus_delta"
            if method == RESIDUAL_POLICY_METHOD
            else "absolute_next_latent"
        )
        if geometry.get("latent_transition") != expected_transition:
            raise ValueError("The evaluation latent transition is inconsistent.")
        if geometry.get("policy_used_at_inference") is not False:
            raise ValueError(
                "The expert-action auxiliary must stay disabled at inference."
            )
    for key in (
        "embed_dim",
        "projection_dim",
        "hidden_dim",
        "history_size",
        "rollout_horizon",
        "max_future_offset",
    ):
        if int(geometry.get(key, 0)) <= 0:
            raise ValueError(f"geometry.{key} must be positive.")
    if float(geometry.get("temperature", 0.0)) <= 0.0:
        raise ValueError("geometry.temperature must be positive.")
    if not 0.0 < float(geometry.get("gamma", 0.0)) <= 1.0:
        raise ValueError("geometry.gamma must lie in (0, 1].")

    planning = protocol.get("planning", {})
    missing = REQUIRED_PLANNING_KEYS - planning.keys()
    if missing:
        raise ValueError(f"Missing planning keys: {sorted(missing)}")
    if planning["elites"] > planning["candidates"]:
        raise ValueError("CEM elites cannot exceed candidates.")
    if planning["receding_horizon"] > planning["horizon"]:
        raise ValueError("Receding horizon cannot exceed planning horizon.")
    if planning["horizon"] > geometry["rollout_horizon"]:
        raise ValueError("Planning exceeds the trained rollout horizon.")
    if planning["action_block"] != planning.get("frame_skip"):
        raise ValueError("Planning action blocks must match training frame skip.")
    if planning["horizon"] * planning["action_block"] > planning["episode_budget"]:
        raise ValueError("The explicit plan exceeds the episode budget.")
    if planning.get("initial_distribution") != "cem_gaussian_no_actor":
        raise ValueError("The method must not use a learned actor warm start.")
    evaluation = protocol.get("evaluation", {})
    if min(int(evaluation.get("episodes", 0)), int(evaluation.get("goal_offset", 0))) <= 0:
        raise ValueError("Evaluation episodes and goal offset must be positive.")


def configure_successor_geometry_evaluation_mode(
    protocol: dict[str, Any],
    *,
    smoke: bool,
    pilot: bool,
) -> dict[str, Any]:
    if smoke and pilot:
        raise ValueError("Smoke and pilot modes are mutually exclusive.")
    configured = deepcopy(protocol)
    if smoke:
        configured["id"] = f"{configured['id']}_smoke"
        configured["evaluation"]["episodes"] = 1
        configured["planning"].update(
            {"candidates": 8, "iterations": 1, "elites": 2, "episode_budget": 25}
        )
    elif pilot:
        configured["id"] = f"{configured['id']}_pilot"
        configured["evaluation"]["episodes"] = 10
        configured["planning"].update(
            {
                "candidates": 128,
                "iterations": 10,
                "elites": 16,
                "episode_budget": 100,
            }
        )
    validate_successor_geometry_evaluation_protocol(configured)
    return configured


def _resolve_geometry_checkpoint(path: str | Path) -> Path:
    requested = Path(path).expanduser().resolve()
    if requested.is_file():
        return requested
    if requested.is_dir():
        files = sorted(requested.glob("*.pt"))
        if len(files) == 1:
            return files[0]
    raise FileNotFoundError(
        "A successor-geometry deployment checkpoint must be a .pt file or a "
        "directory containing exactly one .pt file."
    )


def _validate_geometry_config(
    config: dict[str, Any], protocol: dict[str, Any]
) -> None:
    geometry = protocol["geometry"]
    keys = (
        "objective_version",
        "architecture",
        "embed_dim",
        "projection_dim",
        "hidden_dim",
        "history_size",
        "rollout_horizon",
        "max_future_offset",
        "query_sources",
        "goal_conditioning",
        "negative_sampling",
        "same_episode_negatives",
        "reward",
        "policy",
        "td_bootstrap",
    )
    if protocol["method"] in POLICY_AUXILIARY_METHODS:
        keys += ("latent_transition", "policy_used_at_inference")
    for key in keys:
        if config.get(key) != geometry.get(key):
            raise ValueError(f"Geometry checkpoint {key} differs from protocol.")
    for key in ("temperature", "gamma"):
        if not np.isclose(float(config[key]), float(geometry[key])):
            raise ValueError(f"Geometry checkpoint {key} differs from protocol.")


def _validate_checkpoint_pair(
    *, base_name: str, base_file: Path, geometry_config: dict[str, Any]
) -> None:
    if geometry_config.get("base_export_run_name") != base_name:
        raise ValueError("The geometry and LeWM exports came from different epochs.")
    expected_hash = geometry_config.get("base_checkpoint_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("The geometry checkpoint is missing its paired LeWM hash.")
    if _sha256(base_file) != expected_hash:
        raise ValueError("The geometry checkpoint does not match the LeWM weights.")


def evaluate_successor_geometry_lewm(
    *,
    protocol_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    base_checkpoint_path: str | Path,
    geometry_checkpoint_path: str | Path,
    video: bool = False,
    smoke: bool = False,
    pilot: bool = False,
) -> dict[str, Any]:
    protocol = configure_successor_geometry_evaluation_mode(
        load_successor_geometry_evaluation_protocol(protocol_path),
        smoke=smoke,
        pilot=pilot,
    )
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
    geometry_file = _resolve_geometry_checkpoint(geometry_checkpoint_path)
    geometry, geometry_config, payload = load_successor_geometry_checkpoint(
        geometry_file,
        map_location=device,
        expected_method=protocol["method"],
    )
    _validate_geometry_config(geometry_config, protocol)
    _validate_checkpoint_pair(
        base_name=base_name,
        base_file=base_file,
        geometry_config=geometry_config,
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
    geometry = geometry.to(device).eval()
    geometry.requires_grad_(False)
    world_model_parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    geometry_parameter_count = sum(
        parameter.numel() for parameter in geometry.parameters()
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
    policy = make_successor_geometry_policy(
        world_model=model,
        geometry=geometry,
        planning=planning,
        geometry_config=geometry_config,
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
    _write_json(
        output_dir / "protocol_manifest.json",
        {
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
                "geometry_path": str(geometry_file),
                "geometry_sha256": _sha256(geometry_file),
                "geometry_config": geometry_config,
            },
            "selection": selection,
            "normalization": {"action": action_stats},
            "runtime": runtime,
        },
    )

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
        "world_model_parameter_count": world_model_parameter_count,
        "geometry_parameter_count": geometry_parameter_count,
        "method": protocol["method"],
        "smoke": smoke,
        "pilot": pilot,
        "protocol_manifest": str(output_dir / "protocol_manifest.json"),
    }
    _write_json(output_dir / "results.json", result)
    return _jsonable(result)


__all__ = [
    "configure_successor_geometry_evaluation_mode",
    "evaluate_successor_geometry_lewm",
    "load_successor_geometry_evaluation_protocol",
    "validate_successor_geometry_evaluation_protocol",
]
