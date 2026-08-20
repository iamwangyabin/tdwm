"""Evaluation entry point for independent LS-LeWM checkpoints."""

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
    load_local_successor_heads,
    make_local_successor_policy,
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


def load_ls_evaluation_protocol(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as stream:
        protocol = yaml.safe_load(stream)
    validate_ls_evaluation_protocol(protocol)
    return protocol


def validate_ls_evaluation_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema_version") != 1:
        raise ValueError("LS-LeWM evaluation requires schema_version 1.")
    if protocol.get("method") != "ls_lewm" or protocol.get("environment") != "cube":
        raise ValueError("This evaluator only accepts LS-LeWM on Cube.")
    if protocol.get("runtime", {}).get("stable_worldmodel_version") != "0.1.1":
        raise ValueError("LS-LeWM evaluation requires stable-worldmodel 0.1.1.")
    successor = protocol.get("successor", {})
    if successor.get("objective_version") != 1:
        raise ValueError("LS-LeWM evaluation requires objective_version 1.")
    if successor.get("continuation_policy") != "hindsight_gcbc":
        raise ValueError("The continuation policy must match LS-LeWM training.")
    if successor.get("feature_basis") != "augmented_latent_squared_distance":
        raise ValueError("The successor feature basis differs from training.")
    if successor.get("goal_offset_weighting") != "uniform_offsets":
        raise ValueError("The successor goal-offset weighting differs from training.")
    if successor.get("terminal_condition") != "next_state_is_hindsight_goal":
        raise ValueError("The successor terminal condition differs from training.")
    if not 0.0 <= successor.get("gamma", -1.0) < 1.0:
        raise ValueError("successor.gamma must lie in [0, 1).")

    planning = protocol.get("planning", {})
    missing = REQUIRED_PLANNING_KEYS - planning.keys()
    if missing:
        raise ValueError(f"Missing planning keys: {sorted(missing)}")
    if planning["elites"] > planning["candidates"]:
        raise ValueError("CEM elites cannot exceed candidates.")
    if planning["receding_horizon"] > planning["horizon"]:
        raise ValueError("Receding horizon cannot exceed planning horizon.")
    if planning["horizon"] < successor.get("history_size", 1) - 1:
        raise ValueError("The plan is too short to form the successor history.")
    if planning["action_block"] != planning.get("frame_skip"):
        raise ValueError("Planning action blocks must match the training frame skip.")
    if planning["horizon"] * planning["action_block"] > planning["episode_budget"]:
        raise ValueError("The explicit plan exceeds the episode budget.")
    evaluation = protocol.get("evaluation", {})
    if min(evaluation.get("episodes", 0), evaluation.get("goal_offset", 0)) <= 0:
        raise ValueError("Evaluation episodes and goal offset must be positive.")


def _resolve_heads_checkpoint(path: str | Path) -> Path:
    requested = Path(path).expanduser().resolve()
    if requested.is_file():
        return requested
    if requested.is_dir():
        files = sorted(requested.glob("*.pt"))
        if len(files) == 1:
            return files[0]
    raise FileNotFoundError(
        "An LS-LeWM heads checkpoint must be a .pt file or a directory "
        "containing exactly one .pt file."
    )


def _validate_heads_config(config: dict[str, Any], protocol: dict[str, Any]) -> None:
    successor = protocol["successor"]
    expected = {
        "objective_version": successor["objective_version"],
        "embed_dim": protocol["model"]["embed_dim"],
        "history_size": successor["history_size"],
        "hidden_dim": successor["hidden_dim"],
        "feature_basis": successor["feature_basis"],
        "continuation_policy": successor["continuation_policy"],
        "goal_offset_weighting": successor["goal_offset_weighting"],
        "terminal_condition": successor["terminal_condition"],
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Heads checkpoint {key} differs from protocol.")
    if not np.isclose(float(config["gamma"]), float(successor["gamma"])):
        raise ValueError("Heads checkpoint gamma differs from protocol.")


def _validate_checkpoint_pair(
    *,
    base_name: str,
    base_file: Path,
    heads_config: dict[str, Any],
) -> None:
    """Reject independently selected base and successor exports."""

    if heads_config.get("base_export_run_name") != base_name:
        raise ValueError("Heads checkpoint was not exported with this base run name.")
    expected_hash = heads_config.get("base_checkpoint_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("Heads checkpoint is missing its paired base checkpoint hash.")
    if _sha256(base_file) != expected_hash:
        raise ValueError("Heads checkpoint does not match the selected base weights.")


def evaluate_ls_lewm(
    *,
    protocol_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    base_checkpoint_path: str | Path,
    heads_checkpoint_path: str | Path,
    video: bool = False,
    smoke: bool = False,
) -> dict[str, Any]:
    protocol = load_ls_evaluation_protocol(protocol_path)
    if smoke:
        protocol["id"] = f"{protocol['id']}_smoke"
        protocol["evaluation"]["episodes"] = 1
        protocol["planning"].update(
            {"candidates": 8, "iterations": 1, "elites": 2, "episode_budget": 25}
        )
        validate_ls_evaluation_protocol(protocol)

    dataset_path = Path(dataset_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_source = _resolve_dataset_source(dataset_path, protocol["dataset"])
    compatibility = prepare_cloud_runtime()

    import stable_worldmodel as swm
    import torch
    from sklearn.preprocessing import StandardScaler
    from torchvision.transforms import v2 as transforms

    package_version = importlib.metadata.version("stable-worldmodel")
    if package_version != protocol["runtime"]["stable_worldmodel_version"]:
        raise RuntimeError(
            f"Expected stable-worldmodel 0.1.1, found {package_version}."
        )

    base_name, base_file, base_cache = _resolve_local_export_checkpoint(
        base_checkpoint_path
    )
    heads_file = _resolve_heads_checkpoint(heads_checkpoint_path)
    heads, heads_config = load_local_successor_heads(
        heads_file, map_location="cuda"
    )
    _validate_heads_config(heads_config, protocol)
    _validate_checkpoint_pair(
        base_name=base_name,
        base_file=base_file,
        heads_config=heads_config,
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
    expected_action_dim = (
        int(dataset.get_dim("action")) * protocol["planning"]["action_block"]
    )
    if int(heads_config["action_dim"]) != expected_action_dim:
        raise ValueError("Heads checkpoint action block dimension is incompatible.")

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

    action_stats_path = output_dir / "action_normalization.json"
    if action_stats_path.is_file():
        import json

        with action_stats_path.open() as stream:
            action_stats = json.load(stream)
        action_processor = StandardScaler()
        action_processor.mean_ = np.asarray(action_stats["mean"], dtype=np.float64)
        action_processor.scale_ = np.asarray(action_stats["scale"], dtype=np.float64)
        action_processor.var_ = np.asarray(action_stats["variance"], dtype=np.float64)
        action_processor.n_features_in_ = len(action_processor.mean_)
        action_processor.n_samples_seen_ = int(action_stats["samples"])
    else:
        action_processor = StandardScaler().fit(dataset.get_col_data("action"))
        action_stats = {
            "mean": action_processor.mean_,
            "scale": action_processor.scale_,
            "variance": action_processor.var_,
            "samples": int(action_processor.n_samples_seen_),
        }
        _write_json(action_stats_path, action_stats)

    model = swm.wm.load_pretrained(base_name, cache_dir=str(base_cache)).to("cuda").eval()
    model.requires_grad_(False)
    heads = heads.to("cuda").eval()
    heads.requires_grad_(False)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_count += sum(parameter.numel() for parameter in heads.parameters())

    image = protocol["image_preprocessing"]
    image_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=image["mean"], std=image["std"]),
            transforms.Resize(size=protocol["world"]["image_size"]),
        ]
    )
    policy = make_local_successor_policy(
        world_model=model,
        heads=heads,
        planning=planning,
        successor=protocol["successor"],
        process={"action": action_processor},
        transform={"pixels": image_transform, "goal": image_transform},
        device="cuda",
    )

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
            "heads_path": str(heads_file),
            "heads_sha256": _sha256(heads_file),
            "heads_config": heads_config,
        },
        "selection": selection,
        "normalization": {"action": action_stats},
        "runtime": {
            "stable_worldmodel": package_version,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "tdwm_git_revision": _git_revision(),
            "cuda_device": torch.cuda.get_device_name(0),
            "stablewm_home": os.environ.get("STABLEWM_HOME"),
            "compatibility_adapter": compatibility,
        },
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
        "parameter_count": parameter_count,
        "method": "ls_lewm",
        "smoke": smoke,
        "protocol_manifest": str(output_dir / "protocol_manifest.json"),
    }
    _write_json(output_dir / "results.json", result)
    return _jsonable(result)
