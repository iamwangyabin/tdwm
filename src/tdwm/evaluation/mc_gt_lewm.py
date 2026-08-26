"""Controlled MC-GT-LeWM evaluation on the reproduced LeWM Cube protocol."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from tdwm.adapters import prepare_cloud_runtime
from tdwm.adapters.mc_gt_lewm import (
    load_mc_goal_tail_value,
    make_mc_goal_tail_policy,
)
from tdwm.evaluation.lewm_checkpoint import (
    _git_revision,
    _jsonable,
    _resolve_dataset_source,
    _resolve_local_export_checkpoint,
    _sha256,
    _write_json,
    sample_start_goal_pairs,
)


REQUIRED_PLANNING_KEYS = {
    "horizon",
    "candidates",
    "iterations",
    "elites",
    "initial_variance",
    "action_block",
    "frame_skip",
    "receding_horizon",
    "executed_environment_steps_before_replanning",
    "episode_budget",
    "planning_seed",
    "solver_batch_size",
    "warm_start",
    "plan_config_history_len",
}


def load_mc_gt_evaluation_protocol(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as stream:
        protocol = yaml.safe_load(stream)
    validate_mc_gt_evaluation_protocol(protocol)
    return protocol


def validate_mc_gt_evaluation_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema_version") != 1:
        raise ValueError("MC-GT-LeWM evaluation requires schema_version 1.")
    if protocol.get("method") != "mc_gt_lewm":
        raise ValueError("This evaluator only accepts MC-GT-LeWM.")
    if protocol.get("environment") != "cube":
        raise ValueError("MC-GT-LeWM V0.2 is locked to OGBench Cube.")
    if protocol.get("stage") != "planner_evaluation":
        raise ValueError("MC-GT-LeWM evaluation requires planner_evaluation stage.")
    if protocol.get("runtime", {}).get("stable_worldmodel_version") != "0.1.1":
        raise ValueError("MC-GT-LeWM is locked to stable-worldmodel 0.1.1.")

    planning = protocol.get("planning", {})
    missing = REQUIRED_PLANNING_KEYS - planning.keys()
    if missing:
        raise ValueError(f"Missing planning protocol keys: {sorted(missing)}")
    if planning["elites"] > planning["candidates"]:
        raise ValueError("CEM elites cannot exceed candidates.")
    if planning["receding_horizon"] > planning["horizon"]:
        raise ValueError("Receding horizon cannot exceed the CEM horizon.")
    if planning["action_block"] != planning["frame_skip"]:
        raise ValueError("Planning action blocks must match training frame skip.")
    if planning["horizon"] * planning["action_block"] > planning["episode_budget"]:
        raise ValueError("The planned action sequence exceeds the episode budget.")
    if planning["executed_environment_steps_before_replanning"] != (
        planning["receding_horizon"] * planning["action_block"]
    ):
        raise ValueError("The documented replanning interval differs from PlanConfig.")

    evaluation = protocol.get("evaluation", {})
    if evaluation.get("episodes", 0) <= 0:
        raise ValueError("Evaluation episodes must be positive.")
    if evaluation.get("goal_offset", 0) <= 0:
        raise ValueError("Goal offset must be positive.")

    tail = protocol.get("tail_value", {})
    if tail.get("objective") != "supervised_mc":
        raise ValueError("MC-GT-LeWM evaluation requires supervised MC value targets.")
    if tail.get("history_size") != 3 or tail.get("latent_dim") != 192:
        raise ValueError("MC-GT-LeWM requires LeWM history 3 and latent dim 192.")
    expected_action_block_dim = planning["action_block"] * 5
    if tail.get("action_block_dim") != expected_action_block_dim:
        raise ValueError("The value action block differs from the Cube planner.")
    expected_history_dim = (
        tail["history_size"] * tail["latent_dim"]
        + (tail["history_size"] - 1) * tail["action_block_dim"]
    )
    if tail.get("history_dim") != expected_history_dim:
        raise ValueError("The configured value history dimension is inconsistent.")
    if tail.get("weight") != 1.0:
        raise ValueError("The first MC-GT-LeWM planner evaluation locks lambda_V to 1.")
    if tail.get("target_cost_reduction") != "mean":
        raise ValueError("The MC target must use mean latent cost.")
    if tail.get("upstream_terminal_cost_reduction") != "sum":
        raise ValueError("stable-worldmodel 0.1.1 LeWM uses summed terminal cost.")
    if tail.get("upstream_scale_factor") != tail["latent_dim"]:
        raise ValueError("The tail scale must convert mean cost to upstream sum cost.")


def _resolve_value_checkpoint(path: str | Path) -> Path:
    requested = Path(path).expanduser().resolve()
    if requested.is_file() and requested.suffix == ".pt":
        return requested
    raise FileNotFoundError("Pass the exact goal-tail .pt checkpoint file.")


def _validate_value_checkpoint(
    payload: dict[str, Any],
    value_config: dict[str, Any],
    protocol: dict[str, Any],
    *,
    base_sha256: str,
) -> None:
    expected = protocol["tail_value"]
    checks = {
        "history_size": expected["history_size"],
        "history_dim": expected["history_dim"],
        "goal_dim": expected["latent_dim"],
        "action_block_dim": expected["action_block_dim"],
        "hidden_dim": expected["hidden_dim"],
        "max_goal_offset": expected["max_goal_offset"],
    }
    for key, expected_value in checks.items():
        if int(value_config.get(key, -1)) != int(expected_value):
            raise ValueError(f"Value checkpoint {key} differs from protocol.")
    if not np.isclose(float(value_config["gamma"]), float(expected["gamma"])):
        raise ValueError("Value checkpoint gamma differs from protocol.")
    if payload.get("base_checkpoint_sha256") != base_sha256:
        raise ValueError("The value head was trained with a different LeWM checkpoint.")
    if int(payload.get("epoch", -1)) != int(protocol["value_checkpoint"]["epoch"]):
        raise ValueError("Value checkpoint epoch differs from protocol.")


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


def _evaluate_goal_tail_lewm(
    *,
    protocol_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    base_checkpoint_path: str | Path,
    value_checkpoint_path: str | Path,
    protocol_loader: Callable[[str | Path], dict[str, Any]],
    protocol_validator: Callable[[dict[str, Any]], None],
    value_loader: Callable[..., tuple[Any, dict[str, Any], dict[str, Any]]],
    value_validator: Callable[..., None],
    policy_builder: Callable[..., Any],
    method: str,
    world_model_updater: Callable[[Any, dict[str, Any]], None] | None = None,
    video: bool = False,
    smoke: bool = False,
) -> dict[str, Any]:
    """Evaluate one scalar goal-tail method under the locked LeWM protocol."""

    protocol = protocol_loader(protocol_path)
    if smoke:
        protocol["id"] = f"{protocol['id']}_smoke"
        protocol["evaluation"]["episodes"] = 1
        protocol["planning"].update(
            {"candidates": 8, "iterations": 1, "elites": 2, "episode_budget": 25}
        )
        protocol["smoke"] = True
        protocol_validator(protocol)

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

    base_name, base_file, base_cache = _resolve_local_export_checkpoint(
        base_checkpoint_path
    )
    value_file = _resolve_value_checkpoint(value_checkpoint_path)
    base_hash = _sha256(base_file)
    value_hash = _sha256(value_file)
    if base_hash != protocol["base_checkpoint"]["sha256"]:
        raise ValueError("Base LeWM checkpoint SHA-256 differs from protocol.")
    if value_hash != protocol["value_checkpoint"]["sha256"]:
        raise ValueError(f"{method} checkpoint SHA-256 differs from protocol.")

    value, value_config, value_payload = value_loader(
        value_file, map_location="cuda"
    )
    value_validator(
        value_payload,
        value_config,
        protocol,
        base_sha256=base_hash,
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
    model = swm.wm.load_pretrained(base_name, cache_dir=str(base_cache)).to("cuda")
    if world_model_updater is not None:
        world_model_updater(model, value_payload)
    model.eval()
    model.requires_grad_(False)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != protocol["base_checkpoint"]["parameters"]:
        raise ValueError("The LeWM parameter count differs from protocol.")
    value_parameter_count = sum(parameter.numel() for parameter in value.parameters())
    if value_parameter_count != protocol["value_checkpoint"]["parameters"]:
        raise ValueError("The goal-tail parameter count differs from protocol.")

    image_stats = protocol["image_preprocessing"]
    image_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=image_stats["mean"], std=image_stats["std"]),
            transforms.Resize(size=protocol["world"]["image_size"]),
        ]
    )
    tail_cfg = protocol["tail_value"]
    policy_kwargs = {
        "world_model": model,
        "value": value,
        "planning": planning_cfg,
        "history_size": tail_cfg["history_size"],
        "action_block_dim": tail_cfg["action_block_dim"],
        "tail_weight": tail_cfg["weight"],
        "process": {"action": action_processor},
        "transform": {"pixels": image_transform, "goal": image_transform},
        "device": "cuda",
    }
    planner_diagnostics = protocol.get("planner_diagnostics")
    if planner_diagnostics and planner_diagnostics.get("enabled") is True:
        policy_kwargs["planner_diagnostics"] = planner_diagnostics
    policy = policy_builder(
        **policy_kwargs,
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
            "stablewm_home": os.environ.get("STABLEWM_HOME"),
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
                video=output_dir / "videos" if video else None,
            )
    finally:
        world.close()

    diagnostics_result = None
    diagnostics_recorder = getattr(policy, "tdwm_planner_diagnostics", None)
    if diagnostics_recorder is not None:
        diagnostics_payload = diagnostics_recorder.export()
        diagnostics_path = output_dir / "planner_diagnostics.json"
        _write_json(diagnostics_path, diagnostics_payload)
        diagnostics_result = {
            "path": str(diagnostics_path),
            "record_count": diagnostics_payload["record_count"],
            "solve_count": diagnostics_payload["solve_count"],
            "aggregates": diagnostics_payload["aggregates"],
        }

    result = {
        "metrics": metrics,
        "elapsed_seconds": time.time() - started,
        "base_parameter_count": parameter_count,
        "value_parameter_count": value_parameter_count,
        "method": method,
        "smoke": smoke,
        "protocol_manifest": str(output_dir / "protocol_manifest.json"),
    }
    if diagnostics_result is not None:
        result["planner_diagnostics"] = diagnostics_result
    _write_json(output_dir / "results.json", result)
    return _jsonable(result)


def evaluate_mc_gt_lewm(
    *,
    protocol_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    base_checkpoint_path: str | Path,
    value_checkpoint_path: str | Path,
    video: bool = False,
    smoke: bool = False,
) -> dict[str, Any]:
    """Evaluate MC-GT-LeWM with the baseline's dataset, CEM, and episodes."""

    return _evaluate_goal_tail_lewm(
        protocol_path=protocol_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        base_checkpoint_path=base_checkpoint_path,
        value_checkpoint_path=value_checkpoint_path,
        video=video,
        smoke=smoke,
        protocol_loader=load_mc_gt_evaluation_protocol,
        protocol_validator=validate_mc_gt_evaluation_protocol,
        value_loader=load_mc_goal_tail_value,
        value_validator=_validate_value_checkpoint,
        policy_builder=make_mc_goal_tail_policy,
        method="mc_gt_lewm",
    )
