"""Controlled Joint TD-GT-LeWM evaluation on OGBench Cube."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tdwm.adapters.joint_td_gt_lewm import (
    apply_joint_world_model_state,
    load_joint_td_goal_tail_value,
    make_joint_td_goal_tail_policy,
)
from tdwm.evaluation.mc_gt_lewm import (
    REQUIRED_PLANNING_KEYS,
    _evaluate_goal_tail_lewm,
)


def load_joint_td_gt_evaluation_protocol(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as stream:
        protocol = yaml.safe_load(stream)
    validate_joint_td_gt_evaluation_protocol(protocol)
    return protocol


def validate_joint_td_gt_evaluation_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema_version") != 1:
        raise ValueError("Joint TD-GT-LeWM evaluation requires schema_version 1.")
    if protocol.get("method") != "joint_td_gt_lewm":
        raise ValueError("This evaluator only accepts Joint TD-GT-LeWM.")
    if protocol.get("environment") != "cube":
        raise ValueError("Joint TD-GT-LeWM V0 is locked to OGBench Cube.")
    if protocol.get("stage") != "planner_evaluation":
        raise ValueError("Joint evaluation requires planner_evaluation stage.")
    if protocol.get("runtime", {}).get("stable_worldmodel_version") != "0.1.1":
        raise ValueError("Joint TD-GT-LeWM is locked to stable-worldmodel 0.1.1.")

    planning = protocol.get("planning", {})
    missing = REQUIRED_PLANNING_KEYS - planning.keys()
    if missing:
        raise ValueError(f"Missing planning protocol keys: {sorted(missing)}")
    if planning["horizon"] != 5:
        raise ValueError("Joint training and evaluation must share horizon 5.")
    if planning["elites"] > planning["candidates"]:
        raise ValueError("CEM elites cannot exceed candidates.")
    if planning["receding_horizon"] > planning["horizon"]:
        raise ValueError("Receding horizon cannot exceed the CEM horizon.")
    if planning["action_block"] != planning["frame_skip"]:
        raise ValueError("Planning action blocks must match training frame skip.")
    if planning["executed_environment_steps_before_replanning"] != (
        planning["receding_horizon"] * planning["action_block"]
    ):
        raise ValueError("The documented replanning interval differs from PlanConfig.")

    evaluation = protocol.get("evaluation", {})
    if evaluation.get("episodes", 0) <= 0 or evaluation.get("goal_offset", 0) <= 0:
        raise ValueError("Evaluation episodes and goal offset must be positive.")
    tail = protocol.get("tail_value", {})
    if tail.get("objective") != "one_step_td":
        raise ValueError("Joint evaluation requires a one-step TD value.")
    if tail.get("training_input") != "lewm_predicted_terminal_history":
        raise ValueError("Joint evaluation requires a model-aware tail checkpoint.")
    if tail.get("history_size") != 3 or tail.get("latent_dim") != 192:
        raise ValueError("Joint evaluation requires history 3 and latent dim 192.")
    if tail.get("model_rollout_horizon") != planning["horizon"]:
        raise ValueError("The trained rollout horizon differs from CEM.")
    if tail.get("action_block_dim") != planning["action_block"] * 5:
        raise ValueError("The value action block differs from the Cube planner.")
    expected_history_dim = (
        tail["history_size"] * tail["latent_dim"]
        + (tail["history_size"] - 1) * tail["action_block_dim"]
    )
    if tail.get("history_dim") != expected_history_dim:
        raise ValueError("The configured value history dimension is inconsistent.")
    if tail.get("weight") != 1.0:
        raise ValueError("V0 locks lambda_V to one.")
    if tail.get("upstream_scale_factor") != tail["latent_dim"]:
        raise ValueError("The tail scale must convert mean cost to upstream sum cost.")


def _validate_joint_checkpoint(
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
        "model_rollout_horizon": expected["model_rollout_horizon"],
    }
    for key, expected_value in checks.items():
        if int(value_config.get(key, -1)) != int(expected_value):
            raise ValueError(f"Joint checkpoint {key} differs from protocol.")
    if not np.isclose(float(value_config["gamma"]), float(expected["gamma"])):
        raise ValueError("Joint checkpoint gamma differs from protocol.")
    if value_config.get("input_distribution") != expected["training_input"]:
        raise ValueError("Joint checkpoint input distribution differs from protocol.")
    if payload.get("base_checkpoint_sha256") != base_sha256:
        raise ValueError("The joint model used a different LeWM initialization.")
    if int(payload.get("epoch", -1)) != int(protocol["value_checkpoint"]["epoch"]):
        raise ValueError("Joint checkpoint epoch differs from protocol.")


def evaluate_joint_td_gt_lewm(
    *,
    protocol_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    base_checkpoint_path: str | Path,
    joint_checkpoint_path: str | Path,
    video: bool = False,
    smoke: bool = False,
) -> dict[str, Any]:
    """Evaluate jointly trained LeWM dynamics and value through unchanged CEM."""

    return _evaluate_goal_tail_lewm(
        protocol_path=protocol_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        base_checkpoint_path=base_checkpoint_path,
        value_checkpoint_path=joint_checkpoint_path,
        video=video,
        smoke=smoke,
        protocol_loader=load_joint_td_gt_evaluation_protocol,
        protocol_validator=validate_joint_td_gt_evaluation_protocol,
        value_loader=load_joint_td_goal_tail_value,
        value_validator=_validate_joint_checkpoint,
        policy_builder=make_joint_td_goal_tail_policy,
        world_model_updater=apply_joint_world_model_state,
        method="joint_td_gt_lewm",
    )
