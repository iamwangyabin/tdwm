"""Controlled Cube evaluation for Aligned E2E MC-GT-LeWM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tdwm.adapters.aligned_e2e_mc_gt_lewm import (
    load_aligned_e2e_mc_goal_tail_value,
    make_aligned_e2e_mc_goal_tail_policy,
)
from tdwm.evaluation.mc_gt_lewm import (
    REQUIRED_PLANNING_KEYS,
    _evaluate_goal_tail_lewm,
)


def load_aligned_e2e_mc_gt_evaluation_protocol(
    path: str | Path,
) -> dict[str, Any]:
    with Path(path).open() as stream:
        protocol = yaml.safe_load(stream)
    validate_aligned_e2e_mc_gt_evaluation_protocol(protocol)
    return protocol


def validate_aligned_e2e_mc_gt_evaluation_protocol(
    protocol: dict[str, Any],
) -> None:
    if protocol.get("schema_version") != 1:
        raise ValueError("Aligned E2E MC-GT-LeWM evaluation requires schema 1.")
    if protocol.get("method") != "aligned_e2e_mc_gt_lewm":
        raise ValueError("This evaluator only accepts Aligned E2E MC-GT-LeWM.")
    if protocol.get("environment") != "cube" or protocol.get("stage") != "planner_evaluation":
        raise ValueError("Aligned evaluation is locked to Cube planner evaluation.")
    if protocol.get("runtime", {}).get("stable_worldmodel_version") != "0.1.1":
        raise ValueError("Aligned evaluation requires stable-worldmodel 0.1.1.")

    planning = protocol.get("planning", {})
    missing = REQUIRED_PLANNING_KEYS - planning.keys()
    if missing:
        raise ValueError(f"Missing planning protocol keys: {sorted(missing)}")
    if planning["horizon"] != 5 or planning["action_block"] != planning["frame_skip"]:
        raise ValueError("Evaluation must retain horizon 5 and action block 5.")
    if planning["elites"] > planning["candidates"]:
        raise ValueError("CEM elites cannot exceed candidates.")
    if planning["receding_horizon"] != 1:
        raise ValueError("The aligned o50 evaluation replans after one action block.")
    if planning["executed_environment_steps_before_replanning"] != (
        planning["receding_horizon"] * planning["action_block"]
    ):
        raise ValueError("The documented replanning interval is inconsistent.")

    evaluation = protocol.get("evaluation", {})
    if evaluation.get("episodes", 0) <= 0 or evaluation.get("goal_offset", 0) <= 0:
        raise ValueError("Evaluation episodes and goal offset must be positive.")
    planning_coverage = planning["horizon"] * planning["action_block"]
    if evaluation["goal_offset"] <= planning_coverage:
        raise ValueError("The goal must lie strictly beyond the CEM planning coverage.")
    if evaluation.get("requires_tail_beyond_planning_horizon") is not True:
        raise ValueError("The protocol must explicitly require a beyond-horizon tail.")

    tail = protocol.get("tail_value", {})
    if tail.get("objective_version") != 2 or tail.get("objective") != "supervised_mc":
        raise ValueError("Aligned evaluation requires objective-version-two MC value.")
    if tail.get("architecture") != "squared_shared_potential_anchor":
        raise ValueError("Aligned evaluation requires the anchored value architecture.")
    if tail.get("boundary_condition") != "exact_current_goal_zero":
        raise ValueError("Aligned evaluation requires the exact zero boundary.")
    if tail.get("training_input") != "lewm_predicted_terminal_history":
        raise ValueError("Aligned evaluation requires the model-aware value.")
    if tail.get("history_size") != 3 or tail.get("latent_dim") != 192:
        raise ValueError("Aligned evaluation requires history 3 and latent dim 192.")
    if tail.get("model_rollout_horizon") != planning["horizon"]:
        raise ValueError("Training and evaluation rollout horizons differ.")
    if tail.get("action_block_dim") != planning["action_block"] * 5:
        raise ValueError("The value action block differs from the Cube planner.")
    expected_history_dim = (
        tail["history_size"] * tail["latent_dim"]
        + (tail["history_size"] - 1) * tail["action_block_dim"]
    )
    if tail.get("history_dim") != expected_history_dim:
        raise ValueError("The value history dimension is inconsistent.")
    if tail.get("weight") != 1.0:
        raise ValueError("The first aligned planner evaluation locks tail weight one.")
    if tail.get("upstream_scale_factor") != tail["latent_dim"]:
        raise ValueError("The tail must convert mean MC cost to upstream sum cost.")


def _validate_aligned_checkpoint(
    payload: dict[str, Any],
    value_config: dict[str, Any],
    protocol: dict[str, Any],
    *,
    base_sha256: str,
) -> None:
    del base_sha256
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
            raise ValueError(f"Aligned checkpoint {key} differs from protocol.")
    if not np.isclose(float(value_config["gamma"]), float(expected["gamma"])):
        raise ValueError("Aligned checkpoint gamma differs from protocol.")
    if value_config.get("architecture") != expected["architecture"]:
        raise ValueError("Aligned checkpoint architecture differs from protocol.")
    if value_config.get("boundary_condition") != expected["boundary_condition"]:
        raise ValueError("Aligned checkpoint boundary differs from protocol.")
    if value_config.get("input_distribution") != expected["training_input"]:
        raise ValueError("Aligned checkpoint input distribution differs from protocol.")
    if int(payload.get("epoch", -1)) != int(protocol["value_checkpoint"]["epoch"]):
        raise ValueError("Aligned checkpoint epoch differs from protocol.")


def evaluate_aligned_e2e_mc_gt_lewm(
    *,
    protocol_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    world_model_checkpoint_path: str | Path,
    joint_checkpoint_path: str | Path,
    video: bool = False,
    smoke: bool = False,
) -> dict[str, Any]:
    return _evaluate_goal_tail_lewm(
        protocol_path=protocol_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        base_checkpoint_path=world_model_checkpoint_path,
        value_checkpoint_path=joint_checkpoint_path,
        video=video,
        smoke=smoke,
        protocol_loader=load_aligned_e2e_mc_gt_evaluation_protocol,
        protocol_validator=validate_aligned_e2e_mc_gt_evaluation_protocol,
        value_loader=load_aligned_e2e_mc_goal_tail_value,
        value_validator=_validate_aligned_checkpoint,
        policy_builder=make_aligned_e2e_mc_goal_tail_policy,
        method="aligned_e2e_mc_gt_lewm",
    )
