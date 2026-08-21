"""Controlled TD-GT-LeWM evaluation on the reproduced LeWM Cube protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tdwm.adapters.td_gt_lewm import (
    load_td_goal_tail_value,
    make_td_goal_tail_policy,
)
from tdwm.evaluation.mc_gt_lewm import (
    REQUIRED_PLANNING_KEYS,
    _evaluate_goal_tail_lewm,
)


def load_td_gt_evaluation_protocol(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as stream:
        protocol = yaml.safe_load(stream)
    validate_td_gt_evaluation_protocol(protocol)
    return protocol


def validate_td_gt_evaluation_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema_version") != 1:
        raise ValueError("TD-GT-LeWM evaluation requires schema_version 1.")
    if protocol.get("method") != "td_gt_lewm":
        raise ValueError("This evaluator only accepts TD-GT-LeWM.")
    if protocol.get("environment") != "cube":
        raise ValueError("TD-GT-LeWM V0.2 is locked to OGBench Cube.")
    if protocol.get("stage") != "planner_evaluation":
        raise ValueError("TD-GT-LeWM evaluation requires planner_evaluation stage.")
    if protocol.get("runtime", {}).get("stable_worldmodel_version") != "0.1.1":
        raise ValueError("TD-GT-LeWM is locked to stable-worldmodel 0.1.1.")

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
    if tail.get("objective") != "one_step_td":
        raise ValueError("TD-GT-LeWM evaluation requires a one-step TD value.")
    if tail.get("history_size") != 3 or tail.get("latent_dim") != 192:
        raise ValueError("TD-GT-LeWM requires LeWM history 3 and latent dim 192.")
    if tail.get("action_block_dim") != planning["action_block"] * 5:
        raise ValueError("The value action block differs from the Cube planner.")
    expected_history_dim = (
        tail["history_size"] * tail["latent_dim"]
        + (tail["history_size"] - 1) * tail["action_block_dim"]
    )
    if tail.get("history_dim") != expected_history_dim:
        raise ValueError("The configured value history dimension is inconsistent.")
    if tail.get("weight") != 1.0:
        raise ValueError("The first TD-GT-LeWM planner evaluation locks lambda_V to 1.")
    if tail.get("terminate_bootstrap_at_goal") is not True:
        raise ValueError("TD-GT-LeWM must terminate bootstrap at hindsight goals.")
    if not 0.0 <= tail.get("target_ema_decay", -1.0) < 1.0:
        raise ValueError("The TD target EMA decay is invalid.")
    if tail.get("target_cost_reduction") != "mean":
        raise ValueError("The TD value target must use mean latent cost.")
    if tail.get("upstream_terminal_cost_reduction") != "sum":
        raise ValueError("stable-worldmodel 0.1.1 LeWM uses summed terminal cost.")
    if tail.get("upstream_scale_factor") != tail["latent_dim"]:
        raise ValueError("The tail scale must convert mean cost to upstream sum cost.")


def _validate_td_value_checkpoint(
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
    if not np.isclose(
        float(value_config["target_ema_decay"]),
        float(expected["target_ema_decay"]),
    ):
        raise ValueError("Value checkpoint target EMA differs from protocol.")
    if payload.get("training_state_version") != 1:
        raise ValueError("The TD checkpoint is not the resumable formal export.")
    if payload.get("base_checkpoint_sha256") != base_sha256:
        raise ValueError("The value head was trained with a different LeWM checkpoint.")
    if int(payload.get("epoch", -1)) != int(protocol["value_checkpoint"]["epoch"]):
        raise ValueError("Value checkpoint epoch differs from protocol.")


def evaluate_td_gt_lewm(
    *,
    protocol_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    base_checkpoint_path: str | Path,
    value_checkpoint_path: str | Path,
    video: bool = False,
    smoke: bool = False,
) -> dict[str, Any]:
    """Evaluate TD-GT-LeWM with the baseline's dataset, CEM, and episodes."""

    return _evaluate_goal_tail_lewm(
        protocol_path=protocol_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        base_checkpoint_path=base_checkpoint_path,
        value_checkpoint_path=value_checkpoint_path,
        video=video,
        smoke=smoke,
        protocol_loader=load_td_gt_evaluation_protocol,
        protocol_validator=validate_td_gt_evaluation_protocol,
        value_loader=load_td_goal_tail_value,
        value_validator=_validate_td_value_checkpoint,
        policy_builder=make_td_goal_tail_policy,
        method="td_gt_lewm",
    )
