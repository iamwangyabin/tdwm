"""Planning adapter and unified checkpoint loader for Joint TD-GT-LeWM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from tdwm.adapters.mc_gt_lewm import MCGoalTailLeWM
from tdwm.methods.goal_tail_value import GoalTailValue


class JointTDGoalTailLeWM(MCGoalTailLeWM):
    """Score CEM candidates with jointly trained dynamics and goal tail."""


def load_joint_td_goal_tail_value(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[GoalTailValue, dict[str, Any], dict[str, Any]]:
    """Load the online value from a unified Joint TD-GT-LeWM checkpoint."""

    payload = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    if payload.get("method") != "joint_td_gt_lewm":
        raise ValueError("The checkpoint is not Joint TD-GT-LeWM.")
    if payload.get("objective_version") != 1:
        raise ValueError("Joint TD-GT-LeWM planning requires objective_version 1.")
    if payload.get("training_state_version") != 1:
        raise ValueError("Joint planning requires a resumable unified checkpoint.")
    if "world_model_state_dict" not in payload:
        raise ValueError("The joint checkpoint does not contain LeWM dynamics.")
    config = dict(payload["value_config"])
    if config.get("objective") != "one_step_td":
        raise ValueError("Joint TD-GT-LeWM planning requires a one-step TD value.")
    if config.get("input_distribution") != "lewm_predicted_terminal_history":
        raise ValueError("The value was not trained on LeWM terminal rollouts.")
    value = GoalTailValue(
        history_dim=int(config["history_dim"]),
        goal_dim=int(config["goal_dim"]),
        hidden_dim=int(config["hidden_dim"]),
    )
    value.load_state_dict(payload["value_state_dict"])
    value.eval()
    value.requires_grad_(False)
    return value, config, payload


def apply_joint_world_model_state(
    world_model: nn.Module,
    payload: dict[str, Any],
) -> None:
    """Replace the LeWM initialization with jointly trained dynamics."""

    world_model.load_state_dict(payload["world_model_state_dict"], strict=True)
    world_model.eval()
    world_model.requires_grad_(False)


def make_joint_td_goal_tail_policy(
    *,
    world_model: nn.Module,
    value: GoalTailValue,
    planning: dict[str, Any],
    history_size: int,
    action_block_dim: int,
    tail_weight: float,
    process: dict[str, Any] | None = None,
    transform: dict[str, Any] | None = None,
    device: str | torch.device = "cpu",
):
    """Build unchanged upstream CEM around the unified joint model."""

    import stable_worldmodel as swm

    wrapped = JointTDGoalTailLeWM(
        world_model,
        value,
        history_size=history_size,
        action_block_dim=action_block_dim,
        tail_weight=tail_weight,
    ).to(device)
    wrapped.eval()
    wrapped.requires_grad_(False)
    solver = swm.solver.CEMSolver(
        model=wrapped,
        batch_size=planning["solver_batch_size"],
        num_samples=planning["candidates"],
        var_scale=planning["initial_variance"],
        n_steps=planning["iterations"],
        topk=planning["elites"],
        device=device,
        seed=planning["planning_seed"],
    )
    config = swm.PlanConfig(
        horizon=planning["horizon"],
        receding_horizon=planning["receding_horizon"],
        history_len=planning["plan_config_history_len"],
        action_block=planning["action_block"],
        warm_start=planning["warm_start"],
    )
    return swm.policy.WorldModelPolicy(
        solver=solver,
        config=config,
        process=process,
        transform=transform,
    )
