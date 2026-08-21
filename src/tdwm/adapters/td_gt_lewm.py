"""Stable World Model planning adapter for TD-GT-LeWM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from tdwm.adapters.mc_gt_lewm import MCGoalTailLeWM
from tdwm.methods.goal_tail_value import GoalTailValue


class TDGoalTailLeWM(MCGoalTailLeWM):
    """Apply a TD-trained goal-tail value to LeWM's terminal CEM cost."""


def load_td_goal_tail_value(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[GoalTailValue, dict[str, Any], dict[str, Any]]:
    """Load the online value from a formal TD-GT-LeWM checkpoint."""

    payload = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    if payload.get("method") != "td_gt_lewm":
        raise ValueError("The value checkpoint is not a TD-GT-LeWM checkpoint.")
    if payload.get("objective_version") != 1:
        raise ValueError("TD-GT-LeWM planning requires objective_version 1.")
    config = dict(payload["value_config"])
    if config.get("objective") != "one_step_td":
        raise ValueError("TD-GT-LeWM planning requires a one-step TD value.")
    if config.get("terminate_bootstrap_at_goal") is not True:
        raise ValueError("TD-GT-LeWM must terminate bootstrap at hindsight goals.")
    value = GoalTailValue(
        history_dim=int(config["history_dim"]),
        goal_dim=int(config["goal_dim"]),
        hidden_dim=int(config["hidden_dim"]),
    )
    value.load_state_dict(payload["value_state_dict"])
    value.eval()
    value.requires_grad_(False)
    return value, config, payload


def make_td_goal_tail_policy(
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
    """Build the unchanged upstream CEM policy around TD-GT-LeWM."""

    import stable_worldmodel as swm

    wrapped = TDGoalTailLeWM(
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
