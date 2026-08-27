"""Planning adapter for aligned end-to-end MC GoalTail checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from tdwm.adapters.mc_gt_lewm import make_mc_goal_tail_policy
from tdwm.methods.goal_tail_value import BoundaryAnchoredGoalTailValue


def load_aligned_e2e_mc_goal_tail_value(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[BoundaryAnchoredGoalTailValue, dict[str, Any], dict[str, Any]]:
    payload = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    if payload.get("method") != "aligned_e2e_mc_gt_lewm":
        raise ValueError("The checkpoint is not Aligned E2E MC-GT-LeWM.")
    if payload.get("objective_version") != 2:
        raise ValueError("Aligned E2E MC-GT-LeWM requires objective version 2.")
    if payload.get("deployment_checkpoint_version") != 1:
        raise ValueError("The aligned deployment checkpoint version is unsupported.")
    if "world_model_state_dict" not in payload:
        raise ValueError("The aligned checkpoint must contain the jointly trained LeWM.")
    config = dict(payload["value_config"])
    if config.get("objective") != "supervised_mc":
        raise ValueError("Aligned planning requires supervised MC targets.")
    if config.get("architecture") != "squared_shared_potential_anchor":
        raise ValueError("Aligned planning requires the boundary-anchored value.")
    if config.get("boundary_condition") != "exact_current_goal_zero":
        raise ValueError("The aligned checkpoint does not enforce the zero boundary.")
    if config.get("input_distribution") != "lewm_predicted_terminal_history":
        raise ValueError("The aligned value was not trained on terminal rollouts.")
    value = BoundaryAnchoredGoalTailValue(
        history_dim=int(config["history_dim"]),
        goal_dim=int(config["goal_dim"]),
        history_size=int(config["history_size"]),
        hidden_dim=int(config["hidden_dim"]),
    )
    value.load_state_dict(payload["value_state_dict"])
    value.eval()
    value.requires_grad_(False)
    return value, config, payload


make_aligned_e2e_mc_goal_tail_policy = make_mc_goal_tail_policy


__all__ = [
    "load_aligned_e2e_mc_goal_tail_value",
    "make_aligned_e2e_mc_goal_tail_policy",
]
