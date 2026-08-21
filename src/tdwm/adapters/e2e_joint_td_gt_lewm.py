"""Planning adapter for end-to-end Joint TD-GT-LeWM checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from tdwm.adapters.joint_td_gt_lewm import make_joint_td_goal_tail_policy
from tdwm.methods.goal_tail_value import GoalTailValue


def load_e2e_joint_td_goal_tail_value(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[GoalTailValue, dict[str, Any], dict[str, Any]]:
    payload = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    if payload.get("method") != "e2e_joint_td_gt_lewm":
        raise ValueError("The checkpoint is not E2E Joint TD-GT-LeWM.")
    if payload.get("objective_version") != 1:
        raise ValueError("E2E Joint TD-GT-LeWM requires objective version 1.")
    if payload.get("deployment_checkpoint_version") != 1:
        raise ValueError("The E2E deployment checkpoint version is unsupported.")
    if "world_model_state_dict" not in payload:
        raise ValueError("The E2E checkpoint must contain the jointly trained LeWM.")
    config = dict(payload["value_config"])
    if config.get("objective") != "one_step_td":
        raise ValueError("E2E planning requires the one-step TD value.")
    if config.get("input_distribution") != "lewm_predicted_terminal_history":
        raise ValueError("The E2E value was not trained on imagined terminal histories.")
    value = GoalTailValue(
        history_dim=int(config["history_dim"]),
        goal_dim=int(config["goal_dim"]),
        hidden_dim=int(config["hidden_dim"]),
    )
    value.load_state_dict(payload["value_state_dict"])
    value.eval()
    value.requires_grad_(False)
    return value, config, payload


make_e2e_joint_td_goal_tail_policy = make_joint_td_goal_tail_policy


__all__ = [
    "load_e2e_joint_td_goal_tail_value",
    "make_e2e_joint_td_goal_tail_policy",
]
