from __future__ import annotations

import torch

from tdwm.adapters.e2e_joint_td_gt_lewm import (
    load_e2e_joint_td_goal_tail_value,
)
from tdwm.methods.goal_tail_value import GoalTailValue


def test_e2e_deployment_checkpoint_restores_value_and_requires_world_model(tmp_path):
    value = GoalTailValue(history_dim=11, goal_dim=2, hidden_dim=8)
    checkpoint = tmp_path / "e2e.pt"
    torch.save(
        {
            "method": "e2e_joint_td_gt_lewm",
            "objective_version": 1,
            "deployment_checkpoint_version": 1,
            "world_model_state_dict": {"weight": torch.ones(1)},
            "value_state_dict": value.state_dict(),
            "value_config": {
                "objective": "one_step_td",
                "input_distribution": "lewm_predicted_terminal_history",
                "history_dim": 11,
                "goal_dim": 2,
                "hidden_dim": 8,
            },
        },
        checkpoint,
    )

    restored, config, payload = load_e2e_joint_td_goal_tail_value(checkpoint)

    assert config["objective"] == "one_step_td"
    assert "world_model_state_dict" in payload
    for name, parameter in value.state_dict().items():
        assert torch.equal(parameter, restored.state_dict()[name])
