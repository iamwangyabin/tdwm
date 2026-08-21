from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn
import yaml

from tdwm.adapters.joint_td_gt_lewm import (
    JointTDGoalTailLeWM,
    apply_joint_world_model_state,
    load_joint_td_goal_tail_value,
)
from tdwm.adapters.mc_gt_lewm import MCGoalTailLeWM
from tdwm.evaluation.joint_td_gt_lewm import load_joint_td_gt_evaluation_protocol
from tdwm.methods.goal_tail_value import GoalTailValue


ROOT = Path(__file__).resolve().parents[2]


def test_joint_checkpoint_loads_value_and_replaces_lewm_state(tmp_path):
    world_model = nn.Linear(2, 2)
    joint_world_model = copy.deepcopy(world_model)
    with torch.no_grad():
        joint_world_model.weight.fill_(3.0)
        joint_world_model.bias.fill_(4.0)
    value = GoalTailValue(history_dim=10, goal_dim=2, hidden_dim=7)
    checkpoint = tmp_path / "best.pt"
    config = {
        "history_dim": 10,
        "goal_dim": 2,
        "hidden_dim": 7,
        "history_size": 3,
        "action_block_dim": 2,
        "max_goal_offset": 16,
        "gamma": 0.95,
        "objective": "one_step_td",
        "input_distribution": "lewm_predicted_terminal_history",
        "model_rollout_horizon": 5,
    }
    torch.save(
        {
            "objective_version": 1,
            "training_state_version": 1,
            "method": "joint_td_gt_lewm",
            "world_model_state_dict": joint_world_model.state_dict(),
            "value_state_dict": value.state_dict(),
            "value_config": config,
        },
        checkpoint,
    )

    restored, restored_config, payload = load_joint_td_goal_tail_value(checkpoint)
    apply_joint_world_model_state(world_model, payload)

    assert restored_config == config
    assert payload["method"] == "joint_td_gt_lewm"
    assert not any(parameter.requires_grad for parameter in restored.parameters())
    assert not any(parameter.requires_grad for parameter in world_model.parameters())
    for key, parameter in joint_world_model.state_dict().items():
        assert torch.equal(parameter, world_model.state_dict()[key])


def test_joint_planner_uses_the_same_terminal_cost_adapter():
    assert issubclass(JointTDGoalTailLeWM, MCGoalTailLeWM)


def test_joint_planner_protocol_keeps_the_baseline_cem_and_episode_sample():
    protocol = load_joint_td_gt_evaluation_protocol(
        ROOT / "configs/experiment/joint_td_gt_lewm_cube_seed3072_o25.yaml"
    )
    with (ROOT / "configs/experiment/lewm_cube_seed3072_o25.yaml").open() as stream:
        baseline = yaml.safe_load(stream)
    with (
        ROOT / "configs/experiment/td_gt_lewm_cube_seed3072_o25.yaml"
    ).open() as stream:
        diagnostic = yaml.safe_load(stream)

    assert protocol["evaluation"] == baseline["evaluation"]
    assert protocol["world"] == baseline["world"]
    assert protocol["planning"] == diagnostic["planning"]
    assert protocol["base_checkpoint"] == diagnostic["base_checkpoint"]
    assert protocol["value_checkpoint"]["epoch"] == 1
    assert protocol["value_checkpoint"]["includes_world_model_state"] is True
    assert protocol["tail_value"]["model_rollout_horizon"] == 5
