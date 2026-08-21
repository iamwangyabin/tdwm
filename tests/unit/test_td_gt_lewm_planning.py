from __future__ import annotations

from pathlib import Path

import torch
import yaml

from tdwm.adapters.mc_gt_lewm import MCGoalTailLeWM
from tdwm.adapters.td_gt_lewm import TDGoalTailLeWM, load_td_goal_tail_value
from tdwm.evaluation.td_gt_lewm import load_td_gt_evaluation_protocol
from tdwm.methods.goal_tail_value import GoalTailValue


ROOT = Path(__file__).resolve().parents[2]


def test_td_planner_loads_online_value_from_resumable_checkpoint(tmp_path):
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
        "target_ema_decay": 0.995,
        "terminate_bootstrap_at_goal": True,
    }
    torch.save(
        {
            "objective_version": 1,
            "training_state_version": 1,
            "method": "td_gt_lewm",
            "value_state_dict": value.state_dict(),
            "target_value_state_dict": value.state_dict(),
            "value_config": config,
        },
        checkpoint,
    )

    restored, restored_config, payload = load_td_goal_tail_value(checkpoint)

    assert restored_config == config
    assert payload["method"] == "td_gt_lewm"
    assert not any(parameter.requires_grad for parameter in restored.parameters())
    for key, parameter in value.state_dict().items():
        assert torch.equal(parameter, restored.state_dict()[key])


def test_td_planner_uses_the_same_terminal_history_adapter():
    assert issubclass(TDGoalTailLeWM, MCGoalTailLeWM)


def test_td_gt_planner_protocol_changes_only_the_tail_checkpoint():
    protocol = load_td_gt_evaluation_protocol(
        ROOT / "configs/experiment/td_gt_lewm_cube_seed3072_o25.yaml"
    )
    with (
        ROOT / "configs/experiment/mc_gt_lewm_cube_seed3072_o25.yaml"
    ).open() as stream:
        mc_protocol = yaml.safe_load(stream)
    with (ROOT / "configs/experiment/lewm_cube_seed3072_o25.yaml").open() as stream:
        baseline = yaml.safe_load(stream)

    assert protocol["method"] == "td_gt_lewm"
    assert protocol["evaluation"] == baseline["evaluation"]
    assert protocol["world"] == baseline["world"]
    assert protocol["planning"] == mc_protocol["planning"]
    assert protocol["base_checkpoint"] == mc_protocol["base_checkpoint"]
    assert protocol["tail_value"]["objective"] == "one_step_td"
    assert protocol["tail_value"]["weight"] == 1.0
    assert protocol["value_checkpoint"]["epoch"] == 19
