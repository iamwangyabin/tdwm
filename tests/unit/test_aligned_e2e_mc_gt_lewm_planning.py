from __future__ import annotations

import torch
import yaml

from tdwm.adapters.aligned_e2e_mc_gt_lewm import (
    load_aligned_e2e_mc_goal_tail_value,
)
from tdwm.evaluation.aligned_e2e_mc_gt_lewm import (
    validate_aligned_e2e_mc_gt_evaluation_protocol,
)
from tdwm.methods.goal_tail_value import BoundaryAnchoredGoalTailValue


def test_locked_aligned_o50_evaluation_protocol_is_valid():
    with open(
        "configs/experiment/aligned_e2e_mc_gt_lewm_cube_seed3072_o50.yaml"
    ) as stream:
        protocol = yaml.safe_load(stream)

    validate_aligned_e2e_mc_gt_evaluation_protocol(protocol)

    assert protocol["evaluation"] == {
        "episodes": 50,
        "goal_offset": 50,
        "start_goal_source": "same dataset episode",
        "requires_tail_beyond_planning_horizon": True,
    }
    assert protocol["planning"]["iterations"] == 30
    assert protocol["planning"]["candidates"] == 300
    assert protocol["base_checkpoint"]["epoch"] == 10
    assert protocol["value_checkpoint"]["epoch"] == 10


def test_aligned_deployment_checkpoint_restores_exact_boundary(tmp_path):
    value = BoundaryAnchoredGoalTailValue(
        history_dim=11,
        goal_dim=2,
        history_size=3,
        hidden_dim=8,
    )
    checkpoint = tmp_path / "aligned.pt"
    torch.save(
        {
            "method": "aligned_e2e_mc_gt_lewm",
            "objective_version": 2,
            "deployment_checkpoint_version": 1,
            "world_model_state_dict": {"weight": torch.ones(1)},
            "value_state_dict": value.state_dict(),
            "value_config": {
                "objective": "supervised_mc",
                "architecture": "squared_shared_potential_anchor",
                "boundary_condition": "exact_current_goal_zero",
                "input_distribution": "lewm_predicted_terminal_history",
                "history_size": 3,
                "history_dim": 11,
                "goal_dim": 2,
                "hidden_dim": 8,
            },
        },
        checkpoint,
    )

    restored, config, payload = load_aligned_e2e_mc_goal_tail_value(checkpoint)
    history = torch.randn(4, 11)
    current = restored.current_latent(history)

    assert config["objective"] == "supervised_mc"
    assert "world_model_state_dict" in payload
    assert torch.equal(restored(history, current), torch.zeros(4))


def test_aligned_evaluation_requires_goal_beyond_planning_horizon():
    with open("configs/experiment/e2e_joint_td_gt_lewm_cube_seed3072_o25.yaml") as stream:
        protocol = yaml.safe_load(stream)
    protocol["method"] = "aligned_e2e_mc_gt_lewm"
    protocol["evaluation"].update(
        {"goal_offset": 50, "requires_tail_beyond_planning_horizon": True}
    )
    protocol["planning"].update(
        {
            "receding_horizon": 1,
            "executed_environment_steps_before_replanning": 5,
            "episode_budget": 100,
        }
    )
    protocol["tail_value"].update(
        {
            "objective_version": 2,
            "objective": "supervised_mc",
            "architecture": "squared_shared_potential_anchor",
            "boundary_condition": "exact_current_goal_zero",
            "training_input": "lewm_predicted_terminal_history",
        }
    )

    validate_aligned_e2e_mc_gt_evaluation_protocol(protocol)

    protocol["evaluation"]["goal_offset"] = 25
    try:
        validate_aligned_e2e_mc_gt_evaluation_protocol(protocol)
    except ValueError as error:
        assert "strictly beyond" in str(error)
    else:
        raise AssertionError("goal_offset equal to planning coverage must be rejected")


def test_aligned_evaluation_validates_optional_planner_diagnostics():
    with open(
        "configs/experiment/aligned_e2e_mc_gt_lewm_cube_seed3072_o50.yaml"
    ) as stream:
        protocol = yaml.safe_load(stream)
    protocol["planner_diagnostics"] = {
        "enabled": True,
        "record_iterations": [0, 29],
        "epsilon": 1e-8,
    }

    validate_aligned_e2e_mc_gt_evaluation_protocol(protocol)

    protocol["planner_diagnostics"]["record_iterations"] = [30]
    try:
        validate_aligned_e2e_mc_gt_evaluation_protocol(protocol)
    except ValueError as error:
        assert "within the CEM loop" in str(error)
    else:
        raise AssertionError("out-of-range diagnostic iterations must be rejected")
