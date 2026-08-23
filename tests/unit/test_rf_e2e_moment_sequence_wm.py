from __future__ import annotations

import torch

from tdwm.adapters.rf_successor_lewm import load_rf_successor_checkpoint
from tdwm.evaluation.rf_successor_lewm import (
    load_rf_successor_evaluation_protocol,
)
from tdwm.methods.rf_successor_lewm import (
    ActionPrefixMomentHead,
    moment_sequence_objective,
)
from tdwm.training.rf_successor_lewm import (
    load_rf_successor_training_protocol,
)


def test_e2e_moment_objective_updates_both_online_encoder_branches():
    torch.manual_seed(37)
    head = ActionPrefixMomentHead(
        embed_dim=4,
        action_dim=2,
        history_size=2,
        hidden_dim=7,
        gamma=0.9,
    )
    history = torch.randn(2, 2, 4, requires_grad=True)
    actions = torch.randn(2, 5, 2)
    future = torch.randn(2, 5, 4, requires_grad=True)

    output = moment_sequence_objective(
        head,
        history,
        actions,
        future,
        gamma=0.9,
        detach_target=False,
    )
    output.moment_loss.backward()

    assert history.grad is not None and torch.count_nonzero(history.grad) > 0
    assert future.grad is not None and torch.count_nonzero(future.grad) > 0
    assert any(parameter.grad is not None for parameter in head.parameters())


def test_e2e_moment_protocol_changes_only_target_gradient_routing():
    detached = load_rf_successor_training_protocol(
        "configs/experiment/rf_direct_moment_sequence_wm_cube_train.yaml"
    )
    e2e = load_rf_successor_training_protocol(
        "configs/experiment/rf_e2e_moment_sequence_wm_cube_train.yaml"
    )

    for key in (
        "dataset",
        "split",
        "sequence",
        "model",
        "loss",
        "loader",
        "optimizer",
        "scheduler",
        "training",
    ):
        assert e2e[key] == detached[key]
    assert e2e["joint_objective"]["target_encoder"] == "online_end_to_end"
    assert e2e["successor"]["objective_version"] == 6
    assert e2e["successor"]["target"] == "online_end_to_end_direct_moments"


def test_e2e_moment_evaluation_protocol_stays_successor_only():
    protocol = load_rf_successor_evaluation_protocol(
        "configs/experiment/rf_e2e_moment_sequence_wm_cube_checkpoint_o50.yaml"
    )

    assert protocol["method"] == "rf_e2e_moment_sequence_wm"
    assert protocol["successor"]["objective_version"] == 6
    assert protocol["successor"]["terminal_weight"] == 0.0
    assert protocol["inference_objective"]["autoregressive_rollout"] == "disabled"


def test_e2e_terminal_query_protocol_changes_only_the_goal_query():
    standard = load_rf_successor_evaluation_protocol(
        "configs/experiment/rf_e2e_moment_sequence_wm_cube_checkpoint_o50.yaml"
    )
    terminal = load_rf_successor_evaluation_protocol(
        "configs/experiment/rf_e2e_moment_sequence_wm_cube_terminal_o50.yaml"
    )

    assert terminal["successor"]["planning_query"] == "terminal_moment"
    terminal["successor"].pop("planning_query")
    for key in ("id", "display_name", "inference_objective", "provenance"):
        terminal[key] = standard[key]
    assert terminal == standard


def test_e2e_moment_checkpoint_round_trip(tmp_path):
    head = ActionPrefixMomentHead(
        embed_dim=4,
        action_dim=3,
        history_size=2,
        hidden_dim=6,
        gamma=0.95,
    )
    config = {
        "architecture": "causal_gru_successor_increments",
        "feature_group_reduction": "group_sum",
        "embed_dim": 4,
        "action_dim": 3,
        "history_size": 2,
        "hidden_dim": 6,
        "max_horizon": 5,
        "gamma": 0.95,
        "goal_conditioning": "none",
        "action_conditioning": "causal_prefix",
    }
    checkpoint = tmp_path / "rf_e2e_moment_sequence.pt"
    torch.save(
        {
            "method": "rf_e2e_moment_sequence_wm",
            "objective_version": 6,
            "deployment_checkpoint_version": 1,
            "world_model_state_dict": {},
            "successor_state_dict": head.state_dict(),
            "successor_config": config,
        },
        checkpoint,
    )

    restored, restored_config, payload = load_rf_successor_checkpoint(checkpoint)

    assert isinstance(restored, ActionPrefixMomentHead)
    assert restored_config == config
    assert payload["objective_version"] == 6
