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


def test_direct_moment_objective_stops_future_target_gradient():
    torch.manual_seed(31)
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
    )
    output.moment_loss.backward()

    assert history.grad is not None and torch.count_nonzero(history.grad) > 0
    assert future.grad is None
    assert any(parameter.grad is not None for parameter in head.parameters())
    assert output.moment_mse_by_horizon.shape == (5,)
    assert output.successor_mse_by_horizon.shape == (5,)


def test_direct_moment_protocol_keeps_data_model_and_optimizer_fixed():
    balanced = load_rf_successor_training_protocol(
        "configs/experiment/rf_balanced_successor_sequence_wm_cube_train.yaml"
    )
    direct = load_rf_successor_training_protocol(
        "configs/experiment/rf_direct_moment_sequence_wm_cube_train.yaml"
    )

    for key in ("dataset", "split", "sequence", "model", "loss", "loader"):
        assert direct[key] == balanced[key]
    for key in ("optimizer", "scheduler", "training"):
        assert direct[key] == balanced[key]
    assert direct["joint_objective"]["primitive_prediction"] == (
        "future_moment_sequence"
    )
    assert direct["joint_objective"]["target_encoder"] == "online_stop_gradient"
    assert direct["joint_objective"]["moment_sequence_weight"] == 1.0
    assert direct["successor"]["objective_version"] == 5
    assert direct["successor"]["feature_group_reduction"] == "group_sum"


def test_direct_moment_evaluation_protocol_stays_successor_only():
    protocol = load_rf_successor_evaluation_protocol(
        "configs/experiment/rf_direct_moment_sequence_wm_cube_checkpoint_o50.yaml"
    )

    assert protocol["method"] == "rf_direct_moment_sequence_wm"
    assert protocol["successor"]["objective_version"] == 5
    assert protocol["successor"]["terminal_weight"] == 0.0
    assert protocol["inference_objective"]["autoregressive_rollout"] == "disabled"


def test_direct_moment_checkpoint_round_trip(tmp_path):
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
    checkpoint = tmp_path / "rf_direct_moment_sequence.pt"
    torch.save(
        {
            "method": "rf_direct_moment_sequence_wm",
            "objective_version": 5,
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
    assert payload["objective_version"] == 5
