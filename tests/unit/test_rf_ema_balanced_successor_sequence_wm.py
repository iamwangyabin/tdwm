from __future__ import annotations

import torch

from tdwm.adapters.rf_successor_lewm import load_rf_successor_checkpoint
from tdwm.evaluation.rf_successor_lewm import (
    load_rf_successor_evaluation_protocol,
)
from tdwm.methods.rf_successor_lewm import ActionPrefixMomentHead
from tdwm.training.rf_successor_lewm import (
    load_rf_successor_training_protocol,
)


def test_ema_balanced_protocol_changes_only_target_stabilization():
    balanced = load_rf_successor_training_protocol(
        "configs/experiment/rf_balanced_successor_sequence_wm_cube_train.yaml"
    )
    ema = load_rf_successor_training_protocol(
        "configs/experiment/"
        "rf_ema_balanced_successor_sequence_wm_cube_train.yaml"
    )

    for key in ("dataset", "split", "sequence", "model", "loss", "loader"):
        assert ema[key] == balanced[key]
    for key in ("optimizer", "scheduler", "training"):
        assert ema[key] == balanced[key]
    assert ema["joint_objective"]["target_encoder"] == "ema_stop_gradient"
    assert balanced["joint_objective"]["target_encoder"] == "online_end_to_end"
    assert ema["successor"]["objective_version"] == 4
    assert ema["successor"]["target"] == "ema_direct_monte_carlo"
    assert ema["successor"]["target_world_ema_decay"] == 0.995
    assert ema["successor"]["feature_group_reduction"] == "group_sum"


def test_ema_balanced_evaluation_protocol_is_successor_only():
    protocol = load_rf_successor_evaluation_protocol(
        "configs/experiment/"
        "rf_ema_balanced_successor_sequence_wm_cube_checkpoint_o50.yaml"
    )

    assert protocol["method"] == "rf_ema_balanced_successor_sequence_wm"
    assert protocol["successor"]["objective_version"] == 4
    assert protocol["successor"]["target_world_ema_decay"] == 0.995
    assert protocol["successor"]["terminal_weight"] == 0.0
    assert protocol["inference_objective"]["autoregressive_rollout"] == "disabled"


def test_ema_balanced_checkpoint_round_trip(tmp_path):
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
        "target_world_ema_decay": 0.995,
        "embed_dim": 4,
        "action_dim": 3,
        "history_size": 2,
        "hidden_dim": 6,
        "max_horizon": 5,
        "gamma": 0.95,
        "goal_conditioning": "none",
        "action_conditioning": "causal_prefix",
    }
    checkpoint = tmp_path / "rf_ema_balanced_successor_sequence.pt"
    torch.save(
        {
            "method": "rf_ema_balanced_successor_sequence_wm",
            "objective_version": 4,
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
    assert payload["objective_version"] == 4
