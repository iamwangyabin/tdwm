from __future__ import annotations

import torch

from tdwm.adapters.rf_successor_lewm import load_rf_successor_checkpoint
from tdwm.evaluation.rf_successor_lewm import (
    load_rf_successor_evaluation_protocol,
)
from tdwm.methods.rf_successor_lewm import (
    ActionPrefixMomentHead,
    balanced_successor_mse,
    successor_sequence_objective,
)
from tdwm.training.rf_successor_lewm import (
    load_rf_successor_training_protocol,
)


def test_group_sum_reduction_does_not_divide_vector_error_by_latent_dim():
    prediction = torch.zeros(2, 3, 6)
    target = torch.zeros_like(prediction)
    prediction[..., :-2] = 1.0

    coordinate_mean = balanced_successor_mse(prediction, target)
    group_sum = balanced_successor_mse(
        prediction,
        target,
        vector_reduction="group_sum",
    )

    assert torch.allclose(coordinate_mean, torch.tensor(1.0 / 3.0))
    assert torch.allclose(group_sum, torch.tensor(4.0 / 3.0))


def test_group_balanced_sequence_objective_strengthens_latent_vector_gradient():
    torch.manual_seed(21)
    head = ActionPrefixMomentHead(
        embed_dim=4,
        action_dim=2,
        history_size=2,
        hidden_dim=7,
        gamma=0.9,
    )
    history = torch.randn(2, 2, 4)
    actions = torch.randn(2, 3, 2)
    future = torch.randn(2, 3, 4)

    legacy = successor_sequence_objective(
        head,
        history,
        actions,
        future,
        gamma=0.9,
    )
    balanced = successor_sequence_objective(
        head,
        history,
        actions,
        future,
        gamma=0.9,
        vector_reduction="group_sum",
    )

    assert balanced.successor_loss > legacy.successor_loss
    assert torch.equal(balanced.prediction, legacy.prediction)
    assert torch.equal(balanced.target, legacy.target)


def test_group_balanced_checkpoint_round_trip(tmp_path):
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
    checkpoint = tmp_path / "rf_balanced_successor_sequence.pt"
    torch.save(
        {
            "method": "rf_balanced_successor_sequence_wm",
            "objective_version": 3,
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
    assert payload["objective_version"] == 3


def test_group_balanced_protocols_lock_the_single_changed_factor():
    training = load_rf_successor_training_protocol(
        "configs/experiment/rf_balanced_successor_sequence_wm_cube_train.yaml"
    )
    evaluation = load_rf_successor_evaluation_protocol(
        "configs/experiment/"
        "rf_balanced_successor_sequence_wm_cube_checkpoint_o50.yaml"
    )

    assert training["method"] == "rf_balanced_successor_sequence_wm"
    assert training["successor"]["objective_version"] == 3
    assert training["successor"]["feature_group_reduction"] == "group_sum"
    assert evaluation["successor"]["feature_group_reduction"] == "group_sum"
    assert training["joint_objective"]["successor_sequence_weight"] == 1.0
