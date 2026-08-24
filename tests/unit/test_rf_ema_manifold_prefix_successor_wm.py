from __future__ import annotations

import torch

from tdwm.adapters.rf_successor_lewm import load_rf_successor_checkpoint
from tdwm.evaluation.rf_successor_lewm import (
    load_rf_successor_evaluation_protocol,
)
from tdwm.methods.rf_successor_lewm import (
    ManifoldTransformerMomentHead,
    manifold_sequence_objective,
)
from tdwm.training.rf_successor_lewm import (
    load_rf_successor_training_protocol,
)


def _small_head() -> ManifoldTransformerMomentHead:
    return ManifoldTransformerMomentHead(
        embed_dim=8,
        action_dim=3,
        history_size=2,
        gamma=0.9,
        prefix_depth=2,
        prefix_heads=2,
        prefix_mlp_dim=24,
        predictor_depth=2,
        predictor_mlp_dim=32,
        fusion_dim=24,
        dropout=0.0,
    )


def test_ema_manifold_protocol_changes_only_target_stabilization():
    online = load_rf_successor_training_protocol(
        "configs/experiment/rf_manifold_prefix_successor_wm_cube_train.yaml"
    )
    ema = load_rf_successor_training_protocol(
        "configs/experiment/"
        "rf_ema_manifold_prefix_successor_wm_cube_train.yaml"
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
        assert ema[key] == online[key]
    assert online["joint_objective"]["target_encoder"] == "online_end_to_end"
    assert ema["joint_objective"]["target_encoder"] == "ema_stop_gradient"
    assert ema["successor"]["objective_version"] == 8
    assert ema["successor"]["target"] == "ema_stop_gradient_latents"
    assert ema["successor"]["target_world_ema_decay"] == 0.995


def test_ema_manifold_objective_stops_future_target_gradient():
    torch.manual_seed(59)
    head = _small_head()
    history = torch.randn(2, 2, 8, requires_grad=True)
    actions = torch.randn(2, 5, 3)
    future = torch.randn(2, 5, 8, requires_grad=True)

    output = manifold_sequence_objective(
        head,
        history,
        actions,
        future,
        gamma=0.9,
        detach_target=True,
    )
    output.latent_loss.backward()

    assert future.grad is None
    assert history.grad is not None and torch.count_nonzero(history.grad) > 0
    assert any(parameter.grad is not None for parameter in head.parameters())


def test_ema_manifold_evaluation_protocols_are_valid():
    standard = load_rf_successor_evaluation_protocol(
        "configs/experiment/"
        "rf_ema_manifold_prefix_successor_wm_cube_checkpoint_o50.yaml"
    )
    terminal = load_rf_successor_evaluation_protocol(
        "configs/experiment/"
        "rf_ema_manifold_prefix_successor_wm_cube_terminal_o50.yaml"
    )

    assert standard["method"] == "rf_ema_manifold_prefix_successor_wm"
    assert standard["successor"]["objective_version"] == 8
    assert standard["successor"]["target_world_ema_decay"] == 0.995
    assert terminal["successor"]["planning_query"] == "terminal_moment"


def test_ema_manifold_checkpoint_round_trip(tmp_path):
    head = _small_head()
    config = {
        "architecture": "causal_transformer_manifold_successor",
        "target_world_ema_decay": 0.995,
        "embed_dim": 8,
        "action_dim": 3,
        "history_size": 2,
        "prefix_depth": 2,
        "prefix_heads": 2,
        "prefix_mlp_dim": 24,
        "predictor_depth": 2,
        "predictor_mlp_dim": 32,
        "fusion_dim": 24,
        "dropout": 0.0,
        "max_horizon": 5,
        "gamma": 0.9,
        "goal_conditioning": "none",
        "action_conditioning": "causal_prefix",
    }
    checkpoint = tmp_path / "rf_ema_manifold_prefix_successor.pt"
    torch.save(
        {
            "method": "rf_ema_manifold_prefix_successor_wm",
            "objective_version": 8,
            "deployment_checkpoint_version": 1,
            "world_model_state_dict": {},
            "successor_state_dict": head.state_dict(),
            "successor_config": config,
        },
        checkpoint,
    )

    restored, restored_config, payload = load_rf_successor_checkpoint(checkpoint)

    assert isinstance(restored, ManifoldTransformerMomentHead)
    assert restored_config == config
    assert payload["objective_version"] == 8
