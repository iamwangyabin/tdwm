from __future__ import annotations

import math

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


def test_manifold_prefix_head_is_causal_and_action_conditioned():
    torch.manual_seed(41)
    head = _small_head().eval()
    with torch.no_grad():
        torch.nn.init.normal_(head.fusion[-1].weight, std=0.05)
        for block in head.predictor:
            torch.nn.init.normal_(block.modulation[-1].weight, std=0.05)
    history = torch.randn(2, 2, 8)
    actions = torch.randn(2, 5, 3)
    changed = actions.clone()
    changed[:, 2:] = torch.randn_like(changed[:, 2:])

    original = head.predict_latents(history, actions)
    perturbed = head.predict_latents(history, changed)

    assert original.shape == (2, 5, 8)
    assert torch.allclose(original[:, :2], perturbed[:, :2], atol=1e-6)
    assert not torch.allclose(original[:, 2:], perturbed[:, 2:])
    assert not hasattr(head, "policy")


def test_manifold_prefix_successor_is_exactly_derived_from_predicted_latents():
    torch.manual_seed(43)
    head = _small_head().eval()
    history = torch.randn(3, 2, 8)
    actions = torch.randn(3, 5, 3)
    target = torch.randn(3, 5, 8)

    output = manifold_sequence_objective(
        head,
        history,
        actions,
        target,
        gamma=0.9,
    )

    assert torch.allclose(
        output.moments[..., :-2],
        output.predicted_future / math.sqrt(8),
    )
    assert torch.allclose(
        output.moments[..., -2],
        output.predicted_future.square().mean(dim=-1),
    )
    assert torch.equal(
        output.moments[..., -1], torch.ones_like(output.moments[..., -1])
    )
    assert torch.allclose(
        output.recovered_future, output.predicted_future, atol=1e-6
    )
    assert torch.allclose(
        head(history, actions), output.prediction, atol=1e-6
    )


def test_manifold_latent_objective_updates_both_online_encoder_branches():
    torch.manual_seed(47)
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
    )
    output.latent_loss.backward()

    assert history.grad is not None and torch.count_nonzero(history.grad) > 0
    assert future.grad is not None and torch.count_nonzero(future.grad) > 0
    assert any(parameter.grad is not None for parameter in head.parameters())
    assert output.latent_mse_by_horizon.shape == (5,)
    assert output.successor_mse_by_horizon.shape == (5,)


def test_manifold_protocol_keeps_data_budget_and_encoder_fixed():
    e2e = load_rf_successor_training_protocol(
        "configs/experiment/rf_e2e_moment_sequence_wm_cube_train.yaml"
    )
    manifold = load_rf_successor_training_protocol(
        "configs/experiment/rf_manifold_prefix_successor_wm_cube_train.yaml"
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
        assert manifold[key] == e2e[key]
    assert manifold["joint_objective"]["latent_sequence_weight"] == 1.0
    assert manifold["successor"]["objective_version"] == 7
    assert manifold["successor"]["latent_recovery"] == (
        "direct_manifold_latents"
    )
    assert "feature_group_reduction" not in manifold["successor"]
    assert "hidden_dim" not in manifold["successor"]


def test_manifold_evaluation_protocol_uses_only_the_derived_successor():
    protocol = load_rf_successor_evaluation_protocol(
        "configs/experiment/"
        "rf_manifold_prefix_successor_wm_cube_checkpoint_o50.yaml"
    )

    assert protocol["method"] == "rf_manifold_prefix_successor_wm"
    assert protocol["successor"]["objective_version"] == 7
    assert protocol["successor"]["planning_weight"] == 1.0
    assert protocol["successor"]["terminal_weight"] == 0.0
    assert protocol["inference_objective"]["autoregressive_rollout"] == "disabled"


def test_manifold_terminal_protocol_changes_only_the_goal_query():
    standard = load_rf_successor_evaluation_protocol(
        "configs/experiment/"
        "rf_manifold_prefix_successor_wm_cube_checkpoint_o50.yaml"
    )
    terminal = load_rf_successor_evaluation_protocol(
        "configs/experiment/rf_manifold_prefix_successor_wm_cube_terminal_o50.yaml"
    )

    assert terminal["successor"]["planning_query"] == "terminal_moment"
    terminal["successor"].pop("planning_query")
    for key in ("id", "display_name", "inference_objective", "provenance"):
        terminal[key] = standard[key]
    assert terminal == standard


def test_manifold_prefix_checkpoint_round_trip(tmp_path):
    head = _small_head()
    config = {
        "architecture": "causal_transformer_manifold_successor",
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
    checkpoint = tmp_path / "rf_manifold_prefix_successor.pt"
    torch.save(
        {
            "method": "rf_manifold_prefix_successor_wm",
            "objective_version": 7,
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
    assert payload["objective_version"] == 7
