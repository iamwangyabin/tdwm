from __future__ import annotations

import torch
from torch import nn

from tdwm.adapters.rf_successor_lewm import (
    RewardFreeSuccessorLeWM,
    load_rf_successor_checkpoint,
)
from tdwm.evaluation.rf_successor_lewm import (
    load_rf_successor_evaluation_protocol,
    validate_rf_successor_evaluation_protocol,
)
from tdwm.methods.rf_successor_lewm import ManifoldTransformerMomentHead
from tdwm.training.rf_successor_lewm import (
    _resolve_local_pretrained_lewm_export,
    load_rf_successor_training_protocol,
)


SOURCE_HASH = "0ce38860a672c4a304d6921c6f07158977bb1d2c8f0eed8a002bb7c89502b579"


class _BlendWorld(nn.Module):
    def __init__(self, baseline_future: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("baseline_future", baseline_future)

    def rollout(self, info, action_sequence, history_size=None):
        del history_size
        batch, samples, horizon = action_sequence.shape[:3]
        history = info["emb"].expand(batch, samples, -1, -1)
        future = self.baseline_future[:, :samples, :horizon]
        return {"predicted_emb": torch.cat((history, future), dim=-2)}


class _DirectLatentHead(nn.Module):
    history_size = 1
    embed_dim = 2
    action_dim = 1
    gamma = 0.95

    def predict_latents(self, history, actions):
        del history
        return torch.cat((actions, torch.zeros_like(actions)), dim=-1)

    def predict_moments(self, history, actions):
        return self.predict_latents(history, actions)


def _small_head() -> ManifoldTransformerMomentHead:
    return ManifoldTransformerMomentHead(
        embed_dim=8,
        action_dim=3,
        history_size=2,
        gamma=0.9,
        prefix_depth=1,
        prefix_heads=2,
        prefix_mlp_dim=24,
        predictor_depth=1,
        predictor_mlp_dim=32,
        fusion_dim=24,
        dropout=0.0,
    )


def test_frozen_protocol_has_one_trainable_loss_and_locked_source():
    protocol = load_rf_successor_training_protocol(
        "configs/experiment/"
        "rf_frozen_manifold_prefix_successor_wm_cube_train.yaml"
    )

    assert protocol["method"] == "rf_frozen_manifold_prefix_successor_wm"
    assert protocol["initialization"] == "frozen_pretrained_lewm"
    assert protocol["pretrained_world_model"]["checkpoint_sha256"] == SOURCE_HASH
    assert protocol["pretrained_world_model"]["frozen"] is True
    assert protocol["joint_objective"]["target_encoder"] == "frozen_pretrained"
    assert protocol["successor"]["objective_version"] == 9
    assert protocol["successor"]["target"] == "frozen_pretrained_latents"
    assert protocol["successor"]["pretrained_world_model_sha256"] == SOURCE_HASH
    assert protocol["joint_objective"]["latent_sequence_weight"] == 1.0
    assert protocol["loss"]["sigreg"]["weight"] == 0.0
    assert protocol["training"]["freeze_world_model_from_start"] is True
    assert protocol["training"]["stop_after_epoch"] == 1


def test_frozen_terminal_protocol_changes_only_the_goal_query():
    standard = load_rf_successor_evaluation_protocol(
        "configs/experiment/"
        "rf_frozen_manifold_prefix_successor_wm_cube_checkpoint_o50.yaml"
    )
    terminal = load_rf_successor_evaluation_protocol(
        "configs/experiment/"
        "rf_frozen_manifold_prefix_successor_wm_cube_terminal_o50.yaml"
    )

    assert standard["method"] == "rf_frozen_manifold_prefix_successor_wm"
    assert standard["successor"]["objective_version"] == 9
    assert standard["successor"]["pretrained_world_model_sha256"] == SOURCE_HASH
    assert terminal["successor"]["planning_query"] == "terminal_moment"
    terminal["successor"].pop("planning_query")
    for key in ("id", "display_name", "inference_objective", "provenance"):
        terminal[key] = standard[key]
    assert terminal == standard


def test_reward_free_validation_coefficients_enable_lewm_residual_query():
    protocol = load_rf_successor_evaluation_protocol(
        "configs/experiment/"
        "rf_frozen_manifold_prefix_successor_wm_cube_terminal_o50.yaml"
    )
    protocol["successor"]["planning_query"] = "lewm_direct_terminal_blend"
    protocol["successor"]["lewm_blend_weights"] = [
        0.04,
        0.05,
        0.06,
        0.07,
        0.09,
    ]

    validate_rf_successor_evaluation_protocol(protocol)


def test_lewm_residual_query_recovers_each_endpoint_at_zero_and_one():
    baseline = torch.tensor(
        [[[[1.0, 0.0], [2.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]]
    )
    actions = torch.tensor([[[[1.0], [4.0]], [[1.0], [2.0]]]])
    info = {
        "pixels": torch.zeros(1, 1, 1),
        "emb": torch.zeros(1, 1, 1, 2),
        "goal_emb": torch.tensor([[2.0, 0.0]]),
    }

    baseline_query = RewardFreeSuccessorLeWM(
        _BlendWorld(baseline),
        _DirectLatentHead(),
        max_horizon=2,
        planning_query="lewm_direct_terminal_blend",
        lewm_blend_weights=[0.0, 0.0],
    )
    direct_query = RewardFreeSuccessorLeWM(
        _BlendWorld(baseline),
        _DirectLatentHead(),
        max_horizon=2,
        planning_query="lewm_direct_terminal_blend",
        lewm_blend_weights=[1.0, 1.0],
    )

    assert torch.allclose(
        baseline_query.get_cost(dict(info), actions), torch.tensor([[0.0, 2.0]])
    )
    assert torch.allclose(
        direct_query.get_cost(dict(info), actions), torch.tensor([[2.0, 0.0]])
    )


def test_frozen_checkpoint_round_trip(tmp_path):
    head = _small_head()
    config = {
        "architecture": "causal_transformer_manifold_successor",
        "embed_dim": 8,
        "action_dim": 3,
        "history_size": 2,
        "prefix_depth": 1,
        "prefix_heads": 2,
        "prefix_mlp_dim": 24,
        "predictor_depth": 1,
        "predictor_mlp_dim": 32,
        "fusion_dim": 24,
        "dropout": 0.0,
        "max_horizon": 5,
        "gamma": 0.9,
        "goal_conditioning": "none",
        "action_conditioning": "causal_prefix",
        "pretrained_world_model_sha256": SOURCE_HASH,
    }
    checkpoint = tmp_path / "rf_frozen_manifold_prefix_successor.pt"
    torch.save(
        {
            "method": "rf_frozen_manifold_prefix_successor_wm",
            "objective_version": 9,
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
    assert payload["objective_version"] == 9


def test_pretrained_export_resolver_requires_public_cache_layout(tmp_path):
    checkpoint_dir = tmp_path / "exports" / "checkpoints" / "epoch_10"
    checkpoint_dir.mkdir(parents=True)
    weights = checkpoint_dir / "weights.pt"
    weights.write_bytes(b"weights")

    name, resolved_weights, cache_dir = _resolve_local_pretrained_lewm_export(
        checkpoint_dir
    )

    assert name == "epoch_10"
    assert resolved_weights == weights
    assert cache_dir == tmp_path / "exports"
