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
    load_rf_successor_training_protocol,
)


SOURCE_HASH = "0ce38860a672c4a304d6921c6f07158977bb1d2c8f0eed8a002bb7c89502b579"


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


def test_anchored_e2e_protocol_has_fixed_teacher_and_trainable_student():
    protocol = load_rf_successor_training_protocol(
        "configs/experiment/"
        "rf_anchored_e2e_manifold_prefix_wm_cube_train.yaml"
    )

    assert protocol["method"] == "rf_anchored_e2e_manifold_prefix_wm"
    assert protocol["initialization"] == "anchored_pretrained_lewm"
    assert protocol["seeds"] == [0]
    assert protocol["pretrained_world_model"]["student_frozen"] is False
    assert protocol["pretrained_world_model"]["teacher_frozen"] is True
    assert protocol["joint_objective"]["target_encoder"] == (
        "frozen_teacher_stop_gradient"
    )
    assert protocol["successor"]["objective_version"] == 11
    assert protocol["successor"]["target"] == "frozen_teacher_latents"
    assert protocol["loss"]["geometry_anchor"]["weight"] == 10.0
    assert protocol["optimizer"]["world_model_learning_rate"] == 0.000005
    assert protocol["optimizer"]["successor_learning_rate"] == 0.0001


def test_anchored_e2e_evaluation_uses_the_frozen_teacher_goal_space():
    protocol = load_rf_successor_evaluation_protocol(
        "configs/experiment/"
        "rf_anchored_e2e_manifold_prefix_wm_cube_terminal_o50.yaml"
    )

    assert protocol["successor"]["goal_encoder"] == "frozen_pretrained_teacher"
    assert protocol["successor"]["geometry_anchor_weight"] == 10.0
    assert protocol["successor"]["planning_query"] == "terminal_moment"

    protocol["successor"]["planning_query"] = "lewm_direct_terminal_blend"
    protocol["successor"]["lewm_blend_weights"] = [0.05] * 5
    validate_rf_successor_evaluation_protocol(protocol)


class _ConstantEncoder(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.register_buffer("value", torch.tensor(value))

    def encode(self, info):
        batch = info["pixels"].shape[0]
        return {"emb": self.value.expand(batch, 1, 2)}


class _ZeroHead(nn.Module):
    history_size = 1
    embed_dim = 2
    action_dim = 1
    gamma = 0.95

    def predict_moments(self, history, actions):
        del history
        return torch.zeros(*actions.shape[:-1], 4, device=actions.device)


def test_anchored_planner_encodes_goals_with_teacher_not_student():
    method = RewardFreeSuccessorLeWM(
        _ConstantEncoder(1.0),
        _ZeroHead(),
        max_horizon=1,
        planning_query="terminal_moment",
        goal_world_model=_ConstantEncoder(3.0),
    )
    info = {
        "pixels": torch.zeros(1, 1, 1),
        "goal": torch.zeros(1, 1, 1),
        "emb": torch.ones(1, 1, 1, 2),
    }

    method.get_cost(info, torch.zeros(1, 1, 1, 1))

    assert torch.equal(info["goal_emb"], torch.full((1, 1, 2), 3.0))


def test_anchored_checkpoint_round_trip_requires_teacher_state(tmp_path):
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
        "goal_encoder": "frozen_pretrained_teacher",
        "geometry_anchor_weight": 10.0,
        "pretrained_world_model_sha256": SOURCE_HASH,
    }
    checkpoint = tmp_path / "rf_anchored_e2e_manifold_prefix.pt"
    torch.save(
        {
            "method": "rf_anchored_e2e_manifold_prefix_wm",
            "objective_version": 11,
            "deployment_checkpoint_version": 1,
            "world_model_state_dict": {},
            "target_world_model_state_dict": {},
            "successor_state_dict": head.state_dict(),
            "successor_config": config,
        },
        checkpoint,
    )

    restored, restored_config, payload = load_rf_successor_checkpoint(checkpoint)

    assert isinstance(restored, ManifoldTransformerMomentHead)
    assert restored_config == config
    assert payload["objective_version"] == 11
