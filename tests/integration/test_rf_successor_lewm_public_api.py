from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tdwm.adapters.rf_successor_lewm import RewardFreeSuccessorLeWM
from tdwm.methods.rf_successor_lewm import ActionPrefixSuccessorHead
from tdwm.training.rf_successor_lewm import (
    _build_training_module,
    load_rf_successor_training_protocol,
)

swm = pytest.importorskip("stable_worldmodel")


class TinyEncoder(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(3, embed_dim)

    def forward(self, pixels, interpolate_pos_encoding=True):
        del interpolate_pos_encoding
        pooled = pixels.mean(dim=(-2, -1))
        token = self.projection(pooled).unsqueeze(1)
        return SimpleNamespace(last_hidden_state=token)


class TinyPredictor(nn.Module):
    num_frames = 3

    def forward(self, embeddings, action_embeddings):
        return embeddings + action_embeddings


def test_public_lewm_rollout_drives_action_prefix_successor_cost():
    torch.manual_seed(5)
    embed_dim = 4
    action_dim = 2
    world_model = swm.wm.LeWM(
        encoder=TinyEncoder(embed_dim),
        predictor=TinyPredictor(),
        action_encoder=nn.Linear(action_dim, embed_dim),
    )
    successor = ActionPrefixSuccessorHead(
        embed_dim=embed_dim,
        action_dim=action_dim,
        history_size=3,
        hidden_dim=8,
    )
    method = RewardFreeSuccessorLeWM(
        world_model,
        successor,
        max_horizon=4,
        successor_weight=1.0,
        terminal_weight=0.25,
    )
    info = {
        "pixels": torch.randn(2, 3, 1, 3, 8, 8),
        "goal": torch.randn(2, 3, 1, 3, 8, 8),
    }
    candidates = torch.randn(2, 3, 4, action_dim)

    costs = method.get_cost(info, candidates)

    assert costs.shape == (2, 3)
    assert torch.isfinite(costs).all()
    assert not hasattr(method, "get_action")


def test_joint_training_loss_backpropagates_through_public_lewm():
    torch.manual_seed(6)
    world_model = swm.wm.LeWM(
        encoder=TinyEncoder(embed_dim=4),
        predictor=TinyPredictor(),
        action_encoder=nn.Linear(2, 4),
    )
    world_model.predictor.num_frames = 2
    protocol = load_rf_successor_training_protocol(
        "configs/experiment/rf_successor_lewm_cube_train.yaml"
    )
    protocol["sequence"].update(
        history_frames=2,
        rollout_horizon=2,
        num_steps=5,
    )
    protocol["model"]["embed_dim"] = 4
    protocol["successor"]["hidden_dim"] = 8
    protocol["loss"]["sigreg"].update(knots=3, num_projections=4)
    module = _build_training_module(
        world_model,
        protocol,
        total_steps=2,
        action_block_dim=2,
        device_image_preprocessing=False,
    )
    module.log_dict = lambda *args, **kwargs: None
    batch = {
        "pixels": torch.randn(2, 5, 3, 8, 8),
        "action": torch.randn(2, 5, 2),
    }

    loss = module._forward_loss(batch, "train")
    loss.backward()

    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in module.model.parameters())
    assert any(
        parameter.grad is not None for parameter in module.successor.parameters()
    )
    assert all(
        parameter.grad is None for parameter in module.target_model.parameters()
    )


def test_s_only_training_uses_public_encoder_without_lewm_dynamics_loss():
    torch.manual_seed(7)
    world_model = swm.wm.LeWM(
        encoder=TinyEncoder(embed_dim=4),
        predictor=TinyPredictor(),
        action_encoder=nn.Linear(2, 4),
    )
    protocol = load_rf_successor_training_protocol(
        "configs/experiment/rf_successor_sequence_wm_cube_train.yaml"
    )
    protocol["sequence"].update(
        history_frames=2,
        rollout_horizon=2,
        num_steps=5,
    )
    protocol["model"]["embed_dim"] = 4
    protocol["successor"]["hidden_dim"] = 8
    protocol["loss"]["sigreg"].update(knots=3, num_projections=4)
    module = _build_training_module(
        world_model,
        protocol,
        total_steps=2,
        action_block_dim=2,
        device_image_preprocessing=False,
    )
    module.log_dict = lambda *args, **kwargs: None
    batch = {
        "pixels": torch.randn(2, 5, 3, 8, 8),
        "action": torch.randn(2, 5, 2),
    }

    loss = module._forward_loss(batch, "train")
    loss.backward()

    assert torch.isfinite(loss)
    assert not hasattr(module, "target_model")
    assert module.model.encoder.projection.weight.grad is not None
    assert module.model.action_encoder.weight.grad is None
    assert module.model.action_encoder.weight.requires_grad is False
    assert any(
        parameter.grad is not None for parameter in module.successor.parameters()
    )


def test_ema_manifold_training_keeps_target_encoder_frozen():
    torch.manual_seed(11)
    world_model = swm.wm.LeWM(
        encoder=TinyEncoder(embed_dim=4),
        predictor=TinyPredictor(),
        action_encoder=nn.Linear(2, 4),
    )
    protocol = load_rf_successor_training_protocol(
        "configs/experiment/"
        "rf_ema_manifold_prefix_successor_wm_cube_train.yaml"
    )
    protocol["sequence"].update(
        history_frames=2,
        rollout_horizon=2,
        num_steps=5,
    )
    protocol["model"]["embed_dim"] = 4
    protocol["successor"].update(
        prefix_depth=1,
        prefix_heads=2,
        prefix_mlp_dim=12,
        predictor_depth=1,
        predictor_mlp_dim=16,
        fusion_dim=12,
        dropout=0.0,
        max_horizon=2,
    )
    protocol["loss"]["sigreg"].update(knots=3, num_projections=4)
    module = _build_training_module(
        world_model,
        protocol,
        total_steps=2,
        action_block_dim=2,
        device_image_preprocessing=False,
    )
    module.log_dict = lambda *args, **kwargs: None
    batch = {
        "pixels": torch.randn(2, 5, 3, 8, 8),
        "action": torch.randn(2, 5, 2),
    }

    loss = module._forward_loss(batch, "train")
    loss.backward()

    assert torch.isfinite(loss)
    assert module.use_ema_target is True
    assert module.model.encoder.projection.weight.grad is not None
    assert all(
        parameter.grad is None for parameter in module.target_model.parameters()
    )
    assert any(
        parameter.grad is not None for parameter in module.successor.parameters()
    )
