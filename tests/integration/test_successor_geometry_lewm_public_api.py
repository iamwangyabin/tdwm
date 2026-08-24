from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tdwm.adapters.successor_geometry_lewm import SuccessorGeometryLeWM
from tdwm.methods.successor_geometry_lewm import DirectedSuccessorGeometry
from tdwm.methods.residual_policy_lewm import ResidualLeWM
from tdwm.training.successor_geometry_lewm import (
    _build_training_module,
    load_successor_geometry_training_protocol,
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
    num_frames = 2

    def forward(self, embeddings, action_embeddings):
        return embeddings + action_embeddings


def make_tiny_world_model(embed_dim: int = 4):
    return swm.wm.LeWM(
        encoder=TinyEncoder(embed_dim),
        predictor=TinyPredictor(),
        action_encoder=nn.Linear(2, embed_dim),
    )


def make_tiny_residual_world_model(embed_dim: int = 4):
    return ResidualLeWM(
        encoder=TinyEncoder(embed_dim),
        predictor=TinyPredictor(),
        action_encoder=nn.Linear(2, embed_dim),
        pred_proj=nn.Sequential(nn.Linear(embed_dim, embed_dim)),
    )


def test_public_lewm_rollout_drives_successor_geometry_cost():
    torch.manual_seed(13)
    world_model = make_tiny_world_model()
    geometry = DirectedSuccessorGeometry(
        embed_dim=4,
        projection_dim=3,
        hidden_dim=8,
        temperature=0.1,
    )
    method = SuccessorGeometryLeWM(
        world_model,
        geometry,
        history_size=2,
        max_horizon=4,
    )
    info = {
        "pixels": torch.randn(2, 3, 1, 3, 8, 8),
        "goal": torch.randn(2, 3, 1, 3, 8, 8),
    }
    candidates = torch.randn(2, 3, 4, 2)

    costs = method.get_cost(info, candidates)

    assert costs.shape == (2, 3)
    assert torch.isfinite(costs).all()
    assert torch.all((0.0 <= costs) & (costs <= 2.0))
    assert not hasattr(method, "get_action")


def test_joint_training_backpropagates_geometry_through_lewm_rollout():
    torch.manual_seed(21)
    world_model = make_tiny_world_model()
    protocol = load_successor_geometry_training_protocol(
        "configs/experiment/successor_geometry_lewm_cube_train.yaml"
    )
    protocol["sequence"].update(
        history_frames=2,
        rollout_horizon=2,
        max_future_offset=2,
        num_steps=7,
    )
    protocol["model"]["embed_dim"] = 4
    protocol["geometry"].update(
        projection_dim=3,
        hidden_dim=8,
        rollout_horizon=2,
        max_future_offset=2,
    )
    protocol["loss"]["sigreg"].update(knots=3, num_projections=4)
    module = _build_training_module(
        world_model,
        protocol,
        total_steps=2,
        device_image_preprocessing=False,
    )
    module.log_dict = lambda *args, **kwargs: None
    batch = {
        "pixels": torch.randn(2, 7, 3, 8, 8),
        "action": torch.randn(2, 7, 2),
        "_tdwm_episode_id": torch.tensor([4, 9]),
    }

    loss = module._forward_loss(batch, "train")
    loss.backward()

    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in module.model.parameters())
    assert all(parameter.grad is not None for parameter in module.geometry.parameters())


def test_residual_policy_auxiliary_trains_with_the_shared_public_lewm_rollout():
    torch.manual_seed(22)
    world_model = make_tiny_residual_world_model()
    protocol = load_successor_geometry_training_protocol(
        "configs/experiment/residual_policy_successor_geometry_lewm_cube_train.yaml"
    )
    protocol["sequence"].update(
        history_frames=2,
        rollout_horizon=2,
        max_future_offset=2,
        num_steps=7,
    )
    protocol["model"]["embed_dim"] = 4
    protocol["geometry"].update(
        projection_dim=3,
        hidden_dim=8,
        rollout_horizon=2,
        max_future_offset=2,
    )
    protocol["loss"]["sigreg"].update(knots=3, num_projections=4)
    module = _build_training_module(
        world_model,
        protocol,
        total_steps=2,
        device_image_preprocessing=False,
        action_block_dim=2,
    )
    module.log_dict = lambda *args, **kwargs: None
    batch = {
        "pixels": torch.randn(2, 7, 3, 8, 8),
        "action": torch.randn(2, 7, 2),
        "_tdwm_episode_id": torch.tensor([5, 10]),
    }

    loss = module._forward_loss(batch, "train")
    loss.backward()

    assert torch.isfinite(loss)
    assert module.use_policy_auxiliary is True
    assert module.policy_head.weight.grad is not None
    assert torch.count_nonzero(module.policy_head.weight.grad) > 0
    assert any(parameter.grad is not None for parameter in module.model.parameters())
    assert all(parameter.grad is not None for parameter in module.geometry.parameters())
