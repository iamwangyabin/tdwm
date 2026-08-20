from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tdwm.adapters.local_successor import LocalSuccessorLeWM
from tdwm.methods.local_successor import LocalSuccessorHeads

swm = pytest.importorskip("stable_worldmodel")


class TinyEncoder(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(3, embed_dim)

    def forward(self, pixels, interpolate_pos_encoding=True):
        pooled = pixels.mean(dim=(-2, -1))
        token = self.projection(pooled).unsqueeze(1)
        return SimpleNamespace(last_hidden_state=token)


class TinyPredictor(nn.Module):
    num_frames = 3

    def forward(self, embeddings, action_embeddings):
        return embeddings + action_embeddings


def test_public_lewm_rollout_drives_successor_cost_and_actor():
    torch.manual_seed(11)
    embed_dim = 4
    action_dim = 2
    world_model = swm.wm.LeWM(
        encoder=TinyEncoder(embed_dim),
        predictor=TinyPredictor(),
        action_encoder=nn.Linear(action_dim, embed_dim),
    )
    heads = LocalSuccessorHeads(
        embed_dim=embed_dim,
        action_dim=action_dim,
        history_size=3,
        hidden_dim=8,
    )
    method = LocalSuccessorLeWM(world_model, heads, gamma=0.9)

    expanded_info = {
        "pixels": torch.randn(2, 3, 1, 3, 8, 8),
        "goal": torch.randn(2, 3, 1, 3, 8, 8),
    }
    candidates = torch.randn(2, 3, 4, action_dim)
    costs = method.get_cost(expanded_info, candidates)

    actor_info = {
        "pixels": torch.randn(2, 1, 3, 8, 8),
        "goal": torch.randn(2, 1, 3, 8, 8),
    }
    actions = method.get_action(actor_info, horizon=4)

    assert costs.shape == (2, 3)
    assert torch.isfinite(costs).all()
    assert actions.shape == (2, 4, action_dim)
    assert torch.isfinite(actions).all()
