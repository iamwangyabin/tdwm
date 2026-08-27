"""Directed, reward-free successor geometry for LeWM latents."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class DirectedSuccessorGeometry(nn.Module):
    """Embed query states and goal states in two distinct directed spaces."""

    def __init__(
        self,
        *,
        embed_dim: int,
        projection_dim: int,
        hidden_dim: int,
        temperature: float,
    ) -> None:
        super().__init__()
        if min(embed_dim, projection_dim, hidden_dim) <= 0:
            raise ValueError("All successor-geometry dimensions must be positive.")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        self.embed_dim = int(embed_dim)
        self.projection_dim = int(projection_dim)
        self.temperature = float(temperature)
        self.query_projector = _projection_mlp(embed_dim, hidden_dim, projection_dim)
        self.goal_projector = _projection_mlp(embed_dim, hidden_dim, projection_dim)

    def encode_query(self, latent: torch.Tensor) -> torch.Tensor:
        self._validate_latent(latent)
        return F.normalize(self.query_projector(latent), dim=-1)

    def encode_goal(self, latent: torch.Tensor) -> torch.Tensor:
        self._validate_latent(latent)
        return F.normalize(self.goal_projector(latent), dim=-1)

    def score(self, query: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Return directed cosine compatibility with broadcastable leading axes."""

        return (self.encode_query(query) * self.encode_goal(goal)).sum(dim=-1)

    def logits(self, query: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Return pairwise density-ratio logits for two latent batches."""

        if query.ndim != 2 or goal.ndim != 2:
            raise ValueError("Pairwise logits require rank-two latent batches.")
        return self.encode_query(query) @ self.encode_goal(goal).transpose(0, 1) / (
            self.temperature
        )

    def _validate_latent(self, latent: torch.Tensor) -> None:
        if latent.ndim < 1 or latent.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Expected a latent final dimension of {self.embed_dim}, "
                f"found {tuple(latent.shape)}."
            )


def _projection_mlp(
    embed_dim: int, hidden_dim: int, projection_dim: int
) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(embed_dim),
        nn.Linear(embed_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, projection_dim),
    )


def discounted_horizon_weights(
    max_offset: int,
    *,
    gamma: float,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return normalized finite-horizon geometric occupancy weights."""

    if max_offset <= 0:
        raise ValueError("max_offset must be positive.")
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must lie in (0, 1].")
    offsets = torch.arange(max_offset, device=device, dtype=dtype or torch.float32)
    weights = gamma**offsets
    return weights / weights.sum()


@dataclass(frozen=True)
class SuccessorGeometryOutput:
    loss: torch.Tensor
    raw_loss: torch.Tensor
    real_query_loss: torch.Tensor
    predicted_query_loss: torch.Tensor
    top1: torch.Tensor
    positive_margin: torch.Tensor
    loss_by_offset: torch.Tensor
    top1_by_offset: torch.Tensor


def successor_geometry_objective(
    geometry: DirectedSuccessorGeometry,
    real_queries: torch.Tensor,
    predicted_queries: torch.Tensor,
    future_goals: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    gamma: float,
) -> SuccessorGeometryOutput:
    """Learn discounted directed reachability from real and rollout query states.

    Other windows from the same source episode are masked rather than treated as
    negatives. This avoids teaching the model that another reachable state from
    the same trajectory is unreachable.
    """

    if real_queries.ndim != 2 or predicted_queries.shape != real_queries.shape:
        raise ValueError("real and predicted queries must share shape (pairs, dim).")
    if future_goals.ndim != 3 or future_goals.shape[0] != real_queries.shape[0]:
        raise ValueError("future_goals must have shape (pairs, offsets, dim).")
    if future_goals.shape[-1] != real_queries.shape[-1]:
        raise ValueError("queries and goals must share their latent dimension.")
    if group_ids.ndim != 1 or group_ids.shape[0] != real_queries.shape[0]:
        raise ValueError("group_ids must contain one identifier per query pair.")

    pair_count, max_offset = future_goals.shape[:2]
    if pair_count <= 1:
        raise ValueError("Successor contrastive learning requires multiple pairs.")
    weights = discounted_horizon_weights(
        max_offset,
        gamma=gamma,
        device=real_queries.device,
        dtype=real_queries.dtype,
    )
    same_group = group_ids[:, None].eq(group_ids[None, :])
    diagonal = torch.eye(pair_count, dtype=torch.bool, device=real_queries.device)
    invalid_negative = same_group & ~diagonal
    eligible = (~invalid_negative).sum(dim=-1)
    if torch.any(eligible < 2):
        raise ValueError("Each query needs at least one cross-episode negative.")

    real_features = geometry.encode_query(real_queries)
    predicted_features = geometry.encode_query(predicted_queries)
    labels = torch.arange(pair_count, device=real_queries.device)
    real_losses = []
    predicted_losses = []
    raw_losses = []
    accuracies = []
    margins = []

    for offset in range(max_offset):
        goal_features = geometry.encode_goal(future_goals[:, offset])
        real_logits = real_features @ goal_features.transpose(0, 1)
        predicted_logits = predicted_features @ goal_features.transpose(0, 1)
        real_logits = real_logits / geometry.temperature
        predicted_logits = predicted_logits / geometry.temperature
        real_logits = real_logits.masked_fill(invalid_negative, -torch.inf)
        predicted_logits = predicted_logits.masked_fill(invalid_negative, -torch.inf)

        real_raw = F.cross_entropy(real_logits, labels, reduction="none")
        predicted_raw = F.cross_entropy(predicted_logits, labels, reduction="none")
        normalizer = eligible.to(real_raw.dtype).log()
        real_losses.append((real_raw / normalizer).mean())
        predicted_losses.append((predicted_raw / normalizer).mean())
        raw_losses.append(0.5 * (real_raw.mean() + predicted_raw.mean()))

        combined_logits = 0.5 * (real_logits + predicted_logits)
        accuracies.append(combined_logits.argmax(dim=-1).eq(labels).float().mean())
        positive = combined_logits.diagonal()
        negatives = combined_logits.masked_fill(diagonal, -torch.inf).amax(dim=-1)
        finite_margin = positive - negatives
        finite_margin = finite_margin[torch.isfinite(finite_margin)]
        margins.append(
            finite_margin.mean()
            if finite_margin.numel()
            else positive.new_zeros(())
        )

    real_by_offset = torch.stack(real_losses)
    predicted_by_offset = torch.stack(predicted_losses)
    raw_by_offset = torch.stack(raw_losses)
    top1_by_offset = torch.stack(accuracies)
    margin_by_offset = torch.stack(margins)
    loss_by_offset = 0.5 * (real_by_offset + predicted_by_offset)
    return SuccessorGeometryOutput(
        loss=(weights * loss_by_offset).sum(),
        raw_loss=(weights * raw_by_offset).sum(),
        real_query_loss=(weights * real_by_offset).sum(),
        predicted_query_loss=(weights * predicted_by_offset).sum(),
        top1=(weights * top1_by_offset).sum(),
        positive_margin=(weights * margin_by_offset).sum(),
        loss_by_offset=loss_by_offset,
        top1_by_offset=top1_by_offset,
    )


def successor_geometry_cost(
    geometry: DirectedSuccessorGeometry,
    query: torch.Tensor,
    goal: torch.Tensor,
) -> torch.Tensor:
    """Convert directed compatibility to the bounded cost minimized by CEM."""

    return 1.0 - geometry.score(query, goal)


__all__ = [
    "DirectedSuccessorGeometry",
    "SuccessorGeometryOutput",
    "discounted_horizon_weights",
    "successor_geometry_cost",
    "successor_geometry_objective",
]
