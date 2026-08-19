"""Goal-conditioned long-horizon value components for GT-LeWM."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


def goal_cost(latent: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """Return mean squared latent distance to a goal."""

    if latent.shape[-1] != goal.shape[-1]:
        raise ValueError("latent and goal must share their final dimension.")
    return (latent - goal).pow(2).mean(dim=-1)


def discounted_goal_tail_target(
    future_latents: torch.Tensor,
    goal: torch.Tensor,
    bootstrap: torch.Tensor,
    *,
    gamma: float,
) -> torch.Tensor:
    """Build the normalized discounted N-step latent goal-cost target."""

    _validate_gamma(gamma)
    if future_latents.ndim < 2:
        raise ValueError("future_latents must have shape (..., horizon, dim).")
    if future_latents.shape[:-2] != bootstrap.shape:
        raise ValueError("bootstrap must match future_latents leading dimensions.")
    if future_latents.shape[:-2] != goal.shape[:-1]:
        raise ValueError("goal must match future_latents leading dimensions.")
    horizon = future_latents.shape[-2]
    if horizon <= 0:
        raise ValueError("future_latents must contain at least one future step.")

    weights = torch.pow(
        future_latents.new_tensor(gamma),
        torch.arange(horizon, device=future_latents.device),
    )
    step_costs = goal_cost(future_latents, goal.unsqueeze(-2))
    return (1.0 - gamma) * (step_costs * weights).sum(dim=-1) + (
        gamma**horizon
    ) * bootstrap


class GoalTailValue(nn.Module):
    """Low-capacity scalar value head ``V(z, g)``."""

    def __init__(self, embed_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        hidden_dim = embed_dim if hidden_dim is None else hidden_dim
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(3 * self.embed_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, latent: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        if latent.shape[-1] != self.embed_dim or goal.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Expected latent and goal final dimension {self.embed_dim}."
            )
        if latent.shape != goal.shape:
            try:
                while goal.ndim < latent.ndim:
                    goal = goal.unsqueeze(-2)
                goal = torch.broadcast_to(goal, latent.shape)
            except RuntimeError as exc:
                raise ValueError("latent and goal shapes are not broadcastable.") from exc
        return self.network(torch.cat((latent, goal, latent - goal), dim=-1))


def goal_tail_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean-squared regression loss for the scalar tail value."""

    return (prediction - target).pow(2).mean()


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, *, tau: float) -> None:
    """Move target parameters toward source parameters with an EMA update."""

    if not 0.0 < tau <= 1.0:
        raise ValueError("tau must lie in (0, 1].")
    target_parameters: Iterable[torch.Tensor] = target.parameters()
    source_parameters: Iterable[torch.Tensor] = source.parameters()
    for target_parameter, source_parameter in zip(
        target_parameters, source_parameters, strict=True
    ):
        target_parameter.mul_(1.0 - tau).add_(source_parameter, alpha=tau)


def _validate_gamma(gamma: float) -> None:
    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must lie in [0, 1).")
