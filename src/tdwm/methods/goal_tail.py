"""Goal-conditioned long-horizon value components for GT-LeWM."""

from __future__ import annotations

from dataclasses import dataclass
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
                raise ValueError(
                    "latent and goal shapes are not broadcastable."
                ) from exc
        return self.network(torch.cat((latent, goal, latent - goal), dim=-1))


def goal_tail_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean-squared regression loss for the scalar tail value."""

    return (prediction - target).pow(2).mean()


@dataclass(frozen=True)
class GoalTailTDOutput:
    """Losses and diagnostics for future-goal Bellman supervision."""

    td_loss: torch.Tensor
    boundary_loss: torch.Tensor
    prediction_mean: torch.Tensor
    target_mean: torch.Tensor
    pair_count: int


def future_goal_td_objective(
    value: GoalTailValue,
    target_value: GoalTailValue,
    latents: torch.Tensor,
    *,
    first_current_index: int,
    max_goal_offset: int,
    td_horizon: int,
    gamma: float,
) -> GoalTailTDOutput:
    """Train ``V(z, g)`` on every valid future goal in a latent clip.

    Goals at or before the N-step bootstrap state terminate the return, while
    farther goals bootstrap from the slowly updated target value. Offsets are
    averaged uniformly so nearby goals do not dominate merely because a clip
    contains more of them.
    """

    _validate_gamma(gamma)
    if latents.ndim != 3:
        raise ValueError("latents must have shape (batch, time, dim).")
    if not 0 <= first_current_index < latents.shape[1] - 1:
        raise ValueError("first_current_index must leave at least one future state.")
    if max_goal_offset <= 0:
        raise ValueError("max_goal_offset must be positive.")
    if td_horizon <= 0:
        raise ValueError("td_horizon must be positive.")

    available = latents.shape[1] - first_current_index - 1
    if max_goal_offset > available:
        raise ValueError("max_goal_offset exceeds the available latent future.")

    losses: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    pair_count = 0
    for offset in range(1, max_goal_offset + 1):
        pair_end = latents.shape[1] - offset
        current = latents[:, first_current_index:pair_end]
        goal = latents[:, first_current_index + offset :]
        prediction = value(current, goal).squeeze(-1)

        steps = min(td_horizon, offset)
        future = torch.stack(
            [
                latents[
                    :,
                    first_current_index + step : pair_end + step,
                ]
                for step in range(1, steps + 1)
            ],
            dim=-2,
        )
        with torch.no_grad():
            if steps < offset:
                bootstrap_state = latents[
                    :,
                    first_current_index + steps : pair_end + steps,
                ]
                bootstrap = target_value(
                    bootstrap_state.detach(), goal.detach()
                ).squeeze(-1)
            else:
                bootstrap = prediction.new_zeros(prediction.shape)
            target = discounted_goal_tail_target(
                future.detach(), goal.detach(), bootstrap, gamma=gamma
            )

        losses.append(goal_tail_loss(prediction, target))
        predictions.append(prediction.detach().reshape(-1))
        targets.append(target.detach().reshape(-1))
        pair_count += prediction.numel()

    boundary_states = latents[:, first_current_index:]
    boundary_prediction = value(boundary_states, boundary_states).squeeze(-1)
    return GoalTailTDOutput(
        td_loss=torch.stack(losses).mean(),
        boundary_loss=boundary_prediction.pow(2).mean(),
        prediction_mean=torch.cat(predictions).mean(),
        target_mean=torch.cat(targets).mean(),
        pair_count=pair_count,
    )


@torch.no_grad()
def ema_update(target: nn.Module, source: nn.Module, *, decay: float) -> None:
    """Update ``target`` with EMA decay applied to its previous parameters."""

    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must lie in [0, 1).")
    target_parameters: Iterable[torch.Tensor] = target.parameters()
    source_parameters: Iterable[torch.Tensor] = source.parameters()
    for target_parameter, source_parameter in zip(
        target_parameters, source_parameters, strict=True
    ):
        target_parameter.mul_(decay).add_(source_parameter, alpha=1.0 - decay)


def _validate_gamma(gamma: float) -> None:
    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must lie in [0, 1).")
