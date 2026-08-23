"""Goal-independent latent geometry shared by successor methods."""

from __future__ import annotations

import math

import torch


def successor_feature_basis(latent: torch.Tensor) -> torch.Tensor:
    """Lift a latent so squared goal distance is a linear functional."""

    if latent.ndim < 1 or latent.shape[-1] <= 0:
        raise ValueError("latent must have a non-empty final dimension.")
    dimension = latent.shape[-1]
    scaled = latent / math.sqrt(dimension)
    squared_norm = latent.square().mean(dim=-1, keepdim=True)
    constant = torch.ones_like(squared_norm)
    return torch.cat((scaled, squared_norm, constant), dim=-1)


def goal_cost_weights(goal: torch.Tensor) -> torch.Tensor:
    """Return weights mapping lifted features to mean squared goal distance."""

    if goal.ndim < 1 or goal.shape[-1] <= 0:
        raise ValueError("goal must have a non-empty final dimension.")
    dimension = goal.shape[-1]
    linear = -2.0 * goal / math.sqrt(dimension)
    squared_norm = goal.square().mean(dim=-1, keepdim=True)
    coefficient = torch.ones_like(squared_norm)
    return torch.cat((linear, coefficient, squared_norm), dim=-1)


def latent_goal_cost(latent: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """Compute mean squared latent distance with broadcastable leading axes."""

    if latent.shape[-1] != goal.shape[-1]:
        raise ValueError("latent and goal must share their final dimension.")
    return (latent - goal).square().mean(dim=-1)


def successor_goal_cost(successor: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """Project a successor feature vector onto a latent goal cost."""

    expected = goal.shape[-1] + 2
    if successor.shape[-1] != expected:
        raise ValueError(f"Expected successor final dimension {expected}.")
    return (successor * goal_cost_weights(goal)).sum(dim=-1)


__all__ = [
    "goal_cost_weights",
    "latent_goal_cost",
    "successor_feature_basis",
    "successor_goal_cost",
]
