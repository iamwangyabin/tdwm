"""Reward-free, action-prefix successor supervision for LeWM."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

from tdwm.methods.successor_geometry import successor_feature_basis


def discounted_prefix_mass(
    horizon: int,
    *,
    gamma: float,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-step discount powers and their inclusive prefix sums."""

    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0, 1].")
    powers = torch.pow(
        reference.new_tensor(gamma),
        torch.arange(horizon, device=reference.device),
    )
    return powers, powers.cumsum(dim=0)


def finite_horizon_successor_targets(
    future_latents: torch.Tensor,
    *,
    gamma: float,
) -> torch.Tensor:
    """Build normalized direct-MC successor targets for every prefix horizon.

    ``future_latents[..., h, :]`` is the latent reached after applying the
    first ``h + 1`` candidate actions. No goal, reward, policy, or bootstrap is
    involved in this target.
    """

    if future_latents.ndim < 2 or future_latents.shape[-2] <= 0:
        raise ValueError("future_latents must contain a non-empty time axis.")
    return finite_horizon_successor_from_moments(
        successor_feature_basis(future_latents), gamma=gamma
    )


def finite_horizon_successor_from_moments(
    moments: torch.Tensor,
    *,
    gamma: float,
) -> torch.Tensor:
    """Convert per-horizon future moments into normalized prefix successors."""

    if moments.ndim < 2 or moments.shape[-2] <= 0 or moments.shape[-1] < 3:
        raise ValueError("moments must contain non-empty time and feature axes.")
    powers, mass = discounted_prefix_mass(
        moments.shape[-2], gamma=gamma, reference=moments
    )
    view_shape = (1,) * (moments.ndim - 2) + (-1, 1)
    weighted = moments * powers.view(view_shape)
    return weighted.cumsum(dim=-2) / mass.view(view_shape)


def successor_moments_from_sequence(
    successor: torch.Tensor,
    *,
    gamma: float,
) -> torch.Tensor:
    """Invert an all-horizon successor sequence into future moments."""

    if successor.ndim < 2 or successor.shape[-2] <= 0 or successor.shape[-1] < 3:
        raise ValueError("successor must contain non-empty time and feature axes.")
    powers, mass = discounted_prefix_mass(
        successor.shape[-2], gamma=gamma, reference=successor
    )
    if torch.any(powers == 0):
        raise ValueError("Recovering future moments requires gamma > 0.")
    view_shape = (1,) * (successor.ndim - 2) + (-1, 1)
    weighted = successor * mass.view(view_shape)
    previous = torch.cat(
        (torch.zeros_like(weighted[..., :1, :]), weighted[..., :-1, :]),
        dim=-2,
    )
    return (weighted - previous) / powers.view(view_shape)


def latent_sequence_from_successor(
    successor: torch.Tensor,
    *,
    gamma: float,
) -> torch.Tensor:
    """Recover the latent coordinates represented by successor increments."""

    moments = successor_moments_from_sequence(successor, gamma=gamma)
    latent_dim = moments.shape[-1] - 2
    return moments[..., :latent_dim] * math.sqrt(latent_dim)


def successor_recurrence_residual(
    successor: torch.Tensor,
    predicted_future: torch.Tensor,
    *,
    gamma: float,
) -> torch.Tensor:
    """Measure whether successor increments equal the matching latent feature."""

    if successor.ndim < 2 or predicted_future.ndim != successor.ndim:
        raise ValueError("successor and predicted_future must have matching ranks.")
    if successor.shape[:-1] != predicted_future.shape[:-1]:
        raise ValueError("successor and predicted_future leading shapes must match.")
    if successor.shape[-1] != predicted_future.shape[-1] + 2:
        raise ValueError("successor must use the lifted latent feature dimension.")

    horizon = predicted_future.shape[-2]
    powers, mass = discounted_prefix_mass(
        horizon, gamma=gamma, reference=predicted_future
    )
    view_shape = (1,) * (predicted_future.ndim - 2) + (-1, 1)
    weighted_successor = successor * mass.view(view_shape)
    previous = torch.cat(
        (
            torch.zeros_like(weighted_successor[..., :1, :]),
            weighted_successor[..., :-1, :],
        ),
        dim=-2,
    )
    expected_increment = (
        successor_feature_basis(predicted_future) * powers.view(view_shape)
    )
    return weighted_successor - previous - expected_increment


def balanced_successor_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    vector_reduction: str = "coordinate_mean",
) -> torch.Tensor:
    """Combine vector, squared-norm, and constant feature errors.

    ``coordinate_mean`` preserves the objective used by the first two method
    versions. ``group_sum`` treats the complete scaled latent vector as one
    feature group. Because the vector is scaled by ``sqrt(d)``, summing that
    group's coordinate errors recovers latent MSE instead of dividing it by
    ``d`` a second time.
    """

    if prediction.shape != target.shape or prediction.shape[-1] < 3:
        raise ValueError("prediction and target must be matching lifted features.")
    vector_error = (prediction[..., :-2] - target[..., :-2]).square()
    if vector_reduction == "coordinate_mean":
        vector = vector_error.mean()
    elif vector_reduction == "group_sum":
        vector = vector_error.sum(dim=-1).mean()
    else:
        raise ValueError(
            "vector_reduction must be 'coordinate_mean' or 'group_sum'."
        )
    squared_norm = (prediction[..., -2] - target[..., -2]).square().mean()
    constant = (prediction[..., -1] - target[..., -1]).square().mean()
    return (vector + squared_norm + constant) / 3.0


class ActionPrefixSuccessorHead(nn.Module):
    """Causally summarize each supplied action prefix without seeing a goal."""

    def __init__(
        self,
        *,
        embed_dim: int,
        action_dim: int,
        history_size: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        if min(embed_dim, action_dim, history_size, hidden_dim) <= 0:
            raise ValueError("Successor-head dimensions must be positive.")
        self.embed_dim = int(embed_dim)
        self.action_dim = int(action_dim)
        self.history_size = int(history_size)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = self.embed_dim + 2

        self.history_encoder = nn.Sequential(
            nn.Linear(self.history_size * self.embed_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.prefix_encoder = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )
        # The constant successor coordinate is fixed to one after horizon
        # normalization, so the network only predicts the nontrivial moments.
        self.readout = nn.Linear(self.hidden_dim, self.embed_dim + 1)

    def forward(
        self,
        latent_history: torch.Tensor,
        action_prefix: torch.Tensor,
    ) -> torch.Tensor:
        if latent_history.ndim < 3 or action_prefix.ndim != latent_history.ndim:
            raise ValueError("history and action prefix must have matching ranks.")
        if latent_history.shape[-2:] != (self.history_size, self.embed_dim):
            raise ValueError(
                "latent_history must end with "
                f"({self.history_size}, {self.embed_dim})."
            )
        if action_prefix.shape[:-2] != latent_history.shape[:-2]:
            raise ValueError("history and action prefix leading shapes must match.")
        if action_prefix.shape[-2] <= 0 or action_prefix.shape[-1] != self.action_dim:
            raise ValueError("action_prefix has an invalid time or action dimension.")

        leading = latent_history.shape[:-2]
        flat_batch = math.prod(leading) if leading else 1
        history = latent_history.reshape(flat_batch, -1)
        actions = action_prefix.reshape(flat_batch, action_prefix.shape[-2], -1)
        initial = self.history_encoder(history).unsqueeze(0)
        encoded_actions = self.action_encoder(actions)
        states, _ = self.prefix_encoder(encoded_actions, initial)
        raw = self.readout(states)
        linear = raw[..., : self.embed_dim]
        squared_norm = functional.softplus(raw[..., self.embed_dim :])
        constant = torch.ones_like(squared_norm)
        successor = torch.cat((linear, squared_norm, constant), dim=-1)
        return successor.reshape(*leading, action_prefix.shape[-2], self.output_dim)


class ActionPrefixMomentHead(ActionPrefixSuccessorHead):
    """Predict future moments and construct successors by exact accumulation."""

    def __init__(self, *, gamma: float, **kwargs) -> None:
        super().__init__(**kwargs)
        if not 0.0 < gamma <= 1.0:
            raise ValueError("ActionPrefixMomentHead requires gamma in (0, 1].")
        self.gamma = float(gamma)

    def predict_moments(
        self,
        latent_history: torch.Tensor,
        action_prefix: torch.Tensor,
    ) -> torch.Tensor:
        return super().forward(latent_history, action_prefix)

    def forward(
        self,
        latent_history: torch.Tensor,
        action_prefix: torch.Tensor,
    ) -> torch.Tensor:
        moments = self.predict_moments(latent_history, action_prefix)
        return finite_horizon_successor_from_moments(moments, gamma=self.gamma)


@dataclass(frozen=True)
class MultiHorizonSuccessorOutput:
    """Joint losses for the two descriptions of one future trajectory."""

    prediction: torch.Tensor
    target: torch.Tensor
    latent_loss: torch.Tensor
    successor_loss: torch.Tensor
    recurrence_loss: torch.Tensor
    latent_mse_by_horizon: torch.Tensor
    successor_mse_by_horizon: torch.Tensor
    recurrence_mse_by_horizon: torch.Tensor


@dataclass(frozen=True)
class SuccessorSequenceOutput:
    """The single predictive objective used by the S-only method."""

    prediction: torch.Tensor
    target: torch.Tensor
    moments: torch.Tensor
    recovered_future: torch.Tensor
    successor_loss: torch.Tensor
    successor_mse_by_horizon: torch.Tensor
    recovered_latent_mse_by_horizon: torch.Tensor


@dataclass(frozen=True)
class MomentSequenceOutput:
    """Direct all-horizon moment supervision with derived successors."""

    moments: torch.Tensor
    target_moments: torch.Tensor
    prediction: torch.Tensor
    target: torch.Tensor
    recovered_future: torch.Tensor
    moment_loss: torch.Tensor
    moment_mse_by_horizon: torch.Tensor
    successor_mse_by_horizon: torch.Tensor
    recovered_latent_mse_by_horizon: torch.Tensor


def _mse_by_horizon(error: torch.Tensor) -> torch.Tensor:
    dimensions = tuple(range(error.ndim - 2)) + (error.ndim - 1,)
    return error.square().mean(dim=dimensions)


def multi_horizon_successor_objective(
    head: ActionPrefixSuccessorHead,
    latent_history: torch.Tensor,
    action_prefix: torch.Tensor,
    predicted_future: torch.Tensor,
    target_future: torch.Tensor,
    *,
    gamma: float,
) -> MultiHorizonSuccessorOutput:
    """Supervise latent rollout, direct successor, and their exact overlap."""

    if predicted_future.shape != target_future.shape:
        raise ValueError("predicted_future and target_future must match.")
    if predicted_future.shape[:-2] != latent_history.shape[:-2]:
        raise ValueError("future and history leading shapes must match.")
    if predicted_future.shape[-2] != action_prefix.shape[-2]:
        raise ValueError("future and action-prefix horizons must match.")
    detached_target = target_future.detach()
    target = finite_horizon_successor_targets(detached_target, gamma=gamma)
    prediction = head(latent_history, action_prefix)
    recurrence = successor_recurrence_residual(
        prediction, predicted_future, gamma=gamma
    )
    zeros = torch.zeros_like(recurrence)
    return MultiHorizonSuccessorOutput(
        prediction=prediction,
        target=target,
        latent_loss=(predicted_future - detached_target).square().mean(),
        successor_loss=balanced_successor_mse(prediction, target),
        recurrence_loss=balanced_successor_mse(recurrence, zeros),
        latent_mse_by_horizon=_mse_by_horizon(predicted_future - detached_target),
        successor_mse_by_horizon=_mse_by_horizon(prediction - target),
        recurrence_mse_by_horizon=_mse_by_horizon(recurrence),
    )


def successor_sequence_objective(
    head: ActionPrefixMomentHead,
    latent_history: torch.Tensor,
    action_prefix: torch.Tensor,
    target_future: torch.Tensor,
    *,
    gamma: float,
    vector_reduction: str = "coordinate_mean",
) -> SuccessorSequenceOutput:
    """Train one successor sequence without latent or recurrence losses."""

    if target_future.shape[:-2] != latent_history.shape[:-2]:
        raise ValueError("future and history leading shapes must match.")
    if target_future.shape[-2] != action_prefix.shape[-2]:
        raise ValueError("future and action-prefix horizons must match.")
    if target_future.shape[-1] != head.embed_dim:
        raise ValueError("future latents and the successor head must share a dimension.")
    if not math.isclose(float(gamma), head.gamma):
        raise ValueError("The objective gamma differs from the head gamma.")

    moments = head.predict_moments(latent_history, action_prefix)
    prediction = finite_horizon_successor_from_moments(moments, gamma=gamma)
    target = finite_horizon_successor_targets(target_future, gamma=gamma)
    recovered_future = latent_sequence_from_successor(prediction, gamma=gamma)
    return SuccessorSequenceOutput(
        prediction=prediction,
        target=target,
        moments=moments,
        recovered_future=recovered_future,
        successor_loss=balanced_successor_mse(
            prediction,
            target,
            vector_reduction=vector_reduction,
        ),
        successor_mse_by_horizon=_mse_by_horizon(prediction - target),
        recovered_latent_mse_by_horizon=_mse_by_horizon(
            recovered_future - target_future
        ),
    )


def moment_sequence_objective(
    head: ActionPrefixMomentHead,
    latent_history: torch.Tensor,
    action_prefix: torch.Tensor,
    target_future: torch.Tensor,
    *,
    gamma: float,
    vector_reduction: str = "group_sum",
) -> MomentSequenceOutput:
    """Supervise every future lifted moment with stop-gradient targets."""

    if target_future.shape[:-2] != latent_history.shape[:-2]:
        raise ValueError("future and history leading shapes must match.")
    if target_future.shape[-2] != action_prefix.shape[-2]:
        raise ValueError("future and action-prefix horizons must match.")
    if target_future.shape[-1] != head.embed_dim:
        raise ValueError("future latents and the successor head must share a dimension.")
    if not math.isclose(float(gamma), head.gamma):
        raise ValueError("The objective gamma differs from the head gamma.")

    detached_target = target_future.detach()
    target_moments = successor_feature_basis(detached_target)
    moments = head.predict_moments(latent_history, action_prefix)
    prediction = finite_horizon_successor_from_moments(moments, gamma=gamma)
    target = finite_horizon_successor_from_moments(target_moments, gamma=gamma)
    recovered_future = latent_sequence_from_successor(prediction, gamma=gamma)
    return MomentSequenceOutput(
        moments=moments,
        target_moments=target_moments,
        prediction=prediction,
        target=target,
        recovered_future=recovered_future,
        moment_loss=balanced_successor_mse(
            moments,
            target_moments,
            vector_reduction=vector_reduction,
        ),
        moment_mse_by_horizon=_mse_by_horizon(moments - target_moments),
        successor_mse_by_horizon=_mse_by_horizon(prediction - target),
        recovered_latent_mse_by_horizon=_mse_by_horizon(
            recovered_future - detached_target
        ),
    )


__all__ = [
    "ActionPrefixMomentHead",
    "ActionPrefixSuccessorHead",
    "MultiHorizonSuccessorOutput",
    "MomentSequenceOutput",
    "SuccessorSequenceOutput",
    "balanced_successor_mse",
    "discounted_prefix_mass",
    "finite_horizon_successor_from_moments",
    "finite_horizon_successor_targets",
    "latent_sequence_from_successor",
    "multi_horizon_successor_objective",
    "moment_sequence_objective",
    "successor_moments_from_sequence",
    "successor_recurrence_residual",
    "successor_sequence_objective",
]
