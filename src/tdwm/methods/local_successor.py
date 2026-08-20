"""Local dynamics and policy-conditioned successor components for LS-LeWM."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


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


def successor_td_target(
    next_latent: torch.Tensor,
    bootstrap: torch.Tensor,
    *,
    gamma: float,
    terminal: bool | torch.Tensor = False,
) -> torch.Tensor:
    """Build the normalized one-step successor-feature Bellman target."""

    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must lie in [0, 1).")
    immediate = (1.0 - gamma) * successor_feature_basis(next_latent)
    if bootstrap.shape != immediate.shape:
        raise ValueError("bootstrap must match the lifted next-latent shape.")
    target = immediate + gamma * bootstrap
    if isinstance(terminal, bool):
        return immediate if terminal else target
    terminal = terminal.to(device=target.device, dtype=torch.bool)
    if terminal.shape != target.shape[:-1]:
        raise ValueError("terminal must match the target leading dimensions.")
    return torch.where(terminal.unsqueeze(-1), immediate, target)


def _flatten_history(
    latent_history: torch.Tensor,
    previous_actions: torch.Tensor,
    *,
    history_size: int,
    embed_dim: int,
    action_dim: int,
) -> torch.Tensor:
    if latent_history.shape[-2:] != (history_size, embed_dim):
        raise ValueError(
            "latent_history must end with "
            f"({history_size}, {embed_dim}), found {latent_history.shape[-2:]}."
        )
    if previous_actions.shape[:-2] != latent_history.shape[:-2]:
        raise ValueError("latent and action histories must share leading dimensions.")
    if previous_actions.shape[-2:] != (max(0, history_size - 1), action_dim):
        raise ValueError(
            "previous_actions must end with "
            f"({max(0, history_size - 1)}, {action_dim})."
        )
    return torch.cat(
        (
            latent_history.flatten(start_dim=-2),
            previous_actions.flatten(start_dim=-2),
        ),
        dim=-1,
    )


class GoalConditionedPolicy(nn.Module):
    """Low-capacity continuation policy over normalized action blocks."""

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
            raise ValueError("Policy dimensions must be positive.")
        self.embed_dim = int(embed_dim)
        self.action_dim = int(action_dim)
        self.history_size = int(history_size)
        self.hidden_dim = int(hidden_dim)
        context_dim = history_size * embed_dim + (history_size - 1) * action_dim
        input_dim = context_dim + 2 * embed_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        latent_history: torch.Tensor,
        previous_actions: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        context = _flatten_history(
            latent_history,
            previous_actions,
            history_size=self.history_size,
            embed_dim=self.embed_dim,
            action_dim=self.action_dim,
        )
        if goal.shape != latent_history.shape[:-2] + (self.embed_dim,):
            raise ValueError("goal must match the history leading dimensions.")
        current = latent_history[..., -1, :]
        return self.network(torch.cat((context, goal, current - goal), dim=-1))


class SuccessorPredictor(nn.Module):
    """Goal-policy-conditioned discounted successor feature predictor."""

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
            raise ValueError("Successor dimensions must be positive.")
        self.embed_dim = int(embed_dim)
        self.action_dim = int(action_dim)
        self.history_size = int(history_size)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = self.embed_dim + 2
        context_dim = history_size * embed_dim + (history_size - 1) * action_dim
        input_dim = context_dim + action_dim + 2 * embed_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def forward(
        self,
        latent_history: torch.Tensor,
        previous_actions: torch.Tensor,
        action: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        context = _flatten_history(
            latent_history,
            previous_actions,
            history_size=self.history_size,
            embed_dim=self.embed_dim,
            action_dim=self.action_dim,
        )
        leading = latent_history.shape[:-2]
        if action.shape != leading + (self.action_dim,):
            raise ValueError("action must match the history leading dimensions.")
        if goal.shape != leading + (self.embed_dim,):
            raise ValueError("goal must match the history leading dimensions.")
        current = latent_history[..., -1, :]
        return self.network(
            torch.cat((context, action, goal, current - goal), dim=-1)
        )


class LocalSuccessorHeads(nn.Module):
    """Trainable policy and successor heads sharing a LeWM representation."""

    def __init__(
        self,
        *,
        embed_dim: int,
        action_dim: int,
        history_size: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.action_dim = int(action_dim)
        self.history_size = int(history_size)
        self.hidden_dim = int(hidden_dim)
        common = {
            "embed_dim": self.embed_dim,
            "action_dim": self.action_dim,
            "history_size": self.history_size,
            "hidden_dim": self.hidden_dim,
        }
        self.policy = GoalConditionedPolicy(**common)
        self.successor = SuccessorPredictor(**common)

    def make_target(self) -> "LocalSuccessorHeads":
        target = copy.deepcopy(self)
        target.requires_grad_(False)
        return target


def _latent_histories(latents: torch.Tensor, end_indices: range, size: int) -> torch.Tensor:
    return torch.stack(
        [latents[:, end - size + 1 : end + 1] for end in end_indices], dim=1
    )


def _previous_action_histories(
    actions: torch.Tensor, end_indices: range, size: int
) -> torch.Tensor:
    history = size - 1
    if history == 0:
        batch = actions.shape[0]
        count = len(end_indices)
        return actions.new_empty(batch, count, 0, actions.shape[-1])
    return torch.stack(
        [actions[:, end - history : end] for end in end_indices], dim=1
    )


@dataclass(frozen=True)
class SuccessorTDOutput:
    """Losses and diagnostics for policy-conditioned successor training."""

    td_loss: torch.Tensor
    boundary_loss: torch.Tensor
    policy_loss: torch.Tensor
    scalar_prediction_mean: torch.Tensor
    scalar_target_mean: torch.Tensor
    pair_count: int


def future_goal_successor_objective(
    heads: LocalSuccessorHeads,
    target_heads: LocalSuccessorHeads,
    latents: torch.Tensor,
    actions: torch.Tensor,
    *,
    gamma: float,
    context_latents: torch.Tensor | None = None,
    train_policy: bool = True,
) -> SuccessorTDOutput:
    """Evaluate one-step off-policy TD over every future goal in a clip.

    The current transition comes from the offline dataset. Continuation actions
    come from the same target policy used by MPC, which makes the Bellman target
    an evaluation target for an executable policy rather than an implicit data
    behavior distribution.
    """

    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must lie in [0, 1).")
    if latents.ndim != 3 or actions.ndim != 3:
        raise ValueError("latents and actions must have shape (batch, time, dim).")
    if latents.shape[:2] != actions.shape[:2]:
        raise ValueError("latents and actions must share batch and time axes.")
    if latents.shape[-1] != heads.embed_dim:
        raise ValueError("Unexpected latent dimension.")
    if actions.shape[-1] != heads.action_dim:
        raise ValueError("Unexpected action dimension.")
    if heads.history_size != target_heads.history_size:
        raise ValueError("Online and target heads must share history size.")
    context_latents = latents if context_latents is None else context_latents
    if context_latents.shape != latents.shape:
        raise ValueError("context_latents must match latents.")

    history = heads.history_size
    first_current = history - 1
    max_offset = latents.shape[1] - first_current - 1
    if max_offset <= 0:
        raise ValueError("The clip must contain a transition after the history.")

    td_losses: list[torch.Tensor] = []
    policy_losses: list[torch.Tensor] = []
    scalar_predictions: list[torch.Tensor] = []
    scalar_targets: list[torch.Tensor] = []
    pair_count = 0

    for offset in range(1, max_offset + 1):
        pair_end = latents.shape[1] - offset
        current_indices = range(first_current, pair_end)
        next_indices = range(first_current + 1, pair_end + 1)
        current_history = _latent_histories(
            context_latents, current_indices, history
        )
        next_history = _latent_histories(context_latents, next_indices, history)
        current_previous = _previous_action_histories(
            actions, current_indices, history
        )
        next_previous = _previous_action_histories(actions, next_indices, history)
        current_action = actions[:, first_current:pair_end]
        goals = latents[:, first_current + offset :]

        prediction = heads.successor(
            current_history, current_previous, current_action, goals
        )
        with torch.no_grad():
            detached_goal = goals.detach()
            next_action = target_heads.policy(
                next_history.detach(), next_previous.detach(), detached_goal
            )
            bootstrap = target_heads.successor(
                next_history.detach(),
                next_previous.detach(),
                next_action,
                detached_goal,
            )
            target = successor_td_target(
                latents[:, first_current + 1 : pair_end + 1].detach(),
                bootstrap,
                gamma=gamma,
                terminal=offset == 1,
            )

        td_losses.append((prediction - target).square().mean())
        weights = goal_cost_weights(goals.detach())
        scalar_predictions.append(
            (prediction.detach() * weights).sum(dim=-1).reshape(-1)
        )
        scalar_targets.append((target * weights).sum(dim=-1).reshape(-1))
        pair_count += prediction.shape[0] * prediction.shape[1]

        if train_policy:
            policy_prediction = heads.policy(
                current_history.detach(), current_previous.detach(), goals.detach()
            )
            policy_losses.append(
                (policy_prediction - current_action.detach()).square().mean()
            )

    goal_indices = range(first_current + 1, latents.shape[1])
    goal_history = _latent_histories(context_latents, goal_indices, history)
    goal_previous = _previous_action_histories(actions, goal_indices, history)
    boundary_goals = latents[:, first_current + 1 :]
    with torch.no_grad():
        boundary_action = heads.policy(
            goal_history.detach(), goal_previous.detach(), boundary_goals.detach()
        )
    boundary_prediction = heads.successor(
        goal_history, goal_previous, boundary_action, boundary_goals
    )
    boundary_loss = boundary_prediction.square().mean()

    zero = latents.new_zeros(())
    return SuccessorTDOutput(
        td_loss=torch.stack(td_losses).mean(),
        boundary_loss=boundary_loss,
        policy_loss=torch.stack(policy_losses).mean() if policy_losses else zero,
        scalar_prediction_mean=torch.cat(scalar_predictions).mean(),
        scalar_target_mean=torch.cat(scalar_targets).mean(),
        pair_count=pair_count,
    )


@torch.no_grad()
def ema_update(target: nn.Module, source: nn.Module, *, decay: float) -> None:
    """Update target parameters and floating buffers from a source module."""

    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must lie in [0, 1).")
    target_parameters: Iterable[torch.Tensor] = target.parameters()
    source_parameters: Iterable[torch.Tensor] = source.parameters()
    for target_parameter, source_parameter in zip(
        target_parameters, source_parameters, strict=True
    ):
        target_parameter.mul_(decay).add_(source_parameter, alpha=1.0 - decay)
    for target_buffer, source_buffer in zip(
        target.buffers(), source.buffers(), strict=True
    ):
        if target_buffer.is_floating_point():
            target_buffer.mul_(decay).add_(source_buffer, alpha=1.0 - decay)
        else:
            target_buffer.copy_(source_buffer)
