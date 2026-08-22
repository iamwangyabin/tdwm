"""Supervised Monte Carlo goal-tail value for frozen LeWM latents."""

from __future__ import annotations

import torch
from torch import nn


def latent_goal_cost(latent: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """Return mean squared latent distance along the final dimension."""

    if latent.shape[-1] != goal.shape[-1]:
        raise ValueError("latent and goal must share their final dimension.")
    return (latent - goal).pow(2).mean(dim=-1)


class GoalTailValue(nn.Module):
    """Two-layer MLP implementing ``V(history, goal) -> scalar cost``."""

    def __init__(
        self,
        history_dim: int,
        goal_dim: int,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        if history_dim <= 0 or goal_dim <= 0 or hidden_dim <= 0:
            raise ValueError("history_dim, goal_dim, and hidden_dim must be positive.")
        self.history_dim = int(history_dim)
        self.goal_dim = int(goal_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(self.history_dim + self.goal_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, history: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        if history.shape[:-1] != goal.shape[:-1]:
            raise ValueError("history and goal must share their leading dimensions.")
        if history.shape[-1] != self.history_dim:
            raise ValueError(
                f"Expected history final dimension {self.history_dim}, "
                f"found {history.shape[-1]}."
            )
        if goal.shape[-1] != self.goal_dim:
            raise ValueError(
                f"Expected goal final dimension {self.goal_dim}, "
                f"found {goal.shape[-1]}."
            )
        return self.net(torch.cat((history, goal), dim=-1)).squeeze(-1)


class BoundaryAnchoredGoalTailValue(nn.Module):
    """Non-negative goal tail with an exact ``V(h, z_current) = 0`` boundary.

    A shared scalar potential is evaluated once at the requested goal and once
    at the current latent stored in the history. Squaring their difference
    makes the value non-negative and enforces the terminal boundary by
    construction instead of relying on a soft penalty seen during training.
    """

    def __init__(
        self,
        history_dim: int,
        goal_dim: int,
        history_size: int,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        if history_dim <= 0 or goal_dim <= 0 or hidden_dim <= 0:
            raise ValueError("history_dim, goal_dim, and hidden_dim must be positive.")
        if history_size <= 0:
            raise ValueError("history_size must be positive.")
        if history_dim < history_size * goal_dim:
            raise ValueError("history_dim does not contain the latent history.")
        self.history_dim = int(history_dim)
        self.goal_dim = int(goal_dim)
        self.history_size = int(history_size)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(self.history_dim + self.goal_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def current_latent(self, history: torch.Tensor) -> torch.Tensor:
        """Return the final latent from a flattened LeWM history."""

        if history.shape[-1] != self.history_dim:
            raise ValueError(
                f"Expected history final dimension {self.history_dim}, "
                f"found {history.shape[-1]}."
            )
        start = (self.history_size - 1) * self.goal_dim
        return history[..., start : start + self.goal_dim]

    def _potential(self, history: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((history, goal), dim=-1)).squeeze(-1)

    def forward(self, history: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        if history.shape[:-1] != goal.shape[:-1]:
            raise ValueError("history and goal must share their leading dimensions.")
        if goal.shape[-1] != self.goal_dim:
            raise ValueError(
                f"Expected goal final dimension {self.goal_dim}, "
                f"found {goal.shape[-1]}."
            )
        current = self.current_latent(history)
        delta = self._potential(history, goal) - self._potential(history, current)
        return delta.square()


def build_goal_tail_history(
    latents: torch.Tensor,
    actions: torch.Tensor,
    *,
    current_index: int,
    history_size: int,
) -> torch.Tensor:
    """Flatten ``(z history, preceding actions)`` for the current state."""

    if latents.ndim != 3 or actions.ndim != 3:
        raise ValueError("latents and actions must have shape (batch, time, dim).")
    if latents.shape[:2] != actions.shape[:2]:
        raise ValueError("latents and actions must share batch and time dimensions.")
    if history_size <= 0:
        raise ValueError("history_size must be positive.")
    first_index = current_index - history_size + 1
    if first_index < 0 or current_index >= latents.shape[1]:
        raise ValueError("current_index does not contain the requested history.")

    latent_history = latents[:, first_index : current_index + 1].flatten(1)
    action_history = actions[:, first_index:current_index].flatten(1)
    return torch.cat((latent_history, action_history), dim=-1)


def monte_carlo_goal_tail_targets(
    latents: torch.Tensor,
    *,
    current_index: int,
    max_goal_offset: int,
    gamma: float,
) -> torch.Tensor:
    """Return the MC target for every future offset from 1 to ``max_goal_offset``.

    The returned tensor has shape ``(batch, max_goal_offset)``. Entry ``j`` is
    supervised with goal ``z[current_index + j + 1]`` and contains no bootstrap.
    """

    if latents.ndim != 3:
        raise ValueError("latents must have shape (batch, time, dim).")
    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must lie in [0, 1).")
    if max_goal_offset <= 0:
        raise ValueError("max_goal_offset must be positive.")
    if current_index < 0:
        raise ValueError("current_index must be non-negative.")
    if current_index + max_goal_offset >= latents.shape[1]:
        raise ValueError("latents do not contain the requested future goals.")

    future = latents[
        :, current_index + 1 : current_index + max_goal_offset + 1
    ]
    goals = future
    pairwise_cost = latent_goal_cost(
        future.unsqueeze(1),
        goals.unsqueeze(2),
    )
    offsets = torch.arange(
        1,
        max_goal_offset + 1,
        device=latents.device,
    )
    future_steps = offsets
    valid = future_steps.unsqueeze(0) <= offsets.unsqueeze(1)
    weights = torch.pow(latents.new_tensor(gamma), future_steps - 1)
    return (1.0 - gamma) * (
        pairwise_cost * valid.unsqueeze(0) * weights.reshape(1, 1, -1)
    ).sum(dim=-1)


def select_goal_tail_samples(
    latents: torch.Tensor,
    actions: torch.Tensor,
    goal_offsets: torch.Tensor,
    *,
    current_index: int,
    history_size: int,
    max_goal_offset: int,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select one hindsight goal and MC target per batch element."""

    if goal_offsets.ndim != 1 or goal_offsets.shape[0] != latents.shape[0]:
        raise ValueError("goal_offsets must contain one offset per batch element.")
    if goal_offsets.dtype not in (torch.int32, torch.int64):
        raise TypeError("goal_offsets must use an integer dtype.")
    if torch.any(goal_offsets < 1) or torch.any(goal_offsets > max_goal_offset):
        raise ValueError("goal_offsets must lie in [1, max_goal_offset].")

    history = build_goal_tail_history(
        latents,
        actions,
        current_index=current_index,
        history_size=history_size,
    )
    targets = monte_carlo_goal_tail_targets_for_offsets(
        latents,
        goal_offsets,
        current_index=current_index,
        max_goal_offset=max_goal_offset,
        gamma=gamma,
    )
    rows = torch.arange(latents.shape[0], device=latents.device)
    goal_indices = current_index + goal_offsets
    goals = latents[rows, goal_indices]
    return history, goals, targets


def monte_carlo_goal_tail_targets_for_offsets(
    latents: torch.Tensor,
    goal_offsets: torch.Tensor,
    *,
    current_index: int,
    max_goal_offset: int,
    gamma: float,
) -> torch.Tensor:
    """Return one normalized MC target per requested future-goal offset."""

    if latents.ndim != 3:
        raise ValueError("latents must have shape (batch, time, dim).")
    if goal_offsets.ndim != 1 or goal_offsets.shape[0] != latents.shape[0]:
        raise ValueError("goal_offsets must contain one offset per batch element.")
    if goal_offsets.dtype not in (torch.int32, torch.int64):
        raise TypeError("goal_offsets must use an integer dtype.")
    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must lie in [0, 1).")
    if max_goal_offset <= 0:
        raise ValueError("max_goal_offset must be positive.")
    if current_index < 0 or current_index + max_goal_offset >= latents.shape[1]:
        raise ValueError("latents do not contain the requested future goals.")
    if torch.any(goal_offsets < 1) or torch.any(goal_offsets > max_goal_offset):
        raise ValueError("goal_offsets must lie in [1, max_goal_offset].")

    rows = torch.arange(latents.shape[0], device=latents.device)
    goals = latents[rows, current_index + goal_offsets]
    future = latents[
        :, current_index + 1 : current_index + max_goal_offset + 1
    ]
    step_costs = latent_goal_cost(future, goals.unsqueeze(1))
    future_steps = torch.arange(
        1,
        max_goal_offset + 1,
        device=latents.device,
    )
    valid = future_steps.unsqueeze(0) <= goal_offsets.unsqueeze(1)
    weights = torch.pow(latents.new_tensor(gamma), future_steps - 1)
    return (1.0 - gamma) * (
        step_costs * valid * weights.unsqueeze(0)
    ).sum(dim=-1)
