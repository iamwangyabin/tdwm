"""Stable World Model adapters for GT-LeWM inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from tdwm.methods.goal_tail import (
    GoalTailValue,
    discounted_goal_tail_target,
)


class GoalTailLeWM(nn.Module):
    """Wrap a public LeWM model with the goal-conditioned tail value.

    The wrapper intentionally exposes Stable World Model's ``get_cost``
    protocol instead of replacing its CEM solver.  CEM still proposes
    arbitrary action sequences through LeWM's rollout; this adapter only
    changes the candidate objective to the discounted latent path cost plus a
    terminal value bootstrap.
    """

    def __init__(
        self,
        world_model: nn.Module,
        value: GoalTailValue,
        *,
        gamma: float,
        history_size: int | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 <= gamma < 1.0:
            raise ValueError("gamma must lie in [0, 1).")
        self.world_model = world_model
        self.value = value
        self.gamma = float(gamma)
        self.history_size = history_size

    def encode(self, info: dict[str, Any]) -> dict[str, Any]:
        """Delegate encoding to the public LeWM model method."""

        return self.world_model.encode(info)

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        """Delegate one-step prediction to the public LeWM method."""

        return self.world_model.predict(emb, act_emb)

    def rollout(
        self,
        info: dict[str, Any],
        action_sequence: torch.Tensor,
        history_size: int | None = None,
    ) -> dict[str, Any]:
        """Delegate autoregressive rollout to the public LeWM method."""

        return self.world_model.rollout(
            info,
            action_sequence,
            history_size=history_size or self.history_size,
        )

    def criterion(
        self, info_dict: dict[str, Any], action_candidates: torch.Tensor
    ) -> torch.Tensor:
        """Implement Stable World Model's costable protocol."""

        return self.get_cost(info_dict, action_candidates)

    def get_cost(
        self, info_dict: dict[str, Any], action_candidates: torch.Tensor
    ) -> torch.Tensor:
        """Score candidates with discounted latent path cost and tail value."""

        if "goal" not in info_dict:
            raise AssertionError("goal not in info_dict")
        goal_embedding = self._get_goal_embedding(info_dict)
        rollout_info = self.world_model.rollout(info_dict, action_candidates)
        predicted = rollout_info["predicted_emb"]
        if predicted.ndim != 4:
            raise ValueError("LeWM rollout must return (batch, samples, time, dim).")

        history = self.history_size
        if history is None:
            history = getattr(self.world_model.predictor, "num_frames", 1)
        start = min(int(history), predicted.shape[-2] - 1)
        future = predicted[..., start:, :]
        if future.shape[-2] == 0:
            future = predicted[..., -1:, :]

        goal = self._broadcast_goal(goal_embedding, future)
        terminal_value = self.value(future[..., -1, :], goal).squeeze(-1)
        return discounted_goal_tail_target(
            future,
            goal,
            terminal_value,
            gamma=self.gamma,
        )

    def _get_goal_embedding(self, info_dict: dict[str, Any]) -> torch.Tensor:
        if "goal_emb" in info_dict:
            return info_dict["goal_emb"]

        goal = {
            key: value[:, 0]
            for key, value in info_dict.items()
            if torch.is_tensor(value)
        }
        goal["pixels"] = goal["goal"]
        for key in list(info_dict):
            if key.startswith("goal_"):
                goal[key[len("goal_") :]] = goal.pop(key)
        goal.pop("action", None)
        encoded_goal = self.world_model.encode(goal)
        info_dict["goal_emb"] = encoded_goal["emb"]
        return info_dict["goal_emb"]

    @staticmethod
    def _broadcast_goal(
        goal_embedding: torch.Tensor, future: torch.Tensor
    ) -> torch.Tensor:
        """Normalize cached goal shapes produced by WorldModelPolicy/CEM."""

        goal = goal_embedding
        while goal.ndim > 2:
            goal = goal[..., -1, :]
        if goal.ndim != 2 or goal.shape[0] != future.shape[0]:
            raise ValueError("Expected one goal embedding per environment.")
        return goal.unsqueeze(1).expand(
            future.shape[0], future.shape[1], goal.shape[-1]
        )


def load_goal_tail_value(
    checkpoint_path: str | Path, *, map_location: str | torch.device = "cpu"
) -> tuple[GoalTailValue, dict[str, Any]]:
    """Load the value-head export written by the GT-LeWM trainer."""

    payload = torch.load(checkpoint_path, map_location=map_location)
    config = dict(payload["value_config"])
    value = GoalTailValue(
        embed_dim=int(config["embed_dim"]),
        hidden_dim=int(config["hidden_dim"]),
    )
    value.load_state_dict(payload["value_state_dict"])
    value.eval()
    return value, config


def make_goal_tail_policy(
    *,
    world_model: nn.Module,
    value: GoalTailValue,
    planning: dict[str, Any],
    tail_value: dict[str, Any] | None = None,
    gamma: float | None = None,
    history_size: int | None = None,
    process: dict[str, Any] | None = None,
    transform: dict[str, Any] | None = None,
    device: str | torch.device = "cpu",
):
    """Build the standard Stable World Model CEM policy for GT-LeWM.

    All planning infrastructure remains in ``stable_worldmodel``.  The only
    project-owned object passed to CEM is :class:`GoalTailLeWM`.
    """

    import stable_worldmodel as swm

    tail_config = tail_value or {}
    resolved_gamma = gamma
    if resolved_gamma is None:
        resolved_gamma = tail_config.get("gamma", planning.get("gamma"))
    if resolved_gamma is None:
        raise ValueError("Pass gamma or configure tail_value.gamma for GT-LeWM.")
    wrapped = GoalTailLeWM(
        world_model,
        value,
        gamma=resolved_gamma,
        history_size=history_size,
    ).to(device)
    solver = swm.solver.CEMSolver(
        model=wrapped,
        batch_size=planning["solver_batch_size"],
        num_samples=planning["candidates"],
        var_scale=planning["initial_variance"],
        n_steps=planning["iterations"],
        topk=planning["elites"],
        device=device,
        seed=planning["planning_seed"],
    )
    config = swm.PlanConfig(
        horizon=planning["horizon"],
        receding_horizon=planning["receding_horizon"],
        history_len=history_size or 1,
        action_block=planning["action_block"],
        warm_start=planning["warm_start"],
    )
    return swm.policy.WorldModelPolicy(
        solver=solver,
        config=config,
        process=process,
        transform=transform,
    )
