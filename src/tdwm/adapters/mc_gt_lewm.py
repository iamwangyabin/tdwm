"""Stable World Model planning adapter for MC-GT-LeWM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from tdwm.methods.goal_tail_value import GoalTailValue


class MCGoalTailLeWM(nn.Module):
    """Add the supervised MC goal-tail value to LeWM's terminal CEM cost."""

    def __init__(
        self,
        world_model: nn.Module,
        value: GoalTailValue,
        *,
        history_size: int = 3,
        action_block_dim: int = 25,
        tail_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if history_size <= 0:
            raise ValueError("history_size must be positive.")
        if action_block_dim <= 0:
            raise ValueError("action_block_dim must be positive.")
        if tail_weight < 0.0:
            raise ValueError("tail_weight must be non-negative.")
        expected_history_dim = (
            history_size * value.goal_dim
            + (history_size - 1) * action_block_dim
        )
        if value.history_dim != expected_history_dim:
            raise ValueError(
                f"Expected value history dimension {expected_history_dim}, "
                f"found {value.history_dim}."
            )
        self.world_model = world_model
        self.value = value
        self.history_size = int(history_size)
        self.action_block_dim = int(action_block_dim)
        self.tail_weight = float(tail_weight)

    def encode(self, info: dict[str, Any]) -> dict[str, Any]:
        return self.world_model.encode(info)

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        return self.world_model.predict(emb, act_emb)

    def rollout(
        self,
        info: dict[str, Any],
        action_sequence: torch.Tensor,
        history_size: int | None = None,
    ) -> dict[str, Any]:
        return self.world_model.rollout(
            info,
            action_sequence,
            history_size=history_size or self.history_size,
        )

    def get_cost(
        self, info_dict: dict[str, Any], action_candidates: torch.Tensor
    ) -> torch.Tensor:
        """Return unchanged LeWM terminal cost plus the terminal MC tail value."""

        if "goal" not in info_dict:
            raise AssertionError("goal not in info_dict")
        if action_candidates.ndim != 4:
            raise ValueError(
                "action_candidates must have shape (batch, samples, horizon, dim)."
            )
        if action_candidates.shape[-2] < self.history_size - 1:
            raise ValueError("The plan is too short to build terminal action history.")
        if action_candidates.shape[-1] != self.action_block_dim:
            raise ValueError(
                f"Expected action block dimension {self.action_block_dim}, "
                f"found {action_candidates.shape[-1]}."
            )

        goal_embedding = self._get_goal_embedding(info_dict)
        rollout_info = self.world_model.rollout(
            info_dict,
            action_candidates,
            history_size=self.history_size,
        )
        predicted = rollout_info.get("predicted_emb")
        if predicted is None or predicted.ndim != 4:
            raise ValueError(
                "LeWM rollout must return predicted_emb with shape "
                "(batch, samples, time, dim)."
            )
        if predicted.shape[-2] < self.history_size:
            raise ValueError("LeWM rollout is too short to build terminal history.")
        if predicted.shape[-1] != self.value.goal_dim:
            raise ValueError("LeWM latent dimension differs from the value checkpoint.")

        original_cost = self.world_model.criterion(rollout_info)
        latent_history = predicted[..., -self.history_size :, :].flatten(-2)
        action_count = self.history_size - 1
        action_history = action_candidates[
            ..., action_candidates.shape[-2] - action_count :, :
        ].flatten(-2)
        history = torch.cat((latent_history, action_history), dim=-1)
        goal = self._broadcast_goal(goal_embedding, predicted)
        tail_value = self.value(history, goal)

        # LeWM 0.1.1 sums terminal squared error over latent dimensions, while
        # MC-GT targets use the mean. Scaling V by D preserves the unchanged
        # upstream cost and is rank-equivalent to D * (c_mean + V).
        return original_cost + (
            self.tail_weight * predicted.shape[-1] * tail_value
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
        goal_embedding: torch.Tensor, predicted: torch.Tensor
    ) -> torch.Tensor:
        goal = goal_embedding
        while goal.ndim > 2:
            goal = goal[..., -1, :]
        if goal.ndim != 2 or goal.shape[0] != predicted.shape[0]:
            raise ValueError("Expected one goal embedding per environment.")
        return goal.unsqueeze(1).expand(
            predicted.shape[0], predicted.shape[1], goal.shape[-1]
        )


def load_mc_goal_tail_value(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[GoalTailValue, dict[str, Any], dict[str, Any]]:
    """Load the value head written by the formal MC-GT-LeWM trainer."""

    payload = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    if payload.get("method") != "mc_gt_lewm":
        raise ValueError("The value checkpoint is not an MC-GT-LeWM checkpoint.")
    if payload.get("objective_version") != 1:
        raise ValueError("MC-GT-LeWM planning requires objective_version 1.")
    config = dict(payload["value_config"])
    if config.get("objective") != "supervised_mc":
        raise ValueError("MC-GT-LeWM planning requires supervised MC targets.")
    value = GoalTailValue(
        history_dim=int(config["history_dim"]),
        goal_dim=int(config["goal_dim"]),
        hidden_dim=int(config["hidden_dim"]),
    )
    value.load_state_dict(payload["value_state_dict"])
    value.eval()
    value.requires_grad_(False)
    return value, config, payload


def make_mc_goal_tail_policy(
    *,
    world_model: nn.Module,
    value: GoalTailValue,
    planning: dict[str, Any],
    history_size: int,
    action_block_dim: int,
    tail_weight: float,
    process: dict[str, Any] | None = None,
    transform: dict[str, Any] | None = None,
    device: str | torch.device = "cpu",
):
    """Build the unchanged upstream CEM policy around MC-GT-LeWM."""

    import stable_worldmodel as swm

    wrapped = MCGoalTailLeWM(
        world_model,
        value,
        history_size=history_size,
        action_block_dim=action_block_dim,
        tail_weight=tail_weight,
    ).to(device)
    wrapped.eval()
    wrapped.requires_grad_(False)
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
        history_len=planning["plan_config_history_len"],
        action_block=planning["action_block"],
        warm_start=planning["warm_start"],
    )
    return swm.policy.WorldModelPolicy(
        solver=solver,
        config=config,
        process=process,
        transform=transform,
    )
