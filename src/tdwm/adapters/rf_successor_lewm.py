"""Stable World Model adapter for reward-free Successor-LeWM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from tdwm.methods.rf_successor_lewm import (
    ActionPrefixMomentHead,
    ActionPrefixSuccessorHead,
)
from tdwm.methods.successor_geometry import latent_goal_cost, successor_goal_cost


class RewardFreeSuccessorLeWM(nn.Module):
    """Score supplied action prefixes without learning or invoking a policy."""

    def __init__(
        self,
        world_model: nn.Module,
        successor: ActionPrefixSuccessorHead | ActionPrefixMomentHead,
        *,
        max_horizon: int,
        successor_weight: float = 1.0,
        terminal_weight: float = 0.0,
        clamp_successor_cost: bool = True,
        planning_query: str = "discounted_successor",
    ) -> None:
        super().__init__()
        if max_horizon <= 0:
            raise ValueError("max_horizon must be positive.")
        if min(successor_weight, terminal_weight) < 0.0:
            raise ValueError("Planning weights must be non-negative.")
        if successor_weight + terminal_weight <= 0.0:
            raise ValueError("At least one planning cost must have positive weight.")
        if planning_query not in {"discounted_successor", "terminal_moment"}:
            raise ValueError("Unsupported reward-free planning query.")
        if planning_query == "terminal_moment" and not hasattr(
            successor, "predict_moments"
        ):
            raise ValueError("Terminal-moment planning requires a moment head.")
        self.world_model = world_model
        self.successor = successor
        self.max_horizon = int(max_horizon)
        self.successor_weight = float(successor_weight)
        self.terminal_weight = float(terminal_weight)
        self.clamp_successor_cost = bool(clamp_successor_cost)
        self.planning_query = planning_query

    @property
    def history_size(self) -> int:
        return self.successor.history_size

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

    def criterion(
        self, info_dict: dict[str, Any], action_candidates: torch.Tensor
    ) -> torch.Tensor:
        return self.get_cost(info_dict, action_candidates)

    def get_cost(
        self, info_dict: dict[str, Any], action_candidates: torch.Tensor
    ) -> torch.Tensor:
        """Return a goal query of the candidate-conditioned successor moments."""

        if "goal" not in info_dict and "goal_emb" not in info_dict:
            raise AssertionError("goal not in info_dict")
        if "pixels" not in info_dict or info_dict["pixels"].ndim < 3:
            raise ValueError("RF-Successor-LeWM requires an observation time axis.")
        if action_candidates.ndim != 4:
            raise ValueError(
                "action_candidates must have shape (batch, samples, horizon, dim)."
            )
        if action_candidates.shape[-1] != self.successor.action_dim:
            raise ValueError("Candidate action blocks have the wrong dimension.")
        batch, samples, horizon = action_candidates.shape[:3]
        if not 0 < horizon <= self.max_horizon:
            raise ValueError("Planning horizon exceeds successor training coverage.")

        future = None
        if self.terminal_weight > 0.0:
            observed_frames = int(info_dict["pixels"].shape[2])
            rollout = self.world_model.rollout(
                info_dict,
                action_candidates,
                history_size=self.history_size,
            )
            predicted = rollout.get("predicted_emb")
            if predicted is None or predicted.ndim != 4:
                raise ValueError(
                    "LeWM rollout must return (batch, samples, time, latent_dim)."
                )
            if predicted.shape[:2] != (batch, samples):
                raise ValueError("LeWM rollout does not match the CEM candidate batch.")
            if predicted.shape[-1] != self.successor.embed_dim:
                raise ValueError("LeWM and successor latent dimensions differ.")
            future = predicted[..., observed_frames:, :]
            if future.shape[-2] != horizon:
                raise ValueError("LeWM rollout future does not match the plan horizon.")
            history = self._pad_history(predicted[..., :observed_frames, :])
        else:
            history = self._encoded_history_for_samples(
                info_dict, batch=batch, samples=samples
            )
        if self.planning_query == "terminal_moment":
            score_features = self.successor.predict_moments(
                history, action_candidates
            )[..., -1, :]
        else:
            score_features = self.successor(history, action_candidates)[..., -1, :]
        goal = self._goal_for_samples(info_dict, batch=batch, samples=samples)
        successor_cost = successor_goal_cost(score_features, goal)
        if self.clamp_successor_cost:
            successor_cost = successor_cost.clamp_min(0.0)
        cost = self.successor_weight * successor_cost
        if future is not None:
            cost = cost + self.terminal_weight * latent_goal_cost(
                future[..., -1, :], goal
            )
        return cost

    def _encoded_history_for_samples(
        self,
        info: dict[str, Any],
        *,
        batch: int,
        samples: int,
    ) -> torch.Tensor:
        encoded = info.get("emb")
        if encoded is None:
            initial = {
                key: value[:, 0]
                for key, value in info.items()
                if torch.is_tensor(value)
            }
            initial.pop("action", None)
            initial.pop("act_emb", None)
            encoded = self.world_model.encode(initial)["emb"].detach().unsqueeze(1)
        elif encoded.ndim == 3:
            encoded = encoded.unsqueeze(1)
        if encoded.ndim != 4 or encoded.shape[0] != batch:
            raise ValueError("Encoded history must have shape (batch, samples, time, dim).")
        if encoded.shape[1] == 1:
            encoded = encoded.expand(batch, samples, -1, -1)
        elif encoded.shape[1] != samples:
            raise ValueError("Cached history does not match the CEM sample count.")
        if encoded.shape[-1] != self.successor.embed_dim:
            raise ValueError("Encoded history and successor latent dimensions differ.")
        info["emb"] = encoded
        return self._pad_history(encoded)

    def _pad_history(self, history: torch.Tensor) -> torch.Tensor:
        if history.shape[-1] != self.successor.embed_dim:
            raise ValueError("Unexpected latent history dimension.")
        if history.shape[-2] >= self.history_size:
            return history[..., -self.history_size :, :]
        if history.shape[-2] <= 0:
            raise ValueError("At least one observed latent is required.")
        padding = history[..., :1, :].expand(
            *history.shape[:-2], self.history_size - history.shape[-2], history.shape[-1]
        )
        return torch.cat((padding, history), dim=-2)

    def _goal_for_samples(
        self, info: dict[str, Any], *, batch: int, samples: int
    ) -> torch.Tensor:
        goal = self._get_or_encode_goal(info)
        while goal.ndim > 2:
            goal = goal[..., -1, :]
        if goal.ndim != 2 or goal.shape[0] != batch:
            raise ValueError("Expected one goal embedding per environment.")
        return goal.unsqueeze(1).expand(batch, samples, -1)

    def _get_or_encode_goal(self, info: dict[str, Any]) -> torch.Tensor:
        if "goal_emb" in info:
            return info["goal_emb"]
        goal_info = {
            key: value[:, 0]
            for key, value in info.items()
            if torch.is_tensor(value)
        }
        goal_info["pixels"] = goal_info["goal"]
        for key in list(goal_info):
            if key.startswith("goal_"):
                goal_info[key[len("goal_") :]] = goal_info.pop(key)
        goal_info.pop("action", None)
        encoded = self.world_model.encode(goal_info)["emb"]
        info["goal_emb"] = encoded
        return encoded


def load_rf_successor_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[ActionPrefixSuccessorHead, dict[str, Any], dict[str, Any]]:
    """Load the deployment checkpoint written by the joint trainer."""

    payload = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    method = payload.get("method")
    if method not in {
        "rf_successor_lewm",
        "rf_successor_sequence_wm",
        "rf_balanced_successor_sequence_wm",
        "rf_ema_balanced_successor_sequence_wm",
        "rf_direct_moment_sequence_wm",
        "rf_e2e_moment_sequence_wm",
    }:
        raise ValueError("The checkpoint is not a supported reward-free successor model.")
    objective_version = payload.get("objective_version")
    if objective_version not in {1, 2, 3, 4, 5, 6}:
        raise ValueError("Unsupported reward-free successor objective version.")
    if payload.get("deployment_checkpoint_version") != 1:
        raise ValueError("Unsupported RF-Successor-LeWM checkpoint version.")
    if "world_model_state_dict" not in payload:
        raise ValueError("The deployment checkpoint is missing the joint LeWM.")
    config = dict(payload["successor_config"])
    if config.get("goal_conditioning") != "none":
        raise ValueError("The successor checkpoint is not reward-free.")
    if config.get("action_conditioning") != "causal_prefix":
        raise ValueError("The successor checkpoint is not action-prefix conditioned.")
    head_kwargs = {
        "embed_dim": int(config["embed_dim"]),
        "action_dim": int(config["action_dim"]),
        "history_size": int(config["history_size"]),
        "hidden_dim": int(config["hidden_dim"]),
    }
    if method in {
        "rf_successor_sequence_wm",
        "rf_balanced_successor_sequence_wm",
        "rf_ema_balanced_successor_sequence_wm",
        "rf_direct_moment_sequence_wm",
        "rf_e2e_moment_sequence_wm",
    }:
        expected_version = {
            "rf_successor_sequence_wm": 2,
            "rf_balanced_successor_sequence_wm": 3,
            "rf_ema_balanced_successor_sequence_wm": 4,
            "rf_direct_moment_sequence_wm": 5,
            "rf_e2e_moment_sequence_wm": 6,
        }[method]
        if objective_version != expected_version or config.get(
            "architecture"
        ) != "causal_gru_successor_increments":
            raise ValueError(
                "The successor-sequence checkpoint version or architecture differs."
            )
        if method in {
            "rf_balanced_successor_sequence_wm",
            "rf_ema_balanced_successor_sequence_wm",
            "rf_direct_moment_sequence_wm",
            "rf_e2e_moment_sequence_wm",
        } and config.get("feature_group_reduction") != "group_sum":
            raise ValueError("The balanced checkpoint is missing group-sum reduction.")
        if method == "rf_ema_balanced_successor_sequence_wm" and not 0.0 <= float(
            config.get("target_world_ema_decay", -1.0)
        ) < 1.0:
            raise ValueError("The EMA checkpoint has an invalid target decay.")
        head = ActionPrefixMomentHead(gamma=float(config["gamma"]), **head_kwargs)
    else:
        if method != "rf_successor_lewm":
            raise ValueError("Objective version 1 requires RF-Successor-LeWM.")
        head = ActionPrefixSuccessorHead(**head_kwargs)
    head.load_state_dict(payload["successor_state_dict"])
    head.eval()
    head.requires_grad_(False)
    return head, config, payload


def make_rf_successor_policy(
    *,
    world_model: nn.Module,
    successor: ActionPrefixSuccessorHead | ActionPrefixMomentHead,
    planning: dict[str, Any],
    successor_config: dict[str, Any],
    process: dict[str, Any] | None = None,
    transform: dict[str, Any] | None = None,
    device: str | torch.device = "cpu",
):
    """Build the standard CEM/MPC policy without an action-producing head."""

    import stable_worldmodel as swm

    wrapped = RewardFreeSuccessorLeWM(
        world_model,
        successor,
        max_horizon=int(successor_config["max_horizon"]),
        successor_weight=float(successor_config.get("planning_weight", 1.0)),
        terminal_weight=float(successor_config.get("terminal_weight", 0.0)),
        clamp_successor_cost=bool(
            successor_config.get("clamp_successor_cost", True)
        ),
        planning_query=str(
            successor_config.get("planning_query", "discounted_successor")
        ),
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
        history_len=planning.get("history_len", 1),
        action_block=planning["action_block"],
        warm_start=planning["warm_start"],
    )
    return swm.policy.WorldModelPolicy(
        solver=solver,
        config=config,
        process=process,
        transform=transform,
    )


__all__ = [
    "RewardFreeSuccessorLeWM",
    "load_rf_successor_checkpoint",
    "make_rf_successor_policy",
]
