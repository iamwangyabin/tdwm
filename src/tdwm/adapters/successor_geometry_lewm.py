"""Stable World Model planning adapter for directed successor geometry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from tdwm.methods.successor_geometry_lewm import (
    DirectedSuccessorGeometry,
    successor_geometry_cost,
)
from tdwm.methods.residual_policy_lewm import (
    POLICY_AUXILIARY_METHOD,
    POLICY_AUXILIARY_METHODS,
    RESIDUAL_POLICY_METHOD,
)


METHOD = "successor_geometry_lewm"
SUPPORTED_METHODS = frozenset((METHOD, *POLICY_AUXILIARY_METHODS))


class SuccessorGeometryLeWM(nn.Module):
    """Use LeWM rollouts and a learned directed goal geometry as CEM cost."""

    def __init__(
        self,
        world_model: nn.Module,
        geometry: DirectedSuccessorGeometry,
        *,
        history_size: int,
        max_horizon: int,
    ) -> None:
        super().__init__()
        if min(history_size, max_horizon) <= 0:
            raise ValueError("history_size and max_horizon must be positive.")
        self.world_model = world_model
        self.geometry = geometry
        self.history_size = int(history_size)
        self.max_horizon = int(max_horizon)

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
        if "goal" not in info_dict and "goal_emb" not in info_dict:
            raise AssertionError("goal not in info_dict")
        if "pixels" not in info_dict or info_dict["pixels"].ndim < 3:
            raise ValueError("Successor geometry requires an observation time axis.")
        if action_candidates.ndim != 4:
            raise ValueError(
                "action_candidates must have shape (batch, samples, horizon, dim)."
            )
        batch, samples, horizon = action_candidates.shape[:3]
        if not 0 < horizon <= self.max_horizon:
            raise ValueError("Planning horizon exceeds geometry training coverage.")

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
        if predicted.shape[-1] != self.geometry.embed_dim:
            raise ValueError("LeWM and successor geometry dimensions differ.")
        observed_frames = int(info_dict["pixels"].shape[2])
        future = predicted[..., observed_frames:, :]
        if future.shape[-2] != horizon:
            raise ValueError("LeWM rollout future does not match the plan horizon.")
        terminal = future[..., -1, :]
        goal = self._goal_for_samples(info_dict, batch=batch, samples=samples)
        return successor_geometry_cost(self.geometry, terminal, goal)

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
        goal_info.pop("act_emb", None)
        encoded = self.world_model.encode(goal_info)["emb"]
        info["goal_emb"] = encoded
        return encoded


def load_successor_geometry_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    expected_method: str | None = None,
) -> tuple[DirectedSuccessorGeometry, dict[str, Any], dict[str, Any]]:
    """Load the paired deployment checkpoint produced by the joint trainer."""

    payload = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    method = payload.get("method")
    if method not in SUPPORTED_METHODS:
        raise ValueError("The checkpoint is not a successor-geometry LeWM model.")
    if expected_method is not None and method != expected_method:
        raise ValueError("The checkpoint method differs from the evaluation protocol.")
    if payload.get("objective_version") != 1:
        raise ValueError("Unsupported successor-geometry objective version.")
    if payload.get("deployment_checkpoint_version") != 1:
        raise ValueError("Unsupported successor-geometry checkpoint version.")
    if "world_model_state_dict" not in payload:
        raise ValueError("The deployment checkpoint is missing the joint LeWM.")
    config = dict(payload.get("geometry_config", {}))
    expected = {
        "architecture": "dual_mlp_directed_cosine",
        "goal_conditioning": "future_pairs_only",
        "reward": "none",
        "policy": (
            "expert_action_auxiliary_training_only"
            if method in POLICY_AUXILIARY_METHODS
            else "none"
        ),
        "td_bootstrap": False,
        "query_sources": ["real_terminal", "predicted_terminal"],
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Invalid successor-geometry checkpoint field: {key}.")
    geometry = DirectedSuccessorGeometry(
        embed_dim=int(config["embed_dim"]),
        projection_dim=int(config["projection_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        temperature=float(config["temperature"]),
    )
    geometry.load_state_dict(payload["geometry_state_dict"])
    geometry.eval()
    geometry.requires_grad_(False)
    return geometry, config, payload


def make_successor_geometry_policy(
    *,
    world_model: nn.Module,
    geometry: DirectedSuccessorGeometry,
    planning: dict[str, Any],
    geometry_config: dict[str, Any],
    process: dict[str, Any] | None = None,
    transform: dict[str, Any] | None = None,
    device: str | torch.device = "cpu",
):
    """Build the standard CEM/MPC policy with no learned action policy."""

    import stable_worldmodel as swm

    wrapped = SuccessorGeometryLeWM(
        world_model,
        geometry,
        history_size=int(geometry_config["history_size"]),
        max_horizon=int(geometry_config["rollout_horizon"]),
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
    "METHOD",
    "POLICY_AUXILIARY_METHOD",
    "POLICY_AUXILIARY_METHODS",
    "RESIDUAL_POLICY_METHOD",
    "SUPPORTED_METHODS",
    "SuccessorGeometryLeWM",
    "load_successor_geometry_checkpoint",
    "make_successor_geometry_policy",
]
