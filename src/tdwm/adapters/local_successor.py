"""Stable World Model adapter for the standalone LS-LeWM method."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from tdwm.methods.local_successor import (
    LocalSuccessorHeads,
    latent_goal_cost,
    successor_goal_cost,
)


class LocalSuccessorLeWM(nn.Module):
    """Join short-horizon LeWM rollouts with a policy-consistent TD tail."""

    def __init__(
        self,
        world_model: nn.Module,
        heads: LocalSuccessorHeads,
        *,
        gamma: float,
        clamp_tail_cost: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 <= gamma < 1.0:
            raise ValueError("gamma must lie in [0, 1).")
        self.world_model = world_model
        self.heads = heads
        self.gamma = float(gamma)
        self.clamp_tail_cost = bool(clamp_tail_cost)

    @property
    def history_size(self) -> int:
        return self.heads.history_size

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
        """Score explicit candidate prefixes plus their learned policy tails."""

        if "goal" not in info_dict and "goal_emb" not in info_dict:
            raise AssertionError("goal not in info_dict")
        if "pixels" not in info_dict or info_dict["pixels"].ndim < 3:
            raise ValueError("LS-LeWM requires a time axis in info_dict['pixels'].")
        if action_candidates.ndim != 4:
            raise ValueError(
                "action_candidates must have shape (batch, samples, horizon, dim)."
            )
        if action_candidates.shape[-1] != self.heads.action_dim:
            raise ValueError("Candidate action blocks have the wrong dimension.")

        batch, samples, horizon = action_candidates.shape[:3]
        if horizon <= 0:
            raise ValueError("The planning horizon must be positive.")
        observed_frames = int(info_dict["pixels"].shape[2])
        goal = self._goal_for_samples(info_dict, batch=batch, samples=samples)
        rollout_info = self.world_model.rollout(
            info_dict, action_candidates, history_size=self.history_size
        )
        predicted = rollout_info["predicted_emb"]
        if predicted.ndim != 4 or predicted.shape[:2] != (batch, samples):
            raise ValueError(
                "LeWM rollout must return (batch, samples, time, latent_dim)."
            )
        future = predicted[..., observed_frames:, :]
        if future.shape[-2] != horizon:
            raise ValueError(
                "LeWM rollout future length does not match the CEM horizon: "
                f"{future.shape[-2]} != {horizon}."
            )

        stage_goal = goal.unsqueeze(-2)
        stage_cost = latent_goal_cost(future, stage_goal)
        discounts = torch.pow(
            stage_cost.new_tensor(self.gamma),
            torch.arange(horizon, device=stage_cost.device),
        )
        prefix_cost = (1.0 - self.gamma) * (
            stage_cost * discounts
        ).sum(dim=-1)

        terminal_history = self._pad_latent_history(predicted)
        terminal_previous = self._terminal_previous_actions(action_candidates)
        tail_action = self.heads.policy(
            terminal_history, terminal_previous, goal
        )
        successor = self.heads.successor(
            terminal_history, terminal_previous, tail_action, goal
        )
        tail_cost = successor_goal_cost(successor, goal)
        if self.clamp_tail_cost:
            tail_cost = tail_cost.clamp_min(0.0)
        return prefix_cost + (self.gamma**horizon) * tail_cost

    @torch.no_grad()
    def get_action(
        self,
        info: dict[str, Any],
        horizon: int = 1,
        prefix_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Roll out the continuation policy to initialize missing CEM actions."""

        if horizon <= 0:
            raise ValueError("horizon must be positive.")
        latent_history = self._initial_latent_history(info)
        goal = self._goal_without_samples(info, batch=latent_history.shape[0])
        previous_actions = self._initial_previous_actions(
            info,
            batch=latent_history.shape[0],
            device=latent_history.device,
            dtype=latent_history.dtype,
        )

        if prefix_actions is not None:
            prefix = prefix_actions.to(
                device=latent_history.device, dtype=latent_history.dtype
            )
            if prefix.ndim != 3 or prefix.shape[0] != latent_history.shape[0]:
                raise ValueError(
                    "prefix_actions must have shape (batch, time, action_dim)."
                )
            if prefix.shape[-1] != self.heads.action_dim:
                raise ValueError("prefix_actions have the wrong action dimension.")
            for index in range(prefix.shape[1]):
                action = prefix[:, index]
                latent_history, previous_actions = self._advance_context(
                    latent_history, previous_actions, action
                )

        actions: list[torch.Tensor] = []
        for _ in range(horizon):
            action = self.heads.policy(latent_history, previous_actions, goal)
            actions.append(action)
            latent_history, previous_actions = self._advance_context(
                latent_history, previous_actions, action
            )
        return torch.stack(actions, dim=1)

    def _advance_context(
        self,
        latent_history: torch.Tensor,
        previous_actions: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_actions = torch.cat((previous_actions, action.unsqueeze(-2)), dim=-2)
        action_embeddings = self.world_model.action_encoder(raw_actions)
        next_latent = self.world_model.predict(latent_history, action_embeddings)[
            ..., -1, :
        ]
        latent_history = torch.cat(
            (latent_history[..., 1:, :], next_latent.unsqueeze(-2)), dim=-2
        )
        if self.history_size > 1:
            previous_actions = raw_actions[..., 1:, :]
        return latent_history, previous_actions

    def _initial_latent_history(self, info: dict[str, Any]) -> torch.Tensor:
        if "emb" in info:
            embeddings = info["emb"]
        else:
            if "pixels" not in info:
                raise ValueError("pixels are required to initialize the actor.")
            encoded = self.world_model.encode({"pixels": info["pixels"]})
            embeddings = encoded["emb"]
            info["emb"] = embeddings
        if embeddings.ndim != 3:
            raise ValueError("Actor initialization expects (batch, time, latent).")
        return self._pad_latent_history(embeddings)

    def _initial_previous_actions(
        self,
        info: dict[str, Any],
        *,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        required = self.history_size - 1
        if required == 0:
            return torch.empty(
                batch, 0, self.heads.action_dim, device=device, dtype=dtype
            )
        history = info.get("action_history")
        if history is None:
            return torch.zeros(
                batch, required, self.heads.action_dim, device=device, dtype=dtype
            )
        history = history.to(device=device, dtype=dtype)
        if history.ndim != 3 or history.shape[0] != batch:
            raise ValueError("action_history must have shape (batch, time, dim).")
        if history.shape[-1] != self.heads.action_dim:
            raise ValueError("action_history has the wrong action dimension.")
        if history.shape[1] >= required:
            return history[:, -required:]
        padding = torch.zeros(
            batch,
            required - history.shape[1],
            self.heads.action_dim,
            device=device,
            dtype=dtype,
        )
        return torch.cat((padding, history), dim=1)

    def _terminal_previous_actions(
        self, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        required = self.history_size - 1
        leading = action_candidates.shape[:2]
        if required == 0:
            return action_candidates.new_empty(
                *leading, 0, self.heads.action_dim
            )
        available = action_candidates[..., -required:, :]
        if available.shape[-2] == required:
            return available
        padding = action_candidates.new_zeros(
            *leading, required - available.shape[-2], self.heads.action_dim
        )
        return torch.cat((padding, available), dim=-2)

    def _pad_latent_history(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.shape[-1] != self.heads.embed_dim:
            raise ValueError("Unexpected latent dimension.")
        available = latents.shape[-2]
        if available >= self.history_size:
            return latents[..., -self.history_size :, :]
        padding = latents[..., :1, :].expand(
            *latents.shape[:-2], self.history_size - available, latents.shape[-1]
        )
        return torch.cat((padding, latents), dim=-2)

    def _goal_for_samples(
        self, info: dict[str, Any], *, batch: int, samples: int
    ) -> torch.Tensor:
        goal = self._get_or_encode_goal(info)
        if goal.ndim == 2:
            return goal.unsqueeze(1).expand(batch, samples, -1)
        if goal.ndim == 3:
            if goal.shape[1] == samples:
                return goal
            return goal[:, -1].unsqueeze(1).expand(batch, samples, -1)
        if goal.ndim == 4 and goal.shape[1] == samples:
            return goal[..., -1, :]
        raise ValueError("Could not align goal embeddings with CEM samples.")

    def _goal_without_samples(
        self, info: dict[str, Any], *, batch: int
    ) -> torch.Tensor:
        goal = self._get_or_encode_goal(info)
        if goal.ndim == 2 and goal.shape[0] == batch:
            return goal
        if goal.ndim == 3 and goal.shape[0] == batch:
            return goal[:, -1]
        raise ValueError("Could not align goal embeddings with the actor batch.")

    def _get_or_encode_goal(self, info: dict[str, Any]) -> torch.Tensor:
        if "goal_emb" in info:
            return info["goal_emb"]
        if "goal" not in info:
            raise ValueError("goal is required.")
        goal_pixels = info["goal"]
        if goal_pixels.ndim == 6:
            goal_pixels = goal_pixels[:, 0]
        if goal_pixels.ndim != 5:
            raise ValueError("goal pixels must have batch and time axes.")
        encoded = self.world_model.encode({"pixels": goal_pixels})["emb"]
        info["goal_emb"] = encoded
        return encoded


def load_local_successor_heads(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[LocalSuccessorHeads, dict[str, Any]]:
    """Load the policy and successor export written by the LS-LeWM trainer."""

    payload = torch.load(checkpoint_path, map_location=map_location)
    config = dict(payload["heads_config"])
    heads = LocalSuccessorHeads(
        embed_dim=int(config["embed_dim"]),
        action_dim=int(config["action_dim"]),
        history_size=int(config["history_size"]),
        hidden_dim=int(config["hidden_dim"]),
    )
    heads.load_state_dict(payload["heads_state_dict"])
    heads.eval()
    return heads, config


def make_local_successor_policy(
    *,
    world_model: nn.Module,
    heads: LocalSuccessorHeads,
    planning: dict[str, Any],
    successor: dict[str, Any],
    process: dict[str, Any] | None = None,
    transform: dict[str, Any] | None = None,
    device: str | torch.device = "cpu",
):
    """Build Stable World Model's CEM policy around the LS-LeWM adapter."""

    import stable_worldmodel as swm

    wrapped = LocalSuccessorLeWM(
        world_model,
        heads,
        gamma=float(successor["gamma"]),
        clamp_tail_cost=bool(successor.get("clamp_tail_cost", True)),
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
