"""Stable World Model planning adapter for MC-GT-LeWM."""

from __future__ import annotations

import math
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
        collect_planner_diagnostics: bool = False,
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
        self.collect_planner_diagnostics = bool(collect_planner_diagnostics)
        self.last_cost_components: dict[str, torch.Tensor] | None = None

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

        if self.collect_planner_diagnostics:
            latent_dim = predicted.shape[-1]
            normalized_terminal = original_cost / latent_dim
            normalized_total = normalized_terminal + self.tail_weight * tail_value
            components = {
                "terminal_cost": normalized_terminal.detach(),
                "tail_value": tail_value.detach(),
                "total_cost": normalized_total.detach(),
            }
            if hasattr(self.value, "current_latent"):
                current = self.value.current_latent(history)
                components["boundary_value"] = self.value(history, current).detach()
            self.last_cost_components = components

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


class GoalTailPlannerDiagnosticsRecorder:
    """Record compact component and ranking statistics from public CEM callbacks."""

    output_key = "tdwm_goal_tail_planner_diagnostics"

    def __init__(
        self,
        model: MCGoalTailLeWM,
        *,
        record_iterations: list[int] | tuple[int, ...],
        epsilon: float = 1e-8,
    ) -> None:
        if not record_iterations:
            raise ValueError("record_iterations must not be empty.")
        if epsilon <= 0.0:
            raise ValueError("diagnostic epsilon must be positive.")
        self.model = model
        self.record_iterations = frozenset(int(step) for step in record_iterations)
        self.epsilon = float(epsilon)
        self.records: list[dict[str, float | int]] = []
        self.history: list[list[dict[str, float | int]]] = []
        self._current: list[dict[str, float | int]] = []
        self.solve_index = -1
        self.batch_index = -1

    def reset(self) -> None:
        self.solve_index += 1
        self.batch_index = -1
        self.history = []
        self._current = []

    def start_batch(self) -> None:
        if self._current:
            self.history.append(self._current)
        self.batch_index += 1
        self._current = []

    def end_solve(self) -> None:
        if self._current:
            self.history.append(self._current)
            self._current = []

    def __call__(self, **state: Any) -> None:
        step = int(state["step"])
        if step not in self.record_iterations:
            return
        components = self.model.last_cost_components
        if components is None:
            raise RuntimeError("Goal-tail planner diagnostics have no cost components.")
        record = self._build_record(
            step=step,
            terminal=components["terminal_cost"],
            tail=components["tail_value"],
            total=components["total_cost"],
            topk_indices=state["topk_inds"],
            boundary=components.get("boundary_value"),
        )
        self.records.append(record)
        self._current.append(record)

    def _build_record(
        self,
        *,
        step: int,
        terminal: torch.Tensor,
        tail: torch.Tensor,
        total: torch.Tensor,
        topk_indices: torch.Tensor,
        boundary: torch.Tensor | None,
    ) -> dict[str, float | int]:
        if terminal.shape != tail.shape or terminal.shape != total.shape:
            raise RuntimeError("Planner diagnostic component shapes differ.")
        if terminal.ndim != 2:
            raise RuntimeError("Planner diagnostic costs must be batch by candidate.")
        elite_mask = torch.zeros_like(total, dtype=torch.bool)
        elite_mask.scatter_(1, topk_indices, True)
        nonelite_mask = ~elite_mask
        best_total = total.argmin(dim=1, keepdim=True)
        selected_tail = tail.gather(1, best_total)
        selected_terminal = terminal.gather(1, best_total)
        selected_total = total.gather(1, best_total)
        tail_percentile = (tail <= selected_tail).float().mean(dim=1) * 100.0
        ratio = tail / terminal.abs().clamp_min(self.epsilon)
        k = int(topk_indices.shape[1])
        terminal_topk = terminal.topk(k, dim=1, largest=False).indices
        topk_overlap = (
            terminal_topk.unsqueeze(-1) == topk_indices.unsqueeze(-2)
        ).any(dim=-1).float().mean()
        record: dict[str, float | int] = {
            "solve_index": self.solve_index,
            "batch_index": self.batch_index,
            "iteration": step,
            "candidate_count": int(total.shape[1]),
            "terminal_cost_mean": self._mean(terminal),
            "terminal_cost_std": self._std(terminal),
            "tail_value_mean": self._mean(tail),
            "tail_value_std": self._std(tail),
            "total_cost_mean": self._mean(total),
            "candidate_cost_std": self._std(total),
            "tail_to_terminal_ratio_mean": self._mean(ratio),
            "elite_tail_mean": self._mean(tail[elite_mask]),
            "nonelite_tail_mean": self._mean(tail[nonelite_mask]),
            "terminal_total_rank_correlation": self._mean_rank_correlation(
                terminal, total
            ),
            "tail_total_rank_correlation": self._mean_rank_correlation(tail, total),
            "terminal_total_topk_overlap": float(topk_overlap.item()),
            "best_candidate_changed_fraction": float(
                (terminal.argmin(dim=1) != total.argmin(dim=1)).float().mean().item()
            ),
            "total_best_candidate_tail_percentile": self._mean(tail_percentile),
            "selected_terminal_cost": self._mean(selected_terminal),
            "selected_tail_value": self._mean(selected_tail),
            "selected_total_cost": self._mean(selected_total),
        }
        if boundary is not None:
            record["boundary_value_abs_mean"] = self._mean(boundary.abs())
            record["boundary_value_abs_max"] = float(boundary.abs().max().item())
        return record

    @staticmethod
    def _mean(value: torch.Tensor) -> float:
        return float(value.float().mean().item())

    @staticmethod
    def _std(value: torch.Tensor) -> float:
        return float(value.float().std(unbiased=False).item())

    @staticmethod
    def _mean_rank_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
        left_rank = left.argsort(dim=1).argsort(dim=1).float()
        right_rank = right.argsort(dim=1).argsort(dim=1).float()
        left_rank -= left_rank.mean(dim=1, keepdim=True)
        right_rank -= right_rank.mean(dim=1, keepdim=True)
        numerator = (left_rank * right_rank).sum(dim=1)
        denominator = (
            left_rank.square().sum(dim=1) * right_rank.square().sum(dim=1)
        ).sqrt()
        correlation = numerator / denominator.clamp_min(torch.finfo(torch.float32).eps)
        return float(correlation.mean().item())

    def export(self) -> dict[str, Any]:
        numeric_keys = sorted(
            {
                key
                for record in self.records
                for key, value in record.items()
                if isinstance(value, (int, float))
                and key not in {"solve_index", "batch_index", "iteration"}
            }
        )
        aggregates: dict[str, dict[str, float]] = {}
        for key in numeric_keys:
            values = [
                float(record[key])
                for record in self.records
                if key in record and math.isfinite(float(record[key]))
            ]
            if values:
                aggregates[key] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                }
        return {
            "schema_version": 1,
            "record_iterations": sorted(self.record_iterations),
            "solve_count": self.solve_index + 1,
            "record_count": len(self.records),
            "aggregates": aggregates,
            "records": self.records,
        }


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
    planner_diagnostics: dict[str, Any] | None = None,
):
    """Build the unchanged upstream CEM policy around MC-GT-LeWM."""

    import stable_worldmodel as swm

    wrapped = MCGoalTailLeWM(
        world_model,
        value,
        history_size=history_size,
        action_block_dim=action_block_dim,
        tail_weight=tail_weight,
        collect_planner_diagnostics=planner_diagnostics is not None,
    ).to(device)
    wrapped.eval()
    wrapped.requires_grad_(False)
    recorder = None
    callbacks = None
    if planner_diagnostics is not None:
        recorder = GoalTailPlannerDiagnosticsRecorder(
            wrapped,
            record_iterations=planner_diagnostics["record_iterations"],
            epsilon=float(planner_diagnostics.get("epsilon", 1e-8)),
        )
        callbacks = [recorder]
    solver = swm.solver.CEMSolver(
        model=wrapped,
        batch_size=planning["solver_batch_size"],
        num_samples=planning["candidates"],
        var_scale=planning["initial_variance"],
        n_steps=planning["iterations"],
        topk=planning["elites"],
        device=device,
        seed=planning["planning_seed"],
        callbacks=callbacks,
    )
    config = swm.PlanConfig(
        horizon=planning["horizon"],
        receding_horizon=planning["receding_horizon"],
        history_len=planning["plan_config_history_len"],
        action_block=planning["action_block"],
        warm_start=planning["warm_start"],
    )
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=config,
        process=process,
        transform=transform,
    )
    if recorder is not None:
        policy.tdwm_planner_diagnostics = recorder
    return policy
