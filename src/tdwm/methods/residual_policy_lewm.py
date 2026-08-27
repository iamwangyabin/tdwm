"""Residual LeWM dynamics with a training-only expert-action auxiliary."""

from __future__ import annotations

from dataclasses import dataclass

import stable_worldmodel as swm
import torch
from torch import nn


POLICY_AUXILIARY_METHOD = "policy_auxiliary_successor_geometry_lewm"
RESIDUAL_POLICY_METHOD = "residual_policy_successor_geometry_lewm"
POLICY_AUXILIARY_METHODS = frozenset(
    (POLICY_AUXILIARY_METHOD, RESIDUAL_POLICY_METHOD)
)


class ResidualLeWM(swm.wm.LeWM):
    """Parameterize each public LeWM prediction as the current latent plus a delta."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        output_layers = [
            module
            for module in self.pred_proj.modules()
            if isinstance(module, nn.Linear)
        ]
        if not output_layers:
            raise ValueError("ResidualLeWM requires a linear pred_proj output layer.")
        nn.init.zeros_(output_layers[-1].weight)
        if output_layers[-1].bias is not None:
            nn.init.zeros_(output_layers[-1].bias)

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        delta = super().predict(emb, act_emb)
        if delta.shape != emb.shape:
            raise RuntimeError("Residual LeWM delta must match the input latent shape.")
        return emb + delta


@dataclass(frozen=True)
class ExpertActionWindows:
    """Causal windows for predicting the demonstrated action at the latest latent."""

    history: torch.Tensor
    past_actions: torch.Tensor
    target_actions: torch.Tensor
    target_is_finite: torch.Tensor
    count_per_clip: int


def build_expert_action_windows(
    latents: torch.Tensor,
    actions: torch.Tensor,
    *,
    history_size: int,
) -> ExpertActionWindows:
    """Align ``z[t-H+1:t]`` and past actions with demonstrated action ``a[t]``."""

    if latents.ndim != 3:
        raise ValueError("latents must have shape (batch, time, dim).")
    if actions.ndim != 3 or actions.shape[:2] != latents.shape[:2]:
        raise ValueError("actions must share the latent batch and time axes.")
    if history_size < 2:
        raise ValueError("Expert-action supervision requires history_size >= 2.")
    batch, time = latents.shape[:2]
    count = time - history_size
    if count <= 0:
        raise ValueError("The clip contains no complete expert-action window.")

    history = torch.cat(
        [latents[:, start : start + history_size] for start in range(count)],
        dim=0,
    )
    past_actions = torch.cat(
        [
            actions[:, start : start + history_size - 1]
            for start in range(count)
        ],
        dim=0,
    )
    targets = torch.cat(
        [actions[:, start + history_size - 1] for start in range(count)],
        dim=0,
    )
    target_is_finite = torch.isfinite(targets).all(dim=-1)
    return ExpertActionWindows(
        history=history,
        past_actions=torch.nan_to_num(past_actions, 0.0),
        target_actions=torch.nan_to_num(targets, 0.0),
        target_is_finite=target_is_finite,
        count_per_clip=count,
    )


__all__ = [
    "ExpertActionWindows",
    "POLICY_AUXILIARY_METHOD",
    "POLICY_AUXILIARY_METHODS",
    "RESIDUAL_POLICY_METHOD",
    "ResidualLeWM",
    "build_expert_action_windows",
]
