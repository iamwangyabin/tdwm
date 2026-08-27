from __future__ import annotations

import copy

import torch
from torch import nn

from tdwm.methods.goal_tail_value import GoalTailValue
from tdwm.training.e2e_joint_td_gt_lewm import load_e2e_joint_td_gt_protocol
from tdwm.training.joint_td_gt_lewm import build_joint_td_batch


class TinyLeWM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = nn.Linear(2, 1, bias=False)
        self.action_encoder = nn.Identity()

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        return self.predictor(torch.cat((emb, act_emb), dim=-1))

    def rollout(
        self,
        info: dict[str, torch.Tensor],
        action_sequence: torch.Tensor,
        history_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        history_size = 3 if history_size is None else history_size
        frames = list(info["emb"][:, 0].unbind(1))
        actions = action_sequence[:, 0]
        for _ in range(actions.shape[1] - history_size + 1):
            start = len(frames) - history_size
            latent_window = torch.stack(frames[start:], dim=1)
            action_window = actions[:, start : start + history_size]
            frames.append(self.predict(latent_window, action_window)[:, -1])
        info["predicted_emb"] = torch.stack(frames, dim=1).unsqueeze(1)
        return info


def test_formal_e2e_protocol_is_raw_random_and_full_budget():
    protocol = load_e2e_joint_td_gt_protocol(
        "configs/experiment/e2e_joint_td_gt_lewm_cube_train.yaml"
    )

    assert protocol["initialization"] == "random_from_scratch"
    assert set(protocol["dataset"]["keys_to_load"]) >= {"pixels", "action"}
    assert "latent_cache" not in protocol
    assert protocol["sequence"]["num_steps"] == 24
    assert protocol["sequence"]["model_rollout_horizon"] == 5
    assert (
        protocol["training"]["epochs"]
        * protocol["training"]["optimizer_steps_per_epoch"]
        == 127_960
    )
    assert (
        protocol["loader"]["batch_size"]
        * protocol["sequence"]["prediction_windows"]
        == protocol["loss"]["sigreg"]["effective_batch_size"]
    )


def test_terminal_td_loss_updates_rollout_and_raw_representation_latents():
    torch.manual_seed(0)
    model = TinyLeWM()
    value = GoalTailValue(history_dim=5, goal_dim=1, hidden_dim=8)
    target_value = copy.deepcopy(value).requires_grad_(False)
    latents = torch.randn(2, 9, 1, requires_grad=True)
    actions = torch.randn(2, 5, 1)

    td_batch = build_joint_td_batch(
        model,
        target_value,
        latents,
        actions,
        torch.tensor([1, 1]),
        history_size=3,
        rollout_horizon=2,
        gamma=0.95,
    )
    prediction = value(td_batch.predicted_history, td_batch.goals)
    (prediction - td_batch.targets).pow(2).mean().backward()

    assert model.predictor.weight.grad is not None
    assert model.predictor.weight.grad.abs().sum() > 0
    assert latents.grad is not None
    assert latents.grad[:, :3].abs().sum() > 0
    assert all(parameter.grad is None for parameter in target_value.parameters())
