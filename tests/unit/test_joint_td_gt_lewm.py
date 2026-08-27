from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from tdwm.methods.goal_tail_value import GoalTailValue
from tdwm.training.joint_td_gt_lewm import (
    CachedCubeJointTDDataset,
    build_history_at,
    build_joint_td_batch,
    configure_joint_trainability,
    load_joint_td_gt_protocol,
    teacher_forced_windows,
)


class TinyLeWM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(1, 1)
        self.projector = nn.Linear(1, 1)
        self.predictor = nn.Linear(2, 1, bias=False)
        self.action_encoder = nn.Identity()
        self.pred_proj = nn.Identity()

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        return self.predictor(torch.cat((emb, act_emb), dim=-1))

    def rollout(
        self,
        info: dict[str, torch.Tensor],
        action_sequence: torch.Tensor,
        history_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        history_size = 3 if history_size is None else history_size
        history = info["emb"][:, 0]
        actions = action_sequence[:, 0]
        frames = list(history.unbind(1))
        for _ in range(actions.shape[1] - history.shape[1] + 1):
            start = len(frames) - history_size
            latent_window = torch.stack(frames[start:], dim=1)
            action_window = actions[:, start : start + history_size]
            frames.append(self.predict(latent_window, action_window)[:, -1])
        info["predicted_emb"] = torch.stack(frames, dim=1).unsqueeze(1)
        return info


def test_joint_cached_dataset_reads_full_rollout_and_bootstrap_actions(tmp_path):
    latent_path = tmp_path / "latents.npy"
    latents = np.arange(80, dtype=np.float32).reshape(40, 2)
    actions = np.arange(80, dtype=np.float32).reshape(40, 2)
    np.save(latent_path, latents)
    dataset = CachedCubeJointTDDataset(
        latent_cache_path=latent_path,
        normalized_actions=actions,
        clip_indices=[(0, start) for start in range(4)],
        episode_offsets=[0],
        source_indices=[1],
        frame_skip=2,
        num_steps=6,
        action_blocks=4,
    )

    sample = dataset[0]

    assert torch.equal(
        sample["latents"], torch.from_numpy(latents[[1, 3, 5, 7, 9, 11]])
    )
    assert torch.equal(
        sample["action_blocks"], torch.from_numpy(actions[1:9].reshape(4, 4))
    )


def test_teacher_forced_windows_preserve_each_local_lewm_target():
    latents = torch.arange(7, dtype=torch.float32).reshape(1, 7, 1)
    actions = (10 + torch.arange(6, dtype=torch.float32)).reshape(1, 6, 1)

    histories, action_windows, targets = teacher_forced_windows(
        latents, actions, history_size=3, rollout_horizon=4
    )

    assert torch.equal(
        histories[:, :, 0],
        torch.tensor([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]),
    )
    assert torch.equal(
        action_windows[:, :, 0],
        torch.tensor([[10, 11, 12], [11, 12, 13], [12, 13, 14], [13, 14, 15]]),
    )
    assert torch.equal(
        targets[:, :, 0],
        torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]),
    )


def test_joint_tail_loss_backpropagates_into_lewm_dynamics():
    model = TinyLeWM()
    with torch.no_grad():
        model.predictor.weight.copy_(torch.tensor([[0.5, 0.25]]))
    frozen, dynamics = configure_joint_trainability(model)
    value = GoalTailValue(history_dim=5, goal_dim=1, hidden_dim=4)
    target = copy.deepcopy(value).requires_grad_(False)
    latents = torch.arange(9, dtype=torch.float32).reshape(1, 9, 1)
    actions = torch.ones(1, 5, 1)

    batch = build_joint_td_batch(
        model,
        target,
        latents,
        actions,
        torch.tensor([1]),
        history_size=3,
        rollout_horizon=2,
        gamma=0.95,
    )
    prediction = value(batch.predicted_history, batch.goals)
    loss = (prediction - batch.targets).pow(2).mean()
    loss.backward()

    assert all(parameter.grad is None for parameter in frozen)
    assert all(parameter.grad is not None for parameter in dynamics)
    assert model.predictor.weight.grad.abs().sum() > 0


def test_build_history_at_uses_preceding_actions_only():
    latents = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
    actions = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)

    history = build_history_at(latents, actions, current_index=3, history_size=3)

    assert torch.equal(
        history,
        torch.cat(
            (latents[:, 1:4].flatten(1), actions[:, 1:3].flatten(1)), dim=-1
        ),
    )


def test_formal_joint_protocol_connects_tail_gradient_to_rollout():
    protocol = load_joint_td_gt_protocol(
        "configs/experiment/joint_td_gt_lewm_cube_train.yaml"
    )

    assert protocol["method"] == "joint_td_gt_lewm"
    assert protocol["sequence"]["model_rollout_horizon"] == 5
    assert protocol["base_model"]["frozen_components"] == ["encoder", "projector"]
    assert protocol["base_model"]["frozen_running_statistics"] == ["pred_proj"]
    assert protocol["joint_objective"]["tail_input"] == "predicted_terminal_history"
    assert protocol["joint_objective"]["backpropagate_tail_through_rollout"] is True
    assert protocol["initialization"]["standalone_tail_warm_start"] is False
