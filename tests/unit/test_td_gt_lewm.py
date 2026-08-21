from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from tdwm.methods.goal_tail_value import GoalTailValue
from tdwm.training.td_gt_lewm import (
    CachedCubeTDGoalTailDataset,
    build_td_histories,
    ema_update_target,
    load_td_gt_protocol,
    one_step_td_goal_tail_targets,
)


class FixedTarget(nn.Module):
    def forward(self, history: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return torch.full(history.shape[:-1], 10.0, device=history.device)


def test_td_cached_dataset_loads_current_and_next_action_histories(tmp_path):
    latent_path = tmp_path / "latents.npy"
    latents = np.arange(40, dtype=np.float32).reshape(20, 2)
    actions = np.arange(40, dtype=np.float32).reshape(20, 2)
    np.save(latent_path, latents)
    dataset = CachedCubeTDGoalTailDataset(
        latent_cache_path=latent_path,
        normalized_actions=actions,
        clip_indices=[(0, start) for start in range(4)],
        episode_offsets=[0],
        source_indices=[1],
        frame_skip=2,
        num_steps=4,
        history_size=3,
    )

    sample = dataset[0]

    assert torch.equal(sample["latents"], torch.from_numpy(latents[[1, 3, 5, 7]]))
    assert torch.equal(
        sample["action_blocks"], torch.from_numpy(actions[1:7].reshape(3, 4))
    )


def test_one_step_td_target_terminates_at_goal_and_bootstraps_other_goals():
    latents = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1).expand(2, -1, -1)
    action_blocks = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2).expand(2, -1, -1)
    offsets = torch.tensor([1, 2])

    history, goals, targets, continuation = one_step_td_goal_tail_targets(
        FixedTarget(),
        latents,
        action_blocks,
        offsets,
        history_size=3,
        gamma=0.5,
    )
    expected_history, expected_next = build_td_histories(
        latents, action_blocks, history_size=3
    )

    assert torch.equal(history, expected_history)
    assert torch.equal(goals[:, 0], torch.tensor([3.0, 4.0]))
    assert torch.allclose(targets, torch.tensor([0.0, 5.5]))
    assert torch.equal(continuation, torch.tensor([0.0, 1.0]))
    assert torch.equal(expected_next[:, :3], latents[:, 1:4].flatten(1))


def test_ema_target_uses_only_online_parameters():
    value = GoalTailValue(history_dim=3, goal_dim=1, hidden_dim=2)
    target = GoalTailValue(history_dim=3, goal_dim=1, hidden_dim=2)
    with torch.no_grad():
        for parameter in value.parameters():
            parameter.fill_(10.0)
        for parameter in target.parameters():
            parameter.zero_()

    ema_update_target(target, value, decay=0.9)

    for parameter in target.parameters():
        assert torch.allclose(parameter, torch.ones_like(parameter))


def test_formal_td_gt_protocol_is_a_clean_objective_change():
    protocol = load_td_gt_protocol("configs/experiment/td_gt_lewm_cube_train.yaml")

    assert protocol["method"] == "td_gt_lewm"
    assert protocol["display_name"] == "TD-GT-LeWM"
    assert protocol["base_model"]["frozen"] is True
    assert protocol["tail_value"]["objective"] == "one_step_td"
    assert protocol["tail_value"]["target_network"] is True
    assert protocol["tail_value"]["target_ema_decay"] == 0.995
    assert protocol["initialization"]["mc_gt_warm_start"] is False
    assert protocol["training"]["coverage"] == "all_training_clips_each_epoch"
    assert protocol["planner"]["connected"] is False
