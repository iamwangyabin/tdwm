from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from tdwm.methods.goal_tail_value import (
    GoalTailValue,
    build_goal_tail_history,
    latent_goal_cost,
    monte_carlo_goal_tail_targets,
    select_goal_tail_samples,
)
from tdwm.training.train_goal_tail_cube import (
    build_value_optimizer,
    freeze_world_model,
    load_goal_tail_protocol,
    spearman_correlation,
    split_clip_indices_by_episode,
)


def test_latent_goal_cost_averages_the_latent_dimension():
    latent = torch.tensor([[1.0, 3.0]])
    goal = torch.tensor([[0.0, 1.0]])

    assert torch.equal(latent_goal_cost(latent, goal), torch.tensor([2.5]))


def test_mc_targets_use_each_hindsight_goal_without_bootstrap():
    latents = torch.tensor([[[0.0], [1.0], [2.0], [4.0]]])

    target = monte_carlo_goal_tail_targets(
        latents,
        current_index=0,
        max_goal_offset=3,
        gamma=0.5,
    )

    expected = torch.tensor(
        [[
            0.0,
            0.5 * (1.0 + 0.5 * 0.0),
            0.5 * (9.0 + 0.5 * 4.0 + 0.25 * 0.0),
        ]]
    )
    assert torch.allclose(target, expected)


def test_history_contains_three_latents_and_two_preceding_actions():
    latents = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    actions = torch.tensor([[[10.0, 11.0], [20.0, 21.0], [30.0, 31.0], [40.0, 41.0]]])

    history = build_goal_tail_history(
        latents,
        actions,
        current_index=2,
        history_size=3,
    )

    assert torch.equal(
        history,
        torch.tensor([[1.0, 2.0, 3.0, 10.0, 11.0, 20.0, 21.0]]),
    )


def test_sample_selection_matches_requested_future_offsets():
    latents = torch.tensor(
        [
            [[0.0], [1.0], [2.0], [3.0], [4.0]],
            [[5.0], [6.0], [7.0], [8.0], [9.0]],
        ]
    )
    actions = torch.zeros(2, 5, 1)

    _, goals, targets = select_goal_tail_samples(
        latents,
        actions,
        torch.tensor([1, 2]),
        current_index=2,
        history_size=3,
        max_goal_offset=2,
        gamma=0.5,
    )

    assert torch.equal(goals, torch.tensor([[3.0], [9.0]]))
    assert torch.allclose(targets, torch.tensor([0.0, 0.5]))


def test_goal_tail_value_is_the_requested_two_hidden_layer_mlp():
    value = GoalTailValue(history_dim=7, goal_dim=3, hidden_dim=5)

    prediction = value(torch.zeros(4, 7), torch.zeros(4, 3))

    assert prediction.shape == (4,)
    assert sum(isinstance(module, nn.Linear) for module in value.net) == 3
    assert sum(isinstance(module, nn.SiLU) for module in value.net) == 2


def test_optimizer_contains_only_value_parameters():
    value = GoalTailValue(history_dim=7, goal_dim=3, hidden_dim=5)
    unrelated = nn.Linear(3, 3)
    optimizer = build_value_optimizer(
        value,
        {"type": "AdamW", "learning_rate": 3e-4, "weight_decay": 1e-4},
    )

    optimized = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimized == {id(parameter) for parameter in value.parameters()}
    assert optimized.isdisjoint({id(parameter) for parameter in unrelated.parameters()})


def test_freeze_world_model_disables_gradients_and_training_mode():
    model = nn.Sequential(nn.Linear(2, 2), nn.Dropout())
    model.train()

    frozen = freeze_world_model(model)

    assert frozen is model
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_episode_split_has_no_episode_overlap_and_is_seeded():
    clips = [(episode, start) for episode in range(4) for start in range(3)]

    first = split_clip_indices_by_episode(
        clips, episode_count=4, train_fraction=0.5, seed=42
    )
    second = split_clip_indices_by_episode(
        clips, episode_count=4, train_fraction=0.5, seed=42
    )

    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    assert set(first[2]).isdisjoint(set(first[3]))
    assert {clips[index][0] for index in first[0]} == set(first[2])
    assert {clips[index][0] for index in first[1]} == set(first[3])


def test_spearman_correlation_tracks_cost_ordering_with_ties():
    target = np.array([0.0, 1.0, 1.0, 3.0])

    assert spearman_correlation(target, target) == pytest.approx(1.0)
    assert spearman_correlation(-target, target) == pytest.approx(-1.0)


def test_v0_1_protocol_has_no_td_or_planner_training():
    protocol = load_goal_tail_protocol(
        "configs/experiment/goal_tail_value_cube_train.yaml"
    )

    assert protocol["base_model"]["frozen"] is True
    assert protocol["sequence"]["history_frames"] == 3
    assert protocol["sequence"]["max_goal_offset"] == 16
    assert protocol["tail_value"]["objective"] == "supervised_mc"
    assert protocol["planner"]["connected"] is False
    assert "td_horizon" not in protocol["tail_value"]
    assert "target_ema_decay" not in protocol["tail_value"]
