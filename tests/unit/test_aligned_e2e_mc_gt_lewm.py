from __future__ import annotations

import torch
from torch import nn

from tdwm.methods.goal_tail_value import BoundaryAnchoredGoalTailValue
from tdwm.training.aligned_e2e_mc_gt_lewm import (
    _scale_gradient,
    ema_update_module,
    load_aligned_e2e_mc_gt_protocol,
)


def test_aligned_protocol_uses_independent_world_and_long_tail_views():
    protocol = load_aligned_e2e_mc_gt_protocol(
        "configs/experiment/aligned_e2e_mc_gt_lewm_cube_train.yaml"
    )

    assert protocol["sequence"]["world_num_steps"] == 4
    assert protocol["sequence"]["tail_num_steps"] == 24
    assert protocol["loader"]["world_batch_size"] == 128
    assert protocol["loader"]["world_minimum_unique_episodes_per_batch"] == 128
    assert protocol["loader"]["tail_batch_size"] == 16
    assert protocol["tail_value"]["boundary_condition"] == "exact_current_goal_zero"
    assert protocol["tail_value"]["objective"] == "supervised_mc"
    assert protocol["planner"]["evaluation_goal_offset"] > (
        protocol["planner"]["horizon"] * protocol["sequence"]["frame_skip"]
    )
    assert (
        protocol["training"]["epochs"]
        * protocol["training"]["optimizer_steps_per_epoch"]
        == 127_960
    )


def test_aligned_value_keeps_the_original_head_parameter_budget():
    value = BoundaryAnchoredGoalTailValue(
        history_dim=626,
        goal_dim=192,
        history_size=3,
        hidden_dim=512,
    )

    assert sum(parameter.numel() for parameter in value.parameters()) == 682_497


def test_gradient_scaling_changes_backward_without_changing_forward():
    source = torch.tensor([2.0], requires_grad=True)

    scaled = _scale_gradient(source, 0.1)
    scaled.backward()

    assert torch.equal(scaled.detach(), source.detach())
    assert torch.allclose(source.grad, torch.tensor([0.1]))


def test_ema_world_update_tracks_parameters_and_batchnorm_buffers():
    source = nn.BatchNorm1d(2)
    target = nn.BatchNorm1d(2)
    with torch.no_grad():
        source.weight.fill_(2.0)
        source.running_mean.fill_(4.0)
        source.num_batches_tracked.fill_(7)
        target.weight.zero_()
        target.running_mean.zero_()

    ema_update_module(target, source, decay=0.75)

    assert torch.allclose(target.weight, torch.full_like(target.weight, 0.5))
    assert torch.allclose(target.running_mean, torch.ones_like(target.running_mean))
    assert target.num_batches_tracked.item() == 7
