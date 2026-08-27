from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from tdwm.training.clean_aligned_lewm import (
    VARIANTS,
    clip_parameter_groups,
    compare_state_dicts,
    freeze_batchnorm_running_stats,
    load_clean_aligned_protocol,
    preserve_torch_rng_state,
    state_dict_sha256,
)


def test_clean_protocol_locks_causal_controls_and_terminal_only_inference():
    protocol = load_clean_aligned_protocol(
        "configs/experiment/clean_aligned_lewm_cube_train.yaml"
    )

    assert tuple(protocol["variants"]) == VARIANTS
    assert protocol["variants"]["r0_common_lewm"]["auxiliary"] == "none"
    assert protocol["variants"]["r1_head_only"]["world_gradient_scale"] == 0.0
    assert protocol["variants"]["r2_anchored_mc"]["world_gradient_scale"] == 0.1
    assert protocol["planner"]["cost"] == "terminal_latent_goal_distance"
    assert protocol["planner"]["load_value_head"] is False
    assert all(
        variant["inference"] == "terminal_only"
        for variant in protocol["variants"].values()
    )
    assert protocol["causal_controls"] == {
        "same_initial_world_state": True,
        "same_world_split": True,
        "same_tail_split": True,
        "same_world_batch_order": True,
        "same_tail_batch_order": True,
        "freeze_auxiliary_batchnorm_running_stats": True,
        "preserve_rng_around_auxiliary": True,
        "separate_world_value_gradient_clipping": True,
        "deterministic_algorithms": True,
    }


def test_batchnorm_context_freezes_buffers_but_keeps_gradients():
    model = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3))
    model.train()
    batchnorm = model[1]
    running_mean = batchnorm.running_mean.clone()
    running_variance = batchnorm.running_var.clone()
    batches = batchnorm.num_batches_tracked.clone()

    with freeze_batchnorm_running_stats(model):
        assert model.training is True
        assert batchnorm.training is False
        model(torch.randn(8, 3)).square().mean().backward()

    assert batchnorm.training is True
    assert torch.equal(batchnorm.running_mean, running_mean)
    assert torch.equal(batchnorm.running_var, running_variance)
    assert torch.equal(batchnorm.num_batches_tracked, batches)
    assert model[0].weight.grad is not None
    assert batchnorm.weight.grad is not None


def test_auxiliary_rng_context_restores_cpu_rng():
    torch.manual_seed(17)
    expected = torch.rand(8)
    torch.manual_seed(17)

    with preserve_torch_rng_state():
        torch.rand(100)

    assert torch.equal(torch.rand(8), expected)


def test_world_and_value_gradients_are_clipped_independently():
    world = nn.Parameter(torch.zeros(2))
    value = nn.Parameter(torch.zeros(2))
    world.grad = torch.tensor([3.0, 4.0])
    value.grad = torch.tensor([3000.0, 4000.0])

    world_norm, value_norm = clip_parameter_groups(
        [world],
        [value],
        world_max_norm=1.0,
        value_max_norm=1.0,
    )

    assert world_norm.item() == pytest.approx(5.0)
    assert value_norm.item() == pytest.approx(5000.0)
    assert torch.allclose(world.grad, torch.tensor([0.6, 0.8]), atol=1e-6)
    assert torch.allclose(value.grad, torch.tensor([0.6, 0.8]), atol=1e-6)


class _ToyWorld(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.batchnorm = nn.BatchNorm1d(4)
        self.dropout = nn.Dropout(0.3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.batchnorm(self.linear(inputs)))


def _run_toy_clean_updates(
    world: _ToyWorld,
    value: nn.Module,
    *,
    head_only: bool,
    steps: int,
) -> None:
    world.train()
    value.train()
    optimizer = torch.optim.AdamW(
        [
            {"params": list(world.parameters()), "lr": 1e-3},
            {"params": list(value.parameters()), "lr": 2e-3},
        ],
        weight_decay=1e-3,
    )
    generator = torch.Generator().manual_seed(991)
    batches = [torch.randn(8, 4, generator=generator) for _ in range(steps)]
    long_batches = [torch.randn(8, 4, generator=generator) for _ in range(steps)]
    torch.manual_seed(31337)
    for common_batch, long_batch in zip(batches, long_batches, strict=True):
        optimizer.zero_grad()
        world(common_batch).square().mean().backward()
        if head_only:
            with preserve_torch_rng_state(), freeze_batchnorm_running_stats(world):
                predicted = world(long_batch)
            value(predicted.detach()).square().mean().backward()
        clip_parameter_groups(
            list(world.parameters()),
            list(value.parameters()),
            world_max_norm=1.0,
            value_max_norm=1.0,
        )
        optimizer.step()


@pytest.mark.parametrize("steps", [1, 100])
def test_r0_and_detached_r1_leave_every_world_parameter_and_buffer_identical(steps):
    torch.manual_seed(7)
    initial_world = _ToyWorld()
    initial_value = nn.Linear(4, 1)
    r0_world = copy.deepcopy(initial_world)
    r1_world = copy.deepcopy(initial_world)
    r0_value = copy.deepcopy(initial_value)
    r1_value = copy.deepcopy(initial_value)

    _run_toy_clean_updates(r0_world, r0_value, head_only=False, steps=steps)
    _run_toy_clean_updates(r1_world, r1_value, head_only=True, steps=steps)

    comparison = compare_state_dicts(r0_world.state_dict(), r1_world.state_dict())
    assert comparison["exact_match"] is True
    assert comparison["mismatch_count"] == 0
    assert state_dict_sha256(r0_world.state_dict()) == state_dict_sha256(
        r1_world.state_dict()
    )
    assert any(
        not torch.equal(r0_value.state_dict()[name], r1_value.state_dict()[name])
        for name in r0_value.state_dict()
    )
