from __future__ import annotations

import torch
from torch import nn

from tdwm.adapters.gt_lewm import GoalTailLeWM, load_goal_tail_value
from tdwm.evaluation.gt_lewm import load_protocol, sample_start_goal_pairs
from tdwm.methods.goal_tail import (
    GoalTailValue,
    discounted_goal_tail_target,
    ema_update,
    future_goal_td_objective,
)
from tdwm.training.gt_lewm import load_gt_training_protocol


class FixedValue(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.parameter = nn.Parameter(torch.zeros(()))
        self.value = value

    def forward(self, latent: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return torch.full(latent.shape[:-1] + (1,), self.value, device=latent.device)


class FakeWorldModel(nn.Module):
    def __init__(self, predicted: torch.Tensor) -> None:
        super().__init__()
        self.parameter = nn.Parameter(torch.zeros(()))
        self.predicted = predicted
        self.predictor = type("Predictor", (), {"num_frames": 1})()

    def rollout(self, info, action_sequence):
        return {"predicted_emb": self.predicted.to(action_sequence.device)}


def test_discounted_goal_tail_target_matches_normalized_return():
    future = torch.tensor([[[1.0, 0.0], [2.0, 0.0]]])
    goal = torch.zeros(1, 2)
    bootstrap = torch.tensor([3.0])

    target = discounted_goal_tail_target(future, goal, bootstrap, gamma=0.5)

    expected = 0.5 * (0.5 + 0.5 * 2.0) + 0.25 * 3.0
    assert torch.allclose(target, torch.tensor([expected]))


def test_goal_tail_adapter_scores_path_and_terminal_value():
    predicted = torch.tensor(
        [[[
            [99.0, 99.0],
            [98.0, 98.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ]]]
    )
    model = FakeWorldModel(predicted)
    adapter = GoalTailLeWM(model, FixedValue(2.0), gamma=0.5, history_size=1)
    info = {
        "pixels": torch.zeros(1, 1, 2, 3, 2, 2),
        "goal": torch.zeros(1, 1, 2),
        "goal_emb": torch.zeros(1, 1, 2),
    }
    actions = torch.zeros(1, 1, 3, 1)

    cost = adapter.get_cost(info, actions)

    expected = 0.5 * (0.5 + 0.5 * 2.0 + 0.25 * 4.5) + 0.125 * 2.0
    assert cost.shape == (1, 1)
    assert torch.allclose(cost, torch.tensor([[expected]]))


def test_goal_tail_value_checkpoint_round_trip(tmp_path):
    value = GoalTailValue(embed_dim=4, hidden_dim=6)
    checkpoint = tmp_path / "value.pt"
    torch.save(
        {
            "value_state_dict": value.state_dict(),
            "value_config": {
                "objective_version": 2,
                "embed_dim": 4,
                "hidden_dim": 6,
                "gamma": 0.95,
                "max_goal_offset": 8,
                "td_horizon": 4,
                "target_ema_decay": 0.995,
                "continuation_policy": "offline_dataset_behavior",
            },
        },
        checkpoint,
    )

    restored, config = load_goal_tail_value(checkpoint)

    assert config["embed_dim"] == 4
    for key, parameter in value.state_dict().items():
        assert torch.equal(parameter, restored.state_dict()[key])


def test_gt_protocols_are_separate_from_baseline_protocols():
    train_protocol = load_gt_training_protocol(
        "configs/experiment/gt_lewm_cube_train.yaml"
    )
    eval_protocol = load_protocol("configs/experiment/gt_lewm_cube_checkpoint_o50.yaml")

    assert train_protocol["method"] == "gt_lewm"
    assert train_protocol["sequence"]["num_steps"] == 11
    assert train_protocol["tail_value"]["goal_source"] == "all_future_states_in_clip"
    assert (
        train_protocol["tail_value"]["continuation_policy"]
        == "offline_dataset_behavior"
    )
    assert (
        train_protocol["loader"]["batch_size"]
        * (
            train_protocol["sequence"]["num_steps"]
            - train_protocol["sequence"]["history_frames"]
        )
        == train_protocol["loss"]["sigreg"]["effective_batch_size"]
    )
    assert (
        train_protocol["training"]["epochs"]
        * train_protocol["training"]["optimizer_steps_per_epoch"]
        == 127_960
    )
    assert eval_protocol["method"] == "gt_lewm"
    assert eval_protocol["tail_value"]["max_goal_offset"] == 8
    assert eval_protocol["evaluation"]["goal_offset"] > (
        eval_protocol["planning"]["horizon"]
        * eval_protocol["planning"]["action_block"]
    )


def test_future_goal_td_objective_terminates_at_goal_and_bootstraps_far_goals():
    latents = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1)
    value = FixedValue(2.0)
    target_value = FixedValue(2.0)

    output = future_goal_td_objective(
        value,
        target_value,
        latents,
        first_current_index=0,
        max_goal_offset=3,
        td_horizon=2,
        gamma=0.5,
    )

    assert output.pair_count == 6
    assert torch.allclose(output.td_loss, torch.tensor((4.0 + 2.25 + 0.5625) / 3))
    assert torch.allclose(output.boundary_loss, torch.tensor(4.0))
    assert torch.allclose(output.prediction_mean, torch.tensor(2.0))
    assert torch.allclose(output.target_mean, torch.tensor(0.625))


def test_future_goal_td_objective_shapes_the_latent_but_not_target_network():
    torch.manual_seed(0)
    latents = torch.randn(2, 5, 3, requires_grad=True)
    value = GoalTailValue(embed_dim=3, hidden_dim=4)
    target_value = GoalTailValue(embed_dim=3, hidden_dim=4)
    target_value.requires_grad_(False)

    output = future_goal_td_objective(
        value,
        target_value,
        latents,
        first_current_index=1,
        max_goal_offset=3,
        td_horizon=2,
        gamma=0.9,
    )
    (output.td_loss + output.boundary_loss).backward()

    assert latents.grad is not None
    assert torch.count_nonzero(latents.grad) > 0
    assert all(parameter.grad is None for parameter in target_value.parameters())


def test_ema_update_uses_decay_for_the_previous_target():
    target = nn.Linear(1, 1, bias=False)
    source = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        target.weight.zero_()
        source.weight.fill_(10.0)

    ema_update(target, source, decay=0.9)

    assert torch.allclose(target.weight, torch.ones_like(target.weight))


def test_start_goal_sampler_is_seeded_and_includes_every_valid_rank():
    first = sample_start_goal_pairs(
        torch.tensor([5, 5]).numpy(), goal_offset=2, episodes=6, seed=42
    )
    second = sample_start_goal_pairs(
        torch.tensor([5, 5]).numpy(), goal_offset=2, episodes=6, seed=42
    )

    assert all(
        torch.equal(torch.as_tensor(left), torch.as_tensor(right))
        for left, right in zip(first, second, strict=True)
    )
    assert torch.equal(torch.as_tensor(first[2]), torch.arange(6))
