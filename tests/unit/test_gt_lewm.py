from __future__ import annotations

import torch
from torch import nn

from tdwm.adapters.gt_lewm import GoalTailLeWM, load_goal_tail_value
from tdwm.evaluation.gt_lewm import load_protocol, sample_start_goal_pairs
from tdwm.methods.goal_tail import (
    GoalTailValue,
    discounted_goal_tail_target,
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
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ]]]
    )
    model = FakeWorldModel(predicted)
    adapter = GoalTailLeWM(model, FixedValue(2.0), gamma=0.5, history_size=1)
    info = {"goal": torch.zeros(1, 1, 2), "goal_emb": torch.zeros(1, 1, 2)}
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
                "embed_dim": 4,
                "hidden_dim": 6,
                "gamma": 0.95,
                "horizon": 8,
                "target_tau": 0.99,
            },
        },
        checkpoint,
    )

    restored, config = load_goal_tail_value(checkpoint)

    assert config["embed_dim"] == 4
    for key, parameter in value.state_dict().items():
        assert torch.equal(parameter, restored.state_dict()[key])


def test_gt_protocols_are_separate_from_baseline_protocols():
    train_protocol = load_gt_training_protocol("configs/experiment/gt_lewm_cube_train.yaml")
    eval_protocol = load_protocol("configs/experiment/gt_lewm_cube_checkpoint_o25.yaml")

    assert train_protocol["method"] == "gt_lewm"
    assert train_protocol["sequence"]["num_steps"] == 11
    assert eval_protocol["method"] == "gt_lewm"
    assert eval_protocol["tail_value"]["horizon"] == 8


def test_start_goal_sampler_is_seeded_and_excludes_final_rank():
    first = sample_start_goal_pairs(
        torch.tensor([5, 5]).numpy(), goal_offset=2, episodes=2, seed=42
    )
    second = sample_start_goal_pairs(
        torch.tensor([5, 5]).numpy(), goal_offset=2, episodes=2, seed=42
    )

    assert all(torch.equal(torch.as_tensor(left), torch.as_tensor(right)) for left, right in zip(first, second, strict=True))
    assert int(first[2].max()) < 6
