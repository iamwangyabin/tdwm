from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import yaml

from tdwm.adapters.mc_gt_lewm import MCGoalTailLeWM, load_mc_goal_tail_value
from tdwm.evaluation.mc_gt_lewm import load_mc_gt_evaluation_protocol
from tdwm.methods.goal_tail_value import GoalTailValue


ROOT = Path(__file__).resolve().parents[2]


class RecordingValue(nn.Module):
    history_dim = 10
    goal_dim = 2

    def __init__(self) -> None:
        super().__init__()
        self.parameter = nn.Parameter(torch.zeros(()))
        self.history: torch.Tensor | None = None
        self.goal: torch.Tensor | None = None

    def forward(self, history: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        self.history = history.detach().clone()
        self.goal = goal.detach().clone()
        return history[..., -1]


class FakeLeWM(nn.Module):
    def __init__(self, predicted: torch.Tensor) -> None:
        super().__init__()
        self.parameter = nn.Parameter(torch.zeros(()))
        self.predicted = predicted
        self.rollout_history_size: int | None = None

    def rollout(self, info, action_sequence, history_size=None):
        self.rollout_history_size = history_size
        return {
            **info,
            "predicted_emb": self.predicted.to(action_sequence.device),
        }

    def criterion(self, info):
        terminal = info["predicted_emb"][..., -1, :]
        goal = info["goal_emb"]
        while goal.ndim > 2:
            goal = goal[..., -1, :]
        return (terminal - goal.unsqueeze(1)).pow(2).sum(dim=-1)


def test_mc_goal_tail_cost_preserves_lewm_cost_and_uses_terminal_history():
    predicted = torch.tensor(
        [
            [
                [[9.0, 9.0], [8.0, 8.0], [7.0, 7.0], [1.0, 1.0], [1.0, 0.0], [1.0, 2.0]],
                [[6.0, 6.0], [5.0, 5.0], [4.0, 4.0], [2.0, 2.0], [2.0, 1.0], [3.0, 4.0]],
            ]
        ]
    )
    actions = torch.tensor(
        [
            [
                [[9.0, 9.0], [8.0, 8.0], [7.0, 7.0], [0.1, 0.2], [0.3, 0.5]],
                [[6.0, 6.0], [5.0, 5.0], [4.0, 4.0], [1.1, 1.2], [1.3, 1.5]],
            ]
        ]
    )
    value = RecordingValue()
    world_model = FakeLeWM(predicted)
    adapter = MCGoalTailLeWM(
        world_model,
        value,
        history_size=3,
        action_block_dim=2,
        tail_weight=1.0,
    )
    info = {
        "goal": torch.zeros(1, 1, 1, 2),
        "goal_emb": torch.zeros(1, 1, 2),
    }

    cost = adapter.get_cost(info, actions)

    assert world_model.rollout_history_size == 3
    assert torch.allclose(cost, torch.tensor([[6.0, 28.0]]))
    assert value.history is not None
    assert torch.equal(value.history[..., :6], predicted[..., -3:, :].flatten(-2))
    assert torch.equal(value.history[..., 6:], actions[..., -2:, :].flatten(-2))
    assert value.goal is not None
    assert torch.equal(value.goal, torch.zeros(1, 2, 2))


def test_zero_tail_exactly_recovers_upstream_lewm_cost():
    predicted = torch.randn(2, 3, 6, 4)
    value = GoalTailValue(history_dim=20, goal_dim=4, hidden_dim=8)
    for parameter in value.parameters():
        nn.init.zeros_(parameter)
    world_model = FakeLeWM(predicted)
    adapter = MCGoalTailLeWM(
        world_model,
        value,
        history_size=3,
        action_block_dim=4,
    )
    actions = torch.randn(2, 3, 5, 4)
    info = {
        "goal": torch.zeros(2, 3, 1, 4),
        "goal_emb": torch.randn(2, 1, 4),
    }

    cost = adapter.get_cost(info, actions)
    expected = world_model.criterion(
        {"predicted_emb": predicted, "goal_emb": info["goal_emb"]}
    )

    assert torch.equal(cost, expected)


def test_mc_goal_tail_checkpoint_round_trip(tmp_path):
    value = GoalTailValue(history_dim=10, goal_dim=2, hidden_dim=7)
    checkpoint = tmp_path / "best.pt"
    config = {
        "history_dim": 10,
        "goal_dim": 2,
        "hidden_dim": 7,
        "history_size": 3,
        "action_block_dim": 2,
        "max_goal_offset": 16,
        "gamma": 0.95,
        "objective": "supervised_mc",
    }
    torch.save(
        {
            "objective_version": 1,
            "method": "mc_gt_lewm",
            "value_state_dict": value.state_dict(),
            "value_config": config,
        },
        checkpoint,
    )

    restored, restored_config, payload = load_mc_goal_tail_value(checkpoint)

    assert restored_config == config
    assert payload["method"] == "mc_gt_lewm"
    assert not any(parameter.requires_grad for parameter in restored.parameters())
    for key, parameter in value.state_dict().items():
        assert torch.equal(parameter, restored.state_dict()[key])


def test_mc_gt_planner_protocol_changes_only_the_terminal_cost():
    protocol = load_mc_gt_evaluation_protocol(
        ROOT / "configs/experiment/mc_gt_lewm_cube_seed3072_o25.yaml"
    )
    with (ROOT / "configs/experiment/lewm_cube_seed3072_o25.yaml").open() as stream:
        baseline = yaml.safe_load(stream)

    assert protocol["method"] == "mc_gt_lewm"
    assert protocol["evaluation"] == baseline["evaluation"]
    assert protocol["world"] == baseline["world"]
    assert protocol["planning"]["horizon"] == baseline["planning"]["horizon"]
    for key in (
        "candidates",
        "iterations",
        "elites",
        "initial_variance",
        "action_block",
        "frame_skip",
        "receding_horizon",
        "episode_budget",
        "planning_seed",
        "solver_batch_size",
        "warm_start",
    ):
        assert protocol["planning"][key] == baseline["planning"][key]
    assert protocol["tail_value"]["weight"] == 1.0
    assert protocol["tail_value"]["upstream_scale_factor"] == 192
