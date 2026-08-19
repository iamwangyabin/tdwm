import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn

from tdwm.adapters.goal_tail import GoalTailLeWM, load_goal_tail_value
from tdwm.methods.goal_tail import (
    GoalTailValue,
    discounted_goal_tail_target,
    goal_cost,
    soft_update,
)
from tdwm.training.lewm import load_training_protocol


ROOT = Path(__file__).resolve().parents[2]


class _FakeWorldModel(nn.Module):
    def __init__(self, predicted: torch.Tensor) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.predictor = type("Predictor", (), {"num_frames": 2})()
        self.predicted = predicted

    def rollout(self, info, action_sequence):
        info["predicted_emb"] = self.predicted.to(action_sequence)
        return info


class _ZeroValue(nn.Module):
    def forward(self, latent, goal):
        return torch.zeros((*latent.shape[:-1], 1), device=latent.device)


class GoalTailTest(unittest.TestCase):
    def test_n_step_target_matches_normalized_discounted_cost(self):
        future = torch.tensor([[[1.0], [2.0]]])
        goal = torch.zeros(1, 1)
        bootstrap = torch.zeros(1)

        target = discounted_goal_tail_target(
            future, goal, bootstrap, gamma=0.5
        )

        self.assertTrue(torch.allclose(target, torch.tensor([1.5])))
        self.assertTrue(
            torch.allclose(
                goal_cost(future, goal[:, None]), torch.tensor([[1.0, 4.0]])
            )
        )

    def test_mpc_adapter_scores_discounted_predicted_path(self):
        predicted = torch.tensor([[[[0.0], [1.0], [2.0], [3.0]]]])
        model = GoalTailLeWM(
            _FakeWorldModel(predicted),
            _ZeroValue(),
            gamma=0.5,
            history_size=2,
        )
        info = {"goal": torch.zeros(1, 1, 1, 1), "goal_emb": torch.zeros(1, 1, 1)}
        actions = torch.zeros(1, 1, 2, 1)

        cost = model.get_cost(info, actions)

        self.assertEqual(tuple(cost.shape), (1, 1))
        self.assertTrue(torch.allclose(cost, torch.tensor([[4.25]])))

    def test_value_head_and_ema_are_trainable(self):
        value = GoalTailValue(embed_dim=3)
        target = GoalTailValue(embed_dim=3)
        before = [parameter.detach().clone() for parameter in target.parameters()]
        soft_update(target, value, tau=1.0)
        for expected, actual in zip(value.parameters(), target.parameters(), strict=True):
            self.assertTrue(torch.equal(expected, actual))
        self.assertFalse(
            all(
                torch.equal(old, new)
                for old, new in zip(before, target.parameters(), strict=True)
            )
        )

        prediction = value(torch.randn(4, 3), torch.randn(4, 3)).mean()
        prediction.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in value.parameters()))

    def test_value_head_export_round_trip(self):
        value = GoalTailValue(embed_dim=2, hidden_dim=4)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "value.pt"
            torch.save(
                {
                    "value_state_dict": value.state_dict(),
                    "value_config": {"embed_dim": 2, "hidden_dim": 4},
                },
                path,
            )
            restored, config = load_goal_tail_value(path)

        self.assertEqual(config["embed_dim"], 2)
        for expected, actual in zip(
            value.parameters(), restored.parameters(), strict=True
        ):
            self.assertTrue(torch.equal(expected, actual))

    def test_gt_lewm_training_protocol_is_valid(self):
        protocol = load_training_protocol(
            ROOT / "configs/experiment/gt_lewm_cube_train.yaml"
        )
        self.assertEqual(protocol["method"], "gt_lewm")
        self.assertEqual(protocol["sequence"]["num_steps"], 11)
        self.assertEqual(protocol["tail_value"]["horizon"], 8)


if __name__ == "__main__":
    unittest.main()
