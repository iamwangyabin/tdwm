from __future__ import annotations

import torch
from torch import nn

from tdwm.adapters.rf_successor_lewm import (
    RewardFreeSuccessorLeWM,
    load_rf_successor_checkpoint,
)
from tdwm.evaluation.rf_successor_lewm import (
    load_rf_successor_evaluation_protocol,
)
from tdwm.methods.rf_successor_lewm import (
    ActionPrefixSuccessorHead,
    finite_horizon_successor_targets,
    multi_horizon_successor_objective,
    successor_recurrence_residual,
)
from tdwm.methods.successor_geometry import (
    goal_cost_weights,
    latent_goal_cost,
    successor_feature_basis,
)
from tdwm.training.rf_successor_lewm import (
    build_multi_horizon_windows,
    load_rf_successor_training_protocol,
)


class FakeWorldModel(nn.Module):
    def __init__(self, predicted: torch.Tensor) -> None:
        super().__init__()
        self.parameter = nn.Parameter(torch.zeros(()))
        self.predicted = predicted

    def rollout(self, info, action_sequence, history_size=None):
        del info, history_size
        return {"predicted_emb": self.predicted.to(action_sequence)}

    def encode(self, info):
        return {"emb": info["pixels"]}


class FixedSuccessor(nn.Module):
    def __init__(self, latent: torch.Tensor, *, history_size: int, action_dim: int):
        super().__init__()
        self.embed_dim = int(latent.numel())
        self.action_dim = int(action_dim)
        self.history_size = int(history_size)
        self.register_buffer("value", successor_feature_basis(latent))

    def forward(self, history, actions):
        return self.value.to(actions).expand(
            *actions.shape[:-1], self.value.shape[-1]
        )


def test_direct_successor_targets_include_single_and_all_multi_step_values():
    torch.manual_seed(1)
    future = torch.randn(2, 4, 5)
    goal = torch.randn(2, 5)
    gamma = 0.8

    targets = finite_horizon_successor_targets(future, gamma=gamma)
    weights = goal_cost_weights(goal).unsqueeze(1)
    queried_cost = (targets * weights).sum(dim=-1)
    powers = gamma ** torch.arange(4, dtype=future.dtype)
    stage_cost = latent_goal_cost(future, goal.unsqueeze(1))
    expected = (stage_cost * powers).cumsum(dim=1) / powers.cumsum(dim=0)

    assert torch.allclose(targets[:, 0], successor_feature_basis(future[:, 0]))
    assert torch.allclose(queried_cost, expected, atol=1e-6)


def test_successor_target_exactly_satisfies_latent_increment_recurrence():
    torch.manual_seed(2)
    future = torch.randn(3, 5, 7)
    target = finite_horizon_successor_targets(future, gamma=0.95)

    residual = successor_recurrence_residual(target, future, gamma=0.95)

    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-6)


def test_action_prefix_head_is_causal_and_has_no_goal_or_policy_api():
    torch.manual_seed(3)
    head = ActionPrefixSuccessorHead(
        embed_dim=4,
        action_dim=2,
        history_size=3,
        hidden_dim=8,
    ).eval()
    history = torch.randn(2, 3, 4)
    actions = torch.randn(2, 5, 2)
    changed = actions.clone()
    changed[:, 3:] = torch.randn_like(changed[:, 3:])

    original = head(history, actions)
    perturbed = head(history, changed)

    assert original.shape == (2, 5, 6)
    assert torch.allclose(original[:, :3], perturbed[:, :3])
    assert torch.equal(original[..., -1], torch.ones_like(original[..., -1]))
    assert not hasattr(head, "policy")


def test_joint_objective_updates_world_rollout_and_not_target_latents():
    torch.manual_seed(4)
    head = ActionPrefixSuccessorHead(
        embed_dim=3,
        action_dim=2,
        history_size=2,
        hidden_dim=7,
    )
    history = torch.randn(2, 2, 3, requires_grad=True)
    actions = torch.randn(2, 4, 2)
    predicted = torch.randn(2, 4, 3, requires_grad=True)
    target = torch.randn(2, 4, 3, requires_grad=True)

    output = multi_horizon_successor_objective(
        head,
        history,
        actions,
        predicted,
        target,
        gamma=0.9,
    )
    loss = output.latent_loss + output.successor_loss + output.recurrence_loss
    loss.backward()

    assert history.grad is not None and torch.count_nonzero(history.grad) > 0
    assert predicted.grad is not None and torch.count_nonzero(predicted.grad) > 0
    assert target.grad is None
    assert any(parameter.grad is not None for parameter in head.parameters())


def test_multi_horizon_windows_align_history_actions_and_targets():
    latents = torch.arange(9, dtype=torch.float32).reshape(1, 9, 1)
    target = latents + 100.0
    actions = torch.arange(9, dtype=torch.float32).reshape(1, 9, 1)

    windows = build_multi_horizon_windows(
        latents,
        target,
        actions,
        history_size=3,
        horizon=2,
    )

    assert windows.count_per_clip == 5
    assert torch.equal(windows.history[0, :, 0], torch.tensor([0.0, 1.0, 2.0]))
    assert torch.equal(windows.rollout_actions[0, :, 0], torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert torch.equal(windows.action_prefix[0, :, 0], torch.tensor([2.0, 3.0]))
    assert torch.equal(windows.target_future[0, :, 0], torch.tensor([103.0, 104.0]))
    assert torch.equal(windows.history[-1, :, 0], torch.tensor([4.0, 5.0, 6.0]))


def test_planner_queries_supplied_prefix_without_an_actor():
    # One observed latent followed by two future latents.
    predicted = torch.tensor([[[[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]]]])
    successor = FixedSuccessor(
        torch.tensor([2.0, 0.0]), history_size=3, action_dim=1
    )
    adapter = RewardFreeSuccessorLeWM(
        FakeWorldModel(predicted),
        successor,
        max_horizon=2,
        successor_weight=1.0,
        terminal_weight=0.5,
        clamp_successor_cost=False,
    )
    info = {
        "pixels": torch.zeros(1, 1, 1, 2),
        "goal_emb": torch.zeros(1, 1, 2),
    }
    actions = torch.zeros(1, 1, 2, 1)

    cost = adapter.get_cost(info, actions)

    # Successor point [2, 0] has mean squared cost 2; terminal [4, 0] has 8.
    assert torch.allclose(cost, torch.tensor([[6.0]]))
    assert not hasattr(adapter, "get_action")


def test_reward_free_successor_checkpoint_round_trip(tmp_path):
    head = ActionPrefixSuccessorHead(
        embed_dim=4, action_dim=3, history_size=2, hidden_dim=6
    )
    config = {
        "embed_dim": 4,
        "action_dim": 3,
        "history_size": 2,
        "hidden_dim": 6,
        "max_horizon": 5,
        "goal_conditioning": "none",
        "action_conditioning": "causal_prefix",
    }
    checkpoint = tmp_path / "rf_successor.pt"
    torch.save(
        {
            "method": "rf_successor_lewm",
            "objective_version": 1,
            "deployment_checkpoint_version": 1,
            "world_model_state_dict": {},
            "successor_state_dict": head.state_dict(),
            "successor_config": config,
        },
        checkpoint,
    )

    restored, restored_config, payload = load_rf_successor_checkpoint(checkpoint)

    assert restored_config == config
    assert payload["method"] == "rf_successor_lewm"
    for name, value in head.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])


def test_training_protocol_locks_reward_free_multi_horizon_semantics():
    protocol = load_rf_successor_training_protocol(
        "configs/experiment/rf_successor_lewm_cube_train.yaml"
    )

    assert protocol["method"] == "rf_successor_lewm"
    assert protocol["sequence"]["rollout_horizon"] == 5
    assert protocol["successor"]["goal_conditioning"] == "none"
    assert protocol["successor"]["continuation_policy"] == "none"
    assert protocol["successor"]["td_bootstrap"] is False
    assert protocol["joint_objective"]["multi_step_prediction"] == (
        "open_loop_latent_mse_all_horizons"
    )


def test_evaluation_protocol_uses_cem_candidates_without_an_actor():
    protocol = load_rf_successor_evaluation_protocol(
        "configs/experiment/rf_successor_lewm_cube_checkpoint_o50.yaml"
    )

    assert protocol["method"] == "rf_successor_lewm"
    assert protocol["planning"]["horizon"] == protocol["successor"]["max_horizon"]
    assert protocol["planning"]["initial_distribution"] == (
        "cem_gaussian_no_actor"
    )
    assert protocol["inference_objective"]["learned_action_policy"] is False
