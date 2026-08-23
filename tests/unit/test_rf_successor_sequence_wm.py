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
    ActionPrefixMomentHead,
    finite_horizon_successor_targets,
    latent_sequence_from_successor,
    successor_moments_from_sequence,
    successor_sequence_objective,
)
from tdwm.methods.successor_geometry import successor_feature_basis
from tdwm.training.rf_successor_lewm import (
    load_rf_successor_training_protocol,
)


class EncodeOnlyWorldModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.parameter = nn.Parameter(torch.zeros(()))
        self.rollout_calls = 0

    def encode(self, info):
        return {"emb": info["pixels"] + self.parameter}

    def rollout(self, info, action_sequence, history_size=None):
        del info, action_sequence, history_size
        self.rollout_calls += 1
        raise AssertionError("Successor-only planning must not invoke LeWM rollout.")


def test_all_horizon_successors_exactly_recover_future_latents():
    torch.manual_seed(10)
    future = torch.randn(3, 5, 7)

    successor = finite_horizon_successor_targets(future, gamma=0.95)
    recovered_moments = successor_moments_from_sequence(successor, gamma=0.95)
    recovered_latents = latent_sequence_from_successor(successor, gamma=0.95)

    assert torch.allclose(
        recovered_moments, successor_feature_basis(future), atol=1e-6
    )
    assert torch.allclose(recovered_latents, future, atol=1e-6)


def test_moment_head_enforces_successor_recurrence_by_construction():
    torch.manual_seed(11)
    head = ActionPrefixMomentHead(
        embed_dim=4,
        action_dim=2,
        history_size=3,
        hidden_dim=8,
        gamma=0.9,
    )
    history = torch.randn(2, 3, 4)
    actions = torch.randn(2, 5, 2)

    moments = head.predict_moments(history, actions)
    successor = head(history, actions)
    recovered = successor_moments_from_sequence(successor, gamma=0.9)

    assert successor.shape == (2, 5, 6)
    assert torch.allclose(recovered, moments, atol=1e-6)


def test_s_only_objective_updates_online_future_encoder_and_head():
    torch.manual_seed(12)
    head = ActionPrefixMomentHead(
        embed_dim=3,
        action_dim=2,
        history_size=2,
        hidden_dim=7,
        gamma=0.9,
    )
    history = torch.randn(2, 2, 3, requires_grad=True)
    actions = torch.randn(2, 4, 2)
    target_future = torch.randn(2, 4, 3, requires_grad=True)

    output = successor_sequence_objective(
        head,
        history,
        actions,
        target_future,
        gamma=0.9,
    )
    output.successor_loss.backward()

    assert history.grad is not None and torch.count_nonzero(history.grad) > 0
    assert target_future.grad is not None
    assert torch.count_nonzero(target_future.grad) > 0
    assert any(parameter.grad is not None for parameter in head.parameters())
    assert not hasattr(output, "latent_loss")
    assert not hasattr(output, "recurrence_loss")


def test_s_only_planner_scores_prefix_without_world_model_rollout():
    torch.manual_seed(13)
    world_model = EncodeOnlyWorldModel()
    head = ActionPrefixMomentHead(
        embed_dim=2,
        action_dim=1,
        history_size=3,
        hidden_dim=6,
        gamma=0.95,
    )
    adapter = RewardFreeSuccessorLeWM(
        world_model,
        head,
        max_horizon=5,
        successor_weight=1.0,
        terminal_weight=0.0,
    )
    info = {
        "pixels": torch.randn(1, 4, 1, 2),
        "goal_emb": torch.randn(1, 1, 2),
    }
    candidates = torch.randn(1, 4, 5, 1)

    cost = adapter.get_cost(info, candidates)

    assert cost.shape == (1, 4)
    assert torch.isfinite(cost).all()
    assert world_model.rollout_calls == 0


def test_s_only_checkpoint_round_trip(tmp_path):
    head = ActionPrefixMomentHead(
        embed_dim=4,
        action_dim=3,
        history_size=2,
        hidden_dim=6,
        gamma=0.95,
    )
    config = {
        "architecture": "causal_gru_successor_increments",
        "embed_dim": 4,
        "action_dim": 3,
        "history_size": 2,
        "hidden_dim": 6,
        "max_horizon": 5,
        "gamma": 0.95,
        "goal_conditioning": "none",
        "action_conditioning": "causal_prefix",
    }
    checkpoint = tmp_path / "rf_successor_sequence.pt"
    torch.save(
        {
            "method": "rf_successor_sequence_wm",
            "objective_version": 2,
            "deployment_checkpoint_version": 1,
            "world_model_state_dict": {},
            "successor_state_dict": head.state_dict(),
            "successor_config": config,
        },
        checkpoint,
    )

    restored, restored_config, payload = load_rf_successor_checkpoint(checkpoint)

    assert isinstance(restored, ActionPrefixMomentHead)
    assert restored_config == config
    assert payload["method"] == "rf_successor_sequence_wm"
    for name, value in head.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])


def test_s_only_protocol_has_one_predictive_loss_and_no_ema_target():
    protocol = load_rf_successor_training_protocol(
        "configs/experiment/rf_successor_sequence_wm_cube_train.yaml"
    )
    objective = protocol["joint_objective"]

    assert protocol["method"] == "rf_successor_sequence_wm"
    assert objective["primitive_prediction"] == "successor_sequence"
    assert objective["successor_sequence_weight"] == 1.0
    assert objective["consistency"] == "architectural_discounted_cumsum"
    assert objective["target_encoder"] == "online_end_to_end"
    assert "local_prediction_weight" not in objective
    assert "multi_step_prediction_weight" not in objective
    assert "recurrence_weight" not in objective
    assert "target_world_ema_decay" not in protocol["successor"]


def test_s_only_evaluation_protocol_disables_autoregressive_rollout():
    protocol = load_rf_successor_evaluation_protocol(
        "configs/experiment/rf_successor_sequence_wm_cube_checkpoint_o50.yaml"
    )

    assert protocol["method"] == "rf_successor_sequence_wm"
    assert protocol["successor"]["objective_version"] == 2
    assert protocol["successor"]["terminal_weight"] == 0.0
    assert protocol["inference_objective"]["autoregressive_rollout"] == "disabled"
