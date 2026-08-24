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
    LeWMResidualTransformerHead,
    residual_manifold_sequence_objective,
)
from tdwm.methods.successor_geometry import latent_goal_cost
from tdwm.training.rf_successor_lewm import load_rf_successor_training_protocol


SOURCE_HASH = "0ce38860a672c4a304d6921c6f07158977bb1d2c8f0eed8a002bb7c89502b579"


def _small_head(*, embed_dim: int = 8) -> LeWMResidualTransformerHead:
    return LeWMResidualTransformerHead(
        embed_dim=embed_dim,
        action_dim=3,
        history_size=2,
        gamma=0.9,
        prefix_depth=1,
        prefix_heads=2,
        prefix_mlp_dim=24,
        predictor_depth=1,
        predictor_mlp_dim=32,
        fusion_dim=24,
        dropout=0.0,
    )


def test_residual_head_exactly_recovers_frozen_lewm_at_initialization():
    torch.manual_seed(19)
    head = _small_head()
    history = torch.randn(2, 2, 8)
    actions = torch.randn(2, 4, 3)
    base_future = torch.randn(2, 4, 8)

    correction = head.predict_correction(history, actions, base_future)
    corrected = head.predict_latents(history, actions, base_future)

    assert torch.count_nonzero(correction) == 0
    assert torch.equal(corrected, base_future)


def test_residual_objective_has_one_loss_and_trains_the_zero_output_layer():
    torch.manual_seed(23)
    head = _small_head()
    history = torch.randn(2, 2, 8, requires_grad=True)
    actions = torch.randn(2, 3, 3)
    base_future = torch.randn(2, 3, 8, requires_grad=True)
    target_future = base_future + 0.2 * torch.randn_like(base_future)

    output = residual_manifold_sequence_objective(
        head,
        history,
        actions,
        base_future,
        target_future,
        gamma=0.9,
    )
    output.latent_loss.backward()

    expected = (base_future - target_future).square().mean()
    assert torch.allclose(output.latent_loss, expected)
    assert torch.allclose(
        output.latent_mse_by_horizon,
        output.base_latent_mse_by_horizon,
    )
    assert head.correction_out.weight.grad is not None
    assert torch.count_nonzero(head.correction_out.weight.grad) > 0
    assert history.grad is None
    assert base_future.grad is None


def test_residual_prefix_is_causal():
    torch.manual_seed(29)
    head = _small_head()
    with torch.no_grad():
        head.correction_out.weight.copy_(torch.eye(8))
    history = torch.randn(1, 2, 8)
    base_future = torch.randn(1, 4, 8)
    actions = torch.randn(1, 4, 3)
    changed = actions.clone()
    changed[:, -1] += 10.0

    original = head.predict_correction(history, actions, base_future)
    perturbed = head.predict_correction(history, changed, base_future)

    assert torch.allclose(original[:, :-1], perturbed[:, :-1], atol=1e-6)


def test_residual_protocol_is_frozen_reward_free_and_single_objective():
    train = load_rf_successor_training_protocol(
        "configs/experiment/rf_frozen_residual_prefix_wm_cube_train.yaml"
    )
    evaluation = load_rf_successor_evaluation_protocol(
        "configs/experiment/rf_frozen_residual_prefix_wm_cube_o50.yaml"
    )

    assert train["method"] == "rf_frozen_residual_prefix_wm"
    assert train["initialization"] == "frozen_pretrained_lewm"
    assert train["pretrained_world_model"]["checkpoint_sha256"] == SOURCE_HASH
    assert train["joint_objective"]["goal_conditioning"] == "none"
    assert train["joint_objective"]["policy"] == "none"
    assert train["joint_objective"]["bootstrap"] == "none"
    assert train["joint_objective"]["latent_sequence_weight"] == 1.0
    assert train["loss"]["sigreg"]["weight"] == 0.0
    assert train["training"]["freeze_world_model_from_start"] is True
    assert evaluation["successor"]["objective_version"] == 10
    assert evaluation["successor"]["planning_query"] == "lewm_residual_terminal"


class _RolloutWorld(nn.Module):
    def __init__(self, base_future: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("base_future", base_future)

    def rollout(self, info, action_sequence, history_size=None):
        del history_size
        batch, samples, horizon = action_sequence.shape[:3]
        history = info["emb"].expand(batch, samples, -1, -1)
        future = self.base_future[:, :samples, :horizon]
        return {"predicted_emb": torch.cat((history, future), dim=-2)}


def test_residual_planner_starts_as_exact_lewm_terminal_cost():
    head = LeWMResidualTransformerHead(
        embed_dim=2,
        action_dim=1,
        history_size=1,
        gamma=0.9,
        prefix_depth=1,
        prefix_heads=1,
        prefix_mlp_dim=8,
        predictor_depth=1,
        predictor_mlp_dim=8,
        fusion_dim=8,
        dropout=0.0,
    )
    base_future = torch.tensor(
        [[[[1.0, 0.0], [2.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]]
    )
    actions = torch.zeros(1, 2, 2, 1)
    goal = torch.tensor([[2.0, 0.0]])
    info = {
        "pixels": torch.zeros(1, 1, 1),
        "emb": torch.zeros(1, 1, 1, 2),
        "goal_emb": goal,
    }
    adapter = RewardFreeSuccessorLeWM(
        _RolloutWorld(base_future),
        head,
        max_horizon=2,
        planning_query="lewm_residual_terminal",
    )

    cost = adapter.get_cost(info, actions)
    expected = latent_goal_cost(base_future[..., -1, :], goal.unsqueeze(1))

    assert torch.allclose(cost, expected)


def test_residual_checkpoint_round_trip(tmp_path):
    head = _small_head()
    config = {
        "architecture": "causal_transformer_lewm_residual",
        "embed_dim": 8,
        "action_dim": 3,
        "history_size": 2,
        "prefix_depth": 1,
        "prefix_heads": 2,
        "prefix_mlp_dim": 24,
        "predictor_depth": 1,
        "predictor_mlp_dim": 32,
        "fusion_dim": 24,
        "dropout": 0.0,
        "max_horizon": 5,
        "gamma": 0.9,
        "goal_conditioning": "none",
        "action_conditioning": "causal_prefix",
        "pretrained_world_model_sha256": SOURCE_HASH,
    }
    checkpoint = tmp_path / "rf_frozen_residual_prefix_wm.pt"
    torch.save(
        {
            "method": "rf_frozen_residual_prefix_wm",
            "objective_version": 10,
            "deployment_checkpoint_version": 1,
            "world_model_state_dict": {},
            "successor_state_dict": head.state_dict(),
            "successor_config": config,
        },
        checkpoint,
    )

    restored, restored_config, payload = load_rf_successor_checkpoint(checkpoint)

    assert isinstance(restored, LeWMResidualTransformerHead)
    assert restored_config == config
    assert payload["objective_version"] == 10
