from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from tdwm.adapters.successor_geometry_lewm import (
    RESIDUAL_POLICY_METHOD,
    load_successor_geometry_checkpoint,
)
from tdwm.evaluation.successor_geometry_lewm import (
    load_successor_geometry_evaluation_protocol,
)
from tdwm.methods.successor_geometry_lewm import (
    DirectedSuccessorGeometry,
    discounted_horizon_weights,
    successor_geometry_cost,
    successor_geometry_objective,
)
from tdwm.methods.residual_policy_lewm import (
    ResidualLeWM,
    build_expert_action_windows,
)
from tdwm.training.successor_geometry_lewm import (
    EpisodeDiverseBatchSampler,
    build_successor_geometry_windows,
    load_successor_geometry_training_protocol,
    validate_successor_geometry_training_protocol,
)


class IdentityGeometry(nn.Module):
    temperature = 0.1

    def encode_query(self, latent: torch.Tensor) -> torch.Tensor:
        return F.normalize(latent, dim=-1)

    def encode_goal(self, latent: torch.Tensor) -> torch.Tensor:
        return F.normalize(latent, dim=-1)


def test_discounted_horizon_weights_are_normalized_and_ordered():
    weights = discounted_horizon_weights(5, gamma=0.8)

    assert torch.isclose(weights.sum(), torch.tensor(1.0))
    assert torch.all(weights[:-1] > weights[1:])


def test_successor_objective_prefers_matching_future_pairs():
    geometry = IdentityGeometry()
    queries = torch.eye(4)
    goals = torch.stack((queries, queries), dim=1)
    groups = torch.arange(4)

    aligned = successor_geometry_objective(
        geometry,
        queries,
        queries,
        goals,
        groups,
        gamma=0.95,
    )
    shuffled = successor_geometry_objective(
        geometry,
        queries,
        queries,
        goals.roll(1, dims=0),
        groups,
        gamma=0.95,
    )

    assert aligned.loss < shuffled.loss
    assert torch.isclose(aligned.top1, torch.tensor(1.0))
    assert aligned.positive_margin > 0.0


def test_successor_objective_backpropagates_through_both_queries_and_goals():
    torch.manual_seed(4)
    geometry = DirectedSuccessorGeometry(
        embed_dim=6,
        projection_dim=4,
        hidden_dim=8,
        temperature=0.2,
    )
    real = torch.randn(6, 6, requires_grad=True)
    predicted = torch.randn(6, 6, requires_grad=True)
    goals = torch.randn(6, 3, 6, requires_grad=True)
    groups = torch.arange(6)

    output = successor_geometry_objective(
        geometry,
        real,
        predicted,
        goals,
        groups,
        gamma=0.9,
    )
    output.loss.backward()

    assert torch.isfinite(output.loss)
    assert real.grad is not None
    assert predicted.grad is not None
    assert goals.grad is not None
    assert all(parameter.grad is not None for parameter in geometry.parameters())


def test_same_episode_windows_are_masked_as_negatives():
    geometry = IdentityGeometry()
    queries = torch.eye(4)
    goals = torch.stack((queries, queries), dim=1)
    groups = torch.tensor([7, 7, 8, 9])

    output = successor_geometry_objective(
        geometry,
        queries,
        queries,
        goals,
        groups,
        gamma=1.0,
    )

    assert torch.isfinite(output.loss)
    assert torch.isclose(output.top1, torch.tensor(1.0))


def test_window_builder_aligns_rollout_terminal_and_future_offsets():
    latents = torch.arange(2 * 8, dtype=torch.float32).reshape(2, 8, 1)
    actions = torch.arange(2 * 8, dtype=torch.float32).reshape(2, 8, 1)
    windows = build_successor_geometry_windows(
        latents,
        actions,
        history_size=2,
        rollout_horizon=2,
        max_future_offset=2,
        episode_ids=torch.tensor([10, 11]),
    )

    assert windows.dynamics_count_per_clip == 5
    assert windows.geometry_count_per_clip == 3
    assert windows.history.shape == (10, 2, 1)
    assert windows.target_future.shape == (10, 2, 1)
    assert windows.real_terminal.shape == (6, 1)
    assert windows.future_goals.shape == (6, 2, 1)
    assert windows.real_terminal[:2, 0].tolist() == [3.0, 11.0]
    assert windows.future_goals[:2, :, 0].tolist() == [[4.0, 5.0], [12.0, 13.0]]
    assert windows.group_ids.tolist() == [10, 11, 10, 11, 10, 11]


def test_expert_action_windows_are_causal_and_mask_nonfinite_targets():
    latents = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1)
    actions = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)
    actions[:, 3] = torch.nan

    windows = build_expert_action_windows(latents, actions, history_size=3)

    assert windows.count_per_clip == 3
    assert windows.history[:, :, 0].tolist() == [
        [0.0, 1.0, 2.0],
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
    ]
    assert windows.past_actions.shape == (3, 2, 2)
    assert windows.target_actions[:, 0].tolist() == [4.0, 0.0, 8.0]
    assert windows.target_is_finite.tolist() == [True, False, True]


def test_residual_lewm_is_identity_at_initialization():
    class Predictor(nn.Module):
        def forward(self, embeddings, action_embeddings):
            return embeddings + action_embeddings

    pred_proj = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4))
    model = ResidualLeWM(
        encoder=nn.Identity(),
        predictor=Predictor(),
        action_encoder=nn.Identity(),
        pred_proj=pred_proj,
    )
    embeddings = torch.randn(3, 2, 4)
    action_embeddings = torch.randn(3, 2, 4)

    predicted = model.predict(embeddings, action_embeddings)

    assert torch.equal(predicted, embeddings)
    assert torch.count_nonzero(pred_proj[-1].weight) == 0
    assert torch.count_nonzero(pred_proj[-1].bias) == 0


def test_validation_sampler_guarantees_cross_episode_negatives():
    clip_indices = [
        (episode, start)
        for episode in range(6)
        for start in range(3)
    ]
    source_indices = [7, 0, 12, 4, 16, 9, 1, 13, 5, 17, 10, 2]
    sampler = EpisodeDiverseBatchSampler(
        source_indices,
        clip_indices,
        batch_size=4,
        seed=399,
    )

    batches = list(sampler)
    assert batches
    assert len(batches) == len(sampler)
    for batch in batches:
        episodes = {
            clip_indices[source_indices[position]][0] for position in batch
        }
        assert len(episodes) == len(batch)
    repeated = EpisodeDiverseBatchSampler(
        source_indices,
        clip_indices,
        batch_size=4,
        seed=399,
    )
    assert list(repeated) == batches


def test_directed_geometry_cost_is_bounded_and_not_parameter_tied():
    torch.manual_seed(8)
    geometry = DirectedSuccessorGeometry(
        embed_dim=5,
        projection_dim=3,
        hidden_dim=7,
        temperature=0.1,
    )
    query = torch.randn(4, 5)
    goal = torch.randn(4, 5)
    cost = successor_geometry_cost(geometry, query, goal)

    assert cost.shape == (4,)
    assert torch.all((0.0 <= cost) & (cost <= 2.0))
    assert geometry.query_projector is not geometry.goal_projector
    query_parameters = {id(parameter) for parameter in geometry.query_projector.parameters()}
    goal_parameters = {id(parameter) for parameter in geometry.goal_projector.parameters()}
    assert query_parameters.isdisjoint(goal_parameters)


def test_training_and_evaluation_protocols_are_reward_free_and_end_to_end():
    training = load_successor_geometry_training_protocol(
        "configs/experiment/successor_geometry_lewm_cube_train.yaml"
    )
    evaluation = load_successor_geometry_evaluation_protocol(
        "configs/experiment/successor_geometry_lewm_cube_checkpoint_o50.yaml"
    )

    assert training["initialization"] == "random_from_scratch"
    assert training["objective"]["reward"] == "none"
    assert training["objective"]["policy"] == "none"
    assert training["objective"]["td_bootstrap"] is False
    assert evaluation["geometry"]["planning_cost"] == "one_minus_directed_cosine"
    assert evaluation["planning"]["initial_distribution"] == "cem_gaussian_no_actor"

    invalid = {**training, "objective": {**training["objective"], "reward": "goal"}}
    try:
        validate_successor_geometry_training_protocol(invalid)
    except ValueError as error:
        assert "objective.reward" in str(error)
    else:
        raise AssertionError("A reward-conditioned protocol must be rejected.")


def test_residual_policy_protocol_keeps_the_action_head_out_of_inference():
    training = load_successor_geometry_training_protocol(
        "configs/experiment/residual_policy_successor_geometry_lewm_cube_train.yaml"
    )
    evaluation = load_successor_geometry_evaluation_protocol(
        "configs/experiment/"
        "residual_policy_successor_geometry_lewm_cube_checkpoint_o50.yaml"
    )

    assert training["method"] == RESIDUAL_POLICY_METHOD
    assert training["objective"]["reward"] == "none"
    assert training["model"]["transition_parameterization"] == "residual_delta"
    assert training["policy_auxiliary"]["inference_usage"] == "disabled"
    assert evaluation["planning"]["initial_distribution"] == "cem_gaussian_no_actor"
    assert evaluation["inference_objective"]["learned_action_policy"] is False


def test_successor_geometry_checkpoint_round_trip(tmp_path):
    geometry = DirectedSuccessorGeometry(
        embed_dim=5,
        projection_dim=3,
        hidden_dim=7,
        temperature=0.1,
    )
    config = {
        "architecture": "dual_mlp_directed_cosine",
        "embed_dim": 5,
        "projection_dim": 3,
        "hidden_dim": 7,
        "temperature": 0.1,
        "query_sources": ["real_terminal", "predicted_terminal"],
        "goal_conditioning": "future_pairs_only",
        "reward": "none",
        "policy": "none",
        "td_bootstrap": False,
    }
    checkpoint = tmp_path / "geometry.pt"
    torch.save(
        {
            "method": "successor_geometry_lewm",
            "objective_version": 1,
            "deployment_checkpoint_version": 1,
            "world_model_state_dict": {},
            "geometry_state_dict": geometry.state_dict(),
            "geometry_config": config,
        },
        checkpoint,
    )

    restored, restored_config, payload = load_successor_geometry_checkpoint(checkpoint)

    assert restored_config == config
    assert payload["method"] == "successor_geometry_lewm"
    for expected, actual in zip(
        geometry.parameters(), restored.parameters(), strict=True
    ):
        assert torch.equal(expected, actual)
