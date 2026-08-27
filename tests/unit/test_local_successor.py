from __future__ import annotations

import hashlib

import pytest
import torch
from torch import nn

from tdwm.adapters.local_successor import (
    LocalSuccessorLeWM,
    load_local_successor_heads,
)
from tdwm.evaluation.local_successor import (
    _validate_checkpoint_pair,
    load_ls_evaluation_protocol,
)
from tdwm.methods.local_successor import (
    LocalSuccessorHeads,
    future_goal_successor_objective,
    goal_cost_weights,
    latent_goal_cost,
    successor_feature_basis,
    successor_goal_cost,
    successor_td_target,
)
from tdwm.training.local_successor import load_ls_training_protocol


class FixedPolicy(nn.Module):
    def __init__(self, action_dim: int, value: float = 0.0) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.value = value

    def forward(self, latent_history, previous_actions, goal):
        return torch.full(
            latent_history.shape[:-2] + (self.action_dim,),
            self.value,
            device=latent_history.device,
            dtype=latent_history.dtype,
        )


class FixedSuccessor(nn.Module):
    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("value", value)

    def forward(self, latent_history, previous_actions, action, goal):
        return self.value.to(latent_history).expand(
            *latent_history.shape[:-2], self.value.numel()
        )


class ActionEchoSuccessor(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim

    def forward(self, latent_history, previous_actions, action, goal):
        output = action.new_zeros(*action.shape[:-1], self.output_dim)
        output[..., 0] = action[..., 0]
        return output


class CurrentLatentPolicy(nn.Module):
    def forward(self, latent_history, previous_actions, goal):
        return latent_history[..., -1, :1]


class FakeWorldModel(nn.Module):
    def __init__(self, predicted: torch.Tensor) -> None:
        super().__init__()
        self.parameter = nn.Parameter(torch.zeros(()))
        self.predicted = predicted
        self.action_encoder = nn.Identity()

    def rollout(self, info, action_sequence, history_size=None):
        return {"predicted_emb": self.predicted.to(action_sequence)}

    def predict(self, embeddings, action_embeddings):
        return embeddings + action_embeddings


def test_successor_basis_exactly_linearizes_latent_goal_distance():
    torch.manual_seed(3)
    latent = torch.randn(2, 4, 7)
    goal = torch.randn(2, 4, 7)

    lifted = successor_feature_basis(latent)
    projected = successor_goal_cost(lifted, goal)

    assert lifted.shape[-1] == latent.shape[-1] + 2
    assert torch.allclose(projected, latent_goal_cost(latent, goal), atol=1e-6)
    assert torch.allclose(
        projected,
        (lifted * goal_cost_weights(goal)).sum(dim=-1),
        atol=1e-6,
    )


def test_successor_td_target_has_exact_terminal_boundary():
    torch.manual_seed(4)
    next_latent = torch.randn(3, 5)
    bootstrap = torch.randn(3, 7)
    gamma = 0.8
    immediate = (1.0 - gamma) * successor_feature_basis(next_latent)

    continuing = successor_td_target(
        next_latent, bootstrap, gamma=gamma, terminal=False
    )
    terminal = successor_td_target(
        next_latent, bootstrap, gamma=gamma, terminal=True
    )
    masked = successor_td_target(
        next_latent,
        bootstrap,
        gamma=gamma,
        terminal=torch.tensor([True, False, True]),
    )

    assert torch.allclose(continuing, immediate + gamma * bootstrap)
    assert torch.allclose(terminal, immediate)
    assert torch.allclose(masked[[0, 2]], immediate[[0, 2]])
    assert torch.allclose(masked[1], continuing[1])


def test_successor_objective_uses_executable_target_policy_and_shapes_latent():
    torch.manual_seed(5)
    heads = LocalSuccessorHeads(
        embed_dim=3, action_dim=2, history_size=2, hidden_dim=8
    )
    target_heads = heads.make_target()
    latents = torch.randn(2, 5, 3, requires_grad=True)
    actions = torch.randn(2, 5, 2)

    output = future_goal_successor_objective(
        heads,
        target_heads,
        latents,
        actions,
        gamma=0.9,
    )
    loss = output.td_loss + output.boundary_loss + output.policy_loss
    loss.backward()

    assert output.pair_count == 12
    assert output.td_loss.ndim == 0
    assert output.boundary_loss.ndim == 0
    assert output.policy_loss.ndim == 0
    assert latents.grad is not None
    assert torch.count_nonzero(latents.grad) > 0
    assert any(parameter.grad is not None for parameter in heads.parameters())
    assert all(parameter.grad is None for parameter in target_heads.parameters())


def test_successor_bootstrap_is_conditioned_on_target_policy_action():
    heads = LocalSuccessorHeads(
        embed_dim=1, action_dim=1, history_size=1, hidden_dim=4
    )
    heads.policy = FixedPolicy(action_dim=1)
    heads.successor = FixedSuccessor(torch.zeros(3))
    latents = torch.zeros(1, 4, 1)
    actions = torch.zeros(1, 4, 1)

    def td_loss_for_policy(value: float):
        target = LocalSuccessorHeads(
            embed_dim=1, action_dim=1, history_size=1, hidden_dim=4
        )
        target.policy = FixedPolicy(action_dim=1, value=value)
        target.successor = ActionEchoSuccessor(output_dim=3)
        return future_goal_successor_objective(
            heads,
            target,
            latents,
            actions,
            gamma=0.9,
            train_policy=False,
        ).td_loss

    zero_policy_loss = td_loss_for_policy(0.0)
    nonzero_policy_loss = td_loss_for_policy(2.0)

    assert nonzero_policy_loss > zero_policy_loss


def test_successor_mpc_cost_splices_prefix_and_policy_tail_without_overlap():
    predicted = torch.tensor([[[[99.0, 99.0], [1.0, 0.0], [2.0, 0.0]]]])
    heads = LocalSuccessorHeads(
        embed_dim=2, action_dim=1, history_size=2, hidden_dim=4
    )
    heads.policy = FixedPolicy(action_dim=1)
    heads.successor = FixedSuccessor(torch.tensor([0.0, 0.0, 2.0, 0.0]))
    adapter = LocalSuccessorLeWM(
        FakeWorldModel(predicted), heads, gamma=0.5, clamp_tail_cost=True
    )
    info = {
        "pixels": torch.zeros(1, 1, 1, 3, 2, 2),
        "goal": torch.zeros(1, 1, 1, 2),
        "goal_emb": torch.zeros(1, 1, 2),
    }
    actions = torch.zeros(1, 1, 2, 1)

    cost = adapter.get_cost(info, actions)

    expected_prefix = 0.5 * (0.5 + 0.5 * 2.0)
    expected_tail = 0.5**2 * 2.0
    assert cost.shape == (1, 1)
    assert torch.allclose(cost, torch.tensor([[expected_prefix + expected_tail]]))


def test_successor_actor_rollout_advances_context_for_cem_warm_start():
    heads = LocalSuccessorHeads(
        embed_dim=1, action_dim=1, history_size=1, hidden_dim=4
    )
    heads.policy = FixedPolicy(action_dim=1, value=0.25)
    adapter = LocalSuccessorLeWM(
        FakeWorldModel(torch.empty(0)), heads, gamma=0.9
    )
    info = {
        "emb": torch.zeros(2, 1, 1),
        "goal_emb": torch.ones(2, 1, 1),
    }

    actions = adapter.get_action(info, horizon=3)

    assert actions.shape == (2, 3, 1)
    assert torch.allclose(actions, torch.full_like(actions, 0.25))


def test_successor_actor_applies_warm_start_prefix_before_tail():
    heads = LocalSuccessorHeads(
        embed_dim=1, action_dim=1, history_size=1, hidden_dim=4
    )
    heads.policy = CurrentLatentPolicy()
    adapter = LocalSuccessorLeWM(
        FakeWorldModel(torch.empty(0)), heads, gamma=0.9
    )
    info = {
        "emb": torch.zeros(1, 1, 1),
        "goal_emb": torch.ones(1, 1, 1),
    }

    actions = adapter.get_action(
        info, horizon=2, prefix_actions=torch.tensor([[[0.5]]])
    )

    assert torch.allclose(actions, torch.tensor([[[0.5], [1.0]]]))


def test_local_successor_checkpoint_round_trip(tmp_path):
    heads = LocalSuccessorHeads(
        embed_dim=4, action_dim=3, history_size=2, hidden_dim=6
    )
    checkpoint = tmp_path / "heads.pt"
    config = {
        "objective_version": 1,
        "embed_dim": 4,
        "action_dim": 3,
        "history_size": 2,
        "hidden_dim": 6,
        "gamma": 0.95,
        "continuation_policy": "hindsight_gcbc",
        "goal_offset_weighting": "uniform_offsets",
        "terminal_condition": "next_state_is_hindsight_goal",
        "base_export_run_name": "epoch_01",
        "base_checkpoint_sha256": "0" * 64,
    }
    torch.save(
        {"heads_state_dict": heads.state_dict(), "heads_config": config},
        checkpoint,
    )

    restored, restored_config = load_local_successor_heads(checkpoint)

    assert restored_config == config
    for name, value in heads.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])


def test_ls_checkpoint_pair_is_bound_by_export_name_and_hash(tmp_path):
    base_file = tmp_path / "weights.pt"
    base_file.write_bytes(b"paired LeWM weights")
    digest = hashlib.sha256(base_file.read_bytes()).hexdigest()
    config = {
        "base_export_run_name": "epoch_03",
        "base_checkpoint_sha256": digest,
    }

    _validate_checkpoint_pair(
        base_name="epoch_03", base_file=base_file, heads_config=config
    )

    with pytest.raises(ValueError, match="does not match"):
        _validate_checkpoint_pair(
            base_name="epoch_03",
            base_file=base_file,
            heads_config={**config, "base_checkpoint_sha256": "0" * 64},
        )


def test_ls_protocol_is_independent_and_preserves_sigreg_batch():
    protocol = load_ls_training_protocol(
        "configs/experiment/ls_lewm_cube_train.yaml"
    )

    assert protocol["method"] == "ls_lewm"
    assert protocol["successor"]["continuation_policy"] == "hindsight_gcbc"
    assert protocol["successor"]["td_steps"] == 1
    assert protocol["successor"]["goal_offset_weighting"] == "uniform_offsets"
    assert protocol["sequence"]["num_steps"] == (
        protocol["sequence"]["history_frames"]
        + protocol["successor"]["max_goal_offset"]
    )
    assert (
        protocol["loader"]["batch_size"]
        * (
            protocol["sequence"]["num_steps"]
            - protocol["sequence"]["history_frames"]
        )
        == protocol["loss"]["sigreg"]["effective_batch_size"]
    )


def test_ls_evaluation_uses_successor_tail_and_actor_warm_start():
    protocol = load_ls_evaluation_protocol(
        "configs/experiment/ls_lewm_cube_checkpoint_o50.yaml"
    )

    assert protocol["method"] == "ls_lewm"
    assert protocol["planning"]["initial_distribution"] == (
        "continuation_policy_rollout"
    )
    assert protocol["inference_objective"]["tail"] == (
        "policy_conditioned_vector_successor_goal_cost"
    )
    assert protocol["evaluation"]["goal_offset"] > (
        protocol["planning"]["horizon"] * protocol["planning"]["action_block"]
    )
