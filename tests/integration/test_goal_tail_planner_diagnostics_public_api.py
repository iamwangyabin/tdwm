from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from gymnasium.spaces import Box
from torch import nn

from tdwm.adapters.mc_gt_lewm import GoalTailPlannerDiagnosticsRecorder


swm = pytest.importorskip("stable_worldmodel")


class _DiagnosticCostModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.parameter = nn.Parameter(torch.zeros(()))
        self.last_cost_components = None

    def get_cost(self, info, candidates):
        del info
        terminal = candidates.square().mean(dim=(-2, -1))
        tail = (candidates[..., 0, 0] + 0.5).square()
        total = terminal + tail
        self.last_cost_components = {
            "terminal_cost": terminal.detach(),
            "tail_value": tail.detach(),
            "total_cost": total.detach(),
            "boundary_value": torch.zeros_like(total),
        }
        return total


def test_stable_worldmodel_cem_public_callback_records_goal_tail_components():
    model = _DiagnosticCostModel()
    recorder = GoalTailPlannerDiagnosticsRecorder(
        model,
        record_iterations=[0, 1],
    )
    solver = swm.solver.CEMSolver(
        model=model,
        batch_size=1,
        num_samples=8,
        var_scale=1.0,
        n_steps=2,
        topk=2,
        device="cpu",
        seed=7,
        callbacks=[recorder],
    )
    solver.configure(
        action_space=Box(low=-1.0, high=1.0, shape=(1, 1)),
        n_envs=1,
        config=SimpleNamespace(horizon=2, action_block=1),
    )

    output = solver.solve({"pixels": torch.zeros(1, 1)})
    exported = recorder.export()

    assert output["actions"].shape == (1, 2, 1)
    assert recorder.output_key in output["callbacks"]
    assert exported["solve_count"] == 1
    assert exported["record_count"] == 2
    assert exported["aggregates"]["boundary_value_abs_max"]["max"] == 0.0
