from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tdwm.evaluation.protocol import select_evaluation_points

ROOT = Path(__file__).resolve().parents[2]


def test_evaluation_points_are_fixed_unique_and_in_bounds() -> None:
    lengths = [10, 30, 31, 40, 50]
    first = select_evaluation_points(
        lengths,
        episodes=3,
        goal_offset=25,
        seed=42,
    )
    second = select_evaluation_points(
        lengths,
        episodes=3,
        goal_offset=25,
        seed=42,
    )

    assert first == second
    episode_ids, starts = first
    assert len(set(episode_ids)) == 3
    assert all(lengths[episode] > 25 for episode in episode_ids)
    assert all(
        0 <= start < lengths[episode] - 25
        for episode, start in zip(episode_ids, starts, strict=True)
    )


def test_evaluation_rejects_an_impossible_request() -> None:
    with pytest.raises(ValueError, match="only 1"):
        select_evaluation_points([10, 30], episodes=2, goal_offset=25, seed=0)


def test_world_callable_schema_matches_stable_worldmodel_0_1_1() -> None:
    for path in sorted((ROOT / "configs" / "envs").glob("*.yaml")):
        with path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        for callable_config in config["evaluation"].get("callables", []):
            for argument in callable_config.get("args", {}).values():
                assert "value" in argument
                assert "dataset_key" not in argument
                assert "literal" not in argument
                if argument.get("in_dataset", True) is False:
                    assert not isinstance(argument["value"], str)
