from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tdwm.evaluation.protocol import select_evaluation_points

ROOT = Path(__file__).resolve().parents[2]


def test_evaluation_points_match_published_lewm_sampling() -> None:
    lengths = [27, 28, 30]
    first = select_evaluation_points(
        lengths,
        episodes=5,
        goal_offset=25,
        seed=0,
    )
    second = select_evaluation_points(
        lengths,
        episodes=5,
        goal_offset=25,
        seed=0,
    )

    assert first == second
    episode_ids, starts = first
    assert list(zip(episode_ids, starts, strict=True)) == [
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 3),
    ]
    assert len(set(episode_ids)) < len(episode_ids)
    assert all(lengths[episode] > 25 for episode in episode_ids)
    assert all(
        0 <= start < lengths[episode] - 25
        for episode, start in zip(episode_ids, starts, strict=True)
    )


def test_evaluation_rejects_an_impossible_request() -> None:
    with pytest.raises(ValueError, match="only 4 valid starts"):
        select_evaluation_points([10, 30], episodes=5, goal_offset=25, seed=0)


def test_evaluation_rejects_invalid_lengths() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        select_evaluation_points([[30]], episodes=1, goal_offset=25, seed=0)
    with pytest.raises(ValueError, match="non-negative"):
        select_evaluation_points([-1, 30], episodes=1, goal_offset=25, seed=0)
    with pytest.raises(ValueError, match="only 0 valid starts"):
        select_evaluation_points([], episodes=1, goal_offset=25, seed=0)


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
