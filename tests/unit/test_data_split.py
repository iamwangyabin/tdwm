from __future__ import annotations

import numpy as np
import pytest

from tdwm.training.data import (
    column_statistics,
    make_episode_split,
    make_episode_view,
)


class FakeDataset:
    offsets = np.asarray([0, 2, 4])
    lengths = np.asarray([2, 2, 2])

    def get_col_data(self, key: str) -> np.ndarray:
        assert key == "action"
        return np.asarray([[0.0], [2.0], [100.0], [100.0], [4.0], [6.0]])


def test_episode_split_never_mixes_clips_from_the_same_episode() -> None:
    clips = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0)]
    split = make_episode_split(clips, train_fraction=0.5, seed=42)
    train_indices, validation_indices = split.indices_for(clips)

    train_episodes = {clips[index][0] for index in train_indices}
    validation_episodes = {clips[index][0] for index in validation_indices}
    assert train_episodes == set(split.train_episodes)
    assert validation_episodes == set(split.validation_episodes)
    assert train_episodes.isdisjoint(validation_episodes)
    assert sorted(train_indices + validation_indices) == list(range(len(clips)))


def test_episode_split_is_deterministic() -> None:
    clips = [(episode, start) for episode in range(10) for start in range(2)]
    assert make_episode_split(clips, train_fraction=0.8, seed=7) == (
        make_episode_split(clips, train_fraction=0.8, seed=7)
    )


def test_statistics_only_use_training_episodes() -> None:
    result = column_statistics(FakeDataset(), ["action"], episodes=[0, 2])

    assert result["action"]["mean"] == pytest.approx([3.0])
    assert result["action"]["std"] == pytest.approx([2.2360679])


def test_split_requires_at_least_two_episodes() -> None:
    with pytest.raises(ValueError, match="At least two"):
        make_episode_split([(0, 0), (0, 1)], train_fraction=0.9, seed=42)


def test_episode_view_excludes_other_samples_and_random_goal_lengths() -> None:
    dataset = FakeDataset()
    dataset.clip_indices = [(0, 0), (1, 0), (2, 0)]
    view = make_episode_view(dataset, episodes=[0, 2])

    assert view.clip_indices == [(0, 0), (2, 0)]
    assert view.lengths.tolist() == [2, 0, 2]
    assert dataset.lengths.tolist() == [2, 2, 2]
