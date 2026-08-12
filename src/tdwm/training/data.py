from __future__ import annotations

import random
from copy import copy
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class EpisodeSplit:
    train_episodes: tuple[int, ...]
    validation_episodes: tuple[int, ...]
    seed: int

    def indices_for(
        self, clip_indices: Sequence[tuple[int, int]]
    ) -> tuple[list[int], list[int]]:
        train = set(self.train_episodes)
        validation = set(self.validation_episodes)
        train_indices: list[int] = []
        validation_indices: list[int] = []
        for index, (episode, _) in enumerate(clip_indices):
            if int(episode) in train:
                train_indices.append(index)
            elif int(episode) in validation:
                validation_indices.append(index)
        return train_indices, validation_indices

    def as_dict(self) -> dict[str, Any]:
        return {
            "unit": "episode",
            "seed": self.seed,
            "train_episodes": list(self.train_episodes),
            "validation_episodes": list(self.validation_episodes),
        }


def make_episode_split(
    clip_indices: Iterable[tuple[int, int]],
    *,
    train_fraction: float,
    seed: int,
) -> EpisodeSplit:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1")

    episodes = sorted({int(episode) for episode, _ in clip_indices})
    if len(episodes) < 2:
        raise ValueError("At least two eligible episodes are required for a split")

    random.Random(seed).shuffle(episodes)
    train_count = max(
        1,
        min(len(episodes) - 1, int(len(episodes) * train_fraction)),
    )
    return EpisodeSplit(
        train_episodes=tuple(sorted(episodes[:train_count])),
        validation_episodes=tuple(sorted(episodes[train_count:])),
        seed=seed,
    )


def column_statistics(
    dataset: Any,
    keys: list[str],
    episodes: Iterable[int],
) -> dict[str, Any]:
    episode_ids = tuple(int(episode) for episode in episodes)
    if not episode_ids:
        raise ValueError("Cannot compute statistics without training episodes")

    statistics: dict[str, Any] = {}
    for key in keys:
        all_values = np.asarray(dataset.get_col_data(key), dtype=np.float32)
        chunks = [
            all_values[
                int(dataset.offsets[episode]) : int(dataset.offsets[episode])
                + int(dataset.lengths[episode])
            ]
            for episode in episode_ids
        ]
        values = np.concatenate(chunks, axis=0)
        if values.ndim == 1:
            values = values[:, None]
        values = values.reshape(-1, values.shape[-1])
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        statistics[key] = {
            "mean": mean.astype(float).tolist(),
            "std": std.astype(float).tolist(),
        }
    return statistics


def make_episode_view(dataset: Any, episodes: Iterable[int]) -> Any:
    """Create a shallow dataset view whose samples and lengths use one split."""
    episode_set = {int(episode) for episode in episodes}
    if not episode_set:
        raise ValueError("Cannot create a dataset view without episodes")

    view = copy(dataset)
    view.clip_indices = [
        clip for clip in dataset.clip_indices if int(clip[0]) in episode_set
    ]
    view.lengths = np.asarray(dataset.lengths).copy()
    excluded = [
        index for index in range(len(view.lengths)) if index not in episode_set
    ]
    view.lengths[excluded] = 0
    return view
