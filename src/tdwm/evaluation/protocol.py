from __future__ import annotations

from typing import Any

import numpy as np


OFFICIAL_LEWM_SELECTION_PROTOCOL = "lewm_official_valid_start_v1"


def select_evaluation_points(
    lengths: Any,
    *,
    episodes: int,
    goal_offset: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if goal_offset < 1:
        raise ValueError("goal_offset must be positive")

    episode_lengths = np.asarray(lengths, dtype=np.int64)
    if episode_lengths.ndim != 1:
        raise ValueError("lengths must be one-dimensional")
    if np.any(episode_lengths < 0):
        raise ValueError("lengths must be non-negative")

    # Match the evaluator published with LeWM: sample uniformly over every
    # valid dataset row, rather than uniformly over episodes and then starts.
    # Its ``choice(len(valid_indices) - 1, ...)`` also leaves the final valid
    # row out of the sampling population, which we preserve for exact
    # reproduction of reported numbers.
    valid_starts_per_episode = np.maximum(
        episode_lengths - goal_offset,
        0,
    )
    cumulative_valid_starts = np.cumsum(valid_starts_per_episode)
    selectable_starts = max(int(valid_starts_per_episode.sum()) - 1, 0)
    if selectable_starts < episodes:
        raise ValueError(
            f"Requested {episodes} evaluations but only {selectable_starts} "
            "valid starts are selectable under the official protocol."
        )

    rng = np.random.default_rng(seed)
    selected_flat_indices = np.sort(
        rng.choice(selectable_starts, size=episodes, replace=False)
    )
    episode_ids = np.searchsorted(
        cumulative_valid_starts,
        selected_flat_indices,
        side="right",
    )
    previous_cumulative = np.where(
        episode_ids == 0,
        0,
        cumulative_valid_starts[np.maximum(episode_ids - 1, 0)],
    )
    start_steps = selected_flat_indices - previous_cumulative
    return episode_ids.astype(int).tolist(), start_steps.astype(int).tolist()
