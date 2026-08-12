from __future__ import annotations

from typing import Any

import numpy as np


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

    eligible = [
        index
        for index, length in enumerate(lengths)
        if int(length) > goal_offset
    ]
    if len(eligible) < episodes:
        raise ValueError(
            f"Requested {episodes} evaluations but only {len(eligible)} "
            "episodes are long enough."
        )
    rng = np.random.default_rng(seed)
    selected = sorted(
        int(value)
        for value in rng.choice(eligible, size=episodes, replace=False)
    )
    starts = [
        int(rng.integers(0, int(lengths[episode]) - goal_offset))
        for episode in selected
    ]
    return selected, starts
