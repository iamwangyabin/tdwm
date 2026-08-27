#!/usr/bin/env python3
"""Recompute the archived paired A/C/D Cube O50 statistics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = (
    REPOSITORY_ROOT / "reports/artifacts/aligned_acd_o50_seed3072"
)
DEFAULT_CSV = ARTIFACT_ROOT / "paired_outcomes.csv"
DEFAULT_SUMMARY = ARTIFACT_ROOT / "summary.json"
METHOD_COLUMNS = {"A": "success_A", "C": "success_C", "D": "success_D"}
COMPARISONS = (("A", "C"), ("C", "D"), ("A", "D"))


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"Invalid boolean value {value!r}.")


def _pair_hash(episode_index: int, start_step: int, goal_step: int) -> str:
    payload = {
        "episode_index": episode_index,
        "goal_step": goal_step,
        "start_step": start_step,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _exact_mcnemar(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    lower = min(left_only, right_only)
    probability = 2.0 * sum(
        math.comb(discordant, count) for count in range(lower + 1)
    ) / (2**discordant)
    return min(1.0, probability)


def _holm_adjust(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=raw.get)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        candidate = min(1.0, (total - rank) * raw[name])
        running_max = max(running_max, candidate)
        adjusted[name] = running_max
    return adjusted


def _load_rows(path: Path) -> list[dict[str, Any]]:
    required = {
        "planning_seed",
        "selection_position",
        "selection_hash",
        "episode_index",
        "start_step",
        "goal_step",
        "valid_row_rank",
        "pair_hash",
        *METHOD_COLUMNS.values(),
    }
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
        for source in reader:
            row: dict[str, Any] = {
                "planning_seed": int(source["planning_seed"]),
                "selection_position": int(source["selection_position"]),
                "selection_hash": source["selection_hash"],
                "episode_index": int(source["episode_index"]),
                "start_step": int(source["start_step"]),
                "goal_step": int(source["goal_step"]),
                "valid_row_rank": int(source["valid_row_rank"]),
                "pair_hash": source["pair_hash"],
            }
            for method, column in METHOD_COLUMNS.items():
                row[method] = _parse_bool(source[column])
            expected_hash = _pair_hash(
                row["episode_index"], row["start_step"], row["goal_step"]
            )
            if row["pair_hash"] != expected_hash:
                raise ValueError(
                    f"pair_hash mismatch at planning seed {row['planning_seed']} "
                    f"position {row['selection_position']}."
                )
            rows.append(row)
    if not rows:
        raise ValueError("The paired outcomes CSV is empty.")
    return rows


def _comparison(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    both = sum(row[left] and row[right] for row in rows)
    left_only = sum(row[left] and not row[right] for row in rows)
    right_only = sum(not row[left] and row[right] for row in rows)
    neither = len(rows) - both - left_only - right_only
    delta = right_only - left_only
    return {
        "left": left,
        "right": right,
        "episodes": len(rows),
        "both_success": both,
        "left_only": left_only,
        "right_only": right_only,
        "neither_success": neither,
        "delta_successes": delta,
        "delta_percentage_points": 100.0 * delta / len(rows),
        "mcnemar_exact_two_sided_p": _exact_mcnemar(left_only, right_only),
    }


def summarize(path: Path) -> dict[str, Any]:
    rows = _load_rows(path)
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[row["planning_seed"]].append(row)

    per_selection: dict[str, Any] = {}
    expected_positions: int | None = None
    for seed in sorted(by_seed):
        selected = sorted(by_seed[seed], key=lambda row: row["selection_position"])
        positions = [row["selection_position"] for row in selected]
        if positions != list(range(len(selected))):
            raise ValueError(f"Planning seed {seed} has non-contiguous positions.")
        hashes = {row["selection_hash"] for row in selected}
        if len(hashes) != 1:
            raise ValueError(f"Planning seed {seed} has multiple selection hashes.")
        if expected_positions is None:
            expected_positions = len(selected)
        elif len(selected) != expected_positions:
            raise ValueError("Planning selections do not have equal episode counts.")
        successes = {
            method: sum(row[method] for row in selected) for method in METHOD_COLUMNS
        }
        per_selection[str(seed)] = {
            "episodes": len(selected),
            "selection_sha256": hashes.pop(),
            "successes": successes,
            "success_rates_percent": {
                method: 100.0 * count / len(selected)
                for method, count in successes.items()
            },
            "deltas_percentage_points": {
                "C_minus_A": 100.0 * (successes["C"] - successes["A"]) / len(selected),
                "D_minus_C": 100.0 * (successes["D"] - successes["C"]) / len(selected),
                "D_minus_A": 100.0 * (successes["D"] - successes["A"]) / len(selected),
            },
        }

    totals = {method: sum(row[method] for row in rows) for method in METHOD_COLUMNS}
    comparisons: dict[str, Any] = {}
    raw_p: dict[str, float] = {}
    for left, right in COMPARISONS:
        name = f"{left}_to_{right}"
        comparisons[name] = _comparison(rows, left, right)
        raw_p[name] = comparisons[name]["mcnemar_exact_two_sided_p"]
    adjusted = _holm_adjust(raw_p)
    for name, value in adjusted.items():
        comparisons[name]["holm_adjusted_p_across_three_contrasts"] = value

    pair_to_seeds: dict[str, set[int]] = defaultdict(set)
    episode_occurrences: Counter[int] = Counter()
    for row in rows:
        pair_to_seeds[row["pair_hash"]].add(row["planning_seed"])
        episode_occurrences[row["episode_index"]] += 1
    duplicate_pairs = sorted(
        pair_hash for pair_hash, seeds in pair_to_seeds.items() if len(seeds) > 1
    )
    repeated_episodes = {
        str(episode): count
        for episode, count in sorted(episode_occurrences.items())
        if count > 1
    }

    direction_counts: dict[str, dict[str, int]] = {}
    for delta_name in ("C_minus_A", "D_minus_C", "D_minus_A"):
        values = [
            selection["deltas_percentage_points"][delta_name]
            for selection in per_selection.values()
        ]
        direction_counts[delta_name] = {
            "positive": sum(value > 0.0 for value in values),
            "zero": sum(value == 0.0 for value in values),
            "negative": sum(value < 0.0 for value in values),
        }

    return {
        "schema_version": 1,
        "study": {
            "environment": "swm/OGBCube-v0",
            "goal_offset": 50,
            "training_seed": 3072,
            "planning_selection_seeds": sorted(by_seed),
            "episodes_per_selection": expected_positions,
            "episode_count": len(rows),
            "note": (
                "These are six evaluation selections for one training seed, "
                "not six independently trained models."
            ),
        },
        "methods": {
            "A": "Original LeWM world model with terminal cost only",
            "B": (
                "Original LeWM world model with a boundary-anchored tail trained "
                "in original LeWM coordinates; not yet completed"
            ),
            "B_prime": (
                "Original LeWM world model with the Aligned-coordinate tail; "
                "invalid cross-coordinate diagnostic only"
            ),
            "C": "Aligned world model with terminal cost only",
            "D": "Aligned world model with boundary-anchored MC tail",
        },
        "per_selection": per_selection,
        "pooled": {
            "episodes": len(rows),
            "successes": totals,
            "success_rates_percent": {
                method: 100.0 * count / len(rows)
                for method, count in totals.items()
            },
            "comparisons": comparisons,
            "selection_direction_counts": direction_counts,
            "multiplicity_note": (
                "A-to-C, C-to-D, and A-to-D are reported as three peer "
                "contrasts. Holm-adjusted p-values are included; A-to-C was "
                "not preregistered as a primary contrast."
            ),
        },
        "sample_audit": {
            "rows": len(rows),
            "unique_pair_hashes": len(pair_to_seeds),
            "cross_planning_seed_duplicate_pair_count": len(duplicate_pairs),
            "cross_planning_seed_duplicate_pair_hashes": duplicate_pairs,
            "unique_source_episodes": len(episode_occurrences),
            "source_episodes_used_more_than_once": repeated_episodes,
        },
        "execution_notes": {
            "planned_vectorized_o200": {
                "status": "terminated_by_operating_system",
                "reason": (
                    "Constructing 200 MuJoCo environments exceeded the RTX 3090 "
                    "host's approximately 15 GiB memory budget."
                ),
                "replacement": (
                    "Six separate matched 50-episode planning selections, "
                    "totalling 300 paired episodes."
                ),
                "interpretation_constraint": (
                    "The archive is six-by-O50 and must not be represented as one "
                    "O200 execution."
                ),
            }
        },
    }


def _serialized(summary: dict[str, Any]) -> str:
    return json.dumps(summary, indent=2, sort_keys=True) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--output", type=Path)
    group.add_argument(
        "--check",
        type=Path,
        nargs="?",
        const=DEFAULT_SUMMARY,
        help="Fail unless the recomputed JSON equals the archived summary.",
    )
    args = parser.parse_args()
    summary = summarize(args.csv)
    serialized = _serialized(summary)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    if args.check is not None:
        archived = json.loads(args.check.read_text())
        if archived != summary:
            raise SystemExit(f"Archived summary differs from recomputation: {args.check}")
    print(serialized, end="")


if __name__ == "__main__":
    main()
