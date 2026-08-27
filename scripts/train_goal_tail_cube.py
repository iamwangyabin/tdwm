#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tdwm.training.train_goal_tail_cube import train_goal_tail_cube


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train supervised GoalTailValue on frozen LeWM Cube latents."
    )
    parser.add_argument(
        "--config",
        default="configs/experiment/goal_tail_value_cube_train.yaml",
    )
    parser.add_argument(
        "--dataset",
        default=os.environ.get("TDWM_CUBE_DATASET"),
        help="Path to the locked Cube HDF5 or audited Lance dataset.",
    )
    parser.add_argument(
        "--base-checkpoint-path",
        required=True,
        help="Frozen LeWM export in <cache>/checkpoints/<run> layout.",
    )
    parser.add_argument(
        "--normalization-stats",
        help="Optional LeWM column_normalization.json to reuse exactly.",
    )
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults below TDWM_RUN_ROOT or outputs/.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-steps", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset:
        raise SystemExit("Pass --dataset or set TDWM_CUBE_DATASET.")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            Path(os.environ.get("TDWM_RUN_ROOT", "outputs"))
            / "goal_tail_value_cube_v0_1"
        )
    result = train_goal_tail_cube(
        protocol_path=args.config,
        dataset_path=args.dataset,
        base_checkpoint_path=args.base_checkpoint_path,
        output_dir=output_dir,
        seed=args.seed,
        normalization_stats_path=args.normalization_stats,
        smoke=args.smoke,
        max_steps=args.max_steps,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
