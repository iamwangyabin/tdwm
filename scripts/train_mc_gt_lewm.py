#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tdwm.training.mc_gt_lewm import train_mc_gt_lewm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fully train MC-GT-LeWM on cached frozen Cube latents."
    )
    parser.add_argument(
        "--config",
        default="configs/experiment/mc_gt_lewm_cube_train.yaml",
    )
    parser.add_argument(
        "--dataset",
        default=os.environ.get("TDWM_CUBE_DATASET"),
        help="Path to the locked Cube HDF5 or audited Lance dataset.",
    )
    parser.add_argument("--base-checkpoint-path", required=True)
    parser.add_argument(
        "--normalization-stats",
        required=True,
        help="The reproduced LeWM column_normalization.json.",
    )
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults below TDWM_RUN_ROOT or outputs/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset:
        raise SystemExit("Pass --dataset or set TDWM_CUBE_DATASET.")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            Path(os.environ.get("TDWM_RUN_ROOT", "outputs"))
            / "mc_gt_lewm_cube_full"
        )
    result = train_mc_gt_lewm(
        protocol_path=args.config,
        dataset_path=args.dataset,
        base_checkpoint_path=args.base_checkpoint_path,
        normalization_stats_path=args.normalization_stats,
        output_dir=output_dir,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
