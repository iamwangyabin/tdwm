#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tdwm.training.gt_lewm import train_gt_lewm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train standalone GT-LeWM without modifying the LeWM baseline."
    )
    parser.add_argument(
        "--config",
        default="configs/experiment/gt_lewm_cube_train.yaml",
        help="Locked GT-LeWM training protocol YAML.",
    )
    parser.add_argument(
        "--dataset",
        default=os.environ.get("TDWM_CUBE_DATASET"),
        help="Path to locked Cube HDF5 or audited .lance data.",
    )
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Training root. Defaults below TDWM_RUN_ROOT or outputs/.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--resume", choices=("auto", "never", "required"), default="auto"
    )
    parser.add_argument("--max-steps", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset:
        raise SystemExit("Pass --dataset or set TDWM_CUBE_DATASET.")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(os.environ.get("TDWM_RUN_ROOT", "outputs")) / "gt_lewm_cube_training"
    result = train_gt_lewm(
        protocol_path=args.config,
        dataset_path=args.dataset,
        output_dir=output_dir,
        seed=args.seed,
        smoke=args.smoke,
        resume=args.resume,
        max_steps=args.max_steps,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
