#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tdwm.training.local_successor import train_ls_lewm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train independent LS-LeWM without modifying the LeWM baseline."
    )
    parser.add_argument(
        "--config",
        default="configs/experiment/ls_lewm_cube_train.yaml",
        help="Locked LS-LeWM training protocol YAML.",
    )
    parser.add_argument(
        "--dataset",
        default=os.environ.get("TDWM_CUBE_DATASET"),
        help="Path to the audited Cube HDF5 or Lance dataset.",
    )
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Training root below TDWM_RUN_ROOT or outputs by default.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--resume", choices=("auto", "never", "required"), default="auto"
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset:
        raise SystemExit("Pass --dataset or set TDWM_CUBE_DATASET.")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            Path(os.environ.get("TDWM_RUN_ROOT", "outputs"))
            / "ls_lewm_cube_training"
        )
    result = train_ls_lewm(
        protocol_path=args.config,
        dataset_path=args.dataset,
        output_dir=output_dir,
        seed=args.seed,
        smoke=args.smoke,
        resume=args.resume,
        max_steps=args.max_steps,
        skip_validation=args.skip_validation,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
