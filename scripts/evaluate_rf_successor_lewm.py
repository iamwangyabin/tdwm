#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tdwm.evaluation.rf_successor_lewm import evaluate_rf_successor_lewm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RF-Successor-LeWM through policy-free CEM/MPC."
    )
    parser.add_argument(
        "--config",
        default="configs/experiment/rf_successor_lewm_cube_checkpoint_o50.yaml",
    )
    parser.add_argument(
        "--dataset",
        default=os.environ.get("TDWM_CUBE_DATASET"),
        help="Path to the audited Cube HDF5 or Lance dataset.",
    )
    parser.add_argument("--base-checkpoint-path", required=True)
    parser.add_argument("--successor-checkpoint-path", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--video", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--pilot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset:
        raise SystemExit("Pass --dataset or set TDWM_CUBE_DATASET.")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            Path(os.environ.get("TDWM_RUN_ROOT", "outputs"))
            / "rf_successor_lewm_cube_o50"
        )
    result = evaluate_rf_successor_lewm(
        protocol_path=args.config,
        dataset_path=args.dataset,
        output_dir=output_dir,
        base_checkpoint_path=args.base_checkpoint_path,
        successor_checkpoint_path=args.successor_checkpoint_path,
        video=args.video,
        smoke=args.smoke,
        pilot=args.pilot,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
