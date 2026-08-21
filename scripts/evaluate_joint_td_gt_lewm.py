#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tdwm.evaluation.joint_td_gt_lewm import evaluate_joint_td_gt_lewm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Joint TD-GT-LeWM with the locked Cube CEM protocol."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--dataset",
        default=os.environ.get("TDWM_CUBE_DATASET"),
        help="Path to the audited Cube Lance dataset.",
    )
    parser.add_argument("--base-checkpoint-path", required=True)
    parser.add_argument("--joint-checkpoint-path", required=True)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults below TDWM_RUN_ROOT or outputs/.",
    )
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset:
        raise SystemExit("Pass --dataset or set TDWM_CUBE_DATASET.")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            Path(os.environ.get("TDWM_RUN_ROOT", "outputs"))
            / "joint_td_gt_lewm_cube_seed3072_o25"
        )
    result = evaluate_joint_td_gt_lewm(
        protocol_path=args.config,
        dataset_path=args.dataset,
        output_dir=output_dir,
        base_checkpoint_path=args.base_checkpoint_path,
        joint_checkpoint_path=args.joint_checkpoint_path,
        video=args.video,
        smoke=args.smoke,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
