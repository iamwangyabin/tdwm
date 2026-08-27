#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tdwm.evaluation.gt_lewm import evaluate_gt_lewm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate standalone GT-LeWM without modifying the LeWM baseline."
    )
    parser.add_argument(
        "--config",
        default="configs/experiment/gt_lewm_cube_checkpoint_o50.yaml",
        help="Locked GT-LeWM evaluation protocol YAML.",
    )
    parser.add_argument(
        "--dataset",
        default=os.environ.get("TDWM_CUBE_DATASET"),
        help="Path to cube_single_expert.h5 or audited .lance data.",
    )
    parser.add_argument("--base-checkpoint-path", required=True)
    parser.add_argument("--value-checkpoint-path", required=True)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Run directory. Defaults below TDWM_RUN_ROOT or outputs/.",
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
            Path(os.environ.get("TDWM_RUN_ROOT", "outputs")) / "gt_lewm_cube_o50"
        )
    result = evaluate_gt_lewm(
        protocol_path=args.config,
        dataset_path=args.dataset,
        output_dir=output_dir,
        base_checkpoint_path=args.base_checkpoint_path,
        value_checkpoint_path=args.value_checkpoint_path,
        video=args.video,
        smoke=args.smoke,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
