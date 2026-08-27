#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tdwm.training.clean_aligned_lewm import VARIANTS, train_clean_aligned_lewm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one causally controlled Clean Aligned LeWM variant."
    )
    parser.add_argument(
        "--config",
        default="configs/experiment/clean_aligned_lewm_cube_train.yaml",
    )
    parser.add_argument("--dataset", default=os.environ.get("TDWM_CUBE_DATASET"))
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", default=None)
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
            / "clean_aligned_lewm_cube_full"
        )
    result = train_clean_aligned_lewm(
        protocol_path=args.config,
        dataset_path=args.dataset,
        output_dir=output_dir,
        variant=args.variant,
        seed=args.seed,
        smoke=args.smoke,
        resume=args.resume,
        max_steps=args.max_steps,
        skip_validation=args.skip_validation,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
