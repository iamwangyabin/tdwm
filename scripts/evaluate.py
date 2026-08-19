#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tdwm.evaluation import evaluate_official_lewm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an official LeWM checkpoint with Stable World Model."
    )
    parser.add_argument(
        "--config",
        default="configs/experiment/lewm_cube_checkpoint_o25.yaml",
        help="Locked experiment protocol YAML.",
    )
    parser.add_argument(
        "--dataset",
        default=os.environ.get("TDWM_CUBE_DATASET"),
        help="Path to cube_single_expert.h5 (or set TDWM_CUBE_DATASET).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Run directory. Defaults below TDWM_RUN_ROOT or outputs/.",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--checkpoint-path",
        help="Local Stable World Model export directory (for example, epoch_10).",
    )
    parser.add_argument("--video", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one episode with 8 candidates and 1 CEM iteration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset:
        raise SystemExit("Pass --dataset or set TDWM_CUBE_DATASET.")
    output_dir = args.output_dir
    if output_dir is None:
        run_root = Path(os.environ.get("TDWM_RUN_ROOT", "outputs"))
        output_dir = run_root / "lewm_cube_official_checkpoint_o25"
    result = evaluate_official_lewm(
        protocol_path=args.config,
        dataset_path=args.dataset,
        output_dir=output_dir,
        checkpoint_name=args.checkpoint,
        checkpoint_path=args.checkpoint_path,
        video=args.video,
        smoke=args.smoke,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
