#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tdwm.training.td_gt_lewm import train_td_gt_lewm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fully train TD-GT-LeWM on frozen Cube latents."
    )
    parser.add_argument(
        "--config",
        default="configs/experiment/td_gt_lewm_cube_train.yaml",
    )
    parser.add_argument(
        "--dataset",
        default=os.environ.get("TDWM_CUBE_DATASET"),
        help="Path to the audited Cube Lance dataset.",
    )
    parser.add_argument("--base-checkpoint-path", required=True)
    parser.add_argument("--normalization-stats", required=True)
    parser.add_argument(
        "--latent-cache-dir",
        default=None,
        help="Optional audited MC-GT latent cache directory to reuse.",
    )
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run two training steps and a small checkpoint/validation pass.",
    )
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
            / "td_gt_lewm_cube_full"
        )
    result = train_td_gt_lewm(
        protocol_path=args.config,
        dataset_path=args.dataset,
        base_checkpoint_path=args.base_checkpoint_path,
        normalization_stats_path=args.normalization_stats,
        output_dir=output_dir,
        seed=args.seed,
        latent_cache_dir=args.latent_cache_dir,
        smoke=args.smoke,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
