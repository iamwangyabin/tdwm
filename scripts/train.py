#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tdwm.training import train_lewm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train original LeWM on OGBench-Cube with a locked protocol."
    )
    parser.add_argument(
        "--config",
        default="configs/experiment/lewm_cube_train.yaml",
        help="Locked training protocol YAML.",
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
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Stop after this many optimizer steps; use only for a controlled throughput run.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation; use only for a controlled throughput run.",
    )
    parser.add_argument("--loader-workers", type=int)
    parser.add_argument("--loader-prefetch-factor", type=int)
    parser.add_argument("--validation-loader-workers", type=int)
    parser.add_argument(
        "--stride-aware-lance",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Fetch only frames consumed by the LeWM stride when reading Lance; "
            "enabled by the locked Cube protocol."
        ),
    )
    parser.add_argument(
        "--device-image-preprocessing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Transfer compact uint8 images and apply the released normalization "
            "and resize on the accelerator."
        ),
    )
    parser.add_argument(
        "--compile-model",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Compile LeWM's encode and predict methods with torch.compile; "
            "enabled by the locked protocol after a controlled Cube A/B."
        ),
    )
    parser.add_argument(
        "--run-label",
        help="Append a label to the seed output directory for a separate controlled run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset:
        raise SystemExit("Pass --dataset or set TDWM_CUBE_DATASET.")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(os.environ.get("TDWM_RUN_ROOT", "outputs")) / "lewm_cube_training"
    result = train_lewm(
        protocol_path=args.config,
        dataset_path=args.dataset,
        output_dir=output_dir,
        seed=args.seed,
        smoke=args.smoke,
        resume=args.resume,
        max_steps=args.max_steps,
        skip_validation=args.skip_validation,
        loader_workers=args.loader_workers,
        loader_prefetch_factor=args.loader_prefetch_factor,
        validation_loader_workers=args.validation_loader_workers,
        stride_aware_lance=args.stride_aware_lance,
        device_image_preprocessing=args.device_image_preprocessing,
        compile_model=args.compile_model,
        run_label=args.run_label,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
