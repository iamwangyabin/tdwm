#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from tdwm.training import train_lewm

PUSHT_METHODS = frozenset(
    {"lewm", "pldm", "dino_wm", "gcbc", "gcivl", "gciql", "tdmpc2"}
)


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
        "--block-shuffle",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Group training clips into locality-preserving blocks before "
            "shuffling; use only for a controlled Cube throughput comparison."
        ),
    )
    parser.add_argument(
        "--block-size",
        type=int,
        help="Clip count per locality block when --block-shuffle is enabled.",
    )
    parser.add_argument(
        "--block-prefetch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Read and JPEG-decode a local Lance block before shuffling its "
            "mini-batches; requires --block-shuffle and loader workers."
        ),
    )
    parser.add_argument(
        "--block-prefetch-size",
        type=int,
        help="Clip count staged in a worker's decoded local block cache.",
    )
    parser.add_argument(
        "--episode-streaming",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Sequentially prefetch compressed Lance episodes into one bounded "
            "cache while mixing episodes within each training batch."
        ),
    )
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
            "available only for a controlled Cube A/B."
        ),
    )
    parser.add_argument(
        "--run-label",
        help="Append a label to the seed output directory for a separate controlled run.",
    )
    return parser.parse_args()


def main() -> None:
    # Keep the established Cube CLI unchanged while exposing the PR's
    # dataset-backed PushT adapters through the same script entry point.
    if "--env" in sys.argv or "--method" in sys.argv:
        from tdwm.training.lewm_pusht import build_parser

        args = build_parser().parse_args()
        if args.method == "lewm":
            from tdwm.training.lewm_pusht import run

            run(args)
        elif args.method in PUSHT_METHODS - {"lewm", "tdmpc2"}:
            from tdwm.training.baselines import run

            run(args)
        elif args.method == "tdmpc2":
            raise RuntimeError(
                "TD-MPC2 remains protocol-gated: it requires reward/Q training "
                "and cannot be mixed into this offline PushT comparison."
            )
        else:
            raise ValueError(f"Unknown method: {args.method}")
        return

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
        block_shuffle=args.block_shuffle,
        block_size=args.block_size,
        block_prefetch=args.block_prefetch,
        block_prefetch_size=args.block_prefetch_size,
        episode_streaming=args.episode_streaming,
        stride_aware_lance=args.stride_aware_lance,
        device_image_preprocessing=args.device_image_preprocessing,
        compile_model=args.compile_model,
        run_label=args.run_label,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
