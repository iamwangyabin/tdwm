#!/usr/bin/env python3
"""Train released TD-JEPA on the locally converted OGBench-Cube data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tdwm.training.td_jepa import train_tdjepa_cube


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train released TD-JEPA on Cube Lance-derived episode buffers."
    )
    parser.add_argument("--implementation-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3917)
    parser.add_argument("--train-steps", type=int, default=1_000_000)
    parser.add_argument("--load-episodes", type=int, default=10_000)
    parser.add_argument("--loader-workers", type=int, default=8)
    parser.add_argument("--log-every-updates", type=int, default=1_000)
    parser.add_argument("--checkpoint-every-steps", type=int, default=50_000)
    parser.add_argument("--no-compile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_tdjepa_cube(
        implementation_root=args.implementation_root,
        data_root=args.data_root,
        work_dir=args.work_dir,
        seed=args.seed,
        train_steps=args.train_steps,
        load_episodes=args.load_episodes,
        loader_workers=args.loader_workers,
        log_every_updates=args.log_every_updates,
        checkpoint_every_steps=args.checkpoint_every_steps,
        compile_model=not args.no_compile,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
