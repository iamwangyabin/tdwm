#!/usr/bin/env python3
"""Convert the existing Cube Lance data to TD-JEPA's public OGBench buffer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tdwm.adapters.td_jepa import convert_cube_lance_to_tdjepa_buffer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a TD-JEPA OGBench episode buffer from Cube Lance data."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-interval", type=int, default=100)
    parser.add_argument("--implementation-revision", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = convert_cube_lance_to_tdjepa_buffer(
        args.source,
        args.output_root,
        image_size=args.image_size,
        device=args.device,
        episodes=args.episodes,
        resume=args.resume,
        report_interval=args.report_interval,
        implementation_revision=args.implementation_revision,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
