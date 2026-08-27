#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from tdwm.training.clean_aligned_lewm import compare_state_dicts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare every world-model parameter and buffer exactly."
    )
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_world_state(path: Path) -> dict[str, torch.Tensor]:
    payload: Any = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint is not a mapping: {path}")
    if "world_model_state_dict" in payload:
        return payload["world_model_state_dict"]
    state = payload.get("state_dict", payload)
    prefix = "model."
    selected = {
        name[len(prefix) :]: value
        for name, value in state.items()
        if name.startswith(prefix)
    }
    if not selected:
        raise ValueError(f"No world-model state found in {path}")
    return selected


def main() -> None:
    args = parse_args()
    result = {
        "left": str(args.left.resolve()),
        "right": str(args.right.resolve()),
        **compare_state_dicts(
            load_world_state(args.left),
            load_world_state(args.right),
        ),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    if not result["exact_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
