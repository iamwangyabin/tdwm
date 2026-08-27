#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.metadata
import subprocess
from pathlib import Path

import stable_worldmodel as swm

EXPECTED_VERSION = "0.1.1"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test the pinned Stable World Model installation."
    )
    parser.add_argument("--env", default="swm/PushT-v1")
    parser.add_argument("--dataset")
    args = parser.parse_args()

    version = importlib.metadata.version("stable-worldmodel")
    if version != EXPECTED_VERSION:
        raise RuntimeError(
            f"Expected stable-worldmodel=={EXPECTED_VERSION}, found {version}"
        )
    print(f"stable-worldmodel={version}")
    print(f"stable_worldmodel import={Path(swm.__file__).resolve()}")

    for command in ("envs", "datasets", "checkpoints"):
        subprocess.run(["swm", command], check=True)

    world = swm.World(args.env, num_envs=1, image_shape=(64, 64))
    try:
        world.reset(seed=0)
        print(f"world={args.env} pixels={world.infos['pixels'].shape}")
    finally:
        world.close()

    if args.dataset:
        dataset = swm.data.load_dataset(args.dataset, num_steps=1)
        print(
            f"dataset={Path(args.dataset).resolve()} "
            f"episodes={len(dataset.lengths)} clips={len(dataset)}"
        )


if __name__ == "__main__":
    main()
