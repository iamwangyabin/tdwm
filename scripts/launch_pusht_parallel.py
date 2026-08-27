#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tdwm.training.experiment import (
    canonical_hash,
    dataset_signature,
    git_state,
)

DEFAULT_METHODS = ("pldm", "dino_wm", "gcbc", "gcivl", "gciql")


def _gpu_memory() -> dict[int, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    memory: dict[int, int] = {}
    for line in result.stdout.splitlines():
        index, used = line.split(",", maxsplit=1)
        memory[int(index.strip())] = int(used.strip())
    return memory


def _write_state(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as handle:
        state = json.load(handle)
    return set(state.get("completed", []))


def _externally_running_methods() -> set[str]:
    result = subprocess.run(
        ["pgrep", "-af", "scripts/train.py"],
        check=False,
        capture_output=True,
        text=True,
    )
    methods: set[str] = set()
    for line in result.stdout.splitlines():
        fields = line.split()
        if "--method" not in fields:
            continue
        index = fields.index("--method") + 1
        if index < len(fields):
            methods.add(fields[index])
    return methods


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Keep all free GPUs busy with single-seed PushT baselines."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-sha256")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--checkpoint-steps", type=int, default=1000)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--free-memory-threshold-mib", type=int, default=512)
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    unknown = set(args.methods) - set(DEFAULT_METHODS)
    if unknown:
        raise ValueError(f"Unsupported queued methods: {sorted(unknown)}")

    root = Path(__file__).resolve().parents[1]
    dataset = Path(args.dataset).expanduser().resolve()
    run_root = Path(args.run_root).expanduser().resolve()
    if not dataset.exists():
        raise FileNotFoundError(dataset)
    run_root.mkdir(parents=True, exist_ok=True)
    request_identity = {
        "git": git_state(root),
        "dataset": dataset_signature(dataset, args.dataset_sha256),
        "seed": args.seed,
        "methods": args.methods,
        "workers": args.workers,
        "checkpoint_steps": args.checkpoint_steps,
        "environment_config_sha256": canonical_hash(
            (root / "configs" / "envs" / "pusht.yaml").read_text(
                encoding="utf-8"
            )
        ),
        "method_config_sha256": {
            method: canonical_hash(
                (root / "configs" / "methods" / f"{method}.yaml").read_text(
                    encoding="utf-8"
                )
            )
            for method in args.methods
        },
    }
    request_fingerprint = canonical_hash(request_identity)
    launcher_dir = (
        run_root
        / "launcher"
        / f"pusht_parallel_seed{args.seed}_{request_fingerprint[:12]}"
    )
    launcher_dir.mkdir(parents=True, exist_ok=True)
    state_path = launcher_dir / "state.json"

    completed = _load_completed(state_path)
    failed: dict[str, int] = {}
    attempts = {method: 0 for method in args.methods}
    children: dict[str, tuple[subprocess.Popen[Any], int, Any]] = {}
    pending = [method for method in args.methods if method not in completed]

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment.setdefault("HF_HOME", str(run_root / "huggingface"))
    environment.setdefault(
        "STABLEWM_HOME", str(run_root / "stable_worldmodel")
    )

    while pending or children:
        for method, (process, gpu, log_handle) in list(children.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            log_handle.close()
            del children[method]
            if return_code == 0:
                completed.add(method)
            elif attempts[method] < args.max_attempts:
                pending.append(method)
            else:
                failed[method] = return_code

        external = _externally_running_methods() - set(children)
        memory = _gpu_memory()
        occupied = {gpu for _, gpu, _ in children.values()}
        free_gpus = [
            gpu
            for gpu in args.gpus
            if gpu not in occupied
            and memory.get(gpu, args.free_memory_threshold_mib + 1)
            < args.free_memory_threshold_mib
        ]

        for gpu in free_gpus:
            method = next(
                (candidate for candidate in pending if candidate not in external),
                None,
            )
            if method is None:
                break
            pending.remove(method)
            attempts[method] += 1
            log_path = launcher_dir / f"{method}.attempt{attempts[method]}.log"
            log_handle = log_path.open("a", encoding="utf-8")
            child_environment = environment.copy()
            child_environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            command = [
                sys.executable,
                "scripts/train.py",
                "--env",
                "pusht",
                "--method",
                method,
                "--seed",
                str(args.seed),
                "--dataset",
                str(dataset),
                "--run-root",
                str(run_root),
                "--workers",
                str(args.workers),
                "--checkpoint-steps",
                str(args.checkpoint_steps),
                "--resume",
            ]
            if args.dataset_sha256:
                command.extend(["--dataset-sha256", args.dataset_sha256])
            process = subprocess.Popen(
                command,
                cwd=root,
                env=child_environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            children[method] = (process, gpu, log_handle)

        _write_state(
            state_path,
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "request_fingerprint": request_fingerprint,
                "request_identity": request_identity,
                "seed": args.seed,
                "dataset": str(dataset),
                "run_root": str(run_root),
                "pending": pending,
                "running": {
                    method: {"pid": process.pid, "gpu": gpu}
                    for method, (process, gpu, _) in children.items()
                },
                "external": sorted(external),
                "completed": sorted(completed),
                "failed": failed,
                "attempts": attempts,
                "gpu_memory_mib": memory,
            },
        )
        if pending or children:
            time.sleep(args.poll_seconds)

    if failed:
        raise SystemExit(f"Methods failed after retries: {failed}")


if __name__ == "__main__":
    main()
