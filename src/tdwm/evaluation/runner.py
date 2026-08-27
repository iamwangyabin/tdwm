from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import stable_worldmodel as swm
import torch
from sklearn.preprocessing import StandardScaler
from torchvision.transforms import v2

from tdwm.evaluation.protocol import (
    OFFICIAL_LEWM_SELECTION_PROTOCOL,
    select_evaluation_points,
)
from tdwm.training.experiment import (
    canonical_hash,
    dataset_signature,
    git_state,
)
from tdwm.training.lewm_pusht import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    _load_yaml,
    _repo_root,
    _verify_installed_platform,
    _write_json,
)

WORLD_MODEL_METHODS = frozenset({"lewm", "pldm", "dino_wm"})
POLICY_METHODS = frozenset({"gcbc", "gcivl", "gciql"})


def _image_transform(image_size: int) -> Any:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            v2.Resize((image_size, image_size)),
        ]
    )


def _processors(dataset: Any, keys: list[str]) -> dict[str, StandardScaler]:
    processors: dict[str, StandardScaler] = {}
    for key in keys:
        if key == "pixels" or key not in dataset.column_names:
            continue
        values = np.asarray(dataset.get_col_data(key))
        if values.ndim == 1:
            values = values[:, None]
        values = values.reshape(-1, values.shape[-1])
        values = values[~np.isnan(values).any(axis=1)]
        processor = StandardScaler().fit(values)
        processors[key] = processor
        if key != "action":
            processors[f"goal_{key}"] = processor
    return processors


def _checkpoint_signature(checkpoint: str, run_root: Path) -> dict[str, Any]:
    candidate = Path(checkpoint).expanduser()
    if not candidate.is_absolute():
        candidate = run_root / "checkpoints" / candidate
    if candidate.exists():
        return dataset_signature(candidate)
    return {"name": checkpoint}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def run(args: argparse.Namespace) -> Path:
    _verify_installed_platform()
    if args.method not in WORLD_MODEL_METHODS | POLICY_METHODS:
        raise ValueError(f"Unsupported evaluation method: {args.method}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")

    root = _repo_root()
    environment = _load_yaml(root / "configs" / "envs" / f"{args.env}.yaml")
    method = _load_yaml(root / "configs" / "methods" / f"{args.method}.yaml")
    evaluation = environment["evaluation"]
    planning = environment["planning"]
    episodes = (
        int(evaluation["episodes"])
        if args.episodes is None
        else int(args.episodes)
    )
    goal_offset = int(evaluation["goal_offset_steps"])
    budget = int(evaluation["budget_steps"])
    seed = args.seed if args.seed is not None else int(evaluation["seed"])
    if int(planning["horizon"]) * int(planning["action_block"]) > budget:
        raise ValueError("Planning horizon exceeds the evaluation budget")
    if args.planner_batch_size < 1:
        raise ValueError("planner batch size must be positive")

    dataset_path = Path(args.dataset).expanduser().resolve()
    run_root = Path(args.run_root).expanduser().resolve()
    dataset = swm.data.load_dataset(
        str(dataset_path),
        keys_to_cache=list(
            environment["dataset"]["evaluation_loader"].get(
                "keys_to_cache", []
            )
        ),
    )
    episode_ids, start_steps = select_evaluation_points(
        dataset.lengths,
        episodes=episodes,
        goal_offset=goal_offset,
        seed=seed,
    )

    model = swm.wm.load_pretrained(args.checkpoint, cache_dir=str(run_root))
    model = model.to(args.device).eval()
    model.requires_grad_(False)
    image_size = int(environment["world"]["image_size"])
    transform = {
        "pixels": _image_transform(image_size),
        "goal": _image_transform(image_size),
    }
    process = _processors(
        dataset,
        list(
            environment["dataset"]["evaluation_loader"].get(
                "keys_to_cache", []
            )
        ),
    )

    if args.method in WORLD_MODEL_METHODS:
        solver = swm.solver.CEMSolver(
            model=model,
            batch_size=args.planner_batch_size,
            num_samples=int(planning["candidates"]),
            var_scale=float(planning["initial_variance"]),
            n_steps=int(planning["iterations"]),
            topk=int(planning["elites"]),
            device=args.device,
            seed=seed,
        )
        policy = swm.policy.WorldModelPolicy(
            solver=solver,
            config=swm.PlanConfig(
                horizon=int(planning["horizon"]),
                receding_horizon=int(planning["receding_horizon"]),
                history_len=int(method["sequence"]["history_size"]),
                action_block=int(planning["action_block"]),
            ),
            process=process,
            transform=transform,
        )
    else:
        policy = swm.policy.FeedForwardPolicy(
            model=model,
            process=process,
            transform=transform,
        )

    world_config = environment["world"]
    world = swm.World(
        world_config["env_name"],
        num_envs=episodes,
        image_shape=(image_size, image_size),
        max_episode_steps=max(int(world_config["max_episode_steps"]), 2 * budget),
        render_mode="rgb_array",
        **world_config.get("kwargs", {}),
    )
    try:
        world.set_policy(policy)
        started = time.perf_counter()
        metrics = world.evaluate(
            dataset=dataset,
            episodes_idx=episode_ids,
            start_steps=start_steps,
            goal_offset=goal_offset,
            eval_budget=budget,
            callables=evaluation.get("callables"),
            video=args.video_dir,
        )
    finally:
        world.close()
    elapsed = time.perf_counter() - started

    identity = {
        "git": git_state(root),
        "stable_worldmodel_version": importlib.metadata.version(
            "stable-worldmodel"
        ),
        "environment_config": environment,
        "method_config": method,
        "checkpoint": _checkpoint_signature(args.checkpoint, run_root),
        "dataset": dataset_signature(dataset_path, args.dataset_sha256),
        "seed": seed,
        "selection_protocol": OFFICIAL_LEWM_SELECTION_PROTOCOL,
        "episode_ids": episode_ids,
        "start_steps": start_steps,
    }
    fingerprint = canonical_hash(identity)
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else run_root
        / "evaluations"
        / f"{args.method}_{args.env}_seed{seed}_{fingerprint[:12]}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        output,
        {
            "evaluation_fingerprint": fingerprint,
            "identity": identity,
            "checkpoint": args.checkpoint,
            "episodes": episode_ids,
            "start_steps": start_steps,
            "elapsed_seconds": elapsed,
            "metrics": _jsonable(metrics),
        },
    )
    print(json.dumps(_jsonable(metrics), indent=2, sort_keys=True))
    print(f"Evaluation record: {output}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one TDWM baseline with the shared dataset protocol."
    )
    parser.add_argument("--env", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-sha256")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--planner-batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output")
    parser.add_argument("--video-dir")
    return parser


def main() -> None:
    run(build_parser().parse_args())
