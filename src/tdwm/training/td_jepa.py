"""Launch configuration for the released TD-JEPA Cube baseline."""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import Any


def apply_tdjepa_cube_overrides(
    base_config: dict[str, Any],
    *,
    data_root: str | Path,
    work_dir: str | Path,
    seed: int,
    train_steps: int,
    load_episodes: int,
    loader_workers: int,
    log_every_updates: int,
    checkpoint_every_steps: int,
    compile_model: bool,
    implementation_revision: str,
) -> dict[str, Any]:
    """Apply the first released Cube sweep setting to the requested local data."""

    if min(
        train_steps,
        load_episodes,
        loader_workers,
        log_every_updates,
        checkpoint_every_steps,
    ) <= 0:
        raise ValueError("TD-JEPA training counts must be positive.")

    config = copy.deepcopy(base_config)
    config.update(
        {
            "work_dir": str(Path(work_dir).expanduser().resolve()),
            "seed": seed,
            "num_train_steps": train_steps,
            "log_every_updates": log_every_updates,
            "checkpoint_every_steps": checkpoint_every_steps,
            "eval_every_steps": checkpoint_every_steps,
            "use_wandb": False,
            "evaluations": [],
            "tags": {
                "baseline": "TD-JEPAFlowBC",
                "td_jepa_revision": implementation_revision,
                "dataset_root": str(Path(data_root).expanduser().resolve()),
                "protocol": "tdwm-cube-lance64-v1",
            },
        }
    )
    config["data"].update(
        {
            "dataset_root": str(Path(data_root).expanduser().resolve()),
            "load_n_episodes": load_episodes,
            "num_workers": loader_workers,
        }
    )
    config["agent"]["compile"] = compile_model
    # This is the first Cube configuration in TD-JEPA's released sweep.
    config["agent"]["train"].update(
        {
            "bc_coeff": 3.0,
            "psi_ortho_coef": 1.0,
            "lr_psi": 1.0e-4,
        }
    )
    return config


def train_tdjepa_cube(
    *,
    implementation_root: str | Path,
    data_root: str | Path,
    work_dir: str | Path,
    seed: int = 3917,
    train_steps: int = 1_000_000,
    load_episodes: int = 10_000,
    loader_workers: int = 8,
    log_every_updates: int = 1_000,
    checkpoint_every_steps: int = 50_000,
    compile_model: bool = True,
) -> dict[str, Any]:
    """Train TD-JEPA using the official source tree without modifying it."""

    implementation_root = Path(implementation_root).expanduser().resolve()
    data_root = Path(data_root).expanduser().resolve()
    work_dir = Path(work_dir).expanduser().resolve()
    revision_file = implementation_root / "UPSTREAM_REVISION"
    if not revision_file.is_file():
        raise FileNotFoundError(
            "TD-JEPA source must include the audited UPSTREAM_REVISION record."
        )
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)

    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, str(implementation_root))
    from scripts.train.pixel.launch_td_jepa_ogbench import BASE_CFG
    from train import TrainConfig

    implementation_revision = next(
        (
            line.split("=", 1)[1]
            for line in revision_file.read_text().splitlines()
            if line.startswith("commit=")
        ),
        "unknown",
    )
    raw_config = apply_tdjepa_cube_overrides(
        BASE_CFG,
        data_root=data_root,
        work_dir=work_dir,
        seed=seed,
        train_steps=train_steps,
        load_episodes=load_episodes,
        loader_workers=loader_workers,
        log_every_updates=log_every_updates,
        checkpoint_every_steps=checkpoint_every_steps,
        compile_model=compile_model,
        implementation_revision=implementation_revision,
    )
    config = TrainConfig(**raw_config)
    config.build().train()
    return {
        "work_dir": str(work_dir),
        "config_path": str(work_dir / "config.json"),
        "implementation_revision": implementation_revision,
    }
