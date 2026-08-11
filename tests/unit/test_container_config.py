from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _container_constraints() -> dict[str, str]:
    constraints: dict[str, str] = {}
    for raw_line in (ROOT / "docker" / "constraints.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==", maxsplit=1)
        constraints[name] = version
    return constraints


def test_container_preserves_base_image_cuda_stack() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    constraints = _container_constraints()

    assert constraints["stable-worldmodel"] == "0.1.1"
    assert constraints["stable-pretraining"] == "0.1.7"
    assert "torch" not in constraints
    assert "torchvision" not in constraints
    assert "torchaudio" not in constraints
    assert constraints["lightning"] == "2.4.0"

    assert "ARG BASE_IMAGE" in dockerfile
    assert "FROM ${BASE_IMAGE}" in dockerfile
    assert "tdwm-base-torch.json" in dockerfile
    assert "pytorch/pytorch:" not in dockerfile
    assert '"stable-worldmodel[all]==0.1.1"' in dockerfile
    assert dockerfile.count("python -m pip check") == 2


def test_project_keeps_single_upstream_dependency() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    training_sources = "\n".join(
        (ROOT / "src" / "tdwm" / "training" / name).read_text(
            encoding="utf-8"
        )
        for name in ("lewm.py", "baselines.py")
    )

    assert '"stable-worldmodel[all]==0.1.1"' in pyproject
    assert "stable-pretraining" not in pyproject
    assert "requeue_checkpoint_every_n_steps" not in training_sources
