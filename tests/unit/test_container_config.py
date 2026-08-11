from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _container_constraints() -> dict[str, str]:
    constraints: dict[str, str] = {}
    for raw_line in (ROOT / "docker" / "constraints-cu121.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==", maxsplit=1)
        constraints[name] = version
    return constraints


def test_container_uses_verified_cuda_stack() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    constraints = _container_constraints()

    assert constraints["stable-worldmodel"] == "0.1.1"
    assert constraints["stable-pretraining"] == "0.1.8"
    assert constraints["torch"] == "2.4.1"
    assert constraints["torchvision"] == "0.19.1"
    assert constraints["torchaudio"] == "2.4.1"
    assert constraints["lightning"] == "2.4.0"

    assert (
        "pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime@sha256:" in dockerfile
    )
    assert '"stable-worldmodel[all]==0.1.1"' in dockerfile
    assert dockerfile.count("python -m pip check") == 2


def test_project_keeps_single_upstream_dependency() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert '"stable-worldmodel[all]==0.1.1"' in pyproject
    assert "stable-pretraining" not in pyproject
