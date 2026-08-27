from __future__ import annotations

import hashlib
import json
import string
import subprocess
from pathlib import Path
from typing import Any


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_state(root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    state: dict[str, Any] = {"commit": commit, "dirty": bool(status.strip())}
    if state["dirty"]:
        state["status_sha256"] = hashlib.sha256(status.encode("utf-8")).hexdigest()
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        state["diff_sha256"] = hashlib.sha256(diff).hexdigest()
        untracked_output = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        untracked: list[tuple[str, str]] = []
        for raw_path in untracked_output.split(b"\0"):
            if not raw_path:
                continue
            relative_path = raw_path.decode("utf-8", errors="surrogateescape")
            path = root / relative_path
            if path.is_file():
                untracked.append((relative_path, _file_sha256(path)))
        state["untracked_sha256"] = canonical_hash(untracked)
    return state


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_signature(path: Path, sha256: str | None = None) -> dict[str, Any]:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    signature: dict[str, Any] = {
        "path": str(path),
        "kind": "directory" if path.is_dir() else "file",
    }
    expected_sha256 = sha256.lower() if sha256 else None
    if expected_sha256 and (
        len(expected_sha256) != 64
        or any(character not in string.hexdigits for character in expected_sha256)
    ):
        raise ValueError("dataset sha256 must contain exactly 64 hexadecimal digits")

    if path.is_file():
        stat = path.stat()
        signature.update({"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        if expected_sha256:
            actual_sha256 = _file_sha256(path)
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"Dataset sha256 mismatch for {path}: expected "
                    f"{expected_sha256}, found {actual_sha256}"
                )
            signature["sha256"] = actual_sha256
        return signature

    entries: list[tuple[str, int, int]] = []
    content_entries: list[tuple[str, str]] = []
    for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        stat = child.stat()
        relative_path = str(child.relative_to(path))
        entries.append((relative_path, stat.st_size, stat.st_mtime_ns))
        if expected_sha256:
            content_entries.append((relative_path, _file_sha256(child)))
    signature.update(
        {
            "file_count": len(entries),
            "size_bytes": sum(size for _, size, _ in entries),
            "manifest_sha256": canonical_hash(entries),
        }
    )
    if expected_sha256:
        actual_sha256 = canonical_hash(content_entries)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Dataset sha256 mismatch for {path}: expected "
                f"{expected_sha256}, found {actual_sha256}"
            )
        signature["sha256"] = actual_sha256
    return signature


def prepare_run_directory(
    *,
    run_root: Path,
    requested_run_id: str | None,
    method: str,
    environment: str,
    seed: int,
    identity: dict[str, Any],
) -> tuple[str, Path, str]:
    fingerprint = canonical_hash(identity)
    run_id = requested_run_id or (
        f"{method}_{environment}_seed{seed}_{fingerprint[:12]}"
    )
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run ID must be one non-empty path component")
    run_dir = run_root / "runs" / run_id
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("experiment_fingerprint") != fingerprint:
            raise RuntimeError(
                f"Run directory {run_dir} belongs to a different experiment; "
                "choose another --run-id or remove the stale run outside TDWM."
            )
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(
            f"Run directory {run_dir} is non-empty but has no metadata; "
            "choose another --run-id or inspect the directory manually."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir, fingerprint
