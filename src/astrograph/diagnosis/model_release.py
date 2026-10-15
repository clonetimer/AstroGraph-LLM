from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def model_source_fingerprint(path: str | Path) -> tuple[str, int, int, str, int, int]:
    """Return a cache key that changes when a manifest or its artifact changes."""
    source = Path(path).resolve()
    source_stat = source.stat()
    artifact = source
    if source.suffix.lower() == ".json":
        try:
            manifest = json.loads(source.read_text(encoding="utf-8"))
            model_file = manifest.get("model_file") if isinstance(manifest, dict) else None
            if isinstance(model_file, str):
                artifact = (source.parent / model_file).resolve()
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
    try:
        artifact_stat = artifact.stat()
        artifact_size = artifact_stat.st_size
        artifact_mtime_ns = artifact_stat.st_mtime_ns
    except OSError:
        # Formal resolution below owns validation and error semantics.
        artifact_size = -1
        artifact_mtime_ns = -1
    return (
        str(source),
        source_stat.st_size,
        source_stat.st_mtime_ns,
        str(artifact),
        artifact_size,
        artifact_mtime_ns,
    )


def resolve_release_artifact(
    path: str | Path,
    *,
    manifest_schema: str,
    artifact_schema: str,
) -> tuple[Path, dict[str, Any] | None]:
    """Resolve a model artifact, validating an optional colocated release manifest."""
    source = Path(path).resolve()
    if source.suffix.lower() != ".json":
        allow_unsafe = os.environ.get("ASTROGRAPH_ALLOW_UNMANIFESTED_JOBLIB", "").strip().lower() in {
            "1", "true", "yes"
        }
        if source.suffix.lower() == ".joblib" and allow_unsafe:
            if not source.is_file():
                raise ValueError(f"model artifact is missing: {source}")
            return source, None
        raise ValueError(
            "model release manifest JSON is required; raw joblib loading is disabled. "
            "Set ASTROGRAPH_ALLOW_UNMANIFESTED_JOBLIB=1 only for explicitly trusted local research assets."
        )

    try:
        manifest = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid model release manifest: {source}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != manifest_schema:
        raise ValueError(f"unsupported model release manifest schema: {source}")

    model_file = manifest.get("model_file")
    expected_sha256 = manifest.get("model_sha256")
    if not isinstance(model_file, str) or not model_file.strip():
        raise ValueError("model release manifest requires model_file")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(char not in "0123456789abcdef" for char in expected_sha256.lower())
    ):
        raise ValueError("model release manifest requires a valid model_sha256")

    release_root = source.parent.resolve()
    artifact = (release_root / model_file).resolve()
    if artifact.parent != release_root:
        raise ValueError("model_file must name a file in the release manifest directory")
    if artifact.suffix.lower() != ".joblib" or not artifact.is_file():
        raise ValueError(f"model release artifact is missing or not a joblib file: {artifact}")
    actual_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256.lower():
        raise ValueError(
            f"model release artifact hash mismatch: expected {expected_sha256.lower()}, got {actual_sha256}"
        )

    manifest["resolved_artifact_schema"] = artifact_schema
    return artifact, manifest
