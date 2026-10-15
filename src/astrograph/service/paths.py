from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path


def _project_key(project_root: Path) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", project_root.name).strip("-.") or "project"
    digest = hashlib.sha256(str(project_root).encode("utf-8")).hexdigest()[:12]
    return f"{slug}-{digest}"


def _default_state_base() -> Path:
    if value := os.environ.get("XDG_STATE_HOME", "").strip():
        return Path(value).expanduser()
    if value := os.environ.get("LOCALAPPDATA", "").strip():
        return Path(value).expanduser()
    return Path.home() / ".local" / "state"


def _resolve_external_path(value: str | Path, *, project_root: Path, kind: str) -> Path:
    resolved = Path(value).expanduser().resolve()
    allow_inside = os.environ.get("ASTROGRAPH_ALLOW_PROJECT_STATE", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if not allow_inside and (resolved == project_root or project_root in resolved.parents):
        raise ValueError(
            f"{kind} must be outside project_root; set ASTROGRAPH_ALLOW_PROJECT_STATE=1 "
            "only for an explicit legacy migration"
        )
    return resolved


@dataclass(frozen=True)
class ServicePathLayout:
    """Resolved immutable package root and mutable service roots.

    Mutable state is rejected when it is located inside the source tree unless
    the explicit legacy-migration override is enabled.  This keeps source
    distributions reproducible and supports read-only installations.
    """

    project_root: Path
    state_root: Path
    artifact_root: Path

    @classmethod
    def resolve(
        cls,
        project_root: str | Path,
        *,
        state_dir: str | Path | None = None,
        artifact_dir: str | Path | None = None,
    ) -> "ServicePathLayout":
        root = Path(project_root).expanduser().resolve()

        state_value: str | Path
        if state_dir is not None:
            state_value = state_dir
        elif configured := os.environ.get("ASTROGRAPH_STATE_DIR", "").strip():
            state_value = configured
        else:
            state_value = _default_state_base() / "astrograph" / _project_key(root)
        state_root = _resolve_external_path(
            state_value,
            project_root=root,
            kind="ASTROGRAPH_STATE_DIR",
        )

        artifact_value: str | Path
        if artifact_dir is not None:
            artifact_value = artifact_dir
        elif configured := os.environ.get("ASTROGRAPH_ARTIFACT_DIR", "").strip():
            artifact_value = configured
        else:
            artifact_value = state_root / "artifacts"
        artifact_root = _resolve_external_path(
            artifact_value,
            project_root=root,
            kind="ASTROGRAPH_ARTIFACT_DIR",
        )

        return cls(project_root=root, state_root=state_root, artifact_root=artifact_root)


__all__ = ["ServicePathLayout"]
