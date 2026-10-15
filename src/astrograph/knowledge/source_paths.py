"""Relocate a source-bound artifact without weakening its checksum requirement."""
from __future__ import annotations
import hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_bound_source(recorded_path: str, expected_sha256: str, *, anchor: Path) -> Path:
    """Use the recorded path, or a unique verified suffix inside the local project.

    Historical construction bundles contain author-machine absolute paths. Their
    identity excludes location, but includes content hashes. Never rewrite the
    signed bundle, search the filesystem globally, or accept a changed source.
    """
    recorded = Path(recorded_path)
    if recorded.is_absolute() and recorded.is_file():
        if _digest(recorded) != expected_sha256:
            raise ValueError("bound source file changed")
        return recorded
    root = next((p for p in [anchor.resolve().parent, *anchor.resolve().parents]
                 if (p / "pyproject.toml").is_file() and (p / "src" / "astrograph").is_dir()), None)
    if root is None:
        raise FileNotFoundError(f"bound source unavailable and no project root: {recorded_path}")
    pure = PureWindowsPath(recorded_path) if "\\" in recorded_path else PurePosixPath(recorded_path)
    if ".." in pure.parts:
        raise ValueError("bound source path traversal is forbidden")
    matches = set()
    for index in range(len(pure.parts)):
        suffix = Path(*pure.parts[index:])
        if suffix.is_absolute():
            continue
        candidate = (root / suffix).resolve()
        if not candidate.is_relative_to(root):
            continue
        if candidate.is_file() and _digest(candidate) == expected_sha256:
            matches.add(candidate)
    if len(matches) != 1:
        raise ValueError("bound source relocation requires one checksum-matching local file")
    return matches.pop()
