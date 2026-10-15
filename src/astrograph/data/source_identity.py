from __future__ import annotations

import hashlib
from pathlib import Path

SOURCE_SUFFIXES = {".py", ".json", ".yaml", ".yml"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_simulation_source_hash(source_root: str | Path) -> tuple[str, int]:
    """Reproduce the PHM campaign source-package identity algorithm."""
    root = Path(source_root).resolve()
    files: list[Path] = []
    for candidate in (root / "scripts", root / "src", root / "configs" / "phm"):
        if candidate.exists():
            files.extend(
                path for path in candidate.rglob("*")
                if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES
            )
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest(), len(files)


def validate_checksum_file(checksum_file: str | Path) -> dict:
    checksum_path = Path(checksum_file)
    matched: list[str] = []
    missing: list[str] = []
    mismatched: list[dict] = []
    for line in checksum_path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(None, 1)
        relative = relative.strip().lstrip("*").replace("\\", "/")
        path = checksum_path.parent / relative
        if not path.exists():
            missing.append(relative)
            continue
        actual = sha256_file(path)
        if actual == expected:
            matched.append(relative)
        else:
            mismatched.append({"path": relative, "expected": expected, "actual": actual})
    return {
        "listed": len(matched) + len(missing) + len(mismatched),
        "matched": len(matched),
        "missing": missing,
        "mismatched": mismatched,
    }
