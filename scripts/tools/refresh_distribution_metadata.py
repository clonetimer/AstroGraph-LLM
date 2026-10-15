"""Refresh the governed source inventory without including local VCS/cache state."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EPHEMERAL_PARTS = {"node_modules", ".git", ".claude", ".pytest_cache", "__pycache__", ".tmp_tests", "worktrees", ".venv", "astrograph_core.egg-info"}
EXCLUDED_PREFIXES = {"scripts/integration/out", "runs"}
CONTENT_EXCLUDED = {"FILELIST.txt", "SHA256SUMS.txt", "SOURCE_MANIFEST.json"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def _git_visible_files() -> list[Path] | None:
    if not (ROOT / ".git").exists():
        return None
    result = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return [
        ROOT / item.decode("utf-8", errors="strict")
        for item in result.stdout.split(b"\0")
        if item
    ]


def distribution_files() -> list[Path]:
    git_visible = _git_visible_files()
    candidates = git_visible if git_visible is not None else ROOT.rglob("*")
    return sorted(
        (
            path
            for path in candidates
            if path.is_file()
            and not EPHEMERAL_PARTS.intersection(path.relative_to(ROOT).parts)
            and path.suffix not in EXCLUDED_SUFFIXES
            and not any(
                path.relative_to(ROOT).as_posix() == prefix
                or path.relative_to(ROOT).as_posix().startswith(prefix + "/")
                for prefix in EXCLUDED_PREFIXES
            )
        ),
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    files = distribution_files()
    content = [path for path in files if relative(path) not in CONTENT_EXCLUDED]
    (ROOT / "FILELIST.txt").write_text(
        "\n".join(relative(path) for path in content) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    manifest_path = ROOT / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    release = json.loads((ROOT / "release.json").read_text(encoding="utf-8"))
    manifest["version"] = release.get("version", manifest.get("version"))
    manifest["status"] = release.get("status", manifest.get("status"))
    manifest["architecture_convergence"] = release.get("architecture_convergence", {})
    manifest["content_file_count"] = len(content)
    manifest["content_total_bytes"] = sum(path.stat().st_size for path in content)
    manifest["checksum_entry_count"] = len(distribution_files()) - 1
    manifest["generated"] = date.today().isoformat()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    checksum_files = [path for path in distribution_files() if path.name != "SHA256SUMS.txt"]
    (ROOT / "SHA256SUMS.txt").write_text(
        "\n".join(f"{sha256(path)}  {relative(path)}" for path in checksum_files) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
