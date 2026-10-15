from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

from astrograph.governance import build_tree_manifest


ROOT = Path(__file__).resolve().parents[1]


def _load_refresh_module():
    path = ROOT / "scripts/tools/refresh_distribution_metadata.py"
    spec = importlib.util.spec_from_file_location("refresh_distribution_metadata", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_distribution_inventory_uses_git_visibility_and_excludes_logs(tmp_path, monkeypatch) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("ignored.txt\n*.log\n", encoding="utf-8")
    (tmp_path / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    (tmp_path / "untracked.txt").write_text("candidate\n", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("ignored\n", encoding="utf-8")
    (tmp_path / "runtime.log").write_text("runtime\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", ".gitignore", "tracked.txt"], check=True
    )

    module = _load_refresh_module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    visible = {path.relative_to(tmp_path).as_posix() for path in module.distribution_files()}

    assert ".gitignore" in visible
    assert "tracked.txt" in visible
    assert "untracked.txt" in visible
    assert "ignored.txt" not in visible
    assert "runtime.log" not in visible


def test_governance_tree_excludes_ignored_runtime_files_but_keeps_tracked_models(tmp_path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("*.pt\nruntime/\n", encoding="utf-8")
    (tmp_path / "tracked_model.pt").write_text("tracked\n", encoding="utf-8")
    (tmp_path / "ignored_model.pt").write_text("ignored\n", encoding="utf-8")
    (tmp_path / "candidate.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime/state.sqlite3").write_text("runtime\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", ".gitignore", "candidate.py"], check=True
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "-f", "tracked_model.pt"], check=True
    )

    manifest = build_tree_manifest(tmp_path)

    assert "tracked_model.pt" in manifest
    assert "candidate.py" in manifest
    assert "ignored_model.pt" not in manifest
    assert "runtime/state.sqlite3" not in manifest


def test_root_data_ignore_rule_does_not_hide_python_data_package() -> None:
    ignore_lines = {
        line.strip()
        for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert "/data/" in ignore_lines
    assert "data/" not in ignore_lines
    assert "/data/diagnosis_runs/" in ignore_lines
    assert "/.astrograph-state/" in ignore_lines
