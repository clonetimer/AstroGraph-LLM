from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_post49_document_freshness_checker_passes() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/docs/check_document_freshness.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_current_index_excludes_history_and_research() -> None:
    current = json.loads((ROOT / "docs/CURRENT_DOCS.json").read_text(encoding="utf-8"))
    paths = current["default_current_corpus"]
    assert paths
    assert all(not p.startswith("docs/history/") for p in paths)
    assert all(not p.startswith("research/") for p in paths)
    assert "governance/implementation/post49/RELEASE_READINESS.json" == current["authoritative_order"][0]


def test_history_is_centralized_and_content_addressed() -> None:
    delivery = json.loads((ROOT / "DELIVERY_PROFILE.json").read_text(encoding="utf-8"))
    if not delivery["historical_records_included"]:
        assert delivery["source_kind"] == "CURRENT_SOURCE_ONLY"
        assert not (ROOT / "docs/history").exists()
        assert "docs/history/" in delivery["excluded_history_roots"]
        return
    history = json.loads((ROOT / "docs/history/MANIFEST.json").read_text(encoding="utf-8"))
    assert history["claim_precedence"] == "HISTORY_NEVER_OVERRIDES_CURRENT_RELEASE_READINESS"
    assert history["file_count"] >= 100
    assert all(item["path"].startswith("docs/history/") for item in history["records"])
    # Representative records that previously appeared as current docs must be archived.
    archived = {item.get("original_path"): item["path"] for item in history["records"] if item.get("original_path")}
    assert archived["docs/ADCS_PRODUCTION_QUALIFICATION_20260827.md"].startswith("docs/history/")
    assert archived["docs/implementation/POST48_DEV23_ADCS_METRIC_PROTOCOL.md"].startswith("docs/history/")


def test_release_metadata_points_to_current_document_policy() -> None:
    release = json.loads((ROOT / "release.json").read_text(encoding="utf-8"))
    delivery = json.loads((ROOT / "DELIVERY_PROFILE.json").read_text(encoding="utf-8"))
    assert release["production_paths"]["trusted_data_protocol"] == "docs/CURRENT_ENGINEERING_POLICY.md"
    assert release["documentation"]["current_index"] == "docs/CURRENT_DOCS.json"
    if delivery["historical_records_included"]:
        assert release["documentation"]["history_root"] == "docs/history/"
    else:
        assert release["documentation"]["history_included"] is False
        assert release["documentation"]["history_location"] == "NOT_INCLUDED_IN_THIS_SOURCE_PACKAGE"
    assert release["documentation"]["history_may_override_current"] is False

def test_stable_source_builder_purges_generated_workspace_state(tmp_path: Path) -> None:
    import importlib.util

    script = ROOT / "scripts/release/build_stable_source.py"
    spec = importlib.util.spec_from_file_location("post49_stable_builder", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    staged = tmp_path / "AstroGraph"
    (staged / "scripts/data_driven/__pycache__").mkdir(parents=True)
    (staged / "scripts/data_driven/__pycache__/x.pyc").write_bytes(b"cache")
    (staged / ".pytest_cache").mkdir(parents=True)
    (staged / "src/example.pyc").parent.mkdir(parents=True)
    (staged / "src/example.pyc").write_bytes(b"cache")
    (staged / "keep.txt").write_text("keep", encoding="utf-8")

    module._purge_workspace_state(staged)

    assert not (staged / "scripts/data_driven/__pycache__").exists()
    assert not (staged / ".pytest_cache").exists()
    assert not (staged / "src/example.pyc").exists()
    assert (staged / "keep.txt").read_text(encoding="utf-8") == "keep"
