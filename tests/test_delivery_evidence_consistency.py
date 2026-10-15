from __future__ import annotations

from pathlib import Path

import pytest

from astrograph.governance.delivery_evidence import (
    load_delivery_evidence_registry,
    resolve_delivery_evidence,
    validate_delivery_evidence,
)

ROOT = Path(__file__).resolve().parents[1]


def test_current_source_delivery_declares_historical_dependencies_explicitly():
    report = validate_delivery_evidence(ROOT)
    assert report["status"] == "PASS_WITH_EXTERNAL_DEPENDENCIES"
    assert report["external_declared_count"] >= 8
    assert report["errors"] == []
    assert all(item["consumers"] for item in report["dependencies"])


def test_strict_delivery_rejects_archive_only_dependencies():
    report = validate_delivery_evidence(ROOT, require_bundled=True)
    assert report["status"] == "FAIL"
    assert report["external_declared_count"] >= 8
    assert all(error.startswith("EXTERNAL_NOT_ALLOWED:") for error in report["errors"])


def test_unregistered_missing_evidence_cannot_be_silently_treated_as_external():
    with pytest.raises(KeyError, match="not registered"):
        resolve_delivery_evidence(ROOT, "governance/evidence/unknown/missing.json")


def test_registry_paths_are_unique_and_safe():
    registry = load_delivery_evidence_registry(ROOT)
    paths = [item["path"] for item in registry["dependencies"]]
    ids = [item["dependency_id"] for item in registry["dependencies"]]
    assert len(paths) == len(set(paths))
    assert len(ids) == len(set(ids))
    assert all(not Path(path).is_absolute() and ".." not in Path(path).parts for path in paths)
