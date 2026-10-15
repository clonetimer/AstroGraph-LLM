from __future__ import annotations

import hashlib
import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VERSION = "0.7.1.post49"
BASELINE_MANIFEST = ROOT / "governance/baselines/post48-dev63-rc-source-manifest.json"
READINESS = ROOT / "governance/implementation/post49/RELEASE_READINESS.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_post49_version_and_stable_claim_converge_without_dev_suffix() -> None:
    release = json.loads((ROOT / "release.json").read_text(encoding="utf-8"))
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    init_text = (ROOT / "src/astrograph/__init__.py").read_text(encoding="utf-8")
    claims = json.loads((ROOT / "configs/release_claims/stable_release.json").read_text(encoding="utf-8"))
    assert release["version"] == VERSION == project["project"]["version"]
    assert f'__version__ = "{VERSION}"' in init_text
    assert release["status"].startswith("STABLE_SOFTWARE_RELEASE")
    assert claims["release_id"] == f"AstroGraph-v{VERSION}-stable"
    assert not re.search(r"\bdev\d+\b", VERSION, re.I)


def test_post49_preserves_dev63_protected_assets_and_diagnostic_source() -> None:
    baseline = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))["files"]
    protected_prefixes = ("models/", "data/", "reference/", "benchmarks/")
    for rel, expected in baseline.items():
        if rel.startswith(protected_prefixes):
            path = ROOT / rel
            assert path.is_file(), rel
            assert _sha(path) == expected, rel
    # Stable convergence is not a diagnostic-source rewrite.  Only the runtime
    # package version declaration may change under src/astrograph.
    for rel, expected in baseline.items():
        if rel.startswith("src/astrograph/") and rel not in {
            "src/astrograph/__init__.py",
            "src/astrograph/lifecycle/__init__.py",
            "src/astrograph/governance/health_check.py",
            "src/astrograph/data_driven/unified_platform.py",
            "src/astrograph/services/reasoning.py",
        }:
            path = ROOT / rel
            assert path.is_file(), rel
            assert _sha(path) == expected, rel

    baseline_health = (ROOT / "governance/baselines/post49-pre-doc-normalization-health_check.py").read_text(encoding="utf-8")
    current_health = (ROOT / "src/astrograph/governance/health_check.py").read_text(encoding="utf-8")
    assert current_health == baseline_health.replace(
        "docs/AI_NATIVE_EVOLUTION_ROADMAP.md",
        "docs/history/root/AI_NATIVE_EVOLUTION_ROADMAP.md",
    )

    baseline_unified = (ROOT / "governance/baselines/post49-pre-doc-normalization-unified_platform.py").read_text(encoding="utf-8")
    current_unified = (ROOT / "src/astrograph/data_driven/unified_platform.py").read_text(encoding="utf-8")
    assert current_unified == baseline_unified.replace(
        "docs/AI_NATIVE_UNIFIED_BASELINE_PREREGISTRATION.md",
        "docs/history/root/AI_NATIVE_UNIFIED_BASELINE_PREREGISTRATION.md",
    )

    baseline_reasoning = (ROOT / "governance/baselines/post49-pre-doc-normalization-reasoning.py").read_text(encoding="utf-8")
    current_reasoning = (ROOT / "src/astrograph/services/reasoning.py").read_text(encoding="utf-8")
    assert current_reasoning == baseline_reasoning.replace(
        "docs/AI_NATIVE_EVOLUTION_ROADMAP.md",
        "docs/history/root/AI_NATIVE_EVOLUTION_ROADMAP.md",
    )

    baseline_lifecycle = (ROOT / "governance/baselines/post48-dev63-rc-lifecycle-init.py").read_text(encoding="utf-8")
    current_lifecycle = (ROOT / "src/astrograph/lifecycle/__init__.py").read_text(encoding="utf-8")
    assert current_lifecycle.startswith(baseline_lifecycle)
    delta = current_lifecycle[len(baseline_lifecycle):]
    assert "from .requests import" in delta
    assert "JsonRequestLifecycleStore" in delta
    assert "RequestLifecycleRecord" in delta
    assert "RequestLifecycleStatus" in delta
    assert "RequestTransition" in delta


def test_post49_readiness_inherits_scenario_boundaries_without_promotion() -> None:
    stable = json.loads(READINESS.read_text(encoding="utf-8"))
    rc = json.loads((ROOT / "governance/implementation/post48_dev63/RELEASE_READINESS.json").read_text(encoding="utf-8"))
    assert stable["version"] == VERSION
    assert stable["release_class"] == "STABLE_SOFTWARE_RELEASE"
    assert stable["automatic_production_promotion"] is False
    assert stable["diagnostic_qualification_promoted_by_post49"] is False
    assert stable["scenarios"] == rc["scenarios"]
    assert stable["external_dependencies"] == rc["external_dependencies"]
    statuses = {item["id"]: item["status"] for item in stable["scenarios"]}
    assert statuses[3] == "PROFILE_GATED_PARTIAL"
    assert statuses[4] == "SUPPORTED_EXACT_ADCS_PROFILE"
    assert statuses[6] == "SUPPORTED_NONVOTING_JOINT_EVIDENCE"


def test_post49_release_metadata_removes_current_dev_suffix_but_preserves_history_by_reference() -> None:
    release = json.loads((ROOT / "release.json").read_text(encoding="utf-8"))
    if "spacecraft_scope_convergence" in release:
        stable = release["spacecraft_scope_convergence"]
        assert stable["patch_id"] == "PATCH-R32"
        assert stable["stable_engineering_candidate"] is True
        assert stable["broad_external_diagnostic_qualification"] is False
        assert stable["next_phase"] == "ASTROGRAPH_BENCH_UI"
        serialized = json.dumps({
            "version": release["version"],
            "status": release["status"],
            "spacecraft_scope_convergence": stable,
        })
    else:
        stable = release["stable_post49"]
        assert stable["direct_rc_baseline"] == "governance/implementation/post48_dev63/RELEASE_READINESS.json"
        assert stable["automatic_production_promotion"] is False
        assert stable["diagnostic_qualification_promoted"] is False
        serialized = json.dumps({
            "version": release["version"],
            "status": release["status"],
            "stable_post49": stable,
        })
    assert re.search(r"\bdev\d+\b", serialized, re.I) is None


def test_post49_baseline_is_content_bound_to_uploaded_dev63_rc() -> None:
    contract = json.loads((ROOT / "governance/implementation/post49/stable_release_frozen/CONTRACT.yaml").read_text(encoding="utf-8"))
    assert contract["baseline_version"] == "0.7.1.post48.dev63"
    assert contract["baseline_archive_sha256"] == "c0d6d45df87a874b03417325465c73081e7f0cfe6640bf9125d1fe9baf85a7dc"
    assert contract["baseline_manifest_path"] == "governance/baselines/post48-dev63-rc-source-manifest.json"
    assert contract["baseline_test_count"] == 174
