from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from astrograph.assurance.kg_coverage import KGCoverageAuditError, build_kg_coverage_audit

ROOT = Path(__file__).resolve().parents[2]


def test_project_coverage_audit_reports_distinct_denominators() -> None:
    report = build_kg_coverage_audit(ROOT)
    adcs = report["datasets"]["ADCS_V5_2_2"]
    phm = report["datasets"]["PHM_V6_0"]

    assert adcs["coverage"]["against_dataset_fault_universe"] == {
        "numerator": 4,
        "denominator": 14,
        "ratio": pytest.approx(4 / 14),
    }
    assert adcs["coverage"]["against_strict_registered_mapping_universe"] == {
        "numerator": 4,
        "denominator": 14,
        "ratio": pytest.approx(4 / 14),
    }
    assert phm["coverage"]["against_dataset_fault_universe"] == {
        "numerator": 5,
        "denominator": 17,
        "ratio": pytest.approx(5 / 17),
    }
    assert phm["coverage"]["against_strict_registered_mapping_universe"] == {
        "numerator": 5,
        "denominator": 5,
        "ratio": 1.0,
    }
    assert phm["coverage"]["against_configured_runtime_candidate_universe"] == {
        "numerator": 5,
        "denominator": 5,
        "ratio": 1.0,
    }
    assert phm["claim_boundary"]["configured_runtime_universe_may_be_narrower_than_dataset"] is True
    assert phm["claim_boundary"]["whole_dataset_kg_gain_claim_allowed"] is False
    assert report["selection_boundary"]["post_hoc_only_groups"] == [4, 9]
    assert report["selection_boundary"]["groups_4_9_parameter_selection_prohibited"] is True
    assert len(report["audit_sha256"]) == 64


def test_coverage_audit_is_deterministic() -> None:
    first = build_kg_coverage_audit(ROOT)
    second = build_kg_coverage_audit(ROOT)
    assert first == second


def test_coverage_audit_rejects_declared_rule_mismatch(tmp_path: Path) -> None:
    destination = tmp_path / "project" / "configs" / "datasets"
    destination.mkdir(parents=True)
    for name in ("adcs_v5_2_2", "phm_v6_0"):
        shutil.copytree(ROOT / "configs" / "datasets" / name, destination / name)

    candidate_path = destination / "adcs_v5_2_2" / "candidate_universe.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["configured_rule_evidence_fault_ids"].append("ADCS_FALSE_DECLARATION")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    with pytest.raises(KGCoverageAuditError, match="disagree"):
        build_kg_coverage_audit(tmp_path / "project")



def _copied_dataset_tree(tmp_path: Path) -> Path:
    project_root = tmp_path / "project"
    destination = project_root / "configs" / "datasets"
    destination.mkdir(parents=True)
    for name in ("adcs_v5_2_2", "phm_v6_0"):
        shutil.copytree(ROOT / "configs" / "datasets" / name, destination / name)
    return project_root


def test_coverage_audit_rejects_unmapped_runtime_candidate(tmp_path: Path) -> None:
    project_root = _copied_dataset_tree(tmp_path)
    candidate_path = project_root / "configs/datasets/adcs_v5_2_2/candidate_universe.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["runtime_fault_ids"].append("ADCS_UNMAPPED_RUNTIME_FAULT")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    with pytest.raises(KGCoverageAuditError, match="runtime candidate universe contains unmapped"):
        build_kg_coverage_audit(project_root)


def test_coverage_audit_rejects_unpinned_asset_path(tmp_path: Path) -> None:
    project_root = _copied_dataset_tree(tmp_path)
    candidate_path = project_root / "configs/datasets/adcs_v5_2_2/candidate_universe.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["runtime_evidence_path"] = "configs/datasets/adcs_v5_2_2/other.json"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    with pytest.raises(KGCoverageAuditError, match="does not pin the audited asset"):
        build_kg_coverage_audit(project_root)
