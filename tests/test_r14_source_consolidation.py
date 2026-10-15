from __future__ import annotations

import json
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_r14_unified_source_consolidation_manifest_is_single_source_truth() -> None:
    payload = json.loads((_root() / "governance/merge/PATCH_R14_SOURCE_CONSOLIDATION.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "astrograph-source-consolidation.v1"
    assert payload["patch_id"] == "PATCH-R14"
    comparison = payload["comparison"]
    assert comparison["backend_only_file_count"] == 0
    assert comparison["platform_only_file_count"] == 81
    assert comparison["same_path_content_difference_count"] == 30
    canonical = payload["canonical_tree"]
    assert canonical["one_source_tree"] is True
    assert canonical["separate_backend_source_tree_allowed"] is False
    assert canonical["platform_is_optional_runtime_capability"] is True
    assert canonical["sat_agent_is_optional_upstream_provider"] is True
    assert canonical["matlab_simulink_is_optional_upstream_provider"] is True


def test_r14_research_audit_is_open_and_does_not_claim_confirmatory_execution() -> None:
    payload = json.loads((_root() / "governance/research/PATCH_R14_RESEARCH_AUDIT.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "astrograph-research-audit-registry.v1"
    assert payload["method"]["real_confirmatory_inference_executed"] is False
    assert payload["issue_counts"]["confirmed_defects"] >= 2
    statuses = {item["id"]: item["status"] for item in payload["issues"]}
    assert statuses["R14-GOV-001"] == "CONFIRMED_DEFECT"
    assert statuses["R14-GND-001"] == "CONFIRMED_DEFECT"
