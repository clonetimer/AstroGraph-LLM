from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]


def test_scope_protocol_is_schema_valid_and_complete() -> None:
    root = ROOT / "protocols/acceptance"
    for name in ("diagnosis_scope", "fault_taxonomy", "coverage_matrix", "metric_policy", "reasoning_rubric"):
        Draft202012Validator(json.loads((root / f"{name}.schema.json").read_text(encoding="utf-8"))).validate(
            json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
        )
    scope = json.loads((root / "diagnosis_scope.json").read_text(encoding="utf-8"))
    coverage = json.loads((root / "coverage_matrix.json").read_text(encoding="utf-8"))
    assert scope["object_count"] == 31
    assert len(coverage["rows"]) == 93
    assert sum(1 if row["scenario_type"] == "nominal" else len(row["effect_ids"]) for row in coverage["rows"]) == 176


def test_metric_thresholds_match_project_requirements() -> None:
    policy = json.loads((ROOT / "protocols/acceptance/metric_policy.json").read_text(encoding="utf-8"))
    assert policy["thresholds"] == {
        "classification_accuracy": 0.90, "detection_accuracy": 0.98,
        "localization_accuracy": 0.95, "reasoning_correctness": 0.90,
    }


def test_independent_approval_preserves_frozen_guardrails() -> None:
    approval = json.loads((ROOT / "protocols/acceptance/approvals.json").read_text(encoding="utf-8"))
    evaluator = approval["independent_evaluator"]
    assert evaluator["status"] == "APPROVED"
    assert evaluator["reviewer"] == "Timer"
    assert evaluator["assurance"] == "DECLARED_INDEPENDENT_REVIEWER_IDENTITY"
    declarations = " ".join(evaluator["declarations"])
    for required in ("not be deleted or skipped", "denominators may not be changed", "UNKNOWN may not be expanded", "thresholds may not be lowered"):
        assert required in declarations
