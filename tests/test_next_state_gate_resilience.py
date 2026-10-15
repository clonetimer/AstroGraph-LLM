from __future__ import annotations

from pathlib import Path

from astrograph.evaluation.next_state_gate import audit_frozen_predictions, build_next_state_gate_report


def test_missing_frozen_prediction_files_return_structured_failure(tmp_path: Path) -> None:
    report = audit_frozen_predictions(tmp_path)

    assert report["status"] == "FAIL"
    assert report["file_count"] == 0
    assert len(report["missing_files"]) == report["required_file_count"]


def test_next_state_gate_remains_reportable_when_inputs_are_missing(tmp_path: Path) -> None:
    report = build_next_state_gate_report(outputs_root=tmp_path / "outputs", prediction_root=tmp_path / "predictions")

    assert report["status"] == "BLOCKED"
    assert "FROZEN_MODEL_PREDICTION_AUDIT_NOT_PASS" in report["blockers"]
    assert "KG_BRANCH_DECORRELATION_OR_WEIGHTING_NOT_VALIDATED" in report["blockers"]
