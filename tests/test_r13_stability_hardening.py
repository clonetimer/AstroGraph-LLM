from __future__ import annotations

import pytest

from astrograph.data_driven import ModelInferenceRequest
from astrograph.integrations.platform_knowledge import (
    PlatformKnowledgeClaim,
    PlatformKnowledgeExplanation,
    _validate_explanation,
)
from astrograph.integrations.sat_sim_pipeline import _rw_screening_predict


def _snapshot() -> dict:
    return {
        "candidate_fault_id": "ADCS_RW_JAM",
        "evidence": [{
            "evidence_id": "E1",
            "field": "sensor.adcs.rw.motor_current_a_0",
            "observed_value": 0.2,
            "unit": "A",
        }],
    }


def test_explanation_rejects_measurement_value_not_supported_by_cited_evidence() -> None:
    explanation = PlatformKnowledgeExplanation(
        candidate_fault_id="ADCS_RW_JAM",
        abstained=False,
        summary="Motor current is 900 A.",
        claims=[PlatformKnowledgeClaim(
            kind="OBSERVATION", text="Motor current is 900 A.", citation_ids=["E1"]
        )],
    )
    with pytest.raises(ValueError, match="numerically inconsistent"):
        _validate_explanation(explanation, _snapshot(), [], [])


def test_rw_screening_zero_deviation_is_nominal_not_friction() -> None:
    request = ModelInferenceRequest(
        episode_id="EP-ZERO",
        model_id="rw-engineering-screening",
        model_version="0.1.0-candidate",
        feature_schema_id="sat-sim.rw-screening-projection.v1",
        input_data_sha256="0" * 64,
        registered_fault_ids=[
            "ADCS_RW_FRICTION_INCREASE", "ADCS_RW_JAM", "ADCS_RW_TORQUE_AUTHORITY_LOSS"
        ],
        payload={"features": {
            "raw.rw_speed.paired_rms": 0.0,
            "raw.rw_actual_torque.paired_rms": 0.0,
            "raw.pointing_error.paired_rms": 0.0,
        }},
    )
    raw = _rw_screening_predict(request)
    assert max(raw.class_values, key=raw.class_values.get) == "NORMAL"

import csv
from pathlib import Path

from astrograph.agents.mission.models import AgentNarrative, CitedParagraph
from astrograph.integrations.maintenance_presentation import build_maintenance_presentation
from astrograph.integrations.multidimensional_diagnostics import assess_multidimensional_telemetry
from astrograph.integrations.platform_agent import _validate_narrative
from astrograph.integrations.platform_visualization import build_analysis_series, build_visualization
from astrograph.integrations.temporal_localization import localize_from_visualization


def test_agent_narrative_rejects_measurement_value_not_supported_by_evidence() -> None:
    narrative = AgentNarrative(
        candidate_fault_id="ADCS_RW_JAM",
        paragraphs=[CitedParagraph(text="Motor current is 900 A.", citation_ids=["E1"])],
    )
    with pytest.raises(ValueError, match="numerically inconsistent"):
        _validate_narrative(narrative, snapshot=_snapshot(), visible={"E1"})


def test_maintenance_title_uses_fusion_prediction_not_pipeline_case_identity() -> None:
    report = {
        "job_id": "JOB-R13",
        "status": "SUCCEEDED",
        "artifact_kind": "PIPELINE_RUNTIME",
        "formal_diagnosis_publishable": False,
        "analysis": {"pipeline_result": {
            "fault_id": "ADCS_RW_JAM",
            "disposition": "REVIEW_REQUIRED",
            "blockers": [],
            "fusion": {
                "top_fault_id": "ADCS_RW_FRICTION_INCREASE",
                "candidates": [{"fault_id": "ADCS_RW_FRICTION_INCREASE"}],
            },
        }},
    }
    maintenance = build_maintenance_presentation(
        report=report,
        visualization={"channels": []},
        temporal={"status": "UNAVAILABLE"},
        multidimensional={"status": "UNAVAILABLE", "channels": []},
    )
    assert maintenance["fault_id"] == "ADCS_RW_FRICTION_INCREASE"
    assert maintenance["headline"] == "反作用轮摩擦增大"


def test_visualization_preserves_peak_and_source_row_coverage(tmp_path: Path) -> None:
    telemetry = tmp_path / "telemetry.csv"
    with telemetry.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "a", "b"])
        for index in range(4000):
            missing = index % 2 == 1
            writer.writerow([
                index,
                "" if missing else (100.0 if index == 100 else 0.0),
                "" if missing else (200.0 if index == 100 else 0.0),
            ])
    report = {"artifact_kind": "PIPELINE_RUNTIME", "analysis": {}}
    visualization = build_visualization(expanded=tmp_path, report=report)
    by_name = {channel["name"]: channel for channel in visualization["channels"]}
    assert by_name["a"]["maximum"] == 100.0
    assert by_name["b"]["maximum"] == 200.0
    assert by_name["a"]["coverage_ratio"] == pytest.approx(0.5)
    assert any(point["v"] == 100.0 for point in by_name["a"]["points"])

    analysis = build_analysis_series(expanded=tmp_path, report=report)
    assessment = assess_multidimensional_telemetry(analysis, temporal={"status": "UNAVAILABLE"})
    coverage = {item["channel"]: item["coverage_ratio"] for item in assessment["channels"]}
    assert coverage["a"] == pytest.approx(0.5)
    assert coverage["b"] == pytest.approx(0.5)


def _step_visualization(length: int) -> dict:
    channels = []
    for name in ("a", "b", "c"):
        channels.append({
            "name": name,
            "points": [{"t": index, "v": 0.0 if index < 55 else 10.0} for index in range(length)],
        })
    return {"status": "AVAILABLE", "channels": channels}


def test_temporal_detection_is_prefix_stable_when_later_rows_are_appended() -> None:
    short = localize_from_visualization(_step_visualization(101), diagnostic_fault_id="F")
    long = localize_from_visualization(_step_visualization(1001), diagnostic_fault_id="F")
    assert short["status"] == "DETECTED"
    assert long["status"] == "DETECTED"
    assert short["first_detection_s"] == long["first_detection_s"] == 55.0
