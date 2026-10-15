from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from astrograph.data_driven import DataDrivenPrediction


ACTIVITY_AXIS_SOURCES = {
    "fault_timing": "v0522_fault_timing_runtime_effect",
    "severity": "v0528_all_fault_severity_mapping",
    "noise": "v0523_sensor_noise_runtime_effect",
    "dropout": "v0524_sensor_dropout_runtime_effect",
    "operating_condition": "v0525_operating_condition_runtime_effect",
    "initial_state": "v0526_initial_state_runtime_effect",
    "satellite_instance": "v0527_satellite_instance_runtime_effect",
}

FROZEN_PREDICTION_FILES = [
    "lab-hist-gradient-boosting_predictions.json",
    "lab-logistic-regression_predictions.json",
    "lab-random-forest_predictions.json",
]

DEFAULT_ACTIVITY_AXIS_PREDICTION_REPORT = (
    "v0530_r4f2_activity_axis_inference/activity_axis_prediction_report.json"
)
DEFAULT_KG_FALLBACK_REPORT = (
    "v0530_r4f2_kg_fallback_21_scene/kg_fallback_21_scene_report.json"
)
DEFAULT_KG_REDUNDANCY_REPORT = (
    "v0530_r4f2_kg_redundancy_gate/kg_redundancy_quality_gate_report.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validation_file(axis_root: Path) -> Path | None:
    candidates = sorted(axis_root.glob("*VALIDATION.json"))
    return candidates[0] if candidates else None


def audit_activity_axis_artifacts(outputs_root: Path) -> dict[str, Any]:
    axes: dict[str, dict[str, Any]] = {}
    for axis, directory in ACTIVITY_AXIS_SOURCES.items():
        root = outputs_root / directory
        validation_path = _validation_file(root)
        validation = _read_json(validation_path) if validation_path and validation_path.is_file() else {}
        telemetry_files = sorted(root.rglob("telemetry.csv")) if root.is_dir() else []
        manifest_files = sorted(root.rglob("run_manifest.json")) if root.is_dir() else []
        controlled_count = 0
        manifest_hash_mismatches = []
        for manifest_path in manifest_files:
            manifest = _read_json(manifest_path)
            axis_record = manifest.get("activity_axes", {}).get(axis, {})
            if axis_record.get("controlled") is True:
                controlled_count += 1
            reported = next(
                (
                    item.get("manifest_sha256")
                    for item in validation.get("runs", [])
                    if item.get("scenario") == manifest.get("scenario")
                    and str(item.get("severity", "")).lower() in str(manifest_path).lower()
                ),
                None,
            )
            if reported and reported != _sha256(manifest_path):
                manifest_hash_mismatches.append(str(manifest_path))
        axes[axis] = {
            "source_dir": str(root),
            "validation_file": str(validation_path) if validation_path else None,
            "validation_status": validation.get("status"),
            "telemetry_count": len(telemetry_files),
            "manifest_count": len(manifest_files),
            "controlled_manifest_count": controlled_count,
            "manifest_hash_mismatch_count": len(manifest_hash_mismatches),
            "status": "PASS"
            if validation.get("status") == "PASS"
            and len(telemetry_files) > 0
            and len(manifest_files) > 0
            and controlled_count > 0
            and not manifest_hash_mismatches
            else "FAIL",
            "claim_boundary": validation.get("claim_boundary", ""),
        }
    return {
        "schema_version": "astrograph-v0530-r2-activity-axis-artifact-audit.v1",
        "axis_count": len(axes),
        "pass_count": sum(item["status"] == "PASS" for item in axes.values()),
        "axes": axes,
        "status": "PASS" if axes and all(item["status"] == "PASS" for item in axes.values()) else "FAIL",
    }


def audit_frozen_predictions(prediction_root: Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    total_predictions = 0
    invalid_predictions = 0
    test_predictions = 0
    validation_predictions = 0
    input_hashes: set[str] = set()
    missing_files: list[str] = []
    invalid_files: dict[str, str] = {}
    for filename in FROZEN_PREDICTION_FILES:
        path = prediction_root / filename
        if not path.is_file():
            missing_files.append(filename)
            continue
        try:
            payload = _read_json(path)
        except Exception as exc:
            invalid_files[filename] = f"{type(exc).__name__}: {exc}"
            continue
        split_counts: dict[str, int] = {}
        predictions = payload.get("predictions", {})
        if not isinstance(predictions, dict):
            invalid_files[filename] = "predictions must be a JSON object"
            continue
        for split, rows in predictions.items():
            if not isinstance(rows, list):
                invalid_predictions += 1
                continue
            split_counts[split] = len(rows)
            if split == "TEST":
                test_predictions += len(rows)
            if split == "VALIDATION":
                validation_predictions += len(rows)
            for row in rows:
                try:
                    prediction = DataDrivenPrediction.model_validate(row["prediction"])
                    input_hashes.add(prediction.input_data_sha256)
                    total_predictions += 1
                except Exception:
                    invalid_predictions += 1
        files[filename] = {
            "model_id": payload.get("model_id"),
            "model_version": payload.get("model_version"),
            "sha256": _sha256(path),
            "split_counts": split_counts,
            "calibration_id": payload.get("calibration", {}).get("calibration_id"),
        }
    return {
        "schema_version": "astrograph-v0530-r2-frozen-prediction-audit.v1",
        "prediction_root": str(prediction_root),
        "file_count": len(files),
        "required_file_count": len(FROZEN_PREDICTION_FILES),
        "missing_files": missing_files,
        "invalid_files": invalid_files,
        "total_predictions": total_predictions,
        "validation_predictions": validation_predictions,
        "test_predictions": test_predictions,
        "invalid_predictions": invalid_predictions,
        "unique_input_data_sha256_count": len(input_hashes),
        "files": files,
        "status": "PASS"
        if len(files) == len(FROZEN_PREDICTION_FILES)
        and not missing_files
        and not invalid_files
        and invalid_predictions == 0
        else "FAIL",
        "claim_boundary": (
            "Frozen predictions are existing model outputs. This audit does not retrain, tune, or infer on "
            "V0.5.22-V0.5.28 activity-axis telemetry."
        ),
    }


def audit_linked_activity_axis_predictions(
    report_path: Path,
    *,
    expected_axes: set[str] | None = None,
    expected_model_count: int = 3,
) -> dict[str, Any]:
    expected_axes = expected_axes or set(ACTIVITY_AXIS_SOURCES)
    if not report_path.is_file():
        return {
            "schema_version": "astrograph-v0530-r4f2-linked-activity-axis-prediction-audit.v1",
            "report_path": str(report_path),
            "status": "FAIL",
            "blocking_reasons": ["ACTIVITY_AXIS_PREDICTION_REPORT_MISSING"],
            "axis_count": 0,
            "model_count": 0,
            "sample_count": 0,
            "prediction_count": 0,
            "claim_boundary": "No activity-axis prediction report was available for linkage audit.",
        }
    try:
        payload = _read_json(report_path)
    except Exception as exc:
        return {
            "schema_version": "astrograph-v0530-r4f2-linked-activity-axis-prediction-audit.v1",
            "report_path": str(report_path),
            "status": "FAIL",
            "blocking_reasons": [f"ACTIVITY_AXIS_PREDICTION_REPORT_INVALID:{type(exc).__name__}"],
            "axis_count": 0,
            "model_count": 0,
            "sample_count": 0,
            "prediction_count": 0,
            "claim_boundary": "The activity-axis prediction report could not be parsed.",
        }

    reasons: list[str] = []
    samples = payload.get("samples", [])
    predictions = payload.get("predictions", [])
    sample_by_id = {str(item.get("sample_id")): item for item in samples}
    axes = {str(item.get("axis")) for item in samples}
    model_ids = {str(item.get("model_id")) for item in predictions}
    if payload.get("status") != "PASS":
        reasons.append("ACTIVITY_AXIS_PREDICTION_EXECUTION_NOT_PASS")
    if axes != expected_axes:
        reasons.append("ACTIVITY_AXIS_PREDICTION_AXIS_COVERAGE_INCOMPLETE")
    if len(model_ids) != expected_model_count:
        reasons.append("ACTIVITY_AXIS_PREDICTION_MODEL_COVERAGE_INCOMPLETE")
    expected_prediction_count = len(samples) * len(model_ids)
    if len(predictions) != expected_prediction_count:
        reasons.append("ACTIVITY_AXIS_PREDICTION_COUNT_MISMATCH")
    seen_pairs: set[tuple[str, str]] = set()
    for row in predictions:
        sample_id = str(row.get("sample_id"))
        model_id = str(row.get("model_id"))
        pair = (sample_id, model_id)
        if pair in seen_pairs:
            reasons.append("ACTIVITY_AXIS_PREDICTION_DUPLICATE_SAMPLE_MODEL_PAIR")
        seen_pairs.add(pair)
        sample = sample_by_id.get(sample_id)
        if sample is None:
            reasons.append("ACTIVITY_AXIS_PREDICTION_REFERENCES_UNKNOWN_SAMPLE")
            continue
        if row.get("input_data_sha256") != sample.get("telemetry_sha256"):
            reasons.append("ACTIVITY_AXIS_PREDICTION_INPUT_HASH_NOT_LINKED")
    reasons = list(dict.fromkeys(reasons))
    return {
        "schema_version": "astrograph-v0530-r4f2-linked-activity-axis-prediction-audit.v1",
        "report_path": str(report_path),
        "report_sha256": _sha256(report_path),
        "status": "PASS" if not reasons else "FAIL",
        "blocking_reasons": reasons,
        "axis_count": len(axes),
        "axes": sorted(axes),
        "model_count": len(model_ids),
        "model_ids": sorted(model_ids),
        "sample_count": len(samples),
        "prediction_count": len(predictions),
        "metrics": payload.get("metrics", {}),
        "claim_boundary": (
            "PASS proves only sample-level hash linkage between frozen-model predictions and registered "
            "activity-axis telemetry. It does not establish acceptable cross-condition accuracy."
        ),
    }



def audit_online_kg_fallback_report(
    report_path: Path,
    *,
    expected_scenario_count: int = 21,
) -> dict[str, Any]:
    """Validate workflow-level automatic CSV fallback across the registered scene set."""

    schema_version = "astrograph-v0530-r4f2-online-kg-fallback-audit.v1"
    if not report_path.is_file():
        return {
            "schema_version": schema_version,
            "report_path": str(report_path),
            "status": "FAIL",
            "blocking_reasons": ["KG_FALLBACK_21_SCENE_REPORT_MISSING"],
            "scenario_count": 0,
            "claim_boundary": "No workflow-level fallback report was available.",
        }
    try:
        payload = _read_json(report_path)
    except Exception as exc:
        return {
            "schema_version": schema_version,
            "report_path": str(report_path),
            "status": "FAIL",
            "blocking_reasons": [f"KG_FALLBACK_21_SCENE_REPORT_INVALID:{type(exc).__name__}"],
            "scenario_count": 0,
            "claim_boundary": "The workflow-level fallback report could not be parsed.",
        }

    reasons: list[str] = []
    rows = payload.get("rows", [])
    scenario_names = [str(item.get("scenario")) for item in rows]
    if payload.get("status") != "PASS":
        reasons.append("KG_FALLBACK_21_SCENE_EXECUTION_NOT_PASS")
    if payload.get("scenario_count") != expected_scenario_count or len(rows) != expected_scenario_count:
        reasons.append("KG_FALLBACK_21_SCENE_COUNT_MISMATCH")
    if len(set(scenario_names)) != len(scenario_names):
        reasons.append("KG_FALLBACK_21_SCENE_DUPLICATE_SCENARIO")
    expected_totals = {
        "workflow_success_count": expected_scenario_count,
        "fallback_selected_count": expected_scenario_count,
        "fusion_present_count": expected_scenario_count,
        "csv_backend_count": expected_scenario_count,
    }
    for field, expected in expected_totals.items():
        if payload.get(field) != expected:
            reasons.append(f"KG_FALLBACK_21_SCENE_{field.upper()}_MISMATCH")
    for row in rows:
        if row.get("workflow_status") != "SUCCEEDED":
            reasons.append("KG_FALLBACK_21_SCENE_WORKFLOW_FAILURE")
        if row.get("fallback_selected") is not True:
            reasons.append("KG_FALLBACK_21_SCENE_FALLBACK_NOT_SELECTED")
        if row.get("selected_backend") != "csv_reference_fallback":
            reasons.append("KG_FALLBACK_21_SCENE_SELECTED_BACKEND_INVALID")
        if row.get("kg_graph_backend") != "csv_reference_fallback":
            reasons.append("KG_FALLBACK_21_SCENE_GRAPH_BACKEND_INVALID")
        if row.get("fusion_present") is not True:
            reasons.append("KG_FALLBACK_21_SCENE_FUSION_MISSING")
        if int(row.get("kg_row_count") or 0) <= 0:
            reasons.append("KG_FALLBACK_21_SCENE_EMPTY_GRAPH_RESULT")
        if row.get("error") not in (None, ""):
            reasons.append("KG_FALLBACK_21_SCENE_ROW_ERROR")
    reasons = list(dict.fromkeys(reasons))
    return {
        "schema_version": schema_version,
        "report_path": str(report_path),
        "report_sha256": _sha256(report_path),
        "status": "PASS" if not reasons else "FAIL",
        "blocking_reasons": reasons,
        "scenario_count": len(rows),
        "scenarios": sorted(scenario_names),
        "workflow_success_count": payload.get("workflow_success_count", 0),
        "fallback_selected_count": payload.get("fallback_selected_count", 0),
        "fusion_present_count": payload.get("fusion_present_count", 0),
        "csv_backend_count": payload.get("csv_backend_count", 0),
        "claim_boundary": (
            "PASS proves deterministic workflow-level fallback to the hash-bound CSV Knowledge Release "
            "when Neo4j is unavailable. It does not prove timeout recovery, mid-query disconnect recovery, "
            "or equivalent diagnosis accuracy between backends."
        ),
    }


def audit_kg_redundancy_quality_gate(
    report_path: Path,
    *,
    expected_case_count: int = 7,
) -> dict[str, Any]:
    schema_version = "astrograph-v0530-r4f2-kg-redundancy-quality-gate-audit.v1"
    if not report_path.is_file():
        return {
            "schema_version": schema_version,
            "report_path": str(report_path),
            "status": "FAIL",
            "blocking_reasons": ["KG_REDUNDANCY_QUALITY_GATE_REPORT_MISSING"],
            "case_count": 0,
            "claim_boundary": "No KG redundancy quality-gate report was available.",
        }
    try:
        payload = _read_json(report_path)
    except Exception as exc:
        return {
            "schema_version": schema_version,
            "report_path": str(report_path),
            "status": "FAIL",
            "blocking_reasons": [f"KG_REDUNDANCY_QUALITY_GATE_REPORT_INVALID:{type(exc).__name__}"],
            "case_count": 0,
            "claim_boundary": "The KG redundancy quality-gate report could not be parsed.",
        }

    reasons: list[str] = []
    rows = payload.get("rows", [])
    if payload.get("status") != "PASS":
        reasons.append("KG_REDUNDANCY_QUALITY_GATE_NOT_PASS")
    if payload.get("case_count") != expected_case_count or len(rows) != expected_case_count:
        reasons.append("KG_REDUNDANCY_QUALITY_GATE_CASE_COUNT_MISMATCH")
    identical = int(payload.get("identical_rank_case_count") or 0)
    collapsed = int(payload.get("collapsed_case_count") or 0)
    duplicates = int(payload.get("duplicate_contribution_case_count") or 0)
    if identical <= 0 or collapsed != identical:
        reasons.append("KG_REDUNDANCY_IDENTICAL_BRANCHES_NOT_FULLY_COLLAPSED")
    if duplicates != 0:
        reasons.append("KG_REDUNDANCY_DUPLICATE_CONTRIBUTIONS_REMAIN")
    for row in rows:
        if row.get("kg_rank_orders_identical") is True:
            if row.get("collapsed_kg_source") not in {"KG_FORWARD", "KG_BACKWARD"}:
                reasons.append("KG_REDUNDANCY_ROW_NOT_COLLAPSED")
            if int(row.get("duplicate_contribution_count") or 0) != 0:
                reasons.append("KG_REDUNDANCY_ROW_DUPLICATE_CONTRIBUTION")
    reasons = list(dict.fromkeys(reasons))
    return {
        "schema_version": schema_version,
        "report_path": str(report_path),
        "report_sha256": _sha256(report_path),
        "status": "PASS" if not reasons else "FAIL",
        "blocking_reasons": reasons,
        "case_count": len(rows),
        "identical_rank_case_count": identical,
        "collapsed_case_count": collapsed,
        "duplicate_contribution_case_count": duplicates,
        "claim_boundary": (
            "PASS proves an implemented duplicate-vote quality gate for exactly identical KG rank orders. "
            "It does not prove forward/backward statistical independence or calibrated branch weighting."
        ),
    }

def build_next_state_gate_report(
    *,
    outputs_root: Path,
    prediction_root: Path,
    activity_axis_prediction_report_path: Path | None = None,
    kg_fallback_report_path: Path | None = None,
    kg_redundancy_report_path: Path | None = None,
) -> dict[str, Any]:
    activity = audit_activity_axis_artifacts(outputs_root)
    frozen = audit_frozen_predictions(prediction_root)
    linked_report_path = activity_axis_prediction_report_path or (
        outputs_root / DEFAULT_ACTIVITY_AXIS_PREDICTION_REPORT
    )
    linked = audit_linked_activity_axis_predictions(linked_report_path)
    fallback_report_path = kg_fallback_report_path or (outputs_root / DEFAULT_KG_FALLBACK_REPORT)
    fallback = audit_online_kg_fallback_report(fallback_report_path)
    redundancy_report_path = kg_redundancy_report_path or (outputs_root / DEFAULT_KG_REDUNDANCY_REPORT)
    redundancy = audit_kg_redundancy_quality_gate(redundancy_report_path)
    blockers = []
    if activity["status"] != "PASS":
        blockers.append("REAL_ACTIVITY_AXIS_ARTIFACT_AUDIT_NOT_PASS")
    if frozen["status"] != "PASS":
        blockers.append("FROZEN_MODEL_PREDICTION_AUDIT_NOT_PASS")
    if linked["status"] != "PASS":
        blockers.append("FROZEN_PREDICTIONS_NOT_LINKED_TO_ACTIVITY_AXIS_RUNS")
    if fallback["status"] != "PASS":
        blockers.append("NEO4J_21_SCENE_AUTOMATIC_FALLBACK_NOT_VALIDATED")
    if redundancy["status"] != "PASS":
        blockers.append("KG_BRANCH_DECORRELATION_OR_WEIGHTING_NOT_VALIDATED")
    blockers.extend(
        [
            "REAL_OOD_OR_UNREGISTERED_FAULT_ARTIFACTS_NOT_PRESENT",
            "LOW_PRIVILEGE_READ_ONLY_NEO4J_USER_NOT_VALIDATED",
            "NEO4J_TIMEOUT_DISCONNECT_RECOVERY_NOT_VALIDATED",
        ]
    )
    return {
        "schema_version": "astrograph-v0530-r2-next-state-gate.v1",
        "gate_name": "post-v0530-r1-hard-gap-gate",
        "activity_axis_artifacts": activity,
        "frozen_predictions": frozen,
        "activity_axis_predictions": linked,
        "linked_activity_axis_prediction_status": "PASS" if linked["status"] == "PASS" else "BLOCKED",
        "online_kg_fallback": fallback,
        "kg_redundancy_quality_gate": redundancy,
        "kg_branch_quality_gate_status": "PASS" if redundancy["status"] == "PASS" else "BLOCKED",
        "real_ood_status": "BLOCKED",
        "neo4j_workflow_fallback_status": "PASS" if fallback["status"] == "PASS" else "BLOCKED",
        "agent_runtime_admission": "BLOCKED",
        "blockers": blockers,
        "status": "BLOCKED",
        "claim_boundary": (
            "This gate validates inventory and schema readiness only. It does not claim KG net gain on real "
            "cross-condition data, unknown-fault rejection, or workflow-level Neo4j resilience."
        ),
    }


def write_next_state_gate_report(report: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "next_state_gate_report.json"
    md_path = output_dir / "next_state_gate_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def _markdown(report: dict[str, Any]) -> str:
    activity = report["activity_axis_artifacts"]
    frozen = report["frozen_predictions"]
    linked = report["activity_axis_predictions"]
    fallback = report["online_kg_fallback"]
    redundancy = report["kg_redundancy_quality_gate"]
    lines = [
        "# AstroGraph V0.5.30-R2 Next State Gate",
        "",
        "## Decision",
        "",
        f"- Status: {report['status']}",
        f"- Agent runtime admission: {report['agent_runtime_admission']}",
        "",
        "## Activity Axis Artifacts",
        "",
        "| Axis | Status | Telemetry | Manifests | Controlled Manifests |",
        "|---|---:|---:|---:|---:|",
    ]
    for axis, row in activity["axes"].items():
        lines.append(
            f"| {axis} | {row['status']} | {row['telemetry_count']} | "
            f"{row['manifest_count']} | {row['controlled_manifest_count']} |"
        )
    lines.extend(
        [
            "",
            "## Frozen Predictions",
            "",
            f"- Status: {frozen['status']}",
            f"- Files: {frozen['file_count']}",
            f"- Validation predictions: {frozen['validation_predictions']}",
            f"- Test predictions: {frozen['test_predictions']}",
            f"- Invalid predictions: {frozen['invalid_predictions']}",
            "",
            "## Activity-Axis Prediction Linkage",
            "",
            f"- Status: {linked['status']}",
            f"- Axes: {linked['axis_count']}",
            f"- Models: {linked['model_count']}",
            f"- Samples: {linked['sample_count']}",
            f"- Predictions: {linked['prediction_count']}",
            "",
            "## Online KG Fallback",
            "",
            f"- Status: {fallback['status']}",
            f"- Scenarios: {fallback['scenario_count']}",
            f"- Workflow successes: {fallback.get('workflow_success_count', 0)}",
            f"- Fallback selections: {fallback.get('fallback_selected_count', 0)}",
            f"- Fusion outputs: {fallback.get('fusion_present_count', 0)}",
            "",
            "## KG Redundancy Quality Gate",
            "",
            f"- Status: {redundancy['status']}",
            f"- Cases: {redundancy['case_count']}",
            f"- Identical rank cases: {redundancy.get('identical_rank_case_count', 0)}",
            f"- Collapsed cases: {redundancy.get('collapsed_case_count', 0)}",
            f"- Duplicate-contribution cases: {redundancy.get('duplicate_contribution_case_count', 0)}",
            "",
            "## Blockers",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["blockers"])
    lines.extend(["", "## Claim Boundary", "", report["claim_boundary"], ""])
    return "\n".join(lines)
