from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field


REGISTERED_EXPECTATIONS: dict[str, tuple[str, str | None, bool]] = {
    "nominal": ("NO_REGISTERED_FAULT", None, False),
    "reaction_wheel_friction_increase": ("ACCEPT", "ADCS_RW_FRICTION_INCREASE", False),
    "reaction_wheel_torque_authority_loss": ("ACCEPT", "ADCS_RW_TORQUE_AUTHORITY_LOSS", False),
    "heater_stuck_on": ("ACCEPT", "THERMAL_HEATER_STUCK_ON", False),
    "star_tracker_dropout": ("ACCEPT", "ADCS_STAR_TRACKER_DROPOUT", False),
    "reaction_wheel_jam": ("ACCEPT", "ADCS_RW_JAM", False),
    "battery_internal_resistance_increase": ("ACCEPT", "EPS_BATTERY_INTERNAL_RESISTANCE_INCREASE", False),
    "solar_array_string_failure": ("ACCEPT", "EPS_SOLAR_ARRAY_STRING_FAILURE", False),
    "gyro_bias_drift": ("ACCEPT", "ADCS_GYRO_BIAS_DRIFT", False),
    "transmitter_output_power_loss": ("ACCEPT", "COMM_TRANSMITTER_OUTPUT_POWER_LOSS", False),
    "storage_write_failure": ("ACCEPT", "DATA_STORAGE_WRITE_FAILURE", False),
    "battery_capacity_loss": ("ACCEPT", "EPS_BATTERY_CAPACITY_LOSS", False),
    "solar_efficiency_loss": ("ACCEPT", "EPS_SOLAR_ARRAY_EFFICIENCY_LOSS", False),
    "radiator_rejection_loss": ("ACCEPT", "THERMAL_RADIATOR_REJECTION_LOSS", False),
    "downlink_service_loss": ("ACCEPT", "COMM_DOWNLINK_SERVICE_LOSS", False),
    "temperature_sensor_bias": ("ACCEPT", "THERMAL_TEMPERATURE_SENSOR_BIAS", False),
    "temperature_sensor_drift": ("ACCEPT", "THERMAL_TEMPERATURE_SENSOR_DRIFT", False),
}

OUT_OF_SCOPE_SCENARIOS: set[str] = set()

COMPOSITE_SCENARIOS = {
    "eps_capacity_plus_solar",
    "radiator_plus_heater",
    "rw_friction_plus_star_tracker",
    "transmitter_plus_storage_write",
}

REQUIRED_TOOL_IDS = {
    "semantics.validate",
    "online.runtime_evidence.build",
    "diagnosis.invoke_online_core",
    "online.evidence_graph.build",
    "online.artifacts.persist",
}


class ScenarioExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario: str
    expected_disposition: str
    expected_fault_id: str | None = None
    expected_human_review: bool = False


class ArtifactAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario: str
    artifact_dir: str
    manifest_valid: bool
    missing_files: list[str] = Field(default_factory=list)
    hash_mismatches: list[str] = Field(default_factory=list)
    tool_ids: list[str] = Field(default_factory=list)
    missing_tool_ids: list[str] = Field(default_factory=list)
    model_id: str | None = None
    reasoning_status: str | None = None
    latency_s: float | None = None
    attempts: int | None = None
    repair_attempt_count: int = 0
    degraded_to_reference: bool = False
    invalid_evidence_reference_count: int = 0
    core_comparison_status: str | None = None
    disposition: str | None = None
    accepted_fault_id: str | None = None
    requires_human_review: bool | None = None
    hypothesis_structure_sha256: str | None = None
    decision_structure_sha256: str | None = None


class SimulationSourceAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-v058-simulation-source-audit.v1"
    root: str
    expected_scenario_count: int
    actual_scenario_count: int
    passed_count: int
    passed: bool
    rows: list[dict[str, Any]]
    warnings: list[str] = Field(default_factory=list)


class OnlineAcceptanceReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-v059-online-acceptance.v1"
    reasoner_mode: str
    scenario_count: int
    workflow_success_count: int
    workflow_success_rate: float
    expected_route_match_count: int
    expected_route_match_rate: float
    independent_core_match_count: int
    independent_core_match_rate: float
    human_review_count: int
    rule_fallback_count: int
    invalid_evidence_reference_count: int
    repair_count: int
    model_latency_mean_s: float | None = None
    model_latency_p95_s: float | None = None
    manifest_valid_count: int
    tool_registry_valid_count: int
    disposition_counts: dict[str, int]
    rows: list[dict[str, Any]]
    artifact_audits: list[ArtifactAudit]
    warnings: list[str] = Field(default_factory=list)


class StabilityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-v058-online-stability.v1"
    reasoner_mode: str
    repeats: int
    scenario_count: int
    run_count: int
    exact_signature_stability_mean: float
    minimum_scenario_stability: float
    route_stability_mean: float
    scenario_rows: list[dict[str, Any]]
    aggregate_acceptance: OnlineAcceptanceReport


def scenario_expectation(scenario: str) -> ScenarioExpectation:
    if scenario in REGISTERED_EXPECTATIONS:
        disposition, fault_id, review = REGISTERED_EXPECTATIONS[scenario]
        return ScenarioExpectation(
            scenario=scenario,
            expected_disposition=disposition,
            expected_fault_id=fault_id,
            expected_human_review=review,
        )
    if scenario in OUT_OF_SCOPE_SCENARIOS:
        return ScenarioExpectation(
            scenario=scenario,
            expected_disposition="OUT_OF_SCOPE",
            expected_human_review=False,
        )
    if scenario in COMPOSITE_SCENARIOS:
        return ScenarioExpectation(
            scenario=scenario,
            expected_disposition="HUMAN_REVIEW",
            expected_human_review=True,
        )
    raise KeyError(f"no V0.5.9 online acceptance expectation for scenario: {scenario}")


def all_expected_scenarios() -> list[str]:
    return sorted([*REGISTERED_EXPECTATIONS, *OUT_OF_SCOPE_SCENARIOS, *COMPOSITE_SCENARIOS])


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _canonical_sha256(document: Any) -> str:
    raw = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1)))))
    return ordered[rank]


def audit_artifact_dir(artifact_dir: str | Path, *, scenario: str | None = None) -> ArtifactAudit:
    root = Path(artifact_dir).resolve()
    manifest_path = root / "manifest.json"
    missing: list[str] = []
    mismatches: list[str] = []
    if not manifest_path.is_file():
        return ArtifactAudit(
            scenario=scenario or root.name,
            artifact_dir=str(root),
            manifest_valid=False,
            missing_files=["manifest.json"],
            missing_tool_ids=sorted(REQUIRED_TOOL_IDS),
        )
    manifest = _json(manifest_path)
    for name, info in manifest.get("files", {}).items():
        path = root / str(info.get("path", ""))
        if not path.is_file():
            missing.append(name)
            continue
        if _sha256(path) != info.get("sha256") or path.stat().st_size != info.get("size_bytes"):
            mismatches.append(name)

    trace_path = root / "audit" / "agent_trace.json"
    hypothesis_path = root / "diagnosis" / "hypothesis_set.json"
    decision_path = root / "diagnosis" / "decision_record.json"
    trace = _json(trace_path) if trace_path.is_file() else {}
    hypotheses = _json(hypothesis_path) if hypothesis_path.is_file() else {}
    decision = _json(decision_path) if decision_path.is_file() else {}
    tool_ids = sorted({str(item["tool_id"]) for item in trace.get("events", []) if item.get("tool_id")})
    reasoning_meta: dict[str, Any] = {}
    for item in trace.get("events", []):
        if item.get("action") == "generate_hypotheses":
            reasoning_meta = item.get("output_summary", {}).get("reasoning_backend", {}) or {}
            break
    validation = hypotheses.get("validation_metadata", {}) or {}
    hypothesis_structure = [
        {
            "hypothesis_type": item.get("hypothesis_type"),
            "target_fault_id": item.get("target_fault_id"),
            "status": item.get("status"),
            "supporting_evidence_fact_ids": sorted(item.get("supporting_evidence_fact_ids", [])),
            "contradicting_evidence_fact_ids": sorted(item.get("contradicting_evidence_fact_ids", [])),
            "missing_evidence_definition_ids": sorted(item.get("missing_evidence_definition_ids", [])),
        }
        for item in hypotheses.get("hypotheses", [])
    ]
    decision_structure = {
        "disposition": decision.get("disposition"),
        "accepted_fault_id": decision.get("accepted_fault_id"),
        "requires_human_review": decision.get("requires_human_review"),
        "comparison_status": decision.get("comparison_status"),
        "unresolved_gaps": sorted(decision.get("unresolved_gaps", [])),
    }
    invalid_ids = validation.get("invalid_evidence_reference_ids", []) or validation.get(
        "filtered_evidence_reference_ids", []
    ) or []
    invalid_count = int(validation.get("invalid_evidence_reference_count", len(invalid_ids)) or 0)
    return ArtifactAudit(
        scenario=scenario or str(manifest.get("run_id") or root.name),
        artifact_dir=str(root),
        manifest_valid=not missing and not mismatches,
        missing_files=missing,
        hash_mismatches=mismatches,
        tool_ids=tool_ids,
        missing_tool_ids=sorted(REQUIRED_TOOL_IDS - set(tool_ids)),
        model_id=trace.get("model_id"),
        reasoning_status=reasoning_meta.get("status"),
        latency_s=(float(reasoning_meta["latency_s"]) if reasoning_meta.get("latency_s") is not None else None),
        attempts=(int(reasoning_meta["attempts"]) if reasoning_meta.get("attempts") is not None else None),
        repair_attempt_count=int(validation.get("repair_attempt_count", 0) or 0),
        degraded_to_reference=bool(validation.get("degraded_to_reference", False)),
        invalid_evidence_reference_count=invalid_count,
        core_comparison_status=decision.get("comparison_status"),
        disposition=decision.get("disposition"),
        accepted_fault_id=decision.get("accepted_fault_id"),
        requires_human_review=decision.get("requires_human_review"),
        hypothesis_structure_sha256=_canonical_sha256(hypothesis_structure),
        decision_structure_sha256=_canonical_sha256(decision_structure),
    )


def validate_summary(
    summary: dict[str, Any],
    *,
    artifact_root: str | Path,
    require_expected_routes: bool = True,
) -> OnlineAcceptanceReport:
    mode = str(summary.get("reasoner_mode", "unknown"))
    root = Path(artifact_root).resolve()
    rows = list(summary.get("rows", []))
    audits: list[ArtifactAudit] = []
    route_matches = 0
    core_matches = 0
    success_count = 0
    human_review_count = 0
    fallback_count = 0
    invalid_count = 0
    repair_count = 0
    latencies: list[float] = []
    warnings: list[str] = []
    enriched_rows: list[dict[str, Any]] = []
    for row in rows:
        scenario = str(row["scenario"])
        audit = audit_artifact_dir(root / mode / scenario, scenario=scenario)
        audits.append(audit)
        expected = scenario_expectation(scenario) if require_expected_routes else None
        route_match = True
        if expected is not None:
            route_match = (
                row.get("disposition") == expected.expected_disposition
                and row.get("accepted_fault_id") == expected.expected_fault_id
                and bool(row.get("requires_human_review")) == expected.expected_human_review
            )
        success = row.get("status") == "SUCCEEDED"
        comparison = row.get("comparison_status")
        if expected is None:
            core_match = comparison in {"MATCH", "CONSISTENT_NON_DIAGNOSIS", "NOT_COMPARABLE"}
        elif expected.expected_disposition in {"ACCEPT", "NO_REGISTERED_FAULT"}:
            core_match = comparison == "MATCH"
        elif expected.expected_disposition == "OUT_OF_SCOPE":
            core_match = comparison in {"NOT_COMPARABLE", "CONSISTENT_NON_DIAGNOSIS"}
        else:  # Composite contexts are deliberately blocked even if the single-fault Core sees one member.
            core_match = comparison in {"MATCH", "MISMATCH", "NOT_COMPARABLE", "CONSISTENT_NON_DIAGNOSIS", "COMPOSITE_MEMBER_EVIDENCE_ONLY"}
        success_count += int(success)
        route_matches += int(route_match)
        core_matches += int(core_match)
        human_review_count += int(bool(row.get("requires_human_review")))
        fallback = bool(row.get("degraded_to_reference")) or audit.degraded_to_reference
        fallback_count += int(fallback)
        invalid_count += audit.invalid_evidence_reference_count
        repair_count += int(audit.repair_attempt_count > 0)
        if audit.latency_s is not None:
            latencies.append(audit.latency_s)
        if not route_match:
            warnings.append(f"{scenario}: actual route does not match V0.5.9 expectation")
        if not audit.manifest_valid:
            warnings.append(f"{scenario}: artifact manifest validation failed")
        if audit.missing_tool_ids:
            warnings.append(f"{scenario}: missing ToolRegistry trace IDs {audit.missing_tool_ids}")
        enriched_rows.append(
            {
                **row,
                "route_match": route_match,
                "manifest_valid": audit.manifest_valid,
                "tool_registry_valid": not audit.missing_tool_ids,
                "core_alignment_acceptable": core_match,
                "model_id": audit.model_id,
                "model_latency_s": audit.latency_s,
                "invalid_evidence_reference_count": audit.invalid_evidence_reference_count,
                "hypothesis_structure_sha256": audit.hypothesis_structure_sha256,
                "decision_structure_sha256": audit.decision_structure_sha256,
            }
        )
    total = len(rows)
    return OnlineAcceptanceReport(
        reasoner_mode=mode,
        scenario_count=total,
        workflow_success_count=success_count,
        workflow_success_rate=(success_count / total if total else 0.0),
        expected_route_match_count=route_matches,
        expected_route_match_rate=(route_matches / total if total else 0.0),
        independent_core_match_count=core_matches,
        independent_core_match_rate=(core_matches / total if total else 0.0),
        human_review_count=human_review_count,
        rule_fallback_count=fallback_count,
        invalid_evidence_reference_count=invalid_count,
        repair_count=repair_count,
        model_latency_mean_s=(statistics.mean(latencies) if latencies else None),
        model_latency_p95_s=_p95(latencies),
        manifest_valid_count=sum(item.manifest_valid for item in audits),
        tool_registry_valid_count=sum(not item.missing_tool_ids for item in audits),
        disposition_counts=dict(Counter(str(row.get("disposition")) for row in rows)),
        rows=enriched_rows,
        artifact_audits=audits,
        warnings=warnings,
    )


def load_and_validate_summary(
    summary_path: str | Path,
    *,
    artifact_root: str | Path | None = None,
) -> OnlineAcceptanceReport:
    path = Path(summary_path).resolve()
    root = Path(artifact_root).resolve() if artifact_root else path.parent
    return validate_summary(_json(path), artifact_root=root)


def build_stability_report(
    reports: Iterable[OnlineAcceptanceReport],
    *,
    reasoner_mode: str,
) -> StabilityReport:
    items = list(reports)
    if not items:
        raise ValueError("at least one acceptance report is required")
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in items:
        for row in report.rows:
            by_scenario[str(row["scenario"])].append(row)
    scenario_rows: list[dict[str, Any]] = []
    exact_scores: list[float] = []
    route_scores: list[float] = []
    for scenario in sorted(by_scenario):
        rows = by_scenario[scenario]
        signatures = [
            (
                row.get("status"),
                row.get("disposition"),
                row.get("accepted_fault_id"),
                row.get("comparison_status"),
                bool(row.get("requires_human_review")),
                bool(row.get("degraded_to_reference")),
                int(row.get("invalid_evidence_reference_count", 0) or 0),
                row.get("hypothesis_structure_sha256"),
                row.get("decision_structure_sha256"),
            )
            for row in rows
        ]
        signature_counts = Counter(signatures)
        exact = max(signature_counts.values()) / len(rows)
        route = sum(bool(row.get("route_match")) for row in rows) / len(rows)
        exact_scores.append(exact)
        route_scores.append(route)
        scenario_rows.append(
            {
                "scenario": scenario,
                "repeat_count": len(rows),
                "unique_signature_count": len(signature_counts),
                "exact_signature_stability": exact,
                "expected_route_rate": route,
                "workflow_success_rate": sum(row.get("status") == "SUCCEEDED" for row in rows) / len(rows),
                "rule_fallback_rate": sum(bool(row.get("degraded_to_reference")) for row in rows) / len(rows),
                "invalid_evidence_reference_total": sum(
                    int(row.get("invalid_evidence_reference_count", 0) or 0) for row in rows
                ),
            }
        )
    merged_summary = {
        "reasoner_mode": reasoner_mode,
        "rows": [row for report in items for row in report.rows],
    }
    # Acceptance aggregate reuses the already-audited rows but cannot re-open a single artifact root.
    total_rows = merged_summary["rows"]
    aggregate = OnlineAcceptanceReport(
        reasoner_mode=reasoner_mode,
        scenario_count=len(total_rows),
        workflow_success_count=sum(row.get("status") == "SUCCEEDED" for row in total_rows),
        workflow_success_rate=sum(row.get("status") == "SUCCEEDED" for row in total_rows) / len(total_rows),
        expected_route_match_count=sum(bool(row.get("route_match")) for row in total_rows),
        expected_route_match_rate=sum(bool(row.get("route_match")) for row in total_rows) / len(total_rows),
        independent_core_match_count=sum(bool(row.get("core_alignment_acceptable")) for row in total_rows),
        independent_core_match_rate=sum(bool(row.get("core_alignment_acceptable")) for row in total_rows) / len(total_rows),
        human_review_count=sum(bool(row.get("requires_human_review")) for row in total_rows),
        rule_fallback_count=sum(bool(row.get("degraded_to_reference")) for row in total_rows),
        invalid_evidence_reference_count=sum(
            int(row.get("invalid_evidence_reference_count", 0) or 0) for row in total_rows
        ),
        repair_count=sum(int(row.get("repair_attempt_count", 0) or 0) > 0 for row in total_rows),
        model_latency_mean_s=None,
        model_latency_p95_s=None,
        manifest_valid_count=sum(bool(row.get("manifest_valid")) for row in total_rows),
        tool_registry_valid_count=sum(bool(row.get("tool_registry_valid")) for row in total_rows),
        disposition_counts=dict(Counter(str(row.get("disposition")) for row in total_rows)),
        rows=total_rows,
        artifact_audits=[audit for report in items for audit in report.artifact_audits],
        warnings=[warning for report in items for warning in report.warnings],
    )
    return StabilityReport(
        reasoner_mode=reasoner_mode,
        repeats=len(items),
        scenario_count=len(by_scenario),
        run_count=len(total_rows),
        exact_signature_stability_mean=statistics.mean(exact_scores),
        minimum_scenario_stability=min(exact_scores),
        route_stability_mean=statistics.mean(route_scores),
        scenario_rows=scenario_rows,
        aggregate_acceptance=aggregate,
    )


def audit_simulation_output_root(
    root: str | Path,
    *,
    scenarios: Iterable[str] | None = None,
) -> SimulationSourceAudit:
    base = Path(root).resolve()
    expected = list(scenarios or all_expected_scenarios())
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for scenario in expected:
        run_dir = base / scenario
        telemetry = run_dir / "telemetry.csv"
        manifest_path = run_dir / "run_manifest.json"
        passed = telemetry.is_file() and manifest_path.is_file()
        manifest: dict[str, Any] = {}
        if manifest_path.is_file():
            try:
                manifest = _json(manifest_path)
            except Exception as exc:
                passed = False
                warnings.append(f"{scenario}: invalid run_manifest.json: {exc}")
        if manifest and manifest.get("scenario") != scenario:
            passed = False
            warnings.append(f"{scenario}: manifest scenario mismatch")
        row_count = int(manifest.get("row_count", 0) or 0)
        if telemetry.is_file() and row_count <= 0:
            # Count data rows without loading pandas.
            with telemetry.open("r", encoding="utf-8-sig", errors="replace") as stream:
                row_count = max(0, sum(1 for _ in stream) - 1)
        if row_count <= 0:
            passed = False
            warnings.append(f"{scenario}: no telemetry rows")
        event_count = int(manifest.get("event_count", 0) or 0)
        active_event_rows = int(manifest.get("active_event_rows", 0) or 0)
        if scenario != "nominal" and event_count > 0 and active_event_rows <= 0:
            passed = False
            warnings.append(f"{scenario}: configured events were never active")
        rows.append(
            {
                "scenario": scenario,
                "passed": passed,
                "row_count": row_count,
                "event_count": event_count,
                "active_event_rows": active_event_rows,
                "telemetry_sha256": _sha256(telemetry) if telemetry.is_file() else None,
                "manifest_sha256": _sha256(manifest_path) if manifest_path.is_file() else None,
            }
        )
    actual = [path.name for path in base.iterdir() if path.is_dir() and (path / "telemetry.csv").is_file()] if base.is_dir() else []
    return SimulationSourceAudit(
        root=str(base),
        expected_scenario_count=len(expected),
        actual_scenario_count=len(actual),
        passed_count=sum(bool(row["passed"]) for row in rows),
        passed=all(bool(row["passed"]) for row in rows) and len(rows) == len(expected),
        rows=rows,
        warnings=warnings,
    )
