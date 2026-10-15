from __future__ import annotations

from pathlib import Path

import pytest

from astrograph.evidence import (
    ComparisonOperator,
    EpisodeEvidenceBundle,
    EpisodeWindow,
    EvidenceDefinition,
    EvidenceFact,
    EvidenceFactStatus,
    EvidenceRole,
    FeatureAggregation,
    FeatureDefinition,
    FeatureResult,
)
from astrograph.integrations.maintenance_presentation import build_maintenance_presentation
from astrograph.integrations.upload_report import UploadReportJobStore


def _bundle(statuses: tuple[EvidenceFactStatus, ...]) -> EpisodeEvidenceBundle:
    episode = EpisodeWindow(
        episode_id="EP-MAINT", run_id="RUN-1", paired_nominal_run_id="RUN-N",
        start_s=0.0, end_s=10.0, source="unit-test",
    )
    feature_definitions = []
    feature_results = []
    evidence_definitions = []
    evidence_facts = []
    for index, status in enumerate(statuses, start=1):
        fd_id = f"FD-{index}"
        fr_id = f"FR-{index}"
        ed_id = f"ED-{index}"
        ef_id = f"EF-{index}"
        value = 3.0 if status == EvidenceFactStatus.SATISFIED else 1.0
        if status in {EvidenceFactStatus.MISSING_DATA, EvidenceFactStatus.INVALID_DATA, EvidenceFactStatus.UNKNOWN}:
            value = None
        feature_definitions.append(FeatureDefinition(
            feature_definition_id=fd_id, name=f"feature {index}", source_field_template=f"x{index}",
            aggregation=FeatureAggregation.MAX_VALUE, unit="A", algorithm_version="test-v1",
        ))
        feature_results.append(FeatureResult(
            feature_result_id=fr_id, feature_definition_id=fd_id, run_id="RUN-1",
            paired_nominal_run_id="RUN-N", episode_id="EP-MAINT", resolved_field=f"x{index}",
            component_instance="rw-2", value=value, unit="A", window_start_s=0.0, window_end_s=10.0,
            sample_count=10 if value is not None else 0, algorithm_version="test-v1", input_data_sha256=str(index) * 64,
            quality_flags=([] if value is not None else ["MISSING"]),
        ))
        evidence_definitions.append(EvidenceDefinition(
            evidence_definition_id=ed_id, target_fault_id="FAULT-A", name=f"critical evidence {index}",
            feature_definition_id=fd_id, role=EvidenceRole.REQUIRED,
            comparison=ComparisonOperator.GREATER_OR_EQUAL, threshold=2.0, threshold_unit="A", critical=True,
        ))
        evidence_facts.append(EvidenceFact(
            evidence_fact_id=ef_id, evidence_definition_id=ed_id, target_fault_id="FAULT-A",
            feature_result_id=fr_id, episode_id="EP-MAINT", role=EvidenceRole.REQUIRED,
            status=status, observed_value=value, threshold_value=2.0, unit="A",
            comparison=ComparisonOperator.GREATER_OR_EQUAL,
        ))
    return EpisodeEvidenceBundle(
        episode=episode, analysis_component_instance="rw-2",
        feature_definitions=feature_definitions, feature_results=feature_results,
        evidence_definitions=evidence_definitions, evidence_facts=evidence_facts,
    )


def _report(bundle: EpisodeEvidenceBundle | None, *, job_id: str = "JOB-000000000000000000000000") -> dict:
    pipeline = {
        "fault_id": "FAULT-A", "disposition": "REVIEW_REQUIRED",
        "blockers": ["NAMED_EXPERT_REVIEW_PENDING"],
        "fusion": {"top_fault_id": "FAULT-A", "candidates": [{"fault_id": "FAULT-A"}]},
    }
    if bundle is not None:
        pipeline["episode_evidence_bundle"] = bundle.model_dump(mode="json")
    return {
        "job_id": job_id, "status": "SUCCEEDED", "artifact_kind": "PIPELINE_RUNTIME",
        "formal_diagnosis_publishable": False,
        "analysis": {"pipeline_result": pipeline},
    }


def _maintenance(report: dict) -> dict:
    return build_maintenance_presentation(
        report=report,
        visualization={"channels": []},
        temporal={"status": "UNAVAILABLE", "first_detection_s": None},
        multidimensional={"status": "UNAVAILABLE", "channels": []},
    )


def test_maintenance_v3_never_fabricates_formal_evidence_summary():
    result = _maintenance(_report(None))
    assert result["schema_version"] == "astrograph-maintenance-diagnosis.v8"
    assert result["evidence_summary"]["status"] == "UNAVAILABLE"
    assert result["evidence_summary"]["formal_evidence_only"] is True
    assert "不能用波形统计或模型解释代替" in result["evidence_summary"]["summary_zh"]
    assert result["next_diagnostic_check"]["status"] == "UNAVAILABLE"
    assert result["next_diagnostic_check"]["auto_executable"] is False


def test_candidate_escape_is_operator_visible_without_becoming_evidence():
    report = _report(_bundle((EvidenceFactStatus.SATISFIED,)))
    report["analysis"]["reasoning_stage"] = {
        "reasoning_backend": {
            "candidate_escape_applied": True,
            "candidate_selection": {
                "compact_reasoning_allowed": False,
                "candidate_set_status": "ACTIVE_CANDIDATE_OVERFLOW",
                "escape_action": "DISABLE_COMPACT_REASONING",
                "escape_reason_codes": ["ACTIVE_CANDIDATES_EXCEED_TOP_K"],
                "selected_fault_ids": ["FAULT-A", "FAULT-B"],
                "unselected_mandatory_fault_ids": ["FAULT-C"],
                "configured_top_k": 2,
                "effective_top_k": 2,
                "forced_candidate_count": 3,
            },
        }
    }
    result = _maintenance(report)
    guard = result["candidate_context_guard"]
    assert guard["status"] == "EXPANDED_CONTEXT"
    assert guard["compact_reasoning_allowed"] is False
    assert guard["unselected_mandatory_fault_ids"] == ["FAULT-C"]
    assert "不提高任何故障" in guard["boundary_zh"]
    assert result["evidence_summary"]["critical_satisfied"] == 1


def test_formal_evidence_summary_counts_support_missing_and_contradiction_exactly():
    bundle = _bundle((
        EvidenceFactStatus.SATISFIED,
        EvidenceFactStatus.MISSING_DATA,
        EvidenceFactStatus.CONTRADICTED,
    ))
    result = _maintenance(_report(bundle))
    summary = result["evidence_summary"]
    assert summary["status"] == "AVAILABLE"
    assert summary["critical_total"] == 3
    assert summary["critical_satisfied"] == 1
    assert summary["critical_missing"] == 1
    assert summary["critical_contradicted"] == 1
    assert [x["evidence_definition_id"] for x in summary["supporting_items"]] == ["ED-1"]
    assert [x["evidence_definition_id"] for x in summary["missing_items"]] == ["ED-2"]
    assert [x["evidence_definition_id"] for x in summary["contradicting_items"]] == ["ED-3"]
    # Missing evidence is deliberately prioritized over rechecking an existing contradiction.
    check = result["next_diagnostic_check"]
    assert check["status"] == "ACTION_REQUIRED"
    assert check["evidence_definition_id"] == "ED-2"
    assert check["auto_executable"] is False
    assert check["blocking_reason"] == "MISSING_OR_INVALID_FORMAL_EVIDENCE"


def test_contradicted_formal_evidence_exposes_only_shadow_margin_probe():
    bundle = _bundle((EvidenceFactStatus.CONTRADICTED,))
    result = _maintenance(_report(bundle))
    check = result["next_diagnostic_check"]
    assert check["status"] == "REVIEW_AVAILABLE"
    assert check["evidence_definition_id"] == "ED-1"
    assert check["auto_executable"] is True
    assert check["probe_id"].startswith("PROBE-MARGIN-")
    assert check["shadow_only"] is True


def test_store_executes_only_current_approved_probe_and_cannot_mutate_formal_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    store = UploadReportJobStore(tmp_path / "jobs")
    job = store.create("maintenance probe")
    bundle = _bundle((EvidenceFactStatus.CONTRADICTED,))
    report = _report(bundle, job_id=job["job_id"])
    monkeypatch.setattr(store, "report", lambda _job_id: report)
    monkeypatch.setattr(store, "visualization", lambda _job_id: {"channels": []})
    monkeypatch.setattr(store, "temporal_localization", lambda _job_id: {"status": "UNAVAILABLE", "first_detection_s": None})
    monkeypatch.setattr(store, "multidimensional_assessment", lambda _job_id: {"status": "UNAVAILABLE", "channels": []})
    maintenance = _maintenance(report)
    approved = maintenance["next_diagnostic_check"]["probe_id"]

    with pytest.raises(ValueError, match="currently approved"):
        store.run_maintenance_diagnostic_probe(job["job_id"], probe_id="PROBE-MARGIN-INVENTED")

    result = store.run_maintenance_diagnostic_probe(job["job_id"], probe_id=approved)
    assert result["status"] == "SUCCEEDED"
    assert result["formal_evidence_eligible"] is False
    assert result["formal_evidence_generated"] is False
    assert result["creates_new_raw_observation"] is False
    assert result["formal_evidence_changed"] is False
    assert result["diagnosis_changed"] is False
    assert result["release_status_changed"] is False
    reconciliation = result["gap_reconciliation"]
    assert reconciliation["evidence_definition_id"] == "ED-1"
    assert reconciliation["formal_gap_open_after"] is True
    assert reconciliation["gap_closed_by_probe"] is False
    assert reconciliation["linkage_status"] == "FORMAL_GAP_STILL_OPEN"
    assert "必须通过受治理输入重新生成正式 EvidenceFact" in reconciliation["next_action_zh"]
    assert result["next_diagnostic_check_after"]["auto_executable"] is False
    assert result["next_diagnostic_check_after"]["blocking_reason"] == "SHADOW_REVIEW_ALREADY_COMPLETED_FORMAL_GAP_STILL_OPEN"
    assert (store._job_root(job["job_id"]) / "output" / "diagnostic_probes" / f"{result['result_sha256']}.json").is_file()
    persisted = store.maintenance(job["job_id"])
    assert persisted["probe_review_history"][0]["result_sha256"] == result["result_sha256"]
    assert persisted["probe_review_history"][0]["formal_gap_open_after"] is True
    assert persisted["next_diagnostic_check"]["auto_executable"] is False
    with pytest.raises(ValueError, match="currently approved"):
        store.run_maintenance_diagnostic_probe(job["job_id"], probe_id=approved)


def test_overview_v3_separates_execution_diagnosis_and_release_status(tmp_path: Path):
    store = UploadReportJobStore(tmp_path / "jobs")
    job = store.create("overview")
    root = store._job_root(job["job_id"])
    report = _report(None, job_id=job["job_id"])
    report_path = root / "output" / "report.json"
    report_path.write_text(__import__("json").dumps(report), encoding="utf-8")
    metadata = store.get(job["job_id"])
    metadata.update({
        "status": "SUCCEEDED", "completed_at": "2026-09-17T00:00:00+00:00",
        "artifact_kind": "PIPELINE_RUNTIME", "report_json": "output/report.json",
    })
    store._save(metadata)

    overview = store.overview()
    assert overview["schema_version"] == "astrograph-platform-overview.v3"
    item = next(x for x in overview["recent_jobs"] if x["job_id"] == job["job_id"])
    assert item["execution_status"] == "SUCCEEDED"
    assert item["diagnosis_disposition"] == "REVIEW_REQUIRED"
    assert item["release_status"] == "FORMAL_RELEASE_BLOCKED"
    assert item["requires_human_review"] is True
    assert overview["diagnosis_disposition_counts"]["REVIEW_REQUIRED"] == 1
    assert overview["release_status_counts"]["FORMAL_RELEASE_BLOCKED"] == 1


def test_diagnostic_contribution_is_unavailable_without_explicit_trajectory_audit():
    report = _report(_bundle((EvidenceFactStatus.SATISFIED,)))
    report["analysis"]["agent_execution"] = {"tool_calls": 5, "status": "SUCCEEDED"}
    result = _maintenance(report)
    audit = result["diagnostic_contribution"]
    assert audit["status"] == "UNAVAILABLE"
    assert audit["probe_execution_count"] == 0
    assert "区分‘执行了多步/工具’" in audit["boundary_zh"]


def test_explicit_diagnostic_contribution_audit_is_operator_visible_without_overclaim():
    report = _report(_bundle((EvidenceFactStatus.CONTRADICTED,)))
    report["analysis"]["reasoning_stage"] = {
        "llm_diagnostic_trajectory": {
            "schema_version": "astrograph-llm-controlled-diagnostic-trajectory.v2",
            "substance_audit": {"status": "SUBSTANTIVE_METHOD_PATH"},
            "contribution_audit": {
                "status": "DERIVED_OBSERVATION_ONLY",
                "probe_execution_count": 1,
                "new_observation_count": 1,
                "new_raw_observation_count": 0,
                "formal_evidence_generated_count": 0,
                "resolved_gap_count": 0,
                "candidate_set_changed": False,
                "decision_changed": False,
                "reason_codes": ["PROBE_RECOMPUTED_ALREADY_ADMITTED_INFORMATION_ONLY"],
            },
        }
    }
    result = _maintenance(report)
    audit = result["diagnostic_contribution"]
    assert audit["status"] == "DERIVED_OBSERVATION_ONLY"
    assert audit["structural_status"] == "SUBSTANTIVE_METHOD_PATH"
    assert audit["probe_execution_count"] == 1
    assert audit["new_raw_observation_count"] == 0
    assert audit["formal_evidence_generated_count"] == 0
    assert audit["unresolved_gap_definition_ids"] == ["ED-1"]
    assert audit["unresolved_gap_count"] == 1
    assert audit["gap_linkage_status"] == "NO_CURRENT_GAP_TARGET_RECORDED"
    assert "没有取得新的原始观测" in audit["operator_summary_zh"]


def test_shadow_probe_is_linked_to_current_gap_without_claiming_gap_closure():
    report = _report(_bundle((EvidenceFactStatus.CONTRADICTED,)))
    report["analysis"]["reasoning_stage"] = {
        "llm_diagnostic_trajectory": {
            "schema_version": "astrograph-llm-controlled-diagnostic-trajectory.v2",
            "steps": [{
                "sequence": 1,
                "proposal": {"capability_id": "execute_diagnostic_probe", "probe_id": "PROBE-MARGIN-TEST", "objective": "recheck"},
                "policy_status": "EXECUTED_SHADOW_PROBE",
                "observation": {
                    "observation_id": "OBS-1", "capability_id": "execute_diagnostic_probe", "artifact_sha256": "a" * 64,
                    "content": {"evidence_definition_id": "ED-1", "formal_evidence_generated": False, "creates_new_raw_observation": False},
                },
            }],
            "substance_audit": {"status": "SUBSTANTIVE_METHOD_PATH"},
            "contribution_audit": {
                "status": "DERIVED_OBSERVATION_ONLY", "probe_execution_count": 1, "new_observation_count": 1,
                "new_raw_observation_count": 0, "formal_evidence_generated_count": 0, "resolved_gap_count": 0,
                "candidate_set_changed": False, "decision_changed": False,
                "reason_codes": ["PROBE_RECOMPUTED_ALREADY_ADMITTED_INFORMATION_ONLY"],
            },
        }
    }
    audit = _maintenance(report)["diagnostic_contribution"]
    assert audit["reviewed_evidence_definition_ids"] == ["ED-1"]
    assert audit["reviewed_current_gap_definition_ids"] == ["ED-1"]
    assert audit["unresolved_gap_definition_ids"] == ["ED-1"]
    assert audit["resolved_gap_count"] == 0
    assert audit["gap_linkage_status"] == "REVIEWED_GAP_STILL_UNRESOLVED"
    assert "没有生成新的正式证据" in audit["gap_linkage_summary_zh"]


def test_store_rebuilds_stale_maintenance_contract_instead_of_serving_stale_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    store = UploadReportJobStore(tmp_path / "jobs")
    job = store.create("maintenance schema migration")
    report = _report(_bundle((EvidenceFactStatus.SATISFIED,)), job_id=job["job_id"])
    path = store._job_root(job["job_id"]) / "output" / "maintenance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"schema_version":"astrograph-maintenance-diagnosis.v3","headline":"stale"}', encoding="utf-8")
    monkeypatch.setattr(store, "report", lambda _job_id: report)
    monkeypatch.setattr(store, "visualization", lambda _job_id: {"channels": []})
    monkeypatch.setattr(store, "temporal_localization", lambda _job_id: {"status": "UNAVAILABLE", "first_detection_s": None})
    monkeypatch.setattr(store, "multidimensional_assessment", lambda _job_id: {"status": "UNAVAILABLE", "channels": []})
    result = store.maintenance(job["job_id"])
    assert result["schema_version"] == "astrograph-maintenance-diagnosis.v8"
    assert "diagnostic_contribution" in result
    persisted = __import__("json").loads(path.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == "astrograph-maintenance-diagnosis.v8"


def test_gap_input_contract_translates_missing_evidence_into_machine_readable_admission_requirements():
    bundle = _bundle((EvidenceFactStatus.SATISFIED, EvidenceFactStatus.MISSING_DATA, EvidenceFactStatus.CONTRADICTED))
    result = _maintenance(_report(bundle))
    contracts = {item["evidence_definition_id"]: item for item in result["evidence_gap_input_contracts"]}
    missing = contracts["ED-2"]
    assert missing["contract_id"] == "INPUT-ED-2"
    assert missing["status"] == "INPUT_REQUIRED"
    assert missing["source_field_template"] == "x2"
    assert missing["resolved_source_field"] == "x2"
    assert missing["required_unit"] == "A"
    assert missing["required_window_start_s"] == 0.0
    assert missing["required_window_end_s"] == 10.0
    assert missing["window_source"] == "EPISODE_CONTRACT"
    assert missing["observed_window_start_s"] == 0.0
    assert missing["observed_window_end_s"] == 10.0
    assert missing["unit_contract_scope"] == "FEATURE_RESULT_OUTPUT"
    assert missing["threshold_unit"] == "A"
    assert missing["input_data_sha256"] == "2" * 64
    requirement = {item["requirement_id"]: item for item in missing["admission_requirements"]}
    assert requirement["SOURCE_FIELD_BOUND"]["status"] == "SATISFIED"
    assert requirement["UNIT_MATCH"]["status"] == "SATISFIED"
    assert requirement["SOURCE_HASH_BOUND"]["status"] == "SATISFIED"
    assert requirement["SAMPLE_COVERAGE"]["status"] == "UNSATISFIED"
    assert requirement["FEATURE_VALUE_AVAILABLE"]["status"] == "UNSATISFIED"
    assert "SAMPLE_COVERAGE_EMPTY" in missing["blockers"]
    assert "FEATURE_VALUE_UNAVAILABLE" in missing["blockers"]
    assert result["next_diagnostic_check"]["input_contract_id"] == "INPUT-ED-2"
    assert "不会自行采集数据" in missing["boundary_zh"]


def test_contradicted_evidence_contract_does_not_demand_recollection_when_existing_input_is_traceable():
    result = _maintenance(_report(_bundle((EvidenceFactStatus.CONTRADICTED,))))
    contract = result["evidence_gap_input_contracts"][0]
    assert contract["status"] == "EXISTING_INPUT_READY_FOR_CONTRACT_REVIEW"
    assert contract["blockers"] == []
    assert contract["sample_count"] == 10
    assert contract["feature_algorithm_version"] == "test-v1"
    assert result["next_diagnostic_check"]["input_contract_id"] == contract["contract_id"]
    assert result["next_diagnostic_check"]["auto_executable"] is True


def test_store_exposes_machine_readable_maintenance_input_contracts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    store = UploadReportJobStore(tmp_path / "jobs")
    job = store.create("input contract API")
    report = _report(_bundle((EvidenceFactStatus.MISSING_DATA,)), job_id=job["job_id"])
    monkeypatch.setattr(store, "report", lambda _job_id: report)
    monkeypatch.setattr(store, "visualization", lambda _job_id: {"channels": []})
    monkeypatch.setattr(store, "temporal_localization", lambda _job_id: {"status": "UNAVAILABLE", "first_detection_s": None})
    monkeypatch.setattr(store, "multidimensional_assessment", lambda _job_id: {"status": "UNAVAILABLE", "channels": []})
    payload = store.maintenance_input_contracts(job["job_id"])
    assert payload["schema_version"] == "astrograph-maintenance-input-contracts.v1"
    assert payload["maintenance_schema_version"] == "astrograph-maintenance-diagnosis.v8"
    assert payload["contract_count"] == 1
    assert payload["next_input_contract_id"] == "INPUT-ED-1"
    assert payload["contracts"][0]["source_field_template"] == "x1"
    assert "does not upload data" in payload["claim_boundary"]


def test_episode_window_mismatch_is_an_input_contract_blocker_even_when_feature_is_traceable():
    bundle = _bundle((EvidenceFactStatus.CONTRADICTED,))
    bundle.feature_results[0] = bundle.feature_results[0].model_copy(update={"window_end_s": 9.0})
    result = _maintenance(_report(bundle))
    contract = result["evidence_gap_input_contracts"][0]
    requirements = {item["requirement_id"]: item for item in contract["admission_requirements"]}
    assert requirements["WINDOW_BOUND"]["status"] == "UNSATISFIED"
    assert contract["required_window_end_s"] == 10.0
    assert contract["observed_window_end_s"] == 9.0
    assert "FEATURE_WINDOW_MISMATCH" in contract["blockers"]
    assert contract["status"] == "INPUT_CONTRACT_REPAIR_REQUIRED"


def test_maintenance_input_contract_endpoint_is_part_of_platform_openapi():
    from fastapi.testclient import TestClient
    from astrograph.api import create_app
    paths = TestClient(create_app()).get("/openapi.json").json()["paths"]
    assert "/v1/platform/jobs/{job_id}/maintenance/input-contracts" in paths
    assert "get" in paths["/v1/platform/jobs/{job_id}/maintenance/input-contracts"]



def _supplemental_ready_bundle(status: EvidenceFactStatus = EvidenceFactStatus.MISSING_DATA) -> EpisodeEvidenceBundle:
    bundle = _bundle((status,))
    bundle.feature_definitions[0] = bundle.feature_definitions[0].model_copy(
        update={"algorithm_version": "builtin-max-value@0.5.13"}
    )
    return bundle


def test_r10_contract_exposes_direct_supplemental_admission_only_for_exact_builtin_version():
    supported = _maintenance(_report(_supplemental_ready_bundle()))["evidence_gap_input_contracts"][0]
    assert supported["supplemental_admission_status"] == "SUPPORTED"
    assert supported["supplemental_operator_id"] == "MAX_VALUE"
    assert supported["supplemental_operator_runtime_version"] == "0.5.13"
    assert supported["supplemental_capsule_schema_version"] == "astrograph-maintenance-supplemental-feature-input.v1"

    unsupported = _maintenance(_report(_bundle((EvidenceFactStatus.MISSING_DATA,))))["evidence_gap_input_contracts"][0]
    assert unsupported["supplemental_admission_status"] == "UNSUPPORTED"
    assert "FROZEN_ALGORITHM_VERSION_NOT_EXACT_BUILTIN_MATCH" in unsupported["supplemental_admission_reason_codes"]


def test_r10_store_admits_contract_bound_input_rebuilds_evidence_overlay_without_mutating_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    store = UploadReportJobStore(tmp_path / "jobs")
    job = store.create("supplemental evidence admission")
    bundle = _supplemental_ready_bundle()
    report = _report(bundle, job_id=job["job_id"])
    monkeypatch.setattr(store, "report", lambda _job_id: report)
    monkeypatch.setattr(store, "visualization", lambda _job_id: {"channels": []})
    monkeypatch.setattr(store, "temporal_localization", lambda _job_id: {"status": "UNAVAILABLE", "first_detection_s": None})
    monkeypatch.setattr(store, "multidimensional_assessment", lambda _job_id: {"status": "UNAVAILABLE", "channels": []})

    maintenance_before = store.maintenance(job["job_id"])
    contract = maintenance_before["evidence_gap_input_contracts"][0]
    assert contract["supplemental_admission_status"] == "SUPPORTED"
    assert maintenance_before["evidence_summary"]["critical_missing"] == 1

    template = store.maintenance_input_contract_template(job["job_id"], contract["contract_id"])
    assert template["schema_version"] == "astrograph-maintenance-supplemental-input-template.v1"
    assert template["support"]["status"] == "SUPPORTED"

    capsule = template["capsule"]
    capsule.update({
        "sample_times_s": [0.0, 5.0, 10.0],
        "current_values": [1.0, 3.0, 4.0],
        "source_note": "operator supplied maintenance recheck",
    })
    result = store.admit_maintenance_feature_input(
        job["job_id"], contract_id=contract["contract_id"], capsule_payload=capsule, actor="tester"
    )
    assert result["status"] == "ADMITTED"
    assert result["evidence_fact_before"]["status"] == "MISSING_DATA"
    assert result["evidence_fact_after"]["status"] == "SATISFIED"
    assert result["hypothesis_status_before"] == "INSUFFICIENT_EVIDENCE"
    assert result["hypothesis_status_after"] == "SUPPORTED"
    assert result["original_report_mutated"] is False
    assert result["formal_release_eligible"] is False
    assert result["feature_result"]["quality_flags"] == []

    # The original frozen report remains untouched; maintenance view uses an explicit overlay.
    assert report["analysis"]["pipeline_result"]["episode_evidence_bundle"]["evidence_facts"][0]["status"] == "MISSING_DATA"
    maintenance_after = store.maintenance(job["job_id"])
    summary = maintenance_after["evidence_summary"]
    assert maintenance_after["schema_version"] == "astrograph-maintenance-diagnosis.v8"
    assert summary["critical_satisfied"] == 1
    assert summary["critical_missing"] == 0
    assert summary["formal_evidence_only"] is False
    assert summary["evidence_source_mode"] == "SUPPLEMENTAL_REANALYSIS_OVERLAY"
    assert summary["supplemental_overlay_receipt_sha256"] == result["receipt_sha256"]
    assert summary["original_release_unchanged"] is True
    assert "重新进入完整生产流水线" in summary["boundary_zh"]

    admissions = store.maintenance_supplemental_admissions(job["job_id"])
    assert admissions["count"] == 1
    assert admissions["admissions"][0]["receipt_sha256"] == result["receipt_sha256"]

    provenance = result["provenance_receipt"]
    assert provenance["admission_receipt_sha256"] == result["receipt_sha256"]
    assert provenance["source_job_id"] == job["job_id"]
    assert provenance["formal_release_eligible"] is False
    provenance_index = store.maintenance_supplemental_provenance(job["job_id"])
    assert provenance_index["count"] == 1
    assert provenance_index["receipts"][0]["receipt_sha256"] == provenance["receipt_sha256"]


def test_r10_supplemental_admission_rejects_wrong_field_unit_window_and_client_result_spoofing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    store = UploadReportJobStore(tmp_path / "jobs")
    job = store.create("supplemental admission guards")
    report = _report(_supplemental_ready_bundle(), job_id=job["job_id"])
    monkeypatch.setattr(store, "report", lambda _job_id: report)
    monkeypatch.setattr(store, "visualization", lambda _job_id: {"channels": []})
    monkeypatch.setattr(store, "temporal_localization", lambda _job_id: {"status": "UNAVAILABLE", "first_detection_s": None})
    monkeypatch.setattr(store, "multidimensional_assessment", lambda _job_id: {"status": "UNAVAILABLE", "channels": []})
    contract = store.maintenance(job["job_id"])["evidence_gap_input_contracts"][0]
    template = store.maintenance_input_contract_template(job["job_id"], contract["contract_id"])["capsule"]
    base = {**template, "sample_times_s": [0.0, 5.0, 10.0], "current_values": [1.0, 3.0, 4.0]}

    for patch, message in [
        ({"resolved_source_field": "wrong"}, "resolved_source_field"),
        ({"unit": "V"}, "unit"),
        ({"window_end_s": 9.0}, "window"),
    ]:
        payload = {**base, **patch}
        with pytest.raises(ValueError, match=message):
            store.admit_maintenance_feature_input(
                job["job_id"], contract_id=contract["contract_id"], capsule_payload=payload, actor="tester"
            )

    # Extra client-provided fields (for example a forged FeatureResult/value) are rejected by the strict capsule schema.
    with pytest.raises(Exception):
        store.admit_maintenance_feature_input(
            job["job_id"], contract_id=contract["contract_id"],
            capsule_payload={**base, "feature_result": {"value": 999.0}}, actor="tester",
        )


def test_r10_maintenance_supplemental_endpoints_are_part_of_platform_openapi():
    from fastapi.testclient import TestClient
    from astrograph.api import create_app
    paths = TestClient(create_app()).get("/openapi.json").json()["paths"]
    template_path = "/v1/platform/jobs/{job_id}/maintenance/input-contracts/{contract_id}/template"
    admit_path = "/v1/platform/jobs/{job_id}/maintenance/input-contracts/{contract_id}/admit-feature-input"
    list_path = "/v1/platform/jobs/{job_id}/maintenance/supplemental-admissions"
    assert "get" in paths[template_path]
    assert "post" in paths[admit_path]
    assert "get" in paths[list_path]


def test_r10_identical_supplemental_capsule_replay_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = UploadReportJobStore(tmp_path / "jobs")
    job = store.create("supplemental replay guard")
    report = _report(_supplemental_ready_bundle(), job_id=job["job_id"])
    monkeypatch.setattr(store, "report", lambda _job_id: report)
    monkeypatch.setattr(store, "visualization", lambda _job_id: {"channels": []})
    monkeypatch.setattr(store, "temporal_localization", lambda _job_id: {"status": "UNAVAILABLE", "first_detection_s": None})
    monkeypatch.setattr(store, "multidimensional_assessment", lambda _job_id: {"status": "UNAVAILABLE", "channels": []})
    contract = store.maintenance(job["job_id"])["evidence_gap_input_contracts"][0]
    capsule = store.maintenance_input_contract_template(job["job_id"], contract["contract_id"])["capsule"]
    capsule.update({"sample_times_s": [0.0, 1.0, 2.0], "current_values": [1.0, 3.0, 4.0]})
    store.admit_maintenance_feature_input(
        job["job_id"], contract_id=contract["contract_id"], capsule_payload=capsule, actor="operator"
    )
    # Even a stale client copy of the old contract cannot replay identical bytes.
    from astrograph.integrations.maintenance_input_admission import SupplementalFeatureInputCapsule, admit_supplemental_feature_input
    overlay, _, _ = store._maintenance_supplemental_overlay(job["job_id"])
    assert overlay is not None
    with pytest.raises(ValueError, match="already admitted"):
        admit_supplemental_feature_input(
            bundle=overlay, contract=contract, capsule=SupplementalFeatureInputCapsule.model_validate(capsule), actor="operator"
        )



def test_oag_report_candidate_is_projected_as_review_only_and_keeps_formal_evidence():
    bundle = _bundle((EvidenceFactStatus.MISSING_DATA,))
    report = {
        "job_id": "JOB-OAG-MAINTENANCE-000000000001",
        "status": "SUCCEEDED",
        "artifact_kind": "OAG_DIAGNOSTIC",
        "formal_diagnosis_publishable": False,
        "analysis": {
            "oag_result": {
                "output": {
                    "summary": {
                        "status": "READY_FOR_EXPLANATION",
                        "suspected_fault_id": "FAULT-A",
                    },
                    "assessment": {
                        "conclusion_status": "SUPPORTED_CANDIDATE",
                        "selected_fault_id": "FAULT-A",
                    },
                }
            },
            "oag_input": {"bundle": bundle.model_dump(mode="json")},
        },
    }
    # The maintenance evidence extractor reads the persisted OAG input through
    # the report-compatible bundle locations used by the platform.  Keep one
    # explicit copy here to isolate candidate projection from store persistence.
    report["analysis"]["episode_evidence_bundle"] = bundle.model_dump(mode="json")
    result = _maintenance(report)
    assert result["headline"] == "FAULT-A"
    assert result["diagnosis_status"] == "REVIEW_REQUIRED"
    assert result["status_zh"] == "需要人工复核"
    assert result["release_status_zh"] == "当前仅用于维护复核，不作为正式发布结论"


def test_oag_insufficient_evidence_projects_to_abstain_without_candidate():
    report = {
        "status": "SUCCEEDED",
        "artifact_kind": "OAG_DIAGNOSTIC",
        "formal_diagnosis_publishable": False,
        "analysis": {
            "oag_result": {
                "output": {
                    "summary": {"status": "INSUFFICIENT_EVIDENCE", "suspected_fault_id": None},
                    "assessment": {"conclusion_status": "UNKNOWN", "selected_fault_id": None},
                }
            }
        },
    }
    result = _maintenance(report)
    assert result["diagnosis_status"] == "ABSTAIN"
    assert result["status_zh"] == "证据不足，暂不下结论"
