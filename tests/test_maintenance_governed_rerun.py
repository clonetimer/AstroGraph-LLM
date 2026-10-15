from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from astrograph.api import create_app
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
from astrograph.fusion import FaultRanking, RankingSource, ScoreSemantics
from astrograph.governance.supplemental_provenance import build_supplemental_input_provenance_receipt
from astrograph.integrations.maintenance_rerun import (
    assess_governed_oag_rerun,
    build_governed_oag_rerun_input,
    canonical_sha256,
    verify_governed_rerun_receipt,
    verify_governed_rerun_completion_receipt,
)
from astrograph.integrations.upload_report import UploadReportJobStore
from astrograph.oag.contracts import OAGInput, content_hash
from astrograph.oag.three_task import EvidenceProjectionReceipt, ThreeTaskResult


def _bundle(*, status: EvidenceFactStatus = EvidenceFactStatus.MISSING_DATA, value: float | None = None) -> EpisodeEvidenceBundle:
    episode = EpisodeWindow(
        episode_id="EP-R11", run_id="RUN-R11", paired_nominal_run_id="RUN-NOM",
        start_s=0.0, end_s=10.0, source="r11-unit-test", source_is_label_free=True,
    )
    fd = FeatureDefinition(
        feature_definition_id="FD-R11", name="R11 max current", source_field_template="adcs.rw.motor_current_a_2",
        aggregation=FeatureAggregation.MAX_VALUE, unit="A", algorithm_version="builtin-max-value@0.5.13",
    )
    fr = FeatureResult(
        feature_result_id="FR-R11", feature_definition_id=fd.feature_definition_id,
        run_id=episode.run_id, paired_nominal_run_id=episode.paired_nominal_run_id,
        episode_id=episode.episode_id, resolved_field="adcs.rw.motor_current_a_2",
        component_instance="rw-2", value=value, unit="A", window_start_s=0.0, window_end_s=10.0,
        sample_count=(3 if value is not None else 0), algorithm_version=fd.algorithm_version,
        input_data_sha256=("a" * 64), quality_flags=([] if value is not None else ["MISSING"]),
    )
    ed = EvidenceDefinition(
        evidence_definition_id="ED-R11", target_fault_id="ADCS_RW_JAM", name="R11 critical current",
        feature_definition_id=fd.feature_definition_id, role=EvidenceRole.REQUIRED,
        comparison=ComparisonOperator.GREATER_OR_EQUAL, threshold=2.0, threshold_unit="A", critical=True,
    )
    ef = EvidenceFact(
        evidence_fact_id="EF-R11", evidence_definition_id=ed.evidence_definition_id,
        target_fault_id=ed.target_fault_id, feature_result_id=fr.feature_result_id,
        episode_id=episode.episode_id, role=EvidenceRole.REQUIRED, status=status,
        observed_value=value, threshold_value=2.0, unit="A",
        comparison=ComparisonOperator.GREATER_OR_EQUAL,
    )
    return EpisodeEvidenceBundle(
        episode=episode, analysis_component_instance="rw-2",
        feature_definitions=[fd], feature_results=[fr], evidence_definitions=[ed], evidence_facts=[ef],
    )


def _oag(bundle: EpisodeEvidenceBundle) -> OAGInput:
    ranking = FaultRanking(
        ranking_id="RANK-R11", episode_id=bundle.episode.episode_id,
        source=RankingSource.DATA_DRIVEN, score_semantics=ScoreSemantics.RANKING_SCORE,
        candidates=[], producer_id="r11-upstream", producer_version="1",
        claim_boundary="Frozen upstream ranking for R11 governed evidence-branch replay test.",
    )
    return OAGInput(bundle=bundle, diagnostic_ranking=ranking, decision_policy="evidence_aware", review_evidence=True)




def _oag_with_native_projection(bundle: EpisodeEvidenceBundle) -> OAGInput:
    base = _oag(bundle)
    projection = EvidenceProjectionReceipt(
        source_window_sha256="1" * 64, nominal_window_sha256="2" * 64,
        model_input_sha256="a" * 64,
        evidence_bundle_sha256=content_hash(bundle.model_dump(mode="json")),
        feature_input_hashes={item.feature_result_id: item.input_data_sha256 for item in bundle.feature_results},
        model_artifact_sha256="3" * 64, evidence_package_sha256="4" * 64,
        window_start_s=bundle.episode.start_s, window_end_s=bundle.episode.end_s,
    )
    heads = ThreeTaskResult(
        episode_id=bundle.episode.episode_id, prediction_id="PRED-R11",
        model_id=base.diagnostic_ranking.producer_id, model_version=base.diagnostic_ranking.producer_version,
        input_data_sha256="a" * 64, window_start_s=bundle.episode.start_s, window_end_s=bundle.episode.end_s,
        detection_probability=0.9, detection_threshold=0.5, detected=True,
        component_probabilities={"rw-2": 1.0}, predicted_component="rw-2",
        classification_ranking=base.diagnostic_ranking, evidence_projection=projection,
    )
    return base.model_copy(update={"three_task_result": heads})


def _overlay(original: EpisodeEvidenceBundle) -> EpisodeEvidenceBundle:
    fr = original.feature_results[0].model_copy(update={
        "value": 4.0, "sample_count": 3, "input_data_sha256": "b" * 64, "quality_flags": [],
    })
    ef = original.evidence_facts[0].model_copy(update={
        "status": EvidenceFactStatus.SATISFIED, "observed_value": 4.0,
    })
    return original.model_copy(update={"feature_results": [fr], "evidence_facts": [ef]})


def _source_report(job_id: str, bundle: EpisodeEvidenceBundle) -> dict:
    return {
        "schema_version": "astrograph-upload-report.v1",
        "job_id": job_id,
        "status": "SUCCEEDED",
        "artifact_kind": "OAG_DIAGNOSTIC",
        "formal_diagnosis_publishable": False,
        "analysis": {
            "episode_evidence_bundle": bundle.model_dump(mode="json"),
            "oag_result": {
                "output": {
                    "summary": {
                        "status": "AMBIGUOUS", "suspected_fault_id": None,
                        "diagnostic_decision": {"candidate_fault_ids": ["ADCS_RW_JAM"]},
                    },
                    "assessment": {
                        "conclusion_status": "ALTERNATIVES", "selected_fault_id": None,
                        "bounded_candidate_fault_ids": ["ADCS_RW_JAM"],
                    },
                }
            },
        },
    }


def _provenance(
    *, source_job_id: str, source_report: dict, bundle: EpisodeEvidenceBundle,
    admission_sha: str, parents: list[str] | None = None,
) -> dict:
    return build_supplemental_input_provenance_receipt(
        source_job_id=source_job_id, source_report=source_report,
        original_evidence_bundle=bundle.model_dump(mode="json"),
        contract={"contract_id": "CONTRACT-R11", "evidence_definition_id": "ED-R11"},
        capsule={
            "contract_id": "CONTRACT-R11", "resolved_source_field": "adcs.rw.motor_current_a_2",
            "unit": "A", "window_start_s": 0.0, "window_end_s": 10.0,
            "sample_times_s": [0.0, 5.0, 10.0], "current_values": [1.0, 3.0, 4.0],
        },
        admission_receipt_sha256=admission_sha,
        parent_admission_receipt_sha256s=list(parents or []), actor="tester",
        admitted_at="2026-09-18T00:00:00+00:00",
    ).model_dump(mode="json")




def _admit_overlay(store: UploadReportJobStore, job_id: str, report: dict, monkeypatch) -> dict:
    monkeypatch.setattr(store, "report", lambda requested: report if requested == job_id else (_ for _ in ()).throw(KeyError(requested)))
    monkeypatch.setattr(store, "visualization", lambda _job_id: {"channels": []})
    monkeypatch.setattr(store, "temporal_localization", lambda _job_id: {"status": "UNAVAILABLE", "first_detection_s": None})
    monkeypatch.setattr(store, "multidimensional_assessment", lambda _job_id: {"status": "UNAVAILABLE", "channels": []})
    maintenance = store.maintenance(job_id)
    assert maintenance["fault_id"] == "ADCS_RW_JAM"
    contract = maintenance["evidence_gap_input_contracts"][0]
    assert contract["supplemental_admission_status"] == "SUPPORTED"
    capsule = store.maintenance_input_contract_template(job_id, contract["contract_id"])["capsule"]
    capsule.update({
        "sample_times_s": [0.0, 5.0, 10.0],
        "current_values": [1.0, 3.0, 4.0],
        "source_note": "R11 governed rerun admission test",
    })
    return store.admit_maintenance_feature_input(
        job_id, contract_id=contract["contract_id"], capsule_payload=capsule, actor="tester"
    )


def test_r11_governed_oag_rerun_replaces_only_evidence_branch_and_remains_review_only():
    original_bundle = _bundle()
    overlay = _overlay(original_bundle)
    original = _oag(original_bundle)
    source_job_id = "JOB-" + "A" * 24
    source_report = _source_report(source_job_id, original_bundle)
    readiness = assess_governed_oag_rerun(
        source_job_id=source_job_id, source_artifact_kind="OAG_DIAGNOSTIC",
        original_input=original, overlay_bundle=overlay, source_report=source_report,
        source_oag_input_sha256="1" * 64, supplemental_receipt_sha256s=["2" * 64],
        supplemental_provenance_receipts=[_provenance(
            source_job_id=source_job_id, source_report=source_report, bundle=original_bundle, admission_sha="2" * 64
        )],
    )
    assert readiness.status == "READY"
    assert readiness.raw_telemetry_rerun_performed is False
    assert readiness.formal_release_eligible is False
    rerun = build_governed_oag_rerun_input(original_input=original, overlay_bundle=overlay)
    assert rerun.bundle.evidence_facts[0].status == EvidenceFactStatus.SATISFIED
    assert rerun.diagnostic_ranking == original.diagnostic_ranking
    assert rerun.fusion_config == original.fusion_config
    assert canonical_sha256(rerun.model_dump(mode="json")) == readiness.rerun_input_sha256


def test_r11_non_oag_source_is_fail_closed_for_governed_rerun():
    original = _oag(_bundle())
    source_job_id = "JOB-R11-NON-OAG"
    source_report = {}
    readiness = assess_governed_oag_rerun(
        source_job_id=source_job_id, source_artifact_kind="PIPELINE_RUNTIME",
        original_input=original, overlay_bundle=_overlay(original.bundle), source_report=source_report,
        source_oag_input_sha256="1" * 64, supplemental_receipt_sha256s=["2" * 64],
        supplemental_provenance_receipts=[_provenance(
            source_job_id=source_job_id, source_report=source_report, bundle=original.bundle, admission_sha="2" * 64
        )],
    )
    assert readiness.status == "BLOCKED"
    assert "SOURCE_ARTIFACT_REQUIRES_RAW_INPUT_RERUN" in readiness.blockers


def test_r11_store_materializes_new_immutable_oag_child_job_without_mutating_parent(tmp_path: Path, monkeypatch):
    store = UploadReportJobStore(tmp_path / "jobs")
    source = store.create("R11 source OAG")
    original_bundle = _bundle()
    original_input = _oag(original_bundle)
    store.put_file(source["job_id"], "diagnostic.json", original_input.model_dump_json().encode("utf-8"))
    source_state = store.get(source["job_id"])
    source_state.update(status="SUCCEEDED", stage="COMPLETED", artifact_kind="OAG_DIAGNOSTIC")
    store._save(source_state)
    report = _source_report(source["job_id"], original_bundle)
    admission = _admit_overlay(store, source["job_id"], report, monkeypatch)
    assert admission["provenance_receipt"]["admission_receipt_sha256"] == admission["receipt_sha256"]

    ready = store.maintenance_governed_rerun_readiness(source["job_id"])
    assert ready["status"] == "READY"
    assert ready["supplemental_provenance_receipt_sha256s"] == [admission["provenance_receipt"]["receipt_sha256"]]
    created = store.create_maintenance_governed_rerun(source["job_id"], actor="tester", start=False)
    child_id = created["child_job_id"]
    assert child_id != source["job_id"]
    assert created["preflight"]["artifact_kind"] == "OAG_DIAGNOSTIC"
    assert created["receipt"]["original_report_mutated"] is False
    assert created["receipt"]["raw_telemetry_rerun_performed"] is False
    assert created["receipt"]["formal_release_eligible"] is False
    assert created["receipt"]["supplemental_provenance_receipt_sha256s"] == [
        admission["provenance_receipt"]["receipt_sha256"]
    ]
    verify_governed_rerun_receipt(created["receipt"])

    child_input = OAGInput.model_validate_json(
        (store._job_root(child_id) / "input" / "maintenance_rerun_oag.json").read_text(encoding="utf-8")
    )
    assert child_input.bundle.evidence_facts[0].status == EvidenceFactStatus.SATISFIED
    assert store.get(source["job_id"])["status"] == "SUCCEEDED"
    assert store.get(child_id)["parent_job_id"] == source["job_id"]
    assert (store._job_root(child_id) / "output" / "maintenance_rerun_receipt.json").is_file()


def test_r11_platform_openapi_exposes_provenance_and_governed_rerun_routes():
    paths = TestClient(create_app()).get("/openapi.json").json()["paths"]
    assert "get" in paths["/v1/platform/jobs/{job_id}/maintenance/supplemental-provenance"]
    assert "get" in paths["/v1/platform/jobs/{job_id}/maintenance/governed-reruns"]
    assert "get" in paths["/v1/platform/jobs/{job_id}/maintenance/governed-rerun-readiness"]
    assert "post" in paths["/v1/platform/jobs/{job_id}/maintenance/governed-rerun"]


def test_r11_native_evidence_projection_is_blocked_from_capsule_based_rerun():
    original_bundle = _bundle()
    original = _oag_with_native_projection(original_bundle)
    source_job_id = "JOB-R11-NATIVE"
    source_report = {}
    readiness = assess_governed_oag_rerun(
        source_job_id=source_job_id, source_artifact_kind="OAG_DIAGNOSTIC", original_input=original,
        overlay_bundle=_overlay(original_bundle), source_report=source_report,
        source_oag_input_sha256="1" * 64, supplemental_receipt_sha256s=["2" * 64],
        supplemental_provenance_receipts=[_provenance(
            source_job_id=source_job_id, source_report=source_report, bundle=original_bundle, admission_sha="2" * 64
        )],
    )
    assert readiness.status == "BLOCKED"
    assert "NATIVE_EVIDENCE_PROJECTION_BOUND_TO_ORIGINAL_BUNDLE" in readiness.blockers
    assert readiness.formal_release_eligible is False
    assert readiness.raw_telemetry_rerun_performed is False


def test_r11_source_job_must_be_succeeded_before_store_can_create_rerun(tmp_path: Path, monkeypatch):
    store = UploadReportJobStore(tmp_path / "jobs")
    source = store.create("R11 unfinished source")
    original_bundle = _bundle()
    original_input = _oag(original_bundle)
    store.put_file(source["job_id"], "diagnostic.json", original_input.model_dump_json().encode("utf-8"))
    report = _source_report(source["job_id"], original_bundle)
    _admit_overlay(store, source["job_id"], report, monkeypatch)
    state = store.get(source["job_id"])
    state.update(status="RUNNING", stage="ANALYSIS", artifact_kind="OAG_DIAGNOSTIC")
    store._save(state)
    readiness = store.maintenance_governed_rerun_readiness(source["job_id"])
    assert readiness["status"] == "BLOCKED"
    assert "SOURCE_JOB_NOT_SUCCEEDED" in readiness["blockers"]


def test_r11_overlay_without_verified_provenance_is_blocked(tmp_path: Path, monkeypatch):
    store = UploadReportJobStore(tmp_path / "jobs")
    source = store.create("R11 fabricated overlay guard")
    original_bundle = _bundle()
    store.put_file(source["job_id"], "diagnostic.json", _oag(original_bundle).model_dump_json().encode("utf-8"))
    state = store.get(source["job_id"])
    state.update(status="SUCCEEDED", stage="COMPLETED", artifact_kind="OAG_DIAGNOSTIC")
    store._save(state)
    report = _source_report(source["job_id"], original_bundle)
    monkeypatch.setattr(store, "report", lambda _job_id: report)
    root = store._maintenance_supplemental_root(source["job_id"]); root.mkdir(parents=True, exist_ok=True)
    (root / "current_overlay.json").write_text(json.dumps({
        "schema_version": "astrograph-maintenance-supplemental-evidence-overlay.v1",
        "updated_at": "2026-09-18T00:00:00+00:00", "latest_receipt_sha256": "c" * 64,
        "admission_receipt_sha256s": ["c" * 64], "bundle": _overlay(original_bundle).model_dump(mode="json"),
        "original_report_mutated": False, "formal_release_eligible": False,
    }), encoding="utf-8")
    readiness = store.maintenance_governed_rerun_readiness(source["job_id"])
    assert readiness["status"] == "BLOCKED"
    assert "NO_SUPPLEMENTAL_PROVENANCE_RECEIPT" in readiness["blockers"]



def test_r11_governed_child_executes_full_graph_compatible_oag_and_stays_review_only(tmp_path: Path, monkeypatch):
    from astrograph.oag.cli import demo_input

    root = Path(__file__).resolve().parents[1]
    original = demo_input(root)
    bundle = original.bundle
    feature_definitions = [
        item.model_copy(update={"algorithm_version": "builtin-max-value@0.5.13"})
        if item.feature_definition_id == "FD-ONLINE-RW-DRAG-V1" else item
        for item in bundle.feature_definitions
    ]
    feature_results = []
    for item in bundle.feature_results:
        if item.feature_definition_id == "FD-ONLINE-RW-DRAG-V1":
            item = item.model_copy(update={
                "algorithm_version": "builtin-max-value@0.5.13",
                "value": None, "sample_count": 0, "quality_flags": ["MISSING"],
            })
        feature_results.append(item)
    evidence_facts = []
    for item in bundle.evidence_facts:
        if item.evidence_definition_id == "ED-ONLINE-RW-FRICTION-DIRECT-V1":
            item = item.model_copy(update={"status": EvidenceFactStatus.MISSING_DATA, "observed_value": None})
        evidence_facts.append(item)
    gap_bundle = bundle.model_copy(update={
        "feature_definitions": feature_definitions,
        "feature_results": feature_results,
        "evidence_facts": evidence_facts,
    })
    source_input = original.model_copy(update={
        "bundle": gap_bundle, "decision_policy": "evidence_aware", "review_evidence": True,
    })

    store = UploadReportJobStore(tmp_path / "jobs")
    source = store.create("R11 full governed rerun")
    store.put_file(source["job_id"], "diagnostic.json", source_input.model_dump_json().encode("utf-8"))
    state = store.get(source["job_id"])
    state.update(status="SUCCEEDED", stage="COMPLETED", artifact_kind="OAG_DIAGNOSTIC")
    store._save(state)
    source_report = {
        "schema_version": "astrograph-upload-report.v1",
        "job_id": source["job_id"], "status": "SUCCEEDED", "artifact_kind": "OAG_DIAGNOSTIC",
        "formal_diagnosis_publishable": False,
        "analysis": {
            "episode_evidence_bundle": gap_bundle.model_dump(mode="json"),
            "oag_result": {"output": {
                "summary": {
                    "status": "AMBIGUOUS", "suspected_fault_id": None,
                    "diagnostic_decision": {"candidate_fault_ids": ["ADCS_RW_FRICTION_INCREASE"]},
                },
                "assessment": {
                    "conclusion_status": "ALTERNATIVES", "selected_fault_id": None,
                    "bounded_candidate_fault_ids": ["ADCS_RW_FRICTION_INCREASE"],
                },
            }},
        },
    }
    original_report_method = store.report
    monkeypatch.setattr(
        store, "report",
        lambda requested: source_report if requested == source["job_id"] else original_report_method(requested),
    )
    monkeypatch.setattr(store, "visualization", lambda _job_id: {"channels": []})
    monkeypatch.setattr(store, "temporal_localization", lambda _job_id: {"status": "UNAVAILABLE", "first_detection_s": None})
    monkeypatch.setattr(store, "multidimensional_assessment", lambda _job_id: {"status": "UNAVAILABLE", "channels": []})

    maintenance = store.maintenance(source["job_id"])
    contract = next(
        item for item in maintenance["evidence_gap_input_contracts"]
        if item["evidence_definition_id"] == "ED-ONLINE-RW-FRICTION-DIRECT-V1"
    )
    assert contract["supplemental_admission_status"] == "SUPPORTED"
    capsule = store.maintenance_input_contract_template(source["job_id"], contract["contract_id"])["capsule"]
    capsule.update({
        "sample_times_s": [30.0, 60.0, 90.0],
        "current_values": [0.0005, 0.002, 0.003],
        "source_note": "R11 graph-compatible full child execution regression",
    })
    admission = store.admit_maintenance_feature_input(
        source["job_id"], contract_id=contract["contract_id"], capsule_payload=capsule, actor="tester"
    )
    assert admission["evidence_fact_after"]["status"] == "SATISFIED"
    assert admission["feature_result"]["quality_flags"] == []

    readiness = store.maintenance_governed_rerun_readiness(source["job_id"])
    assert readiness["status"] == "READY", readiness["blockers"]
    created = store.create_maintenance_governed_rerun(source["job_id"], actor="tester", start=False)
    result = store.run(created["child_job_id"])
    assert result["status"] == "SUCCEEDED", result.get("error")
    assert result["artifact_kind"] == "OAG_DIAGNOSTIC"
    assert result["formal_diagnosis_publishable"] is False
    assert result["rerun_provenance"]["formal_release_eligible"] is False
    assert result["rerun_provenance"]["raw_telemetry_rerun_performed"] is False
    summary = result["analysis"]["oag_result"]["output"]["summary"]
    assert summary["status"] == "READY_FOR_EXPLANATION"
    assert summary["suspected_fault_id"] == "ADCS_RW_FRICTION_INCREASE"
    completion_path = store._job_root(created["child_job_id"]) / "output" / "maintenance_rerun_completion_receipt.json"
    completion = verify_governed_rerun_completion_receipt(
        json.loads(completion_path.read_text(encoding="utf-8"))
    )
    assert completion.child_diagnosis_status == "READY_FOR_EXPLANATION"
    assert completion.child_selected_fault_id == "ADCS_RW_FRICTION_INCREASE"
    assert completion.child_release_status == "FORMAL_RELEASE_BLOCKED"
    assert completion.formal_release_eligible is False
    child_state = store.get(created["child_job_id"])
    assert child_state["rerun_provenance"]["completion_receipt_sha256"] == completion.receipt_sha256
    assert store.get(source["job_id"])["status"] == "SUCCEEDED"


def test_r11_governed_rerun_receipt_detects_lineage_tampering(tmp_path: Path, monkeypatch):
    store = UploadReportJobStore(tmp_path / "jobs")
    source = store.create("R11 receipt tamper")
    bundle = _bundle(); store.put_file(source["job_id"], "diagnostic.json", _oag(bundle).model_dump_json().encode())
    state = store.get(source["job_id"]); state.update(status="SUCCEEDED", stage="COMPLETED", artifact_kind="OAG_DIAGNOSTIC"); store._save(state)
    report = _source_report(source["job_id"], bundle)
    _admit_overlay(store, source["job_id"], report, monkeypatch)
    created = store.create_maintenance_governed_rerun(source["job_id"], actor="tester", start=False)
    receipt = dict(created["receipt"]); receipt["child_job_id"] = "JOB-TAMPERED"
    import pytest
    with pytest.raises(ValueError, match="receipt_sha256 mismatch"):
        verify_governed_rerun_receipt(receipt)


def test_r11_governed_rerun_completion_receipt_detects_tampering(tmp_path: Path, monkeypatch):
    store = UploadReportJobStore(tmp_path / "jobs")
    source = store.create("R11 completion receipt tamper")
    bundle = _bundle()
    store.put_file(source["job_id"], "diagnostic.json", _oag(bundle).model_dump_json().encode())
    state = store.get(source["job_id"]); state.update(status="SUCCEEDED", stage="COMPLETED", artifact_kind="OAG_DIAGNOSTIC"); store._save(state)
    report = _source_report(source["job_id"], bundle)
    _admit_overlay(store, source["job_id"], report, monkeypatch)
    created = store.create_maintenance_governed_rerun(source["job_id"], actor="tester", start=False)
    child_id = created["child_job_id"]
    child_input = OAGInput.model_validate_json(
        (store._job_root(child_id) / "input" / "maintenance_rerun_oag.json").read_text(encoding="utf-8")
    )
    child_report = {
        "job_id": child_id, "formal_diagnosis_publishable": False,
        "rerun_provenance": {"source_job_id": source["job_id"], "receipt_sha256": created["receipt"]["receipt_sha256"]},
        "analysis": {"oag_result": {"output": {"summary": {"status": "AMBIGUOUS", "suspected_fault_id": None}}}},
    }
    from astrograph.integrations.maintenance_rerun import build_governed_rerun_completion_receipt
    completion = build_governed_rerun_completion_receipt(
        prepared_receipt=created["receipt"], source_report=report,
        source_oag_input_sha256=created["receipt"]["source_oag_input_sha256"],
        actual_rerun_input=child_input, child_report=child_report, completed_at="2026-09-18T00:01:00+00:00",
    )
    tampered = dict(completion); tampered["child_release_status"] = "FORMAL_RELEASE_BLOCKED"; tampered["child_diagnosis_status"] = "SUPPORTED"
    import pytest
    with pytest.raises(ValueError, match="completion receipt_sha256 mismatch"):
        verify_governed_rerun_completion_receipt(tampered)


def test_r11_completion_core_rechecks_frozen_source_report_and_input_identity(tmp_path: Path, monkeypatch):
    store = UploadReportJobStore(tmp_path / "jobs")
    source = store.create("R11 frozen-source completion guard")
    bundle = _bundle()
    store.put_file(source["job_id"], "diagnostic.json", _oag(bundle).model_dump_json().encode("utf-8"))
    state = store.get(source["job_id"]); state.update(status="SUCCEEDED", stage="COMPLETED", artifact_kind="OAG_DIAGNOSTIC"); store._save(state)
    report = _source_report(source["job_id"], bundle)
    _admit_overlay(store, source["job_id"], report, monkeypatch)
    created = store.create_maintenance_governed_rerun(source["job_id"], actor="tester", start=False)
    child_id = created["child_job_id"]
    child_input = OAGInput.model_validate_json(
        (store._job_root(child_id) / "input" / "maintenance_rerun_oag.json").read_text(encoding="utf-8")
    )
    child_report = {
        "job_id": child_id, "formal_diagnosis_publishable": False,
        "rerun_provenance": {"source_job_id": source["job_id"], "receipt_sha256": created["receipt"]["receipt_sha256"]},
        "analysis": {"oag_result": {"output": {"summary": {"status": "AMBIGUOUS"}}}},
    }
    from astrograph.integrations.maintenance_rerun import build_governed_rerun_completion_receipt
    import pytest
    with pytest.raises(ValueError, match="source report changed"):
        build_governed_rerun_completion_receipt(
            prepared_receipt=created["receipt"], source_report={**report, "mutated": True},
            source_oag_input_sha256=created["receipt"]["source_oag_input_sha256"],
            actual_rerun_input=child_input, child_report=child_report, completed_at="later",
        )
    with pytest.raises(ValueError, match="source OAG input changed"):
        build_governed_rerun_completion_receipt(
            prepared_receipt=created["receipt"], source_report=report,
            source_oag_input_sha256="f" * 64,
            actual_rerun_input=child_input, child_report=child_report, completed_at="later",
        )


def test_r11_equivalent_governed_rerun_is_idempotently_blocked_until_prior_child_fails(tmp_path: Path, monkeypatch):
    store = UploadReportJobStore(tmp_path / "jobs")
    source = store.create("R11 idempotent governed rerun")
    bundle = _bundle()
    store.put_file(source["job_id"], "diagnostic.json", _oag(bundle).model_dump_json().encode("utf-8"))
    state = store.get(source["job_id"]); state.update(status="SUCCEEDED", stage="COMPLETED", artifact_kind="OAG_DIAGNOSTIC"); store._save(state)
    report = _source_report(source["job_id"], bundle)
    _admit_overlay(store, source["job_id"], report, monkeypatch)

    first = store.create_maintenance_governed_rerun(source["job_id"], actor="tester", start=False)
    child_id = first["child_job_id"]
    readiness = store.maintenance_governed_rerun_readiness(source["job_id"])
    assert readiness["status"] == "BLOCKED"
    assert "EQUIVALENT_GOVERNED_RERUN_EXISTS" in readiness["blockers"]
    assert readiness["equivalent_child_job_id"] == child_id
    assert readiness["equivalent_child_status"] in {"CREATED", "UPLOADING"}

    import pytest
    with pytest.raises(ValueError, match="EQUIVALENT_GOVERNED_RERUN_EXISTS"):
        store.create_maintenance_governed_rerun(source["job_id"], actor="tester", start=False)

    history = store.maintenance_governed_reruns(source["job_id"])
    assert history["count"] == 1
    assert history["reruns"][0]["child_job_id"] == child_id
    assert history["reruns"][0]["completed"] is False

    failed = store.get(child_id); failed.update(status="FAILED", stage="FAILED"); store._save(failed)
    retry_readiness = store.maintenance_governed_rerun_readiness(source["job_id"])
    assert retry_readiness["status"] == "READY"
    assert retry_readiness["equivalent_child_job_id"] is None


def test_r11_governed_rerun_history_binds_completion_receipt(tmp_path: Path, monkeypatch):
    from astrograph.oag.cli import demo_input
    root = Path(__file__).resolve().parents[1]
    original = demo_input(root)
    bundle = original.bundle
    # Use the graph-compatible full execution helper path by introducing one supported missing max-value fact.
    feature_definitions = [
        item.model_copy(update={"algorithm_version": "builtin-max-value@0.5.13"})
        if item.feature_definition_id == "FD-ONLINE-RW-DRAG-V1" else item
        for item in bundle.feature_definitions
    ]
    feature_results = [
        item.model_copy(update={"algorithm_version": "builtin-max-value@0.5.13", "value": None, "sample_count": 0, "quality_flags": ["MISSING"]})
        if item.feature_definition_id == "FD-ONLINE-RW-DRAG-V1" else item
        for item in bundle.feature_results
    ]
    evidence_facts = [
        item.model_copy(update={"status": EvidenceFactStatus.MISSING_DATA, "observed_value": None})
        if item.evidence_definition_id == "ED-ONLINE-RW-FRICTION-DIRECT-V1" else item
        for item in bundle.evidence_facts
    ]
    gap_bundle = bundle.model_copy(update={"feature_definitions": feature_definitions, "feature_results": feature_results, "evidence_facts": evidence_facts})
    source_input = original.model_copy(update={"bundle": gap_bundle, "decision_policy": "evidence_aware", "review_evidence": True})

    store = UploadReportJobStore(tmp_path / "jobs")
    source = store.create("R11 completion history")
    store.put_file(source["job_id"], "diagnostic.json", source_input.model_dump_json().encode("utf-8"))
    state = store.get(source["job_id"]); state.update(status="SUCCEEDED", stage="COMPLETED", artifact_kind="OAG_DIAGNOSTIC"); store._save(state)
    source_report = {
        "schema_version": "astrograph-upload-report.v1", "job_id": source["job_id"], "status": "SUCCEEDED",
        "artifact_kind": "OAG_DIAGNOSTIC", "formal_diagnosis_publishable": False,
        "analysis": {"episode_evidence_bundle": gap_bundle.model_dump(mode="json"), "oag_result": {"output": {
            "summary": {"status": "AMBIGUOUS", "suspected_fault_id": None, "diagnostic_decision": {"candidate_fault_ids": ["ADCS_RW_FRICTION_INCREASE"]}},
            "assessment": {"conclusion_status": "ALTERNATIVES", "selected_fault_id": None, "bounded_candidate_fault_ids": ["ADCS_RW_FRICTION_INCREASE"]},
        }}},
    }
    original_report_method = store.report
    monkeypatch.setattr(store, "report", lambda requested: source_report if requested == source["job_id"] else original_report_method(requested))
    monkeypatch.setattr(store, "visualization", lambda _job_id: {"channels": []})
    monkeypatch.setattr(store, "temporal_localization", lambda _job_id: {"status": "UNAVAILABLE", "first_detection_s": None})
    monkeypatch.setattr(store, "multidimensional_assessment", lambda _job_id: {"status": "UNAVAILABLE", "channels": []})
    maintenance = store.maintenance(source["job_id"])
    contract = next(item for item in maintenance["evidence_gap_input_contracts"] if item["evidence_definition_id"] == "ED-ONLINE-RW-FRICTION-DIRECT-V1")
    capsule = store.maintenance_input_contract_template(source["job_id"], contract["contract_id"])["capsule"]
    capsule.update({"sample_times_s": [30.0, 60.0, 90.0], "current_values": [0.0005, 0.002, 0.003], "source_note": "R11 history"})
    store.admit_maintenance_feature_input(source["job_id"], contract_id=contract["contract_id"], capsule_payload=capsule, actor="tester")
    created = store.create_maintenance_governed_rerun(source["job_id"], actor="tester", start=False)
    result = store.run(created["child_job_id"])
    assert result["status"] == "SUCCEEDED"
    history = store.maintenance_governed_reruns(source["job_id"])
    entry = next(item for item in history["reruns"] if item["child_job_id"] == created["child_job_id"])
    assert entry["completed"] is True
    assert entry["completion_receipt"]["child_release_status"] == "FORMAL_RELEASE_BLOCKED"
    assert entry["completion_receipt"]["formal_release_eligible"] is False


def test_r11_atomic_idempotency_claim_blocks_concurrent_equivalent_creation(tmp_path: Path, monkeypatch):
    store = UploadReportJobStore(tmp_path / "jobs")
    source = store.create("R11 concurrent claim")
    bundle = _bundle()
    store.put_file(source["job_id"], "diagnostic.json", _oag(bundle).model_dump_json().encode("utf-8"))
    state = store.get(source["job_id"]); state.update(status="SUCCEEDED", stage="COMPLETED", artifact_kind="OAG_DIAGNOSTIC"); store._save(state)
    report = _source_report(source["job_id"], bundle)
    _admit_overlay(store, source["job_id"], report, monkeypatch)
    readiness_payload = store.maintenance_governed_rerun_readiness(source["job_id"])
    from astrograph.integrations.maintenance_rerun import MaintenanceGovernedRerunReadiness
    readiness = MaintenanceGovernedRerunReadiness.model_validate({
        key: readiness_payload[key] for key in MaintenanceGovernedRerunReadiness.model_fields
    })
    provenance = readiness_payload["supplemental_provenance_receipt_sha256s"]
    claim_path, _ = store._claim_governed_rerun_creation(source["job_id"], readiness, provenance, actor="tester")
    import pytest
    with pytest.raises(ValueError, match="CREATION_IN_PROGRESS"):
        store._claim_governed_rerun_creation(source["job_id"], readiness, provenance, actor="tester-2")
    claim_path.unlink()

