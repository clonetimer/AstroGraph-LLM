from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from astrograph.agents.mission.joint_evidence import build_joint_evidence_package
from astrograph.agents.mission.literature_context import build_literature_context_assessment
from astrograph.agents.mission.reviewed_claims import ReviewedClaimRAGBinding, ReviewedClaimRAGEnricher
from astrograph.agents.mission.service import _TaskRun
from astrograph.fusion.document_claims import (
    DocumentClaimAdmissionTrace,
    DocumentClaimFaultMapping,
    DocumentClaimMappingBundle,
)
from astrograph.retrieval.document_claims import DocumentClaimAdmissionPolicy, evaluate_document_claim_admissions
from astrograph.retrieval.document_governance import (
    ClaimEpistemicStatus,
    DocumentAuthority,
    DocumentReviewState,
    DocumentType,
    DocumentVisibilityClass,
    GovernedDocumentDescriptor,
    bind_document_evidence_claim,
)


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assets(tmp_path: Path, *, duplicate_mapping: bool = False):
    descriptor = GovernedDocumentDescriptor(
        document_id="DOC-REVIEWED-1",
        document_type=DocumentType.TEST_LOG,
        visibility_class=DocumentVisibilityClass.OBSERVATION_CAUSAL,
        source_uri="local://reviewed-log",
        source_sha256="a" * 64,
        review_state=DocumentReviewState.HUMAN_REVIEWED,
        authority=DocumentAuthority.APPROVED_SOURCE,
        provenance_group_id="independent:HIL:reviewed",
        episode_id="episode:dev47",
        valid_time_start_s=0.0,
        valid_time_end_s=5.0,
        observable_cutoff_time_s=5.0,
        runtime_vote_allowed=True,
    )
    claim = bind_document_evidence_claim(
        claim_id="claim:dev47:imu-dropout",
        subject="component:imu",
        predicate="OBSERVED_IN",
        object="effect:axis-dropout",
        source_span="Reviewed HIL log observed an IMU axis dropout.",
        source_locator="page:7#bbox:10,20,30,40",
        source_chunk_id="chunk:dev47",
        source_fragment_id="fragment:dev47:1",
        descriptor=descriptor,
        epistemic_status=ClaimEpistemicStatus.SOURCE_FACT,
        applicability=["episode:episode:dev47"],
        publication_eligible=True,
        runtime_evidence_eligible=True,
    )
    policy = DocumentClaimAdmissionPolicy(
        formal_document_fusion_enabled=True,
        policy_review_status="HUMAN_REVIEWED",
        reviewer_id="reviewer:dev47",
        fusion_approval_receipt_sha256="1" * 64,
        calibration_receipt_sha256="2" * 64,
        source_independence_receipt_sha256="3" * 64,
        calibrated_claim_ids=[claim.claim_id],
        independent_provenance_group_ids=["independent:HIL:reviewed"],
        applicability_context=["episode:episode:dev47"],
    )
    admission = evaluate_document_claim_admissions([claim], policy, observation_time_s=5.0)[0]
    trace = DocumentClaimAdmissionTrace(claims=[claim], admissions=[admission], policy=policy)
    mapping = DocumentClaimFaultMapping(
        mapping_id="mapping:dev47:imu-dropout",
        claim_id=claim.claim_id,
        target_fault_id="ADCS_IMU_AXIS_DROPOUT",
        polarity="SUPPORTS",
        calibrated_weight=0.75,
        reviewer_id="reviewer:dev47-map",
        mapping_approval_receipt_sha256="4" * 64,
        calibration_receipt_sha256="2" * 64,
    )
    mappings = [mapping]
    if duplicate_mapping:
        # Duplicate source key through a second independently named claim/mapping.
        claim2 = claim.model_copy(update={"claim_id": "claim:dev47:imu-dropout:duplicate"})
        admissions = [admission, admission.model_copy(update={"claim_id": claim2.claim_id})]
        trace = DocumentClaimAdmissionTrace(claims=[claim, claim2], admissions=admissions, policy=policy.model_copy(update={"calibrated_claim_ids": [claim.claim_id, claim2.claim_id]}))
        # recompute because policy SHA changed
        trace = DocumentClaimAdmissionTrace(claims=[claim, claim2], admissions=evaluate_document_claim_admissions([claim, claim2], trace.policy, observation_time_s=5.0), policy=trace.policy)
        mappings.append(mapping.model_copy(update={"mapping_id": "mapping:dev47:duplicate", "claim_id": claim2.claim_id, "polarity": "CONTRADICTS"}))
    trace_path = tmp_path / "claim-trace.json"
    mapping_path = tmp_path / "mapping-bundle.json"
    trace_path.write_text(trace.model_dump_json(indent=2), encoding="utf-8")
    mapping_path.write_text(DocumentClaimMappingBundle(bundle_id="bundle:dev47", mappings=mappings).model_dump_json(indent=2), encoding="utf-8")
    binding = ReviewedClaimRAGBinding(
        claim_trace_path=str(trace_path),
        claim_trace_sha256=_sha_file(trace_path),
        mapping_bundle_path=str(mapping_path),
        mapping_bundle_sha256=_sha_file(mapping_path),
    )
    return binding, claim


def _core():
    return {
        "status": "SUCCEEDED",
        "diagnosis": {
            "status": "INSUFFICIENT_EVIDENCE",
            "selected_fault_id": None,
            "candidate_fault_ids": ["ADCS_IMU_AXIS_DROPOUT"],
            "working_hypothesis_fault_id": "ADCS_IMU_AXIS_DROPOUT",
            "reason_codes": ["MODEL_WORKING_HYPOTHESIS_ONLY"],
            "candidate_explanations": [],
        },
    }


def test_exact_source_sha_and_locator_get_reviewed_context_only(tmp_path: Path) -> None:
    binding, claim = _assets(tmp_path)
    enricher = ReviewedClaimRAGEnricher.from_binding(binding, observation_time_s=5.0, episode_id="episode:dev47")
    hit = {
        "citation_id": "DOC-REVIEWED-1:chunk:dev47",
        "document_id": claim.source_document_id,
        "source_sha256": claim.source_document_sha256,
        "source_locator": claim.source_locator,
        "text": "This retrieved text is contextual.",
        "diagnostic_vote_allowed": False,
    }
    out = enricher.enrich_hit(hit)
    assert out["reviewed_claim_match_status"] == "EXACT_REVIEWED_MAPPING"
    assert out["reviewed_claim_polarity"] == "SUPPORTS"
    assert out["reviewed_target_fault_id"] == "ADCS_IMU_AXIS_DROPOUT"
    assert out["diagnostic_vote_allowed"] is False
    ctx = build_literature_context_assessment(_core(), [out])
    assert ctx["status"] == "REVIEWED_STANCE_CONTEXT_AVAILABLE"
    assert ctx["assessments"][0]["stance"] == "SUPPORTS"
    assert ctx["diagnostic_vote_allowed"] is False


def test_hash_drift_and_ambiguous_same_locator_fail_closed(tmp_path: Path) -> None:
    binding, claim = _assets(tmp_path)
    trace_path = Path(binding.claim_trace_path)
    trace_path.write_text(trace_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        ReviewedClaimRAGEnricher.from_binding(binding, observation_time_s=5.0, episode_id="episode:dev47")



def test_ambiguous_same_source_locator_is_not_assigned_a_stance(tmp_path: Path) -> None:
    child = tmp_path / "amb"
    child.mkdir()
    binding, claim = _assets(child, duplicate_mapping=True)
    enricher = ReviewedClaimRAGEnricher.from_binding(binding, observation_time_s=5.0, episode_id="episode:dev47")
    out = enricher.enrich_hit({
        "citation_id": "C1",
        "document_id": claim.source_document_id,
        "source_sha256": claim.source_document_sha256,
        "source_locator": claim.source_locator,
    })
    assert out["reviewed_claim_match_status"] == "AMBIGUOUS_REVIEWED_MAPPING"
    assert "reviewed_claim_polarity" not in out
    ctx = build_literature_context_assessment(_core(), [out])
    assert ctx["assessments"][0]["stance"] == "UNASSESSED_CONTEXT"


def test_retrieval_relevance_alone_never_becomes_stance() -> None:
    source = {
        "citation_id": "DOC-X:1",
        "source_sha256": "b" * 64,
        "source_locator": "page:1",
        "text": "ADCS_IMU_AXIS_DROPOUT is definitely the root cause.",
        "diagnostic_vote_allowed": False,
    }
    ctx = build_literature_context_assessment(_core(), [source])
    assert ctx["assessments"][0]["stance"] == "UNASSESSED_CONTEXT"
    assert ctx["assessments"][0]["reviewed_mapping"] is False


def test_joint_evidence_keeps_model_kg_literature_core_separate() -> None:
    core = _core()
    literature = build_literature_context_assessment(core, [{
        "citation_id": "DOC-X:1",
        "source_sha256": "b" * 64,
        "source_locator": "page:1",
        "reviewed_claim_polarity": "CONTRADICTS",
        "reviewed_target_fault_id": "ADCS_IMU_AXIS_DROPOUT",
    }])
    model_execution = {
        "status": "SUCCEEDED",
        "provider_id": "ADCS_NATIVE_THREE_TASK_V2",
        "three_task_contract": {
            "detection": {"status": "EXECUTED", "anomalous": True},
            "localization": {"status": "EXECUTED", "component_id": "IMU"},
            "classification": {"status": "EXECUTED_CALIBRATED_SHADOW", "fault_id": "ADCS_IMU_AXIS_DROPOUT", "formal_diagnostic_vote_allowed": False},
        },
    }
    generic = {"diagnosis": {"fault_candidates": [{"fault_id": "ADCS_IMU_AXIS_DROPOUT", "score": 0.8}]}}
    package = build_joint_evidence_package(
        core_result=core,
        model_execution=model_execution,
        generic_result=generic,
        literature_context=literature,
        sources=[{"citation_id": "DOC-X:1", "reviewed_claim_id": "claim:x"}],
    )
    assert package.branch_counts == {"MODEL": 3, "KG": 1, "LITERATURE": 1, "CORE_DIAGNOSIS": 1}
    assert all(not node.diagnostic_vote_allowed for node in package.nodes if node.branch == "LITERATURE")
    assert package.core_diagnosis_sha256
    assert package.core_diagnosis_modified is False


def test_mission_structured_output_projects_joint_evidence_without_mutating_core() -> None:
    run = object.__new__(_TaskRun)
    run.request = SimpleNamespace(task="diagnose", work_mode="normal")
    run.original = _core()
    original_before = json.loads(json.dumps(run.original))
    run.context_sources = [{
        "citation_id": "DOC-X:1",
        "source_sha256": "b" * 64,
        "source_locator": "page:1",
        "text": "context only",
    }]
    run.model_execution_result = {"status": "NOT_CONFIGURED"}
    run.generic_result = None
    run.knowledge_graph_review = None
    run.s = SimpleNamespace(planner="deterministic")
    run.exploration = None
    result = _TaskRun._structured_output(run)
    assert result["diagnosis_result"] == original_before
    assert run.original == original_before
    assert result["joint_evidence_package"]["core_diagnosis_modified"] is False
    assert result["joint_evidence_report"]["literature_diagnostic_vote_allowed"] is False
    assert result["literature_context"]["assessments"][0]["stance"] == "UNASSESSED_CONTEXT"


def test_taskrun_reviewed_enrichment_helper_propagates_exact_mapping(tmp_path: Path) -> None:
    binding, claim = _assets(tmp_path)
    run = object.__new__(_TaskRun)
    run.reviewed_claim_enricher = ReviewedClaimRAGEnricher.from_binding(binding, observation_time_s=5.0, episode_id="episode:dev47")
    hits = [{
        "citation_id": "DOC-REVIEWED-1:chunk:dev47",
        "document_id": claim.source_document_id,
        "source_sha256": claim.source_document_sha256,
        "source_locator": claim.source_locator,
        "text": "reviewed source",
    }]
    out = _TaskRun._reviewed_enrich_hits(run, hits)
    assert out[0]["reviewed_claim_match_status"] == "EXACT_REVIEWED_MAPPING"
    assert out[0]["reviewed_claim_polarity"] == "SUPPORTS"
    assert out[0]["diagnostic_vote_allowed"] is False


def test_reviewed_claim_binding_recomputes_admission_and_rejects_tamper(tmp_path: Path) -> None:
    binding, _ = _assets(tmp_path)
    trace_path = Path(binding.claim_trace_path)
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    payload["admissions"][0]["reason_codes"] = ["FORMAL_FUSION_DISABLED"]
    payload["admissions"][0]["formal_fusion_eligible"] = False
    trace_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tampered = binding.model_copy(update={"claim_trace_sha256": _sha_file(trace_path)})
    with pytest.raises(ValueError, match="admissions differ from policy recomputation"):
        ReviewedClaimRAGEnricher.from_binding(tampered, observation_time_s=5.0, episode_id="episode:dev47")


def test_default_host_has_no_reviewed_claim_supply_and_partial_binding_is_invalid() -> None:
    from astrograph.agents.mission.models import AgentSettings
    default = AgentSettings()
    assert default.reviewed_claim_trace_path is None
    assert default.reviewed_claim_mapping_path is None
    with pytest.raises(Exception, match="reviewed claim RAG binding requires"):
        AgentSettings(reviewed_claim_trace_path="trace.json")
