from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from astrograph.fusion import (
    BiRRFConfig,
    FaultRanking,
    RankedFault,
    RankingSource,
    ScoreSemantics,
    fuse_rankings,
)
from astrograph.fusion.document_claims import (
    DocumentClaimAdmissionTrace,
    DocumentClaimFaultMapping,
    DocumentClaimMappingBundle,
    DocumentClaimOnlineShadowConfig,
    build_online_document_claim_shadow,
    build_document_claim_forward_ranking,
)
from astrograph.retrieval.document_claims import (
    DocumentClaimAdmissionPolicy,
    evaluate_document_claim_admissions,
)
from astrograph.retrieval.document_governance import (
    ClaimEpistemicStatus,
    DocumentAuthority,
    DocumentReviewState,
    DocumentType,
    DocumentVisibilityClass,
    GovernedDocumentDescriptor,
    bind_document_evidence_claim,
)


def _eligible_claim_and_admission():
    descriptor = GovernedDocumentDescriptor(
        document_id="log:independent:adapter",
        document_type=DocumentType.TEST_LOG,
        visibility_class=DocumentVisibilityClass.OBSERVATION_CAUSAL,
        source_uri="local://independent-adapter-log",
        source_sha256="a" * 64,
        review_state=DocumentReviewState.HUMAN_REVIEWED,
        authority=DocumentAuthority.APPROVED_SOURCE,
        provenance_group_id="independent:HIL:adapter",
        episode_id="episode:adapter",
        valid_time_start_s=0.0,
        valid_time_end_s=5.0,
        observable_cutoff_time_s=5.0,
        runtime_vote_allowed=True,
    )
    claim = bind_document_evidence_claim(
        claim_id="claim:adapter:wheel-current-rise",
        subject="component:wheel",
        predicate="OBSERVED_IN",
        object="effect:current-rise",
        source_span="Independent HIL log recorded a current rise.",
        source_locator="page:1",
        source_chunk_id="chunk:adapter",
        source_fragment_id="fragment:adapter:1",
        descriptor=descriptor,
        epistemic_status=ClaimEpistemicStatus.SOURCE_FACT,
        applicability=["episode:episode:adapter"],
        publication_eligible=True,
        runtime_evidence_eligible=True,
    )
    policy = DocumentClaimAdmissionPolicy(
        formal_document_fusion_enabled=True,
        policy_review_status="HUMAN_REVIEWED",
        reviewer_id="reviewer:adapter",
        fusion_approval_receipt_sha256="1" * 64,
        calibration_receipt_sha256="2" * 64,
        source_independence_receipt_sha256="3" * 64,
        calibrated_claim_ids=[claim.claim_id],
        independent_provenance_group_ids=["independent:HIL:adapter"],
        applicability_context=["episode:episode:adapter"],
    )
    return claim, evaluate_document_claim_admissions([claim], policy)[0]


def test_document_fusion_requires_every_scope_constraint_and_episode_binding():
    claim, _ = _eligible_claim_and_admission()
    policy = DocumentClaimAdmissionPolicy(
        formal_document_fusion_enabled=True,
        policy_review_status='HUMAN_REVIEWED', reviewer_id='reviewer:adapter',
        fusion_approval_receipt_sha256='1'*64,
        calibration_receipt_sha256='2'*64,
        source_independence_receipt_sha256='3'*64,
        calibrated_claim_ids=[claim.claim_id],
        independent_provenance_group_ids=[claim.source_provenance_group_id],
        applicability_context=['episode:episode:adapter','graph-scope:SAT-X'],
    )
    mismatch = claim.model_copy(update={
        'applicability':['episode:episode:OTHER','graph-scope:SAT-X']})
    result = evaluate_document_claim_admissions([mismatch],policy)[0]
    assert not result.formal_fusion_eligible
    assert 'APPLICABILITY_NOT_ESTABLISHED' in result.reason_codes
    missing_episode = claim.model_copy(update={'applicability':['graph-scope:SAT-X']})
    result = evaluate_document_claim_admissions([missing_episode],policy)[0]
    assert not result.formal_fusion_eligible
    assert 'OBSERVATION_EPISODE_BINDING_MISSING' in result.reason_codes
    valid = claim.model_copy(update={
        'applicability':['episode:episode:adapter','graph-scope:SAT-X']})
    assert evaluate_document_claim_admissions([valid],policy)[0].formal_fusion_eligible
    # Defensive guard even when a deserialized legacy claim claims voting
    # eligibility despite its train-only origin.
    quarantined = valid.model_copy(update={
        'source_visibility_class':DocumentVisibilityClass.RESOLVED_CASE_TRAIN_ONLY})
    result = evaluate_document_claim_admissions([quarantined],policy)[0]
    assert not result.formal_fusion_eligible
    assert 'SOURCE_TRUTH_OR_TRAIN_ONLY_NOT_RUNTIME_ELIGIBLE' in result.reason_codes
    low_authority = valid.model_copy(update={
        'source_authority':DocumentAuthority.ENGINEERING_REFERENCE})
    result = evaluate_document_claim_admissions([low_authority],policy)[0]
    assert not result.formal_fusion_eligible
    assert 'SOURCE_AUTHORITY_NOT_APPROVED' in result.reason_codes
    unbound = valid.model_copy(update={'source_visibility_class':None})
    result = evaluate_document_claim_admissions([unbound],policy)[0]
    assert not result.formal_fusion_eligible
    assert 'SOURCE_VISIBILITY_NOT_BOUND' in result.reason_codes


def _mapping(claim_id: str) -> DocumentClaimFaultMapping:
    return DocumentClaimFaultMapping(
        mapping_id="mapping:adapter:wheel-current-rise-to-fault-b",
        claim_id=claim_id,
        target_fault_id="FAULT_B",
        polarity="SUPPORTS",
        calibrated_weight=0.8,
        reviewer_id="reviewer:mapping",
        mapping_approval_receipt_sha256="4" * 64,
        calibration_receipt_sha256="2" * 64,
    )


def _write_json(path, payload) -> str:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_admitted_exact_mapping_builds_lineage_complete_kg_forward_ranking() -> None:
    claim, admission = _eligible_claim_and_admission()
    build = build_document_claim_forward_ranking(
        episode_id="episode:adapter",
        claims=[claim],
        admissions=[admission],
        mappings=[_mapping(claim.claim_id)],
        registered_fault_ids=["FAULT_A", "FAULT_B"],
        source_artifact_id="artifact:document-claim-adapter",
        source_artifact_sha256="5" * 64,
    )
    ranking = build.ranking
    assert ranking.source == RankingSource.KG_FORWARD
    assert ranking.kg_fault_coverage is not None
    assert ranking.kg_fault_coverage.coverage_basis == (
        "REVIEWED_CALIBRATED_DOCUMENT_CLAIM_MAPPINGS"
    )
    by_fault = {item.fault_id: item for item in ranking.candidates}
    assert by_fault["FAULT_B"].score == 0.8
    assert by_fault["FAULT_B"].contributes_to_fusion is True
    assert by_fault["FAULT_B"].source_evidence_ids == [
        f"DOCUMENT_CLAIM:{claim.claim_id}"
    ]
    assert by_fault["FAULT_B"].correlation_group_ids == [
        "DOCUMENT_PROVENANCE:independent:HIL:adapter"
    ]

    data = FaultRanking(
        ranking_id="ranking:data:adapter",
        episode_id="episode:adapter",
        source=RankingSource.DATA_DRIVEN,
        score_semantics=ScoreSemantics.UNCALIBRATED_CLASS_PROBABILITY,
        candidates=[
            RankedFault(fault_id="FAULT_A", rank=1, score=0.6, source=RankingSource.DATA_DRIVEN),
            RankedFault(fault_id="FAULT_B", rank=2, score=0.4, source=RankingSource.DATA_DRIVEN),
        ],
        producer_id="fixture-data",
        producer_version="1",
    )
    backward = FaultRanking(
        ranking_id="ranking:backward:adapter",
        episode_id="episode:adapter",
        source=RankingSource.KG_BACKWARD,
        score_semantics=ScoreSemantics.EVIDENCE_STRENGTH,
        candidates=[],
        producer_id="fixture-backward",
        producer_version="1",
    )
    fused = fuse_rankings(data, ranking, backward, config=BiRRFConfig())
    assert fused.episode_id == "episode:adapter"
    assert any(item.fault_id == "FAULT_B" for item in fused.candidates)


def test_unmapped_eligible_claim_is_rejected_instead_of_silently_dropped() -> None:
    claim, admission = _eligible_claim_and_admission()
    with pytest.raises(ValueError, match="lack exact fault mappings"):
        build_document_claim_forward_ranking(
            episode_id="episode:adapter",
            claims=[claim],
            admissions=[admission],
            mappings=[],
            registered_fault_ids=["FAULT_A", "FAULT_B"],
            source_artifact_id="artifact:document-claim-adapter",
            source_artifact_sha256="5" * 64,
        )

    mismatched = _mapping(claim.claim_id).model_copy(
        update={"calibration_receipt_sha256": "9" * 64}
    )
    with pytest.raises(ValueError, match="different calibration receipts"):
        build_document_claim_forward_ranking(
            episode_id="episode:adapter",
            claims=[claim],
            admissions=[admission],
            mappings=[mismatched],
            registered_fault_ids=["FAULT_A", "FAULT_B"],
            source_artifact_id="artifact:document-claim-adapter",
            source_artifact_sha256="5" * 64,
        )


def test_ineligible_claim_produces_no_document_vote() -> None:
    claim, _ = _eligible_claim_and_admission()
    admission = evaluate_document_claim_admissions(
        [claim], DocumentClaimAdmissionPolicy()
    )[0]
    build = build_document_claim_forward_ranking(
        episode_id="episode:adapter",
        claims=[claim],
        admissions=[admission],
        mappings=[_mapping(claim.claim_id)],
        registered_fault_ids=["FAULT_A", "FAULT_B"],
        source_artifact_id="artifact:document-claim-adapter",
        source_artifact_sha256="5" * 64,
    )
    assert build.consumed_claim_ids == []
    assert all(not item.contributes_to_fusion for item in build.ranking.candidates)
    assert build.ranking.kg_fault_coverage.supported_fault_ids == ()


def test_disabled_online_shadow_has_no_artifact_binding_or_runtime_effect() -> None:
    result = build_online_document_claim_shadow(
        config=DocumentClaimOnlineShadowConfig(),
        episode_id="episode:does-not-read-files",
        registered_fault_ids=["FAULT_A"],
    )
    assert result.status == "DISABLED"
    assert result.ranking_build is None
    assert result.candidate_fusion_modified is False
    with pytest.raises(ValidationError, match="must not bind artifact paths"):
        DocumentClaimOnlineShadowConfig(
            enabled=False,
            claim_trace_path="does-not-exist.json",
        )


def test_online_shadow_recomputes_admission_and_builds_reviewed_ranking(tmp_path) -> None:
    claim, admission = _eligible_claim_and_admission()
    policy = DocumentClaimAdmissionPolicy(
        formal_document_fusion_enabled=True,
        policy_review_status="HUMAN_REVIEWED",
        reviewer_id="reviewer:adapter",
        fusion_approval_receipt_sha256="1" * 64,
        calibration_receipt_sha256="2" * 64,
        source_independence_receipt_sha256="3" * 64,
        calibrated_claim_ids=[claim.claim_id],
        independent_provenance_group_ids=["independent:HIL:adapter"],
        applicability_context=["episode:episode:adapter"],
    )
    trace = DocumentClaimAdmissionTrace(
        claims=[claim], admissions=[admission], policy=policy
    )
    mappings = DocumentClaimMappingBundle(
        bundle_id="mapping-bundle:test", mappings=[_mapping(claim.claim_id)]
    )
    trace_path = tmp_path / "claim_trace.json"
    mapping_path = tmp_path / "mappings.json"
    trace_sha = _write_json(trace_path, trace.model_dump(mode="json"))
    mapping_sha = _write_json(mapping_path, mappings.model_dump(mode="json"))
    result = build_online_document_claim_shadow(
        config=DocumentClaimOnlineShadowConfig(
            enabled=True,
            claim_trace_path=str(trace_path),
            claim_trace_sha256=trace_sha,
            mapping_bundle_path=str(mapping_path),
            mapping_bundle_sha256=mapping_sha,
            source_artifact_id="artifact:test",
        ),
        episode_id="episode:adapter",
        registered_fault_ids=["FAULT_A", "FAULT_B"],
    )
    assert result.status == "SHADOW_RANKING_READY"
    assert result.ranking_build.consumed_claim_ids == [claim.claim_id]
    assert result.candidate_fusion_modified is False


def test_online_shadow_rejects_hash_and_admission_tampering(tmp_path) -> None:
    claim, admission = _eligible_claim_and_admission()
    policy = DocumentClaimAdmissionPolicy()
    trace = DocumentClaimAdmissionTrace(
        claims=[claim], admissions=[admission], policy=policy
    )
    mappings = DocumentClaimMappingBundle(bundle_id="mapping-bundle:test")
    trace_path = tmp_path / "claim_trace.json"
    mapping_path = tmp_path / "mappings.json"
    trace_sha = _write_json(trace_path, trace.model_dump(mode="json"))
    mapping_sha = _write_json(mapping_path, mappings.model_dump(mode="json"))
    config = DocumentClaimOnlineShadowConfig(
        enabled=True,
        claim_trace_path=str(trace_path),
        claim_trace_sha256=trace_sha,
        mapping_bundle_path=str(mapping_path),
        mapping_bundle_sha256=mapping_sha,
        source_artifact_id="artifact:test",
    )
    with pytest.raises(ValueError, match="differ from policy recomputation"):
        build_online_document_claim_shadow(
            config=config,
            episode_id="episode:adapter",
            registered_fault_ids=["FAULT_A"],
        )
    with pytest.raises(ValueError, match="trace SHA-256 mismatch"):
        build_online_document_claim_shadow(
            config=config.model_copy(update={"claim_trace_sha256": "f" * 64}),
            episode_id="episode:adapter",
            registered_fault_ids=["FAULT_A"],
        )
