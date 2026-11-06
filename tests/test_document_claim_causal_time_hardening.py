"""P0 regressions: claim admission and independent ranking use trusted as-of time."""
from __future__ import annotations

import pytest
from astrograph.retrieval.document_claims import evaluate_document_claim_admissions
from astrograph.fusion.document_claims import build_document_claim_forward_ranking
from tests.test_document_claim_fusion_adapter import _eligible_claim_and_admission, _mapping


def _policy_for(claim):
    from astrograph.retrieval.models import DocumentClaimAdmissionPolicy
    return DocumentClaimAdmissionPolicy(
        formal_document_fusion_enabled=True,
        policy_review_status="HUMAN_REVIEWED",
        reviewer_id="reviewer:causal",
        fusion_approval_receipt_sha256="1" * 64,
        calibration_receipt_sha256="2" * 64,
        source_independence_receipt_sha256="3" * 64,
        calibrated_claim_ids=[claim.claim_id],
        independent_provenance_group_ids=[claim.source_provenance_group_id],
        applicability_context=["episode:episode:adapter"],
    )


@pytest.mark.parametrize("asof,expected", [
    (None, "INFERENCE_TIME_REQUIRED"),
    (0.0, "FUTURE_DOCUMENT_EVIDENCE_BLOCKED"),
    (4.999, "FUTURE_DOCUMENT_EVIDENCE_BLOCKED"),
    (5.0, "FORMAL_FUSION_ELIGIBLE"),
    (10.0, "FORMAL_FUSION_ELIGIBLE"),
])
def test_time_bounded_claim_requires_request_time(asof, expected):
    claim, _ = _eligible_claim_and_admission()
    admission = evaluate_document_claim_admissions(
        [claim], _policy_for(claim), observation_time_s=asof
    )[0]
    assert expected in admission.reason_codes
    assert admission.formal_fusion_eligible == (expected == "FORMAL_FUSION_ELIGIBLE")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, True])
def test_time_validation_rejects_invalid_asof(bad):
    claim, _ = _eligible_claim_and_admission()
    with pytest.raises(ValueError, match="observation_time_s"):
        evaluate_document_claim_admissions([claim], _policy_for(claim), observation_time_s=bad)


def _rank(claim, admission, **kwargs):
    return build_document_claim_forward_ranking(
        episode_id=kwargs.pop("episode_id", "episode:adapter"),
        claims=[claim], admissions=[admission], mappings=[_mapping(claim.claim_id)],
        registered_fault_ids=["FAULT_A", "FAULT_B"],
        source_artifact_id="artifact:causal", source_artifact_sha256="5" * 64,
        **kwargs,
    )


def test_builder_refuses_future_or_missing_time_even_with_previously_eligible_admission():
    claim, admission = _eligible_claim_and_admission()
    with pytest.raises(ValueError, match="observation_time_s"):
        _rank(claim, admission)
    with pytest.raises(ValueError, match="future observation"):
        _rank(claim, admission, observation_time_s=4.99)
    result = _rank(claim, admission, observation_time_s=5.0)
    assert result.consumed_claim_ids == [claim.claim_id]
    assert result.observation_time_s == 5.0
    assert result.ranking.metadata["observation_time_s"] == 5.0
    result_later = _rank(claim, admission, observation_time_s=8.0)
    assert result_later.ranking.ranking_id != result.ranking.ranking_id


def test_builder_refuses_cross_episode_or_forged_review():
    claim, admission = _eligible_claim_and_admission()
    with pytest.raises(ValueError, match="requested episode"):
        _rank(claim, admission, episode_id="episode:other", observation_time_s=5.0)
    forged = claim.model_copy(update={"source_runtime_vote_allowed": False})
    with pytest.raises(ValueError, match="approved runtime authority"):
        _rank(forged, admission, observation_time_s=5.0)


def test_causal_stance_enricher_requires_inference_time(tmp_path):
    from astrograph.agents.mission.reviewed_claims import ReviewedClaimRAGEnricher
    from astrograph.fusion.document_claims import DocumentClaimAdmissionTrace, DocumentClaimMappingBundle
    from astrograph.agents.mission.reviewed_claims import ReviewedClaimRAGBinding
    import hashlib
    claim, admission = _eligible_claim_and_admission()
    trace = DocumentClaimAdmissionTrace(claims=[claim], admissions=evaluate_document_claim_admissions(
        [claim], _policy_for(claim), observation_time_s=5.0
    ), policy=_policy_for(claim))
    mapping = DocumentClaimMappingBundle(bundle_id="bundle:causal", mappings=[_mapping(claim.claim_id)])
    trace_path = tmp_path / "trace.json"; map_path = tmp_path / "map.json"
    trace_path.write_text(trace.model_dump_json(), encoding="utf-8")
    map_path.write_text(mapping.model_dump_json(), encoding="utf-8")
    binding = ReviewedClaimRAGBinding(
        claim_trace_path=str(trace_path), claim_trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        mapping_bundle_path=str(map_path), mapping_bundle_sha256=hashlib.sha256(map_path.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="trusted observation_time_s"):
        ReviewedClaimRAGEnricher.from_binding(binding)
    with pytest.raises(ValueError, match="differ from policy recomputation"):
        ReviewedClaimRAGEnricher.from_binding(binding, observation_time_s=4.0, episode_id="episode:adapter")
    assert ReviewedClaimRAGEnricher.from_binding(binding, observation_time_s=5.0, episode_id="episode:adapter").reviewed_mapping_count == 1
    assert ReviewedClaimRAGEnricher.from_binding(binding, observation_time_s=5.0, episode_id="episode:other").reviewed_mapping_count == 0


def test_online_shadow_rejects_future_claim_from_same_episode(tmp_path):
    import hashlib
    from astrograph.fusion.document_claims import (
        DocumentClaimAdmissionTrace, DocumentClaimMappingBundle,
        DocumentClaimOnlineShadowConfig, build_online_document_claim_shadow,
    )
    claim, _ = _eligible_claim_and_admission()
    policy = _policy_for(claim)
    stored_admissions = evaluate_document_claim_admissions(
        [claim], policy, observation_time_s=5.0
    )
    trace = DocumentClaimAdmissionTrace(claims=[claim], admissions=stored_admissions, policy=policy)
    mapping = DocumentClaimMappingBundle(bundle_id="bundle:causal", mappings=[_mapping(claim.claim_id)])
    trace_path = tmp_path / "trace.json"; map_path = tmp_path / "map.json"
    trace_path.write_text(trace.model_dump_json(), encoding="utf-8")
    map_path.write_text(mapping.model_dump_json(), encoding="utf-8")
    config = DocumentClaimOnlineShadowConfig(
        enabled=True,
        claim_trace_path=str(trace_path),
        claim_trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        mapping_bundle_path=str(map_path),
        mapping_bundle_sha256=hashlib.sha256(map_path.read_bytes()).hexdigest(),
        source_artifact_id="artifact:causal",
    )
    kwargs = dict(config=config, episode_id="episode:adapter", registered_fault_ids=["FAULT_A", "FAULT_B"])
    with pytest.raises(ValueError, match="trusted finite observation_time_s"):
        build_online_document_claim_shadow(**kwargs)
    with pytest.raises(ValueError, match="differ from policy recomputation"):
        build_online_document_claim_shadow(**kwargs, observation_time_s=4.99)
    accepted = build_online_document_claim_shadow(**kwargs, observation_time_s=5.0)
    assert accepted.status == "SHADOW_RANKING_READY"
    assert accepted.candidate_fusion_modified is False
    with pytest.raises(ValueError, match="not applicable to online episode"):
        build_online_document_claim_shadow(
            config=config, episode_id="episode:other", registered_fault_ids=["FAULT_A", "FAULT_B"],
            observation_time_s=5.0,
        )