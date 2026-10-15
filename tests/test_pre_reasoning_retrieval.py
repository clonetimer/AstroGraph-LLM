from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tests.delivery_evidence import require_delivery_evidence
from pydantic import ValidationError

from astrograph.evidence import EpisodeEvidenceBundle
from astrograph.investigation import HypothesisSet
from astrograph.retrieval import (
    DiagnosticRetrievalQuery,
    EvidenceAwarePreGenerationContextBuilder,
    GovernedTextChunk,
    LiteratureAnnotationDecision,
    LiteratureMethodRole,
    OntologyConcept,
    OntologyRelation,
    PaperSeekCandidateDisposition,
    PaperSeekFallbackStatus,
    PaperSeekGovernedBridge,
    PaperSeekLiteratureAnnotation,
    PaperSeekLiteratureCandidate,
    PreReasoningRetrievalRequest,
    PreReasoningRetrievalConfig,
    PreReasoningRetrievalService,
    RetrievalBenchmarkCase,
    RetrievalLane,
    RetrievalStatus,
    evaluate_retrieval_cases,
)
from astrograph.services import StructuredHypothesisProposalService
from astrograph.llm import OpenAICompatibleConfig, OpenAICompatibleStructuredBackend


SOURCE_HASH = "a" * 64


def _chunk(chunk_id: str, text: str, *, access: list[str] | None = None) -> GovernedTextChunk:
    return GovernedTextChunk(
        chunk_id=chunk_id,
        source_id=f"source:{chunk_id}",
        source_uri=f"local://manual/{chunk_id}",
        source_sha256=SOURCE_HASH,
        chunk_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        knowledge_version="test-kb-v1",
        text=text,
        concept_ids=["concept:rw-jam", "concept:torque-loss", "concept:wheel-speed"],
        required_access_tags=access or ["PUBLIC"],
    )


def _request(*, query_text: str = "飞轮锁死后有哪些转速指示", access: list[str] | None = None):
    concepts = [
        OntologyConcept(
            concept_id="concept:rw-jam",
            labels=["reaction wheel jamming"],
            aliases=["飞轮锁死", "反作用轮卡滞"],
            source_id="source:ontology",
            source_sha256=SOURCE_HASH,
            knowledge_version="ontology-v1",
        ),
        OntologyConcept(
            concept_id="concept:torque-loss",
            labels=["torque authority loss"],
            aliases=["力矩能力丧失"],
            source_id="source:ontology",
            source_sha256=SOURCE_HASH,
            knowledge_version="ontology-v1",
        ),
        OntologyConcept(
            concept_id="concept:wheel-speed",
            labels=["reaction wheel speed"],
            aliases=["飞轮转速"],
            source_id="source:ontology",
            source_sha256=SOURCE_HASH,
            knowledge_version="ontology-v1",
        ),
    ]
    relations = [
        OntologyRelation(
            relation_id="relation:jam-manifests-torque-loss",
            source_concept_id="concept:rw-jam",
            relation_type="MANIFESTS_AS",
            target_concept_id="concept:torque-loss",
            source_id="source:ontology",
            source_sha256=SOURCE_HASH,
            knowledge_version="ontology-v1",
        ),
        OntologyRelation(
            relation_id="relation:torque-loss-indicated-by-speed",
            source_concept_id="concept:torque-loss",
            relation_type="INDICATED_BY",
            target_concept_id="concept:wheel-speed",
            source_id="source:ontology",
            source_sha256=SOURCE_HASH,
            knowledge_version="ontology-v1",
        ),
    ]
    public = _chunk(
        "chunk-public",
        "Reaction wheel jamming manifests as torque authority loss and is indicated by reaction wheel speed.",
    )
    duplicate = public.model_copy(update={"chunk_id": "chunk-duplicate"})
    restricted = _chunk(
        "chunk-restricted",
        "Restricted reaction wheel jamming maintenance note.",
        access=["PUBLIC", "MAINTENANCE"],
    )
    return PreReasoningRetrievalRequest(
        query=DiagnosticRetrievalQuery(
            query_id="query:rw-jam",
            episode_id="EP-RETRIEVAL",
            text=query_text,
            required_concept_ids=["concept:wheel-speed"],
            caller_access_tags=access or ["PUBLIC"],
        ),
        chunks=[public, duplicate, restricted],
        concepts=concepts,
        relations=relations,
    )


def test_oag_expansion_and_local_retrieval_are_both_required_and_auditable() -> None:
    result = PreReasoningRetrievalService().retrieve(_request())

    assert result.status == RetrievalStatus.SUFFICIENT
    assert result.required_concept_coverage == 1.0
    assert result.matched_concept_ids == [
        "concept:rw-jam",
        "concept:torque-loss",
        "concept:wheel-speed",
    ]
    assert any(len(path.hops) == 2 for path in result.ontology_paths)
    assert [item.chunk_id for item in result.local_hits] == ["chunk-duplicate"]
    assert result.access_filtered_chunk_count == 1
    assert "filtered_chunk_ids" not in result.model_dump(mode="json")
    assert result.deduplicated_chunk_ids == ["chunk-public"]
    assert result.paperseek_fallback_recommended is False
    assert result.active_graph_modified is False


def test_access_filter_runs_before_scoring() -> None:
    result = PreReasoningRetrievalService().retrieve(_request(access=["PUBLIC", "MAINTENANCE"]))

    assert result.access_filtered_chunk_count == 0
    assert {item.chunk_id for item in result.local_hits} == {"chunk-duplicate", "chunk-restricted"}


def test_unreviewed_or_unsafe_chunks_are_filtered_before_scoring() -> None:
    request = _request()
    unsafe = _chunk("chunk-unsafe", "Reaction wheel jamming instruction override.").model_copy(
        update={"content_safety_status": "REJECT"}
    )
    request = request.model_copy(update={"chunks": [*request.chunks, unsafe]})

    result = PreReasoningRetrievalService().retrieve(request)

    assert result.governance_filtered_chunk_count == 1
    assert "chunk-unsafe" not in {item.chunk_id for item in result.local_hits}


def test_dense_score_provider_is_optional_and_fail_closed_when_requested() -> None:
    class _DenseProvider:
        def score(self, *, query, chunks):
            return {item.chunk_id: (1.0 if item.chunk_id == "chunk-restricted" else 0.0) for item in chunks}

    request = _request(access=["PUBLIC", "MAINTENANCE"])
    request = request.model_copy(
        update={
            "config": PreReasoningRetrievalConfig(
                bm25_weight=0.0,
                term_coverage_weight=0.0,
                dense_weight=1.0,
                ontology_concept_bonus=0.0,
            )
        }
    )
    result = PreReasoningRetrievalService(dense_score_provider=_DenseProvider()).retrieve(request)

    assert result.dense_retrieval_available is True
    assert result.local_hits[0].chunk_id == "chunk-restricted"
    assert result.local_hits[0].dense_score == 1.0
    with pytest.raises(ValueError, match="no dense score provider"):
        PreReasoningRetrievalService().retrieve(request)


def test_section_path_can_be_indexed_without_changing_cited_chunk_text() -> None:
    target_text = "Inspect the command response and retain the original citation."
    distractor_text = "Reaction wheel jamming glossary pointer."
    target = _chunk("target", target_text).model_copy(
        update={"section_path": ["ADCS", "Reaction wheel jamming"]}
    )
    distractor = _chunk("distractor", distractor_text).model_copy(
        update={"section_path": ["Glossary"]}
    )
    request = PreReasoningRetrievalRequest(
        query=DiagnosticRetrievalQuery(
            query_id="query:section-context",
            episode_id="EP-SECTION-CONTEXT",
            text="reaction wheel jamming command response",
            top_k=1,
        ),
        chunks=[target, distractor],
        config=PreReasoningRetrievalConfig(
            bm25_weight=0.7,
            term_coverage_weight=0.3,
            dense_weight=0.0,
            minimum_local_score=0.0,
            require_local_and_ontology=False,
            include_section_path_in_lexical_index=True,
        ),
    )

    result = PreReasoningRetrievalService().retrieve(request)

    assert result.local_hits[0].chunk_id == "target"
    assert result.local_hits[0].snippet == target_text
    assert result.local_hits[0].section_path == ["ADCS", "Reaction wheel jamming"]


def test_no_match_recommends_offline_fallback_without_modifying_graph() -> None:
    request = _request(query_text="unrelated cryogenic valve")
    result = PreReasoningRetrievalService().retrieve(request)

    assert result.status == RetrievalStatus.NO_MATCH
    assert result.reason_codes == [
        "LOCAL_RETRIEVAL_BELOW_THRESHOLD",
        "ONTOLOGY_CONCEPT_COVERAGE_BELOW_THRESHOLD",
        "NO_RETRIEVAL_EVIDENCE",
    ]
    assert result.paperseek_fallback_recommended is True
    assert result.active_graph_modified is False

    fallback = PaperSeekGovernedBridge.decide_fallback(request.query, result)
    assert fallback.status == PaperSeekFallbackStatus.OFFLINE_DISCOVERY_REQUESTED
    assert fallback.discovery_request.execution_mode == "OFFLINE_ASYNC"
    assert fallback.discovery_request.online_diagnosis_waits_for_result is False
    assert fallback.active_corpus_modified is False
    assert fallback.active_graph_modified is False


def test_sufficient_dual_retrieval_does_not_trigger_paperseek() -> None:
    request = _request()
    result = PreReasoningRetrievalService().retrieve(request)

    fallback = PaperSeekGovernedBridge.decide_fallback(request.query, result)

    assert fallback.status == PaperSeekFallbackStatus.NOT_TRIGGERED
    assert fallback.discovery_request is None


def test_paperseek_candidate_only_enters_review_queue_and_never_updates_active_assets() -> None:
    candidate = PaperSeekLiteratureCandidate(
        candidate_id="paper:10.1000/example",
        title="Knowledge-enhanced spacecraft fault diagnosis",
        abstract="A reviewed candidate abstract.",
        doi="10.1000/example",
        provider="Crossref",
        provider_record_sha256="b" * 64,
        relevance_score=0.90,
        source_quality_score=0.80,
        license_status="ALLOWED",
        domain_applicability_status="IN_SCOPE",
    )

    admission = PaperSeekGovernedBridge.assess_candidate(candidate)

    assert admission.disposition == PaperSeekCandidateDisposition.CORPUS_REVIEW_READY
    assert admission.corpus_candidate_created is True
    assert admission.active_corpus_modified is False
    assert admission.graph_candidate_created is False
    assert admission.active_graph_modified is False
    assert admission.requires_human_review is True


def test_paperseek_duplicate_or_unapproved_license_is_rejected() -> None:
    candidate = PaperSeekLiteratureCandidate(
        candidate_id="paper:duplicate",
        title="Duplicate candidate",
        stable_uri="https://example.test/paper",
        provider="OpenAlex",
        provider_record_sha256="c" * 64,
        relevance_score=0.95,
        source_quality_score=0.95,
        license_status="REVIEW_REQUIRED",
        domain_applicability_status="IN_SCOPE",
        duplicate_of_candidate_id="paper:existing",
    )

    admission = PaperSeekGovernedBridge.assess_candidate(candidate)

    assert admission.disposition == PaperSeekCandidateDisposition.REJECT
    assert admission.reason_codes == ["DUPLICATE_CANDIDATE", "LICENSE_NOT_APPROVED"]


def test_paperseek_batch_admission_deduplicates_normalized_doi_and_title() -> None:
    first = PaperSeekLiteratureCandidate(
        candidate_id="paper:first",
        title="Satellite Fault Diagnosis: A Review",
        doi="10.1000/ASTRO",
        provider="Crossref",
        provider_record_sha256="d" * 64,
        relevance_score=0.90,
        source_quality_score=0.80,
        license_status="ALLOWED",
        domain_applicability_status="IN_SCOPE",
    )
    duplicate = first.model_copy(
        update={
            "candidate_id": "paper:duplicate-normalized",
            "title": "Satellite fault diagnosis — a review",
            "doi": " 10.1000/astro ",
            "provider_record_sha256": "e" * 64,
        }
    )

    admissions = PaperSeekGovernedBridge.assess_candidates([first, duplicate])

    assert admissions[0].disposition == PaperSeekCandidateDisposition.CORPUS_REVIEW_READY
    assert admissions[1].disposition == PaperSeekCandidateDisposition.REJECT
    assert "DUPLICATE_CANDIDATE" in admissions[1].reason_codes


def test_small_frozen_contract_benchmark_reports_retrieval_and_rejection_metrics() -> None:
    positive = _request()
    negative = _request(query_text="unrelated cryogenic valve")

    report = evaluate_retrieval_cases([
        RetrievalBenchmarkCase(
            case_id="rw-jam-two-hop",
            request=positive,
            relevant_chunk_ids=["chunk-duplicate"],
            expected_status=RetrievalStatus.SUFFICIENT,
            expected_concept_ids=[
                "concept:rw-jam", "concept:torque-loss", "concept:wheel-speed"
            ],
        ),
        RetrievalBenchmarkCase(
            case_id="negative-unrelated",
            request=negative,
            relevant_chunk_ids=[],
            expected_status=RetrievalStatus.NO_MATCH,
        ),
    ])

    assert report.case_count == 2
    assert report.mean_recall_at_k == 1.0
    assert report.mean_precision_at_k == 1.0
    assert report.mean_ndcg_at_k == 1.0
    assert report.mean_reciprocal_rank == 1.0
    assert report.mean_concept_recall == 1.0
    assert report.mean_experience_recall == 1.0
    assert report.knowledge_candidate_leakage_rate == 0.0
    assert report.status_accuracy == 1.0
    assert report.negative_rejection_rate == 1.0
    assert report.citation_hash_integrity == 1.0
    assert len(report.benchmark_sha256) == 64
    assert report.retrieval_system_id.endswith(";dense=none")
    assert report.case_results[0]["top_k"] == positive.query.top_k
    assert report.report_sha256 == report.compute_sha256()


def test_benchmark_binds_dense_provider_and_frozen_cases() -> None:
    class _DenseProvider:
        provider_id = "dense-test"
        provider_version = "1"

        def score(self, *, query, chunks):
            return {
                item.chunk_id: (1.0 if item.chunk_id == "chunk-restricted" else 0.0)
                for item in chunks
            }

    request = _request(access=["PUBLIC", "MAINTENANCE"]).model_copy(update={
        "config": PreReasoningRetrievalConfig(
            bm25_weight=0.0, term_coverage_weight=0.0, dense_weight=1.0,
            ontology_concept_bonus=0.0,
        )
    })
    case = RetrievalBenchmarkCase(
        case_id="dense", request=request, relevant_chunk_ids=["chunk-restricted"],
        expected_status=RetrievalStatus.SUFFICIENT,
    )
    report = evaluate_retrieval_cases(
        [case], service=PreReasoningRetrievalService(dense_score_provider=_DenseProvider())
    )
    assert report.mean_reciprocal_rank == 1.0
    assert report.retrieval_system_id.endswith(";dense=dense-test@1")
    changed = case.model_copy(update={"case_id": "dense-renamed"})
    assert evaluate_retrieval_cases(
        [changed], service=PreReasoningRetrievalService(dense_score_provider=_DenseProvider())
    ).benchmark_sha256 != report.benchmark_sha256


def test_benchmark_measures_experience_recall_and_candidate_leakage() -> None:
    request = _request(query_text="approved unique lesson")
    experience = _chunk("experience-hit", "approved unique lesson").model_copy(update={
        "source_id": "experience:EXP-HIT", "lane": RetrievalLane.EXPERIENCE,
    })
    request = request.model_copy(update={
        "query": request.query.model_copy(update={"top_k": 1}),
        "chunks": [experience, *request.chunks],
        "knowledge_update_candidates": [
            {"experience_id": "EXP-HIT", "active_graph_modified": False},
            {"experience_id": "EXP-NOT-HIT", "active_graph_modified": False},
        ],
    })
    report = evaluate_retrieval_cases([RetrievalBenchmarkCase(
        case_id="experience-hit", request=request,
        relevant_chunk_ids=["experience-hit"], expected_status=RetrievalStatus.SUFFICIENT,
        expected_concept_ids=["concept:wheel-speed"],
        expected_experience_source_ids=["experience:EXP-HIT"],
        forbidden_candidate_experience_ids=["EXP-NOT-HIT"],
    )])
    assert report.mean_experience_recall == 1.0
    assert report.knowledge_candidate_leakage_rate == 0.0
    assert report.case_results[0]["knowledge_candidate_experience_ids"] == ["EXP-HIT"]


def test_chunk_text_hash_mismatch_is_rejected() -> None:
    with pytest.raises(ValidationError, match="chunk_sha256"):
        GovernedTextChunk(
            chunk_id="chunk-bad",
            source_id="source:bad",
            source_uri="local://bad",
            source_sha256=SOURCE_HASH,
            chunk_sha256=hashlib.sha256(b"original").hexdigest(),
            knowledge_version="v1",
            text="modified",
        )


class _CaptureBackend:
    model_id = "capture"
    model_version = "1"

    def __init__(self) -> None:
        self.payload = None

    def generate_structured(self, *, agent_id, output_model, payload):
        self.payload = payload
        return HypothesisSet.model_validate(payload["reference_hypothesis_set"])


def test_retrieval_is_injected_before_structured_generation_and_cannot_change_reference() -> None:
    bundle = EpisodeEvidenceBundle.model_validate_json(
        Path("tests/fixtures/evidence/heater_stuck_on/diagnosis/episode_evidence_bundle.json")
        .read_text(encoding="utf-8")
    )
    request = _request().model_copy(
        update={
            "query": _request().query.model_copy(update={"episode_id": bundle.episode.episode_id})
        }
    )
    retrieval = PreReasoningRetrievalService().retrieve(request)
    backend = _CaptureBackend()

    result = StructuredHypothesisProposalService().reason(
        bundle,
        backend=backend,
        retrieval_context=retrieval,
    )

    assert backend.payload["pre_reasoning_retrieval"]["package_id"] == retrieval.package_id
    assert backend.payload["candidate_fusion"] is None
    assert result.validation_metadata["pre_reasoning_retrieval_package_id"] == retrieval.package_id
    assert result.validation_metadata["repair_succeeded"] is True


def test_evidence_aware_builder_constructs_hash_pinned_context_from_diagnosis() -> None:
    bundle = EpisodeEvidenceBundle.model_validate_json(
        Path("tests/fixtures/evidence/heater_stuck_on/diagnosis/episode_evidence_bundle.json")
        .read_text(encoding="utf-8")
    )
    builder = EvidenceAwarePreGenerationContextBuilder(
        runtime_evidence_path="configs/datasets/phm_v6_0/runtime_evidence.json",
        fault_catalog_path="configs/fault_modes/lab_fault_catalog.json",
    )

    request = builder.build_request(bundle)

    assert request is not None
    assert request.query.construction_mode == "EVIDENCE_AWARE"
    assert request.query.grounding_fault_ids == ["THERMAL_HEATER_STUCK_ON"]
    assert request.query.grounding_evidence_fact_ids
    result = PreReasoningRetrievalService().retrieve(request)
    assert result.status == RetrievalStatus.SUFFICIENT
    assert result.query_construction_mode == "EVIDENCE_AWARE"
    assert result.grounding_fault_ids == ["THERMAL_HEATER_STUCK_ON"]
    assert result.grounding_evidence_fact_ids == request.query.grounding_evidence_fact_ids
    assert {item.source_sha256 for item in result.local_hits} <= {
        builder.runtime_sha256,
        builder.catalog_sha256,
    }
    assert any(
        hop.relation_type == "INDICATED_BY"
        for path in result.ontology_paths
        for hop in path.hops
    )
    assert result.active_graph_modified is False


def test_evidence_aware_builder_can_add_reviewed_tbox_paths_without_voting() -> None:
    bundle = EpisodeEvidenceBundle.model_validate_json(
        Path("tests/fixtures/evidence/heater_stuck_on/diagnosis/episode_evidence_bundle.json")
        .read_text(encoding="utf-8")
    )
    graph_path = require_delivery_evidence(
        Path.cwd(),
        "governance/evidence/research/document_corpus/reviewed_tbox_retrieval_r001/reviewed_tbox_retrieval_graph.json",
    )
    require_delivery_evidence(
        Path.cwd(),
        "governance/evidence/research/document_corpus/reviewed_tbox_retrieval_r001/build_receipt.json",
    )
    graph_root = graph_path.parent
    builder = EvidenceAwarePreGenerationContextBuilder(
        runtime_evidence_path="configs/datasets/phm_v6_0/runtime_evidence.json",
        fault_catalog_path="configs/fault_modes/lab_fault_catalog.json",
        reviewed_tbox_graph_path=graph_root / "reviewed_tbox_retrieval_graph.json",
        reviewed_tbox_projection_receipt_path=graph_root / "build_receipt.json",
    )
    request = builder.build_request(
        bundle,
        reviewed_tbox_seed_concept_ids=["mechanism:thermal-exchange-degradation"],
    )
    assert request is not None
    assert request.ontology_asset_audits[0]["runtime_vote_allowed"] is False
    result = PreReasoningRetrievalService().retrieve(request)
    assert result.ontology_seed_concept_ids == [
        "fault:THERMAL_HEATER_STUCK_ON",
        "mechanism:thermal-exchange-degradation",
    ]
    assert any(
        path.seed_concept_id == "mechanism:thermal-exchange-degradation"
        and path.hops[0].relation_type == "MANIFESTS_AS"
        for path in result.ontology_paths
    )
    assert result.ontology_asset_audits[0]["diagnostic_evidence_allowed"] is False
    assert result.active_graph_modified is False


@pytest.mark.parametrize(
    "prompt_profile",
    ["legacy_v051", "evidence_protocol_v052", "evidence_protocol_v059_compact"],
)
def test_every_llm_prompt_profile_exposes_the_same_retrieval_trust_boundary(
    prompt_profile: str,
) -> None:
    backend = OpenAICompatibleStructuredBackend(
        OpenAICompatibleConfig(model="test-model", prompt_profile=prompt_profile)
    )

    messages = backend._messages(output_model=HypothesisSet, payload={})

    assert "pre_reasoning_retrieval" in messages[0]["content"]
    assert "不是正式episode证据" in messages[0]["content"]
    assert "不得提供新故障ID或覆盖BiRRF" in messages[0]["content"]


def test_paperseek_annotation_records_method_role_and_non_claim_boundary() -> None:
    candidate = PaperSeekLiteratureCandidate(
        candidate_id="arxiv:2210.03629",
        title="ReAct: Synergizing Reasoning and Acting in Language Models",
        abstract="Interleaves reasoning and actions.",
        stable_uri="https://arxiv.org/abs/2210.03629",
        provider="arXiv",
        provider_record_sha256="b" * 64,
        relevance_score=0.9,
        source_quality_score=0.9,
        license_status="ALLOWED",
        domain_applicability_status="IN_SCOPE",
    )
    annotation = PaperSeekLiteratureAnnotation.select(
        candidate,
        discovery_run_id="PAPERSEEK-20260902-LLM-DIAG",
        decision=LiteratureAnnotationDecision.INCLUDE,
        method_roles=[LiteratureMethodRole.TOOL_PLANNING],
        supported_design_claims=["Reasoning can interleave with tool actions and observations."],
        unsupported_project_claims=["Does not establish satellite fault-diagnosis benefit."],
        astrograph_compatibility="Supports a typed read-only diagnostic action loop.",
        selection_rationale="Direct architectural method anchor.",
    )
    assert annotation.runtime_use_allowed is False
    assert annotation.active_corpus_modified is False
    assert annotation.active_graph_modified is False
    assert annotation.requires_human_confirmation is True


def test_included_paperseek_annotation_cannot_omit_unsupported_claims() -> None:
    candidate = PaperSeekLiteratureCandidate(
        candidate_id="arxiv:test",
        title="Test paper",
        stable_uri="https://arxiv.org/abs/test",
        provider="arXiv",
        provider_record_sha256="c" * 64,
        relevance_score=0.9,
        source_quality_score=0.9,
        license_status="ALLOWED",
        domain_applicability_status="IN_SCOPE",
    )
    with pytest.raises(ValueError, match="unsupported AstroGraph claims"):
        PaperSeekLiteratureAnnotation.select(
            candidate,
            discovery_run_id="RUN-1",
            decision=LiteratureAnnotationDecision.INCLUDE,
            method_roles=[LiteratureMethodRole.METHOD_ANCHOR],
            supported_design_claims=["A transferable design claim."],
            unsupported_project_claims=[],
            astrograph_compatibility="Compatible.",
            selection_rationale="Relevant.",
        )


def test_checked_in_paperseek_annotations_are_schema_valid_and_non_runtime() -> None:
    path = require_delivery_evidence(
        Path.cwd(),
        "governance/evidence/ai_native/paperseek_llm_diagnostic_loop_20260903/literature_annotations.json",
    )
    annotations = [
        PaperSeekLiteratureAnnotation.model_validate(item)
        for item in json.loads(path.read_text(encoding="utf-8"))
    ]
    assert len(annotations) == 8
    assert all(item.decision == LiteratureAnnotationDecision.INCLUDE for item in annotations)
    assert all(not item.runtime_use_allowed for item in annotations)
    assert all(item.unsupported_project_claims for item in annotations)
