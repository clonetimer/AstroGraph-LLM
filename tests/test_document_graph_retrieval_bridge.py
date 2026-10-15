from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from astrograph.knowledge import process_governed_document
from astrograph.diagnosis.structured_output import _retrieval_trace
from astrograph.knowledge.schema_v3 import (
    GraphEdgeV3,
    GraphNodeV3,
    KnowledgeGraphV3,
    KnowledgeNodeTypeV3,
    KnowledgeRelationTypeV3,
    LifecycleStatus,
)
from astrograph.llm.openai_compatible import OpenAICompatibleStructuredBackend
from astrograph.retrieval.document_graph_bridge import build_document_graph_trace_links
from astrograph.retrieval.document_claims import (
    DocumentClaimAdmissionPolicy,
    evaluate_document_claim_admissions,
    materialize_document_graph_claims,
)
from astrograph.retrieval.document_governance import (
    ClaimEpistemicStatus,
    DocumentAccessContext,
    DocumentAccessMode,
    DocumentAuthority,
    DocumentReviewState,
    DocumentType,
    DocumentVisibilityClass,
    GovernedDocumentDescriptor,
    bind_document_evidence_claim,
)
from astrograph.retrieval.models import (
    DiagnosticRetrievalQuery,
    DocumentGraphTraceLink,
    PreReasoningRetrievalPackage,
    PreReasoningRetrievalRequest,
)
from astrograph.retrieval.service import PreReasoningRetrievalService


def test_document_is_retrieved_and_traced_to_candidate_kg_fragment(tmp_path) -> None:
    path = tmp_path / "design.md"
    path.write_text(
        "# Reaction wheel\n\n## Failure mechanism\n\nBearing drag causes current rise.\n",
        encoding="utf-8",
    )
    source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    processed = process_governed_document(
        descriptor=GovernedDocumentDescriptor(
            document_id="design:wheel:fixture",
            document_type=DocumentType.DESIGN_PROPOSAL,
            visibility_class=DocumentVisibilityClass.DESIGN_PUBLIC,
            source_uri=str(path),
            source_sha256=source_sha,
            review_state=DocumentReviewState.HUMAN_REVIEWED,
            authority=DocumentAuthority.NON_NORMATIVE,
            runtime_vote_allowed=False,
        ),
        source_path=path,
        independence_group="fixture-design-origin",
        license_id="test-fixture",
    )
    links = build_document_graph_trace_links(processed)
    assert links
    assert all(item.link_status == "STAGING_TRACE_ONLY" for item in links)
    assert all(not item.graph_item_ids for item in links)
    assert all(item.source_locators for item in links)

    package = PreReasoningRetrievalService().retrieve(
        PreReasoningRetrievalRequest(
            query=DiagnosticRetrievalQuery(
                query_id="query:wheel-doc",
                episode_id="episode:fixture",
                text="bearing drag current rise",
                document_access_context=DocumentAccessContext(),
                top_k=4,
            ),
            chunks=processed.retrieval_ingestion.chunks,
            document_graph_trace_links=links,
        )
    )
    hit_ids = {item.chunk_id for item in package.local_hits}
    assert package.document_graph_trace_links
    assert all(item.retrieval_chunk_id in hit_ids for item in package.document_graph_trace_links)
    assert package.active_graph_modified is False
    assert all(item.runtime_vote_allowed is False for item in package.document_graph_trace_links)
    compact = OpenAICompatibleStructuredBackend._compact_pre_reasoning_retrieval(
        package.model_dump(mode="json"), set()
    )
    compact_hit_ids = {item["chunk_id"] for item in compact["local_hits"]}
    assert compact_hit_ids == hit_ids
    assert compact["document_graph_trace_links"]
    assert all(
        item["retrieval_chunk_id"] in compact_hit_ids
        for item in compact["document_graph_trace_links"]
    )
    without_links = package.model_dump(mode="json")
    without_links["document_graph_trace_links"] = []
    unlinked_compact = OpenAICompatibleStructuredBackend._compact_pre_reasoning_retrieval(
        without_links, set()
    )
    assert unlinked_compact["local_hits"] == []


def test_trace_link_cannot_be_rebound_to_another_source(tmp_path) -> None:
    text = "wheel current rise"
    from astrograph.retrieval.models import GovernedTextChunk

    chunk = GovernedTextChunk(
        chunk_id="chunk:one",
        source_id="source:one",
        source_uri="local://one",
        source_sha256="a" * 64,
        chunk_sha256=hashlib.sha256(text.encode()).hexdigest(),
        knowledge_version="v1",
        text=text,
    )
    link = DocumentGraphTraceLink(
        retrieval_chunk_id=chunk.chunk_id,
        source_document_id="source:other",
        source_document_sha256=chunk.source_sha256,
        knowledge_section_id="section:test:one",
        knowledge_fragment_id="fragment:test:one",
        graph_scope="TBOX_CANDIDATE",
        link_status="STAGING_TRACE_ONLY",
        source_locators=["line:1"],
    )
    with pytest.raises(ValidationError, match="source does not match"):
        PreReasoningRetrievalRequest(
            query=DiagnosticRetrievalQuery(
                query_id="query:test", episode_id="episode:test", text="wheel"
            ),
            chunks=[chunk],
            document_graph_trace_links=[link],
        )


def test_rag_fragment_can_open_candidate_subgraph_without_promoting_it(tmp_path) -> None:
    path = tmp_path / "candidate.md"
    path.write_text("# Mechanism\n\nBearing drag causes current rise.\n", encoding="utf-8")
    source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    processed = process_governed_document(
        descriptor=GovernedDocumentDescriptor(
            document_id="design:candidate:fixture",
            document_type=DocumentType.DESIGN_PROPOSAL,
            visibility_class=DocumentVisibilityClass.DESIGN_PUBLIC,
            source_uri=str(path),
            source_sha256=source_sha,
            review_state=DocumentReviewState.REVIEW_REQUIRED,
            authority=DocumentAuthority.NON_NORMATIVE,
        ),
        source_path=path,
        independence_group="candidate-origin",
        license_id="test-fixture",
    )
    fragment_id = processed.crosswalk[0].knowledge_fragment_id
    candidate = KnowledgeGraphV3(
        graph_id="graph:candidate:fixture",
        nodes=[
            GraphNodeV3(
                node_id="mechanism:bearing-drag-candidate",
                node_type=KnowledgeNodeTypeV3.FAILURE_MECHANISM,
                source_id="source:candidate:fixture",
                source_sha256=source_sha,
                lifecycle_status=LifecycleStatus.DRAFT,
                properties={
                    "source_fragment_ids": [fragment_id],
                    "source_document_id": processed.knowledge_staging.document_descriptor.document_id,
                    "source_document_sha256": source_sha,
                },
            ),
            GraphNodeV3(
                node_id="effect:current-rise-candidate",
                node_type=KnowledgeNodeTypeV3.EFFECT_PATTERN,
                source_id="source:candidate:fixture",
                source_sha256=source_sha,
                lifecycle_status=LifecycleStatus.DRAFT,
                properties={
                    "source_fragment_ids": [fragment_id],
                    "source_document_id": processed.knowledge_staging.document_descriptor.document_id,
                    "source_document_sha256": source_sha,
                },
            ),
        ],
        edges=[
            GraphEdgeV3(
                edge_id="edge:bearing-drag-manifests-current-rise",
                source_id="mechanism:bearing-drag-candidate",
                relation_type=KnowledgeRelationTypeV3.MANIFESTS_AS,
                target_id="effect:current-rise-candidate",
                provenance_id="source:candidate:fixture",
                provenance_sha256=source_sha,
                lifecycle_status=LifecycleStatus.DRAFT,
                properties={
                    "source_fragment_ids": [fragment_id],
                    "source_document_id": processed.knowledge_staging.document_descriptor.document_id,
                    "source_document_sha256": source_sha,
                    "epistemic_status": "DESIGN_PROPOSAL",
                },
            )
        ],
    )
    links = build_document_graph_trace_links(processed, candidate_graph=candidate)
    linked = [item for item in links if item.link_status == "CANDIDATE_GRAPH_LINK"]
    assert linked
    assert "mechanism:bearing-drag-candidate" in linked[0].graph_item_ids
    assert "edge:bearing-drag-manifests-current-rise" in linked[0].graph_item_ids
    assert linked[0].graph_id == candidate.graph_id
    assert linked[0].runtime_vote_allowed is False
    assert linked[0].active_graph_modified is False

    claims = materialize_document_graph_claims(
        processed, graph=candidate, trace_links=links
    )
    assert len(claims) == 1
    assert claims[0].predicate == "MANIFESTS_AS"
    assert claims[0].source_fragment_id == fragment_id
    assert claims[0].runtime_evidence_eligible is False
    admissions = evaluate_document_claim_admissions(
        claims, DocumentClaimAdmissionPolicy()
    )
    assert admissions[0].formal_fusion_eligible is False
    assert "CLAIM_NOT_CALIBRATED" in admissions[0].reason_codes
    assert "SOURCE_INDEPENDENCE_NOT_ESTABLISHED" in admissions[0].reason_codes
    enriched = PreReasoningRetrievalService().retrieve(
        PreReasoningRetrievalRequest(
            query=DiagnosticRetrievalQuery(
                query_id="query:candidate-claim",
                episode_id="episode:offline",
                text="bearing drag current rise",
                document_access_context=DocumentAccessContext(
                    mode=DocumentAccessMode.CORPUS_BUILD
                ),
            ),
            chunks=processed.retrieval_ingestion.chunks,
            document_graph_trace_links=links,
            document_evidence_claims=claims,
            document_claim_admission_policy=DocumentClaimAdmissionPolicy(),
        )
    )
    assert [item.claim_id for item in enriched.document_evidence_claims] == [
        claims[0].claim_id
    ]
    assert enriched.document_claim_admissions[0].formal_fusion_eligible is False
    output_trace = _retrieval_trace(enriched.model_dump(mode="json"))
    assert output_trace["document_evidence_claims"][0]["claim_id"] == claims[0].claim_id
    assert output_trace["document_claim_admissions"][0]["formal_fusion_eligible"] is False
    tampered = enriched.model_dump(mode="json")
    tampered["document_evidence_claims"][0]["source_document_id"] = "source:forged"
    with pytest.raises(ValidationError, match="provenance differs"):
        PreReasoningRetrievalPackage.model_validate(tampered)

    wrong_source = candidate.model_copy(deep=True)
    wrong_source.nodes[0].properties["source_document_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="not its governed document source"):
        build_document_graph_trace_links(processed, candidate_graph=wrong_source)


def test_document_claim_requires_all_explicit_fusion_gates() -> None:
    descriptor = GovernedDocumentDescriptor(
        document_id="log:independent:fixture",
        document_type=DocumentType.TEST_LOG,
        visibility_class=DocumentVisibilityClass.OBSERVATION_CAUSAL,
        source_uri="local://independent-log",
        source_sha256="b" * 64,
        review_state=DocumentReviewState.HUMAN_REVIEWED,
        authority=DocumentAuthority.APPROVED_SOURCE,
        provenance_group_id="independent:HIL:1",
        episode_id="episode:claim-gate",
        valid_time_start_s=0.0,
        valid_time_end_s=10.0,
        observable_cutoff_time_s=10.0,
        runtime_vote_allowed=True,
    )
    claim = bind_document_evidence_claim(
        claim_id="claim:independent:fixture",
        subject="component:wheel",
        predicate="OBSERVED_IN",
        object="effect:current-rise",
        source_span="Current rise observed during the registered interval.",
        source_locator="page:1",
        source_chunk_id="chunk:independent",
        source_fragment_id="fragment:independent:1",
        descriptor=descriptor,
        epistemic_status=ClaimEpistemicStatus.SOURCE_FACT,
        applicability=["episode:episode:claim-gate"],
        publication_eligible=True,
        runtime_evidence_eligible=True,
    )
    blocked = evaluate_document_claim_admissions(
        [claim], DocumentClaimAdmissionPolicy()
    )[0]
    assert blocked.formal_fusion_eligible is False
    admitted = evaluate_document_claim_admissions(
        [claim],
        DocumentClaimAdmissionPolicy(
            formal_document_fusion_enabled=True,
            policy_review_status="HUMAN_REVIEWED",
            reviewer_id="reviewer:fixture",
            fusion_approval_receipt_sha256="1" * 64,
            calibration_receipt_sha256="2" * 64,
            source_independence_receipt_sha256="3" * 64,
            calibrated_claim_ids=[claim.claim_id],
            independent_provenance_group_ids=["independent:HIL:1"],
            applicability_context=["episode:episode:claim-gate"],
        ),
    )[0]
    assert admitted.formal_fusion_eligible is True
    assert admitted.reason_codes == ["FORMAL_FUSION_ELIGIBLE"]
    with pytest.raises(ValidationError, match="requires named review"):
        DocumentClaimAdmissionPolicy(formal_document_fusion_enabled=True)
