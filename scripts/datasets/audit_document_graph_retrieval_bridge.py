#!/usr/bin/env python3
"""Audit one real document through RAG and candidate-KG traceability paths."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from astrograph.knowledge import process_governed_document
from astrograph.knowledge.construction import DocumentKnowledgeStagingBundle
from astrograph.knowledge.schema_v3 import KnowledgeGraphV3
from astrograph.llm.openai_compatible import OpenAICompatibleStructuredBackend
from astrograph.retrieval.document_governance import (
    DocumentAccessContext,
    DocumentAccessMode,
    DocumentAuthority,
    DocumentReviewState,
    DocumentType,
    DocumentVisibilityClass,
    GovernedDocumentDescriptor,
)
from astrograph.retrieval.document_graph_bridge import build_document_graph_trace_links
from astrograph.retrieval.document_claims import (
    DocumentClaimAdmissionPolicy,
    materialize_document_graph_claims,
)
from astrograph.retrieval.models import DiagnosticRetrievalQuery, PreReasoningRetrievalRequest
from astrograph.retrieval.service import PreReasoningRetrievalService
from astrograph.retrieval.v3_tbox_adapter import load_reviewed_v3_tbox_retrieval_assets


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
SOURCE = WORKSPACE / "智能故障检测软件_设计报告.docx"
OUTPUT = ROOT / "governance/evidence/research/document_corpus/document_graph_retrieval_bridge_r001"
CANDIDATE_ROOT = (
    ROOT / ".astrograph-state/research/document_corpus/document_graph_scope_pilot_r015/episode"
    "/adcs-v522/doc_anomaly_b3bd6e3e0a62736346fe6be2"
)


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def main() -> int:
    source_sha = _file_sha(SOURCE)
    processed = process_governed_document(
        descriptor=GovernedDocumentDescriptor(
            document_id="design:astrograph:early-software-report",
            document_type=DocumentType.DESIGN_PROPOSAL,
            visibility_class=DocumentVisibilityClass.DESIGN_PUBLIC,
            source_uri=str(SOURCE),
            source_sha256=source_sha,
            review_state=DocumentReviewState.REVIEW_REQUIRED,
            authority=DocumentAuthority.NON_NORMATIVE,
            runtime_vote_allowed=False,
        ),
        source_path=SOURCE,
        independence_group="early-design-report-origin",
        license_id="project-owned",
    )
    links = build_document_graph_trace_links(processed)
    projection = ROOT / "governance/evidence/research/document_corpus/reviewed_tbox_retrieval_r001"
    concepts, relations, tbox_audit = load_reviewed_v3_tbox_retrieval_assets(
        graph_path=projection / "reviewed_tbox_retrieval_graph.json",
        projection_receipt_path=projection / "build_receipt.json",
    )
    request = PreReasoningRetrievalRequest(
        query=DiagnosticRetrievalQuery(
            query_id="query:early-design-dual-path-audit",
            episode_id="episode:offline-document-audit",
            text="卫星 故障 诊断 知识图谱 智能 推理",
            seed_concept_ids=["mechanism:actuation-authority-loss"],
            maximum_ontology_hops=1,
            top_k=8,
            document_access_context=DocumentAccessContext(mode=DocumentAccessMode.CORPUS_BUILD),
        ),
        chunks=processed.retrieval_ingestion.chunks,
        concepts=concepts,
        relations=relations,
        ontology_asset_audits=[tbox_audit],
        document_graph_trace_links=links,
    )
    package = PreReasoningRetrievalService().retrieve(request)
    package_payload = package.model_dump(mode="json")
    if not package.local_hits or not package.document_graph_trace_links:
        raise ValueError("real document did not produce retrievable cross-path trace links")
    if not package.ontology_paths or not all(
        hop.source_fragment_ids
        for path in package.ontology_paths
        for hop in path.hops
    ):
        raise ValueError("reviewed graph paths did not retain source fragments")
    if any(item.link_status != "STAGING_TRACE_ONLY" for item in links):
        raise ValueError("unreviewed document unexpectedly claimed reviewed graph membership")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    trace_path = OUTPUT / "retrieval_package_trace.json"
    trace_path.write_text(
        json.dumps(package_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Rebuild one real r015 document and bind its persisted unreviewed graph.
    candidate_staging_path = CANDIDATE_ROOT / "knowledge_staging.json"
    candidate_graph_path = CANDIDATE_ROOT / "candidate_graph.json"
    candidate_staging = DocumentKnowledgeStagingBundle.model_validate_json(
        candidate_staging_path.read_text(encoding="utf-8")
    )
    candidate_graph = KnowledgeGraphV3.model_validate_json(
        candidate_graph_path.read_text(encoding="utf-8")
    )
    candidate_processed = process_governed_document(
        descriptor=candidate_staging.document_descriptor,
        source_path=candidate_staging.source_profile.source_path,
        independence_group=candidate_staging.source_profile.independence_group,
        license_id=candidate_staging.source_profile.license_id,
    )
    if candidate_processed.knowledge_staging.staging_sha256 != candidate_staging.staging_sha256:
        raise ValueError("r015 document staging did not rebuild identically")
    candidate_links = build_document_graph_trace_links(
        candidate_processed, candidate_graph=candidate_graph
    )
    candidate_descriptor = candidate_staging.document_descriptor
    candidate_query = DiagnosticRetrievalQuery(
        query_id="query:r015-candidate-subgraph-audit",
        episode_id=candidate_descriptor.episode_id or "episode:missing",
        text="telemetry anomaly screening baseline comparison channel shift",
        top_k=8,
        document_access_context=DocumentAccessContext(
            mode=DocumentAccessMode.RUNTIME_DIAGNOSIS,
            episode_id=candidate_descriptor.episode_id,
            current_provenance_group_id=candidate_descriptor.provenance_group_id,
            observation_time_s=candidate_descriptor.observable_cutoff_time_s,
        ),
    )
    preliminary_candidate_package = PreReasoningRetrievalService().retrieve(
        PreReasoningRetrievalRequest(
            query=candidate_query,
            chunks=candidate_processed.retrieval_ingestion.chunks,
            document_graph_trace_links=candidate_links,
        )
    )
    document_claims = materialize_document_graph_claims(
        candidate_processed,
        graph=candidate_graph,
        trace_links=preliminary_candidate_package.document_graph_trace_links,
    )
    admission_policy = DocumentClaimAdmissionPolicy()
    candidate_package = PreReasoningRetrievalService().retrieve(
        PreReasoningRetrievalRequest(
            query=candidate_query,
            chunks=candidate_processed.retrieval_ingestion.chunks,
            document_graph_trace_links=candidate_links,
            document_evidence_claims=document_claims,
            document_claim_admission_policy=admission_policy,
        )
    )
    candidate_package_payload = candidate_package.model_dump(mode="json")
    candidate_hit_links = [
        item for item in candidate_package.document_graph_trace_links
        if item.link_status == "CANDIDATE_GRAPH_LINK"
    ]
    if not candidate_hit_links or any(
        item.runtime_vote_allowed or item.active_graph_modified for item in candidate_hit_links
    ):
        raise ValueError("r015 retrieval did not expose a bounded candidate-subgraph link")
    claim_admissions = candidate_package.document_claim_admissions
    if not document_claims or any(item.formal_fusion_eligible for item in claim_admissions):
        raise ValueError("unreviewed and uncalibrated r015 claims must remain context-only")
    claim_payload = {
        "claims": [item.model_dump(mode="json") for item in document_claims],
        "admissions": [item.model_dump(mode="json") for item in claim_admissions],
        "policy": admission_policy.model_dump(mode="json"),
    }
    claim_path = OUTPUT / "document_claim_admission_trace.json"
    claim_path.write_text(
        json.dumps(claim_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    candidate_trace_path = OUTPUT / "candidate_graph_retrieval_trace.json"
    candidate_trace_path.write_text(
        json.dumps(candidate_package_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    early_generation_context = (
        OpenAICompatibleStructuredBackend._compact_pre_reasoning_retrieval(
            package_payload, set()
        )
    )
    candidate_generation_context = (
        OpenAICompatibleStructuredBackend._compact_pre_reasoning_retrieval(
            candidate_package_payload, set()
        )
    )
    if (
        len(early_generation_context["local_hits"]) != len(package.local_hits)
        or len(candidate_generation_context["local_hits"]) != len(candidate_package.local_hits)
        or not candidate_generation_context["document_graph_trace_links"]
    ):
        raise ValueError("governed linked document hits were lost before structured generation")
    generation_context = {
        "early_design": early_generation_context,
        "candidate_graph_document": candidate_generation_context,
    }
    generation_context_path = OUTPUT / "pre_generation_context_trace.json"
    generation_context_path.write_text(
        json.dumps(generation_context, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    receipt = {
        "schema_version": "astrograph-document-graph-retrieval-bridge-audit.v1",
        "status": "PASS",
        "source_document_sha256": source_sha,
        "processing_sha256": processed.processing_sha256,
        "staging_sha256": processed.knowledge_staging.staging_sha256,
        "parsed_element_count": len(processed.retrieval_ingestion.elements),
        "retrieval_chunk_count": len(processed.retrieval_ingestion.chunks),
        "crosswalk_count": len(processed.crosswalk),
        "crosswalk_complete": len(processed.crosswalk) == len(processed.retrieval_ingestion.elements),
        "candidate_kg_staged": processed.candidate_kg_staged,
        "candidate_graph_built": processed.candidate_graph_built,
        "publication_allowed": processed.publication_allowed,
        "runtime_vote_allowed": processed.runtime_vote_allowed,
        "retrieval_hit_count": len(package.local_hits),
        "hit_to_fragment_link_count": len(package.document_graph_trace_links),
        "ontology_path_count": len(package.ontology_paths),
        "graph_hops_with_source_fragments": sum(
            bool(hop.source_fragment_ids)
            for path in package.ontology_paths
            for hop in path.hops
        ),
        "active_graph_modified": package.active_graph_modified,
        "candidate_graph_id": candidate_graph.graph_id,
        "candidate_graph_file_sha256": _file_sha(candidate_graph_path),
        "candidate_staging_file_sha256": _file_sha(candidate_staging_path),
        "candidate_graph_node_count": len(candidate_graph.nodes),
        "candidate_graph_edge_count": len(candidate_graph.edges),
        "candidate_retrieval_hit_count": len(candidate_package.local_hits),
        "candidate_hit_to_subgraph_link_count": len(candidate_hit_links),
        "candidate_linked_graph_item_count": len({
            graph_item_id
            for item in candidate_hit_links
            for graph_item_id in item.graph_item_ids
        }),
        "candidate_trace_file_sha256": _file_sha(candidate_trace_path),
        "candidate_trace_semantic_sha256": _canonical_sha(candidate_package_payload),
        "early_hits_retained_for_generation": len(
            early_generation_context["local_hits"]
        ),
        "early_generation_access_mode": "CORPUS_BUILD",
        "early_runtime_admission_allowed": False,
        "candidate_hits_retained_for_generation": len(
            candidate_generation_context["local_hits"]
        ),
        "candidate_graph_links_retained_for_generation": sum(
            item.get("link_status") == "CANDIDATE_GRAPH_LINK"
            for item in candidate_generation_context["document_graph_trace_links"]
        ),
        "materialized_document_claim_count": len(document_claims),
        "formal_fusion_eligible_claim_count": sum(
            item.formal_fusion_eligible for item in claim_admissions
        ),
        "claim_admission_reason_counts": {
            reason: sum(reason in item.reason_codes for item in claim_admissions)
            for reason in sorted({
                reason for item in claim_admissions for reason in item.reason_codes
            })
        },
        "document_claim_trace_file_sha256": _file_sha(claim_path),
        "document_claim_trace_semantic_sha256": _canonical_sha(claim_payload),
        "pre_generation_context_file_sha256": _file_sha(generation_context_path),
        "pre_generation_context_semantic_sha256": _canonical_sha(generation_context),
        "trace_file_sha256": _file_sha(trace_path),
        "trace_semantic_sha256": _canonical_sha(package_payload),
        "builder_source_sha256": _file_sha(Path(__file__).resolve()),
        "implementation_source_sha256": {
            relative: _file_sha(ROOT / relative)
            for relative in (
                "src/astrograph/retrieval/document_graph_bridge.py",
                "src/astrograph/retrieval/document_governance.py",
                "src/astrograph/retrieval/models.py",
                "src/astrograph/retrieval/service.py",
                "src/astrograph/retrieval/v3_tbox_adapter.py",
                "src/astrograph/llm/openai_compatible.py",
                "src/astrograph/retrieval/document_claims.py",
                "src/astrograph/diagnosis/structured_output.py",
            )
        },
        "claim_boundary": (
            "The document is processed into both governed retrieval chunks and candidate-KG "
            "fragments. Cross-links are provenance only; the unreviewed candidate graph is not "
            "published and cannot supply diagnosis evidence or a runtime vote."
        ),
    }
    receipt_path = OUTPUT / "build_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
