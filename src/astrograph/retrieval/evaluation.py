from __future__ import annotations

import hashlib
import json
import math

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import PreReasoningRetrievalRequest, RetrievalStatus
from .service import PreReasoningRetrievalService


class RetrievalBenchmarkCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    request: PreReasoningRetrievalRequest
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    expected_status: RetrievalStatus
    expected_concept_ids: list[str] = Field(default_factory=list)
    expected_experience_source_ids: list[str] = Field(default_factory=list)
    forbidden_candidate_experience_ids: list[str] = Field(default_factory=list)


class RetrievalBenchmarkReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-pre-reasoning-retrieval-benchmark-report.v1"
    benchmark_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    retrieval_system_id: str = Field(min_length=1)
    case_count: int
    positive_case_count: int
    negative_case_count: int
    mean_recall_at_k: float = Field(ge=0.0, le=1.0)
    mean_precision_at_k: float = Field(ge=0.0, le=1.0)
    mean_ndcg_at_k: float = Field(ge=0.0, le=1.0)
    mean_reciprocal_rank: float = Field(ge=0.0, le=1.0)
    mean_concept_recall: float = Field(ge=0.0, le=1.0)
    mean_experience_recall: float = Field(ge=0.0, le=1.0)
    knowledge_candidate_leakage_rate: float = Field(ge=0.0, le=1.0)
    status_accuracy: float = Field(ge=0.0, le=1.0)
    negative_rejection_rate: float = Field(ge=0.0, le=1.0)
    citation_hash_integrity: float = Field(ge=0.0, le=1.0)
    case_results: list[dict]
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_boundary: str = (
        "Small deterministic contract benchmark only; it does not establish production retrieval quality, "
        "LLM faithfulness, GraphRAG gain, or domain-wide corpus coverage."
    )

    @model_validator(mode="after")
    def validate_hash(self) -> "RetrievalBenchmarkReport":
        if self.report_sha256 != self.compute_sha256():
            raise ValueError("retrieval benchmark report_sha256 mismatch")
        return self

    def compute_sha256(self) -> str:
        payload = self.model_dump(mode="json", exclude={"report_sha256"})
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def evaluate_retrieval_cases(
    cases: list[RetrievalBenchmarkCase],
    *,
    service: PreReasoningRetrievalService | None = None,
) -> RetrievalBenchmarkReport:
    if not cases or len({item.case_id for item in cases}) != len(cases):
        raise ValueError("retrieval benchmark cases must be nonempty and uniquely identified")
    service = service or PreReasoningRetrievalService()
    benchmark_sha256 = hashlib.sha256(json.dumps(
        [item.model_dump(mode="json") for item in sorted(cases, key=lambda item: item.case_id)],
        ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    provider = service.dense_score_provider
    provider_id = "none"
    if provider is not None:
        provider_id = str(
            getattr(provider, "provider_id", provider.__class__.__qualname__)
        )
        provider_version = getattr(provider, "provider_version", None)
        if provider_version:
            provider_id += "@" + str(provider_version)
    retrieval_system_id = f"{service.service_id}@{service.service_version};dense={provider_id}"
    recalls: list[float] = []
    precisions: list[float] = []
    ndcgs: list[float] = []
    reciprocal_ranks: list[float] = []
    concept_recalls: list[float] = []
    experience_recalls: list[float] = []
    status_matches = 0
    negative_matches = 0
    hash_checks = 0
    hash_total = 0
    candidate_leak_cases = 0
    candidate_checked_cases = 0
    rows: list[dict] = []
    positives = [item for item in cases if item.relevant_chunk_ids]
    negatives = [item for item in cases if not item.relevant_chunk_ids]
    for case in cases:
        result = service.retrieve(case.request)
        retrieved = [item.chunk_id for item in result.local_hits]
        relevant = set(case.relevant_chunk_ids)
        matched = relevant & set(retrieved)
        recall = len(matched) / len(relevant) if relevant else None
        precision = len(matched) / len(retrieved) if relevant and retrieved else (0.0 if relevant else None)
        if relevant:
            dcg = sum(
                1.0 / math.log2(rank + 1)
                for rank, chunk_id in enumerate(retrieved, start=1)
                if chunk_id in relevant
            )
            ideal_count = min(len(relevant), len(retrieved))
            ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
            ndcg = dcg / ideal_dcg if ideal_dcg else 0.0
            reciprocal_rank = next(
                (1.0 / rank for rank, chunk_id in enumerate(retrieved, start=1) if chunk_id in relevant),
                0.0,
            )
        else:
            ndcg = None
            reciprocal_rank = None
        expected_concepts = set(case.expected_concept_ids)
        concept_recall = (
            len(expected_concepts & set(result.matched_concept_ids)) / len(expected_concepts)
            if expected_concepts else None
        )
        if recall is not None:
            recalls.append(recall)
        if precision is not None:
            precisions.append(precision)
        if ndcg is not None:
            ndcgs.append(ndcg)
        if reciprocal_rank is not None:
            reciprocal_ranks.append(reciprocal_rank)
        if concept_recall is not None:
            concept_recalls.append(concept_recall)
        expected_experience = set(case.expected_experience_source_ids)
        experience_recall = (
            len(expected_experience & set(result.experience_source_ids)) / len(expected_experience)
            if expected_experience else None
        )
        if experience_recall is not None:
            experience_recalls.append(experience_recall)
        forbidden_candidates = set(case.forbidden_candidate_experience_ids)
        returned_candidate_ids = {
            str(item.get("experience_id")) for item in result.knowledge_update_candidates
        }
        leaked_candidates = sorted(forbidden_candidates & returned_candidate_ids)
        if forbidden_candidates:
            candidate_checked_cases += 1
            candidate_leak_cases += int(bool(leaked_candidates))
        status_matches += int(result.status == case.expected_status)
        if not relevant:
            negative_matches += int(result.status == case.expected_status and not result.local_hits)
        request_chunks = {item.chunk_id: item for item in case.request.chunks}
        for hit in result.local_hits:
            hash_total += 1
            hash_checks += int(request_chunks[hit.chunk_id].chunk_sha256 == hit.chunk_sha256)
        rows.append(
            {
                "case_id": case.case_id,
                "top_k": case.request.query.top_k,
                "status": result.status.value,
                "retrieved_chunk_ids": retrieved,
                "recall_at_k": recall,
                "precision_at_k": precision,
                "ndcg_at_k": ndcg,
                "reciprocal_rank": reciprocal_rank,
                "concept_recall": concept_recall,
                "experience_recall": experience_recall,
                "experience_source_ids": result.experience_source_ids,
                "knowledge_candidate_experience_ids": sorted(returned_candidate_ids),
                "leaked_candidate_experience_ids": leaked_candidates,
            }
        )
    body = dict(
        benchmark_sha256=benchmark_sha256,
        retrieval_system_id=retrieval_system_id,
        case_count=len(cases),
        positive_case_count=len(positives),
        negative_case_count=len(negatives),
        mean_recall_at_k=sum(recalls) / len(recalls) if recalls else 0.0,
        mean_precision_at_k=sum(precisions) / len(precisions) if precisions else 0.0,
        mean_ndcg_at_k=sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
        mean_reciprocal_rank=(
            sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
        ),
        mean_concept_recall=sum(concept_recalls) / len(concept_recalls) if concept_recalls else 0.0,
        mean_experience_recall=(
            sum(experience_recalls) / len(experience_recalls) if experience_recalls else 1.0
        ),
        knowledge_candidate_leakage_rate=(
            candidate_leak_cases / candidate_checked_cases if candidate_checked_cases else 0.0
        ),
        status_accuracy=status_matches / len(cases),
        negative_rejection_rate=negative_matches / len(negatives) if negatives else 0.0,
        citation_hash_integrity=hash_checks / hash_total if hash_total else 1.0,
        case_results=rows,
    )
    provisional = RetrievalBenchmarkReport.model_construct(report_sha256="0" * 64, **body)
    report_sha256 = provisional.compute_sha256()
    return RetrievalBenchmarkReport(report_sha256=report_sha256, **body)
