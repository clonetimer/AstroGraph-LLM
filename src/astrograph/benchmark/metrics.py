"""Explicit-denominator diagnosis metrics; unavailable measurements remain null.

Ground truth belongs here, never in the OAG inference request. A nonempty evidence
list is not evidence coverage, and schema validity is not a hallucination test.
"""
from __future__ import annotations

from collections import Counter
from typing import Literal
import math

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ClaimAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    claim_id: str = Field(min_length=1)
    support: Literal["SUPPORTED", "UNSUPPORTED", "UNASSESSED"]
    assessor_id: str = Field(min_length=1)
    assessment_basis: str = Field(min_length=1)


class BenchmarkRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    case_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    true_fault_id: str = Field(min_length=1)
    predicted_fault_id: str | None = None
    ranked_fault_ids: list[str] = Field(default_factory=list)
    cited_evidence_ids: list[str] = Field(default_factory=list)
    # None = no annotation supplied; [] = annotated, no applicable evidence.
    expected_evidence_ids: list[str] | None = None
    claim_assessments: list[ClaimAssessment] | None = None
    # Caller must specify what was measured; do not compare unlike scopes.
    latency_s: float = Field(ge=0)
    latency_scope: Literal["END_TO_END", "CORE_ONLY", "REPLAY"] = "END_TO_END"

    @field_validator("case_id", "dataset_id", "group_id", "true_fault_id")
    @classmethod
    def nonblank_identity(cls, value):
        if not value.strip():
            raise ValueError("case identities and ground truth must be nonblank")
        return value

    @field_validator("ranked_fault_ids", "cited_evidence_ids", "expected_evidence_ids")
    @classmethod
    def unique_ids(cls, value):
        if value is None:
            return value
        if any(not item.strip() for item in value):
            raise ValueError("identifiers must be nonblank")
        if len(value) != len(set(value)):
            raise ValueError("duplicate identifiers are not independent evidence/candidates")
        return value

    @model_validator(mode="after")
    def validate_claim_ids(self):
        ids = [item.claim_id for item in (self.claim_assessments or [])]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate claim IDs")
        if self.predicted_fault_id is not None and not self.predicted_fault_id.strip():
            raise ValueError("predicted_fault_id must be nonblank or None (abstention)")
        return self


def _ratio(n: int, d: int) -> float | None:
    return n / d if d else None


def _percentile(values: list[float], q: float) -> float:
    """Linear interpolation, same definition for all sample sizes."""
    data = sorted(values)
    point = (len(data) - 1) * q
    lo, hi = math.floor(point), math.ceil(point)
    return data[lo] + (data[hi] - data[lo]) * (point - lo)


def evaluate(records: list[BenchmarkRecord | dict], *, k: int = 3) -> dict:
    """Evaluate one arm. Duplicate runs and mixed latency scopes are errors.

    Evidence coverage is micro-averaged recall of independently annotated evidence
    IDs. Unsupported-claim rate is evaluated only on explicitly assessed claims;
    report its review coverage alongside it, never assume unassessed = supported.
    """
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    if not records:
        raise ValueError("cannot evaluate an empty benchmark")
    rows = [BenchmarkRecord.model_validate(item.model_dump() if isinstance(item, BenchmarkRecord) else item)
            for item in records]
    keys = [(r.dataset_id, r.case_id) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate (dataset_id, case_id) records")
    scopes = {r.latency_scope for r in rows}
    if len(scopes) != 1:
        raise ValueError("mixed latency scopes are not comparable")
    correct = sum(r.predicted_fault_id == r.true_fault_id for r in rows)
    topk_hits = sum(r.true_fault_id in r.ranked_fault_ids[:k] for r in rows)
    annotated = [r for r in rows if r.expected_evidence_ids is not None]
    required = sum(len(r.expected_evidence_ids or []) for r in annotated)
    covered = sum(len(set(r.expected_evidence_ids or []) & set(r.cited_evidence_ids)) for r in annotated)
    assessments = [c for r in rows for c in (r.claim_assessments or [])]
    counts = Counter(c.support for c in assessments)
    reviewed = counts["SUPPORTED"] + counts["UNSUPPORTED"]
    latencies = [r.latency_s for r in rows]
    return {
        "schema_version": "astrograph-benchmark-metrics.v2",
        "case_count": len(rows),
        "independent_group_count": len({(r.dataset_id, r.group_id) for r in rows}),
        "accuracy": correct / len(rows),
        "top_k": k,
        "top_k_recall": topk_hits / len(rows),
        "evidence_coverage": _ratio(covered, required),
        "hallucination_rate": _ratio(counts["UNSUPPORTED"], reviewed),
        "claim_review_coverage": _ratio(reviewed, len(assessments)),
        "diagnosis_latency_s": {
            "scope": next(iter(scopes)), "mean": sum(latencies) / len(rows),
            "p50": _percentile(latencies, .5), "p95": _percentile(latencies, .95),
            "min": min(latencies), "max": max(latencies),
        },
        "denominators": {"accuracy_cases": len(rows), "top_k_cases": len(rows),
                         "annotated_evidence_cases": len(annotated),
                         "required_evidence_ids": required, "covered_evidence_ids": covered,
                         "submitted_claims": len(assessments), "assessed_claims": reviewed,
                         "unsupported_claims": counts["UNSUPPORTED"]},
        "claim_boundary": (
            "Single-label exact-match benchmark. Evidence coverage requires independent annotation. "
            "Hallucination rate means unsupported/assessed claims under the supplied assessor, not "
            "an automatic guarantee of semantic truth. Null denotes unavailable/not applicable."
        ),
    }
