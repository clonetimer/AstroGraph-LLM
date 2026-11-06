from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CandidateStatus = Literal[
    "PROPOSED",
    "UNDER_VERIFICATION",
    "SUPPORTED",
    "CONTRADICTED",
    "INSUFFICIENT_EVIDENCE",
    "OUT_OF_SCOPE",
]
CandidateSource = Literal[
    "REFERENCE_HYPOTHESIS_SET",
    "EVIDENCE_NAVIGATOR",
    "DETERMINISTIC_PERTURBATION",
    "BIRRF_FUSION",
]
CandidateTopKMode = Literal["fixed", "adaptive_forced_count"]
CandidateSetStatus = Literal[
    "COMPLETE",
    "NOMINAL_SHORT_CIRCUIT",
    "ACTIVE_CANDIDATE_OVERFLOW",
    "MANDATORY_CANDIDATE_OVERFLOW",
    "OPEN_SET_REVIEW_REQUIRED",
]
CandidateEscapeAction = Literal["NONE", "DISABLE_COMPACT_REASONING"]

ACTIVE_CANDIDATE_STATUSES = {
    "SUPPORTED",
    "INSUFFICIENT_EVIDENCE",
    "UNDER_VERIFICATION",
}


class CandidateSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fault_id: str = Field(min_length=1)
    status: CandidateStatus
    supporting_evidence_fact_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_fact_ids: list[str] = Field(default_factory=list)
    missing_evidence_definition_ids: list[str] = Field(default_factory=list)
    source: CandidateSource = "REFERENCE_HYPOTHESIS_SET"
    source_rank: int | None = Field(default=None, ge=1)
    source_score: float | None = Field(default=None, allow_inf_nan=False)
    source_score_semantics: str | None = None


class CandidateSelectionInput(BaseModel):
    """Stable boundary for deterministic and future Evidence Navigator candidates."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["astrograph-candidate-selection-input.v1"] = (
        "astrograph-candidate-selection-input.v1"
    )
    registered_fault_ids: list[str]
    candidates: list[CandidateSignal]
    top_k: int = Field(default=4, ge=1, le=16)
    nominal_if_all_contradicted: bool = True

    @model_validator(mode="after")
    def validate_candidate_whitelist(self) -> "CandidateSelectionInput":
        registered = set(self.registered_fault_ids)
        candidate_ids = [item.fault_id for item in self.candidates]
        unknown = sorted(set(candidate_ids) - registered)
        if unknown:
            raise ValueError(f"candidate fault IDs are not registered: {unknown}")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate fault IDs must be unique")
        return self


class RankedCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fault_id: str
    status: CandidateStatus
    score: int = Field(description="Deprecated display-only ordinal; selection uses lexicographic evidence semantics.")
    plausibility_rank: int | None = Field(default=None, ge=1)
    investigation_priority: int = Field(default=0, ge=0)
    ranking_reason_codes: list[str] = Field(default_factory=list)
    supporting_count: int
    contradicting_count: int
    missing_count: int
    source: CandidateSource
    source_rank: int | None = None
    source_score: float | None = None
    source_score_semantics: str | None = None


class CandidateSelectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["astrograph-candidate-selection-result.v1"] = (
        "astrograph-candidate-selection-result.v1"
    )
    top_k: int
    selected_fault_ids: list[str]
    ranked_candidates: list[RankedCandidate]
    forced_candidate_count: int
    nominal_short_circuit: bool
    compact_reasoning_allowed: bool = True
    candidate_set_status: CandidateSetStatus = "COMPLETE"
    escape_action: CandidateEscapeAction = "NONE"
    unselected_mandatory_fault_ids: list[str] = Field(default_factory=list)
    escape_reason_codes: list[str] = Field(default_factory=list)
    ranking_semantics: str = (
        "LEXICOGRAPHIC: status > fewer contradictions > more support > fewer missing > source score > source rank"
    )


def _status_priority(status: CandidateStatus) -> int:
    return {
        "SUPPORTED": 6,
        "INSUFFICIENT_EVIDENCE": 5,
        "UNDER_VERIFICATION": 4,
        "PROPOSED": 3,
        "OUT_OF_SCOPE": 2,
        "CONTRADICTED": 1,
    }[status]


def candidate_score(item: CandidateSignal) -> tuple[int, int, int, int, float, int, str]:
    """Return the deterministic lexicographic plausibility key.

    Missing evidence is never a positive plausibility signal, and contradiction
    precedence cannot be overwhelmed by accumulating support counts.
    """
    support = len(item.supporting_evidence_fact_ids)
    missing = len(item.missing_evidence_definition_ids)
    contradicting = len(item.contradicting_evidence_fact_ids)
    source_score = float(item.source_score) if item.source_score is not None else float("-inf")
    source_rank = -(item.source_rank or 10**9)
    return (
        _status_priority(item.status),
        -contradicting,
        support,
        -missing,
        source_score,
        source_rank,
        item.fault_id,
    )


def candidate_display_score(item: CandidateSignal) -> int:
    """Backward-compatible display ordinal only; never use it for selection."""
    support = min(len(item.supporting_evidence_fact_ids), 99)
    missing = min(len(item.missing_evidence_definition_ids), 99)
    contradicting = min(len(item.contradicting_evidence_fact_ids), 99)
    return _status_priority(item.status) * 10000 - contradicting * 100 + support * 10 - missing


def _investigation_priority(item: CandidateSignal) -> int:
    if item.status == "INSUFFICIENT_EVIDENCE":
        return 100 + min(len(item.missing_evidence_definition_ids), 99)
    if item.status == "UNDER_VERIFICATION":
        return 80 + min(len(item.missing_evidence_definition_ids), 19)
    if item.status == "PROPOSED":
        return 50 + min(len(item.missing_evidence_definition_ids), 19)
    return 0


def _ranking_reason_codes(item: CandidateSignal) -> list[str]:
    reasons = [f"STATUS_{item.status}"]
    if item.contradicting_evidence_fact_ids:
        reasons.append("HAS_CONTRADICTION")
    if item.supporting_evidence_fact_ids:
        reasons.append("HAS_SUPPORT")
    if item.missing_evidence_definition_ids:
        reasons.append("HAS_MISSING_EVIDENCE")
    if item.source_score is not None:
        reasons.append("HAS_SOURCE_SCORE")
    return reasons


def resolve_candidate_top_k(
    candidates: list[CandidateSignal],
    *,
    mode: CandidateTopKMode = "fixed",
    configured_top_k: int = 4,
    min_top_k: int = 2,
    max_top_k: int = 6,
) -> int:
    """Resolve the effective candidate count without changing ranking semantics.

    ``adaptive_forced_count`` keeps the clean path compact while expanding when
    the deterministic evidence layer marks more candidates as active or uncertain.
    The returned value is always bounded to the public selector range [1, 16].
    """

    if not 1 <= configured_top_k <= 16:
        raise ValueError("configured_top_k must be between 1 and 16")
    if not 1 <= min_top_k <= 16 or not 1 <= max_top_k <= 16:
        raise ValueError("adaptive Top-K bounds must be between 1 and 16")
    if min_top_k > max_top_k:
        raise ValueError("min_top_k must not exceed max_top_k")
    if mode == "fixed":
        return configured_top_k
    if mode != "adaptive_forced_count":
        raise ValueError(f"unsupported candidate Top-K mode: {mode}")
    forced_count = sum(item.status in ACTIVE_CANDIDATE_STATUSES for item in candidates)
    return max(min_top_k, min(max_top_k, forced_count))


def select_candidates(request: CandidateSelectionInput) -> CandidateSelectionResult:
    ranked = sorted(request.candidates, key=candidate_score, reverse=True)
    forced = [
        item
        for item in ranked
        if item.status in ACTIVE_CANDIDATE_STATUSES
    ]
    if request.nominal_if_all_contradicted and not forced:
        return CandidateSelectionResult(
            top_k=request.top_k,
            selected_fault_ids=[],
            ranked_candidates=[_ranked(item, plausibility_rank=index) for index, item in enumerate(ranked, start=1)],
            forced_candidate_count=0,
            nominal_short_circuit=True,
            candidate_set_status="NOMINAL_SHORT_CIRCUIT",
        )
    selected = forced[: request.top_k]
    if len(selected) < request.top_k:
        selected_ids = {item.fault_id for item in selected}
        remaining = [item for item in ranked if item.fault_id not in selected_ids]
        selected.extend(remaining[: request.top_k - len(selected)])
    # Keep the V0.5.9 public payload deterministic and order-insensitive.
    selected_ids = sorted(item.fault_id for item in selected)
    selected_set = set(selected_ids)
    unselected_forced = sorted(item.fault_id for item in forced if item.fault_id not in selected_set)
    compact_allowed = not unselected_forced
    return CandidateSelectionResult(
        top_k=request.top_k,
        selected_fault_ids=selected_ids,
        ranked_candidates=[_ranked(item, plausibility_rank=index) for index, item in enumerate(ranked, start=1)],
        forced_candidate_count=len(forced),
        nominal_short_circuit=False,
        compact_reasoning_allowed=compact_allowed,
        candidate_set_status=("COMPLETE" if compact_allowed else "ACTIVE_CANDIDATE_OVERFLOW"),
        escape_action=("NONE" if compact_allowed else "DISABLE_COMPACT_REASONING"),
        unselected_mandatory_fault_ids=unselected_forced,
        escape_reason_codes=([] if compact_allowed else ["ACTIVE_CANDIDATES_EXCEED_TOP_K"]),
    )


def _ranked(item: CandidateSignal, *, plausibility_rank: int | None = None) -> RankedCandidate:
    return RankedCandidate(
        fault_id=item.fault_id,
        status=item.status,
        score=candidate_display_score(item),
        plausibility_rank=plausibility_rank,
        investigation_priority=_investigation_priority(item),
        ranking_reason_codes=_ranking_reason_codes(item),
        supporting_count=len(item.supporting_evidence_fact_ids),
        contradicting_count=len(item.contradicting_evidence_fact_ids),
        missing_count=len(item.missing_evidence_definition_ids),
        source=item.source,
        source_rank=item.source_rank,
        source_score=item.source_score,
        source_score_semantics=item.source_score_semantics,
    )


def selection_input_from_reference(
    reference_items: list[dict],
    *,
    registered_fault_ids: list[str],
    top_k: int,
) -> CandidateSelectionInput:
    candidates = []
    for item in reference_items:
        fault_id = item.get("target_fault_id")
        if item.get("hypothesis_type") != "REGISTERED_FAULT" or not fault_id:
            continue
        candidates.append(CandidateSignal(
            fault_id=str(fault_id),
            status=str(item.get("status", "OUT_OF_SCOPE")),
            supporting_evidence_fact_ids=[str(value) for value in item.get("supporting_evidence_fact_ids", [])],
            contradicting_evidence_fact_ids=[str(value) for value in item.get("contradicting_evidence_fact_ids", [])],
            missing_evidence_definition_ids=[str(value) for value in item.get("missing_evidence_definition_ids", [])],
        ))
    return CandidateSelectionInput(
        registered_fault_ids=registered_fault_ids,
        candidates=candidates,
        top_k=top_k,
    )
