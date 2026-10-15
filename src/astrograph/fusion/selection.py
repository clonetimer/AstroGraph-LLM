from __future__ import annotations

from astrograph.llm.candidate_selection import (
    ACTIVE_CANDIDATE_STATUSES,
    CandidateSelectionResult,
    CandidateSignal,
    RankedCandidate,
    candidate_display_score,
)

from .models import CandidateFusionPackage, RankingSource


def _reference_by_fault(reference_items: list[dict]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for item in reference_items:
        if item.get("hypothesis_type") != "REGISTERED_FAULT" or not item.get("target_fault_id"):
            continue
        result[str(item["target_fault_id"])] = item
    return result


def _ranked(item: CandidateSignal) -> RankedCandidate:
    return RankedCandidate(
        fault_id=item.fault_id,
        status=item.status,
        score=candidate_display_score(item),
        supporting_count=len(item.supporting_evidence_fact_ids),
        contradicting_count=len(item.contradicting_evidence_fact_ids),
        missing_count=len(item.missing_evidence_definition_ids),
        source=item.source,
        source_rank=item.source_rank,
        source_score=item.source_score,
        source_score_semantics=item.source_score_semantics,
    )


def select_fused_candidates(
    package: CandidateFusionPackage,
    reference_items: list[dict],
    *,
    mode: str = "fixed",
    configured_top_k: int = 4,
    min_top_k: int = 2,
    max_top_k: int = 6,
) -> tuple[CandidateSelectionResult, dict]:
    if mode not in {"fixed", "adaptive_forced_count"}:
        raise ValueError(f"unsupported candidate Top-K mode: {mode}")
    if not 1 <= configured_top_k <= 16:
        raise ValueError("configured_top_k must be between 1 and 16")
    if not 1 <= min_top_k <= max_top_k <= 16:
        raise ValueError("invalid adaptive Top-K bounds")

    reference = _reference_by_fault(reference_items)
    candidates: list[CandidateSignal] = []
    for fused in package.birrf_result.candidates:
        item = reference.get(fused.fault_id, {})
        candidates.append(
            CandidateSignal(
                fault_id=fused.fault_id,
                status=str(item.get("status", "PROPOSED")),
                supporting_evidence_fact_ids=[str(value) for value in item.get("supporting_evidence_fact_ids", [])],
                contradicting_evidence_fact_ids=[str(value) for value in item.get("contradicting_evidence_fact_ids", [])],
                missing_evidence_definition_ids=[str(value) for value in item.get("missing_evidence_definition_ids", [])],
                source="BIRRF_FUSION",
                source_rank=fused.final_rank,
                source_score=fused.fused_score,
                source_score_semantics=package.birrf_result.fused_score_semantics,
            )
        )

    active_ids = {
        item.fault_id for item in candidates if item.status in ACTIVE_CANDIDATE_STATUSES
    }
    branch_tops = {
        item.fault_id
        for ranking in (
            package.data_ranking,
            package.forward_ranking,
            package.backward_ranking,
        )
        for item in ranking.candidates
        if item.rank == min((candidate.rank for candidate in ranking.candidates), default=0)
    }
    mandatory_ids = active_ids | branch_tops
    if mode == "fixed":
        effective_top_k = configured_top_k
    else:
        effective_top_k = max(min_top_k, min(max_top_k, len(mandatory_ids)))

    all_contradicted = bool(candidates) and all(item.status == "CONTRADICTED" for item in candidates)
    normal_is_top = package.birrf_result.normal_rank == 1
    if all_contradicted and normal_is_top:
        open_set_review = bool(package.birrf_result.open_set_actions)
        result = CandidateSelectionResult(
            top_k=effective_top_k,
            selected_fault_ids=[],
            ranked_candidates=[_ranked(item) for item in candidates],
            forced_candidate_count=len(mandatory_ids),
            nominal_short_circuit=True,
            compact_reasoning_allowed=not open_set_review,
            candidate_set_status=("OPEN_SET_REVIEW_REQUIRED" if open_set_review else "NOMINAL_SHORT_CIRCUIT"),
            escape_action=("DISABLE_COMPACT_REASONING" if open_set_review else "NONE"),
            escape_reason_codes=(["OPEN_SET_REQUIRES_UNTRUNCATED_CONTEXT"] if open_set_review else []),
        )
    else:
        ordered_ids = [item.fault_id for item in candidates]
        selected_ordered = [fault_id for fault_id in ordered_ids if fault_id in mandatory_ids]
        selected_ordered.extend(fault_id for fault_id in ordered_ids if fault_id not in mandatory_ids)
        selected_ordered = selected_ordered[:effective_top_k]
        selected_ids = sorted(selected_ordered)
        unselected_mandatory = sorted(mandatory_ids - set(selected_ids))
        open_set_review = bool(package.birrf_result.open_set_actions)
        compact_allowed = not unselected_mandatory and not open_set_review
        if open_set_review:
            candidate_set_status = "OPEN_SET_REVIEW_REQUIRED"
        elif unselected_mandatory:
            candidate_set_status = "MANDATORY_CANDIDATE_OVERFLOW"
        else:
            candidate_set_status = "COMPLETE"
        reason_codes = []
        if unselected_mandatory:
            reason_codes.append("MANDATORY_CANDIDATES_EXCEED_TOP_K")
        if open_set_review:
            reason_codes.append("OPEN_SET_REQUIRES_UNTRUNCATED_CONTEXT")
        result = CandidateSelectionResult(
            top_k=effective_top_k,
            selected_fault_ids=selected_ids,
            ranked_candidates=[_ranked(item) for item in candidates],
            forced_candidate_count=len(mandatory_ids),
            nominal_short_circuit=False,
            compact_reasoning_allowed=compact_allowed,
            candidate_set_status=candidate_set_status,
            escape_action=("NONE" if compact_allowed else "DISABLE_COMPACT_REASONING"),
            unselected_mandatory_fault_ids=unselected_mandatory,
            escape_reason_codes=reason_codes,
        )

    snapshot = result.model_dump(mode="json")
    snapshot.update(
        {
            "selection_source": "BIRRF_FUSION",
            "top_k_mode": mode,
            "configured_top_k": configured_top_k,
            "effective_top_k": effective_top_k,
            "candidate_top_k_min": min_top_k,
            "candidate_top_k_max": max_top_k,
            "mandatory_fault_ids": sorted(mandatory_ids),
            "branch_top_fault_ids": {
                RankingSource.DATA_DRIVEN.value: package.birrf_result.metadata.get("data_top_fault_id"),
                RankingSource.KG_FORWARD.value: package.birrf_result.metadata.get("forward_top_fault_id"),
                RankingSource.KG_BACKWARD.value: package.birrf_result.metadata.get("backward_top_fault_id"),
            },
            "branch_top_fault_id_sets": {
                RankingSource.DATA_DRIVEN.value: package.birrf_result.metadata.get("data_top_fault_ids", []),
                RankingSource.KG_FORWARD.value: package.birrf_result.metadata.get("forward_top_fault_ids", []),
                RankingSource.KG_BACKWARD.value: package.birrf_result.metadata.get("backward_top_fault_ids", []),
            },
            "fusion_id": package.birrf_result.fusion_id,
            "conflict_flags": package.birrf_result.conflict_flags,
        }
    )
    return result, snapshot
