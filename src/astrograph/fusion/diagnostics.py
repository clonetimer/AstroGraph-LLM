from __future__ import annotations

from dataclasses import dataclass

from .models import BiRRFConfig, BiRRFFault, FaultRanking, RankingSource, ScoreSemantics


@dataclass(frozen=True)
class FusionDiagnosticReport:
    blocking_conflicts: tuple[str, ...]
    quality_warnings: tuple[str, ...]
    kg_branch_correlation: float | None
    kg_top1_overlap: float | None
    kg_candidate_overlap: float | None
    data_conflict_gate_applied: bool
    data_score_semantics: str

    @property
    def conflict_flags(self) -> list[str]:
        return list(dict.fromkeys([*self.blocking_conflicts, *self.quality_warnings]))


def rank_correlation(ranking1: FaultRanking, ranking2: FaultRanking) -> float | None:
    faults1 = {candidate.fault_id: candidate.rank for candidate in ranking1.candidates}
    faults2 = {candidate.fault_id: candidate.rank for candidate in ranking2.candidates}
    common_faults = sorted(set(faults1) & set(faults2))
    if len(common_faults) < 2:
        return None
    left = [float(faults1[fault]) for fault in common_faults]
    right = [float(faults2[fault]) for fault in common_faults]
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_delta = [value - left_mean for value in left]
    right_delta = [value - right_mean for value in right]
    denominator = (
        sum(value * value for value in left_delta) * sum(value * value for value in right_delta)
    ) ** 0.5
    if denominator <= 0.0:
        return None
    correlation = sum(a * b for a, b in zip(left_delta, right_delta)) / denominator
    return max(-1.0, min(1.0, correlation))


def candidate_overlap(ranking1: FaultRanking, ranking2: FaultRanking) -> float | None:
    faults1 = {candidate.fault_id for candidate in ranking1.candidates}
    faults2 = {candidate.fault_id for candidate in ranking2.candidates}
    union = faults1 | faults2
    return len(faults1 & faults2) / len(union) if union else None


def top_fault_ids(ranking: FaultRanking) -> list[str]:
    if not ranking.candidates:
        return []
    best_rank = min(item.rank for item in ranking.candidates)
    return sorted(item.fault_id for item in ranking.candidates if item.rank == best_rank)


def unique_top_fault_id(ranking: FaultRanking) -> str | None:
    top = top_fault_ids(ranking)
    return top[0] if len(top) == 1 else None


def evaluate_fusion_diagnostics(
    *,
    data_ranking: FaultRanking,
    forward_ranking: FaultRanking,
    backward_ranking: FaultRanking,
    candidates: list[BiRRFFault],
    config: BiRRFConfig,
    collapsed_kg_source: str | None,
) -> FusionDiagnosticReport:
    data_top_ids = top_fault_ids(data_ranking)
    forward_top_ids = top_fault_ids(forward_ranking)
    backward_top_ids = top_fault_ids(backward_ranking)
    data_top = data_top_ids[0] if len(data_top_ids) == 1 else None
    forward_top = forward_top_ids[0] if len(forward_top_ids) == 1 else None
    backward_top = backward_top_ids[0] if len(backward_top_ids) == 1 else None

    kg_top1_overlap: float | None = None
    if forward_top_ids and backward_top_ids:
        top_union = set(forward_top_ids) | set(backward_top_ids)
        kg_top1_overlap = len(set(forward_top_ids) & set(backward_top_ids)) / len(top_union)
    kg_branch_correlation = rank_correlation(forward_ranking, backward_ranking)
    kg_candidate_overlap = candidate_overlap(forward_ranking, backward_ranking)

    quality_warnings: list[str] = []
    if len(data_top_ids) > 1:
        quality_warnings.append(f"DATA_TOP_RANK_TIED={len(data_top_ids)}")
    if len(forward_top_ids) > 1:
        quality_warnings.append(f"KG_FORWARD_TOP_RANK_TIED={len(forward_top_ids)}")
    if len(backward_top_ids) > 1:
        quality_warnings.append(f"KG_BACKWARD_TOP_RANK_TIED={len(backward_top_ids)}")
    if collapsed_kg_source is not None:
        quality_warnings.append(f"KG_REDUNDANT_BRANCH_COLLAPSED={collapsed_kg_source}")
    if data_top and forward_top and data_top != forward_top:
        quality_warnings.append("DATA_TOP_DIFFERS_FROM_FORWARD_GRAPH_TOP")
    if data_top and backward_top and data_top != backward_top:
        quality_warnings.append("DATA_TOP_DIFFERS_FROM_BACKWARD_GRAPH_TOP")
    if kg_branch_correlation is not None and kg_branch_correlation > 0.7:
        quality_warnings.append(f"KG_BRANCH_HIGH_CORRELATION={kg_branch_correlation:.3f}")
    if kg_top1_overlap == 0.0:
        quality_warnings.append(
            f"KG_BRANCH_TOP1_MISMATCH: forward={forward_top}, backward={backward_top}"
        )

    backward_by_fault = {item.fault_id: item for item in backward_ranking.candidates}
    blocking_conflicts: list[str] = []
    data_conflict_gate_applied = False
    if data_top is not None:
        top = next((item for item in data_ranking.candidates if item.fault_id == data_top), None)
        if top is None:
            raise ValueError("unique data top fault is absent from data ranking candidates")
        calibrated_probability = (
            data_ranking.score_semantics == ScoreSemantics.CALIBRATED_CLASS_PROBABILITY
            and bool(data_ranking.calibration_id)
        )
        if not calibrated_probability:
            quality_warnings.append(
                "DATA_CONFLICT_GATE_SKIPPED_UNCALIBRATED_SCORE="
                f"{data_ranking.score_semantics.value}"
            )
        else:
            data_conflict_gate_applied = True
            backward_item = backward_by_fault.get(top.fault_id)
            backward_coverage = (
                float(backward_item.metadata["coverage"])
                if backward_item is not None and "coverage" in backward_item.metadata
                else None
            )
            fused_top_ids = sorted(item.fault_id for item in candidates if item.final_rank == 1)
            fused_top = fused_top_ids[0] if len(fused_top_ids) == 1 else None
            if top.score >= config.data_conflict_probability and backward_coverage is None:
                quality_warnings.append("BACKWARD_EVIDENCE_COVERAGE_UNAVAILABLE")
                if fused_top == top.fault_id:
                    blocking_conflicts.append("BACKWARD_EVIDENCE_COVERAGE_UNAVAILABLE")
            elif (
                top.score >= config.data_conflict_probability
                and backward_coverage is not None
                and backward_coverage < config.backward_conflict_coverage
            ):
                if fused_top == top.fault_id:
                    blocking_conflicts.append("HIGH_DATA_SCORE_LOW_BACKWARD_EVIDENCE_COVERAGE")
                else:
                    quality_warnings.append("HIGH_DATA_SCORE_LOW_BACKWARD_EVIDENCE_COVERAGE")
                    if fused_top is not None:
                        quality_warnings.append(f"DATA_TOP_OVERRIDDEN_BY_FUSION={fused_top}")

    return FusionDiagnosticReport(
        blocking_conflicts=tuple(blocking_conflicts),
        quality_warnings=tuple(quality_warnings),
        kg_branch_correlation=kg_branch_correlation,
        kg_top1_overlap=kg_top1_overlap,
        kg_candidate_overlap=kg_candidate_overlap,
        data_conflict_gate_applied=data_conflict_gate_applied,
        data_score_semantics=data_ranking.score_semantics.value,
    )
