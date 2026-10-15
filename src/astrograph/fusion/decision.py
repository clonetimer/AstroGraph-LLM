"""Evidence-aware *diagnostic* adjudication of the existing BiRRF result.

This is not flight/control authorization. RRF remains a rank score. Missing KG
coverage is unknown, never counter-evidence. A model-only hypothesis is retained
as such rather than discarded or advertised as a knowledge-verified diagnosis.
"""
from __future__ import annotations

from typing import Any
from astrograph.evidence import EpisodeEvidenceBundle
from .models import CandidateFusionPackage

# These legacy gates encode absent/low support, not an observed contradiction.
_COVERAGE_GATES = frozenset({
    "BACKWARD_EVIDENCE_COVERAGE_UNAVAILABLE",
    "HIGH_DATA_SCORE_LOW_BACKWARD_EVIDENCE_COVERAGE",
    "THREE_TASK_DETECTOR_NOT_TRIGGERED",
})


def _active_tops(ranking) -> list[str]:
    active = [c for c in ranking.candidates if c.contributes_to_fusion]
    if not active:
        return []
    rank = min(c.rank for c in active)
    return sorted(c.fault_id for c in active if c.rank == rank)


class FusionDecisionService:
    """Conservative evidence arbitration, without learned or label-tuned thresholds.

    No source weight is changed here. Counter-evidence can exclude a hypothesis;
    absence of knowledge cannot. Unresolved disagreement returns a candidate set.
    """
    policy_id = "evidence-aware-fusion.v1"

    def decide(self, *, bundle: EpisodeEvidenceBundle,
               package: CandidateFusionPackage, summary: dict[str, Any],
               detected: bool | None = None, identical_reference: bool = False,
               maximum_candidates: int = 3) -> dict[str, Any]:
        if maximum_candidates < 1:
            raise ValueError("maximum_candidates must be positive")
        if package.episode_id != bundle.episode.episode_id:
            raise ValueError("decision episode mismatch")
        fused = package.birrf_result
        # The decision consumes the complete candidate table, never a UI top-k.
        rows = {c["fault_id"]: c for c in summary["candidates"]}
        if set(rows) != {c.fault_id for c in fused.candidates}:
            raise ValueError("decision requires the complete fusion candidate table")
        declared = {d.target_fault_id for d in bundle.evidence_definitions}
        supported = {f for f, c in rows.items() if c["eligible_for_explanation"]}
        contradicted = {f for f, c in rows.items() if c["contradicting_evidence_ids"]}
        data_top_all = _active_tops(package.data_ranking)
        data_top = [f for f in data_top_all if f not in contradicted]
        kg_top = _active_tops(package.forward_ranking)
        blocking = sorted(set(fused.blocking_conflicts) - _COVERAGE_GATES)
        warnings = list(fused.quality_warnings)
        warnings += ["SUPPORT_GAP_NOT_COUNTEREVIDENCE:" + g
                     for g in fused.blocking_conflicts if g in _COVERAGE_GATES]
        if fused.open_set_actions:
            blocking.append("OPEN_SET_REQUIRES_REVIEW")
        selected = None
        reasons: list[str] = []
        proposed: set[str] = set()
        model_mechanism_disagreement = detected is False and bool(supported)
        if identical_reference:
            status = "NORMAL_REFERENCE_MATCH"
            reasons.append("IDENTICAL_PAIRED_INPUT_NO_OBSERVED_DEVIATION")
        elif blocking:
            status = "BLOCKED"
            reasons.extend(blocking)
        elif model_mechanism_disagreement:
            status = "CANDIDATE_SET"
            proposed = set(supported)
            reasons.append("DETECTOR_AND_EXECUTABLE_MECHANISM_DISAGREE")
        elif supported:
            proposed = set(supported) | set(data_top)
            nomination = fused.top_fault_id
            # A lower supported candidate may replace contradicted data tops, but
            # must not replace an uncovered competitor merely for lacking a KG rule.
            if len(proposed) == 1:
                selected = next(iter(proposed))
            elif (nomination in supported and data_top == [nomination]):
                selected = nomination
            if selected is not None:
                status = "SUPPORTED_SINGLETON"
                reasons.append("FUSED_HYPOTHESIS_WITH_EXECUTABLE_SUPPORT")
                if any(f != selected and f in contradicted for f in data_top_all):
                    reasons.append("DATA_TOP_REJECTED_BY_OBSERVED_COUNTEREVIDENCE")
            else:
                status = "CANDIDATE_SET"
                reasons.append("COMPETING_HYPOTHESES_REQUIRE_DISCRIMINATING_OBSERVATION")
        elif detected is False:
            status = "NO_DETECTED_ANOMALY"
            reasons.append("DETECTOR_NOT_TRIGGERED_IS_NOT_PROOF_OF_HEALTH")
        elif len(data_top) == 1:
            selected = data_top[0]
            proposed = set(data_top)
            status = "MODEL_ONLY"
            reasons.append("RETAIN_MODEL_HYPOTHESIS_WITHOUT_KG_CONFIRMATION")
            if selected not in declared:
                reasons.append("KG_COVERAGE_ABSENT_NOT_NEGATIVE_EVIDENCE")
            else:
                reasons.append("KG_SUPPORT_INCOMPLETE_OR_NOT_APPLICABLE")
        elif data_top:
            proposed = set(data_top)
            status = "CANDIDATE_SET"
            reasons.append("DATA_RANK_TIE_RETAINED")
        else:
            status = "INSUFFICIENT_EVIDENCE"
            reasons.append("NO_NONCONTRADICTED_SUPPORTED_OR_DATA_HYPOTHESIS")
        # Preserve overflow candidates in an audit field; never choose alphabetically.
        overflow = len(proposed) > maximum_candidates and selected is None
        if overflow:
            status = "CANDIDATE_OVERFLOW"
            reasons.append("CANDIDATE_BUDGET_EXCEEDED_WITHOUT_ARBITRARY_TRUNCATION")
        candidate_ids = [selected] if selected else sorted(proposed)
        # Keep a *working preference* separate from a supported singleton. A
        # competing KG explanation is not sufficient to erase a noncontradicted
        # data hypothesis. This preference NEVER changes the confirmed decision,
        # suppresses alternatives, stops the Agent, or authorizes control.
        working = selected
        working_basis = "CURRENT_DIAGNOSTIC_HYPOTHESIS" if selected else None
        if (status == "CANDIDATE_SET" and not model_mechanism_disagreement
                and len(data_top) == 1 and data_top[0] in proposed):
            working = data_top[0]
            working_basis = "NONCONTRADICTED_DATA_TOP_PENDING_KNOWLEDGE_DISAMBIGUATION"
        explanation_rows = []
        for candidate in sorted(fused.candidates, key=lambda c: (c.final_rank, c.fault_id)):
            f = candidate.fault_id
            row = rows[f]
            coverage = ("CONTRADICTED" if f in contradicted else
                        "SUPPORTED" if f in supported else
                        "UNCOVERED" if f not in declared else "UNKNOWN_OR_INCOMPLETE")
            explanation_rows.append({
                "fault_id": f, "data_top": f in data_top_all,
                "selected": f == selected, "knowledge_state": coverage,
                "fusion_rank": candidate.final_rank,
                "fusion_score": candidate.fused_score,
                "source_ranks": candidate.source_ranks,
                "reciprocal_contributions": candidate.reciprocal_contributions,
                "dependency_adjusted": candidate.dependency_adjusted,
                "supporting_evidence_ids": row["supporting_evidence_ids"],
                "contradicting_evidence_ids": row["contradicting_evidence_ids"],
                "missing_required_definition_ids": row["missing_required_definition_ids"],
                "graph_paths": row["graph_paths"],
            })
        return {
            "schema_version": "astrograph-fusion-diagnostic-decision.v1",
            "policy_id": self.policy_id, "status": status,
            "selected_fault_id": selected, "candidate_fault_ids": candidate_ids,
            "working_hypothesis_fault_id": working,
            "working_hypothesis_basis": working_basis,
            "working_hypothesis_is_confirmed": False,
            "requires_discriminating_evidence": (status in {"CANDIDATE_SET", "CANDIDATE_OVERFLOW", "MODEL_ONLY", "INSUFFICIENT_EVIDENCE"}
                or (status == "NO_DETECTED_ANOMALY" and any(c["missing_required_definition_ids"] for c in rows.values()))),
            "alternatives": sorted(proposed - ({selected} if selected else set())),
            "candidate_overflow": overflow, "candidate_set_is_confirmed_class": False,
            "knowledge_verified": status == "SUPPORTED_SINGLETON",
            "detected": detected, "detector_mechanism_disagreement": model_mechanism_disagreement,
            "data_top_fault_ids": data_top_all, "knowledge_top_fault_ids": kg_top,
            "fusion_top_fault_id": fused.top_fault_id,
            "decision_changed_data_top": bool(selected and data_top_all != [selected]),
            "blocking_conflicts": blocking, "quality_warnings": sorted(set(warnings)),
            "reason_codes": reasons, "candidate_explanations": explanation_rows,
            "score_semantics": "RECIPROCAL_RANK_SCORE_NOT_PROBABILITY",
            "calibrated_probability": None, "control_actions_authorized": False,
            "root_cause_confirmed": False,
            "claim_boundary": "Diagnostic hypothesis at the configured fault granularity; not flight qualification or independently confirmed ground truth.",
        }


def render_decision(decision: dict[str, Any]) -> dict[str, Any]:
    """Deterministic, inspectable report; works without an LLM or external service."""
    selected = decision["selected_fault_id"]
    if selected:
        level = "有可执行机理支持的诊断假设" if decision["knowledge_verified"] else "仅数据模型支持的待核验假设"
        conclusion = f"{selected}：{level}。不代表已确认或唯一根因。"
    else:
        conclusion = "当前状态：" + decision["status"] + "；未选择唯一故障类别。"
        if decision.get("working_hypothesis_fault_id"):
            conclusion += "优先检查假设为 " + decision["working_hypothesis_fault_id"] + "，不是知识已确认结论；竞争候选仍然保留。"
    citations = sorted({e for row in decision["candidate_explanations"]
                        for key in ("supporting_evidence_ids", "contradicting_evidence_ids")
                        for e in row[key]})
    return {"generation": "DETERMINISTIC_EVIDENCE_REPORT", "conclusion": conclusion,
            "reason_codes": decision["reason_codes"], "evidence_ids": citations,
            "candidate_explanations": decision["candidate_explanations"],
            "limitations": ["BiRRF 分数不是故障概率。", "相似案例和未核实文档不是当前设备观测。",
                            "未授权设备控制；结论仍需独立核实。"]}
