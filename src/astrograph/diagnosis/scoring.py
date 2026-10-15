from __future__ import annotations

from collections import defaultdict

from astrograph.config import DiagnosisPolicy
from astrograph.evidence import EvidenceConstraintCatalog
from astrograph.schemas import AnomalyEvidence, CandidateScore, DiagnosisResult, DiagnosisStatus, KnowledgeEdge


def diagnose(
    anomalies: list[AnomalyEvidence],
    edges: list[KnowledgeEdge],
    *,
    policy: DiagnosisPolicy | None = None,
    knowledge_version: str = "",
    operating_condition_ids: set[str] | None = None,
    constraints: EvidenceConstraintCatalog | None = None,
    temporal_evaluations: dict[str, list[dict]] | None = None,
) -> DiagnosisResult:
    policy = policy or DiagnosisPolicy()
    if not anomalies:
        return DiagnosisResult(
            status=DiagnosisStatus.UNKNOWN,
            reasons=["No anomaly evidence provided."],
            policy_version=policy.policy_version,
            knowledge_version=knowledge_version,
            claim_boundary=policy.claim_boundary,
        )
    if not edges:
        return DiagnosisResult(
            status=DiagnosisStatus.UNKNOWN,
            reasons=["No reviewed/provisional knowledge edges matched the anomalies."],
            policy_version=policy.policy_version,
            knowledge_version=knowledge_version,
            claim_boundary=policy.claim_boundary,
        )

    confidence = {a.anomaly_id: a.confidence for a in anomalies}
    operating_condition_ids = operating_condition_ids or set()
    temporal_evaluations = temporal_evaluations or {}
    support_scores: dict[str, float] = defaultdict(float)
    raw_scores: dict[str, float] = defaultdict(float)
    names: dict[str, str] = {}
    contributions: dict[str, list[dict]] = defaultdict(list)
    matched_anomalies: dict[str, set[str]] = defaultdict(set)
    direct_counts: dict[str, int] = defaultdict(int)

    for edge in edges:
        if edge.anomaly_id not in confidence:
            continue
        if edge.excluded_condition_ids and operating_condition_ids.intersection(edge.excluded_condition_ids):
            continue
        if (
            operating_condition_ids
            and edge.applicable_condition_ids
            and not operating_condition_ids.intersection(edge.applicable_condition_ids)
        ):
            continue
        raw = confidence[edge.anomaly_id] * edge.weight
        role_factor = policy.role_factors.get(edge.edge_role, 0.0)
        level_factor = policy.evidence_level_factors.get(edge.evidence_level, 0.0)
        provisional_factor = policy.provisional_factor if edge.provisional else 1.0
        adjusted = raw * role_factor * level_factor * provisional_factor
        raw_scores[edge.fault_id] += raw
        support_scores[edge.fault_id] += adjusted
        names[edge.fault_id] = edge.fault_name
        matched_anomalies[edge.fault_id].add(edge.anomaly_id)
        if edge.edge_role == "direct":
            direct_counts[edge.fault_id] += 1
        contributions[edge.fault_id].append(
            {
                "anomaly_id": edge.anomaly_id,
                "anomaly_name": edge.anomaly_name,
                "confidence": confidence[edge.anomaly_id],
                "weight": edge.weight,
                "edge_role": edge.edge_role,
                "evidence_level": edge.evidence_level,
                "provisional": edge.provisional,
                "raw_contribution": raw,
                "adjusted_contribution": adjusted,
                "source_ref": edge.source_ref,
                "applicable_condition_ids": edge.applicable_condition_ids,
                "operating_condition_ids": sorted(operating_condition_ids),
            }
        )

    anomaly_count = max(1, len(anomalies))
    candidates: list[CandidateScore] = []
    for fault_id, accumulated_support in support_scores.items():
        coverage = len(matched_anomalies[fault_id]) / anomaly_count
        coverage_factor = policy.coverage_floor + (1.0 - policy.coverage_floor) * coverage
        support_after_coverage = accumulated_support * coverage_factor

        missing_critical: list[dict] = []
        critical_penalty = 0.0
        blocked = False
        if constraints:
            for requirement in constraints.requirements_for(fault_id):
                matches = [
                    anomaly_id
                    for anomaly_id in requirement.any_of_anomaly_ids
                    if confidence.get(anomaly_id, -1.0) >= requirement.min_confidence
                ]
                if len(matches) >= requirement.minimum_matches:
                    continue
                detail = {
                    "requirement_id": requirement.requirement_id,
                    "any_of_anomaly_ids": requirement.any_of_anomaly_ids,
                    "minimum_matches": requirement.minimum_matches,
                    "matched_anomaly_ids": matches,
                    "min_confidence": requirement.min_confidence,
                    "action": requirement.action,
                    "rationale": requirement.rationale,
                    "evidence_level": requirement.evidence_level,
                }
                missing_critical.append(detail)
                if requirement.action == "block_diagnosis":
                    blocked = True
                else:
                    critical_penalty += requirement.penalty

        counter_items: list[dict] = []
        counter_penalty = 0.0
        if constraints:
            for rule in constraints.counter_rules_for(fault_id):
                observed_confidence = confidence.get(rule.anomaly_id)
                if observed_confidence is None or observed_confidence < rule.min_confidence:
                    continue
                penalty = observed_confidence * rule.penalty_weight
                counter_penalty += penalty
                counter_items.append(
                    {
                        "rule_id": rule.rule_id,
                        "anomaly_id": rule.anomaly_id,
                        "confidence": observed_confidence,
                        "penalty_weight": rule.penalty_weight,
                        "penalty": penalty,
                        "rationale": rule.rationale,
                        "evidence_level": rule.evidence_level,
                    }
                )

        temporal_items = list(temporal_evaluations.get(fault_id, []))
        temporal_penalty = sum(float(item.get("penalty", 0.0)) for item in temporal_items if not item.get("satisfied", False))
        temporal_blocked = any(
            (not item.get("satisfied", False)) and item.get("action") == "block_diagnosis"
            for item in temporal_items
        )
        blocked = blocked or temporal_blocked
        final_score = max(0.0, support_after_coverage - critical_penalty - counter_penalty - temporal_penalty)
        candidates.append(
            CandidateScore(
                fault_id=fault_id,
                fault_name=names[fault_id],
                score=final_score,
                raw_score=raw_scores[fault_id],
                evidence_count=len(contributions[fault_id]),
                direct_evidence_count=direct_counts[fault_id],
                evidence_coverage=coverage,
                contributions=sorted(contributions[fault_id], key=lambda item: item["anomaly_id"]),
                support_score=support_after_coverage,
                counter_evidence_penalty=counter_penalty + critical_penalty + temporal_penalty,
                counter_evidence=counter_items,
                critical_evidence_satisfied=not missing_critical,
                missing_critical_evidence=missing_critical,
                eligible_for_diagnosis=not blocked,
                temporal_constraints_satisfied=not any(not item.get("satisfied", False) for item in temporal_items),
                temporal_constraint_penalty=temporal_penalty,
                temporal_constraint_evaluations=temporal_items,
            )
        )

    candidates.sort(key=lambda item: (-item.score, item.fault_id))
    if not candidates:
        return DiagnosisResult(
            status=DiagnosisStatus.UNKNOWN,
            reasons=["Matched edges contributed no valid score."],
            policy_version=policy.policy_version,
            knowledge_version=knowledge_version,
            claim_boundary=policy.claim_boundary,
        )

    eligible = [item for item in candidates if item.eligible_for_diagnosis]
    display_top = candidates[0]
    common_base = dict(
        candidates=candidates,
        policy_version=policy.policy_version,
        knowledge_version=knowledge_version,
        claim_boundary=policy.claim_boundary,
    )
    if not eligible:
        return DiagnosisResult(
            status=DiagnosisStatus.UNKNOWN,
            top_score=display_top.score,
            score_margin=0.0,
            reasons=[
                "All matched candidates are blocked because required critical evidence is missing.",
                *[
                    f"{item.fault_id}: missing {len(item.missing_critical_evidence)} critical evidence requirement(s)."
                    for item in candidates
                    if not item.eligible_for_diagnosis
                ],
            ],
            **common_base,
        )

    top = eligible[0]
    second_score = eligible[1].score if len(eligible) > 1 else 0.0
    margin = top.score - second_score
    common = dict(top_score=top.score, score_margin=margin, **common_base)
    if top.evidence_count < policy.min_evidence_count:
        return DiagnosisResult(
            status=DiagnosisStatus.UNKNOWN,
            reasons=[f"Evidence count {top.evidence_count} is below minimum {policy.min_evidence_count}."],
            **common,
        )
    if top.score < policy.min_score:
        reason = f"Top adjusted score {top.score:.4f} is below minimum {policy.min_score:.4f}."
        if top.counter_evidence_penalty > 0:
            reason += f" Counter-evidence/constraint penalty={top.counter_evidence_penalty:.4f}."
        return DiagnosisResult(status=DiagnosisStatus.UNKNOWN, reasons=[reason], **common)
    if len(eligible) > 1 and margin < policy.min_margin:
        return DiagnosisResult(
            status=DiagnosisStatus.AMBIGUOUS,
            reasons=[f"Top-two eligible adjusted-score margin {margin:.4f} is below minimum {policy.min_margin:.4f}."],
            **common,
        )
    reasons = ["Deterministic score, evidence-count, critical-evidence, and margin requirements passed."]
    if top.counter_evidence:
        reasons.append(f"Counter-evidence penalty {top.counter_evidence_penalty:.4f} was applied before decision.")
    return DiagnosisResult(
        status=DiagnosisStatus.DIAGNOSED,
        conclusion_fault_id=top.fault_id,
        conclusion_fault_name=top.fault_name,
        reasons=reasons,
        **common,
    )
