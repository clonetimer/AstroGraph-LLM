"""A deterministic projection of native BiRRF and verified graph associations.

No threshold detector and no LLM constructs this summary. Scores remain ranking
scores. A path in a graph is never promoted to a current observation.
"""
from __future__ import annotations
from astrograph.evidence import EpisodeEvidenceBundle, EvidenceFactStatus, EvidenceRole
from astrograph.fusion import CandidateFusionPackage
from astrograph.knowledge.query import GraphQueryExecutionResult
from .contracts import content_hash


def build_diagnostic_summary(
    bundle: EpisodeEvidenceBundle,
    package: CandidateFusionPackage,
    query_results: list[GraphQueryExecutionResult],
    *,
    top_k: int = 5,
    graph_identity: dict | None = None,
    resolve_supported_ties: bool = False,
    resolve_unsupported_top: bool = False,
) -> dict:
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if package.episode_id != bundle.episode.episode_id:
        raise ValueError("summary episode mismatch")
    if any(q.truncated for q in query_results):
        raise ValueError("truncated graph results cannot establish complete evidence")
    facts = {f.evidence_fact_id: f for f in bundle.evidence_facts}
    if len(facts) != len(bundle.evidence_facts):
        raise ValueError("duplicate evidence fact IDs")
    definitions = {d.evidence_definition_id: d for d in bundle.evidence_definitions}
    features = {f.feature_result_id: f for f in bundle.feature_results}
    features_by_definition = {f.feature_definition_id: f for f in bundle.feature_results}
    fused = package.birrf_result
    # Do not arbitrarily break a tie at the top-k boundary.
    ordered = sorted(fused.candidates, key=lambda c: (c.final_rank, c.fault_id))
    cutoff = ordered[min(top_k, len(ordered))-1].final_rank if ordered else 0
    # Coverage-aware resolution must inspect every candidate; truncating before
    # checking eligibility could hide a second supported mechanism.
    selected = [c for c in ordered if resolve_unsupported_top or c.final_rank <= cutoff]
    candidates = []
    for candidate in selected:
        unknown = set(candidate.supporting_evidence_fact_ids) - set(facts)
        if unknown:
            raise ValueError(f"fused candidate references unknown evidence: {sorted(unknown)}")
        related = [f for f in facts.values() if f.target_fault_id == candidate.fault_id]
        supporting = sorted(f.evidence_fact_id for f in related
                            if f.status == EvidenceFactStatus.SATISFIED and
                            f.role in {EvidenceRole.REQUIRED, EvidenceRole.SUPPORTING})
        contradicting = sorted(f.evidence_fact_id for f in related if
                               f.status == EvidenceFactStatus.CONTRADICTED or
                               (f.status == EvidenceFactStatus.SATISFIED and
                                f.role in {EvidenceRole.COUNTER, EvidenceRole.EXCLUSION}))
        # Definition IDs, including absent facts, are deliberately retained.
        required = [d for d in definitions.values() if d.target_fault_id == candidate.fault_id
                    and d.critical and d.role == EvidenceRole.REQUIRED]
        satisfied_defs = {f.evidence_definition_id for f in related if f.status == EvidenceFactStatus.SATISFIED}
        missing = sorted(d.evidence_definition_id for d in required if d.evidence_definition_id not in satisfied_defs)
        paths = []
        for query in query_results:
            for row in query.rows:
                if candidate.fault_id in row.path_entity_ids:
                    paths.append({"query_plan_id": query.query_plan_id,
                                  "entity_ids": row.path_entity_ids,
                                  "relations": row.path_relation_types})
        candidates.append({
            "fault_id": candidate.fault_id, "rank": candidate.final_rank,
            "fusion_score": candidate.fused_score,
            "score_semantics": "RECIPROCAL_RANK_SCORE_NOT_PROBABILITY",
            "source_ranks": candidate.source_ranks,
            "reciprocal_contributions": candidate.reciprocal_contributions,
            "dependency_adjusted": candidate.dependency_adjusted,
            "supporting_evidence_ids": supporting,
            "contradicting_evidence_ids": contradicting,
            "missing_required_definition_ids": missing,
            "graph_paths": paths,
            "eligible_for_explanation": bool(supporting and paths and not contradicting and not missing),
        })
    nominated = next((c for c in candidates if c["fault_id"] == fused.top_fault_id), None)
    tie_resolution = None
    if resolve_supported_ties and fused.top_fault_id is None:
        tied = [c for c in candidates if c['rank'] == 1]
        supported = [c for c in tied if c['eligible_for_explanation']]
        if len(tied) > 1 and len(supported) == 1:
            nominated = supported[0]
            tie_resolution = {'policy': 'UNIQUE_SUPPORTED_TOP',
                              'tied_fault_ids': sorted(c['fault_id'] for c in tied),
                              'selected_fault_id': nominated['fault_id'],
                              'ranking_modified': False}
    if resolve_unsupported_top and (nominated is None or not nominated['eligible_for_explanation']):
        supported = [c for c in candidates if c['eligible_for_explanation']]
        if len(supported) == 1:
            nominated = supported[0]
            tie_resolution = {'policy': 'UNIQUE_SUPPORTED_CANDIDATE',
                              'rejected_top_fault_ids': sorted(c['fault_id'] for c in candidates
                                  if c['rank'] == 1 and not c['eligible_for_explanation']),
                              'selected_fault_id': nominated['fault_id'],
                              'selected_original_rank': nominated['rank'],
                              'ranking_modified': False}
    if fused.blocking_conflicts or fused.open_set_actions:
        status = "BLOCKED"
    elif not candidates or not any(c["eligible_for_explanation"] for c in candidates):
        status = "INSUFFICIENT_EVIDENCE"
    elif fused.top_fault_id is None and tie_resolution is None:
        status = "AMBIGUOUS"
    elif nominated is None or not nominated["eligible_for_explanation"]:
        status = "BLOCKED"
    else:
        status = "READY_FOR_EXPLANATION"
    referenced = sorted({eid for c in candidates for key in
                         ["supporting_evidence_ids", "contradicting_evidence_ids"] for eid in c[key]})
    evidence = []
    for eid in referenced:
        f = facts[eid]
        feature = features.get(f.feature_result_id)
        evidence.append({"evidence_id": eid, "status": f.status.value, "role": f.role.value,
                         "observed_value": f.observed_value, "threshold_value": f.threshold_value,
                         "comparison": f.comparison.value, "unit": f.unit,
                         "feature_field": feature.resolved_field if feature else None})
        conjuncts = definitions[f.evidence_definition_id].and_feature_conditions
        if conjuncts:
            evidence[-1]['conjunctive_observations'] = [{
                'feature_definition_id': c.feature_definition_id,
                'observed_value': features_by_definition[c.feature_definition_id].value
                    if c.feature_definition_id in features_by_definition else None,
                'unit': c.threshold_unit, 'comparison': c.comparison.value,
                'threshold': c.threshold,
            } for c in conjuncts]
    summary = {
        "schema_version": "astrograph-birrf-diagnostic-summary.v1",
        "generation": "NATIVE_GRAPH_QUERY_PLUS_BIRRF_PROJECTION",
        "public_episode_id": "EP-" + content_hash(package.episode_id)[:20],
        "window": {"start_s": bundle.episode.start_s, "end_s": bundle.episode.end_s},
        "component": bundle.analysis_component_instance,
        "status": status,
        "suspected_fault_id": nominated['fault_id'] if status == "READY_FOR_EXPLANATION" else None,
        "calibrated_probability": None,
        "candidates": candidates, "evidence": evidence,
        "blocking_conflicts": list(fused.blocking_conflicts),
        "open_set_actions": list(fused.open_set_actions),
        "quality_warnings": list(fused.quality_warnings),
        "graph_identity": graph_identity or {},
        "fusion_id": fused.fusion_id,
        "claim_boundary": (
            "Candidate explanation input, not a flight diagnosis or control authorization. "
            "BiRRF scores are not calibrated probabilities. Missing/contradictory evidence is retained."
        ),
    }
    if resolve_supported_ties or resolve_unsupported_top:
        summary['tie_resolution'] = tie_resolution
    summary["summary_sha256"] = content_hash(summary)
    return summary
