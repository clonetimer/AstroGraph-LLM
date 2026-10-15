from __future__ import annotations

from collections import defaultdict
import math
from typing import Iterable

from astrograph.evidence import EpisodeEvidenceBundle, EvidenceFactStatus, EvidenceRole
from astrograph.evidence.executable import stable_id
from astrograph.lineage import observation_correlation_group_id
from astrograph.knowledge.spatiotemporal_fault_graph import SpatiotemporalTemplateMatch

from .coverage import build_kg_fault_coverage_declaration
from .models import (
    FaultRanking,
    KGFaultCoverageDeclaration,
    RankedFault,
    RankingSource,
    ScoreSemantics,
)

_EVIDENCE_LEVEL_WEIGHT = {"E0": 0.70, "E1": 0.80, "E2": 0.90, "E3": 1.00, "E4": 1.00}
_POSITIVE_ROLE_WEIGHT = {EvidenceRole.REQUIRED: 1.0, EvidenceRole.SUPPORTING: 0.6}
_COUNTER_ROLE_WEIGHT = {EvidenceRole.COUNTER: 0.8, EvidenceRole.EXCLUSION: 1.0}

KG_RANKING_PRODUCER_VERSION = "0.7.1.post24+b006.wip.kgcoverage.v1"


def _fact_lineage(bundle: EpisodeEvidenceBundle, fact) -> tuple[list[str], list[str], bool]:
    """Resolve one evidence fact back to the physical observation it consumed."""

    feature_results = {item.feature_result_id: item for item in bundle.feature_results}
    feature_definitions = {item.feature_definition_id: item for item in bundle.feature_definitions}
    result = feature_results.get(fact.feature_result_id)
    if result is None:
        return [f"EVIDENCE_FACT:{fact.evidence_fact_id}"], [], False
    definition = feature_definitions.get(result.feature_definition_id)
    semantic_ids = (
        list(definition.required_semantics)
        if definition is not None and definition.required_semantics
        else [result.resolved_field]
    )
    source_ids = [
        f"EVIDENCE_FACT:{fact.evidence_fact_id}",
        f"FEATURE_RESULT:{result.feature_result_id}",
        f"INPUT_DATA_SHA256:{result.input_data_sha256}",
    ]
    groups = [
        observation_correlation_group_id(
            episode_id=result.episode_id,
            window_start_s=result.window_start_s,
            window_end_s=result.window_end_s,
            source_field=result.resolved_field,
            semantic_signal_id=semantic_id,
        )
        for semantic_id in semantic_ids
    ]
    return source_ids, groups, True


def _ranked(
    *,
    episode_id: str,
    source: RankingSource,
    score_semantics: ScoreSemantics,
    scores: dict[str, float],
    evidence: dict[str, list[str]],
    source_evidence: dict[str, list[str]] | None = None,
    correlation_groups: dict[str, list[str]] | None = None,
    lineage_complete_by_fault: dict[str, bool] | None = None,
    producer_id: str,
    producer_version: str,
    metadata_by_fault: dict[str, dict] | None = None,
    ranking_metadata: dict | None = None,
    kg_fault_coverage: KGFaultCoverageDeclaration,
) -> FaultRanking:
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    candidates: list[RankedFault] = []
    previous_score: float | None = None
    competition_rank = 0
    for index, (fault_id, score) in enumerate(ordered, start=1):
        numeric_score = float(score)
        if previous_score is None or not math.isclose(
            numeric_score, previous_score, rel_tol=1e-12, abs_tol=1e-12
        ):
            competition_rank = index
            previous_score = numeric_score
        candidates.append(
            RankedFault(
                fault_id=fault_id,
                rank=competition_rank,
                score=numeric_score,
                source=source,
                evidence_fact_ids=sorted(set(evidence.get(fault_id, []))),
                source_evidence_ids=sorted(set((source_evidence or {}).get(fault_id, []))),
                correlation_group_ids=sorted(set((correlation_groups or {}).get(fault_id, []))),
                lineage_complete=bool((lineage_complete_by_fault or {}).get(fault_id, False)),
                contributes_to_fusion=(
                    numeric_score > 0.0
                    if score_semantics in {
                        ScoreSemantics.EVIDENCE_STRENGTH,
                        ScoreSemantics.EVIDENCE_COVERAGE,
                    }
                    else True
                ),
                metadata=(metadata_by_fault or {}).get(fault_id, {}),
            )
        )
    return FaultRanking(
        ranking_id=stable_id(
            "RANK",
            {
                "episode_id": episode_id,
                "source": source.value,
                "faults": [(item.fault_id, round(item.score, 12)) for item in candidates],
                "kg_fault_coverage_sha256": kg_fault_coverage.declaration_sha256,
            },
        ),
        episode_id=episode_id,
        source=source,
        score_semantics=score_semantics,
        candidates=candidates,
        producer_id=producer_id,
        producer_version=producer_version,
        kg_fault_coverage=kg_fault_coverage,
        metadata=ranking_metadata or {},
    )


def build_forward_graph_ranking(
    bundle: EpisodeEvidenceBundle,
    *,
    spatiotemporal_matches: Iterable[SpatiotemporalTemplateMatch] | None = None,
    spatiotemporal_weight: float = 0.0,
) -> FaultRanking:
    """Rank registered faults from activated graph-associated evidence."""

    if not 0.0 <= spatiotemporal_weight <= 0.25:
        raise ValueError("spatiotemporal_weight must be between 0.0 and 0.25")
    definitions = {item.evidence_definition_id: item for item in bundle.evidence_definitions}
    positive: dict[str, float] = defaultdict(float)
    negative: dict[str, float] = defaultdict(float)
    evidence: dict[str, list[str]] = defaultdict(list)
    source_evidence: dict[str, list[str]] = defaultdict(list)
    correlation_groups: dict[str, list[str]] = defaultdict(list)
    metadata: dict[str, dict] = defaultdict(lambda: {"positive_weight": 0.0, "counter_weight": 0.0})
    registered = sorted({item.target_fault_id for item in bundle.evidence_definitions})

    for fact in bundle.evidence_facts:
        definition = definitions.get(fact.evidence_definition_id)
        if definition is None:
            continue
        level = _EVIDENCE_LEVEL_WEIGHT.get(definition.evidence_level, 0.7)
        critical = 1.25 if definition.critical else 1.0
        if definition.role in _POSITIVE_ROLE_WEIGHT and fact.status == EvidenceFactStatus.SATISFIED:
            contribution = _POSITIVE_ROLE_WEIGHT[definition.role] * level * critical
            positive[fact.target_fault_id] += contribution
            metadata[fact.target_fault_id]["positive_weight"] += contribution
            evidence[fact.target_fault_id].append(fact.evidence_fact_id)
            fact_source_ids, fact_groups, _ = _fact_lineage(bundle, fact)
            source_evidence[fact.target_fault_id].extend(fact_source_ids)
            correlation_groups[fact.target_fault_id].extend(fact_groups)
        elif definition.role in _COUNTER_ROLE_WEIGHT and fact.status == EvidenceFactStatus.CONTRADICTED:
            contribution = _COUNTER_ROLE_WEIGHT[definition.role] * level * critical
            negative[fact.target_fault_id] += contribution
            metadata[fact.target_fault_id]["counter_weight"] += contribution
            evidence[fact.target_fault_id].append(fact.evidence_fact_id)
            fact_source_ids, fact_groups, _ = _fact_lineage(bundle, fact)
            source_evidence[fact.target_fault_id].extend(fact_source_ids)
            correlation_groups[fact.target_fault_id].extend(fact_groups)

    scores = {fault_id: max(0.0, positive[fault_id] - negative[fault_id]) for fault_id in registered}
    accepted_matches: dict[str, SpatiotemporalTemplateMatch] = {}
    ignored_match_reasons: list[dict[str, str]] = []
    for raw_match in spatiotemporal_matches or []:
        match = (
            raw_match
            if isinstance(raw_match, SpatiotemporalTemplateMatch)
            else SpatiotemporalTemplateMatch.model_validate(raw_match)
        )
        if match.fault_id not in scores:
            ignored_match_reasons.append({"template_id": match.template_id, "reason": "UNREGISTERED_FAULT"})
            continue
        if not match.eligible_for_rerank:
            ignored_match_reasons.append({"template_id": match.template_id, "reason": "INELIGIBLE_MATCH"})
            continue
        if match.review_status != "HUMAN_REVIEWED":
            ignored_match_reasons.append({"template_id": match.template_id, "reason": "NOT_HUMAN_REVIEWED"})
            continue
        previous = accepted_matches.get(match.fault_id)
        if previous is None or (match.score, match.template_id) > (previous.score, previous.template_id):
            accepted_matches[match.fault_id] = match
    if spatiotemporal_weight > 0.0:
        for fault_id, match in accepted_matches.items():
            contribution = spatiotemporal_weight * match.score
            scores[fault_id] += contribution
            metadata[fault_id]["spatiotemporal_template_id"] = match.template_id
            metadata[fault_id]["spatiotemporal_match_score"] = match.score
            metadata[fault_id]["spatiotemporal_rerank_contribution"] = contribution
            metadata[fault_id]["spatiotemporal_supporting_episode_ids"] = match.supporting_episode_ids
            source_evidence[fault_id].append(f"TEMPLATE:{match.template_id}")
            correlation_groups[fault_id].append(
                f"SPATIOTEMPORAL_TEMPLATE:{match.template_id}"
            )
    return _ranked(
        episode_id=bundle.episode.episode_id,
        source=RankingSource.KG_FORWARD,
        score_semantics=ScoreSemantics.EVIDENCE_STRENGTH,
        scores=scores,
        evidence=evidence,
        source_evidence=source_evidence,
        correlation_groups=correlation_groups,
        lineage_complete_by_fault={
            fault_id: (scores.get(fault_id, 0.0) <= 0.0 or bool(source_evidence.get(fault_id)))
            for fault_id in registered
        },
        producer_id="evidence-definition-forward-ranking",
        producer_version=KG_RANKING_PRODUCER_VERSION,
        metadata_by_fault=dict(metadata),
        ranking_metadata={
            "semantics": "activated evidence keys -> registered fault associations",
            "neo4j_required": False,
            "spatiotemporal_rerank_enabled": spatiotemporal_weight > 0.0,
            "spatiotemporal_weight": spatiotemporal_weight,
            "accepted_spatiotemporal_template_ids": sorted(
                match.template_id for match in accepted_matches.values()
            ),
            "ignored_spatiotemporal_matches": ignored_match_reasons,
            "lineage_schema_version": "astrograph-observation-correlation.v1",
            "spatiotemporal_claim_boundary": (
                "Reviewed approximate episode-template matches may only re-rank KG_FORWARD; "
                "they do not add an independent BiRRF vote or establish causal propagation."
            ),
        },
        kg_fault_coverage=build_kg_fault_coverage_declaration(bundle),
    )


def build_backward_evidence_ranking(
    bundle: EpisodeEvidenceBundle,
    *,
    candidate_fault_ids: Iterable[str] | None = None,
) -> FaultRanking:
    """Rank candidates by weighted coverage of graph-expected evidence."""

    wanted = set(candidate_fault_ids or (item.target_fault_id for item in bundle.evidence_definitions))
    definitions_by_fault: dict[str, list] = defaultdict(list)
    facts = {item.evidence_definition_id: item for item in bundle.evidence_facts}
    for definition in bundle.evidence_definitions:
        if definition.target_fault_id in wanted:
            definitions_by_fault[definition.target_fault_id].append(definition)

    scores: dict[str, float] = {}
    evidence: dict[str, list[str]] = defaultdict(list)
    source_evidence: dict[str, list[str]] = defaultdict(list)
    correlation_groups: dict[str, list[str]] = defaultdict(list)
    metadata: dict[str, dict] = {}
    for fault_id in sorted(wanted):
        expected_weight = 0.0
        matched_weight = 0.0
        counter_weight = 0.0
        critical_missing = 0
        observed_expected = 0
        for definition in definitions_by_fault.get(fault_id, []):
            level = _EVIDENCE_LEVEL_WEIGHT.get(definition.evidence_level, 0.7)
            critical = 1.25 if definition.critical else 1.0
            fact = facts.get(definition.evidence_definition_id)
            if definition.role in _POSITIVE_ROLE_WEIGHT:
                weight = _POSITIVE_ROLE_WEIGHT[definition.role] * level * critical
                expected_weight += weight
                if fact is None or fact.status in {
                    EvidenceFactStatus.MISSING_DATA,
                    EvidenceFactStatus.INVALID_DATA,
                    EvidenceFactStatus.UNKNOWN,
                }:
                    if definition.critical:
                        critical_missing += 1
                    continue
                observed_expected += 1
                evidence[fault_id].append(fact.evidence_fact_id)
                fact_source_ids, fact_groups, _ = _fact_lineage(bundle, fact)
                source_evidence[fault_id].extend(fact_source_ids)
                correlation_groups[fault_id].extend(fact_groups)
                if fact.status == EvidenceFactStatus.SATISFIED:
                    matched_weight += weight
            elif definition.role in _COUNTER_ROLE_WEIGHT:
                if fact is not None and fact.status == EvidenceFactStatus.CONTRADICTED:
                    weight = _COUNTER_ROLE_WEIGHT[definition.role] * level * critical
                    counter_weight += weight
                    evidence[fault_id].append(fact.evidence_fact_id)
                    fact_source_ids, fact_groups, _ = _fact_lineage(bundle, fact)
                    source_evidence[fault_id].extend(fact_source_ids)
                    correlation_groups[fault_id].extend(fact_groups)

        coverage = matched_weight / expected_weight if expected_weight > 0.0 else 0.0
        critical_positive_count = sum(
            1
            for item in definitions_by_fault.get(fault_id, [])
            if item.critical and item.role in _POSITIVE_ROLE_WEIGHT
        )
        completeness = max(
            0.0,
            1.0 - (critical_missing / max(1, critical_positive_count)),
        )
        consistency = max(
            0.0,
            1.0 - counter_weight / expected_weight,
        ) if expected_weight > 0.0 else 0.0
        score = max(0.0, min(1.0, coverage * completeness * consistency))
        scores[fault_id] = score
        metadata[fault_id] = {
            "expected_positive_weight": expected_weight,
            "matched_positive_weight": matched_weight,
            "coverage": coverage,
            "completeness": completeness,
            "consistency": consistency,
            "critical_missing_count": critical_missing,
            "critical_positive_definition_count": critical_positive_count,
            "observed_expected_count": observed_expected,
            "activated_counter_weight": counter_weight,
            "backward_score": score,
        }

    return _ranked(
        episode_id=bundle.episode.episode_id,
        source=RankingSource.KG_BACKWARD,
        score_semantics=ScoreSemantics.EVIDENCE_COVERAGE,
        scores=scores,
        evidence=evidence,
        source_evidence=source_evidence,
        correlation_groups=correlation_groups,
        lineage_complete_by_fault={
            fault_id: (scores.get(fault_id, 0.0) <= 0.0 or bool(source_evidence.get(fault_id)))
            for fault_id in wanted
        },
        producer_id="fault-expected-evidence-backward-ranking",
        producer_version=KG_RANKING_PRODUCER_VERSION,
        metadata_by_fault=metadata,
        ranking_metadata={
            "semantics": "candidate fault -> expected evidence coverage",
            "lineage_schema_version": "astrograph-observation-correlation.v1",
        },
        kg_fault_coverage=build_kg_fault_coverage_declaration(bundle),
    )
