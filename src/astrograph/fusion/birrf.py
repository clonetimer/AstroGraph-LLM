from __future__ import annotations

from dataclasses import dataclass
import math

from astrograph.evidence.executable import stable_id

from .diagnostics import (
    candidate_overlap,
    evaluate_fusion_diagnostics,
    rank_correlation,
    top_fault_ids,
    unique_top_fault_id,
)
from .models import (
    BiRRFConfig,
    BiRRFFault,
    BiRRFResult,
    FaultRanking,
    KGPartialCoverageAction,
    RankingSource,
    ScoreSemantics,
)


@dataclass(frozen=True)
class BranchWeightResolution:
    configured: dict[str, float]
    effective: dict[str, float]
    identical_kg_rankings: bool
    kg_rank_correlation: float | None
    kg_candidate_overlap: float | None
    collapsed_kg_source: str | None
    collapse_reason: str | None
    kg_fault_universe_coverage: dict[str, float | None]
    # Legacy name retained for compatibility: these are episode ranking candidates,
    # not executable capability. New consumers must use declared_executable_fault_ids.
    kg_declared_fault_ids: dict[str, list[str]]
    kg_declared_executable_fault_ids: dict[str, list[str]]
    kg_coverage_declaration_sha256: dict[str, str | None]
    kg_coverage_source_artifact_ids: dict[str, str | None]
    kg_coverage_source_artifact_sha256: dict[str, str | None]
    kg_unsupported_expected_fault_ids: dict[str, list[str]]
    kg_out_of_universe_declared_fault_ids: dict[str, list[str]]
    kg_active_fault_ids_outside_declaration: dict[str, list[str]]
    kg_active_fault_ids_outside_expected_universe: dict[str, list[str]]
    suppressed_kg_sources: list[str]
    kg_suppression_reasons: dict[str, str]
    attenuated_kg_sources: list[str]
    kg_attenuation_factors: dict[str, float]
    kg_attenuation_reasons: dict[str, str]


@dataclass(frozen=True)
class CandidateContributionResolution:
    raw: dict[str, float]
    adjusted: dict[str, float]
    shared_correlation_groups: dict[str, list[str]]
    dependency_adjusted: bool
    incomplete_lineage_sources: list[str]


def _resolve_candidate_contributions(
    raw: dict[str, float],
    items_by_source: dict[str, object],
    *,
    collapse_shared_groups: bool,
) -> CandidateContributionResolution:
    """Cap a connected set of lineage-dependent branches to one effective vote.

    A branch contribution is indivisible at this layer.  Therefore any declared
    shared correlation group conservatively connects the full branch
    contributions for that fault.  The connected component is scaled so its sum
    equals its strongest original contribution, preventing duplicated upstream
    observations from manufacturing additional support.
    """

    adjusted = dict(raw)
    group_sources: dict[str, list[str]] = {}
    incomplete: list[str] = []
    for source_name, raw_item in items_by_source.items():
        item = raw_item  # narrow, local duck typing keeps the helper independent of Pydantic internals
        if not item.lineage_complete:
            incomplete.append(source_name)
        for group_id in item.correlation_group_ids:
            group_sources.setdefault(group_id, []).append(source_name)
    shared = {
        group_id: sorted(set(source_names))
        for group_id, source_names in group_sources.items()
        if len(set(source_names)) > 1
    }
    if not collapse_shared_groups or not shared:
        return CandidateContributionResolution(
            raw=dict(raw),
            adjusted=adjusted,
            shared_correlation_groups=shared,
            dependency_adjusted=False,
            incomplete_lineage_sources=sorted(incomplete),
        )

    parent = {source_name: source_name for source_name in raw}

    def find(source_name: str) -> str:
        while parent[source_name] != source_name:
            parent[source_name] = parent[parent[source_name]]
            source_name = parent[source_name]
        return source_name

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for sources in shared.values():
        anchor = sources[0]
        for source_name in sources[1:]:
            union(anchor, source_name)

    components: dict[str, list[str]] = {}
    for source_name in raw:
        components.setdefault(find(source_name), []).append(source_name)

    dependency_adjusted = False
    for sources in components.values():
        if len(sources) < 2:
            continue
        total = sum(raw[source_name] for source_name in sources)
        cap = max(raw[source_name] for source_name in sources)
        if total <= cap or total <= 0.0:
            continue
        scale = cap / total
        for source_name in sources:
            adjusted[source_name] = raw[source_name] * scale
        dependency_adjusted = True

    return CandidateContributionResolution(
        raw=dict(raw),
        adjusted=adjusted,
        shared_correlation_groups=shared,
        dependency_adjusted=dependency_adjusted,
        incomplete_lineage_sources=sorted(incomplete),
    )


def resolve_branch_weights(
    forward_ranking: FaultRanking,
    backward_ranking: FaultRanking,
    config: BiRRFConfig,
) -> BranchWeightResolution:
    configured = {
        RankingSource.DATA_DRIVEN.value: config.data_weight,
        RankingSource.KG_FORWARD.value: config.forward_weight,
        RankingSource.KG_BACKWARD.value: config.backward_weight,
    }
    effective = dict(configured)
    expected_faults = set(config.expected_fault_universe)
    kg_rankings = {
        RankingSource.KG_FORWARD.value: forward_ranking,
        RankingSource.KG_BACKWARD.value: backward_ranking,
    }

    ranked_candidates = {
        source_name: sorted({item.fault_id for item in ranking.candidates})
        for source_name, ranking in kg_rankings.items()
    }
    declared_executable: dict[str, list[str]] = {}
    declaration_hashes: dict[str, str | None] = {}
    source_artifact_ids: dict[str, str | None] = {}
    source_artifact_hashes: dict[str, str | None] = {}
    coverage: dict[str, float | None] = {}
    unsupported_expected: dict[str, list[str]] = {}
    out_of_universe_declared: dict[str, list[str]] = {}
    active_outside_declaration: dict[str, list[str]] = {}
    active_outside_expected: dict[str, list[str]] = {}
    suppressed: list[str] = []
    suppression_reasons: dict[str, str] = {}
    attenuated: list[str] = []
    attenuation_factors = {source_name: 1.0 for source_name in kg_rankings}
    attenuation_reasons: dict[str, str] = {}

    def suppress(source_name: str, reason: str) -> None:
        effective[source_name] = 0.0
        attenuation_factors[source_name] = 0.0
        if source_name not in suppressed:
            suppressed.append(source_name)
        suppression_reasons[source_name] = reason

    for source_name, ranking in kg_rankings.items():
        declaration = ranking.kg_fault_coverage
        active_faults = {
            item.fault_id for item in ranking.candidates if item.contributes_to_fusion
        }
        if declaration is None:
            supported_faults: set[str] = set()
            declared_executable[source_name] = []
            declaration_hashes[source_name] = None
            source_artifact_ids[source_name] = None
            source_artifact_hashes[source_name] = None
        else:
            supported_faults = set(declaration.supported_fault_ids)
            declared_executable[source_name] = sorted(supported_faults)
            declaration_hashes[source_name] = declaration.declaration_sha256
            source_artifact_ids[source_name] = declaration.source_artifact_id
            source_artifact_hashes[source_name] = declaration.source_artifact_sha256

        coverage[source_name] = (
            len(supported_faults & expected_faults) / len(expected_faults)
            if expected_faults and declaration is not None
            else None
        )
        unsupported_expected[source_name] = (
            sorted(expected_faults - supported_faults) if expected_faults else []
        )
        out_of_universe_declared[source_name] = (
            sorted(supported_faults - expected_faults) if expected_faults else []
        )
        active_outside_declaration[source_name] = sorted(active_faults - supported_faults)
        active_outside_expected[source_name] = (
            sorted(active_faults - expected_faults) if expected_faults else []
        )

        if not config.enforce_kg_fault_universe_coverage or effective[source_name] <= 0.0:
            continue
        if declaration is None:
            suppress(source_name, "MISSING_EXECUTABLE_COVERAGE_DECLARATION")
            continue
        if active_outside_declaration[source_name]:
            suppress(source_name, "ACTIVE_CANDIDATE_OUTSIDE_EXECUTABLE_COVERAGE_DECLARATION")
            continue
        if active_outside_expected[source_name]:
            suppress(source_name, "ACTIVE_CANDIDATE_OUTSIDE_EXPECTED_FAULT_UNIVERSE")
            continue

        branch_coverage = coverage[source_name]
        if (
            branch_coverage is not None
            and branch_coverage < config.minimum_kg_fault_universe_coverage
        ):
            if config.kg_partial_coverage_action == KGPartialCoverageAction.SUPPRESS_BRANCH:
                suppress(source_name, "EXECUTABLE_FAULT_UNIVERSE_COVERAGE_BELOW_MINIMUM")
            else:
                minimum = config.minimum_kg_fault_universe_coverage
                factor = min(1.0, max(0.0, branch_coverage / minimum)) if minimum > 0.0 else 1.0
                effective[source_name] *= factor
                attenuation_factors[source_name] = factor
                if factor <= 0.0:
                    suppress(source_name, "ZERO_EXECUTABLE_FAULT_UNIVERSE_COVERAGE")
                elif factor < 1.0:
                    attenuated.append(source_name)
                    attenuation_reasons[source_name] = (
                        "SCALED_BY_EXECUTABLE_FAULT_UNIVERSE_COVERAGE"
                    )

    forward_order = {(item.fault_id, item.rank) for item in forward_ranking.candidates}
    backward_order = {(item.fault_id, item.rank) for item in backward_ranking.candidates}
    collapsed_kg_source: str | None = None
    collapse_reason: str | None = None
    identical = bool(forward_order) and forward_order == backward_order
    correlation = rank_correlation(forward_ranking, backward_ranking)
    overlap = candidate_overlap(forward_ranking, backward_ranking)
    highly_correlated = (
        not identical
        and config.collapse_highly_correlated_kg_rankings
        and correlation is not None
        and correlation >= config.kg_redundancy_correlation_threshold
        and overlap is not None
        and overlap >= config.kg_redundancy_min_overlap
    )
    if (
        ((config.collapse_identical_kg_rankings and identical) or highly_correlated)
        and effective[RankingSource.KG_FORWARD.value] > 0.0
        and effective[RankingSource.KG_BACKWARD.value] > 0.0
    ):
        forward_name = RankingSource.KG_FORWARD.value
        backward_name = RankingSource.KG_BACKWARD.value
        if effective[forward_name] >= effective[backward_name]:
            retained = forward_name
            collapsed_kg_source = backward_name
        else:
            retained = backward_name
            collapsed_kg_source = forward_name
        # A redundant observation is one vote, not two independent votes. Retain
        # the stronger post-coverage trust instead of manufacturing extra evidence.
        effective[retained] = max(effective[forward_name], effective[backward_name])
        effective[collapsed_kg_source] = 0.0
        collapse_reason = "IDENTICAL_RANKING" if identical else "HIGH_CORRELATION_AND_OVERLAP"

    return BranchWeightResolution(
        configured=configured,
        effective=effective,
        identical_kg_rankings=identical,
        kg_rank_correlation=correlation,
        kg_candidate_overlap=overlap,
        collapsed_kg_source=collapsed_kg_source,
        collapse_reason=collapse_reason,
        kg_fault_universe_coverage=coverage,
        kg_declared_fault_ids=ranked_candidates,
        kg_declared_executable_fault_ids=declared_executable,
        kg_coverage_declaration_sha256=declaration_hashes,
        kg_coverage_source_artifact_ids=source_artifact_ids,
        kg_coverage_source_artifact_sha256=source_artifact_hashes,
        kg_unsupported_expected_fault_ids=unsupported_expected,
        kg_out_of_universe_declared_fault_ids=out_of_universe_declared,
        kg_active_fault_ids_outside_declaration=active_outside_declaration,
        kg_active_fault_ids_outside_expected_universe=active_outside_expected,
        suppressed_kg_sources=sorted(suppressed),
        kg_suppression_reasons=suppression_reasons,
        attenuated_kg_sources=sorted(attenuated),
        kg_attenuation_factors=attenuation_factors,
        kg_attenuation_reasons=attenuation_reasons,
    )


class BiRRFEngine:
    """Pure deterministic rank-fusion engine.

    It consumes already-built rankings. It does not query KG backends, apply open-set
    policy, or decide whether a result is releasable.
    """

    def fuse(
        self,
        data_ranking: FaultRanking,
        forward_ranking: FaultRanking,
        backward_ranking: FaultRanking,
        *,
        config: BiRRFConfig | None = None,
        normal_probability: float | None = None,
        normal_rank: int | None = None,
    ) -> BiRRFResult:
        config = config or BiRRFConfig()
        rankings = {
            RankingSource.DATA_DRIVEN.value: data_ranking,
            RankingSource.KG_FORWARD.value: forward_ranking,
            RankingSource.KG_BACKWARD.value: backward_ranking,
        }
        weights = resolve_branch_weights(forward_ranking, backward_ranking, config)
        smoothing = {
            RankingSource.DATA_DRIVEN.value: config.data_smoothing,
            RankingSource.KG_FORWARD.value: config.forward_smoothing,
            RankingSource.KG_BACKWARD.value: config.backward_smoothing,
        }
        all_faults = sorted({item.fault_id for ranking in rankings.values() for item in ranking.candidates})
        by_source = {
            name: {item.fault_id: item for item in ranking.candidates}
            for name, ranking in rankings.items()
        }
        fused_rows: list[tuple] = []
        for fault_id in all_faults:
            source_ranks: dict[str, int] = {}
            source_scores: dict[str, float] = {}
            raw_contributions: dict[str, float] = {}
            evidence: list[str] = []
            source_evidence_by_source: dict[str, list[str]] = {}
            correlation_groups_by_source: dict[str, list[str]] = {}
            lineage_complete_by_source: dict[str, bool] = {}
            active_items: dict[str, object] = {}
            for source_name in (
                RankingSource.DATA_DRIVEN.value,
                RankingSource.KG_FORWARD.value,
                RankingSource.KG_BACKWARD.value,
            ):
                item = by_source[source_name].get(fault_id)
                if item is None or weights.effective[source_name] <= 0.0:
                    continue
                source_ranks[source_name] = item.rank
                source_scores[source_name] = item.score
                evidence.extend(item.evidence_fact_ids)
                source_evidence_by_source[source_name] = list(item.source_evidence_ids)
                correlation_groups_by_source[source_name] = list(item.correlation_group_ids)
                lineage_complete_by_source[source_name] = item.lineage_complete
                if not item.contributes_to_fusion:
                    continue
                contribution = weights.effective[source_name] / (smoothing[source_name] + item.rank)
                raw_contributions[source_name] = contribution
                active_items[source_name] = item
            contribution_resolution = _resolve_candidate_contributions(
                raw_contributions,
                active_items,
                collapse_shared_groups=config.collapse_shared_correlation_groups,
            )
            total = sum(contribution_resolution.adjusted.values())
            fused_rows.append(
                (
                    fault_id,
                    total,
                    source_ranks,
                    source_scores,
                    contribution_resolution,
                    evidence,
                    source_evidence_by_source,
                    correlation_groups_by_source,
                    lineage_complete_by_source,
                )
            )

        fused_rows.sort(key=lambda item: (-item[1], item[0]))
        candidates: list[BiRRFFault] = []
        previous_score: float | None = None
        competition_rank = 0
        dependency_adjusted_fault_ids: list[str] = []
        incomplete_lineage_by_fault: dict[str, list[str]] = {}
        for index, (
            fault_id,
            score,
            source_ranks,
            source_scores,
            contribution_resolution,
            evidence,
            source_evidence_by_source,
            correlation_groups_by_source,
            lineage_complete_by_source,
        ) in enumerate(
            fused_rows, start=1
        ):
            if previous_score is None or not math.isclose(
                score, previous_score, rel_tol=1e-12, abs_tol=1e-12
            ):
                competition_rank = index
                previous_score = score
            if contribution_resolution.dependency_adjusted:
                dependency_adjusted_fault_ids.append(fault_id)
            if contribution_resolution.incomplete_lineage_sources:
                incomplete_lineage_by_fault[fault_id] = list(
                    contribution_resolution.incomplete_lineage_sources
                )
            candidates.append(
                BiRRFFault(
                    fault_id=fault_id,
                    final_rank=competition_rank,
                    fused_score=score,
                    source_ranks=source_ranks,
                    source_scores=source_scores,
                    raw_reciprocal_contributions=contribution_resolution.raw,
                    reciprocal_contributions=contribution_resolution.adjusted,
                    supporting_evidence_fact_ids=sorted(set(evidence)),
                    source_evidence_ids_by_source=source_evidence_by_source,
                    correlation_group_ids_by_source=correlation_groups_by_source,
                    lineage_complete_by_source=lineage_complete_by_source,
                    shared_correlation_groups=contribution_resolution.shared_correlation_groups,
                    dependency_adjusted=contribution_resolution.dependency_adjusted,
                )
            )

        diagnostics = evaluate_fusion_diagnostics(
            data_ranking=data_ranking,
            forward_ranking=forward_ranking,
            backward_ranking=backward_ranking,
            candidates=candidates,
            config=config,
            collapsed_kg_source=weights.collapsed_kg_source,
        )
        data_top = unique_top_fault_id(data_ranking)
        forward_top = unique_top_fault_id(forward_ranking)
        backward_top = unique_top_fault_id(backward_ranking)
        fused_top_ids = sorted(item.fault_id for item in candidates if item.final_rank == 1)

        blocking_conflicts = list(diagnostics.blocking_conflicts)
        quality_warnings = list(diagnostics.quality_warnings)
        if dependency_adjusted_fault_ids:
            quality_warnings.append(
                f"EVIDENCE_LINEAGE_DEPENDENCY_ADJUSTED={len(dependency_adjusted_fault_ids)}"
            )
        if incomplete_lineage_by_fault and config.warn_on_incomplete_evidence_lineage:
            quality_warnings.append(
                f"EVIDENCE_LINEAGE_INCOMPLETE={len(incomplete_lineage_by_fault)}"
            )
        if incomplete_lineage_by_fault and config.require_complete_evidence_lineage:
            blocking_conflicts.append("EVIDENCE_LINEAGE_INCOMPLETE")
        for source_name in weights.suppressed_kg_sources:
            branch_coverage = weights.kg_fault_universe_coverage[source_name]
            coverage_text = "UNAVAILABLE" if branch_coverage is None else f"{branch_coverage:.6f}"
            reason = weights.kg_suppression_reasons[source_name]
            if reason == "EXECUTABLE_FAULT_UNIVERSE_COVERAGE_BELOW_MINIMUM":
                quality_warnings.append(
                    "KG_BRANCH_SUPPRESSED_INCOMPLETE_FAULT_COVERAGE="
                    f"{source_name}:{coverage_text}"
                )
            else:
                quality_warnings.append(
                    "KG_BRANCH_SUPPRESSED_INVALID_COVERAGE_CONTRACT="
                    f"{source_name}:{reason}:{coverage_text}"
                )
        for source_name in weights.attenuated_kg_sources:
            branch_coverage = weights.kg_fault_universe_coverage[source_name]
            quality_warnings.append(
                "KG_BRANCH_ATTENUATED_INCOMPLETE_FAULT_COVERAGE="
                f"{source_name}:{branch_coverage:.6f}:"
                f"factor={weights.kg_attenuation_factors[source_name]:.6f}"
            )
        blocking_conflicts = list(dict.fromkeys(blocking_conflicts))
        normal_score_semantics = data_ranking.score_semantics.value
        reported_normal_probability = normal_probability
        if data_ranking.score_semantics == ScoreSemantics.NORMALIZED_RANKING_MASS:
            if normal_probability is not None:
                quality_warnings.append("NORMALIZED_RANKING_MASS_NOT_EXPOSED_AS_NORMAL_PROBABILITY")
            reported_normal_probability = None
        quality_warnings = list(dict.fromkeys(quality_warnings))

        return BiRRFResult(
            fusion_id=stable_id(
                "FUSION",
                {
                    "episode_id": data_ranking.episode_id,
                    "rankings": {name: ranking.ranking_id for name, ranking in rankings.items()},
                    "kg_coverage_declarations": weights.kg_coverage_declaration_sha256,
                    "config": config.model_dump(mode="json"),
                },
            ),
            episode_id=data_ranking.episode_id,
            config=config,
            candidates=candidates,
            source_ranking_ids={name: ranking.ranking_id for name, ranking in rankings.items()},
            top_fault_id=fused_top_ids[0] if len(fused_top_ids) == 1 else None,
            normal_probability=reported_normal_probability,
            normal_rank=normal_rank,
            normal_score_semantics=normal_score_semantics,
            conflict_flags=list(dict.fromkeys([*blocking_conflicts, *quality_warnings])),
            blocking_conflicts=blocking_conflicts,
            quality_warnings=quality_warnings,
            open_set_actions=[],
            metadata={
                "data_top_fault_id": data_top,
                "forward_top_fault_id": forward_top,
                "backward_top_fault_id": backward_top,
                "kg_branch_correlation": diagnostics.kg_branch_correlation,
                "kg_top1_overlap": diagnostics.kg_top1_overlap,
                "kg_candidate_overlap": diagnostics.kg_candidate_overlap,
                "data_conflict_gate_applied": diagnostics.data_conflict_gate_applied,
                "data_score_semantics": diagnostics.data_score_semantics,
                "fused_score_semantics": "RECIPROCAL_RANK_SCORE_NOT_PROBABILITY",
                "normal_score_semantics": normal_score_semantics,
                "legacy_normal_probability_suppressed": (
                    normal_probability is not None and reported_normal_probability is None
                ),
                "kg_rank_orders_identical": weights.identical_kg_rankings,
                "collapsed_kg_source": weights.collapsed_kg_source,
                "kg_collapse_reason": weights.collapse_reason,
                "configured_branch_weights": weights.configured,
                "effective_branch_weights": weights.effective,
                "kg_coverage_semantics_version": config.kg_fault_coverage_semantics_version,
                "kg_fault_universe_coverage": weights.kg_fault_universe_coverage,
                "kg_executable_fault_universe_coverage": weights.kg_fault_universe_coverage,
                "kg_declared_fault_ids": weights.kg_declared_fault_ids,
                "kg_ranked_candidate_fault_ids": weights.kg_declared_fault_ids,
                "kg_declared_executable_fault_ids": weights.kg_declared_executable_fault_ids,
                "kg_coverage_declaration_sha256": weights.kg_coverage_declaration_sha256,
                "kg_coverage_source_artifact_ids": weights.kg_coverage_source_artifact_ids,
                "kg_coverage_source_artifact_sha256": weights.kg_coverage_source_artifact_sha256,
                "kg_unsupported_expected_fault_ids": weights.kg_unsupported_expected_fault_ids,
                "kg_out_of_universe_declared_fault_ids": (
                    weights.kg_out_of_universe_declared_fault_ids
                ),
                "kg_active_fault_ids_outside_declaration": (
                    weights.kg_active_fault_ids_outside_declaration
                ),
                "kg_active_fault_ids_outside_expected_universe": (
                    weights.kg_active_fault_ids_outside_expected_universe
                ),
                "suppressed_kg_sources": weights.suppressed_kg_sources,
                "kg_suppression_reasons": weights.kg_suppression_reasons,
                "attenuated_kg_sources": weights.attenuated_kg_sources,
                "kg_attenuation_factors": weights.kg_attenuation_factors,
                "kg_attenuation_reasons": weights.kg_attenuation_reasons,
                "kg_partial_coverage_action": config.kg_partial_coverage_action.value,
                "expected_fault_universe": list(config.expected_fault_universe),
                "minimum_kg_fault_universe_coverage": (
                    config.minimum_kg_fault_universe_coverage
                ),
                "data_top_fault_ids": top_fault_ids(data_ranking),
                "forward_top_fault_ids": top_fault_ids(forward_ranking),
                "backward_top_fault_ids": top_fault_ids(backward_ranking),
                "fused_top_fault_ids": fused_top_ids,
                "evidence_lineage_dependency_adjusted_fault_ids": dependency_adjusted_fault_ids,
                "incomplete_evidence_lineage_by_fault": incomplete_lineage_by_fault,
                "quality_warnings": quality_warnings,
            },
        )


def fuse_rankings(
    data_ranking: FaultRanking,
    forward_ranking: FaultRanking,
    backward_ranking: FaultRanking,
    *,
    config: BiRRFConfig | None = None,
    normal_probability: float | None = None,
    normal_rank: int | None = None,
) -> BiRRFResult:
    return BiRRFEngine().fuse(
        data_ranking,
        forward_ranking,
        backward_ranking,
        config=config,
        normal_probability=normal_probability,
        normal_rank=normal_rank,
    )
