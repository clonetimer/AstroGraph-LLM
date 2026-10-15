from __future__ import annotations

from typing import Any, Iterable

from astrograph.evidence import EpisodeEvidenceBundle
from astrograph.knowledge.query import GraphEvidenceAssociation
from astrograph.knowledge.spatiotemporal_fault_graph import SpatiotemporalTemplateMatch

from .associations import (
    build_backward_evidence_ranking_from_associations,
    build_forward_graph_ranking_from_associations,
)
from .birrf import BiRRFEngine, fuse_rankings
from .models import BiRRFConfig, CandidateFusionPackage, FaultRanking, RankingSource
from .open_set import OpenSetFusionPolicy
from .ranking import build_backward_evidence_ranking, build_forward_graph_ranking


def _validate_data_ranking(
    bundle: EpisodeEvidenceBundle,
    data_ranking: FaultRanking,
    registered_fault_ids: Iterable[str] | None = None,
) -> list[str]:
    if data_ranking.source != RankingSource.DATA_DRIVEN:
        raise ValueError("data_ranking must use DATA_DRIVEN source")
    if data_ranking.episode_id != bundle.episode.episode_id:
        raise ValueError("data ranking and evidence bundle episode IDs differ")
    evidence_faults = {item.target_fault_id for item in bundle.evidence_definitions}
    registered = sorted(set(registered_fault_ids) if registered_fault_ids is not None else evidence_faults)
    if not evidence_faults.issubset(registered):
        raise ValueError("registered fault universe omits configured evidence faults")
    unknown = {item.fault_id for item in data_ranking.candidates} - set(registered)
    if unknown:
        raise ValueError(f"data-driven ranking contains unregistered faults: {sorted(unknown)}")
    return registered


class RuntimeFusionService:
    """Runtime composition boundary for candidate fusion.

    Ranking construction, graph-association validation, BiRRF math, diagnostics, and
    open-set policy live in separate components. This service only composes them into
    the stable ``CandidateFusionPackage`` contract.
    """

    def __init__(self, *, engine: BiRRFEngine | None = None) -> None:
        self.engine = engine or BiRRFEngine()
        self.open_set_policy = OpenSetFusionPolicy()

    def build(
        self,
        *,
        bundle: EpisodeEvidenceBundle,
        data_ranking: FaultRanking,
        config: BiRRFConfig | None = None,
        normal_probability: float | None = None,
        normal_rank: int | None = None,
        spatiotemporal_matches: Iterable[SpatiotemporalTemplateMatch] | None = None,
        spatiotemporal_weight: float = 0.0,
        forward_associations: Iterable[GraphEvidenceAssociation] | None = None,
        backward_associations: Iterable[GraphEvidenceAssociation] | None = None,
        open_set_assessment: Any | None = None,
        runtime_metadata: dict[str, Any] | None = None,
        fused_metadata: dict[str, Any] | None = None,
        quality_warnings: Iterable[str] | None = None,
        registered_fault_ids: Iterable[str] | None = None,
    ) -> CandidateFusionPackage:
        registered = _validate_data_ranking(bundle, data_ranking, registered_fault_ids)
        effective_config = self.open_set_policy.effective_config(config or BiRRFConfig(), open_set_assessment)
        graph_query_backed = forward_associations is not None or backward_associations is not None
        if graph_query_backed:
            if forward_associations is None or backward_associations is None:
                raise ValueError("forward and backward graph associations must be provided together")
            forward = build_forward_graph_ranking_from_associations(
                bundle,
                forward_associations,
                spatiotemporal_matches=spatiotemporal_matches,
                spatiotemporal_weight=spatiotemporal_weight,
            )
            backward = build_backward_evidence_ranking_from_associations(
                bundle,
                backward_associations,
                candidate_fault_ids=registered,
            )
        else:
            forward = build_forward_graph_ranking(
                bundle,
                spatiotemporal_matches=spatiotemporal_matches,
                spatiotemporal_weight=spatiotemporal_weight,
            )
            backward = build_backward_evidence_ranking(bundle, candidate_fault_ids=registered)
        fused = self.engine.fuse(
            data_ranking,
            forward,
            backward,
            config=effective_config,
            normal_probability=normal_probability,
            normal_rank=normal_rank,
        )
        package_metadata: dict[str, Any] = {
            "fusion_controls_llm_candidate_order_only": True,
            "formal_decision_remains_deterministic": True,
            "spatiotemporal_rerank_enabled": spatiotemporal_weight > 0.0,
            "spatiotemporal_branch_count": 0,
        }
        if graph_query_backed:
            package_metadata.update({"graph_query_backed": True, "graph_query_mode": "PARAMETERIZED_READ_ONLY"})
        package_metadata.update(runtime_metadata or {})
        if fused_metadata or quality_warnings:
            merged_fused_metadata = {**fused.metadata, **(fused_metadata or {})}
            merged_warnings = list(dict.fromkeys([*fused.quality_warnings, *(quality_warnings or [])]))
            fused = fused.model_copy(update={"metadata": merged_fused_metadata, "quality_warnings": merged_warnings})
        package = CandidateFusionPackage(
            episode_id=bundle.episode.episode_id,
            data_ranking=data_ranking,
            forward_ranking=forward,
            backward_ranking=backward,
            birrf_result=fused,
            registered_fault_ids=registered,
            metadata=package_metadata,
        )
        return self.open_set_policy.apply_assessment(package, open_set_assessment)


def build_candidate_fusion_package(
    *,
    bundle: EpisodeEvidenceBundle,
    data_ranking: FaultRanking,
    config: BiRRFConfig | None = None,
    normal_probability: float | None = None,
    normal_rank: int | None = None,
    spatiotemporal_matches: Iterable[SpatiotemporalTemplateMatch] | None = None,
    spatiotemporal_weight: float = 0.0,
) -> CandidateFusionPackage:
    return RuntimeFusionService().build(
        bundle=bundle,
        data_ranking=data_ranking,
        config=config,
        normal_probability=normal_probability,
        normal_rank=normal_rank,
        spatiotemporal_matches=spatiotemporal_matches,
        spatiotemporal_weight=spatiotemporal_weight,
    )


def build_candidate_fusion_package_from_graph_associations(
    *,
    bundle: EpisodeEvidenceBundle,
    data_ranking: FaultRanking,
    forward_associations: Iterable[GraphEvidenceAssociation],
    backward_associations: Iterable[GraphEvidenceAssociation],
    config: BiRRFConfig | None = None,
    normal_probability: float | None = None,
    normal_rank: int | None = None,
    spatiotemporal_matches: Iterable[SpatiotemporalTemplateMatch] | None = None,
    spatiotemporal_weight: float = 0.0,
) -> CandidateFusionPackage:
    return RuntimeFusionService().build(
        bundle=bundle,
        data_ranking=data_ranking,
        forward_associations=forward_associations,
        backward_associations=backward_associations,
        config=config,
        normal_probability=normal_probability,
        normal_rank=normal_rank,
        spatiotemporal_matches=spatiotemporal_matches,
        spatiotemporal_weight=spatiotemporal_weight,
    )


__all__ = [
    "RuntimeFusionService",
    "build_backward_evidence_ranking",
    "build_backward_evidence_ranking_from_associations",
    "build_candidate_fusion_package",
    "build_candidate_fusion_package_from_graph_associations",
    "build_forward_graph_ranking",
    "build_forward_graph_ranking_from_associations",
    "fuse_rankings",
]
