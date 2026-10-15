from __future__ import annotations

from pathlib import Path

import pytest

from astrograph.anomaly import AnomalyEpisode
from astrograph.evidence import EpisodeEvidenceBundle
from astrograph.fusion import build_forward_graph_ranking
from astrograph.knowledge import (
    SpatiotemporalFaultTemplateCatalog,
    SpatiotemporalTemplateMatch,
    build_component_spatiotemporal_graph,
    match_spatiotemporal_templates,
)


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "configs/knowledge/spatiotemporal_fault_templates.json"


def _episode(anomaly_id: str, start_s: float, end_s: float, *, confidence: float = 0.9) -> AnomalyEpisode:
    suffix = anomaly_id.lower().replace("_", "-")
    return AnomalyEpisode(
        episode_id=f"episode-{suffix}-00000000",
        run_id="run-spatiotemporal-test-0001",
        case_id="case-spatiotemporal-test",
        anomaly_id=anomaly_id,
        name=anomaly_id,
        start_s=start_s,
        end_s=end_s,
        first_observed_at_s=start_s,
        peak_confidence=confidence,
    )


def _bundle(case_id: str) -> EpisodeEvidenceBundle:
    path = ROOT / "tests/fixtures/evidence" / case_id / "diagnosis/episode_evidence_bundle.json"
    return EpisodeEvidenceBundle.model_validate_json(path.read_text(encoding="utf-8"))


def test_catalog_builds_component_episode_graph_and_matches_reviewed_template() -> None:
    catalog = SpatiotemporalFaultTemplateCatalog.from_json(CATALOG_PATH)
    episodes = [
        _episode("ANOM_RW_FRICTION_ESTIMATE_HIGH", 10.0, 20.0),
        _episode("ANOM_RW_CURRENT_DEVIATION", 12.0, 22.0),
    ]
    graph = build_component_spatiotemporal_graph(episodes, catalog)
    matches = match_spatiotemporal_templates(graph, catalog)

    assert len(graph.nodes) == 2
    assert {item.component_instance_id for item in graph.nodes} == {
        "component:adcs:reaction-wheel:analysis"
    }
    assert {item.relation_type.value for item in graph.edges} >= {"OVERLAPS", "SAME_COMPONENT"}
    friction = next(item for item in matches if item.fault_id == "ADCS_RW_FRICTION_INCREASE")
    assert friction.eligible_for_rerank is True
    assert friction.score == pytest.approx(1.0)
    assert friction.matched_edge_ids == ["edge:rw-friction-to-response"]
    assert set(friction.supporting_episode_ids) == {item.episode_id for item in episodes}


def test_missing_response_is_auditable_and_not_eligible_for_rerank() -> None:
    catalog = SpatiotemporalFaultTemplateCatalog.from_json(CATALOG_PATH)
    graph = build_component_spatiotemporal_graph(
        [_episode("ANOM_RW_FRICTION_ESTIMATE_HIGH", 10.0, 20.0)],
        catalog,
    )
    friction = next(
        item for item in match_spatiotemporal_templates(graph, catalog)
        if item.fault_id == "ADCS_RW_FRICTION_INCREASE"
    )
    assert friction.eligible_for_rerank is False
    assert friction.missing_required_roles == ["component_response"]
    assert friction.missing_required_edge_ids == ["edge:rw-friction-to-response"]


def test_unmapped_episode_is_reported_instead_of_guessed() -> None:
    catalog = SpatiotemporalFaultTemplateCatalog.from_json(CATALOG_PATH)
    episode = _episode("ANOM_UNKNOWN_COMPONENT", 1.0, 2.0)
    graph = build_component_spatiotemporal_graph([episode], catalog)
    assert graph.nodes == []
    assert graph.unmapped_episode_ids == [episode.episode_id]


def test_forward_rerank_is_explicit_bounded_and_does_not_add_a_birrf_branch() -> None:
    bundle = _bundle("reaction_wheel_friction_increase")
    baseline = build_forward_graph_ranking(bundle)
    match = SpatiotemporalTemplateMatch(
        template_id="template:test:friction",
        fault_id="ADCS_RW_FRICTION_INCREASE",
        score=1.0,
        eligible_for_rerank=True,
        node_score=1.0,
        edge_score=1.0,
        supporting_episode_ids=["episode-support-00000001"],
        review_status="HUMAN_REVIEWED",
    )
    reranked = build_forward_graph_ranking(
        bundle,
        spatiotemporal_matches=[match],
        spatiotemporal_weight=0.2,
    )
    baseline_score = next(
        item.score for item in baseline.candidates if item.fault_id == "ADCS_RW_FRICTION_INCREASE"
    )
    candidate = next(
        item for item in reranked.candidates if item.fault_id == "ADCS_RW_FRICTION_INCREASE"
    )
    assert candidate.score == pytest.approx(baseline_score + 0.2)
    assert reranked.metadata["spatiotemporal_rerank_enabled"] is True
    assert reranked.metadata["spatiotemporal_weight"] == pytest.approx(0.2)
    assert reranked.source.value == "KG_FORWARD"
    with pytest.raises(ValueError, match="between 0.0 and 0.25"):
        build_forward_graph_ranking(bundle, spatiotemporal_matches=[match], spatiotemporal_weight=0.3)


def test_unreviewed_template_match_cannot_affect_forward_ranking() -> None:
    bundle = _bundle("reaction_wheel_friction_increase")
    baseline = build_forward_graph_ranking(bundle)
    match = SpatiotemporalTemplateMatch(
        template_id="template:test:unreviewed",
        fault_id="ADCS_RW_FRICTION_INCREASE",
        score=1.0,
        eligible_for_rerank=True,
        node_score=1.0,
        edge_score=1.0,
        review_status="MACHINE_VALIDATED",
    )
    reranked = build_forward_graph_ranking(
        bundle,
        spatiotemporal_matches=[match],
        spatiotemporal_weight=0.2,
    )
    assert [item.score for item in reranked.candidates] == [item.score for item in baseline.candidates]
    assert reranked.metadata["ignored_spatiotemporal_matches"] == [
        {"template_id": "template:test:unreviewed", "reason": "NOT_HUMAN_REVIEWED"}
    ]
