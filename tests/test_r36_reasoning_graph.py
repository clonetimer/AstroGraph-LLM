"""Reconstructed R36.3 contract tests from the source handoff."""
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from astrograph.diagnosis.reasoning_graph import (
    BayesianDiagnosticReasoner,
    DiagnosticEdgeRelation,
    DiagnosticNodeType,
    DiagnosticReasoningEdge,
    DiagnosticReasoningGraph,
    DiagnosticReasoningNode,
    ReasoningStatus,
)
from astrograph.evidence.quality import EvidenceSourceCategory, QualityFactor, assess_evidence_quality


def complete_quality(evidence_id: str):
    factor = QualityFactor(value=1.0, basis="test")
    return assess_evidence_quality(
        evidence_id=evidence_id,
        source_category=EvidenceSourceCategory.SIMULATION,
        source_reliability=factor,
        measurement_quality=factor,
        temporal_consistency=factor,
        model_agreement=factor,
    )


def test_bayesian_update_uses_declared_lr_and_quality():
    graph = DiagnosticReasoningGraph(
        graph_id="G1",
        nodes=(
            DiagnosticReasoningNode(node_id="F", node_type=DiagnosticNodeType.FAULT, label="fault", prior_probability=0.1),
            DiagnosticReasoningNode(node_id="E", node_type=DiagnosticNodeType.EVIDENCE, label="evidence", evidence_quality=complete_quality("E")),
        ),
        edges=(
            DiagnosticReasoningEdge(
                edge_id="X", source_id="E", target_id="F",
                relation=DiagnosticEdgeRelation.SUPPORTS, confidence=1.0,
                source_ref="test", likelihood_ratio=9.0,
            ),
        ),
    )
    result = BayesianDiagnosticReasoner().infer(
        graph, observed_evidence_ids={"E"}, at_time=datetime.now(timezone.utc)
    )
    assert result.status == ReasoningStatus.COMPLETE
    assert result.posteriors[0].posterior_probability == pytest.approx(0.5)
    assert result.posteriors[0].used_evidence_ids == ("E",)


def test_incomplete_quality_is_excluded_fail_closed():
    partial = assess_evidence_quality(
        evidence_id="E", source_category=EvidenceSourceCategory.MODEL_OUTPUT,
        source_reliability=QualityFactor(value=1.0, basis="test"),
    )
    graph = DiagnosticReasoningGraph(
        graph_id="G2",
        nodes=(
            DiagnosticReasoningNode(node_id="F", node_type=DiagnosticNodeType.FAULT, label="fault", prior_probability=0.2),
            DiagnosticReasoningNode(node_id="E", node_type=DiagnosticNodeType.EVIDENCE, label="evidence", evidence_quality=partial),
        ),
        edges=(
            DiagnosticReasoningEdge(edge_id="X", source_id="E", target_id="F", relation=DiagnosticEdgeRelation.SUPPORTS, confidence=1.0, source_ref="test", likelihood_ratio=2.0),
        ),
    )
    result = BayesianDiagnosticReasoner().infer(
        graph, observed_evidence_ids={"E"}, at_time=datetime.now(timezone.utc)
    )
    assert result.status == ReasoningStatus.INSUFFICIENT_EVIDENCE
    assert result.excluded_evidence == {"E": "EVIDENCE_QUALITY_INCOMPLETE"}
    assert result.posteriors[0].posterior_probability == pytest.approx(0.2)


def test_probability_edges_must_have_semantically_valid_lr():
    with pytest.raises(ValidationError):
        DiagnosticReasoningEdge(
            edge_id="bad", source_id="E", target_id="F",
            relation=DiagnosticEdgeRelation.SUPPORTS, confidence=1.0,
            source_ref="test", likelihood_ratio=0.5,
        )


def test_inference_requires_timezone_aware_time():
    graph = DiagnosticReasoningGraph(
        graph_id="G3",
        nodes=(DiagnosticReasoningNode(node_id="F", node_type=DiagnosticNodeType.FAULT, label="fault", prior_probability=0.2),),
        edges=(),
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        BayesianDiagnosticReasoner().infer(graph, observed_evidence_ids=set(), at_time=datetime.now())


def test_edge_validity_requires_timezone_aware_time():
    with pytest.raises(ValidationError, match="timezone-aware"):
        DiagnosticReasoningEdge(
            edge_id="bad-time", source_id="E", target_id="F",
            relation=DiagnosticEdgeRelation.SUPPORTS, confidence=1.0,
            source_ref="test", likelihood_ratio=2.0,
            valid_from=datetime(2026, 1, 1),
        )


def test_extreme_counterevidence_is_numerically_stable():
    nodes = [
        DiagnosticReasoningNode(
            node_id="F", node_type=DiagnosticNodeType.FAULT,
            label="fault", prior_probability=0.5,
        )
    ]
    edges = []
    for index in range(3):
        evidence_id = f"E{index}"
        nodes.append(DiagnosticReasoningNode(
            node_id=evidence_id, node_type=DiagnosticNodeType.EVIDENCE,
            label="evidence", evidence_quality=complete_quality(evidence_id),
        ))
        edges.append(DiagnosticReasoningEdge(
            edge_id=f"X{index}", source_id=evidence_id, target_id="F",
            relation=DiagnosticEdgeRelation.CONTRADICTS, confidence=1.0,
            source_ref="test", likelihood_ratio=1e-300,
        ))
    result = BayesianDiagnosticReasoner().infer(
        DiagnosticReasoningGraph(graph_id="extreme", nodes=tuple(nodes), edges=tuple(edges)),
        observed_evidence_ids={"E0", "E1", "E2"},
        at_time=datetime.now(timezone.utc),
    )
    assert result.status == ReasoningStatus.COMPLETE
    assert result.posteriors[0].posterior_probability == pytest.approx(1e-15)


def test_target_without_evidence_makes_global_result_partial():
    graph = DiagnosticReasoningGraph(
        graph_id="multi-target",
        nodes=(
            DiagnosticReasoningNode(node_id="F1", node_type=DiagnosticNodeType.FAULT, label="f1", prior_probability=0.5),
            DiagnosticReasoningNode(node_id="F2", node_type=DiagnosticNodeType.FAULT, label="f2", prior_probability=0.5),
            DiagnosticReasoningNode(node_id="E", node_type=DiagnosticNodeType.EVIDENCE, label="e", evidence_quality=complete_quality("E")),
        ),
        edges=(DiagnosticReasoningEdge(
            edge_id="X", source_id="E", target_id="F1",
            relation=DiagnosticEdgeRelation.SUPPORTS, confidence=1.0,
            source_ref="test", likelihood_ratio=2.0,
        ),),
    )
    result = BayesianDiagnosticReasoner().infer(
        graph, observed_evidence_ids={"E"}, at_time=datetime.now(timezone.utc)
    )
    assert result.status == ReasoningStatus.PARTIAL
    assert result.posteriors[1].used_edge_ids == ()
