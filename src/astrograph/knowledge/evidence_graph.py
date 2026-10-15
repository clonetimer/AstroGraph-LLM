from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from astrograph.evidence import EpisodeEvidenceBundle, EvidenceFactStatus
from astrograph.investigation import DecisionRecord, HypothesisSet
from .schema import AssertionStatus, KnowledgeNodeType, KnowledgeRelationType, ReviewStatus


class EvidenceGraphNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1)
    node_type: KnowledgeNodeType
    properties: dict[str, Any] = Field(default_factory=dict)


class EvidenceGraphEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1)
    relation_type: KnowledgeRelationType
    target_id: str = Field(min_length=1)
    assertion_status: AssertionStatus = AssertionStatus.DERIVED
    evidence_source_id: str | None = None
    review_status: ReviewStatus = ReviewStatus.MACHINE_VALIDATED
    properties: dict[str, Any] = Field(default_factory=dict)


class EpisodeEvidenceGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-episode-evidence-graph.v2"
    episode_id: str = Field(min_length=1)
    nodes: list[EvidenceGraphNode]
    edges: list[EvidenceGraphEdge]
    persistent_raw_telemetry: bool = False
    claim_boundary: str = (
        "Graph stores schema-validated evidence summaries and provenance references; relation scores are associative support, not causal effects; raw high-rate telemetry remains outside the graph."
    )


def build_episode_evidence_graph(
    bundle: EpisodeEvidenceBundle,
    hypothesis_set: HypothesisSet,
    decision: DecisionRecord,
) -> EpisodeEvidenceGraph:
    nodes: list[EvidenceGraphNode] = []
    edges: list[EvidenceGraphEdge] = []
    episode = bundle.episode
    nodes.append(
        EvidenceGraphNode(
            node_id=episode.episode_id,
            node_type=KnowledgeNodeType.DIAGNOSIS_EPISODE,
            properties=episode.model_dump(mode="json"),
        )
    )
    component_id = bundle.analysis_component_instance
    if bundle.response_window is not None:
        nodes.append(
            EvidenceGraphNode(
                node_id=bundle.response_window.episode_id,
                node_type=KnowledgeNodeType.RESPONSE_WINDOW,
                properties=bundle.response_window.model_dump(mode="json"),
            )
        )
        edges.append(
            EvidenceGraphEdge(
                source_id=episode.episode_id,
                relation_type=KnowledgeRelationType.HAS_RESPONSE_WINDOW,
                target_id=bundle.response_window.episode_id,
            )
        )
    nodes.append(
        EvidenceGraphNode(
            node_id=component_id,
            node_type=KnowledgeNodeType.COMPONENT_INSTANCE,
            properties={
                "analysis_wheel_index": bundle.analysis_wheel_index,
                "selected_wheel_index": bundle.selected_wheel_index,
                "wheel_selection_status": bundle.wheel_selection_status,
            },
        )
    )
    edges.append(
        EvidenceGraphEdge(
            source_id=episode.episode_id,
            relation_type=KnowledgeRelationType.INVOLVES_COMPONENT,
            target_id=component_id,
        )
    )
    for definition in bundle.feature_definitions:
        nodes.append(
            EvidenceGraphNode(
                node_id=definition.feature_definition_id,
                node_type=KnowledgeNodeType.FEATURE_DEFINITION,
                properties=definition.model_dump(mode="json"),
            )
        )
    for result in bundle.feature_results:
        nodes.append(
            EvidenceGraphNode(
                node_id=result.feature_result_id,
                node_type=KnowledgeNodeType.FEATURE_RESULT,
                properties=result.model_dump(mode="json"),
            )
        )
        edges.extend(
            [
                EvidenceGraphEdge(
                    source_id=episode.episode_id,
                    relation_type=KnowledgeRelationType.HAS_FEATURE_RESULT,
                    target_id=result.feature_result_id,
                ),
                EvidenceGraphEdge(
                    source_id=result.feature_result_id,
                    relation_type=KnowledgeRelationType.INSTANCE_OF,
                    target_id=result.feature_definition_id,
                ),
                EvidenceGraphEdge(
                    source_id=result.feature_result_id,
                    relation_type=KnowledgeRelationType.MEASURED_ON,
                    target_id=component_id,
                ),
            ]
        )
    for definition in bundle.evidence_definitions:
        nodes.append(
            EvidenceGraphNode(
                node_id=definition.evidence_definition_id,
                node_type=KnowledgeNodeType.EVIDENCE_DEFINITION,
                properties=definition.model_dump(mode="json"),
            )
        )
        edges.append(
            EvidenceGraphEdge(
                source_id=definition.evidence_definition_id,
                relation_type=KnowledgeRelationType.USES_FEATURE,
                target_id=definition.feature_definition_id,
            )
        )
    fact_by_id = {item.evidence_fact_id: item for item in bundle.evidence_facts}
    for fact in bundle.evidence_facts:
        nodes.append(
            EvidenceGraphNode(
                node_id=fact.evidence_fact_id,
                node_type=KnowledgeNodeType.EVIDENCE_FACT,
                properties=fact.model_dump(mode="json"),
            )
        )
        edges.extend(
            [
                EvidenceGraphEdge(
                    source_id=episode.episode_id,
                    relation_type=KnowledgeRelationType.HAS_EVIDENCE_FACT,
                    target_id=fact.evidence_fact_id,
                ),
                EvidenceGraphEdge(
                    source_id=fact.evidence_fact_id,
                    relation_type=KnowledgeRelationType.INSTANCE_OF,
                    target_id=fact.evidence_definition_id,
                ),
                EvidenceGraphEdge(
                    source_id=fact.evidence_fact_id,
                    relation_type=KnowledgeRelationType.DERIVED_FROM,
                    target_id=fact.feature_result_id,
                ),
            ]
        )
    for hypothesis in hypothesis_set.hypotheses:
        nodes.append(
            EvidenceGraphNode(
                node_id=hypothesis.hypothesis_id,
                node_type=KnowledgeNodeType.DIAGNOSTIC_HYPOTHESIS,
                properties=hypothesis.model_dump(mode="json"),
            )
        )
        edges.append(
            EvidenceGraphEdge(
                source_id=episode.episode_id,
                relation_type=KnowledgeRelationType.HAS_HYPOTHESIS,
                target_id=hypothesis.hypothesis_id,
            )
        )
        for fact_id in hypothesis.supporting_evidence_fact_ids:
            edges.append(
                EvidenceGraphEdge(
                    source_id=hypothesis.hypothesis_id,
                    relation_type=KnowledgeRelationType.SUPPORTED_BY,
                    target_id=fact_id,
                    properties={"fact_status": fact_by_id[fact_id].status.value if fact_id in fact_by_id else "UNKNOWN"},
                )
            )
        for fact_id in hypothesis.contradicting_evidence_fact_ids:
            edges.append(
                EvidenceGraphEdge(
                    source_id=hypothesis.hypothesis_id,
                    relation_type=KnowledgeRelationType.CONTRADICTED_BY,
                    target_id=fact_id,
                    properties={
                        "fact_status": fact_by_id[fact_id].status.value if fact_id in fact_by_id else EvidenceFactStatus.UNKNOWN.value
                    },
                )
            )
    nodes.append(
        EvidenceGraphNode(
            node_id=decision.decision_id,
            node_type=KnowledgeNodeType.DECISION_RECORD,
            properties=decision.model_dump(mode="json"),
        )
    )
    edges.append(
        EvidenceGraphEdge(
            source_id=decision.decision_id,
            relation_type=KnowledgeRelationType.DECIDES_EPISODE,
            target_id=episode.episode_id,
        )
    )
    if decision.accepted_hypothesis_id:
        edges.append(
            EvidenceGraphEdge(
                source_id=decision.decision_id,
                relation_type=KnowledgeRelationType.ACCEPTS_HYPOTHESIS,
                target_id=decision.accepted_hypothesis_id,
            )
        )
    return EpisodeEvidenceGraph(
        episode_id=episode.episode_id,
        nodes=nodes,
        edges=edges,
        persistent_raw_telemetry=False,
    )
