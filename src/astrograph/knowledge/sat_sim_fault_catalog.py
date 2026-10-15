from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .evidence_graph import EvidenceGraphEdge, EvidenceGraphNode
from .schema import AssertionStatus, KnowledgeNodeType, KnowledgeRelationType, ReviewStatus


class ValidatedPropagationPath(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path_id: str = Field(min_length=1)
    target_subsystem: str = Field(min_length=1)
    telemetry_fields: list[str] = Field(min_length=1)
    interpretation: str = Field(min_length=1)
    evidence_source_id: str = Field(min_length=1)
    assertion_status: AssertionStatus = AssertionStatus.DERIVED
    review_status: ReviewStatus = ReviewStatus.MACHINE_VALIDATED
    applicable_conditions: list[str] = Field(default_factory=list)
    claim_boundary: str = "Observed associative whole-spacecraft coupling; not a causal-effect estimate."


class FaultCatalogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    effect_id: str = Field(min_length=1)
    subsystem: str = Field(min_length=1)
    component: str = Field(min_length=1)
    mechanism: str = Field(min_length=1)
    direct_symptoms: list[str] = Field(min_length=1)
    secondary_symptoms: list[str] = Field(default_factory=list)
    telemetry_fields: list[str] = Field(min_length=1)
    propagation_targets: list[str] = Field(default_factory=list)
    evidence_source_id: str = Field(min_length=1)
    assertion_status: AssertionStatus = AssertionStatus.DERIVED
    review_status: ReviewStatus = ReviewStatus.MACHINE_VALIDATED
    applicable_conditions: list[str] = Field(default_factory=list)
    validated_propagation_paths: list[ValidatedPropagationPath] = Field(default_factory=list)


class SatSimFaultCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "astrograph-sat-sim-fault-catalog.v1"
    entries: list[FaultCatalogEntry]
    claim_boundary: str = (
        "Laboratory Basilisk effect catalog with schema-validated evidence paths. Entries describe "
        "mechanism-consistent associations and propagation hypotheses, not estimated causal effects."
    )

    @model_validator(mode="after")
    def unique_effect_ids(self) -> "SatSimFaultCatalog":
        ids = [entry.effect_id for entry in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("effect_id values must be unique")
        return self

    def by_effect_id(self) -> dict[str, FaultCatalogEntry]:
        return {entry.effect_id: entry for entry in self.entries}


class FaultCatalogGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "astrograph-sat-sim-fault-catalog-graph.v1"
    nodes: list[EvidenceGraphNode]
    edges: list[EvidenceGraphEdge]
    effect_coverage: list[str]
    claim_boundary: str = "Evidence-path graph; relation edges are not causal effect estimates."


def load_sat_sim_fault_catalog(path: Path) -> SatSimFaultCatalog:
    return SatSimFaultCatalog.model_validate(json.loads(path.read_text(encoding="utf-8")))


def build_fault_catalog_graph(catalog: SatSimFaultCatalog) -> FaultCatalogGraph:
    nodes: dict[str, EvidenceGraphNode] = {}
    edges: list[EvidenceGraphEdge] = []
    for entry in catalog.entries:
        subsystem_id = f"subsystem:{entry.subsystem}"
        component_id = f"component_type:{entry.subsystem}:{entry.component}"
        fault_id = f"fault_mode:{entry.effect_id}"
        nodes.setdefault(subsystem_id, EvidenceGraphNode(node_id=subsystem_id, node_type=KnowledgeNodeType.SUBSYSTEM,
                                                         properties={"name": entry.subsystem}))
        nodes.setdefault(component_id, EvidenceGraphNode(node_id=component_id, node_type=KnowledgeNodeType.COMPONENT_TYPE,
                                                         properties={"name": entry.component}))
        nodes[fault_id] = EvidenceGraphNode(node_id=fault_id, node_type=KnowledgeNodeType.FAULT_MODE,
                                           properties={"effect_id": entry.effect_id, "mechanism": entry.mechanism,
                                                       "applicable_conditions": entry.applicable_conditions})
        common = dict(assertion_status=entry.assertion_status, evidence_source_id=entry.evidence_source_id,
                      review_status=entry.review_status)
        edges.append(EvidenceGraphEdge(source_id=component_id, relation_type=KnowledgeRelationType.PART_OF,
                                       target_id=subsystem_id, **common))
        edges.append(EvidenceGraphEdge(source_id=fault_id, relation_type=KnowledgeRelationType.AFFECTS,
                                       target_id=component_id, **common))
        for index, symptom in enumerate(entry.direct_symptoms):
            anomaly_id=f"anomaly:{entry.effect_id}:direct:{index}"
            nodes[anomaly_id]=EvidenceGraphNode(node_id=anomaly_id,node_type=KnowledgeNodeType.ANOMALY,
                                                properties={"description":symptom,"symptom_level":"direct"})
            edges.append(EvidenceGraphEdge(source_id=fault_id,relation_type=KnowledgeRelationType.OBSERVED_AS,
                                           target_id=anomaly_id,**common))
        for index, symptom in enumerate(entry.secondary_symptoms):
            anomaly_id=f"anomaly:{entry.effect_id}:secondary:{index}"
            nodes[anomaly_id]=EvidenceGraphNode(node_id=anomaly_id,node_type=KnowledgeNodeType.ANOMALY,
                                                properties={"description":symptom,"symptom_level":"secondary"})
            edges.append(EvidenceGraphEdge(source_id=fault_id,relation_type=KnowledgeRelationType.PROPAGATES_TO,
                                           target_id=anomaly_id,assertion_status=AssertionStatus.INFERRED,
                                           evidence_source_id=entry.evidence_source_id,review_status=entry.review_status,
                                           properties={"claim_boundary":"laboratory propagation hypothesis"}))
        for field in entry.telemetry_fields:
            channel_id=f"telemetry:{field}"
            nodes.setdefault(channel_id,EvidenceGraphNode(node_id=channel_id,node_type=KnowledgeNodeType.TELEMETRY_CHANNEL,
                                                          properties={"field":field}))
            edges.append(EvidenceGraphEdge(source_id=fault_id,relation_type=KnowledgeRelationType.USES_TELEMETRY_FIELD,
                                           target_id=channel_id,**common))
        for target in entry.propagation_targets:
            target_id=f"subsystem:{target}"
            nodes.setdefault(target_id,EvidenceGraphNode(node_id=target_id,node_type=KnowledgeNodeType.SUBSYSTEM,
                                                         properties={"name":target}))
            edges.append(EvidenceGraphEdge(source_id=fault_id,relation_type=KnowledgeRelationType.PROPAGATES_TO,
                                           target_id=target_id,assertion_status=AssertionStatus.INFERRED,
                                           evidence_source_id=entry.evidence_source_id,review_status=entry.review_status,
                                           properties={"claim_boundary":"laboratory propagation hypothesis"}))
        for path in entry.validated_propagation_paths:
            fact_id=f"evidence_fact:propagation:{entry.effect_id}:{path.path_id}"
            target_id=f"subsystem:{path.target_subsystem}"
            nodes[fact_id]=EvidenceGraphNode(
                node_id=fact_id,
                node_type=KnowledgeNodeType.EVIDENCE_FACT,
                properties={
                    "path_id":path.path_id,
                    "interpretation":path.interpretation,
                    "telemetry_fields":path.telemetry_fields,
                    "applicable_conditions":path.applicable_conditions,
                    "claim_boundary":path.claim_boundary,
                },
            )
            nodes.setdefault(target_id,EvidenceGraphNode(node_id=target_id,node_type=KnowledgeNodeType.SUBSYSTEM,
                                                         properties={"name":path.target_subsystem}))
            path_common=dict(assertion_status=path.assertion_status,evidence_source_id=path.evidence_source_id,
                             review_status=path.review_status)
            edges.append(EvidenceGraphEdge(source_id=fault_id,relation_type=KnowledgeRelationType.HAS_EVIDENCE_FACT,
                                           target_id=fact_id,**path_common))
            edges.append(EvidenceGraphEdge(source_id=fact_id,relation_type=KnowledgeRelationType.PROPAGATES_TO,
                                           target_id=target_id,**path_common,
                                           properties={"claim_boundary":path.claim_boundary}))
    return FaultCatalogGraph(nodes=list(nodes.values()),edges=edges,
                             effect_coverage=sorted(entry.effect_id for entry in catalog.entries))


__all__=["ValidatedPropagationPath","FaultCatalogEntry","SatSimFaultCatalog","FaultCatalogGraph","load_sat_sim_fault_catalog","build_fault_catalog_graph"]
