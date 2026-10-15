"""Time-valid probabilistic evidence graph using tempered likelihood ratios.

The topology records diagnostic relations.  It is not a structural causal
model: CAUSES/EXPLAINS edges are descriptive and do not enter the numerical
update implemented here.
"""
from __future__ import annotations

import math
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.evidence.quality import (
    EvidenceQualityAssessment,
    EvidenceQualityStatus,
)


class DiagnosticNodeType(str, Enum):
    FAULT = "FAULT"
    COMPONENT = "COMPONENT"
    SENSOR = "SENSOR"
    PHYSICS = "PHYSICS"
    EVIDENCE = "EVIDENCE"
    TIME = "TIME"
    ACTION = "ACTION"


class DiagnosticEdgeRelation(str, Enum):
    CAUSES = "CAUSES"
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    EXPLAINS = "EXPLAINS"


class ReasoningStatus(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class _Strict(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )


class DiagnosticReasoningNode(_Strict):
    node_id: str = Field(min_length=1)
    node_type: DiagnosticNodeType
    label: str = Field(min_length=1)

    prior_probability: float | None = Field(
        default=None,
        gt=0.0,
        lt=1.0,
    )

    evidence_quality: EvidenceQualityAssessment | None = None

    @model_validator(mode="after")
    def validate_role(self) -> "DiagnosticReasoningNode":
        if (
            self.prior_probability is not None
            and self.node_type
            not in {
                DiagnosticNodeType.FAULT,
                DiagnosticNodeType.COMPONENT,
            }
        ):
            raise ValueError(
                "prior probability is only valid for FAULT/COMPONENT targets"
            )

        if self.evidence_quality is not None:
            if self.node_type != DiagnosticNodeType.EVIDENCE:
                raise ValueError(
                    "evidence quality is only valid on EVIDENCE nodes"
                )

            if self.evidence_quality.evidence_id != self.node_id:
                raise ValueError(
                    "evidence quality must bind to the evidence node id"
                )

        return self


class DiagnosticReasoningEdge(_Strict):
    edge_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)

    relation: DiagnosticEdgeRelation

    confidence: float = Field(ge=0.0, le=1.0)
    source_ref: str = Field(min_length=1)

    valid_from: datetime | None = None
    valid_to: datetime | None = None

    likelihood_ratio: float | None = Field(
        default=None,
        gt=0.0,
    )

    @model_validator(mode="after")
    def validate_probability_semantics(
        self,
    ) -> "DiagnosticReasoningEdge":

        for name, value in (
            ("valid_from", self.valid_from),
            ("valid_to", self.valid_to),
        ):
            if value is not None and (
                value.tzinfo is None or value.utcoffset() is None
            ):
                raise ValueError(f"{name} must be timezone-aware")

        if (
            self.valid_from
            and self.valid_to
            and self.valid_from > self.valid_to
        ):
            raise ValueError("edge validity interval is inverted")

        if self.relation == DiagnosticEdgeRelation.SUPPORTS:
            if (
                self.likelihood_ratio is None
                or self.likelihood_ratio < 1.0
            ):
                raise ValueError(
                    "SUPPORTS edge requires likelihood_ratio >= 1"
                )

        elif self.relation == DiagnosticEdgeRelation.CONTRADICTS:
            if (
                self.likelihood_ratio is None
                or self.likelihood_ratio > 1.0
            ):
                raise ValueError(
                    "CONTRADICTS edge requires "
                    "0 < likelihood_ratio <= 1"
                )

        elif self.likelihood_ratio is not None:
            raise ValueError(
                "likelihood_ratio is only valid on "
                "SUPPORTS/CONTRADICTS edges"
            )

        return self

    def active_at(self, at_time: datetime) -> bool:
        if (
            self.valid_from is not None
            and at_time < self.valid_from
        ):
            return False

        if (
            self.valid_to is not None
            and at_time > self.valid_to
        ):
            return False

        return True


class DiagnosticReasoningGraph(_Strict):
    schema_version: Literal[
        "astrograph-diagnostic-reasoning-graph.v1"
    ] = "astrograph-diagnostic-reasoning-graph.v1"

    graph_id: str = Field(min_length=1)

    nodes: tuple[DiagnosticReasoningNode, ...]
    edges: tuple[DiagnosticReasoningEdge, ...]

    claim_boundary: str = (
        "Time-valid probabilistic evidence structure with explicit tempered "
        "likelihood-ratio inputs. CAUSES/EXPLAINS topology is descriptive; "
        "the graph is not a structural causal model and does not establish "
        "causal proof or flight qualification."
    )

    @model_validator(mode="after")
    def validate_graph(self) -> "DiagnosticReasoningGraph":
        node_ids = [node.node_id for node in self.nodes]

        if len(node_ids) != len(set(node_ids)):
            raise ValueError(
                "diagnostic reasoning node ids must be unique"
            )

        edge_ids = [edge.edge_id for edge in self.edges]

        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError(
                "diagnostic reasoning edge ids must be unique"
            )

        known = {
            node.node_id: node
            for node in self.nodes
        }

        for edge in self.edges:
            if (
                edge.source_id not in known
                or edge.target_id not in known
            ):
                raise ValueError(
                    "diagnostic reasoning edge references "
                    "an unknown node"
                )

            if edge.relation in {
                DiagnosticEdgeRelation.SUPPORTS,
                DiagnosticEdgeRelation.CONTRADICTS,
            }:
                if (
                    known[edge.source_id].node_type
                    != DiagnosticNodeType.EVIDENCE
                ):
                    raise ValueError(
                        "probabilistic support/contradiction "
                        "must originate from EVIDENCE"
                    )

                if known[edge.target_id].node_type not in {
                    DiagnosticNodeType.FAULT,
                    DiagnosticNodeType.COMPONENT,
                }:
                    raise ValueError(
                        "probabilistic support/contradiction must "
                        "target FAULT/COMPONENT"
                    )

        return self


class PosteriorEstimate(_Strict):
    target_id: str

    target_type: Literal[
        DiagnosticNodeType.FAULT,
        DiagnosticNodeType.COMPONENT,
    ]

    prior_probability: float = Field(gt=0.0, lt=1.0)
    posterior_probability: float = Field(gt=0.0, lt=1.0)

    used_edge_ids: tuple[str, ...] = ()
    used_evidence_ids: tuple[str, ...] = ()


class DiagnosticInferenceResult(_Strict):
    schema_version: Literal[
        "astrograph-diagnostic-inference-result.v1"
    ] = "astrograph-diagnostic-inference-result.v1"

    graph_id: str
    status: ReasoningStatus
    evaluated_at: datetime

    posteriors: tuple[PosteriorEstimate, ...]
    excluded_evidence: dict[str, str]

    claim_boundary: str = (
        "Generalized Bayesian estimate using declared priors and tempered "
        "likelihood ratios. Evidence screening scores act as likelihood "
        "temperatures; this is not an ordinary independent-evidence Bayesian "
        "posterior, calibrated truth probability or causal proof."
    )


class TemperedLikelihoodReasoner:

    @staticmethod
    def _stable_logistic(log_odds: float) -> float:
        """Return sigmoid(log_odds) without overflowing at extreme evidence."""
        if log_odds >= 0.0:
            tail = math.exp(-log_odds)
            return 1.0 / (1.0 + tail)
        head = math.exp(log_odds)
        return head / (1.0 + head)

    def infer(
        self,
        graph: DiagnosticReasoningGraph,
        *,
        observed_evidence_ids: set[str],
        at_time: datetime,
        require_complete_quality: bool = True,
    ) -> DiagnosticInferenceResult:

        if (
            at_time.tzinfo is None
            or at_time.utcoffset() is None
        ):
            raise ValueError(
                "timezone-aware inference time required"
            )

        nodes = {
            node.node_id: node
            for node in graph.nodes
        }

        for evidence_id in observed_evidence_ids:
            node = nodes.get(evidence_id)

            if (
                node is None
                or node.node_type
                != DiagnosticNodeType.EVIDENCE
            ):
                raise ValueError(
                    "observed evidence is not an EVIDENCE node: "
                    f"{evidence_id}"
                )

        excluded: dict[str, str] = {}
        usable_confidence: dict[str, float] = {}

        for evidence_id in observed_evidence_ids:
            quality = nodes[evidence_id].evidence_quality

            if quality is None:
                if require_complete_quality:
                    excluded[evidence_id] = (
                        "EVIDENCE_QUALITY_MISSING"
                    )
                    continue

                usable_confidence[evidence_id] = 1.0
                continue

            if (
                quality.status
                != EvidenceQualityStatus.COMPLETE
                or quality.screening_score is None
            ):
                excluded[evidence_id] = (
                    "EVIDENCE_QUALITY_INCOMPLETE"
                )
                continue

            usable_confidence[evidence_id] = (
                quality.screening_score
            )

        posteriors: list[PosteriorEstimate] = []

        for target in graph.nodes:
            if target.node_type not in {
                DiagnosticNodeType.FAULT,
                DiagnosticNodeType.COMPONENT,
            }:
                continue

            if target.prior_probability is None:
                continue

            log_odds = math.log(
                target.prior_probability
                / (1.0 - target.prior_probability)
            )

            used_edges: list[str] = []
            used_evidence: list[str] = []

            for edge in graph.edges:
                if (
                    edge.target_id != target.node_id
                    or edge.source_id
                    not in usable_confidence
                ):
                    continue

                if edge.relation not in {
                    DiagnosticEdgeRelation.SUPPORTS,
                    DiagnosticEdgeRelation.CONTRADICTS,
                }:
                    continue

                if not edge.active_at(at_time):
                    excluded.setdefault(
                        edge.source_id,
                        "EVIDENCE_EDGE_OUTSIDE_TIME_VALIDITY",
                    )
                    continue

                weight = (
                    edge.confidence
                    * usable_confidence[edge.source_id]
                )

                log_odds += (
                    math.log(float(edge.likelihood_ratio))
                    * weight
                )

                used_edges.append(edge.edge_id)
                used_evidence.append(edge.source_id)

            posterior = self._stable_logistic(log_odds)

            posterior = min(
                max(posterior, 1.0e-15),
                1.0 - 1.0e-15,
            )

            posteriors.append(
                PosteriorEstimate(
                    target_id=target.node_id,
                    target_type=target.node_type,
                    prior_probability=(
                        target.prior_probability
                    ),
                    posterior_probability=posterior,
                    used_edge_ids=tuple(used_edges),
                    used_evidence_ids=tuple(
                        dict.fromkeys(used_evidence)
                    ),
                )
            )

        used_count = sum(
            len(item.used_edge_ids)
            for item in posteriors
        )

        if used_count == 0:
            status = (
                ReasoningStatus.INSUFFICIENT_EVIDENCE
            )
        elif excluded or any(not item.used_edge_ids for item in posteriors):
            status = ReasoningStatus.PARTIAL
        else:
            status = ReasoningStatus.COMPLETE

        return DiagnosticInferenceResult(
            graph_id=graph.graph_id,
            status=status,
            evaluated_at=at_time,
            posteriors=tuple(posteriors),
            excluded_evidence=excluded,
        )


# Compatibility name for R36 callers.  New research text and code should use
# TemperedLikelihoodReasoner because the update is generalized/power Bayesian,
# not an ordinary independent-evidence Bayesian posterior.
BayesianDiagnosticReasoner = TemperedLikelihoodReasoner


__all__ = [
    "DiagnosticNodeType",
    "DiagnosticEdgeRelation",
    "ReasoningStatus",
    "DiagnosticReasoningNode",
    "DiagnosticReasoningEdge",
    "DiagnosticReasoningGraph",
    "PosteriorEstimate",
    "DiagnosticInferenceResult",
    "TemperedLikelihoodReasoner",
    "BayesianDiagnosticReasoner",
]
