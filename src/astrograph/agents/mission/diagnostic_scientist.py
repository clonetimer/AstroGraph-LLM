"""Bounded R36 diagnostic experiment executor.

The compatibility class name is retained, but hypotheses, evidence requests
and experiment plans are supplied by the caller.  This module does not claim
autonomous hypothesis generation or optimal experimental design.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from astrograph.agents.controlled_tools import (
    ControlledToolRegistry,
    ToolApproval,
)

from astrograph.diagnosis.reasoning_graph import (
    BayesianDiagnosticReasoner,
    DiagnosticEdgeRelation,
    DiagnosticNodeType,
    DiagnosticReasoningEdge,
    DiagnosticReasoningGraph,
    DiagnosticReasoningNode,
    PosteriorEstimate,
)

from astrograph.evidence.quality import (
    EvidenceQualityAssessment,
    EvidenceQualityStatus,
)


class _Strict(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )


class ScientistStatus(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    INCOMPLETE = "INCOMPLETE"


class DiagnosticHypothesisProposal(_Strict):
    hypothesis_id: str = Field(min_length=1)
    fault_id: str = Field(min_length=1)
    label: str = Field(min_length=1)

    prior_probability: float = Field(
        gt=0.0,
        lt=1.0,
    )

    rationale: str = Field(min_length=1)


class EvidenceRequest(_Strict):
    request_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    required_observation: str = Field(min_length=1)


class ExperimentPlan(_Strict):
    plan_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    tool_id: str = Field(min_length=1)

    args: dict[str, Any] = Field(
        default_factory=dict
    )


class ExperimentEvidence(_Strict):
    evidence_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)

    likelihood_ratio: float = Field(gt=0.0)

    edge_confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    source_ref: str = Field(min_length=1)

    quality: EvidenceQualityAssessment

    @model_validator(mode="after")
    def complete_quality(
        self,
    ) -> "ExperimentEvidence":

        if (
            self.quality.evidence_id
            != self.evidence_id
        ):
            raise ValueError(
                "experiment evidence quality "
                "identity mismatch"
            )

        if (
            self.quality.status
            != EvidenceQualityStatus.COMPLETE
            or
            self.quality.screening_score
            is None
        ):
            raise ValueError(
                "experiment evidence requires "
                "COMPLETE evidence quality"
            )

        return self


class DiagnosticScientistResult(_Strict):
    schema_version: Literal[
        "astrograph-diagnostic-scientist-result.v1"
    ] = "astrograph-diagnostic-scientist-result.v1"

    status: ScientistStatus

    posteriors: tuple[
        PosteriorEstimate,
        ...
    ] = ()

    accepted_evidence_ids: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()

    tool_invocation_records: tuple[
        dict[str, Any],
        ...
    ] = ()

    control_action_authorized: Literal[False] = False

    knowledge_write_authorized: Literal[False] = False

    release_authority: Literal[False] = False

    claim_boundary: str = (
        "Controlled execution of caller-supplied diagnostic hypotheses, "
        "evidence requests and experiment plans. "
        "Tool results are evidence inputs only; "
        "the loop has no spacecraft control, "
        "knowledge-publication, or release authority."
    )


def _normalize_quality_payload(
    value: Any,
) -> Any:

    if not isinstance(value, dict):
        return value

    result = dict(value)

    # These are computed fields.
    # Never trust them on re-ingress.
    for key in (
        "status",
        "completeness",
        "screening_score",
        "overall_confidence",
    ):
        result.pop(key, None)

    return result


def _normalize_experiment_payload(
    value: Any,
) -> dict[str, Any]:

    if hasattr(value, "model_dump"):
        value = value.model_dump(
            mode="json"
        )

    if not isinstance(value, dict):
        raise ValueError(
            "experiment tool must return an object"
        )

    result = dict(value)

    if "quality" in result:
        result["quality"] = (
            _normalize_quality_payload(
                result["quality"]
            )
        )

    return result


class DiagnosticScientistAgent:

    agent_id = (
        "astrograph-diagnostic-scientist-r36"
    )

    def __init__(
        self,
        registry: ControlledToolRegistry,
        *,
        reasoner:
            BayesianDiagnosticReasoner | None = None,
    ):
        self.registry = registry
        self.reasoner = (
            reasoner
            or BayesianDiagnosticReasoner()
        )

    def run(
        self,
        *,
        hypotheses:
            list[DiagnosticHypothesisProposal],
        requests:
            list[EvidenceRequest],
        plans:
            list[ExperimentPlan],
        approvals:
            dict[str, ToolApproval] | None = None,
        evaluated_at:
            datetime | None = None,
    ) -> DiagnosticScientistResult:

        approvals = approvals or {}

        evaluated_at = (
            evaluated_at
            or datetime.now(timezone.utc)
        )

        hypothesis_by_id = {
            item.hypothesis_id: item
            for item in hypotheses
        }

        request_by_id = {
            item.request_id: item
            for item in requests
        }

        if (
            len(hypothesis_by_id)
            != len(hypotheses)
        ):
            raise ValueError(
                "duplicate hypothesis_id"
            )

        if (
            len(request_by_id)
            != len(requests)
        ):
            raise ValueError(
                "duplicate request_id"
            )

        if any(
            request.hypothesis_id
            not in hypothesis_by_id
            for request in requests
        ):
            raise ValueError(
                "evidence request references "
                "unknown hypothesis"
            )

        blockers: list[str] = []
        accepted: list[ExperimentEvidence] = []
        records: list[dict[str, Any]] = []

        for plan in plans:

            request = request_by_id.get(
                plan.request_id
            )

            if (
                request is None
                or
                plan.hypothesis_id
                not in hypothesis_by_id
                or
                request.hypothesis_id
                != plan.hypothesis_id
            ):
                blockers.append(
                    f"PLAN_BINDING_INVALID:"
                    f"{plan.plan_id}"
                )
                continue

            invocation = self.registry.invoke(
                self.agent_id,
                plan.tool_id,
                dict(plan.args),
                approval=approvals.get(
                    plan.plan_id
                ),
            )

            records.append(
                dict(
                    invocation.get("record")
                    or {}
                )
            )

            if not invocation.get("permitted"):
                blockers.append(
                    f"TOOL_DENIED:"
                    f"{plan.plan_id}:"
                    f"{plan.tool_id}"
                )
                continue

            try:
                evidence = (
                    ExperimentEvidence
                    .model_validate(
                        _normalize_experiment_payload(
                            invocation.get("result")
                        )
                    )
                )

            except Exception as exc:
                blockers.append(
                    "INVALID_EVIDENCE_RESULT:"
                    f"{plan.plan_id}:"
                    f"{type(exc).__name__}"
                )
                continue

            if (
                evidence.hypothesis_id
                != plan.hypothesis_id
            ):
                blockers.append(
                    "EVIDENCE_HYPOTHESIS_MISMATCH:"
                    f"{plan.plan_id}"
                )
                continue

            accepted.append(evidence)

        if not accepted:
            return DiagnosticScientistResult(
                status=ScientistStatus.INCOMPLETE,
                blockers=tuple(
                    dict.fromkeys(
                        blockers
                        or [
                            "NO_QUALIFIED_"
                            "EXPERIMENT_EVIDENCE"
                        ]
                    )
                ),
                tool_invocation_records=tuple(
                    records
                ),
            )

        nodes = [
            DiagnosticReasoningNode(
                node_id=hypothesis.fault_id,
                node_type=(
                    DiagnosticNodeType.FAULT
                ),
                label=hypothesis.label,
                prior_probability=(
                    hypothesis.prior_probability
                ),
            )
            for hypothesis in hypotheses
        ]

        edges: list[
            DiagnosticReasoningEdge
        ] = []

        hypothesis_fault = {
            item.hypothesis_id:
                item.fault_id
            for item in hypotheses
        }

        for index, evidence in enumerate(
            accepted,
            1,
        ):
            nodes.append(
                DiagnosticReasoningNode(
                    node_id=evidence.evidence_id,
                    node_type=(
                        DiagnosticNodeType.EVIDENCE
                    ),
                    label=evidence.evidence_id,
                    evidence_quality=(
                        evidence.quality
                    ),
                )
            )

            relation = (
                DiagnosticEdgeRelation.SUPPORTS
                if evidence.likelihood_ratio >= 1.0
                else
                DiagnosticEdgeRelation.CONTRADICTS
            )

            edges.append(
                DiagnosticReasoningEdge(
                    edge_id=(
                        f"experiment-edge-{index}"
                    ),
                    source_id=(
                        evidence.evidence_id
                    ),
                    target_id=(
                        hypothesis_fault[
                            evidence.hypothesis_id
                        ]
                    ),
                    relation=relation,
                    confidence=(
                        evidence.edge_confidence
                    ),
                    source_ref=(
                        evidence.source_ref
                    ),
                    likelihood_ratio=(
                        evidence.likelihood_ratio
                    ),
                )
            )

        inference = self.reasoner.infer(
            DiagnosticReasoningGraph(
                graph_id=(
                    "diagnostic-scientist-r36"
                ),
                nodes=tuple(nodes),
                edges=tuple(edges),
            ),
            observed_evidence_ids={
                item.evidence_id
                for item in accepted
            },
            at_time=evaluated_at,
        )

        status = (
            ScientistStatus.PARTIAL
            if blockers
            or inference.excluded_evidence
            else
            ScientistStatus.COMPLETE
        )

        blockers.extend(
            "INFERENCE_EXCLUDED:"
            f"{key}:{value}"
            for key, value
            in inference.excluded_evidence.items()
        )

        return DiagnosticScientistResult(
            status=status,
            posteriors=inference.posteriors,
            accepted_evidence_ids=tuple(
                item.evidence_id
                for item in accepted
            ),
            blockers=tuple(
                dict.fromkeys(blockers)
            ),
            tool_invocation_records=tuple(
                records
            ),
        )


__all__ = [
    "ScientistStatus",
    "DiagnosticHypothesisProposal",
    "EvidenceRequest",
    "ExperimentPlan",
    "ExperimentEvidence",
    "DiagnosticScientistResult",
    "DiagnosticScientistAgent",
]
