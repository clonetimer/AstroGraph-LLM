"""Governed offline-evaluation artifact contract for non-model improvement layers."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from astrograph.agents.mission.storage import sha
from .knowledge_feedback import KnowledgeFeedbackStore

NonModelLayer = Literal["planning", "memory", "tool", "knowledge", "execution"]

REQUIRED_CHECKS: dict[str, tuple[str, ...]] = {
    "planning": ("TRIGGER_CASE_REPLAY", "REGRESSION_SUITE", "PLAN_SAFETY", "CORE_DIAGNOSIS_NOT_REGRESSED"),
    "memory": ("TRIGGER_CASE_REPLAY", "REGRESSION_SUITE", "MEMORY_ISOLATION", "NO_CONFIRMED_MEMORY_AUTO_WRITE"),
    "tool": ("TRIGGER_CASE_REPLAY", "REGRESSION_SUITE", "TOOL_PERMISSION_BOUNDARY", "NO_PERMISSION_ESCALATION"),
    "knowledge": ("TRIGGER_CASE_REPLAY", "GRAPH_REGRESSION", "SOURCE_PROVENANCE", "NO_ACTIVE_KG_WRITE"),
    "execution": ("TRIGGER_CASE_REPLAY", "RUNTIME_REGRESSION", "TIMEOUT_CANCEL_SAFETY", "NO_PRODUCTION_CONFIG_WRITE"),
}


class LayerEvaluationArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-layer-evaluation-artifact.v1"] = "astrograph-layer-evaluation-artifact.v1"
    evaluation_id: str = Field(min_length=1)
    improvement_candidate_id: str = Field(min_length=1)
    feedback_id: str = Field(min_length=1)
    target_layer: NonModelLayer
    evaluator_id: str = Field(min_length=1)
    evaluator_version: str = Field(min_length=1)
    checks: dict[str, bool]
    metrics: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)
    evaluated_artifact_refs: list[str] = Field(default_factory=list)
    truth_oracle_used_for_tuning: Literal[False] = False
    production_modified: Literal[False] = False

    @field_validator("evidence_refs", "evaluated_artifact_refs")
    @classmethod
    def unique_refs(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("evaluation references must be unique")
        return value


class LayerEvaluationQualification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-layer-evaluation-qualification.v1"] = "astrograph-layer-evaluation-qualification.v1"
    evaluation_id: str
    improvement_candidate_id: str
    target_layer: NonModelLayer
    status: Literal["ELIGIBLE_FOR_GOVERNED_PROMOTION_REVIEW", "REJECTED"]
    required_checks: list[str]
    failed_required_checks: list[str]
    missing_required_checks: list[str]
    evaluation_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    upgrade_proposal: dict[str, Any] | None = None
    requires_human_gate: Literal[True] = True
    automatic_production_promotion: Literal[False] = False
    production_modified: Literal[False] = False
    claim_boundary: str = (
        "Non-model offline-evaluation qualification only. Passing all layer-specific checks makes the candidate "
        "eligible for governed promotion review; it does not deploy planning, memory, tool, knowledge, or execution changes."
    )


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qualify_approved_non_model_candidate(
    *,
    store: KnowledgeFeedbackStore,
    improvement_candidate_id: str,
    evaluation_artifact_path: str | Path,
    expected_sha256: str,
) -> LayerEvaluationQualification:
    candidate = store.export_candidate(improvement_candidate_id)
    if candidate.get("status") != "APPROVED_FOR_OFFLINE_EVALUATION":
        raise PermissionError("improvement candidate is not approved for offline evaluation")
    layer = candidate.get("target_layer")
    if layer == "model":
        raise ValueError("model candidates must use the existing model challenger evaluator")
    if layer not in REQUIRED_CHECKS:
        raise ValueError("no registered offline evaluator contract for target layer")
    path = Path(evaluation_artifact_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _file_sha(path)
    if actual != expected_sha256:
        raise ValueError("layer evaluation artifact SHA-256 mismatch")
    artifact = LayerEvaluationArtifact.model_validate_json(path.read_text(encoding="utf-8"))
    if artifact.improvement_candidate_id != improvement_candidate_id:
        raise ValueError("evaluation artifact does not bind the approved improvement candidate")
    if artifact.feedback_id != candidate.get("feedback_id"):
        raise ValueError("evaluation artifact feedback_id does not match lifecycle candidate")
    if artifact.target_layer != layer:
        raise ValueError("evaluation artifact target layer does not match lifecycle candidate")
    missing_evidence = sorted(set(candidate.get("evidence_refs") or []) - set(artifact.evidence_refs))
    if missing_evidence:
        raise ValueError(f"evaluation artifact omitted candidate evidence references: {missing_evidence}")

    required = list(REQUIRED_CHECKS[layer])
    missing = sorted(set(required) - set(artifact.checks))
    failed = sorted(check for check in required if artifact.checks.get(check) is False)
    status = "ELIGIBLE_FOR_GOVERNED_PROMOTION_REVIEW" if not missing and not failed else "REJECTED"
    proposal = None
    if status == "ELIGIBLE_FOR_GOVERNED_PROMOTION_REVIEW":
        proposal = {
            "proposal_id": "LUPG-" + sha({"candidate_id": improvement_candidate_id, "evaluation_sha256": actual})[:20].upper(),
            "improvement_candidate_id": improvement_candidate_id,
            "feedback_id": candidate["feedback_id"],
            "target_layer": layer,
            "evaluation_id": artifact.evaluation_id,
            "evaluation_artifact_sha256": actual,
            "requires_human_gate": True,
            "promoted": False,
        }
    return LayerEvaluationQualification(
        evaluation_id=artifact.evaluation_id,
        improvement_candidate_id=improvement_candidate_id,
        target_layer=layer,
        status=status,
        required_checks=required,
        failed_required_checks=failed,
        missing_required_checks=missing,
        evaluation_artifact_sha256=actual,
        upgrade_proposal=proposal,
    )


__all__=["LayerEvaluationArtifact","LayerEvaluationQualification","REQUIRED_CHECKS","qualify_approved_non_model_candidate"]
