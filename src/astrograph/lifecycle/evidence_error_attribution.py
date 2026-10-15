"""Deterministic, review-only error-attribution suggestions from bound evidence nodes."""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict

from astrograph.agents.mission.joint_evidence import JointEvidencePackage
from .evidence_feedback_bridge import EvidenceFeedbackBinding
from .knowledge_feedback import ExpertFeedback


_LAYER_MAP: dict[str, tuple[str, ...]] = {
    "MODEL": ("model",),
    "KG": ("knowledge",),
    "LITERATURE": ("knowledge", "memory"),
    "CORE_DIAGNOSIS": ("planning", "execution"),
}


class EvidenceErrorAttribution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-evidence-error-attribution.v1"] = "astrograph-evidence-error-attribution.v1"
    feedback_id: str
    binding_status: str
    referenced_branches: list[str]
    suggested_target_layers: list[str]
    expert_target_layers: list[str]
    overlap_layers: list[str]
    expert_only_layers: list[str]
    suggestion_only_layers: list[str]
    status: Literal["ALIGNED", "PARTIALLY_ALIGNED", "NO_STRUCTURED_EVIDENCE_BINDING"]
    expert_decision_overridden: Literal[False] = False
    production_modified: Literal[False] = False
    claim_boundary: str = (
        "Deterministic triage suggestion only. Expert target_layers remain authoritative for candidate creation; "
        "the attribution does not change planner, model, knowledge, memory, tools, or execution settings."
    )


def attribute_feedback_error_layers(
    *,
    feedback: ExpertFeedback,
    binding: EvidenceFeedbackBinding,
    joint_evidence_package: JointEvidencePackage | dict[str, Any],
) -> EvidenceErrorAttribution:
    package = joint_evidence_package if isinstance(joint_evidence_package, JointEvidencePackage) else JointEvidencePackage.model_validate(joint_evidence_package)
    if binding.joint_evidence_package_id != package.package_id:
        raise ValueError("evidence feedback binding does not match the supplied joint evidence package")
    if binding.feedback_id != feedback.feedback_id:
        raise ValueError("evidence feedback binding does not match the supplied expert feedback")
    by_id={node.evidence_id:node for node in package.nodes}
    branches=[]
    for ref in binding.referenced_evidence_ids:
        node=by_id.get(ref)
        if node is None:
            raise ValueError("bound evidence reference is absent from joint evidence package")
        branches.append(node.branch)
    branches=sorted(set(branches))
    suggested=sorted({layer for branch in branches for layer in _LAYER_MAP[branch]})
    expert=sorted(set(feedback.target_layers))
    overlap=sorted(set(suggested)&set(expert))
    expert_only=sorted(set(expert)-set(suggested))
    suggestion_only=sorted(set(suggested)-set(expert))
    if not binding.referenced_evidence_ids:
        status="NO_STRUCTURED_EVIDENCE_BINDING"
    elif not expert_only and not suggestion_only:
        status="ALIGNED"
    else:
        status="PARTIALLY_ALIGNED"
    return EvidenceErrorAttribution(
        feedback_id=feedback.feedback_id,
        binding_status=binding.status,
        referenced_branches=branches,
        suggested_target_layers=suggested,
        expert_target_layers=expert,
        overlap_layers=overlap,
        expert_only_layers=expert_only,
        suggestion_only_layers=suggestion_only,
        status=status,
    )


__all__=["EvidenceErrorAttribution","attribute_feedback_error_layers"]
