"""Bind expert feedback to immutable Mission joint-evidence nodes."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from astrograph.agents.mission.joint_evidence import JointEvidencePackage
from astrograph.agents.mission.storage import sha
from .knowledge_feedback import ExpertFeedback


class EvidenceFeedbackBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-evidence-feedback-binding.v1"] = "astrograph-evidence-feedback-binding.v1"
    feedback_id: str
    task_id: str
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    joint_evidence_package_id: str
    joint_evidence_package_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    referenced_evidence_ids: list[str]
    referenced_branches: list[str]
    referenced_roles: list[str]
    status: Literal["BOUND", "UNBOUND_NO_JOINT_EVIDENCE_REFERENCE"]
    production_modified: Literal[False] = False


def _package(value: JointEvidencePackage | dict[str, Any]) -> JointEvidencePackage:
    if isinstance(value, JointEvidencePackage):
        return value
    return JointEvidencePackage.model_validate(value)


def bind_feedback_to_joint_evidence(
    *,
    feedback: ExpertFeedback,
    joint_evidence_package: JointEvidencePackage | dict[str, Any],
    original_result: dict[str, Any],
) -> EvidenceFeedbackBinding:
    package = _package(joint_evidence_package)
    result_sha = sha(original_result)
    if result_sha != feedback.result_sha256:
        raise ValueError("expert feedback result hash does not match the bound diagnosis result")
    if package.core_diagnosis_sha256 != result_sha:
        raise ValueError("joint evidence package is not bound to the expert-reviewed diagnosis result")

    by_id = {item.evidence_id: item for item in package.nodes}
    refs = [ref for ref in feedback.evidence_refs if ref.startswith("EVID-")]
    if len(refs) != len(set(refs)):
        raise ValueError("expert feedback joint-evidence references must be unique")
    unknown = sorted(set(refs) - set(by_id))
    if unknown:
        raise ValueError(f"expert feedback references unknown joint evidence nodes: {unknown}")
    selected = [by_id[ref] for ref in refs]
    return EvidenceFeedbackBinding(
        feedback_id=feedback.feedback_id,
        task_id=feedback.task_id,
        result_sha256=result_sha,
        joint_evidence_package_id=package.package_id,
        joint_evidence_package_sha256=sha(package.model_dump(mode="json")),
        referenced_evidence_ids=refs,
        referenced_branches=sorted({item.branch for item in selected}),
        referenced_roles=sorted({item.role for item in selected}),
        status="BOUND" if refs else "UNBOUND_NO_JOINT_EVIDENCE_REFERENCE",
    )


def evidence_reference_catalog(package: JointEvidencePackage | dict[str, Any]) -> list[dict[str, Any]]:
    parsed = _package(package)
    return [
        {
            "evidence_id": item.evidence_id,
            "branch": item.branch,
            "role": item.role,
            "state": item.state,
            "fault_id": item.fault_id,
            "source_ref": item.source_ref,
            "source_sha256": item.source_sha256,
            "source_locator": item.source_locator,
            "diagnostic_vote_allowed": item.diagnostic_vote_allowed,
        }
        for item in parsed.nodes
    ]


__all__ = ["EvidenceFeedbackBinding", "bind_feedback_to_joint_evidence", "evidence_reference_catalog"]
