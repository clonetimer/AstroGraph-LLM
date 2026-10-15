"""Orchestrate evidence-bound expert correction into existing governed stores.

The processor deliberately does not approve, evaluate, promote, or deploy any
candidate.  It only validates the correction, records the existing lifecycle
candidates, and appends the content-addressed failure-regression case.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from astrograph.agents.mission.joint_evidence import JointEvidencePackage
from astrograph.agents.mission.storage import sha
from .evidence_error_attribution import EvidenceErrorAttribution, attribute_feedback_error_layers
from .evidence_feedback_bridge import EvidenceFeedbackBinding, bind_feedback_to_joint_evidence
from .feedback_regression_bridge import bind_feedback_to_failure_regression
from .knowledge_feedback import ExpertFeedback, KnowledgeFeedbackStore


class ControlledFeedbackProcessingPacket(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-controlled-feedback-processing-packet.v1"] = "astrograph-controlled-feedback-processing-packet.v1"
    packet_id: str = Field(min_length=1)
    feedback_id: str
    task_id: str
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_binding: EvidenceFeedbackBinding
    error_attribution: EvidenceErrorAttribution
    lifecycle_receipt: dict[str, Any]
    failure_regression_receipt: dict[str, Any]
    improvement_candidate_ids: list[str]
    requires_offline_evaluation: Literal[True] = True
    production_modified: Literal[False] = False
    automatic_promotion_allowed: Literal[False] = False
    claim_boundary: str = (
        "Governed intake packet only. Improvement candidates remain pending review/offline evaluation; "
        "no planner/model/tool/knowledge/memory/execution production state is changed."
    )


def process_evidence_bound_feedback(
    *,
    store: KnowledgeFeedbackStore,
    feedback: ExpertFeedback,
    original_result: dict[str, Any],
    joint_evidence_package: JointEvidencePackage | dict[str, Any],
    regression_root: str | Path,
    event_id: str,
    input_data_sha256: str | None = None,
    model_refs: list[str] | None = None,
    knowledge_version: str | None = None,
) -> ControlledFeedbackProcessingPacket:
    """Validate first, then use the existing idempotent lifecycle/regression stores."""
    package = joint_evidence_package if isinstance(joint_evidence_package, JointEvidencePackage) else JointEvidencePackage.model_validate(joint_evidence_package)
    # Validate all content-addressed relations before the first durable write.
    binding = bind_feedback_to_joint_evidence(
        feedback=feedback, joint_evidence_package=package, original_result=original_result
    )
    attribution = attribute_feedback_error_layers(
        feedback=feedback, binding=binding, joint_evidence_package=package
    )
    # Reviewer permission is also checked before the failure JSONL is touched.
    store._reviewer(feedback.reviewer)

    lifecycle = store.submit_feedback(feedback, event_id=event_id)
    regression = bind_feedback_to_failure_regression(
        feedback=feedback,
        original_result=original_result,
        regression_root=regression_root,
        input_data_sha256=input_data_sha256,
        model_refs=model_refs,
        knowledge_version=knowledge_version,
        joint_evidence_package=package.model_dump(mode="json"),
    )
    candidate_ids = list(lifecycle.get("candidate_ids") or [])
    if len(candidate_ids) != len(feedback.target_layers):
        raise RuntimeError("lifecycle store did not create exactly one candidate per expert target layer")
    packet_payload = {
        "feedback_id": feedback.feedback_id,
        "task_id": feedback.task_id,
        "result_sha256": feedback.result_sha256,
        "joint_evidence_package_id": package.package_id,
        "candidate_ids": candidate_ids,
        "failure_case_id": (regression.get("failure_case") or {}).get("case_id"),
    }
    return ControlledFeedbackProcessingPacket(
        packet_id="CFP-" + sha(packet_payload)[:20].upper(),
        feedback_id=feedback.feedback_id,
        task_id=feedback.task_id,
        result_sha256=feedback.result_sha256,
        evidence_binding=binding,
        error_attribution=attribution,
        lifecycle_receipt=lifecycle,
        failure_regression_receipt=regression,
        improvement_candidate_ids=candidate_ids,
    )


__all__=["ControlledFeedbackProcessingPacket","process_evidence_bound_feedback"]
