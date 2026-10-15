"""Bind expert corrections to content-addressed failure-regression cases."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from astrograph.data_driven.failure_regression import FailureCase, FailureRegressionSet
from astrograph.evidence.executable import stable_id
from astrograph.agents.mission.storage import sha

from .knowledge_feedback import ExpertFeedback
from .evidence_feedback_bridge import bind_feedback_to_joint_evidence
from .evidence_error_attribution import attribute_feedback_error_layers


def _selected_fault(result: dict[str, Any]) -> str | None:
    diagnosis=result.get("diagnosis",{}) if isinstance(result,dict) else {}
    return diagnosis.get("selected_fault_id") or diagnosis.get("conclusion_fault_id") or diagnosis.get("fusion_top_fault_id")


def bind_feedback_to_failure_regression(
    *,
    feedback: ExpertFeedback,
    original_result: dict[str, Any],
    regression_root: str | Path,
    input_data_sha256: str | None = None,
    model_refs: list[str] | None = None,
    knowledge_version: str | None = None,
    joint_evidence_package: dict[str, Any] | None = None,
) -> dict[str, Any]:
    actual=sha(original_result)
    if actual != feedback.result_sha256:
        raise ValueError("expert feedback result hash does not match the bound diagnosis result")
    if feedback.feedback_type != "DIAGNOSIS_CORRECTION" or not feedback.verified_fault_id:
        return {"status":"NOT_APPLICABLE","feedback_id":feedback.feedback_id,"failure_case_created":False}
    predicted=_selected_fault(original_result)
    failure_type="MISSED" if not predicted else "MISDIAGNOSIS"
    predicted_for_schema=predicted or "NO_FAULT_SELECTED"
    case_id=stable_id("FAIL",{
        "feedback_id":feedback.feedback_id,"task_id":feedback.task_id,"result_sha256":actual,
        "truth":feedback.verified_fault_id,"predicted":predicted_for_schema,
    })
    evidence_binding=None
    evidence_attribution=None
    if joint_evidence_package is not None:
        bound=bind_feedback_to_joint_evidence(
            feedback=feedback,joint_evidence_package=joint_evidence_package,original_result=original_result
        )
        evidence_binding=bound.model_dump(mode="json")
        evidence_attribution=attribute_feedback_error_layers(
            feedback=feedback,binding=bound,joint_evidence_package=joint_evidence_package
        ).model_dump(mode="json")
    case=FailureCase(
        case_id=case_id, episode_id=feedback.task_id, failure_type=failure_type,
        truth_fault_id=feedback.verified_fault_id, predicted_fault_id=predicted_for_schema,
        evidence_summary=feedback.diagnosis_error_summary, feedback_id=feedback.feedback_id,
        task_id=feedback.task_id, result_sha256=actual, input_data_sha256=input_data_sha256,
        model_refs=list(model_refs or []), knowledge_version=knowledge_version,
        evidence_refs=list(feedback.evidence_refs),
        metadata={"truth_source_ref":feedback.truth_source_ref,"target_layers":list(feedback.target_layers),
                  "joint_evidence_binding":evidence_binding,"evidence_error_attribution":evidence_attribution},
        recorded_at=feedback.reviewed_at.astimezone(__import__("datetime").timezone.utc).isoformat(),
    )
    store=FailureRegressionSet(Path(regression_root));path=store.add(case)
    return {
        "status":"RECORDED","feedback_id":feedback.feedback_id,
        "failure_case_created":True,"failure_case":case.model_dump(mode="json"),
        "regression_set_path":str(path),"joint_evidence_binding":evidence_binding,
        "evidence_error_attribution":evidence_attribution,"production_modified":False,
    }


__all__=["bind_feedback_to_failure_regression"]
