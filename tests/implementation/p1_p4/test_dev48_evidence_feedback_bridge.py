from __future__ import annotations

from datetime import datetime, timezone

import pytest

from astrograph.agents.mission.joint_evidence import build_joint_evidence_package, joint_evidence_report
from astrograph.agents.mission.literature_context import build_literature_context_assessment
from astrograph.agents.mission.storage import sha
from astrograph.lifecycle import ExpertFeedback, bind_feedback_to_joint_evidence, evidence_reference_catalog
from astrograph.lifecycle.feedback_regression_bridge import bind_feedback_to_failure_regression


def _core():
    return {
        "status": "SUCCEEDED",
        "diagnosis": {
            "status": "INSUFFICIENT_EVIDENCE",
            "selected_fault_id": None,
            "candidate_fault_ids": ["ADCS_IMU_AXIS_DROPOUT"],
            "working_hypothesis_fault_id": "ADCS_IMU_AXIS_DROPOUT",
            "reason_codes": ["MODEL_WORKING_HYPOTHESIS_ONLY"],
            "candidate_explanations": [],
        },
    }


def _package(core=None):
    core = core or _core()
    source = {
        "citation_id": "DOC-REVIEWED-1:chunk:1",
        "source_sha256": "a" * 64,
        "source_locator": "page:7",
        "reviewed_claim_polarity": "SUPPORTS",
        "reviewed_target_fault_id": "ADCS_IMU_AXIS_DROPOUT",
        "reviewed_claim_id": "claim:dev48:1",
        "reviewed_mapping_id": "mapping:dev48:1",
    }
    literature = build_literature_context_assessment(core, [source])
    model_execution = {
        "status": "SUCCEEDED",
        "provider_id": "ADCS_NATIVE_THREE_TASK_V2",
        "three_task_contract": {
            "detection": {"status": "EXECUTED", "anomalous": True},
            "localization": {"status": "EXECUTED", "component_id": "IMU"},
            "classification": {"status": "EXECUTED_CALIBRATED_SHADOW", "fault_id": "ADCS_IMU_AXIS_DROPOUT", "formal_diagnostic_vote_allowed": False},
        },
    }
    generic = {"diagnosis": {"fault_candidates": [{"fault_id": "ADCS_IMU_AXIS_DROPOUT", "score": 0.8}]}}
    return build_joint_evidence_package(
        core_result=core,
        model_execution=model_execution,
        generic_result=generic,
        literature_context=literature,
        sources=[source],
    )


def _feedback(core, evidence_refs):
    return ExpertFeedback(
        feedback_id="feedback:dev48:1",
        task_id="episode:dev48",
        result_sha256=sha(core),
        feedback_type="DIAGNOSIS_CORRECTION",
        reviewer="expert:dev48",
        reviewed_at=datetime.now(timezone.utc),
        truth_source_ref="truth:new-independent-review",
        verified_fault_id="ADCS_RW_FRICTION_INCREASE",
        target_layers=("model", "knowledge"),
        diagnosis_error_summary="The working hypothesis was not the verified fault.",
        recommended_change="Review classifier and causal knowledge for this episode.",
        evidence_refs=tuple(evidence_refs),
    )


def test_joint_evidence_report_exposes_stable_expert_reference_catalog() -> None:
    package = _package()
    report = joint_evidence_report(package)
    catalog = report["expert_feedback_reference_catalog"]
    assert len(catalog) == len(package.nodes)
    assert {row["evidence_id"] for row in catalog} == {node.evidence_id for node in package.nodes}
    assert any(row["branch"] == "MODEL" for row in catalog)
    assert any(row["branch"] == "LITERATURE" for row in catalog)


def test_expert_feedback_binds_exact_joint_evidence_nodes() -> None:
    core = _core(); package = _package(core)
    model_node = next(node for node in package.nodes if node.branch == "MODEL" and node.role == "FAULT_CLASSIFICATION")
    literature_node = next(node for node in package.nodes if node.branch == "LITERATURE")
    feedback = _feedback(core, [model_node.evidence_id, literature_node.evidence_id])
    bound = bind_feedback_to_joint_evidence(feedback=feedback, joint_evidence_package=package, original_result=core)
    assert bound.status == "BOUND"
    assert bound.referenced_evidence_ids == [model_node.evidence_id, literature_node.evidence_id]
    assert bound.referenced_branches == ["LITERATURE", "MODEL"]
    assert bound.production_modified is False


def test_unknown_evidence_id_and_wrong_core_binding_fail_closed() -> None:
    core = _core(); package = _package(core)
    feedback = _feedback(core, ["EVID-MODEL-" + "0" * 20])
    with pytest.raises(ValueError, match="unknown joint evidence nodes"):
        bind_feedback_to_joint_evidence(feedback=feedback, joint_evidence_package=package, original_result=core)

    changed = _core(); changed["diagnosis"]["reason_codes"].append("CHANGED_AFTER_REVIEW")
    valid_node = next(node for node in package.nodes if node.branch == "MODEL")
    feedback2 = _feedback(changed, [valid_node.evidence_id])
    with pytest.raises(ValueError, match="joint evidence package is not bound"):
        bind_feedback_to_joint_evidence(feedback=feedback2, joint_evidence_package=package, original_result=changed)


def test_legacy_feedback_without_evidence_node_remains_explicitly_unbound() -> None:
    core = _core(); package = _package(core)
    feedback = _feedback(core, ["legacy:evidence:free-text-reference"])
    bound = bind_feedback_to_joint_evidence(feedback=feedback, joint_evidence_package=package, original_result=core)
    assert bound.status == "UNBOUND_NO_JOINT_EVIDENCE_REFERENCE"
    assert bound.referenced_evidence_ids == []


def test_failure_regression_case_persists_joint_evidence_binding(tmp_path) -> None:
    core = _core(); package = _package(core)
    model_node = next(node for node in package.nodes if node.branch == "MODEL" and node.role == "FAULT_CLASSIFICATION")
    feedback = _feedback(core, [model_node.evidence_id])
    result = bind_feedback_to_failure_regression(
        feedback=feedback,
        original_result=core,
        regression_root=tmp_path,
        input_data_sha256="f" * 64,
        model_refs=["ADCS_NATIVE_THREE_TASK_V2"],
        knowledge_version="kg:dev48",
        joint_evidence_package=package.model_dump(mode="json"),
    )
    assert result["status"] == "RECORDED"
    assert result["joint_evidence_binding"]["status"] == "BOUND"
    case = result["failure_case"]
    assert case["metadata"]["joint_evidence_binding"]["referenced_evidence_ids"] == [model_node.evidence_id]
    assert case["evidence_refs"] == [model_node.evidence_id]


def test_reference_catalog_helper_is_read_only_and_complete() -> None:
    package = _package()
    rows = evidence_reference_catalog(package)
    assert len(rows) == len(package.nodes)
    literature = [row for row in rows if row["branch"] == "LITERATURE"]
    assert literature and all(row["diagnostic_vote_allowed"] is False for row in literature)
