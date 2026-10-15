from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from astrograph.agents.mission.joint_evidence import build_joint_evidence_package
from astrograph.agents.mission.literature_context import build_literature_context_assessment
from astrograph.agents.mission.storage import sha
from astrograph.lifecycle import (
    ExpertFeedback,
    KnowledgeFeedbackStore,
    LifecyclePolicy,
    process_evidence_bound_feedback,
)
from astrograph.data_driven.failure_regression import FailureRegressionSet


def _core():
    return {"status":"SUCCEEDED","diagnosis":{"status":"INSUFFICIENT_EVIDENCE","selected_fault_id":None,
        "candidate_fault_ids":["FAULT_A"],"working_hypothesis_fault_id":"FAULT_A","reason_codes":["WORKING_ONLY"],
        "candidate_explanations":[]}}


def _package(core):
    source={"citation_id":"DOC:1","source_sha256":"a"*64,"source_locator":"page:1",
            "reviewed_claim_polarity":"SUPPORTS","reviewed_target_fault_id":"FAULT_A","reviewed_claim_id":"claim:a"}
    lit=build_literature_context_assessment(core,[source])
    return build_joint_evidence_package(core_result=core,
        model_execution={"status":"SUCCEEDED","three_task_contract":{"classification":{"status":"EXECUTED","fault_id":"FAULT_A","formal_diagnostic_vote_allowed":False}}},
        generic_result={"diagnosis":{"fault_candidates":[{"fault_id":"FAULT_A","score":0.7}]}},
        literature_context=lit,sources=[source])


def _feedback(core, package, reviewer="expert:dev50", refs=None):
    if refs is None:
        refs=[next(x for x in package.nodes if x.branch=="MODEL").evidence_id,
              next(x for x in package.nodes if x.branch=="KG").evidence_id]
    return ExpertFeedback(feedback_id="feedback:dev50",task_id="episode:dev50",result_sha256=sha(core),
        feedback_type="DIAGNOSIS_CORRECTION",reviewer=reviewer,reviewed_at=datetime(2026,9,16,4,0,tzinfo=timezone.utc),
        truth_source_ref="truth:independent",verified_fault_id="FAULT_B",target_layers=("model","knowledge"),
        diagnosis_error_summary="FAULT_A was not the verified fault.",recommended_change="Evaluate model and knowledge challengers.",
        evidence_refs=tuple(refs))


def _store(tmp_path: Path, reviewers=("expert:dev50",)):
    policy=LifecyclePolicy(approved_reviewers=reviewers,permitted_document_roots=(str(tmp_path),))
    return KnowledgeFeedbackStore(tmp_path/"lifecycle.sqlite",policy)


def test_controlled_processor_records_candidates_and_failure_case_without_promotion(tmp_path: Path) -> None:
    core=_core();package=_package(core);feedback=_feedback(core,package);store=_store(tmp_path)
    packet=process_evidence_bound_feedback(store=store,feedback=feedback,original_result=core,
        joint_evidence_package=package,regression_root=tmp_path/"regression",event_id="event:dev50:1",
        input_data_sha256="f"*64,model_refs=["model:v2"],knowledge_version="kg:v1")
    assert packet.evidence_binding.status=="BOUND"
    assert packet.error_attribution.overlap_layers==["knowledge","model"]
    assert len(packet.improvement_candidate_ids)==2
    assert packet.failure_regression_receipt["failure_case_created"] is True
    assert packet.production_modified is False and packet.automatic_promotion_allowed is False
    for cid in packet.improvement_candidate_ids:
        candidate=store.export_candidate(cid)
        assert candidate["status"]=="PENDING_REVIEW"
        assert candidate["active_release_allowed"] is False


def test_controlled_processor_is_idempotent_for_same_feedback_event(tmp_path: Path) -> None:
    core=_core();package=_package(core);feedback=_feedback(core,package);store=_store(tmp_path)
    kwargs=dict(store=store,feedback=feedback,original_result=core,joint_evidence_package=package,
        regression_root=tmp_path/"regression",event_id="event:dev50:idempotent")
    first=process_evidence_bound_feedback(**kwargs)
    second=process_evidence_bound_feedback(**kwargs)
    assert first.packet_id==second.packet_id
    assert first.improvement_candidate_ids==second.improvement_candidate_ids
    cases=FailureRegressionSet(tmp_path/"regression").cases()
    assert len(cases)==1
    assert cases[0].recorded_at==feedback.reviewed_at.isoformat()


def test_unknown_evidence_reference_blocks_before_any_durable_write(tmp_path: Path) -> None:
    core=_core();package=_package(core);feedback=_feedback(core,package,refs=["EVID-MODEL-"+"0"*20]);store=_store(tmp_path)
    with pytest.raises(ValueError,match="unknown joint evidence nodes"):
        process_evidence_bound_feedback(store=store,feedback=feedback,original_result=core,
            joint_evidence_package=package,regression_root=tmp_path/"regression",event_id="event:bad-ref")
    assert not (tmp_path/"regression"/"failure_regression_set.jsonl").exists()
    with store._db() as db:
        assert db.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]==0
        assert db.execute("SELECT COUNT(*) FROM improvement_candidates").fetchone()[0]==0


def test_unapproved_reviewer_blocks_before_failure_case_write(tmp_path: Path) -> None:
    core=_core();package=_package(core);feedback=_feedback(core,package,reviewer="expert:not-approved");store=_store(tmp_path)
    with pytest.raises(PermissionError,match="reviewer is not approved"):
        process_evidence_bound_feedback(store=store,feedback=feedback,original_result=core,
            joint_evidence_package=package,regression_root=tmp_path/"regression",event_id="event:bad-reviewer")
    assert not (tmp_path/"regression"/"failure_regression_set.jsonl").exists()


def test_same_event_id_with_changed_feedback_remains_fail_closed(tmp_path: Path) -> None:
    core=_core();package=_package(core);feedback=_feedback(core,package);store=_store(tmp_path)
    process_evidence_bound_feedback(store=store,feedback=feedback,original_result=core,
        joint_evidence_package=package,regression_root=tmp_path/"regression",event_id="event:reuse")
    changed=feedback.model_copy(update={"recommended_change":"Different proposed change."})
    with pytest.raises(ValueError,match="idempotency key reused"):
        process_evidence_bound_feedback(store=store,feedback=changed,original_result=core,
            joint_evidence_package=package,regression_root=tmp_path/"regression",event_id="event:reuse")
