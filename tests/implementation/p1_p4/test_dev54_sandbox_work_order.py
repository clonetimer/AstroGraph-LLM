from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from astrograph.lifecycle import (
    ALLOWED_CHANGE_KINDS,
    MODEL_REQUIRED_CHECKS,
    ExpertFeedback,
    KnowledgeFeedbackStore,
    LifecyclePolicy,
    REQUIRED_CHECKS,
    build_improvement_work_order,
    seal_sandbox_challenger,
)

LAYERS=("planning","memory","tool","knowledge","model","execution")


def _store(tmp_path):
    return KnowledgeFeedbackStore(tmp_path/"lifecycle.sqlite",LifecyclePolicy(
        approved_reviewers=("expert:dev54",),permitted_document_roots=(str(tmp_path),)))


def _candidate(store, layer: str, *, approve=True):
    fb=ExpertFeedback(feedback_id=f"feedback:dev54:{layer}",task_id=f"episode:dev54:{layer}",result_sha256="d"*64,
        feedback_type="DIAGNOSIS_CORRECTION",reviewer="expert:dev54",reviewed_at=datetime(2026,9,16,8,0,tzinfo=timezone.utc),
        truth_source_ref="truth:independent",verified_fault_id="FAULT_B",target_layers=(layer,),
        diagnosis_error_summary="wrong diagnosis",recommended_change=f"prepare {layer} challenger",
        evidence_refs=(f"EVID-{layer.upper()}-DEV54",))
    cid=store.submit_feedback(fb,event_id=f"submit:{layer}")["candidate_ids"][0]
    if approve:
        store.review_candidate(candidate_id=cid,decision="APPROVE_FOR_OFFLINE_EVALUATION",reviewer="expert:dev54",
                               reason="prepare offline challenger only",event_id=f"approve:{layer}")
    return fb,cid


def _baseline(tmp_path, layer):
    p=tmp_path/f"baseline-{layer}.json";p.write_text('{"version":1}',encoding="utf-8");return p


@pytest.mark.parametrize("layer",LAYERS)
def test_approved_candidate_builds_layer_specific_hash_pinned_work_order(tmp_path, layer) -> None:
    store=_store(tmp_path);fb,cid=_candidate(store,layer);base=_baseline(tmp_path,layer)
    kind=ALLOWED_CHANGE_KINDS[layer][0]
    order,path,digest=build_improvement_work_order(store=store,improvement_candidate_id=cid,sandbox_root=tmp_path/"sandbox",
                                                   change_kind=kind,baseline_artifacts={"baseline":base})
    assert path.is_file() and len(digest)==64
    assert order.target_layer==layer and order.change_kind==kind
    assert order.feedback_id==fb.feedback_id
    assert order.baseline_artifacts[0].sha256
    assert Path(order.challenger_root).is_relative_to((tmp_path/"sandbox").resolve())
    expected=list(MODEL_REQUIRED_CHECKS if layer=="model" else REQUIRED_CHECKS[layer])
    assert order.required_evaluation_checks==expected
    assert order.production_write_allowed is False and order.automatic_promotion_allowed is False


def test_pending_candidate_cannot_create_work_order(tmp_path) -> None:
    store=_store(tmp_path);_,cid=_candidate(store,"planning",approve=False);base=_baseline(tmp_path,"planning")
    with pytest.raises(PermissionError,match="not approved"):
        build_improvement_work_order(store=store,improvement_candidate_id=cid,sandbox_root=tmp_path/"sandbox",
                                     change_kind="PLANNER_POLICY",baseline_artifacts={"baseline":base})


def test_layer_disallows_wrong_change_kind(tmp_path) -> None:
    store=_store(tmp_path);_,cid=_candidate(store,"memory");base=_baseline(tmp_path,"memory")
    with pytest.raises(ValueError,match="not allowed"):
        build_improvement_work_order(store=store,improvement_candidate_id=cid,sandbox_root=tmp_path/"sandbox",
                                     change_kind="TOOL_SELECTION_POLICY",baseline_artifacts={"baseline":base})


def test_work_order_is_deterministic_for_same_candidate_and_baseline(tmp_path) -> None:
    store=_store(tmp_path);_,cid=_candidate(store,"knowledge");base=_baseline(tmp_path,"knowledge")
    a,pa,ha=build_improvement_work_order(store=store,improvement_candidate_id=cid,sandbox_root=tmp_path/"sandbox",
                                         change_kind="KNOWLEDGE_PROPOSAL",baseline_artifacts={"baseline":base})
    b,pb,hb=build_improvement_work_order(store=store,improvement_candidate_id=cid,sandbox_root=tmp_path/"sandbox",
                                         change_kind="KNOWLEDGE_PROPOSAL",baseline_artifacts={"baseline":base})
    assert a.work_order_id==b.work_order_id and pa==pb and ha==hb


def test_changed_challenger_inside_sandbox_can_be_sealed(tmp_path) -> None:
    store=_store(tmp_path);_,cid=_candidate(store,"execution");base=_baseline(tmp_path,"execution")
    order,path,digest=build_improvement_work_order(store=store,improvement_candidate_id=cid,sandbox_root=tmp_path/"sandbox",
                                                   change_kind="RUNTIME_POLICY",baseline_artifacts={"baseline":base})
    challenger=Path(order.challenger_root)/"runtime.json";challenger.write_text('{"version":2}',encoding="utf-8")
    seal,seal_path,seal_sha=seal_sandbox_challenger(work_order_path=path,expected_work_order_sha256=digest,
                                                    challenger_artifacts={"challenger":challenger})
    assert seal.ready_for_offline_evaluation is True and seal.changed_from_baseline is True
    assert seal.production_modified is False and seal.automatic_promotion_allowed is False
    assert seal_path.is_file() and len(seal_sha)==64


def test_challenger_outside_sandbox_is_rejected(tmp_path) -> None:
    store=_store(tmp_path);_,cid=_candidate(store,"tool");base=_baseline(tmp_path,"tool")
    order,path,digest=build_improvement_work_order(store=store,improvement_candidate_id=cid,sandbox_root=tmp_path/"sandbox",
                                                   change_kind="TOOL_SELECTION_POLICY",baseline_artifacts={"baseline":base})
    outside=tmp_path/"outside.json";outside.write_text('{"version":2}',encoding="utf-8")
    with pytest.raises(PermissionError,match="outside work-order sandbox"):
        seal_sandbox_challenger(work_order_path=path,expected_work_order_sha256=digest,challenger_artifacts={"bad":outside})


def test_work_order_hash_drift_is_rejected_before_sealing(tmp_path) -> None:
    store=_store(tmp_path);_,cid=_candidate(store,"planning");base=_baseline(tmp_path,"planning")
    order,path,digest=build_improvement_work_order(store=store,improvement_candidate_id=cid,sandbox_root=tmp_path/"sandbox",
                                                   change_kind="PLANNER_PROMPT",baseline_artifacts={"baseline":base})
    challenger=Path(order.challenger_root)/"prompt.txt";challenger.write_text('candidate',encoding="utf-8")
    path.write_text(path.read_text(encoding="utf-8")+'\n',encoding="utf-8")
    with pytest.raises(ValueError,match="work order SHA-256 mismatch"):
        seal_sandbox_challenger(work_order_path=path,expected_work_order_sha256=digest,challenger_artifacts={"prompt":challenger})


def test_unchanged_baseline_content_cannot_be_sealed_as_challenger(tmp_path) -> None:
    store=_store(tmp_path);_,cid=_candidate(store,"memory");base=_baseline(tmp_path,"memory")
    order,path,digest=build_improvement_work_order(store=store,improvement_candidate_id=cid,sandbox_root=tmp_path/"sandbox",
                                                   change_kind="MEMORY_CANDIDATE",baseline_artifacts={"baseline":base})
    challenger=Path(order.challenger_root)/"memory.json";challenger.write_bytes(base.read_bytes())
    with pytest.raises(ValueError,match="do not contain a content change"):
        seal_sandbox_challenger(work_order_path=path,expected_work_order_sha256=digest,challenger_artifacts={"memory":challenger})


def test_symlink_challenger_cannot_be_sealed(tmp_path) -> None:
    store=_store(tmp_path);_,cid=_candidate(store,"knowledge");base=_baseline(tmp_path,"knowledge")
    order,path,digest=build_improvement_work_order(store=store,improvement_candidate_id=cid,sandbox_root=tmp_path/"sandbox",
                                                   change_kind="ONTOLOGY_MAPPING_CANDIDATE",baseline_artifacts={"baseline":base})
    target=Path(order.challenger_root)/"target.json";target.write_text('{"version":2}',encoding="utf-8")
    link=Path(order.challenger_root)/"link.json"
    try: link.symlink_to(target)
    except (OSError,NotImplementedError): pytest.skip("symlink unavailable")
    with pytest.raises(ValueError,match="non-symlink"):
        seal_sandbox_challenger(work_order_path=path,expected_work_order_sha256=digest,challenger_artifacts={"link":link})
