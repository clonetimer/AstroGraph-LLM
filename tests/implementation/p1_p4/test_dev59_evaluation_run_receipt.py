from __future__ import annotations

import hashlib

import pytest

from astrograph.data_driven.feedback import TaskMetrics
from astrograph.lifecycle import (
    EvaluationRunReceipt, evaluate_approved_improvement_candidate,
    write_evaluation_run_receipt, model_metric_payload,
)
from tests.implementation.p1_p4.test_dev53_evolution_dispatcher import (
    _candidate, _sealed_model_dispatch_args, _sealed_non_model_kwargs, _store,
)


def test_model_dispatcher_result_contains_verified_evaluation_run_receipt(tmp_path) -> None:
    store=_store(tmp_path);fb,cid=_candidate(store,"model")
    packet=evaluate_approved_improvement_candidate(store=store,improvement_candidate_id=cid,
        **_sealed_model_dispatch_args(tmp_path,store,cid,fb.feedback_id))
    assert packet.evaluator_result["evaluation_run_id"].startswith("ERUN-")
    assert len(packet.evaluator_result["evaluation_run_receipt_sha256"])==64


def test_non_model_dispatcher_result_contains_verified_evaluation_run_receipt(tmp_path) -> None:
    store=_store(tmp_path);fb,cid=_candidate(store,"planning")
    packet=evaluate_approved_improvement_candidate(store=store,improvement_candidate_id=cid,
        **_sealed_non_model_kwargs(tmp_path,store,fb,cid,"planning"))
    assert packet.evaluator_result["evaluation_run_id"].startswith("ERUN-")
    assert len(packet.evaluator_result["evaluation_run_receipt_sha256"])==64


def test_model_metric_payload_change_after_receipt_is_rejected(tmp_path) -> None:
    store=_store(tmp_path);fb,cid=_candidate(store,"model")
    args=_sealed_model_dispatch_args(tmp_path,store,cid,fb.feedback_id)
    kw=dict(args["model_evaluation_kwargs"])
    challenger=kw["challenger"]
    kw["challenger"]=challenger.model_copy(update={"metrics":TaskMetrics(
        detection_macro_f1=.99,localization_macro_f1=.99,classification_macro_f1=.99,end_to_end_macro_f1=.99)})
    args["model_evaluation_kwargs"]=kw
    with pytest.raises(ValueError,match="result payload does not match"):
        evaluate_approved_improvement_candidate(store=store,improvement_candidate_id=cid,**args)


def test_non_model_check_payload_change_after_receipt_is_rejected(tmp_path) -> None:
    store=_store(tmp_path);fb,cid=_candidate(store,"tool")
    args=_sealed_non_model_kwargs(tmp_path,store,fb,cid,"tool")
    ep=args["non_model_evaluation_artifact_path"]
    import json
    payload=json.loads(ep.read_text(encoding="utf-8"));payload["checks"]["NO_PERMISSION_ESCALATION"]=False
    ep.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    args["non_model_expected_sha256"]=hashlib.sha256(ep.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match="result payload does not match|missing required source reference"):
        evaluate_approved_improvement_candidate(store=store,improvement_candidate_id=cid,**args)


def test_evaluation_run_receipt_hash_drift_fails_closed(tmp_path) -> None:
    store=_store(tmp_path);fb,cid=_candidate(store,"memory")
    args=_sealed_non_model_kwargs(tmp_path,store,fb,cid,"memory")
    rp=args["non_model_evaluation_receipt_path"]
    rp.write_text(rp.read_text(encoding="utf-8")+'\n',encoding="utf-8")
    with pytest.raises(ValueError,match="evaluation run receipt SHA-256 mismatch"):
        evaluate_approved_improvement_candidate(store=store,improvement_candidate_id=cid,**args)


def test_model_receipt_must_include_frozen_split_audit_ref(tmp_path) -> None:
    store=_store(tmp_path);fb,cid=_candidate(store,"model")
    args=_sealed_model_dispatch_args(tmp_path,store,cid,fb.feedback_id)
    old=EvaluationRunReceipt.model_validate_json(args["model_evaluation_receipt_path"].read_text(encoding="utf-8"))
    _,rp,rsha=write_evaluation_run_receipt(root=tmp_path/"bad-runs",improvement_candidate_id=cid,
        feedback_id=fb.feedback_id,target_layer="model",work_order_sha256=old.work_order_sha256,
        challenger_seal_sha256=old.challenger_seal_sha256,evaluator_id=old.evaluator_id,evaluator_version=old.evaluator_version,
        result_kind="MODEL_TASK_METRICS",result_payload=old.result_payload,source_refs=["split:other"])
    args["model_evaluation_receipt_path"]=rp;args["model_expected_evaluation_receipt_sha256"]=rsha
    with pytest.raises(ValueError,match="missing required source reference"):
        evaluate_approved_improvement_candidate(store=store,improvement_candidate_id=cid,**args)
