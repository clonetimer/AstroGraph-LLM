from __future__ import annotations

import pytest

from astrograph.lifecycle import evaluate_approved_improvement_candidate
from tests.implementation.p1_p4.test_dev53_evolution_dispatcher import (
    _candidate, _sealed_model_dispatch_args, _sealed_non_model_kwargs, _store,
)


def test_model_route_requires_sealed_work_order_and_challenger_seal(tmp_path) -> None:
    store=_store(tmp_path);fb,cid=_candidate(store,"model")
    args=_sealed_model_dispatch_args(tmp_path,store,cid,fb.feedback_id)
    model_kwargs=args.pop("model_evaluation_kwargs")
    with pytest.raises(ValueError,match="requires sealed work order, challenger seal, and evaluation run receipt"):
        evaluate_approved_improvement_candidate(store=store,improvement_candidate_id=cid,
                                                model_evaluation_kwargs=model_kwargs)


def test_model_route_preserves_sealed_artifact_lineage_in_native_result(tmp_path) -> None:
    store=_store(tmp_path);fb,cid=_candidate(store,"model")
    packet=evaluate_approved_improvement_candidate(store=store,improvement_candidate_id=cid,
                                                    **_sealed_model_dispatch_args(tmp_path,store,cid,fb.feedback_id))
    assert packet.evaluator_route=="MODEL_CHALLENGER"
    assert packet.evaluator_result["artifact_lineage_verified"] is True
    assert len(packet.evaluator_result["work_order_sha256"])==64
    assert len(packet.evaluator_result["challenger_seal_sha256"])==64
    assert packet.review_status=="ELIGIBLE_FOR_HUMAN_REVIEW"


def test_non_model_route_remains_fully_sealed(tmp_path) -> None:
    store=_store(tmp_path);fb,cid=_candidate(store,"memory")
    packet=evaluate_approved_improvement_candidate(store=store,improvement_candidate_id=cid,
        **_sealed_non_model_kwargs(tmp_path,store,fb,cid,"memory"))
    assert packet.evaluator_route=="LAYER_SPECIFIC_ARTIFACT"
    assert packet.evaluator_result["schema_version"]=="astrograph-sealed-layer-evaluation-qualification.v1"


def test_model_seal_arguments_are_rejected_for_non_model_candidate(tmp_path) -> None:
    store=_store(tmp_path);fb,cid=_candidate(store,"tool")
    with pytest.raises(ValueError,match="model sealed evaluation arguments are invalid"):
        evaluate_approved_improvement_candidate(store=store,improvement_candidate_id=cid,
                                                model_work_order_path=tmp_path/"dummy")
