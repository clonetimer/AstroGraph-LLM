from __future__ import annotations

import pytest

from tests.implementation.p1_p4.test_dev53_evolution_dispatcher import (
    _candidate, _sealed_model_dispatch_args, _sealed_non_model_kwargs, _store,
)
from astrograph.lifecycle import evaluate_approved_improvement_candidate


def test_unsealed_non_model_shortcut_is_no_longer_available_from_unified_dispatcher(tmp_path) -> None:
    store=_store(tmp_path);_,cid=_candidate(store,"planning")
    with pytest.raises(ValueError,match="requires sealed work order"):
        evaluate_approved_improvement_candidate(store=store,improvement_candidate_id=cid,
                                                 non_model_evaluation_artifact_path=tmp_path/"legacy.json",
                                                 non_model_expected_sha256="0"*64)


def test_sealed_non_model_chain_is_exposed_in_dispatcher_native_result(tmp_path) -> None:
    store=_store(tmp_path);fb,cid=_candidate(store,"execution")
    kwargs=_sealed_non_model_kwargs(tmp_path,store,fb,cid,"execution")
    packet=evaluate_approved_improvement_candidate(store=store,improvement_candidate_id=cid,**kwargs)
    native=packet.evaluator_result
    assert native["schema_version"]=="astrograph-sealed-layer-evaluation-qualification.v1"
    assert len(native["work_order_sha256"])==64 and len(native["challenger_seal_sha256"])==64
    assert packet.review_status=="ELIGIBLE_FOR_HUMAN_REVIEW"


def test_model_dispatcher_route_remains_unchanged_by_sealed_non_model_upgrade(tmp_path) -> None:
    store=_store(tmp_path);fb,cid=_candidate(store,"model")
    packet=evaluate_approved_improvement_candidate(store=store,improvement_candidate_id=cid,
                                                    **_sealed_model_dispatch_args(tmp_path,store,cid,fb.feedback_id))
    assert packet.evaluator_route=="MODEL_CHALLENGER"
    assert packet.native_evaluator_status=="PENDING_HUMAN_GATE"
