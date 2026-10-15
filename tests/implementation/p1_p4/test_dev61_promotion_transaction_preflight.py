from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from astrograph.lifecycle import (
    PromotionTransactionPreflight,
    build_promotion_review_packet,
    build_promotion_transaction_preflight,
    evaluate_approved_improvement_candidate,
    load_and_validate_promotion_transaction_preflight,
    validate_promotion_transaction_preflight,
    write_promotion_review_packet,
    write_promotion_transaction_preflight,
)
from tests.implementation.p1_p4.test_dev53_evolution_dispatcher import (
    _candidate, _sealed_model_dispatch_args, _sealed_non_model_kwargs, _store,
)


def _review_file(tmp_path, layer="model"):
    store = _store(tmp_path)
    fb, cid = _candidate(store, layer)
    if layer == "model":
        args = _sealed_model_dispatch_args(tmp_path, store, cid, fb.feedback_id)
        receipt_sha = args["model_expected_evaluation_receipt_sha256"]
    else:
        args = _sealed_non_model_kwargs(tmp_path, store, fb, cid, layer)
        receipt_sha = args["non_model_expected_evaluation_receipt_sha256"]
    review = evaluate_approved_improvement_candidate(store=store, improvement_candidate_id=cid, **args)
    packet = build_promotion_review_packet(review=review, expected_evaluation_run_receipt_sha256=receipt_sha)
    path, digest = write_promotion_review_packet(root=tmp_path/"review", packet=packet)
    return packet, path, digest


def test_promotion_preflight_freezes_current_production_baseline_without_writing(tmp_path) -> None:
    packet, path, digest = _review_file(tmp_path)
    root = tmp_path/"production"; root.mkdir()
    target = root/"active-model.json"; target.write_text('{"version":"v1"}', encoding="utf-8")
    before = target.read_bytes()
    preflight = build_promotion_transaction_preflight(
        promotion_review_packet_path=path,
        expected_promotion_review_packet_sha256=digest,
        production_root=root,
        write_targets={"active-model": target},
    )
    assert preflight.promotion_review_packet_id == packet.packet_id
    assert preflight.preflight_status == "READY_FOR_GOVERNED_TRANSACTION_REVIEW"
    assert preflight.rollback_baseline_refs == ("baseline:active-model",)
    assert target.read_bytes() == before
    assert preflight.production_modified is False and preflight.writes_applied is False


def test_preflight_rejects_production_target_outside_approved_root(tmp_path) -> None:
    _, path, digest = _review_file(tmp_path, "planning")
    root = tmp_path/"production"; root.mkdir()
    outside = tmp_path/"outside.json"; outside.write_text("{}", encoding="utf-8")
    with pytest.raises(PermissionError, match="escaped approved production root"):
        build_promotion_transaction_preflight(
            promotion_review_packet_path=path,
            expected_promotion_review_packet_sha256=digest,
            production_root=root,
            write_targets={"planner": outside},
        )


def test_preflight_detects_production_baseline_drift_before_any_write(tmp_path) -> None:
    _, path, digest = _review_file(tmp_path, "execution")
    root = tmp_path/"production"; root.mkdir()
    target = root/"runtime.json"; target.write_text('{"timeout":10}', encoding="utf-8")
    preflight = build_promotion_transaction_preflight(
        promotion_review_packet_path=path,
        expected_promotion_review_packet_sha256=digest,
        production_root=root,
        write_targets={"runtime": target},
    )
    target.write_text('{"timeout":20}', encoding="utf-8")
    with pytest.raises(ValueError, match="baseline changed"):
        validate_promotion_transaction_preflight(preflight)


def test_preflight_requires_exact_rollback_coverage(tmp_path) -> None:
    _, path, digest = _review_file(tmp_path, "memory")
    root = tmp_path/"production"; root.mkdir()
    target = root/"memory.json"; target.write_text("{}", encoding="utf-8")
    preflight = build_promotion_transaction_preflight(
        promotion_review_packet_path=path,
        expected_promotion_review_packet_sha256=digest,
        production_root=root,
        write_targets={"memory": target},
    )
    payload = preflight.model_dump(mode="json")
    payload["rollback_baseline_refs"] = []
    with pytest.raises(ValidationError, match="rollback coverage"):
        PromotionTransactionPreflight.model_validate(payload)


def test_persisted_preflight_hash_and_live_baseline_are_both_verified(tmp_path) -> None:
    _, path, digest = _review_file(tmp_path, "tool")
    root = tmp_path/"production"; root.mkdir()
    target = root/"tools.json"; target.write_text('{"allow":[]}', encoding="utf-8")
    preflight = build_promotion_transaction_preflight(
        promotion_review_packet_path=path,
        expected_promotion_review_packet_sha256=digest,
        production_root=root,
        write_targets={"tools": target},
    )
    pp, psha = write_promotion_transaction_preflight(root=tmp_path/"preflight", preflight=preflight)
    assert load_and_validate_promotion_transaction_preflight(path=pp, expected_sha256=psha).transaction_id == preflight.transaction_id
    pp.write_text(pp.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="preflight SHA-256 mismatch"):
        load_and_validate_promotion_transaction_preflight(path=pp, expected_sha256=psha)
