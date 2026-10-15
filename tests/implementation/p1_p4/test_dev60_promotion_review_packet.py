from __future__ import annotations

import hashlib
import json

import pytest

from astrograph.agents.mission.storage import sha
from astrograph.lifecycle import (
    build_promotion_review_packet,
    evaluate_approved_improvement_candidate,
    load_and_validate_promotion_review_packet,
    write_promotion_review_packet,
)
from tests.implementation.p1_p4.test_dev53_evolution_dispatcher import (
    _candidate,
    _sealed_model_dispatch_args,
    _sealed_non_model_kwargs,
    _store,
)


def _eligible_model_review(tmp_path):
    store = _store(tmp_path)
    fb, cid = _candidate(store, "model")
    args = _sealed_model_dispatch_args(tmp_path, store, cid, fb.feedback_id)
    review = evaluate_approved_improvement_candidate(store=store, improvement_candidate_id=cid, **args)
    return fb, cid, args, review


def test_model_promotion_review_packet_binds_exact_evidence_lineage(tmp_path) -> None:
    fb, cid, args, review = _eligible_model_review(tmp_path)
    receipt_sha = args["model_expected_evaluation_receipt_sha256"]
    packet = build_promotion_review_packet(review=review, expected_evaluation_run_receipt_sha256=receipt_sha)
    assert packet.improvement_candidate_id == cid
    assert packet.feedback_id == fb.feedback_id
    assert packet.target_layer == "model"
    assert packet.evaluation_run_receipt_sha256 == receipt_sha
    assert packet.work_order_sha256 == args["model_expected_work_order_sha256"]
    assert packet.challenger_seal_sha256 == args["model_expected_challenger_seal_sha256"]
    assert packet.requires_human_gate is True
    assert packet.automatic_production_promotion is False
    assert packet.production_modified is False


def test_non_model_promotion_review_packet_uses_same_contract(tmp_path) -> None:
    store = _store(tmp_path)
    fb, cid = _candidate(store, "planning")
    args = _sealed_non_model_kwargs(tmp_path, store, fb, cid, "planning")
    review = evaluate_approved_improvement_candidate(store=store, improvement_candidate_id=cid, **args)
    packet = build_promotion_review_packet(
        review=review,
        expected_evaluation_run_receipt_sha256=args["non_model_expected_evaluation_receipt_sha256"],
    )
    assert packet.target_layer == "planning"
    assert packet.work_order_sha256 == args["non_model_expected_work_order_sha256"]
    assert packet.challenger_seal_sha256 == args["non_model_expected_challenger_seal_sha256"]
    assert packet.review_state == "PENDING_GOVERNED_PROMOTION_REVIEW"


def test_rejected_candidate_cannot_create_promotion_review_packet(tmp_path) -> None:
    store = _store(tmp_path)
    fb, cid = _candidate(store, "knowledge")
    args = _sealed_non_model_kwargs(tmp_path, store, fb, cid, "knowledge", fail="GRAPH_REGRESSION")
    review = evaluate_approved_improvement_candidate(store=store, improvement_candidate_id=cid, **args)
    assert review.review_status == "REJECTED"
    with pytest.raises(PermissionError, match="human-review-eligible"):
        build_promotion_review_packet(
            review=review,
            expected_evaluation_run_receipt_sha256=args["non_model_expected_evaluation_receipt_sha256"],
        )


def test_promotion_review_packet_receipt_substitution_fails_closed(tmp_path) -> None:
    _, _, args, review = _eligible_model_review(tmp_path)
    with pytest.raises(ValueError, match="evaluation-run receipt binding mismatch"):
        build_promotion_review_packet(review=review, expected_evaluation_run_receipt_sha256="0" * 64)


def test_persisted_promotion_review_packet_is_content_addressed_and_tamper_detected(tmp_path) -> None:
    fb, cid, args, review = _eligible_model_review(tmp_path)
    packet = build_promotion_review_packet(
        review=review,
        expected_evaluation_run_receipt_sha256=args["model_expected_evaluation_receipt_sha256"],
    )
    path, digest = write_promotion_review_packet(root=tmp_path / "promotion-review", packet=packet)
    loaded = load_and_validate_promotion_review_packet(
        path=path,
        expected_sha256=digest,
        expected_improvement_candidate_id=cid,
        expected_feedback_id=fb.feedback_id,
        expected_target_layer="model",
        expected_governed_review_packet_sha256=sha(review.model_dump(mode="json")),
        expected_evaluation_run_receipt_sha256=args["model_expected_evaluation_receipt_sha256"],
    )
    assert loaded.packet_id == packet.packet_id
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["upgrade_proposal"]["tampered"] = True
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_and_validate_promotion_review_packet(
            path=path,
            expected_sha256=digest,
            expected_improvement_candidate_id=cid,
            expected_feedback_id=fb.feedback_id,
            expected_target_layer="model",
            expected_governed_review_packet_sha256=sha(review.model_dump(mode="json")),
            expected_evaluation_run_receipt_sha256=args["model_expected_evaluation_receipt_sha256"],
        )
