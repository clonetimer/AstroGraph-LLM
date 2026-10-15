from __future__ import annotations

import hashlib

import pytest

from astrograph.lifecycle import prepare_governed_evolution_rehearsal
from tests.implementation.p1_p4.test_dev53_evolution_dispatcher import (
    _candidate, _sealed_model_dispatch_args, _sealed_non_model_kwargs, _store,
)


@pytest.mark.parametrize("layer", ("planning", "memory", "tool", "knowledge", "model", "execution"))
def test_all_six_layers_rehearse_to_preflight_without_production_write(tmp_path, layer) -> None:
    case = tmp_path/layer; case.mkdir()
    store = _store(case)
    fb, cid = _candidate(store, layer)
    if layer == "model":
        kwargs = _sealed_model_dispatch_args(case, store, cid, fb.feedback_id)
    else:
        kwargs = _sealed_non_model_kwargs(case, store, fb, cid, layer)
    production = case/"production"; production.mkdir()
    target = production/f"active-{layer}.json"; target.write_text('{"version":1}', encoding="utf-8")
    before = target.read_bytes()
    receipt, path, digest = prepare_governed_evolution_rehearsal(
        store=store,
        improvement_candidate_id=cid,
        dispatcher_kwargs=kwargs,
        artifact_root=case/"artifacts",
        production_root=production,
        write_targets={f"active-{layer}": target},
    )
    assert receipt.target_layer == layer
    assert receipt.final_state == "READY_FOR_GOVERNED_PROMOTION_REVIEW"
    assert receipt.production_modified is False
    assert receipt.automatic_production_promotion is False
    assert receipt.requires_human_gate is True
    assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert target.read_bytes() == before


def test_rehearsal_stops_before_preflight_when_native_evaluator_rejects(tmp_path) -> None:
    store = _store(tmp_path)
    fb, cid = _candidate(store, "knowledge")
    kwargs = _sealed_non_model_kwargs(tmp_path, store, fb, cid, "knowledge", fail="GRAPH_REGRESSION")
    production = tmp_path/"production"; production.mkdir()
    target = production/"kg.json"; target.write_text("{}", encoding="utf-8")
    with pytest.raises(PermissionError, match="stopped because offline evaluation is not eligible"):
        prepare_governed_evolution_rehearsal(
            store=store, improvement_candidate_id=cid, dispatcher_kwargs=kwargs,
            artifact_root=tmp_path/"artifacts", production_root=production,
            write_targets={"active-kg": target},
        )
    assert not (tmp_path/"artifacts"/"promotion-preflight").exists()


def test_rehearsal_stops_on_pending_candidate_before_evaluator_runs(tmp_path) -> None:
    store = _store(tmp_path)
    _, cid = _candidate(store, "planning", approve=False)
    production = tmp_path/"production"; production.mkdir()
    target = production/"planner.json"; target.write_text("{}", encoding="utf-8")
    with pytest.raises(PermissionError, match="not approved for offline evaluation"):
        prepare_governed_evolution_rehearsal(
            store=store, improvement_candidate_id=cid, dispatcher_kwargs={}, artifact_root=tmp_path/"artifacts",
            production_root=production, write_targets={"planner":target},
        )


def test_rehearsal_receipt_binds_promotion_packet_and_preflight_hashes(tmp_path) -> None:
    store = _store(tmp_path)
    fb, cid = _candidate(store, "tool")
    kwargs = _sealed_non_model_kwargs(tmp_path, store, fb, cid, "tool")
    production=tmp_path/"production"; production.mkdir(); target=production/"tools.json"; target.write_text("{}",encoding="utf-8")
    receipt, _, _ = prepare_governed_evolution_rehearsal(
        store=store, improvement_candidate_id=cid, dispatcher_kwargs=kwargs,
        artifact_root=tmp_path/"artifacts", production_root=production, write_targets={"tools":target})
    packet_path=tmp_path/"artifacts"/"promotion-review"/f"{receipt.promotion_review_packet_id}.json"
    preflight_path=tmp_path/"artifacts"/"promotion-preflight"/f"{receipt.promotion_transaction_id}.json"
    assert hashlib.sha256(packet_path.read_bytes()).hexdigest()==receipt.promotion_review_packet_sha256
    assert hashlib.sha256(preflight_path.read_bytes()).hexdigest()==receipt.promotion_preflight_sha256
