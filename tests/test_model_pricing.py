from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from astrograph.api import create_app
from astrograph.integrations.execution_observability import RunJournal, execution_snapshot, run_scope
from astrograph.integrations.model_pricing import ModelPriceBook, calculate_cost
from astrograph.integrations.stage_execution import for_stage


def effective(days=-1):
    return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()


def add_price(book, *, input_rate="2.0", output_rate="4.0", effective_from=None, source_ref="https://provider.example/pricing"):
    return book.add(
        provider="test-provider", model="test-model", currency="USD",
        input_per_million_tokens=input_rate, output_per_million_tokens=output_rate,
        effective_from=effective_from or effective(), source_type="OFFICIAL_PUBLISHED",
        source_ref=source_ref, created_by="admin",
    )


def test_pricebook_resolves_effective_version_and_is_append_only(tmp_path):
    book = ModelPriceBook(tmp_path)
    old_effective = effective(-10)
    new_effective = effective(1)
    old = add_price(book, input_rate="2", output_rate="4", effective_from=old_effective)
    new = add_price(book, input_rate="3", output_rate="6", effective_from=new_effective,
                    source_ref="https://provider.example/pricing-next")
    before_new = book.resolve(provider="test-provider", model="test-model", at=effective(-1))
    after_new = book.resolve(provider="test-provider", model="test-model", at=effective(2))
    assert before_new["price_id"] == old["price_id"]
    assert after_new["price_id"] == new["price_id"]
    assert [x["revision"] for x in book.list()] == [2, 1]
    with pytest.raises(ValueError, match="PRICE_VERSION_ALREADY_EXISTS"):
        add_price(book, effective_from=old_effective)
    with sqlite3.connect(book.path) as db:
        with pytest.raises(sqlite3.IntegrityError, match="APPEND_ONLY"):
            db.execute("UPDATE model_price_versions SET input_per_million_tokens='0' WHERE price_id=?",
                       (old["price_id"],))
        with pytest.raises(sqlite3.IntegrityError, match="APPEND_ONLY"):
            db.execute("DELETE FROM model_price_versions WHERE price_id=?", (old["price_id"],))


def test_calculate_cost_requires_price_and_complete_usage():
    assert calculate_cost(None, {"input_tokens": 10, "output_tokens": 5})["status"] == "PRICING_NOT_CONFIGURED"
    pricing = {
        "status": "PRICE_PINNED", "currency": "USD",
        "input_per_million_tokens": "2", "output_per_million_tokens": "4",
    }
    assert calculate_cost(pricing, {"input_tokens": None, "output_tokens": 5})["status"] == "USAGE_MISSING"
    cost = calculate_cost(pricing, {"input_tokens": 1_000_000, "output_tokens": 500_000})
    assert cost == {"status": "CALCULATED", "currency": "USD", "amount": "4.000000000000"}


class FakeTransportBackend:
    model_id = "test-model"
    config = SimpleNamespace(provider="test-provider")

    def __init__(self, invoke):
        self.invoke = invoke

    def generate_structured(self, **kwargs):
        return self.invoke()


def test_run_journal_pins_price_and_reports_per_request_and_aggregate_cost(tmp_path):
    platform_root = tmp_path / "platform"
    job_root = tmp_path / "job"
    book = ModelPriceBook(platform_root)
    price = add_price(book)
    bundle = {"task_id": "JOB-PRICE", "input_sha256": "a" * 64, "sources": []}
    run = RunJournal(job_root, task_id="JOB-PRICE", kind="analysis", bundle=bundle,
                     model="test-model", pricing_root=platform_root)

    def invoke():
        return run.http_call(
            lambda observer: {"usage": {"prompt_tokens": 1_000_000, "completion_tokens": 500_000}},
            request_sha256="b" * 64,
        )

    with run_scope(run):
        for_stage(FakeTransportBackend(invoke), "analysis").generate_structured()
    run.finish("SUCCEEDED")

    snapshot = execution_snapshot(job_root, "JOB-PRICE")
    request = snapshot["runs"][0]["requests"][0]
    assert request["provider"] == "test-provider"
    assert request["model"] == "test-model"
    assert request["pricing"]["price_id"] == price["price_id"]
    assert request["cost"] == {"status": "CALCULATED", "currency": "USD", "amount": "4.000000000000"}
    assert snapshot["usage"]["cost_status"] == "CALCULATED"
    assert snapshot["usage"]["estimated_cost"] == "4.000000000000"
    assert snapshot["usage"]["cost_currency"] == "USD"


def test_run_journal_marks_failed_transport_as_possible_unmetered_billing(tmp_path):
    platform_root = tmp_path / "platform"
    job_root = tmp_path / "job"
    add_price(ModelPriceBook(platform_root))
    bundle = {"task_id": "JOB-FAIL", "input_sha256": "c" * 64, "sources": []}
    run = RunJournal(job_root, task_id="JOB-FAIL", kind="analysis", bundle=bundle,
                     model="test-model", pricing_root=platform_root)

    def invoke():
        return run.http_call(
            lambda observer: (_ for _ in ()).throw(TimeoutError("transport timeout")),
            request_sha256="d" * 64,
        )

    with run_scope(run):
        with pytest.raises(TimeoutError):
            for_stage(FakeTransportBackend(invoke), "analysis").generate_structured()
    run.finish("FAILED", "MODEL_GENERATION_FAILED")
    snapshot = execution_snapshot(job_root, "JOB-FAIL")
    assert snapshot["usage"]["cost_status"] == "POSSIBLE_UNMETERED_BILLING"
    assert snapshot["usage"]["estimated_cost"] is None
    assert snapshot["usage"]["possible_unmetered_requests"] == 1


def test_run_without_pricebook_never_assumes_zero_cost(tmp_path):
    bundle = {"task_id": "JOB-NOPRICE", "input_sha256": "e" * 64, "sources": []}
    run = RunJournal(tmp_path / "job", task_id="JOB-NOPRICE", kind="analysis",
                     bundle=bundle, model="unpriced-model", pricing_root=tmp_path / "platform")

    def invoke():
        return run.http_call(
            lambda observer: {"usage": {"prompt_tokens": 20, "completion_tokens": 10}},
            request_sha256="f" * 64,
        )

    with run_scope(run):
        for_stage(FakeTransportBackend(invoke), "analysis").generate_structured()
    run.finish("SUCCEEDED")
    usage = execution_snapshot(tmp_path / "job", "JOB-NOPRICE")["usage"]
    assert usage["cost_status"] == "PRICING_NOT_CONFIGURED"
    assert usage["estimated_cost"] is None


def test_model_pricing_api_is_admin_scoped_and_append_only(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTROGRAPH_AUTH_MODE", "local")
    monkeypatch.setenv("ASTROGRAPH_PLATFORM_ROOT", str(tmp_path / "api-platform"))
    with TestClient(create_app(), base_url="http://localhost") as client:
        endpoint = "/v1/platform/settings/model-pricing"
        empty = client.get(endpoint)
        assert empty.status_code == 200
        assert empty.json()["prices"] == []
        saved = client.post(endpoint, json={
            "provider": "test-provider", "model": "test-model", "currency": "USD",
            "input_per_million_tokens": "1.25", "output_per_million_tokens": "5.5",
            "effective_from": effective(), "source_type": "ADMIN_ESTIMATE",
            "source_ref": "Internal approved pricing memo 2026-11",
        })
        assert saved.status_code == 200, saved.text
        assert saved.json()["price"]["revision"] == 1
        listed = client.get(endpoint).json()
        assert listed["prices"][0]["created_by"] == "local:testclient"
        assert listed["prices"][0]["input_per_million_tokens"] == "1.25"
