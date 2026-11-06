from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from astrograph.api import PlatformLLMSettingsRequest
from astrograph.integrations.execution_observability import (
    RunJournal, execution_snapshot, run_scope,
)
from astrograph.integrations.stage_execution import for_stage
from astrograph.integrations.upload_report import UploadReportJobStore


def bundle(task_id: str) -> dict:
    return {"task_id": task_id, "input_sha256": "a" * 64, "sources": []}


def pricing(provider="test-provider", model="test-model", *, version="price-2026-11"):
    return {
        "status": "CONFIGURED", "provider": provider, "model": model,
        "pricing_version": version, "currency": "USD",
        "input_usd_per_million_tokens": 2.0,
        "output_usd_per_million_tokens": 10.0,
    }


def test_actual_usage_is_costed_against_frozen_versioned_price(tmp_path):
    task_id = "JOB-" + "A" * 24
    run = RunJournal(
        tmp_path, task_id=task_id, kind="analysis", bundle=bundle(task_id),
        model="test-model", provider="test-provider",
        pricing_snapshots={"analysis": pricing()},
    )
    with run_scope(run):
        run.http_call(
            lambda observer: {"usage": {"prompt_tokens": 100, "completion_tokens": 50}},
            request_sha256="b" * 64,
        )
        run.finish("SUCCEEDED")

    snapshot = execution_snapshot(tmp_path, task_id)
    request = snapshot["runs"][0]["requests"][0]
    assert request["pricing"]["pricing_version"] == "price-2026-11"
    assert request["cost"] == {
        "status": "CALCULATED", "currency": "USD", "estimated_cost": "0.00070000",
    }
    assert snapshot["usage"]["estimated_cost"] == "0.00070000"
    assert snapshot["usage"]["cost_status"] == "CALCULATED"


def test_verification_stage_uses_its_own_model_and_price_snapshot(tmp_path):
    task_id = "JOB-" + "B" * 24
    run = RunJournal(
        tmp_path, task_id=task_id, kind="analysis", bundle=bundle(task_id),
        model="analysis-model", provider="analysis-provider",
        pricing_snapshots={
            "analysis": pricing("analysis-provider", "analysis-model", version="analysis-v1"),
            "verification": pricing("review-provider", "review-model", version="review-v2"),
        },
    )

    class ReviewBackend:
        model_id = "review-model"
        config = SimpleNamespace(provider="review-provider")

        def generate_structured(self, **kwargs):
            return run.http_call(
                lambda observer: {"usage": {"prompt_tokens": 200, "completion_tokens": 100}},
                request_sha256="c" * 64,
            )

    with run_scope(run):
        run.event("INDEPENDENT_VERIFICATION")
        for_stage(ReviewBackend(), "verification").generate_structured()
        run.finish("SUCCEEDED")

    request = execution_snapshot(tmp_path, task_id)["runs"][0]["requests"][0]
    assert request["stage"] == "INDEPENDENT_VERIFICATION"
    assert request["model"] == "review-model"
    assert request["provider"] == "review-provider"
    assert request["pricing"]["pricing_version"] == "review-v2"
    assert request["cost"]["estimated_cost"] == "0.00140000"


def test_missing_price_or_usage_is_never_reported_as_zero_cost(tmp_path):
    task_id = "JOB-" + "C" * 24
    run = RunJournal(
        tmp_path, task_id=task_id, kind="analysis", bundle=bundle(task_id),
        model="test-model", provider="test-provider",
        pricing_snapshots={"analysis": {"status": "PRICING_NOT_CONFIGURED"}},
    )
    with run_scope(run):
        run.http_call(lambda observer: {"usage": {"prompt_tokens": 100, "completion_tokens": 50}},
                      request_sha256="d" * 64)
        run.finish("SUCCEEDED")
    snapshot = execution_snapshot(tmp_path, task_id)
    assert snapshot["usage"]["estimated_cost"] is None
    assert snapshot["usage"]["cost_status"] == "PRICING_NOT_CONFIGURED"

    task_id_2 = "JOB-" + "D" * 24
    run_2 = RunJournal(
        tmp_path, task_id=task_id_2, kind="analysis", bundle=bundle(task_id_2),
        model="test-model", provider="test-provider",
        pricing_snapshots={"analysis": pricing()},
    )
    with run_scope(run_2):
        run_2.http_call(lambda observer: {"choices": []}, request_sha256="e" * 64)
        run_2.finish("SUCCEEDED")
    snapshot_2 = execution_snapshot(tmp_path, task_id_2)
    assert snapshot_2["runs"][0]["requests"][0]["cost"]["status"] == "USAGE_NOT_REPORTED"
    assert snapshot_2["usage"]["estimated_cost"] is None


def test_provider_or_model_mismatch_blocks_pricing_estimate(tmp_path):
    task_id = "JOB-" + "E" * 24
    run = RunJournal(
        tmp_path, task_id=task_id, kind="analysis", bundle=bundle(task_id),
        model="other-model", provider="test-provider",
        pricing_snapshots={"analysis": pricing()},
    )
    with run_scope(run):
        run.http_call(lambda observer: {"usage": {"prompt_tokens": 10, "completion_tokens": 10}},
                      request_sha256="f" * 64)
        run.finish("SUCCEEDED")
    snapshot = execution_snapshot(tmp_path, task_id)
    assert snapshot["runs"][0]["requests"][0]["cost"]["status"] == "PRICING_IDENTITY_MISMATCH"
    assert snapshot["usage"]["estimated_cost"] is None


def test_failed_paid_request_cost_remains_unknown_not_zero(tmp_path):
    task_id = "JOB-" + "F" * 24
    run = RunJournal(
        tmp_path, task_id=task_id, kind="analysis", bundle=bundle(task_id),
        model="test-model", provider="test-provider",
        pricing_snapshots={"analysis": pricing()},
    )
    with run_scope(run):
        with pytest.raises(RuntimeError):
            run.http_call(
                lambda observer: (_ for _ in ()).throw(RuntimeError("provider disconnected")),
                request_sha256="1" * 64,
            )
    snapshot = execution_snapshot(tmp_path, task_id)
    request = snapshot["runs"][0]["requests"][0]
    assert request["status"] == "FAILED"
    assert request["cost"]["status"] == "USAGE_UNKNOWN_AFTER_FAILURE"
    assert request["cost"]["estimated_cost"] is None
    assert snapshot["usage"]["estimated_cost"] is None


def test_job_freezes_analysis_and_verification_pricing_snapshots(tmp_path):
    store = UploadReportJobStore(tmp_path / "jobs")

    def save_profile(name, version, input_rate, output_rate):
        return store.state_repository.save_service_profile(
            service_kind="llm", name=name, validation_status="PASS",
            config={
                "enabled": True, "base_url": "http://127.0.0.1:1234/v1",
                "model": name, "provider": "lmstudio",
                "pricing_version": version, "pricing_currency": "USD",
                "input_usd_per_million_tokens": input_rate,
                "output_usd_per_million_tokens": output_rate,
            },
        )

    primary = save_profile("analysis-model", "analysis-v1", 1.0, 3.0)
    verifier = save_profile("review-model", "review-v1", 2.0, 8.0)
    job = store.create(
        "冻结模型计价",
        analysis_engine_id="llm:" + primary["profile_id"],
        phase_engine_ids={"verification": "llm:" + verifier["profile_id"]},
    )
    frozen = store.get(job["job_id"])["model_pricing_snapshots"]
    assert frozen["analysis"]["pricing_version"] == "analysis-v1"
    assert frozen["verification"]["pricing_version"] == "review-v1"
    assert frozen["verification"]["input_usd_per_million_tokens"] == 2.0

    store.state_repository.save_service_profile(
        service_kind="llm", name="review-model", profile_id=verifier["profile_id"],
        validation_status="PASS",
        config={
            "enabled": True, "base_url": "http://127.0.0.1:1234/v1",
            "model": "review-model", "provider": "lmstudio",
            "pricing_version": "review-v2", "pricing_currency": "USD",
            "input_usd_per_million_tokens": 99.0,
            "output_usd_per_million_tokens": 99.0,
        },
    )
    assert store.get(job["job_id"])["model_pricing_snapshots"]["verification"]["pricing_version"] == "review-v1"


def test_llm_pricing_configuration_requires_complete_versioned_rates():
    base = {
        "enabled": True, "base_url": "http://127.0.0.1:1234/v1",
        "model": "test-model", "provider": "lmstudio",
    }
    configured = PlatformLLMSettingsRequest.model_validate({
        **base, "pricing_version": "test-v1",
        "input_usd_per_million_tokens": 0, "output_usd_per_million_tokens": 0,
    })
    assert configured.pricing_version == "test-v1"
    with pytest.raises(ValidationError):
        PlatformLLMSettingsRequest.model_validate({
            **base, "pricing_version": "partial-v1", "input_usd_per_million_tokens": 1,
        })
    with pytest.raises(ValidationError):
        PlatformLLMSettingsRequest.model_validate({
            **base, "pricing_version": "bad-v1",
            "input_usd_per_million_tokens": -1, "output_usd_per_million_tokens": 1,
        })



def test_pricing_profile_roundtrip_is_versioned_and_never_exposes_secret(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import astrograph.integrations.platform_llm_probe as probe
    from astrograph.api import create_app

    monkeypatch.setenv("ASTROGRAPH_AUTH_MODE", "local")
    monkeypatch.setenv("ASTROGRAPH_PLATFORM_ROOT", str(tmp_path / "profile-api"))
    for key in (
        "ASTROGRAPH_PLATFORM_LLM_BASE_URL", "ASTROGRAPH_PLATFORM_LLM_MODEL",
        "ASTROGRAPH_PLATFORM_LLM_PROVIDER", "ASTROGRAPH_PLATFORM_LLM_ACTIVE_PROFILE_ID",
        "ASTROGRAPH_LLM_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(probe, "probe_openai_compatible_service", lambda **kwargs: {
        "status": "READY", "model_listed": True,
    })

    with TestClient(create_app()) as client:
        saved = client.post("/v1/platform/settings/llm/profiles", json={
            "name": "priced-test-model", "make_default": True, "activate": True,
            "settings": {
                "enabled": True, "provider": "vllm",
                "base_url": "http://127.0.0.1:9000/v1", "model": "priced-model",
                "timeout_s": 90, "response_format_strategy": "json_object",
                "structured_max_tokens": 512, "pricing_version": "provider-price-2026-11",
                "pricing_currency": "USD", "input_usd_per_million_tokens": 2.0,
                "output_usd_per_million_tokens": 10.0,
            },
        })
        assert saved.status_code == 200, saved.text
        assert "secret_ref" not in saved.json()["profile"]
        listed = client.get("/v1/platform/settings/llm/profiles")
        assert listed.status_code == 200
        profile = listed.json()["profiles"][0]
        assert profile["config"]["pricing_version"] == "provider-price-2026-11"
        assert profile["config"]["input_usd_per_million_tokens"] == 2.0
        assert "api_key" not in profile["config"]

        runtime = client.get("/v1/platform/settings/llm")
        assert runtime.status_code == 200
        assert runtime.json()["pricing_configured"] is True
        assert runtime.json()["pricing_version"] == "provider-price-2026-11"
        assert runtime.json()["output_usd_per_million_tokens"] == 10.0
        assert "api_key" not in runtime.json()
