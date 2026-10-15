from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from astrograph.upstream import (
    build_manual_request_bundle,
    build_manual_response_bundle,
    build_pipeline_runtime_recompute_request,
    provider_capabilities,
    verify_manual_response_bundle,
)
from astrograph.upstream.contracts import canonical_sha256
from astrograph.upstream.manual_provider import sha256_bytes


def _fixtures() -> tuple[bytes, bytes]:
    root = Path(__file__).resolve().parent / "fixtures" / "sat_sim_closure"
    return (root / "runtime_rw_jam.json").read_bytes(), (root / "mapping_bundle.json").read_bytes()


def _request():
    runtime, mapping = _fixtures()
    return build_pipeline_runtime_recompute_request(
        source_job_id="JOB-" + "A" * 24,
        source_runtime_sha256=sha256_bytes(runtime),
        source_mapping_bundle_sha256=sha256_bytes(mapping),
        maintenance_input_contract_ids=["EIC-2", "EIC-1", "EIC-1"],
    )


def _response_bundle(request, *, extra: bool = False, request_sha: str | None = None) -> bytes:
    runtime, mapping = _fixtures()
    response = {
        "schema_version": "astrograph-upstream-recompute-response.v1",
        "request_id": request.request_id,
        "request_sha256": request_sha or request.request_sha256,
        "provider_id": "manual-import",
        "provider_mode": "MANUAL_EXCHANGE",
        "runtime_filename": "runtime.json",
        "runtime_sha256": sha256_bytes(runtime),
        "mapping_filename": "mapping.json",
        "mapping_sha256": sha256_bytes(mapping),
        "source_raw_input_receipt_sha256": None,
        "claim_boundary": "test",
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("response.json", json.dumps(response, sort_keys=True))
        archive.writestr("runtime.json", runtime)
        archive.writestr("mapping.json", mapping)
        if extra:
            archive.writestr("untracked.txt", b"bad")
    return buf.getvalue()


def test_r13_optional_providers_never_disable_manual_core(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ASTROGRAPH_SAT_AGENT_RECOMPUTE_URL", raising=False)
    monkeypatch.delenv("ASTROGRAPH_MATLAB_RECOMPUTE_URL", raising=False)
    caps = provider_capabilities()
    assert caps["automatic_provider_required"] is False
    by_id = {item["provider_id"]: item for item in caps["providers"]}
    assert by_id["manual-import"]["status"] == "MANUAL"
    assert by_id["manual-import"]["configured"] is True
    assert by_id["sat-agent-platform"]["status"] == "UNAVAILABLE"
    assert by_id["matlab-simulink"]["status"] == "UNAVAILABLE"


def test_r13_request_identity_is_deterministic_and_contracts_are_sorted() -> None:
    a = _request()
    b = _request()
    assert a == b
    assert a.maintenance_input_contract_ids == ["EIC-1", "EIC-2"]
    assert a.derived_runtime_patch_allowed is False
    assert a.request_id.startswith("URR-")
    assert len(a.request_sha256) == 64


def test_r13_manual_request_bundle_is_self_contained() -> None:
    request = _request()
    data = build_manual_request_bundle(request)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert set(archive.namelist()) == {"request.json", "response_template.json", "README.txt"}
        payload = json.loads(archive.read("request.json"))
        assert payload["request_sha256"] == request.request_sha256
        assert json.loads(archive.read("response_template.json"))["request_id"] == request.request_id


def test_r13_manual_response_admission_validates_complete_runtime_mapping() -> None:
    request = _request()
    admitted = verify_manual_response_bundle(
        request=request,
        bundle_bytes=_response_bundle(request),
        admitted_at="2026-09-18T00:00:00+00:00",
        actor="tester",
    )
    assert admitted.admission_receipt["status"] == "ADMITTED"
    assert admitted.admission_receipt["cross_project_identity_verified"] is True
    assert admitted.admission_receipt["derived_runtime_patch_used"] is False
    assert admitted.admission_receipt["formal_release_eligible"] is False
    assert admitted.runtime["schema_version"] == "sat-sim.diagnostic-pipeline-runtime.v1"


def test_r13_manual_response_rejects_request_mismatch_and_extra_files() -> None:
    request = _request()
    with pytest.raises(ValueError, match="not bound"):
        verify_manual_response_bundle(
            request=request,
            bundle_bytes=_response_bundle(request, request_sha="0" * 64),
            admitted_at="2026-09-18T00:00:00+00:00",
            actor="tester",
        )
    with pytest.raises(ValueError, match="exactly"):
        verify_manual_response_bundle(
            request=request,
            bundle_bytes=_response_bundle(request, extra=True),
            admitted_at="2026-09-18T00:00:00+00:00",
            actor="tester",
        )


def test_r13_request_tamper_is_detected() -> None:
    request = _request().model_dump(mode="json")
    request["source_runtime_sha256"] = "0" * 64
    from astrograph.upstream.contracts import validate_upstream_request
    with pytest.raises(ValueError, match="request_sha256"):
        validate_upstream_request(request)


def test_r13_manual_response_packager_roundtrips_without_optional_provider_dependencies() -> None:
    request = _request()
    runtime, mapping = _fixtures()
    bundle = build_manual_response_bundle(
        request=request,
        runtime_bytes=runtime,
        mapping_bytes=mapping,
        source_raw_input_receipt_sha256="1" * 64,
    )
    admitted = verify_manual_response_bundle(
        request=request,
        bundle_bytes=bundle,
        admitted_at="2026-09-18T00:00:00+00:00",
        actor="tester",
    )
    assert admitted.response["source_raw_input_receipt_sha256"] == "1" * 64
    assert admitted.admission_receipt["source_raw_input_receipt_sha256"] == "1" * 64
    assert admitted.admission_receipt["status"] == "ADMITTED"


def test_r13_provider_capabilities_are_exposed_without_optional_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient
    from astrograph.api import create_app
    monkeypatch.delenv("ASTROGRAPH_SAT_AGENT_RECOMPUTE_URL", raising=False)
    monkeypatch.delenv("ASTROGRAPH_MATLAB_RECOMPUTE_URL", raising=False)
    client = TestClient(create_app())
    response = client.get("/v1/platform/upstream/providers")
    assert response.status_code == 200
    payload = response.json()
    assert payload["automatic_provider_required"] is False
    assert payload["default_provider_id"] == "manual-import"
