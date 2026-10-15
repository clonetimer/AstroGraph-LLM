from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from astrograph.api import create_app
from astrograph.integrations.upload_report import UploadReportJobStore
import astrograph.upstream.manual_provider as manual


def _fixture_bytes() -> tuple[bytes, bytes]:
    root = Path(__file__).resolve().parent / "fixtures" / "sat_sim_closure"
    return (root / "runtime_rw_jam.json").read_bytes(), (root / "mapping_bundle.json").read_bytes()


def _put_pipeline(store: UploadReportJobStore, job_id: str) -> None:
    runtime, mapping = _fixture_bytes()
    store.put_file(job_id, "runtime.json", runtime)
    store.put_file(job_id, "mapping_bundle.json", mapping)


def _response_bundle(request: dict) -> bytes:
    runtime, mapping = _fixture_bytes()
    response = {
        "schema_version": "astrograph-upstream-recompute-response.v1",
        "request_id": request["request_id"],
        "request_sha256": request["request_sha256"],
        "provider_id": "manual-import",
        "provider_mode": "MANUAL_EXCHANGE",
        "runtime_filename": "runtime.json",
        "runtime_sha256": manual.sha256_bytes(runtime),
        "mapping_filename": "mapping.json",
        "mapping_sha256": manual.sha256_bytes(mapping),
        "source_raw_input_receipt_sha256": None,
        "claim_boundary": "test manual response",
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("response.json", json.dumps(response, sort_keys=True))
        archive.writestr("runtime.json", runtime)
        archive.writestr("mapping.json", mapping)
    return buf.getvalue()


def test_r13_store_manual_exchange_creates_pipeline_child_and_completion(tmp_path: Path) -> None:
    store = UploadReportJobStore(tmp_path / "jobs")
    source = store.create("r13 pipeline source")
    _put_pipeline(store, source["job_id"])
    source_report = store.run(source["job_id"])
    assert source_report["status"] == "SUCCEEDED"
    source_report_bytes = (store._job_root(source["job_id"]) / "output" / "report.json").read_bytes()

    view = store.maintenance_upstream_recompute_request(source["job_id"])
    assert view["manual_exchange_available"] is True
    request = view["request"]
    providers = {item["provider_id"]: item for item in view["providers"]["providers"]}
    assert providers["manual-import"]["status"] == "MANUAL"

    bundle_path = store.maintenance_upstream_recompute_request_bundle_path(source["job_id"])
    with zipfile.ZipFile(bundle_path) as archive:
        assert "request.json" in archive.namelist()

    created = store.create_manual_upstream_recompute_child(
        source["job_id"], response_bundle=_response_bundle(request), actor="tester", start=False
    )
    child_id = created["child_job_id"]
    assert created["preflight"]["artifact_kind"] == "PIPELINE_RUNTIME"
    child = store.get(child_id)
    assert child["upstream_recompute_provenance"]["raw_telemetry_recompute_verified"] is False
    assert child["upstream_recompute_provenance"]["formal_release_eligible"] is False

    report = store.run(child_id)
    assert report["status"] == "SUCCEEDED"
    assert report["artifact_kind"] == "PIPELINE_RUNTIME"
    assert (store._job_root(source["job_id"]) / "output" / "report.json").read_bytes() == source_report_bytes
    completion = json.loads(
        (store._job_root(child_id) / "output" / "upstream_recompute_completion_receipt.json").read_text(encoding="utf-8")
    )
    assert completion["status"] == "COMPLETED"
    assert completion["source_job_id"] == source["job_id"]
    assert completion["raw_telemetry_recompute_verified"] is False
    assert completion["formal_release_eligible"] is False
    history = store.maintenance_upstream_recompute_history(source["job_id"])
    assert history["items"][0]["child_job_id"] == child_id
    assert history["items"][0]["completion_receipt"]["receipt_sha256"] == completion["receipt_sha256"]


def test_r13_equivalent_manual_response_is_idempotent(tmp_path: Path) -> None:
    store = UploadReportJobStore(tmp_path / "jobs")
    source = store.create("r13 idempotent")
    _put_pipeline(store, source["job_id"])
    assert store.run(source["job_id"])["status"] == "SUCCEEDED"
    request = store.maintenance_upstream_recompute_request(source["job_id"])["request"]
    bundle = _response_bundle(request)
    first = store.create_manual_upstream_recompute_child(source["job_id"], response_bundle=bundle, start=False)
    with pytest.raises(ValueError, match="EQUIVALENT_MANUAL_UPSTREAM_RERUN_EXISTS"):
        store.create_manual_upstream_recompute_child(source["job_id"], response_bundle=bundle, start=False)
    failed = store.get(first["child_job_id"])
    failed["status"] = "FAILED"
    store._save(failed)
    retry = store.create_manual_upstream_recompute_child(source["job_id"], response_bundle=bundle, start=False)
    assert retry["child_job_id"] != first["child_job_id"]


def test_r13_completion_fails_when_source_upload_drifted(tmp_path: Path) -> None:
    store = UploadReportJobStore(tmp_path / "jobs")
    source = store.create("r13 drift")
    _put_pipeline(store, source["job_id"])
    assert store.run(source["job_id"])["status"] == "SUCCEEDED"
    request = store.maintenance_upstream_recompute_request(source["job_id"])["request"]
    created = store.create_manual_upstream_recompute_child(
        source["job_id"], response_bundle=_response_bundle(request), start=False
    )
    runtime_path = store._job_root(source["job_id"]) / "input" / "runtime.json"
    raw = runtime_path.read_bytes()
    replacement = bytearray(raw)
    replacement[-2] = ord(" ") if replacement[-2] != ord(" ") else ord("\t")
    runtime_path.write_bytes(bytes(replacement))
    result = store.run(created["child_job_id"])
    assert result["status"] == "FAILED"
    assert "sha256 drift" in result["error"]["message"]
    assert not (store._job_root(created["child_job_id"]) / "output" / "upstream_recompute_completion_receipt.json").exists()


def test_r13_api_supports_zero_dependency_manual_exchange(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASTROGRAPH_PLATFORM_ROOT", str(tmp_path / "platform"))
    monkeypatch.delenv("ASTROGRAPH_SAT_AGENT_RECOMPUTE_URL", raising=False)
    monkeypatch.delenv("ASTROGRAPH_MATLAB_RECOMPUTE_URL", raising=False)
    client = TestClient(create_app())
    providers = client.get("/v1/platform/upstream/providers")
    assert providers.status_code == 200
    by_id = {item["provider_id"]: item for item in providers.json()["providers"]}
    assert by_id["manual-import"]["status"] == "MANUAL"
    assert by_id["sat-agent-platform"]["status"] == "UNAVAILABLE"
    assert by_id["matlab-simulink"]["status"] == "UNAVAILABLE"

    source = client.post("/v1/platform/jobs", json={"task": "r13 manual api"}).json()
    runtime, mapping = _fixture_bytes()
    assert client.put(f"/v1/platform/jobs/{source['job_id']}/files/runtime.json", content=runtime).status_code == 200
    assert client.put(f"/v1/platform/jobs/{source['job_id']}/files/mapping_bundle.json", content=mapping).status_code == 200
    assert client.post(f"/v1/platform/jobs/{source['job_id']}/run").status_code == 200
    view = client.get(f"/v1/platform/jobs/{source['job_id']}/maintenance/upstream-recompute")
    assert view.status_code == 200
    request = view.json()["request"]
    download = client.get(f"/v1/platform/jobs/{source['job_id']}/maintenance/upstream-recompute/request-bundle")
    assert download.status_code == 200
    assert download.headers["content-type"].startswith("application/zip")
    created = client.post(
        f"/v1/platform/jobs/{source['job_id']}/maintenance/upstream-recompute/manual-response?start=false",
        content=_response_bundle(request),
        headers={"Content-Type": "application/zip"},
    )
    assert created.status_code == 200
    assert created.json()["child_job_id"].startswith("JOB-")
