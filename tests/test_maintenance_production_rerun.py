from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from astrograph.api import create_app
from astrograph.integrations.maintenance_production_rerun import (
    RAW_BINDING_SCHEMA_VERSION,
    ProductionRawInputPatchCapsule,
    admit_public_case_raw_patch,
    assess_maintenance_production_rerun,
    build_public_case_raw_input_contracts,
    build_public_case_raw_patch_template,
    materialize_public_case_raw_patch,
    verify_production_rerun_completion_receipt,
)
from astrograph.integrations.sat_sim_public_case import _load_public_case
from astrograph.integrations.upload_report import UploadReportJobStore


def _public_case_files(*, with_binding: bool = True) -> dict[str, bytes]:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["time_s", "eps.battery.soc"], lineterminator="\n")
    writer.writeheader()
    writer.writerows([
        {"time_s": "0", "eps.battery.soc": "0.8"},
        {"time_s": "1", "eps.battery.soc": "0.7"},
        {"time_s": "2", "eps.battery.soc": "0.6"},
    ])
    telemetry = buffer.getvalue().encode("utf-8")
    digest = hashlib.sha256(telemetry).hexdigest()
    manifest = {
        "schema_version": "sat-sim.astrograph-public-case.v1",
        "case_id": "r12-public-001",
        "telemetry": {
            "file": "telemetry.csv",
            "sha256": digest,
            "rows": 3,
            "fields": ["time_s", "eps.battery.soc"],
            "units": {"eps.battery.soc": "1"},
        },
    }
    if with_binding:
        manifest["maintenance_raw_input_binding"] = {
            "schema_version": RAW_BINDING_SCHEMA_VERSION,
            "source_telemetry_sha256": digest,
            "time_field": "time_s",
            "full_window_replacement_required": True,
            "patchable_fields": [{"field": "eps.battery.soc", "unit": "1"}],
        }
    return {
        "manifest.json": (json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"),
        "telemetry.csv": telemetry,
    }


def _write_case(root: Path, files: dict[str, bytes]) -> None:
    root.mkdir(parents=True)
    for name, data in files.items():
        (root / name).write_bytes(data)


def _put_all(store: UploadReportJobStore, job_id: str, files: dict[str, bytes]) -> None:
    for name, data in files.items():
        store.put_file(job_id, name, data)


def test_r12_public_case_raw_contract_requires_explicit_binding(tmp_path: Path) -> None:
    root = tmp_path / "case"
    _write_case(root, _public_case_files(with_binding=False))
    result = build_public_case_raw_input_contracts(root)
    assert result["status"] == "BLOCKED"
    assert "RAW_INPUT_BINDING_UNAVAILABLE" in result["blockers"]


def test_r12_public_case_raw_patch_materializes_new_valid_case(tmp_path: Path) -> None:
    root = tmp_path / "case"
    _write_case(root, _public_case_files())
    contracts = build_public_case_raw_input_contracts(root)
    assert contracts["status"] == "AVAILABLE"
    contract = contracts["contracts"][0]
    template = build_public_case_raw_patch_template(contract, root)
    capsule_payload = template["capsule"]
    capsule_payload["values"] = [0.9, 0.85, 0.8]
    capsule = ProductionRawInputPatchCapsule.model_validate(capsule_payload)
    receipt = admit_public_case_raw_patch(
        case_root=root, contract=contract, capsule=capsule, actor="tester", admitted_at="2026-09-18T00:00:00+00:00"
    )
    patch = materialize_public_case_raw_patch(
        case_root=root,
        admissions=[{"receipt": receipt, "capsule": capsule.model_dump(mode="json"), "contract": contract}],
        source_job_id="JOB-000000000000000000000000",
    )
    child = tmp_path / "child"
    child.mkdir()
    (child / "manifest.json").write_bytes(patch["manifest_bytes"])
    (child / patch["telemetry_filename"]).write_bytes(patch["telemetry_bytes"])
    manifest, telemetry = _load_public_case(child)
    assert manifest["case_id"].startswith("r12-public-001-RERUN-")
    assert manifest["maintenance_raw_patch_provenance"]["raw_telemetry_rerun_performed"] is True
    with telemetry.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [float(row["eps.battery.soc"]) for row in rows] == [0.9, 0.85, 0.8]
    assert patch["source_telemetry_sha256"] != patch["patched_telemetry_sha256"]


def test_r12_raw_patch_rejects_timestamp_drift(tmp_path: Path) -> None:
    root = tmp_path / "case"
    _write_case(root, _public_case_files())
    contract = build_public_case_raw_input_contracts(root)["contracts"][0]
    template = build_public_case_raw_patch_template(contract, root)["capsule"]
    template["sample_times_s"] = [0.0, 1.1, 2.0]
    template["values"] = [0.9, 0.85, 0.8]
    capsule = ProductionRawInputPatchCapsule.model_validate(template)
    with pytest.raises(ValueError, match="timestamps do not exactly match"):
        admit_public_case_raw_patch(
            case_root=root, contract=contract, capsule=capsule, actor="tester", admitted_at="2026-09-18T00:00:00+00:00"
        )


def test_r12_pipeline_runtime_remains_fail_closed() -> None:
    readiness = assess_maintenance_production_rerun(
        source_artifact_kind="PIPELINE_RUNTIME",
        source_report={"job_id": "JOB-X"},
        materialized_patch=None,
        raw_contract_status="BLOCKED",
        raw_contract_blockers=["UPSTREAM_RECOMPUTE_ADAPTER_REQUIRED"],
    )
    assert readiness.status == "BLOCKED"
    assert "UPSTREAM_RECOMPUTE_ADAPTER_REQUIRED" in readiness.blockers
    assert readiness.raw_telemetry_rerun_planned is False


def test_r12_store_creates_and_completes_raw_public_child(tmp_path: Path) -> None:
    store = UploadReportJobStore(tmp_path / "jobs")
    source = store.create("r12 source")
    _put_all(store, source["job_id"], _public_case_files())
    source_report = store.run(source["job_id"])
    assert source_report["status"] == "SUCCEEDED"
    source_report_bytes = (store._job_root(source["job_id"]) / "output" / "report.json").read_bytes()

    contracts = store.maintenance_production_raw_input_contracts(source["job_id"])
    contract = contracts["contracts"][0]
    template = store.maintenance_production_raw_input_template(source["job_id"], contract["contract_id"])
    capsule = dict(template["capsule"])
    capsule["values"] = [0.91, 0.86, 0.81]
    admitted = store.admit_maintenance_production_raw_input(
        source["job_id"], contract_id=contract["contract_id"], capsule_payload=capsule, actor="tester"
    )
    assert admitted["production_rerun_readiness"]["status"] == "READY"

    created = store.create_maintenance_production_rerun(source["job_id"], actor="tester", start=False)
    child_id = created["child_job_id"]
    child_report = store.run(child_id)
    assert child_report["status"] == "SUCCEEDED"
    assert child_report["artifact_kind"] == "PUBLIC_CASE"
    assert (store._job_root(source["job_id"]) / "output" / "report.json").read_bytes() == source_report_bytes
    child_job = store.get(child_id)
    link = child_job["production_rerun_provenance"]
    assert link["raw_telemetry_rerun_performed"] is True
    assert link["formal_release_eligible"] is False
    completion_path = store._job_root(child_id) / "output" / "maintenance_production_rerun_completion_receipt.json"
    completion = verify_production_rerun_completion_receipt(json.loads(completion_path.read_text(encoding="utf-8")))
    assert completion["source_job_id"] == source["job_id"]
    assert completion["child_job_id"] == child_id
    assert completion["raw_telemetry_rerun_performed"] is True
    assert completion["formal_release_eligible"] is False
    assert completion["child_formal_diagnosis_publishable"] is False


def test_r12_api_exposes_raw_contract_admission_and_rerun(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASTROGRAPH_PLATFORM_ROOT", str(tmp_path / "platform"))
    app = create_app()
    client = TestClient(app)
    job = client.post("/v1/platform/jobs", json={"task": "r12 api"}).json()
    for name, data in _public_case_files().items():
        assert client.put(f"/v1/platform/jobs/{job['job_id']}/files/{name}", content=data).status_code == 200
    assert client.post(f"/v1/platform/jobs/{job['job_id']}/run").status_code == 200
    contracts = client.get(f"/v1/platform/jobs/{job['job_id']}/maintenance/production-rerun/raw-input-contracts")
    assert contracts.status_code == 200
    contract = contracts.json()["contracts"][0]
    template = client.get(
        f"/v1/platform/jobs/{job['job_id']}/maintenance/production-rerun/raw-input-contracts/{contract['contract_id']}/template"
    ).json()
    capsule = template["capsule"]
    capsule["values"] = [0.93, 0.88, 0.83]
    admitted = client.post(
        f"/v1/platform/jobs/{job['job_id']}/maintenance/production-rerun/raw-input-contracts/{contract['contract_id']}/admit",
        json=capsule,
    )
    assert admitted.status_code == 200
    assert admitted.json()["production_rerun_readiness"]["status"] == "READY"
    created = client.post(
        f"/v1/platform/jobs/{job['job_id']}/maintenance/production-rerun",
        json={"start": False},
    )
    assert created.status_code == 200
    assert created.json()["child_job_id"].startswith("JOB-")

def test_r12_equivalent_raw_rerun_is_idempotent_until_failed(tmp_path: Path) -> None:
    store = UploadReportJobStore(tmp_path / "jobs")
    source = store.create("r12 idempotency")
    _put_all(store, source["job_id"], _public_case_files())
    assert store.run(source["job_id"])["status"] == "SUCCEEDED"
    contract = store.maintenance_production_raw_input_contracts(source["job_id"])["contracts"][0]
    capsule = store.maintenance_production_raw_input_template(source["job_id"], contract["contract_id"])["capsule"]
    capsule["values"] = [0.92, 0.87, 0.82]
    store.admit_maintenance_production_raw_input(
        source["job_id"], contract_id=contract["contract_id"], capsule_payload=capsule, actor="tester"
    )
    first = store.create_maintenance_production_rerun(source["job_id"], actor="tester", start=False)
    child_id = first["child_job_id"]
    readiness = store.maintenance_production_rerun_readiness(source["job_id"])
    assert readiness["status"] == "BLOCKED"
    assert "EQUIVALENT_PRODUCTION_RERUN_EXISTS" in readiness["blockers"]
    assert readiness["equivalent_child_job_id"] == child_id
    with pytest.raises(ValueError, match="EQUIVALENT_PRODUCTION_RERUN_EXISTS"):
        store.create_maintenance_production_rerun(source["job_id"], actor="tester", start=False)

    failed = store.get(child_id)
    failed["status"] = "FAILED"
    store._save(failed)
    retried = store.create_maintenance_production_rerun(source["job_id"], actor="tester", start=False)
    assert retried["child_job_id"] != child_id


def test_r12_source_raw_input_drift_blocks_completion(tmp_path: Path) -> None:
    store = UploadReportJobStore(tmp_path / "jobs")
    source = store.create("r12 drift")
    _put_all(store, source["job_id"], _public_case_files())
    assert store.run(source["job_id"])["status"] == "SUCCEEDED"
    contract = store.maintenance_production_raw_input_contracts(source["job_id"])["contracts"][0]
    capsule = store.maintenance_production_raw_input_template(source["job_id"], contract["contract_id"])["capsule"]
    capsule["values"] = [0.95, 0.9, 0.85]
    store.admit_maintenance_production_raw_input(
        source["job_id"], contract_id=contract["contract_id"], capsule_payload=capsule, actor="tester"
    )
    created = store.create_maintenance_production_rerun(source["job_id"], actor="tester", start=False)
    child_id = created["child_job_id"]

    # Simulate out-of-band corruption of the supposedly immutable source upload.
    # _safe_expand() reconstructs from input/, so mutate the stored upload rather than
    # the disposable expanded/ directory.
    telemetry = store._job_root(source["job_id"]) / "input" / "telemetry.csv"
    telemetry.write_text("time_s,eps.battery.soc\n0,0.1\n1,0.1\n2,0.1\n", encoding="utf-8")
    # Because the source manifest hash no longer matches telemetry, the post-run
    # production lineage check must fail closed instead of issuing a completion receipt.
    result = store.run(child_id)
    assert result["status"] == "FAILED"
    assert not (store._job_root(child_id) / "output" / "maintenance_production_rerun_completion_receipt.json").exists()
