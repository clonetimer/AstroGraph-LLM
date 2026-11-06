from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from astrograph.integrations.platform_job_queue import (
    PersistentPlatformJobQueue,
    PlatformQueueConflict,
)
from astrograph.integrations.upload_report import UploadReportJobStore
from fastapi.testclient import TestClient
from astrograph.api import create_app


def test_claim_is_atomic_across_independent_queue_connections(tmp_path: Path) -> None:
    first = PersistentPlatformJobQueue(tmp_path)
    second = PersistentPlatformJobQueue(tmp_path)
    for index in range(6):
        first.enqueue(f"JOB-{index:024X}")

    barrier = threading.Barrier(8)
    claims = []
    lock = threading.Lock()

    def claim(index: int) -> None:
        barrier.wait()
        queue = first if index % 2 else second
        value = queue.claim_next(f"worker-{index}", max_running=2, lease_seconds=30)
        if value:
            with lock:
                claims.append(value["job_id"])

    threads = [threading.Thread(target=claim, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert len(claims) == 2
    assert len(set(claims)) == 2
    statuses = [
        first.get(f"JOB-{index:024X}")["status"]
        for index in range(6)
    ]
    assert statuses.count("RUNNING") == 2
    assert statuses.count("QUEUED") == 4


def test_expired_worker_lease_becomes_interrupted_and_never_auto_replays(tmp_path: Path) -> None:
    queue = PersistentPlatformJobQueue(tmp_path)
    job_id = "JOB-" + "A" * 24
    queue.enqueue(job_id)
    first = queue.claim_next("worker-before-crash", lease_seconds=1, max_running=1, now=100.0)
    assert first and first["attempt"] == 1

    assert queue.recover_expired(now=102.0) == [job_id]
    assert queue.get(job_id)["status"] == "INTERRUPTED"
    assert queue.claim_next("replacement-worker", now=102.0) is None
    with pytest.raises(PlatformQueueConflict, match="DUPLICATE_MODEL_COST_CONFIRMATION_REQUIRED"):
        queue.resume(job_id, confirm_duplicate_risk=False, actor="operator")

    resumed = queue.resume(job_id, confirm_duplicate_risk=True, actor="operator")
    assert resumed["status"] == "QUEUED"
    assert resumed["recovery_count"] == 1
    second = queue.claim_next("replacement-worker", lease_seconds=30, max_running=1)
    assert second and second["job_id"] == job_id
    assert second["attempt"] == 2


def test_running_cancel_is_cooperative_and_blocks_report_publication(tmp_path: Path) -> None:
    queue = PersistentPlatformJobQueue(tmp_path)
    job_id = "JOB-" + "B" * 24
    queue.enqueue(job_id)
    claimed = queue.claim_next("worker-1", max_running=1)
    assert claimed and claimed["status"] == "RUNNING"

    request = PersistentPlatformJobQueue(tmp_path).request_cancel(job_id, reason="operator:test")
    assert request["cancel_result"] == "CANCEL_REQUESTED"
    heartbeat = queue.heartbeat(job_id, "worker-1")
    assert heartbeat["owned"] is True
    assert heartbeat["cancel_requested"] is True
    assert queue.begin_publish(job_id, "worker-1") is False
    final = queue.finish(job_id, "worker-1", "CANCELLED")
    assert final["status"] == "CANCELLED"


def test_cancel_after_publish_guard_is_rejected_as_too_late(tmp_path: Path) -> None:
    queue = PersistentPlatformJobQueue(tmp_path)
    job_id = "JOB-" + "C" * 24
    queue.enqueue(job_id)
    queue.claim_next("worker-1", max_running=1)
    assert queue.begin_publish(job_id, "worker-1") is True
    cancelled = queue.request_cancel(job_id)
    assert cancelled["cancel_result"] == "TOO_LATE"
    final = queue.finish(job_id, "worker-1", "SUCCEEDED")
    assert final["status"] == "SUCCEEDED"


def test_queued_cancel_is_immediate_and_not_claimable(tmp_path: Path) -> None:
    queue = PersistentPlatformJobQueue(tmp_path)
    job_id = "JOB-" + "D" * 24
    queue.enqueue(job_id)
    cancelled = queue.request_cancel(job_id)
    assert cancelled["cancel_result"] == "CANCELLED"
    assert queue.claim_next("worker-1") is None


def test_store_cancellation_is_observed_across_queue_connections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from astrograph.integrations.stage_execution import AnalysisCancelled, check_cancelled

    monkeypatch.setenv("ASTROGRAPH_PLATFORM_QUEUE_HEARTBEAT_SECONDS", "0.05")
    monkeypatch.setenv("ASTROGRAPH_PLATFORM_QUEUE_LEASE_SECONDS", "3")
    store = UploadReportJobStore(tmp_path / "platform")
    job = store.create("cooperative cancellation")
    job_id = job["job_id"]
    store.queue.enqueue(job_id)
    worker_id = "worker-cancel-test"
    assert store.queue.claim_next(worker_id, max_running=1)

    started = threading.Event()

    def blocked_run(_job_id, *, before_publish=None):
        started.set()
        try:
            while True:
                check_cancelled()
                time.sleep(0.01)
        except AnalysisCancelled:
            return {"job_id": _job_id, "status": "CANCELLED"}

    store._run_unlocked = blocked_run
    result = {}

    def run():
        result["value"] = store.run(job_id, worker_id=worker_id)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert started.wait(3)
    other_process_view = PersistentPlatformJobQueue(store.root)
    other_process_view.request_cancel(job_id, reason="cross-process-test")
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert result["value"]["status"] == "CANCELLED"
    assert store.queue.get(job_id)["status"] == "CANCELLED"


def test_interrupted_complete_report_is_recovered_without_model_replay(tmp_path: Path) -> None:
    store = UploadReportJobStore(tmp_path / "platform")
    job = store.create("recover complete report")
    job_id = job["job_id"]
    store.queue.enqueue(job_id)
    store.queue.claim_next("worker-crashed", lease_seconds=1, max_running=1, now=100.0)
    store.queue.recover_expired(now=102.0)

    output = store._job_root(job_id) / "output"
    (output / "report.json").write_text(json.dumps({
        "job_id": job_id, "status": "SUCCEEDED", "completed_at": "2026-11-06T00:00:00+00:00",
    }), encoding="utf-8")
    (output / "report.md").write_text("# Completed report\n", encoding="utf-8")

    result = store.resume(job_id, confirm_duplicate_risk=False, actor="operator")
    assert result["recovered_from_complete_report"] is True
    assert result["queue"]["status"] == "SUCCEEDED"
    assert store.get(job_id)["status"] == "SUCCEEDED"


def test_interrupted_partial_output_is_archived_before_explicit_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = UploadReportJobStore(tmp_path / "platform")
    monkeypatch.setattr(store, "ensure_queue_workers", lambda **kwargs: [])
    job = store.create("recover partial report")
    job_id = job["job_id"]
    store.queue.enqueue(job_id)
    store.queue.claim_next("worker-crashed", lease_seconds=1, max_running=1, now=100.0)
    store.queue.recover_expired(now=102.0)
    output = store._job_root(job_id) / "output"
    (output / "partial-attempt.log").write_text("model request may already have been sent", encoding="utf-8")
    checkpoint = output / "knowledge" / "checkpoint.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint_payload = json.dumps({
        "schema_version": "astrograph-long-report-checkpoint.v1",
        "input_sha256": "f" * 64, "completed": [],
    })
    checkpoint.write_text(checkpoint_payload, encoding="utf-8")

    with pytest.raises(PlatformQueueConflict, match="DUPLICATE_MODEL_COST_CONFIRMATION_REQUIRED"):
        store.resume(job_id, confirm_duplicate_risk=False, actor="operator")
    assert (output / "partial-attempt.log").is_file()

    result = store.resume(job_id, confirm_duplicate_risk=True, actor="operator")
    assert result["status"] == "QUEUED"
    assert result["duplicate_model_cost_confirmed"] is True
    assert result["checkpoint_preserved"] is True
    assert (output / "knowledge" / "checkpoint.json").read_text(encoding="utf-8") == checkpoint_payload
    assert not (output / "partial-attempt.log").exists()
    archived = store._job_root(job_id) / "interrupted-attempts" / "attempt-001" / "partial-attempt.log"
    assert archived.is_file()


def test_worker_entrypoint_drains_a_persisted_job_in_a_separate_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import csv
    import hashlib
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["time_s", "eps.battery.soc"])
    writer.writeheader()
    writer.writerows([{"time_s": 0, "eps.battery.soc": 0.8}, {"time_s": 1, "eps.battery.soc": 0.7}])
    telemetry = buffer.getvalue().encode("utf-8")
    manifest = {
        "schema_version": "sat-sim.astrograph-public-case.v1",
        "case_id": "queue-process-case",
        "telemetry": {
            "file": "telemetry.csv", "sha256": hashlib.sha256(telemetry).hexdigest(),
            "rows": 2, "fields": ["time_s", "eps.battery.soc"],
        },
    }
    store = UploadReportJobStore(tmp_path / "platform")
    monkeypatch.setattr(store, "ensure_queue_workers", lambda **kwargs: [])
    job = store.create("separate worker process")
    store.put_file(job["job_id"], "manifest.json", json.dumps(manifest).encode("utf-8"))
    store.put_file(job["job_id"], "telemetry.csv", telemetry)
    store.start(job["job_id"])

    source_root = Path(__file__).resolve().parents[1] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source_root) + (os.pathsep + env.get("PYTHONPATH", ""))
    env["ASTROGRAPH_PLATFORM_ROOT"] = str(store.root)
    env["ASTROGRAPH_PLATFORM_QUEUE_WORKERS"] = "1"
    result = subprocess.run(
        [
            sys.executable, "-m", "astrograph.integrations.platform_worker",
            "--root", str(store.root), "--queue-slot", "0", "--idle-timeout", "1",
        ],
        cwd=str(source_root.parent), env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert store.get(job["job_id"])["status"] == "SUCCEEDED"
    assert store.queue.get(job["job_id"])["status"] == "SUCCEEDED"


def test_api_cancel_endpoint_exposes_persisted_queue_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASTROGRAPH_AUTH_MODE", "local")
    monkeypatch.setenv("ASTROGRAPH_PLATFORM_ROOT", str(tmp_path / "api-platform"))
    monkeypatch.setenv("ASTROGRAPH_PLATFORM_QUEUE_WORKERS", "1")
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: type("P", (), {"pid": 4242})())
    with TestClient(create_app()) as client:
        created = client.post("/v1/platform/jobs", json={"task": "API queue cancellation"})
        assert created.status_code == 200, created.text
        job_id = created.json()["job_id"]
        import hashlib
        import io
        import csv
        store = UploadReportJobStore(tmp_path / "api-platform")
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=["time_s", "eps.battery.soc"])
        writer.writeheader()
        writer.writerows([
            {"time_s": 0, "eps.battery.soc": 0.8},
            {"time_s": 1, "eps.battery.soc": 0.7},
        ])
        telemetry = buffer.getvalue().encode("utf-8")
        manifest = {
            "schema_version": "sat-sim.astrograph-public-case.v1",
            "case_id": "api-cancel-case",
            "telemetry": {
                "file": "telemetry.csv", "sha256": hashlib.sha256(telemetry).hexdigest(),
                "rows": 2, "fields": ["time_s", "eps.battery.soc"],
            },
        }
        store.put_file(job_id, "manifest.json", json.dumps(manifest).encode("utf-8"))
        store.put_file(job_id, "telemetry.csv", telemetry)
        started = client.post(f"/v1/platform/jobs/{job_id}/start")
        assert started.status_code == 202, started.text
        cancelled = client.post(
            f"/v1/platform/jobs/{job_id}/cancel",
            json={"reason": "operator test"},
        )
        assert cancelled.status_code == 200, cancelled.text
        assert cancelled.json()["cancel_result"] == "CANCELLED"
        snapshot = client.get(f"/v1/platform/jobs/{job_id}")
        assert snapshot.status_code == 200
        assert snapshot.json()["queue"]["status"] == "CANCELLED"


def test_api_resume_requires_explicit_duplicate_cost_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASTROGRAPH_AUTH_MODE", "local")
    root = tmp_path / "api-resume-platform"
    monkeypatch.setenv("ASTROGRAPH_PLATFORM_ROOT", str(root))
    monkeypatch.setenv("ASTROGRAPH_PLATFORM_QUEUE_WORKERS", "1")
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: type("P", (), {"pid": 4242})())
    store = UploadReportJobStore(root)
    job = store.create("API interrupted recovery")
    store.queue.enqueue(job["job_id"])
    store.queue.claim_next("crashed-worker", lease_seconds=1, max_running=1, now=100.0)
    store.queue.recover_expired(now=102.0)

    with TestClient(create_app()) as client:
        refused = client.post(
            f"/v1/platform/jobs/{job['job_id']}/resume",
            json={"confirm_duplicate_risk": False},
        )
        assert refused.status_code == 422
        resumed = client.post(
            f"/v1/platform/jobs/{job['job_id']}/resume",
            json={"confirm_duplicate_risk": True},
        )
        assert resumed.status_code == 202, resumed.text
        assert resumed.json()["queue"]["status"] == "QUEUED"
        assert resumed.json()["duplicate_model_cost_confirmed"] is True


def test_platform_ui_exposes_queue_cancellation_and_recovery_controls() -> None:
    resources = Path(__file__).resolve().parents[1] / "src" / "astrograph" / "resources"
    html = (resources / "platform_ui.html").read_text(encoding="utf-8")
    js = (resources / "platform_job.js").read_text(encoding="utf-8")
    assert 'id="cancel-job"' in html
    assert 'id="resume-job"' in html
    assert "/cancel" in js and "/resume" in js
    assert "CANCEL_REQUESTED" in js and "INTERRUPTED" in js
    assert "confirm_duplicate_risk:true" in js
