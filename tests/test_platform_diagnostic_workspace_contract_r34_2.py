from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from astrograph.integrations.diagnostic_workspace_presentation import (
    DiagnosticWorkspaceSourceBundle,
    WORKSPACE_SOURCE_SCHEMA_VERSION,
)
from astrograph.integrations.upload_report import UploadReportJobStore
from tests.test_platform_diagnostic_workspace_contract_r34_1 import (
    JOB_ID,
    _graph,
    _inputs,
    _simulation_source,
)


def _stub_frozen_inputs(store: UploadReportJobStore, monkeypatch: pytest.MonkeyPatch):
    report, maintenance, visualization = _inputs()
    monkeypatch.setattr(store, "get", lambda job_id: {"job_id": job_id, "status": "SUCCEEDED"})
    monkeypatch.setattr(store, "report", lambda _job_id: report)
    monkeypatch.setattr(store, "maintenance", lambda _job_id: maintenance)
    monkeypatch.setattr(store, "visualization", lambda _job_id: visualization)
    return report, maintenance, visualization


def test_concurrent_first_retention_is_atomic_first_writer_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = UploadReportJobStore(tmp_path / "jobs")
    _stub_frozen_inputs(store, monkeypatch)
    source = _simulation_source()

    # Repeat the race with fresh identities so this also detects partial JSON or
    # replace-based last-writer-wins behavior under different schedules.
    for index in range(8):
        job_id = f"JOB-{index:024X}"
        (store.root / job_id / "output").mkdir(parents=True)
        barrier = threading.Barrier(3)

        def retain(graph):
            barrier.wait(timeout=5)
            return store.retain_diagnostic_workspace_sources(
                job_id,
                system_model_graph=graph,
                simulation_sources=(source,),
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(retain, _graph()), pool.submit(retain, None)]
            barrier.wait(timeout=5)
            outcomes = []
            for future in futures:
                try:
                    outcomes.append(("RETURN", future.result(timeout=10)))
                except ValueError as exc:
                    outcomes.append(("IMMUTABLE", str(exc)))

        assert [kind for kind, _ in outcomes].count("RETURN") == 1
        assert [kind for kind, _ in outcomes].count("IMMUTABLE") == 1
        assert "immutable after retention" in next(value for kind, value in outcomes if kind == "IMMUTABLE")
        persisted = json.loads(store._diagnostic_workspace_source_path(job_id).read_text(encoding="utf-8"))
        DiagnosticWorkspaceSourceBundle.model_validate(persisted)
        assert persisted == next(value for kind, value in outcomes if kind == "RETURN")
        assert not list((store.root / job_id / "output").glob("*.tmp"))


def test_concurrent_identical_retention_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = UploadReportJobStore(tmp_path / "jobs")
    _stub_frozen_inputs(store, monkeypatch)
    (store.root / JOB_ID / "output").mkdir(parents=True)
    source = _simulation_source()
    barrier = threading.Barrier(3)

    def retain():
        barrier.wait(timeout=5)
        return store.retain_diagnostic_workspace_sources(
            JOB_ID, system_model_graph=_graph(), simulation_sources=(source,),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(retain), pool.submit(retain)]
        barrier.wait(timeout=5)
        first, second = (future.result(timeout=10) for future in futures)
    assert first == second
    assert first["schema_version"] == WORKSPACE_SOURCE_SCHEMA_VERSION


def test_retained_source_rejects_candidate_context_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = UploadReportJobStore(tmp_path / "jobs")
    report, maintenance, visualization = _inputs()
    current = {"maintenance": maintenance}
    (store.root / JOB_ID / "output").mkdir(parents=True)
    monkeypatch.setattr(store, "get", lambda job_id: {"job_id": job_id, "status": "SUCCEEDED"})
    monkeypatch.setattr(store, "report", lambda _job_id: report)
    monkeypatch.setattr(store, "maintenance", lambda _job_id: current["maintenance"])
    monkeypatch.setattr(store, "visualization", lambda _job_id: visualization)

    store.retain_diagnostic_workspace_sources(
        JOB_ID, system_model_graph=_graph(), simulation_sources=(_simulation_source(),),
    )
    current["maintenance"] = dict(maintenance, fault_id="PUMP:FailedOff")
    with pytest.raises(ValueError, match="candidate context drift"):
        store.diagnostic_workspace(JOB_ID)


def test_legacy_unbound_source_bundle_is_rejected() -> None:
    with pytest.raises(ValueError):
        DiagnosticWorkspaceSourceBundle.model_validate({
            "schema_version": "astrograph-diagnostic-workspace-source-bundle.v1",
            "job_id": JOB_ID,
            "authority_snapshot_sha256": "a" * 64,
            "system_model_graph": None,
            "simulation_sources": [],
            "bundle_sha256": "b" * 64,
        })
