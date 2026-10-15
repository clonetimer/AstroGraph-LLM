from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from fastapi.testclient import TestClient

from astrograph.api import create_app
from astrograph.config import DiagnosisPolicy
from astrograph.knowledge.repository import CsvKnowledgeRepository
from astrograph.service.core import AstroGraphService
from astrograph.service.factory import build_service_from_environment
from astrograph.service.paths import ServicePathLayout


ROOT = Path(__file__).resolve().parents[1]


def test_default_project_health_reports_missing_knowledge_without_raising(tmp_path: Path) -> None:
    project = tmp_path / "project"
    state = tmp_path / "state"
    shutil.copytree(ROOT / "configs", project / "configs")
    service = build_service_from_environment(project, state_dir=state)
    report = service.health()

    assert report.status == "DEGRADED"
    assert report.details["repository"]["status"] == "FAIL"
    assert report.details["repository"]["error"] == "KNOWLEDGE_EDGE_FILE_MISSING"


def test_factory_places_all_mutable_service_stores_under_external_state_root(tmp_path: Path) -> None:
    project = tmp_path / "project"
    state = tmp_path / "state"
    artifacts = tmp_path / "artifacts"
    shutil.copytree(ROOT / "configs", project / "configs")

    service = build_service_from_environment(
        project,
        state_dir=state,
        artifact_dir=artifacts,
    )
    mutable_paths = {
        service.audit_store.path,
        service.case_registry.path,
        service.idempotency_store.path,
        service.episode_store.path,
        service.request_lifecycle_store.path,
        service.knowledge_governance_store.path,
        service.knowledge_governance_store.active_release_path,
        service.knowledge_governance_store.history_dir,
        service.experiment_registry.path,
        service.evidence_registry.root,
        *service.persistence_coordinator.components.values(),
    }
    for mutable_path in mutable_paths:
        Path(mutable_path).resolve().relative_to(state.resolve())
    assert service.state_root == state.resolve()
    assert service.artifact_root == artifacts.resolve()
    assert not (project / "data" / "idempotency" / "requests_v0530.sqlite3").exists()


def test_service_path_layout_rejects_state_inside_source_tree(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    with pytest.raises(ValueError, match="outside project_root"):
        ServicePathLayout.resolve(project, state_dir=project / "data" / "runtime")


def test_service_path_layout_rejects_artifacts_inside_source_tree(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    with pytest.raises(ValueError, match="outside project_root"):
        ServicePathLayout.resolve(
            project,
            state_dir=tmp_path / "state",
            artifact_dir=project / "runtime-artifacts",
        )


def test_csv_repository_health_reports_schema_and_content(tmp_path: Path) -> None:
    edges = tmp_path / "causes.csv"
    edges.write_text(
        "fault_id,fault_name,anomaly_id,anomaly_name,weight\n"
        "F-1,Fault,A-1,Anomaly,1.0\n",
        encoding="utf-8",
    )
    report = CsvKnowledgeRepository(edges).health_check()

    assert report["status"] == "PASS"
    assert report["row_count"] == 1
    assert len(report["content_hash"]) == 64


def test_liveness_does_not_require_repository_and_readiness_returns_503(tmp_path: Path) -> None:
    missing = CsvKnowledgeRepository(tmp_path / "missing.csv")
    service = AstroGraphService(repository=missing, policy=DiagnosisPolicy())
    client = TestClient(create_app(service))

    live = client.get("/live")
    ready = client.get("/ready")
    health = client.get("/health")

    assert live.status_code == 200
    assert live.json()["status"] == "PASS"
    assert ready.status_code == 503
    assert ready.json()["status"] == "DEGRADED"
    assert health.status_code == 200
    assert health.json()["status"] == "DEGRADED"
