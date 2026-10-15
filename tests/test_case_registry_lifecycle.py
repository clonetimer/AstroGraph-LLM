from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from astrograph.api import create_app
from astrograph.cases import DiagnosticCaseRecord, DiagnosticCaseRegistry
from astrograph.config import DiagnosisPolicy
from astrograph.knowledge.repository import CsvKnowledgeRepository
from astrograph.service.core import AstroGraphService


def _record(case_record_id: str = "case-record-00000001") -> DiagnosticCaseRecord:
    return DiagnosticCaseRecord(
        case_record_id=case_record_id,
        request_id="request-1",
        anomaly_count=1,
        anomaly_hash="a" * 64,
        diagnosis_status="DIAGNOSED",
        conclusion_fault_id="ADCS_RW_JAM",
        conclusion_fault_name="Reaction wheel jam",
        policy_version="policy-v1",
        knowledge_version="kg-v1",
        audit_id="audit-1",
    )


def _service(tmp_path: Path) -> AstroGraphService:
    registry = DiagnosticCaseRegistry(tmp_path / "cases.json")
    registry.upsert(_record())
    return AstroGraphService(
        repository=CsvKnowledgeRepository(tmp_path / "missing.csv"),
        policy=DiagnosisPolicy(),
        case_registry=registry,
    )


def test_case_registry_get_and_delete(tmp_path: Path) -> None:
    registry = DiagnosticCaseRegistry(tmp_path / "cases.json")
    record = _record()
    registry.upsert(record)

    assert registry.get(record.case_record_id) == record
    assert registry.delete(record.case_record_id) is True
    assert registry.get(record.case_record_id) is None
    assert registry.delete(record.case_record_id) is False


def test_case_api_lists_and_deletes_records_in_local_mode(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ASTROGRAPH_AUTH_MODE", raising=False)
    monkeypatch.delenv("ASTROGRAPH_API_TOKENS_JSON", raising=False)
    client = TestClient(create_app(_service(tmp_path)))

    listed = client.get("/v1/cases")
    deleted = client.delete("/v1/cases/case-record-00000001")
    missing = client.get("/v1/cases/case-record-00000001")

    assert listed.status_code == 200
    assert listed.json()["count"] == 1
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert missing.status_code == 404


def test_case_delete_requires_admin_in_bearer_mode(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ASTROGRAPH_AUTH_MODE", "bearer")
    monkeypatch.setenv(
        "ASTROGRAPH_API_TOKENS_JSON",
        json.dumps(
            {
                "viewer-token": {"subject": "viewer", "roles": ["viewer"]},
                "admin-token": {"subject": "admin", "roles": ["admin"]},
            }
        ),
    )
    client = TestClient(create_app(_service(tmp_path)))

    denied = client.delete(
        "/v1/cases/case-record-00000001",
        headers={"Authorization": "Bearer viewer-token"},
    )
    deleted = client.delete(
        "/v1/cases/case-record-00000001",
        headers={"Authorization": "Bearer admin-token"},
    )

    assert denied.status_code == 403
    assert deleted.status_code == 200
