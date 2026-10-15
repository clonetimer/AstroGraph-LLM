from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from astrograph.api import create_app
from astrograph.config import DiagnosisPolicy
from astrograph.knowledge.repository import CsvKnowledgeRepository
from astrograph.service.core import AstroGraphService
from astrograph.security import ApiAuthConfig


def _service(tmp_path: Path) -> AstroGraphService:
    return AstroGraphService(
        repository=CsvKnowledgeRepository(tmp_path / "missing.csv"),
        policy=DiagnosisPolicy(),
    )


def test_default_local_mode_allows_loopback_client(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("ASTROGRAPH_AUTH_MODE", raising=False)
    monkeypatch.delenv("ASTROGRAPH_API_TOKENS_JSON", raising=False)
    client = TestClient(create_app(_service(tmp_path)))

    response = client.get("/v1/knowledge/active-release")

    assert response.status_code == 200
    assert response.headers["X-AstroGraph-Auth-Mode"] == "local"


def test_bearer_mode_enforces_authentication_and_roles(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ASTROGRAPH_AUTH_MODE", "bearer")
    monkeypatch.setenv(
        "ASTROGRAPH_API_TOKENS_JSON",
        json.dumps(
            {
                "viewer-token": {"subject": "viewer-a", "roles": ["viewer"]},
                "operator-token": {"subject": "operator-a", "roles": ["operator"]},
                "admin-token": {"subject": "admin-a", "roles": ["admin"]},
            }
        ),
    )
    client = TestClient(create_app(_service(tmp_path)))

    unauthenticated = client.get("/v1/knowledge/active-release")
    viewer = client.get(
        "/v1/knowledge/active-release",
        headers={"Authorization": "Bearer viewer-token"},
    )
    viewer_write = client.post(
        "/v1/compatibility/simulation",
        headers={"Authorization": "Bearer viewer-token"},
        json={},
    )
    operator_validation = client.post(
        "/v1/compatibility/simulation",
        headers={"Authorization": "Bearer operator-token"},
        json={},
    )
    admin = client.get(
        "/v1/admin/persistence/status",
        headers={"Authorization": "Bearer admin-token"},
    )

    assert unauthenticated.status_code == 401
    assert viewer.status_code == 200
    assert viewer_write.status_code == 403
    assert operator_validation.status_code == 422
    assert admin.status_code == 200


def test_bearer_identity_overrides_self_reported_knowledge_actor(monkeypatch) -> None:
    monkeypatch.setenv("ASTROGRAPH_AUTH_MODE", "bearer")
    monkeypatch.setenv(
        "ASTROGRAPH_API_TOKENS_JSON",
        json.dumps({"admin-token": {"subject": "approved-admin", "roles": ["admin"]}}),
    )

    class CaptureService:
        def create_knowledge_draft(self, **kwargs):
            return kwargs

    client = TestClient(create_app(CaptureService()))
    response = client.post(
        "/v1/knowledge/drafts",
        headers={"Authorization": "Bearer admin-token"},
        json={
            "title": "Change",
            "description": "",
            "candidate_release_path": "candidate.json",
            "created_by": "spoofed-user",
        },
    )

    assert response.status_code == 200
    assert response.json()["created_by"] == "approved-admin"


def test_local_mode_rejects_forwarded_loopback_requests(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ASTROGRAPH_AUTH_MODE", "local")
    monkeypatch.delenv("ASTROGRAPH_ALLOW_FORWARDED_LOCAL_AUTH", raising=False)
    client = TestClient(create_app(_service(tmp_path)))

    response = client.get(
        "/v1/knowledge/active-release",
        headers={"X-Forwarded-For": "203.0.113.10"},
    )

    assert response.status_code == 401


def test_production_mode_rejects_local_or_disabled_auth(monkeypatch) -> None:
    monkeypatch.setenv("ASTROGRAPH_DEPLOYMENT_MODE", "production")
    monkeypatch.setenv("ASTROGRAPH_AUTH_MODE", "local")
    with pytest.raises(RuntimeError, match="requires ASTROGRAPH_AUTH_MODE=bearer"):
        ApiAuthConfig.from_environment()
