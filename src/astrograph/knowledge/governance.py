from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from astrograph.knowledge.release import KnowledgeRelease, validate_knowledge_release
from astrograph.knowledge.patches import (
    KnowledgePatchImpactReport,
    KnowledgePatchValidationReport,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _change_id(base_release_id: str, candidate_release_id: str, title: str) -> str:
    return hashlib.sha256(f"{base_release_id}|{candidate_release_id}|{title}".encode("utf-8")).hexdigest()[:32]


class KnowledgeChangeStatus(str, Enum):
    DRAFT = "DRAFT"
    SUBMITTED = "SUBMITTED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    PUBLISHED = "PUBLISHED"
    ROLLED_BACK = "ROLLED_BACK"


class KnowledgeChangeTransition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: KnowledgeChangeStatus
    at: str = Field(default_factory=_now)
    actor: str
    comment: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class KnowledgeChangeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-knowledge-change.v1"
    change_id: str = Field(min_length=16)
    title: str = Field(min_length=1)
    description: str = ""
    base_release_id: str
    candidate_release_id: str
    candidate_release_path: str
    status: KnowledgeChangeStatus = KnowledgeChangeStatus.DRAFT
    created_by: str
    reviewer: str = ""
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
    validation: dict[str, Any] = Field(default_factory=dict)
    previous_release_snapshot: dict[str, Any] = Field(default_factory=dict)
    published_release_snapshot: dict[str, Any] = Field(default_factory=dict)
    transitions: list[KnowledgeChangeTransition] = Field(default_factory=list)
    claim_boundary: str = (
        "Engineering knowledge governance record. Actor strings are audit metadata only; authentication and authorization "
        "must be enforced by the deployment environment."
    )


class JsonKnowledgeGovernanceStore:
    schema_version = "astrograph-knowledge-governance-store.v1"

    def __init__(
        self,
        path: str | Path,
        *,
        project_root: str | Path,
        active_release_path: str | Path,
        history_dir: str | Path,
    ):
        self.path = Path(path)
        self.project_root = Path(project_root).resolve()
        self.active_release_path = Path(active_release_path)
        if not self.active_release_path.is_absolute():
            self.active_release_path = self.project_root / self.active_release_path
        self.history_dir = Path(history_dir)
        if not self.history_dir.is_absolute():
            self.history_dir = self.project_root / self.history_dir
        self._lock = Lock()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": self.schema_version, "changes": {}}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != self.schema_version:
            raise ValueError("unsupported knowledge governance store schema")
        payload.setdefault("changes", {})
        return payload

    def _save(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def _resolve_inside_project(self, path_value: str | Path) -> Path:
        path = Path(path_value)
        resolved = path.resolve() if path.is_absolute() else (self.project_root / path).resolve()
        if resolved != self.project_root and self.project_root not in resolved.parents:
            raise ValueError("knowledge governance paths must remain inside project_root")
        return resolved

    def _candidate_path(self, record: KnowledgeChangeRequest) -> Path:
        return self._resolve_inside_project(record.candidate_release_path)

    def _active_release(self) -> KnowledgeRelease | None:
        if not self.active_release_path.exists():
            return None
        return KnowledgeRelease.from_json(self.active_release_path)

    def _persist_record(self, payload: dict[str, Any], record: KnowledgeChangeRequest) -> KnowledgeChangeRequest:
        record.updated_at = _now()
        payload["changes"][record.change_id] = record.model_dump(mode="json")
        self._save(payload)
        return record

    def create_draft(
        self,
        *,
        title: str,
        description: str,
        candidate_release_path: str,
        created_by: str,
        patch_validation: dict[str, Any] | None = None,
        impact_report: dict[str, Any] | None = None,
    ) -> KnowledgeChangeRequest:
        candidate_path = self._resolve_inside_project(candidate_release_path)
        candidate = KnowledgeRelease.from_json(candidate_path)
        normalized_candidate_path = str(candidate_path.relative_to(self.project_root)).replace("\\", "/")
        active = self._active_release()
        base_release_id = active.release_id if active else ""
        compiled_patch_release = candidate.claim_boundary.startswith("Compiled from KPATCH-")
        if (patch_validation is None) != (impact_report is None):
            raise ValueError("patch validation and impact report must be provided together")
        if compiled_patch_release and patch_validation is None:
            raise ValueError(
                "compiled knowledge patch release requires deterministic validation and impact replay"
            )
        if patch_validation is not None:
            typed_validation = KnowledgePatchValidationReport.model_validate(patch_validation)
            typed_impact = KnowledgePatchImpactReport.model_validate(impact_report)
            if typed_validation.status != "READY_FOR_SANDBOX":
                raise ValueError("knowledge patch deterministic validation is not ready")
            if typed_impact.status != "PASS":
                raise ValueError("knowledge patch sandbox impact replay did not pass")
            if typed_validation.candidate_id != typed_impact.candidate_id:
                raise ValueError("knowledge patch validation and impact candidate IDs differ")
            if typed_impact.base_release_id != base_release_id:
                raise ValueError("knowledge patch impact base release is stale")
            if typed_impact.candidate_release_id != candidate.release_id:
                raise ValueError("knowledge patch impact candidate release differs from draft")
            if typed_validation.active_graph_modified is not False:
                raise ValueError("knowledge patch validation must not modify the active graph")
            if typed_impact.active_graph_modified is not False:
                raise ValueError("knowledge patch impact replay must not modify the active graph")
            patch_validation = typed_validation.model_dump(mode="json")
            impact_report = typed_impact.model_dump(mode="json")
        change_id = _change_id(base_release_id, candidate.release_id, title)
        with self._lock:
            payload = self._load()
            existing = payload["changes"].get(change_id)
            if existing:
                return KnowledgeChangeRequest.model_validate(existing)
            record = KnowledgeChangeRequest(
                change_id=change_id,
                title=title,
                description=description,
                base_release_id=base_release_id,
                candidate_release_id=candidate.release_id,
                candidate_release_path=normalized_candidate_path,
                created_by=created_by,
                validation=(
                    {
                        "patch_preflight": patch_validation,
                        "sandbox_impact": impact_report,
                    }
                    if patch_validation is not None
                    else {}
                ),
                transitions=[
                    KnowledgeChangeTransition(
                        status=KnowledgeChangeStatus.DRAFT,
                        actor=created_by,
                        comment="Knowledge change draft created.",
                        details={
                            "patch_preflight_attached": patch_validation is not None,
                            "sandbox_impact_attached": impact_report is not None,
                        },
                    )
                ],
            )
            return self._persist_record(payload, record)

    def _record(self, payload: dict[str, Any], change_id: str) -> KnowledgeChangeRequest:
        raw = payload["changes"].get(change_id)
        if raw is None:
            raise KeyError(change_id)
        return KnowledgeChangeRequest.model_validate(raw)

    def submit(self, change_id: str, *, actor: str, comment: str = "") -> KnowledgeChangeRequest:
        with self._lock:
            payload = self._load()
            record = self._record(payload, change_id)
            if record.status != KnowledgeChangeStatus.DRAFT:
                raise ValueError("only DRAFT changes can be submitted")
            candidate = KnowledgeRelease.from_json(self._candidate_path(record))
            validation = validate_knowledge_release(candidate, self.project_root)
            if validation["status"] != "PASS":
                raise ValueError("candidate knowledge release failed validation")
            record.validation = {**record.validation, "knowledge_release_validation": validation}
            record.status = KnowledgeChangeStatus.SUBMITTED
            record.transitions.append(
                KnowledgeChangeTransition(
                    status=record.status,
                    actor=actor,
                    comment=comment or "Submitted for review after deterministic validation passed.",
                )
            )
            return self._persist_record(payload, record)

    def review(
        self,
        change_id: str,
        *,
        reviewer: str,
        approve: bool,
        comment: str,
    ) -> KnowledgeChangeRequest:
        with self._lock:
            payload = self._load()
            record = self._record(payload, change_id)
            if record.status != KnowledgeChangeStatus.SUBMITTED:
                raise ValueError("only SUBMITTED changes can be reviewed")
            record.reviewer = reviewer
            record.status = KnowledgeChangeStatus.APPROVED if approve else KnowledgeChangeStatus.REJECTED
            record.transitions.append(
                KnowledgeChangeTransition(status=record.status, actor=reviewer, comment=comment)
            )
            return self._persist_record(payload, record)

    def _write_release_atomic(self, release: KnowledgeRelease) -> None:
        self.active_release_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.active_release_path.with_suffix(self.active_release_path.suffix + ".tmp")
        tmp.write_text(json.dumps(release.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.active_release_path)

    def _archive(self, release: KnowledgeRelease) -> None:
        self.history_dir.mkdir(parents=True, exist_ok=True)
        path = self.history_dir / f"{release.release_id}.json"
        path.write_text(json.dumps(release.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")

    def publish(self, change_id: str, *, actor: str, comment: str = "") -> KnowledgeChangeRequest:
        with self._lock:
            payload = self._load()
            record = self._record(payload, change_id)
            if record.status != KnowledgeChangeStatus.APPROVED:
                raise ValueError("only APPROVED changes can be published")
            candidate = KnowledgeRelease.from_json(self._candidate_path(record))
            validation = validate_knowledge_release(candidate, self.project_root)
            if validation["status"] != "PASS":
                raise ValueError("candidate knowledge release failed validation at publish time")
            if candidate.release_status != "engineering_released":
                raise ValueError("published candidate must have release_status=engineering_released")
            active = self._active_release()
            if active and active.release_id != record.base_release_id:
                raise ValueError("active release changed since draft creation; rebase is required")
            if active:
                self._archive(active)
                record.previous_release_snapshot = active.model_dump(mode="json")
            self._archive(candidate)
            self._write_release_atomic(candidate)
            record.published_release_snapshot = candidate.model_dump(mode="json")
            record.validation = validation
            record.status = KnowledgeChangeStatus.PUBLISHED
            record.transitions.append(
                KnowledgeChangeTransition(
                    status=record.status,
                    actor=actor,
                    comment=comment or "Approved knowledge release published atomically.",
                )
            )
            return self._persist_record(payload, record)

    def rollback(self, change_id: str, *, actor: str, comment: str) -> KnowledgeChangeRequest:
        with self._lock:
            payload = self._load()
            record = self._record(payload, change_id)
            if record.status != KnowledgeChangeStatus.PUBLISHED:
                raise ValueError("only PUBLISHED changes can be rolled back")
            if not record.previous_release_snapshot:
                raise ValueError("no previous release snapshot is available for rollback")
            current = self._active_release()
            if current and current.release_id != record.candidate_release_id:
                raise ValueError("active release no longer matches this change; rollback is unsafe")
            previous = KnowledgeRelease.model_validate(record.previous_release_snapshot)
            self._archive(previous)
            self._write_release_atomic(previous)
            record.status = KnowledgeChangeStatus.ROLLED_BACK
            record.transitions.append(
                KnowledgeChangeTransition(status=record.status, actor=actor, comment=comment)
            )
            return self._persist_record(payload, record)

    def get(self, change_id: str) -> KnowledgeChangeRequest | None:
        raw = self._load()["changes"].get(change_id)
        return None if raw is None else KnowledgeChangeRequest.model_validate(raw)

    def list_records(self) -> list[KnowledgeChangeRequest]:
        return sorted(
            [KnowledgeChangeRequest.model_validate(item) for item in self._load()["changes"].values()],
            key=lambda item: (item.created_at, item.change_id),
        )
