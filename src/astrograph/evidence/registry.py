from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from astrograph.audit.store import canonical_json
from astrograph.evidence.models import EdgeReviewRecord, EvidenceSourceRecord


def sha256_path(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class EvidenceRegistry:
    schema_version = "astrograph-evidence-registry.v1"

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.sources_path = self.root / "evidence_sources.json"
        self.reviews_path = self.root / "edge_reviews.json"

    @staticmethod
    def _empty(kind: str) -> dict[str, Any]:
        return {
            "schema_version": "astrograph-evidence-registry.v1",
            "registry_kind": kind,
            "updated_at": "",
            "records": [],
        }

    def _load(self, path: Path, kind: str) -> dict[str, Any]:
        if not path.exists():
            return self._empty(kind)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != self.schema_version:
            raise ValueError(f"unsupported evidence registry schema: {path}")
        if payload.get("registry_kind") != kind:
            raise ValueError(f"unexpected evidence registry kind: {path}")
        return payload

    @staticmethod
    def _stable(record: dict[str, Any]) -> dict[str, Any]:
        volatile = {"registered_at", "reviewed_at"}
        return {key: value for key, value in record.items() if key not in volatile}

    def _upsert(self, path: Path, kind: str, key_name: str, record: dict[str, Any]) -> None:
        payload = self._load(path, kind)
        records = payload["records"]
        key = record[key_name]
        for index, current in enumerate(records):
            if current[key_name] != key:
                continue
            if canonical_json(self._stable(current)) != canonical_json(self._stable(record)):
                raise ValueError(f"{key_name} collision with different content: {key}")
            records[index] = record
            break
        else:
            records.append(record)
        records.sort(key=lambda item: item[key_name])
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def upsert_source(self, source: EvidenceSourceRecord) -> None:
        self._upsert(
            self.sources_path,
            "evidence_sources",
            "source_id",
            source.model_dump(mode="json"),
        )

    def upsert_review(self, review: EdgeReviewRecord) -> None:
        self._upsert(
            self.reviews_path,
            "edge_reviews",
            "review_id",
            review.model_dump(mode="json"),
        )

    def list_sources(self) -> list[EvidenceSourceRecord]:
        payload = self._load(self.sources_path, "evidence_sources")
        return [EvidenceSourceRecord.model_validate(item) for item in payload["records"]]

    def list_reviews(self) -> list[EdgeReviewRecord]:
        payload = self._load(self.reviews_path, "edge_reviews")
        return [EdgeReviewRecord.model_validate(item) for item in payload["records"]]

    def get_source(self, source_id: str) -> EvidenceSourceRecord | None:
        return next((item for item in self.list_sources() if item.source_id == source_id), None)
