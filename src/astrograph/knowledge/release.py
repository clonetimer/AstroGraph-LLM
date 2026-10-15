from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class KnowledgeRelease(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-knowledge-release.v1"
    release_id: str = Field(min_length=1)
    knowledge_version: str = Field(min_length=1)
    ontology_version: str = Field(min_length=1)
    release_status: Literal["draft", "engineering_released", "retired"]
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    edges_file: str
    edges_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fault_modes_file: str = "data/knowledge_seed/fault_modes.csv"
    anomalies_file: str = "data/knowledge_seed/anomalies.csv"
    evidence_levels: list[str] = Field(default_factory=list)
    review_statuses: list[str] = Field(default_factory=list)
    edge_count: int = Field(ge=0)
    claim_boundary: str

    @model_validator(mode="after")
    def validate_levels(self) -> "KnowledgeRelease":
        invalid = set(self.evidence_levels) - {"E0", "E1", "E2", "E3", "E4"}
        if invalid:
            raise ValueError(f"invalid evidence levels: {sorted(invalid)}")
        return self

    @classmethod
    def from_json(cls, path: str | Path) -> "KnowledgeRelease":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_knowledge_release(release: KnowledgeRelease, project_root: str | Path) -> dict:
    root = Path(project_root)
    edges_path = root / release.edges_file
    checks: list[dict] = []

    def add(check_id: str, passed: bool, message: str, **details) -> None:
        checks.append({"check_id": check_id, "status": "PASS" if passed else "FAIL", "message": message, "details": details})

    add("edges_file_exists", edges_path.exists(), f"edges file: {edges_path}")
    if not edges_path.exists():
        return {"status": "FAIL", "release_id": release.release_id, "checks": checks}
    actual_hash = sha256_file(edges_path)
    add("edges_sha256", actual_hash == release.edges_sha256, "edges SHA256 matches release", actual=actual_hash)

    rows = list(csv.DictReader(edges_path.open(encoding="utf-8-sig", newline="")))
    add("edge_count", len(rows) == release.edge_count, "edge count matches release", actual=len(rows))
    versions = {row.get("knowledge_version", "") for row in rows}
    add(
        "knowledge_version_consistency",
        versions == {release.knowledge_version},
        "all edges use the release knowledge_version",
        actual=sorted(versions),
    )
    keys = [(row.get("fault_id", ""), row.get("anomaly_id", "")) for row in rows]
    add("unique_edges", len(keys) == len(set(keys)), "fault-anomaly edges are unique")
    required = {"fault_id", "fault_name", "anomaly_id", "anomaly_name", "weight", "edge_role", "evidence_level", "review_status", "knowledge_version"}
    missing = [index + 2 for index, row in enumerate(rows) if not required.issubset(row) or any(row.get(key, "") == "" for key in required)]
    add("required_fields", not missing, "required edge fields are populated", bad_rows=missing[:20])
    levels = sorted({row.get("evidence_level", "") for row in rows})
    statuses = sorted({row.get("review_status", "") for row in rows})
    add("evidence_levels", levels == sorted(release.evidence_levels), "evidence level inventory matches", actual=levels)
    add("review_statuses", statuses == sorted(release.review_statuses), "review status inventory matches", actual=statuses)
    status = "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL"
    return {
        "schema_version": "astrograph-knowledge-release-validation.v1",
        "status": status,
        "release_id": release.release_id,
        "knowledge_version": release.knowledge_version,
        "release_status": release.release_status,
        "checks": checks,
        "claim_boundary": release.claim_boundary,
    }
