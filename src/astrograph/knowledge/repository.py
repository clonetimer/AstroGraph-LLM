from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Protocol

from astrograph.knowledge.release import KnowledgeRelease, validate_knowledge_release
from astrograph.schemas import KnowledgeEdge


class KnowledgeRepository(Protocol):
    backend_id: str

    @property
    def knowledge_version(self) -> str: ...

    def retrieve(self, anomaly_ids: list[str]) -> list[KnowledgeEdge]: ...


class CsvKnowledgeRepository:
    backend_id = "csv"

    def __init__(
        self,
        edges_csv: str | Path,
        *,
        knowledge_version: str = "astrograph-ontology-v1-e0",
        release_path: str | Path | None = None,
        project_root: str | Path | None = None,
    ):
        self.edges_csv = Path(edges_csv)
        self.release = KnowledgeRelease.from_json(release_path) if release_path else None
        if self.release:
            root = Path(project_root) if project_root else self.edges_csv.resolve().parents[2]
            validation = validate_knowledge_release(self.release, root)
            if validation["status"] != "PASS":
                raise ValueError(f"knowledge release validation failed: {validation}")
            if self.release.knowledge_version != knowledge_version:
                raise ValueError("repository knowledge_version does not match release")
        self._knowledge_version = knowledge_version

    @property
    def knowledge_version(self) -> str:
        return self._knowledge_version

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.edges_csv.read_bytes()).hexdigest()

    def health_check(self) -> dict:
        required_columns = {
            "fault_id",
            "fault_name",
            "anomaly_id",
            "anomaly_name",
            "weight",
        }
        if not self.edges_csv.is_file():
            return {
                "status": "FAIL",
                "knowledge_version": self._knowledge_version,
                "edges_csv": str(self.edges_csv),
                "error": "KNOWLEDGE_EDGE_FILE_MISSING",
            }
        try:
            with self.edges_csv.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                columns = set(reader.fieldnames or [])
                missing = sorted(required_columns - columns)
                row_count = sum(1 for _ in reader)
            if missing:
                return {
                    "status": "FAIL",
                    "knowledge_version": self._knowledge_version,
                    "edges_csv": str(self.edges_csv),
                    "missing_columns": missing,
                    "row_count": row_count,
                }
            return {
                "status": "PASS" if row_count > 0 else "FAIL",
                "knowledge_version": self._knowledge_version,
                "edges_csv": str(self.edges_csv),
                "row_count": row_count,
                "content_hash": self.content_hash,
                "error": None if row_count > 0 else "KNOWLEDGE_EDGE_FILE_EMPTY",
            }
        except Exception as exc:
            return {
                "status": "FAIL",
                "knowledge_version": self._knowledge_version,
                "edges_csv": str(self.edges_csv),
                "error": f"{type(exc).__name__}: {exc}",
            }

    def retrieve(self, anomaly_ids: list[str]) -> list[KnowledgeEdge]:
        wanted = set(anomaly_ids)
        edges: list[KnowledgeEdge] = []
        seen: set[tuple[str, str]] = set()
        with self.edges_csv.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["anomaly_id"] not in wanted:
                    continue
                key = (row["fault_id"], row["anomaly_id"])
                if key in seen:
                    raise ValueError(f"Duplicate knowledge edge: {key}")
                seen.add(key)
                edges.append(
                    KnowledgeEdge(
                        fault_id=row["fault_id"],
                        fault_name=row["fault_name"],
                        anomaly_id=row["anomaly_id"],
                        anomaly_name=row["anomaly_name"],
                        weight=float(row["weight"]),
                        edge_role=row.get("edge_role", "direct"),
                        evidence_level=row.get("evidence_level", "E0"),
                        evidence_source=row.get("evidence_source", "synthetic_quick_campaign_seed"),
                        source_ref=row.get("source_ref", ""),
                        review_status=row.get("review_status", "provisional"),
                        provisional=row.get("provisional", "true").strip().lower() == "true",
                        knowledge_version=row.get("knowledge_version") or self._knowledge_version,
                        applicable_condition_ids=[item for item in row.get("applicable_condition_ids", "").split("|") if item],
                        excluded_condition_ids=[item for item in row.get("excluded_condition_ids", "").split("|") if item],
                    )
                )
        return sorted(edges, key=lambda item: (item.fault_id, item.anomaly_id))


class Neo4jKnowledgeRepository:
    """Neo4j-backed repository. It never falls back to mock data on failure."""

    backend_id = "neo4j"

    def __init__(self, uri: str, user: str, password: str, *, knowledge_version: str = "astrograph-ontology-v1-e0"):
        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._knowledge_version = knowledge_version

    @property
    def knowledge_version(self) -> str:
        return self._knowledge_version

    def close(self) -> None:
        self.driver.close()

    def health_check(self) -> dict:
        self.driver.verify_connectivity()
        query = """
        OPTIONAL MATCH (r:KnowledgeRelease {knowledge_version: $knowledge_version})
        WITH count(r) AS release_count
        OPTIONAL MATCH (:FaultMode)-[edge:CAUSES {knowledge_version: $knowledge_version}]->(:Anomaly)
        RETURN release_count, count(edge) AS edge_count
        """
        with self.driver.session() as session:
            record = session.run(query, knowledge_version=self._knowledge_version).single()
            return {
                "status": "PASS" if record and record["release_count"] >= 1 and record["edge_count"] >= 1 else "FAIL",
                "knowledge_version": self._knowledge_version,
                "release_count": 0 if record is None else int(record["release_count"]),
                "edge_count": 0 if record is None else int(record["edge_count"]),
            }

    def retrieve(self, anomaly_ids: list[str]) -> list[KnowledgeEdge]:
        query = """
        MATCH (f:FaultMode)-[r:CAUSES]->(a:Anomaly)
        WHERE a.id IN $anomaly_ids
          AND coalesce(r.knowledge_version, $knowledge_version) = $knowledge_version
        RETURN f.id AS fault_id, f.name AS fault_name,
               a.id AS anomaly_id, a.name AS anomaly_name,
               r.weight AS weight, r.edge_role AS edge_role,
               r.evidence_level AS evidence_level,
               r.evidence_source AS evidence_source,
               coalesce(r.source_ref, '') AS source_ref,
               coalesce(r.review_status, 'provisional') AS review_status,
               coalesce(r.provisional, true) AS provisional,
               coalesce(r.knowledge_version, $knowledge_version) AS knowledge_version,
               coalesce(r.applicable_condition_ids, []) AS applicable_condition_ids,
               coalesce(r.excluded_condition_ids, []) AS excluded_condition_ids
        ORDER BY f.id, a.id
        """
        with self.driver.session() as session:
            records = session.run(
                query,
                anomaly_ids=anomaly_ids,
                knowledge_version=self._knowledge_version,
            )
            return [KnowledgeEdge(**dict(record)) for record in records]


def csv_repository_from_release(
    project_root: str | Path,
    release_path: str | Path = "data/knowledge_releases/active_release.json",
) -> CsvKnowledgeRepository:
    """Build a CSV repository from the release manifest instead of a hard-coded seed file."""
    root = Path(project_root).resolve()
    manifest_path = Path(release_path)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    release = KnowledgeRelease.from_json(manifest_path)
    return CsvKnowledgeRepository(
        root / release.edges_file,
        knowledge_version=release.knowledge_version,
        release_path=manifest_path,
        project_root=root,
    )
