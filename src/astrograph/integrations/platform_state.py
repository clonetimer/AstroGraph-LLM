from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


SCHEMA_VERSION = 3


class PlatformStateRepository:
    """SQLite control-plane repository for the local AstroGraph platform.

    Large diagnosis artifacts remain file-backed. SQLite stores queryable control
    state: service profiles, provider registrations, maintenance projections,
    temporal localization, annotations, and job indexes.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._migrate()

    @classmethod
    def from_environment(cls, *, platform_root: str | Path | None = None) -> "PlatformStateRepository":
        configured = os.environ.get("ASTROGRAPH_PLATFORM_STATE_DB", "").strip()
        if configured:
            return cls(configured)
        root = Path(platform_root).expanduser().resolve() if platform_root else Path(
            os.environ.get("ASTROGRAPH_PLATFORM_ROOT") or (Path.home() / ".astrograph" / "platform")
        ).expanduser().resolve()
        return cls(root / "control_plane.db")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            connection = sqlite3.connect(self.path, timeout=30.0)
            connection.row_factory = sqlite3.Row
            try:
                connection.execute("PRAGMA busy_timeout=30000")
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("PRAGMA synchronous=NORMAL")
                connection.execute("PRAGMA foreign_keys=ON")
                yield connection
                connection.commit()
            finally:
                connection.close()

    def _migrate(self) -> None:
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS service_profiles (
                    profile_id TEXT PRIMARY KEY,
                    service_kind TEXT NOT NULL,
                    name TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    secret_ref TEXT,
                    validation_status TEXT NOT NULL DEFAULT 'UNKNOWN',
                    verified_at TEXT,
                    is_default INTEGER NOT NULL DEFAULT 0 CHECK(is_default IN (0,1)),
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(service_kind, name)
                )
                """
            )
            db.execute("CREATE INDEX IF NOT EXISTS idx_service_profiles_kind ON service_profiles(service_kind, is_default DESC, updated_at DESC)")
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_providers (
                    provider_id TEXT PRIMARY KEY,
                    provider_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    secret_ref TEXT,
                    enabled INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN (0,1)),
                    validation_status TEXT NOT NULL DEFAULT 'UNKNOWN',
                    verified_at TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS job_index (
                    job_id TEXT PRIMARY KEY,
                    task TEXT,
                    work_mode TEXT,
                    status TEXT,
                    artifact_kind TEXT,
                    candidate TEXT,
                    disposition TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    report_path TEXT
                )
                """
            )
            db.execute("CREATE INDEX IF NOT EXISTS idx_job_index_updated ON job_index(updated_at DESC)")
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS temporal_localizations (
                    job_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS maintenance_presentations (
                    job_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS multidimensional_assessments (
                    job_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS maintenance_annotations (
                    annotation_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    annotation_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    actor TEXT,
                    created_at REAL NOT NULL
                )
                """
            )
            db.execute("CREATE INDEX IF NOT EXISTS idx_annotations_job ON maintenance_annotations(job_id, created_at DESC)")
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS diagnostic_history_jobs (
                    job_id TEXT PRIMARY KEY,
                    fault_id TEXT,
                    first_detection_s REAL,
                    confirmed_detection_s REAL,
                    temporal_method TEXT,
                    temporal_primary_source TEXT,
                    temporal_agreement TEXT,
                    updated_at REAL NOT NULL
                )
                """
            )
            db.execute("CREATE INDEX IF NOT EXISTS idx_diag_hist_fault ON diagnostic_history_jobs(fault_id, updated_at DESC)")
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS diagnostic_dimension_index (
                    job_id TEXT NOT NULL,
                    fault_id TEXT,
                    dimension TEXT NOT NULL,
                    label_zh TEXT,
                    level TEXT,
                    index_value REAL,
                    affected_channels INTEGER,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(job_id, dimension),
                    FOREIGN KEY(job_id) REFERENCES diagnostic_history_jobs(job_id) ON DELETE CASCADE
                )
                """
            )
            db.execute("CREATE INDEX IF NOT EXISTS idx_diag_dimension_fault ON diagnostic_dimension_index(fault_id, dimension, level)")
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS diagnostic_channel_index (
                    job_id TEXT NOT NULL,
                    fault_id TEXT,
                    channel TEXT NOT NULL,
                    label_zh TEXT,
                    dimension TEXT NOT NULL,
                    level TEXT,
                    index_value REAL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(job_id, channel, dimension),
                    FOREIGN KEY(job_id) REFERENCES diagnostic_history_jobs(job_id) ON DELETE CASCADE
                )
                """
            )
            db.execute("CREATE INDEX IF NOT EXISTS idx_diag_channel_fault ON diagnostic_channel_index(fault_id, channel, level)")
            db.execute(
                "INSERT OR REPLACE INTO schema_meta(key,value) VALUES('schema_version',?)",
                (str(SCHEMA_VERSION),),
            )

    @staticmethod
    def _decode(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        result = dict(row)
        for key in ("config_json", "payload_json"):
            if key in result:
                result[key.removesuffix("_json")] = json.loads(result.pop(key))
        for key in ("is_default", "enabled"):
            if key in result:
                result[key] = bool(result[key])
        return result

    def save_service_profile(
        self, *, service_kind: str, name: str, config: dict[str, Any], secret_ref: str | None = None,
        validation_status: str = "UNKNOWN", verified_at: str | None = None, make_default: bool = False,
        profile_id: str | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        service_kind = service_kind.strip().lower()
        name = name.strip()
        if not service_kind or not name:
            raise ValueError("service_kind and name are required")
        with self.connect() as db:
            existing = db.execute(
                "SELECT profile_id, created_at FROM service_profiles WHERE service_kind=? AND name=?",
                (service_kind, name),
            ).fetchone()
            pid = str(existing["profile_id"]) if existing else (profile_id or "PROFILE-" + uuid.uuid4().hex[:20].upper())
            created = float(existing["created_at"]) if existing else now
            if make_default:
                db.execute("UPDATE service_profiles SET is_default=0 WHERE service_kind=?", (service_kind,))
            db.execute(
                """
                INSERT INTO service_profiles(profile_id,service_kind,name,config_json,secret_ref,validation_status,verified_at,is_default,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(profile_id) DO UPDATE SET
                    service_kind=excluded.service_kind,name=excluded.name,config_json=excluded.config_json,
                    secret_ref=COALESCE(excluded.secret_ref, service_profiles.secret_ref),validation_status=excluded.validation_status,
                    verified_at=excluded.verified_at,is_default=excluded.is_default,updated_at=excluded.updated_at
                """,
                (pid, service_kind, name, json.dumps(config, ensure_ascii=False, sort_keys=True), secret_ref,
                 validation_status, verified_at, int(make_default), created, now),
            )
        return self.get_service_profile(pid) or {}

    def list_service_profiles(self, service_kind: str | None = None) -> list[dict[str, Any]]:
        with self.connect() as db:
            if service_kind:
                rows = db.execute(
                    "SELECT * FROM service_profiles WHERE service_kind=? ORDER BY is_default DESC, updated_at DESC",
                    (service_kind.strip().lower(),),
                ).fetchall()
            else:
                rows = db.execute("SELECT * FROM service_profiles ORDER BY service_kind, is_default DESC, updated_at DESC").fetchall()
        return [self._decode(row) or {} for row in rows]

    def get_service_profile(self, profile_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM service_profiles WHERE profile_id=?", (profile_id,)).fetchone()
        return self._decode(row)

    def default_service_profile(self, service_kind: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM service_profiles WHERE service_kind=? ORDER BY is_default DESC, updated_at DESC LIMIT 1",
                (service_kind.strip().lower(),),
            ).fetchone()
        return self._decode(row)

    def set_default_service_profile(self, profile_id: str) -> dict[str, Any]:
        profile = self.get_service_profile(profile_id)
        if profile is None:
            raise KeyError(profile_id)
        with self.connect() as db:
            db.execute("UPDATE service_profiles SET is_default=0 WHERE service_kind=?", (profile["service_kind"],))
            db.execute("UPDATE service_profiles SET is_default=1, updated_at=? WHERE profile_id=?", (time.time(), profile_id))
        return self.get_service_profile(profile_id) or {}

    def delete_service_profile(self, profile_id: str) -> bool:
        with self.connect() as db:
            return db.execute("DELETE FROM service_profiles WHERE profile_id=?", (profile_id,)).rowcount == 1

    def save_tool_provider(
        self, *, provider_id: str, provider_type: str, name: str, config: dict[str, Any],
        secret_ref: str | None = None, enabled: bool = True, validation_status: str = "UNKNOWN",
        verified_at: str | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        with self.connect() as db:
            existing = db.execute("SELECT created_at FROM tool_providers WHERE provider_id=?", (provider_id,)).fetchone()
            created = float(existing["created_at"]) if existing else now
            db.execute(
                """
                INSERT INTO tool_providers(provider_id,provider_type,name,config_json,secret_ref,enabled,validation_status,verified_at,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(provider_id) DO UPDATE SET provider_type=excluded.provider_type,name=excluded.name,
                    config_json=excluded.config_json,secret_ref=COALESCE(excluded.secret_ref, tool_providers.secret_ref),enabled=excluded.enabled,
                    validation_status=excluded.validation_status,verified_at=excluded.verified_at,updated_at=excluded.updated_at
                """,
                (provider_id, provider_type, name, json.dumps(config, ensure_ascii=False, sort_keys=True), secret_ref,
                 int(enabled), validation_status, verified_at, created, now),
            )
        return self.get_tool_provider(provider_id) or {}

    def get_tool_provider(self, provider_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM tool_providers WHERE provider_id=?", (provider_id,)).fetchone()
        return self._decode(row)

    def list_tool_providers(self, *, enabled_only: bool = False) -> list[dict[str, Any]]:
        query = "SELECT * FROM tool_providers" + (" WHERE enabled=1" if enabled_only else "") + " ORDER BY provider_type,name"
        with self.connect() as db:
            rows = db.execute(query).fetchall()
        return [self._decode(row) or {} for row in rows]

    def delete_tool_provider(self, provider_id: str) -> bool:
        with self.connect() as db:
            return db.execute("DELETE FROM tool_providers WHERE provider_id=?", (provider_id,)).rowcount == 1

    def upsert_job(self, payload: dict[str, Any], *, report_path: str | None = None) -> None:
        candidate = None
        disposition = None
        if payload.get("status") == "SUCCEEDED" and isinstance(payload.get("analysis"), dict):
            analysis = payload["analysis"]
            pipeline = analysis.get("pipeline_result") or {}
            diagnosis = analysis.get("diagnosis") or {}
            candidate = pipeline.get("fault_id") or (pipeline.get("fusion") or {}).get("top_fault_id") or diagnosis.get("selected_fault_id")
            disposition = pipeline.get("disposition") or diagnosis.get("disposition")
        with self.connect() as db:
            db.execute(
                """
                INSERT INTO job_index(job_id,task,work_mode,status,artifact_kind,candidate,disposition,created_at,updated_at,report_path)
                VALUES(?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(job_id) DO UPDATE SET task=excluded.task,work_mode=excluded.work_mode,status=excluded.status,
                    artifact_kind=excluded.artifact_kind,candidate=COALESCE(excluded.candidate,job_index.candidate),
                    disposition=COALESCE(excluded.disposition,job_index.disposition),updated_at=excluded.updated_at,
                    report_path=COALESCE(excluded.report_path,job_index.report_path)
                """,
                (payload.get("job_id"), payload.get("task"), payload.get("work_mode"), payload.get("status"),
                 payload.get("artifact_kind"), candidate, disposition, payload.get("created_at"), payload.get("updated_at") or payload.get("completed_at"), report_path),
            )

    def save_temporal_localization(self, job_id: str, payload: dict[str, Any]) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO temporal_localizations(job_id,payload_json,updated_at) VALUES(?,?,?) ON CONFLICT(job_id) DO UPDATE SET payload_json=excluded.payload_json,updated_at=excluded.updated_at",
                (job_id, json.dumps(payload, ensure_ascii=False, sort_keys=True), time.time()),
            )

    def get_temporal_localization(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM temporal_localizations WHERE job_id=?", (job_id,)).fetchone()
        decoded = self._decode(row)
        return None if decoded is None else decoded.get("payload")

    def save_multidimensional_assessment(self, job_id: str, payload: dict[str, Any]) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO multidimensional_assessments(job_id,payload_json,updated_at) VALUES(?,?,?) ON CONFLICT(job_id) DO UPDATE SET payload_json=excluded.payload_json,updated_at=excluded.updated_at",
                (job_id, json.dumps(payload, ensure_ascii=False, sort_keys=True), time.time()),
            )

    def get_multidimensional_assessment(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM multidimensional_assessments WHERE job_id=?", (job_id,)).fetchone()
        decoded = self._decode(row)
        return None if decoded is None else decoded.get("payload")

    def save_maintenance_presentation(self, job_id: str, payload: dict[str, Any]) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO maintenance_presentations(job_id,payload_json,updated_at) VALUES(?,?,?) ON CONFLICT(job_id) DO UPDATE SET payload_json=excluded.payload_json,updated_at=excluded.updated_at",
                (job_id, json.dumps(payload, ensure_ascii=False, sort_keys=True), time.time()),
            )

    def get_maintenance_presentation(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM maintenance_presentations WHERE job_id=?", (job_id,)).fetchone()
        decoded = self._decode(row)
        return None if decoded is None else decoded.get("payload")

    def add_annotation(self, *, job_id: str, annotation_type: str, payload: dict[str, Any], actor: str = "operator") -> dict[str, Any]:
        annotation_id = "ANN-" + uuid.uuid4().hex[:20].upper()
        created = time.time()
        with self.connect() as db:
            db.execute(
                "INSERT INTO maintenance_annotations(annotation_id,job_id,annotation_type,payload_json,actor,created_at) VALUES(?,?,?,?,?,?)",
                (annotation_id, job_id, annotation_type, json.dumps(payload, ensure_ascii=False, sort_keys=True), actor, created),
            )
        return {"annotation_id": annotation_id, "job_id": job_id, "annotation_type": annotation_type, "payload": payload, "actor": actor, "created_at": created}

    def annotations(self, job_id: str) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute("SELECT * FROM maintenance_annotations WHERE job_id=? ORDER BY created_at", (job_id,)).fetchall()
        return [self._decode(row) or {} for row in rows]


    def index_diagnostic_history(
        self,
        *,
        job_id: str,
        fault_id: str | None,
        temporal: dict[str, Any],
        multidimensional: dict[str, Any],
    ) -> None:
        """Normalize maintenance results for cross-task comparison.

        Raw telemetry remains artifact-backed. This index stores only bounded summary
        rows so history queries do not need to scan report JSON blobs.
        """
        now = time.time()
        with self.connect() as db:
            db.execute(
                """
                INSERT INTO diagnostic_history_jobs(job_id,fault_id,first_detection_s,confirmed_detection_s,temporal_method,temporal_primary_source,temporal_agreement,updated_at)
                VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(job_id) DO UPDATE SET fault_id=excluded.fault_id,first_detection_s=excluded.first_detection_s,
                    confirmed_detection_s=excluded.confirmed_detection_s,temporal_method=excluded.temporal_method,
                    temporal_primary_source=excluded.temporal_primary_source,temporal_agreement=excluded.temporal_agreement,updated_at=excluded.updated_at
                """,
                (
                    job_id, fault_id, temporal.get("first_detection_s"), temporal.get("confirmed_detection_s"),
                    temporal.get("method"), temporal.get("primary_source"), temporal.get("agreement"), now,
                ),
            )
            db.execute("DELETE FROM diagnostic_dimension_index WHERE job_id=?", (job_id,))
            db.execute("DELETE FROM diagnostic_channel_index WHERE job_id=?", (job_id,))
            for item in multidimensional.get("dimension_summary") or []:
                db.execute(
                    "INSERT INTO diagnostic_dimension_index(job_id,fault_id,dimension,label_zh,level,index_value,affected_channels,updated_at) VALUES(?,?,?,?,?,?,?,?)",
                    (
                        job_id, fault_id, item.get("dimension"), item.get("label_zh"), item.get("level"),
                        item.get("index"), item.get("affected_channels"), now,
                    ),
                )
            for channel in multidimensional.get("channels") or []:
                channel_id = str(channel.get("channel") or "")
                if not channel_id:
                    continue
                for dim in channel.get("dimensions") or []:
                    db.execute(
                        "INSERT INTO diagnostic_channel_index(job_id,fault_id,channel,label_zh,dimension,level,index_value,updated_at) VALUES(?,?,?,?,?,?,?,?)",
                        (
                            job_id, fault_id, channel_id, channel.get("label_zh"), dim.get("dimension"), dim.get("level"), dim.get("index"), now,
                        ),
                    )

    def diagnostic_history_summary(self, *, fault_id: str | None = None, limit: int = 200) -> dict[str, Any]:
        limit = max(1, min(int(limit), 2000))
        where = " WHERE fault_id=?" if fault_id else ""
        params: tuple[Any, ...] = (fault_id,) if fault_id else ()
        with self.connect() as db:
            jobs = db.execute(
                f"SELECT * FROM diagnostic_history_jobs{where} ORDER BY updated_at DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
            selected_ids = [str(row["job_id"]) for row in jobs]
            if not selected_ids:
                return {
                    "schema_version": "astrograph-diagnostic-history-summary.v1",
                    "fault_id": fault_id,
                    "sample_count": 0,
                    "dimensions": [],
                    "channels": [],
                    "temporal": {"count": 0},
                }
            placeholders = ",".join("?" for _ in selected_ids)
            dims = db.execute(
                f"SELECT dimension,MAX(label_zh) label_zh,COUNT(*) samples,SUM(CASE WHEN level='SIGNIFICANT' THEN 1 ELSE 0 END) significant_count,SUM(CASE WHEN level='WATCH' THEN 1 ELSE 0 END) watch_count,AVG(index_value) mean_index,MAX(index_value) max_index FROM diagnostic_dimension_index WHERE job_id IN ({placeholders}) GROUP BY dimension ORDER BY significant_count DESC, watch_count DESC, mean_index DESC",
                tuple(selected_ids),
            ).fetchall()
            channels = db.execute(
                f"SELECT channel,MAX(label_zh) label_zh,COUNT(DISTINCT job_id) samples,SUM(CASE WHEN level='SIGNIFICANT' THEN 1 ELSE 0 END) significant_events,SUM(CASE WHEN level='WATCH' THEN 1 ELSE 0 END) watch_events,COUNT(DISTINCT CASE WHEN level!='STABLE' THEN dimension END) affected_dimensions,MAX(index_value) max_index FROM diagnostic_channel_index WHERE job_id IN ({placeholders}) GROUP BY channel HAVING significant_events>0 OR watch_events>0 ORDER BY significant_events DESC, watch_events DESC, max_index DESC LIMIT 12",
                tuple(selected_ids),
            ).fetchall()
        detections = [float(row["first_detection_s"]) for row in jobs if row["first_detection_s"] is not None]
        detections_sorted = sorted(detections)
        temporal = {"count": len(detections)}
        if detections_sorted:
            mid = len(detections_sorted)//2
            temporal.update({
                "minimum_s": detections_sorted[0],
                "maximum_s": detections_sorted[-1],
                "median_s": detections_sorted[mid] if len(detections_sorted)%2 else (detections_sorted[mid-1]+detections_sorted[mid])/2.0,
            })
        return {
            "schema_version": "astrograph-diagnostic-history-summary.v1",
            "fault_id": fault_id,
            "sample_count": len(jobs),
            "dimensions": [dict(row) for row in dims],
            "channels": [dict(row) for row in channels],
            "temporal": temporal,
            "boundary_zh": "历史聚合仅用于维护对照与阈值研究，不构成新的诊断概率或正式证据投票。",
        }
