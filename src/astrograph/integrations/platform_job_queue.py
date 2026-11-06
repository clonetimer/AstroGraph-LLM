"""SQLite-backed multi-process queue for external-diagnosis platform jobs.

The queue stores control state only. Input and report artifacts remain file-backed.
A worker lease expiring never causes an automatic paid-model retry: the job becomes
INTERRUPTED and requires an explicit, audited operator resume.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sqlite3
import time
from typing import Any, Iterator


ACTIVE_STATES = ("RUNNING", "CANCEL_REQUESTED", "COMMITTING")
TERMINAL_STATES = ("SUCCEEDED", "FAILED", "CANCELLED", "INTERRUPTED")


class PlatformQueueConflict(ValueError):
    """The requested queue transition is not valid for the current state."""


class PersistentPlatformJobQueue:
    def __init__(self, platform_root: str | Path) -> None:
        self.root = Path(platform_root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "platform_job_queue.sqlite3"
        self._migrate()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(str(self.path), timeout=30.0, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA busy_timeout=30000")
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            yield db
        finally:
            db.close()

    def _migrate(self) -> None:
        with self._connect() as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS platform_job_queue (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL CHECK(status IN (
                        'QUEUED','RUNNING','CANCEL_REQUESTED','COMMITTING',
                        'SUCCEEDED','FAILED','CANCELLED','INTERRUPTED'
                    )),
                    queued_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    started_at REAL,
                    completed_at REAL,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    recovery_count INTEGER NOT NULL DEFAULT 0,
                    worker_id TEXT,
                    heartbeat_at REAL,
                    lease_expires_at REAL,
                    cancel_requested_at REAL,
                    cancel_reason TEXT,
                    recovery_actor TEXT,
                    recovery_confirmed_at REAL,
                    last_error_code TEXT
                )
                """
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS idx_platform_queue_claim "
                "ON platform_job_queue(status, queued_at, job_id)"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS idx_platform_queue_lease "
                "ON platform_job_queue(status, lease_expires_at)"
            )

    @staticmethod
    def _decode(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return dict(row) if row is not None else None

    @staticmethod
    def _recover_expired_locked(db: sqlite3.Connection, now: float) -> list[str]:
        rows = db.execute(
            """
            SELECT job_id,status FROM platform_job_queue
            WHERE status IN ('RUNNING','COMMITTING','CANCEL_REQUESTED')
              AND lease_expires_at IS NOT NULL AND lease_expires_at<=?
            """,
            (now,),
        ).fetchall()
        recovered = []
        for row in rows:
            job_id = str(row["job_id"])
            previous = str(row["status"])
            status = "CANCELLED" if previous == "CANCEL_REQUESTED" else "INTERRUPTED"
            error = "CANCELLED_AFTER_WORKER_EXIT" if status == "CANCELLED" else "WORKER_LEASE_EXPIRED"
            db.execute(
                """
                UPDATE platform_job_queue
                SET status=?,updated_at=?,completed_at=?,worker_id=NULL,
                    heartbeat_at=NULL,lease_expires_at=NULL,last_error_code=?
                WHERE job_id=? AND status=? AND lease_expires_at<=?
                """,
                (status, now, now, error, job_id, previous, now),
            )
            recovered.append(job_id)
        return recovered

    def recover_expired(self, *, now: float | None = None) -> list[str]:
        timestamp = time.time() if now is None else float(now)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            return self._recover_expired_locked(db, timestamp)

    def enqueue(self, job_id: str, *, allow_failed_retry: bool = True) -> dict[str, Any]:
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM platform_job_queue WHERE job_id=?", (job_id,)
            ).fetchone()
            if row is None:
                db.execute(
                    """
                    INSERT INTO platform_job_queue(job_id,status,queued_at,updated_at)
                    VALUES(?,?,?,?)
                    """,
                    (job_id, "QUEUED", now, now),
                )
            else:
                status = str(row["status"])
                if status == "FAILED" and allow_failed_retry:
                    db.execute(
                        """
                        UPDATE platform_job_queue SET status='QUEUED',queued_at=?,updated_at=?,
                            completed_at=NULL,worker_id=NULL,heartbeat_at=NULL,lease_expires_at=NULL,
                            cancel_requested_at=NULL,cancel_reason=NULL,last_error_code=NULL
                        WHERE job_id=? AND status='FAILED'
                        """,
                        (now, now, job_id),
                    )
                elif status in ("QUEUED", "RUNNING", "CANCEL_REQUESTED", "COMMITTING", "SUCCEEDED"):
                    pass
                elif status == "INTERRUPTED":
                    raise PlatformQueueConflict("INTERRUPTED_JOB_REQUIRES_EXPLICIT_RESUME")
                elif status == "CANCELLED":
                    raise PlatformQueueConflict("CANCELLED_JOB_CANNOT_BE_RESTARTED")
                else:
                    raise PlatformQueueConflict("INVALID_QUEUE_STATE")
            result = db.execute(
                "SELECT * FROM platform_job_queue WHERE job_id=?", (job_id,)
            ).fetchone()
            return self._decode(result) or {}

    def claim_next(
        self,
        worker_id: str,
        *,
        lease_seconds: float = 30.0,
        max_running: int = 2,
        now: float | None = None,
    ) -> dict[str, Any] | None:
        if not worker_id or lease_seconds <= 0 or max_running < 1:
            raise ValueError("INVALID_QUEUE_CLAIM")
        timestamp = time.time() if now is None else float(now)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._recover_expired_locked(db, timestamp)
            active = db.execute(
                """
                SELECT COUNT(*) AS n FROM platform_job_queue
                WHERE status IN ('RUNNING','CANCEL_REQUESTED','COMMITTING')
                  AND lease_expires_at>?
                """,
                (timestamp,),
            ).fetchone()
            if int(active["n"] or 0) >= max_running:
                return None
            row = db.execute(
                "SELECT job_id FROM platform_job_queue WHERE status='QUEUED' "
                "ORDER BY queued_at,job_id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            job_id = str(row["job_id"])
            changed = db.execute(
                """
                UPDATE platform_job_queue
                SET status='RUNNING',worker_id=?,attempt=attempt+1,
                    started_at=COALESCE(started_at,?),heartbeat_at=?,
                    lease_expires_at=?,updated_at=?,last_error_code=NULL
                WHERE job_id=? AND status='QUEUED'
                """,
                (worker_id, timestamp, timestamp, timestamp + lease_seconds, timestamp, job_id),
            ).rowcount
            if changed != 1:
                return None
            claimed = db.execute(
                "SELECT * FROM platform_job_queue WHERE job_id=?", (job_id,)
            ).fetchone()
            return self._decode(claimed)

    def heartbeat(
        self, job_id: str, worker_id: str, *, lease_seconds: float = 30.0,
        now: float | None = None,
    ) -> dict[str, Any]:
        timestamp = time.time() if now is None else float(now)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                """
                UPDATE platform_job_queue
                SET heartbeat_at=?,lease_expires_at=?,updated_at=?
                WHERE job_id=? AND worker_id=?
                  AND status IN ('RUNNING','CANCEL_REQUESTED','COMMITTING')
                """,
                (timestamp, timestamp + lease_seconds, timestamp, job_id, worker_id),
            )
            row = db.execute(
                "SELECT * FROM platform_job_queue WHERE job_id=?", (job_id,)
            ).fetchone()
            decoded = self._decode(row)
            if not decoded or decoded.get("worker_id") != worker_id:
                return {"owned": False, "cancel_requested": True, "status": None}
            return {
                "owned": True,
                "cancel_requested": decoded["status"] == "CANCEL_REQUESTED",
                "status": decoded["status"],
                "lease_expires_at": decoded.get("lease_expires_at"),
            }

    def begin_publish(self, job_id: str, worker_id: str) -> bool:
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            changed = db.execute(
                """
                UPDATE platform_job_queue
                SET status='COMMITTING',updated_at=?,heartbeat_at=?,lease_expires_at=?
                WHERE job_id=? AND worker_id=? AND status='RUNNING'
                """,
                (now, now, now + 30.0, job_id, worker_id),
            ).rowcount
            return changed == 1

    def finish(
        self, job_id: str, worker_id: str, status: str, *,
        error_code: str | None = None,
    ) -> dict[str, Any]:
        if status not in ("SUCCEEDED", "FAILED", "CANCELLED"):
            raise ValueError("INVALID_QUEUE_TERMINAL_STATUS")
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM platform_job_queue WHERE job_id=?", (job_id,)
            ).fetchone()
            decoded = self._decode(row)
            if not decoded or decoded.get("worker_id") != worker_id:
                raise PlatformQueueConflict("QUEUE_WORKER_LEASE_LOST")
            current = str(decoded["status"])
            if current == "CANCEL_REQUESTED":
                status = "CANCELLED"
                error_code = error_code or "USER_CANCELLED"
            elif current == "COMMITTING" and status == "SUCCEEDED":
                pass
            elif current == "RUNNING" and status in ("FAILED", "CANCELLED"):
                pass
            elif current == "RUNNING" and status == "SUCCEEDED":
                raise PlatformQueueConflict("PUBLISH_GUARD_NOT_ENTERED")
            elif current != "COMMITTING":
                raise PlatformQueueConflict("INVALID_QUEUE_FINISH_STATE")
            db.execute(
                """
                UPDATE platform_job_queue SET status=?,updated_at=?,completed_at=?,
                    worker_id=NULL,heartbeat_at=NULL,lease_expires_at=?,last_error_code=?
                WHERE job_id=? AND worker_id=?
                """,
                (status, now, now, now, error_code, job_id, worker_id),
            )
            result = db.execute(
                "SELECT * FROM platform_job_queue WHERE job_id=?", (job_id,)
            ).fetchone()
            return self._decode(result) or {}

    def request_cancel(self, job_id: str, *, reason: str = "USER_REQUEST") -> dict[str, Any]:
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM platform_job_queue WHERE job_id=?", (job_id,)
            ).fetchone()
            decoded = self._decode(row)
            if decoded is None:
                raise KeyError(job_id)
            status = str(decoded["status"])
            if status == "QUEUED":
                next_status = "CANCELLED"
                db.execute(
                    """
                    UPDATE platform_job_queue SET status='CANCELLED',updated_at=?,
                        completed_at=?,cancel_requested_at=?,cancel_reason=?,last_error_code='USER_CANCELLED'
                    WHERE job_id=? AND status='QUEUED'
                    """,
                    (now, now, now, reason[:200], job_id),
                )
            elif status == "RUNNING":
                next_status = "CANCEL_REQUESTED"
                db.execute(
                    """
                    UPDATE platform_job_queue SET status='CANCEL_REQUESTED',updated_at=?,
                        cancel_requested_at=?,cancel_reason=?
                    WHERE job_id=? AND status='RUNNING'
                    """,
                    (now, now, reason[:200], job_id),
                )
            elif status == "INTERRUPTED":
                next_status = "CANCELLED"
                db.execute(
                    """
                    UPDATE platform_job_queue SET status='CANCELLED',updated_at=?,
                        completed_at=?,cancel_requested_at=?,cancel_reason=?,last_error_code='USER_CANCELLED'
                    WHERE job_id=? AND status='INTERRUPTED'
                    """,
                    (now, now, now, reason[:200], job_id),
                )
            elif status == "COMMITTING":
                next_status = "TOO_LATE"
            else:
                next_status = status
            result = db.execute(
                "SELECT * FROM platform_job_queue WHERE job_id=?", (job_id,)
            ).fetchone()
            value = self._decode(result) or {}
            value["cancel_result"] = next_status
            return value

    def resume(
        self, job_id: str, *, confirm_duplicate_risk: bool, actor: str,
    ) -> dict[str, Any]:
        if not confirm_duplicate_risk:
            raise PlatformQueueConflict("DUPLICATE_MODEL_COST_CONFIRMATION_REQUIRED")
        if not actor.strip():
            raise PlatformQueueConflict("RECOVERY_ACTOR_REQUIRED")
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM platform_job_queue WHERE job_id=?", (job_id,)
            ).fetchone()
            decoded = self._decode(row)
            if decoded is None:
                raise KeyError(job_id)
            if decoded["status"] != "INTERRUPTED":
                raise PlatformQueueConflict("ONLY_INTERRUPTED_JOBS_CAN_BE_RESUMED")
            db.execute(
                """
                UPDATE platform_job_queue SET status='QUEUED',queued_at=?,updated_at=?,
                    recovery_count=recovery_count+1,recovery_actor=?,recovery_confirmed_at=?,
                    worker_id=NULL,heartbeat_at=NULL,lease_expires_at=NULL
                WHERE job_id=? AND status='INTERRUPTED'
                """,
                (now, now, actor[:160], now, job_id),
            )
            result = db.execute(
                "SELECT * FROM platform_job_queue WHERE job_id=?", (job_id,)
            ).fetchone()
            return self._decode(result) or {}

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM platform_job_queue WHERE job_id=?", (job_id,)
            ).fetchone()
        return self._decode(row)

    def has_work(self) -> bool:
        now = time.time()
        with self._connect() as db:
            row = db.execute(
                """
                SELECT 1 FROM platform_job_queue
                WHERE status='QUEUED'
                   OR (status IN ('RUNNING','CANCEL_REQUESTED','COMMITTING')
                       AND lease_expires_at>?)
                LIMIT 1
                """,
                (now,),
            ).fetchone()
        return row is not None

    def snapshot(self, job_id: str) -> dict[str, Any] | None:
        row = self.get(job_id)
        if row is None:
            return None
        return {
            "status": row["status"],
            "attempt": row["attempt"],
            "recovery_count": row["recovery_count"],
            "queued_at": row["queued_at"],
            "started_at": row["started_at"],
            "updated_at": row["updated_at"],
            "completed_at": row["completed_at"],
            "heartbeat_at": row["heartbeat_at"],
            "lease_expires_at": row["lease_expires_at"],
            "cancel_requested_at": row["cancel_requested_at"],
            "cancel_reason": row["cancel_reason"],
            "last_error_code": row["last_error_code"],
            "recovery_actor": row["recovery_actor"],
            "recovery_confirmed_at": row["recovery_confirmed_at"],
        }
