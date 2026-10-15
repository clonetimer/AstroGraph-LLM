from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Lock, Thread, get_ident
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

from astrograph.audit.store import sha256_json
from astrograph.errors import IdempotencyConflictError, IdempotencyLeaseLostError


class IdempotencyRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-idempotency-record.v1"
    idempotency_key: str = Field(min_length=1)
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    response: dict[str, Any]
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class JsonIdempotencyStore:
    """Single-process JSON idempotency store with per-key execution locking.

    Producers for different idempotency keys may run concurrently. State reads and
    atomic file replacement remain serialized so concurrent completions cannot lose
    each other's records. The class remains an engineering single-node store; a
    transactional database is still required for multi-host deployments.
    """

    schema_version = "astrograph-idempotency-store.v1"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._state_lock = Lock()
        self._key_locks_guard = Lock()
        self._key_locks: dict[str, Lock] = {}

    def _lock_for_key(self, key: str) -> Lock:
        with self._key_locks_guard:
            return self._key_locks.setdefault(key, Lock())

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": self.schema_version, "records": {}}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != self.schema_version:
            raise ValueError("unsupported idempotency store schema")
        records = payload.get("records")
        if not isinstance(records, dict):
            raise ValueError("idempotency store records must be an object")
        return payload

    def _atomic_write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.{get_ident()}.tmp"
        )
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            # Best-effort directory durability on POSIX. Windows does not allow
            # opening directories with os.open in the same way.
            if os.name == "posix":
                directory_fd = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _validate_existing(
        current: dict[str, Any] | None,
        *,
        key: str,
        request_hash: str,
    ) -> IdempotencyRecord | None:
        if current is None:
            return None
        record = IdempotencyRecord.model_validate(current)
        if record.request_hash != request_hash:
            raise IdempotencyConflictError(
                "The idempotency key was already used with a different request payload.",
                details={"idempotency_key": key},
            )
        return record

    def execute(
        self,
        key: str,
        request_payload: Any,
        producer: Callable[[], dict[str, Any]],
    ) -> tuple[dict[str, Any], bool]:
        request_hash = sha256_json(request_payload)
        key_lock = self._lock_for_key(key)

        # Same-key calls serialize; different keys do not hold the state lock while
        # expensive diagnosis/model/LLM work is running.
        with key_lock:
            with self._state_lock:
                payload = self._load()
                existing = self._validate_existing(
                    payload["records"].get(key),
                    key=key,
                    request_hash=request_hash,
                )
            if existing is not None:
                return existing.response, True

            response = producer()
            record = IdempotencyRecord(
                idempotency_key=key,
                request_hash=request_hash,
                response_hash=sha256_json(response),
                response=response,
            )

            with self._state_lock:
                # Reload after producer completion because other keys may have been
                # committed while this producer was running.
                payload = self._load()
                existing = self._validate_existing(
                    payload["records"].get(key),
                    key=key,
                    request_hash=request_hash,
                )
                if existing is not None:
                    return existing.response, True
                payload["records"][key] = record.model_dump(mode="json")
                self._atomic_write(payload)
            return response, False


class SqliteIdempotencyStore:
    """Transactional idempotency store for multi-process single-node services.

    Producers execute outside database transactions.  A RUNNING claim carries a
    unique owner token and a monotonically increasing lease epoch.  Heartbeats
    refresh the claim, while stale reclaim and final commit use owner/epoch/time
    compare-and-swap fencing so an old worker cannot delete or publish a newer
    worker's result.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        wait_timeout_s: float = 120.0,
        poll_interval_s: float = 0.05,
        stale_after_s: float = 600.0,
        heartbeat_interval_s: float | None = None,
    ) -> None:
        import sqlite3
        import time

        if wait_timeout_s <= 0.0:
            raise ValueError("wait_timeout_s must be positive")
        if poll_interval_s <= 0.0:
            raise ValueError("poll_interval_s must be positive")
        if stale_after_s <= 0.01:
            raise ValueError("stale_after_s must be greater than 0.01 seconds")
        if heartbeat_interval_s is None:
            heartbeat_interval_s = max(0.001, min(stale_after_s / 3.0, 5.0))
        if not 0.0 < float(heartbeat_interval_s) < stale_after_s:
            raise ValueError("heartbeat_interval_s must be positive and smaller than stale_after_s")

        self.path = Path(path)
        self.wait_timeout_s = float(wait_timeout_s)
        self.poll_interval_s = float(poll_interval_s)
        self.stale_after_s = float(stale_after_s)
        self.heartbeat_interval_s = float(heartbeat_interval_s)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        deadline = time.monotonic() + max(1.0, min(self.wait_timeout_s, 30.0))
        while True:
            try:
                with sqlite3.connect(self.path, timeout=30.0) as connection:
                    connection.execute("PRAGMA busy_timeout=30000")
                    connection.execute("PRAGMA journal_mode=WAL")
                    connection.execute("PRAGMA synchronous=FULL")
                    connection.execute(
                        """
                        CREATE TABLE IF NOT EXISTS idempotency_records (
                            idempotency_key TEXT PRIMARY KEY,
                            request_hash TEXT NOT NULL,
                            status TEXT NOT NULL CHECK(status IN ('RUNNING', 'COMPLETED')),
                            owner_token TEXT NOT NULL,
                            lease_epoch INTEGER NOT NULL DEFAULT 1,
                            response_hash TEXT,
                            response_json TEXT,
                            created_at REAL NOT NULL,
                            updated_at REAL NOT NULL
                        )
                        """
                    )
                    columns = {
                        str(row[1])
                        for row in connection.execute("PRAGMA table_info(idempotency_records)")
                    }
                    if "lease_epoch" not in columns:
                        connection.execute(
                            "ALTER TABLE idempotency_records "
                            "ADD COLUMN lease_epoch INTEGER NOT NULL DEFAULT 1"
                        )
                    connection.commit()
                break
            except sqlite3.OperationalError as exc:
                if not self._is_transient_lock_error(exc) or time.monotonic() >= deadline:
                    raise
                time.sleep(max(0.01, min(self.poll_interval_s, 0.1)))

    @staticmethod
    def _is_transient_lock_error(exc: BaseException) -> bool:
        message = str(exc).lower()
        return "locked" in message or "busy" in message

    def _renew_lease(
        self,
        *,
        key: str,
        request_hash: str,
        owner: str,
        lease_epoch: int,
        stop: Event,
        lease_lost: Event,
        errors: list[BaseException],
    ) -> None:
        import sqlite3
        import time

        # Renew immediately.  A transient SQLite lock is contention, not proof
        # that ownership was lost, so retry it at a short cadence.
        while not stop.is_set():
            try:
                with sqlite3.connect(self.path, timeout=30.0) as connection:
                    connection.execute("PRAGMA busy_timeout=30000")
                    updated = connection.execute(
                        """
                        UPDATE idempotency_records
                        SET updated_at = ?
                        WHERE idempotency_key = ?
                          AND request_hash = ?
                          AND status = 'RUNNING'
                          AND owner_token = ?
                          AND lease_epoch = ?
                        """,
                        (time.time(), key, request_hash, owner, lease_epoch),
                    ).rowcount
                    connection.commit()
                if updated != 1:
                    lease_lost.set()
                    return
            except sqlite3.OperationalError as exc:
                if self._is_transient_lock_error(exc):
                    stop.wait(min(self.poll_interval_s, self.heartbeat_interval_s / 4.0))
                    continue
                errors.append(exc)
                lease_lost.set()
                return
            except BaseException as exc:  # surfaced by the owner thread
                errors.append(exc)
                lease_lost.set()
                return
            if stop.wait(self.heartbeat_interval_s):
                return

    def _delete_owned_running_claim(
        self,
        *,
        key: str,
        request_hash: str,
        owner: str,
        lease_epoch: int,
    ) -> None:
        import sqlite3

        with sqlite3.connect(self.path, timeout=30.0) as connection:
            connection.execute("PRAGMA busy_timeout=30000")
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                DELETE FROM idempotency_records
                WHERE idempotency_key = ?
                  AND request_hash = ?
                  AND status = 'RUNNING'
                  AND owner_token = ?
                  AND lease_epoch = ?
                """,
                (key, request_hash, owner, lease_epoch),
            )
            connection.commit()

    def execute(
        self,
        key: str,
        request_payload: Any,
        producer: Callable[[], dict[str, Any]],
    ) -> tuple[dict[str, Any], bool]:
        import sqlite3
        import time
        import uuid

        request_hash = sha256_json(request_payload)
        owner = uuid.uuid4().hex
        deadline = time.monotonic() + self.wait_timeout_s
        lease_epoch: int | None = None

        while lease_epoch is None:
            now = time.time()
            with sqlite3.connect(self.path, timeout=30.0) as connection:
                connection.execute("PRAGMA busy_timeout=30000")
                row = connection.execute(
                    """
                    SELECT request_hash, status, owner_token, lease_epoch,
                           response_json, updated_at
                    FROM idempotency_records WHERE idempotency_key = ?
                    """,
                    (key,),
                ).fetchone()

                if row is None:
                    connection.execute("BEGIN IMMEDIATE")
                    inserted = connection.execute(
                        """
                        INSERT INTO idempotency_records
                        (idempotency_key, request_hash, status, owner_token,
                         lease_epoch, created_at, updated_at)
                        VALUES (?, ?, 'RUNNING', ?, 1, ?, ?)
                        ON CONFLICT(idempotency_key) DO NOTHING
                        """,
                        (key, request_hash, owner, now, now),
                    ).rowcount
                    connection.commit()
                    if inserted == 1:
                        lease_epoch = 1
                        break
                    continue

                (
                    stored_hash,
                    status,
                    stored_owner,
                    stored_epoch,
                    response_json,
                    updated_at,
                ) = row
                if stored_hash != request_hash:
                    raise IdempotencyConflictError(
                        "The idempotency key was already used with a different request payload.",
                        details={"idempotency_key": key},
                    )
                if status == "COMPLETED":
                    return json.loads(response_json or "{}"), True

                if now - float(updated_at) >= self.stale_after_s:
                    next_epoch = int(stored_epoch) + 1
                    connection.execute("BEGIN IMMEDIATE")
                    updated = connection.execute(
                        """
                        UPDATE idempotency_records
                        SET owner_token = ?, lease_epoch = ?, updated_at = ?
                        WHERE idempotency_key = ?
                          AND request_hash = ?
                          AND status = 'RUNNING'
                          AND owner_token = ?
                          AND lease_epoch = ?
                          AND updated_at = ?
                        """,
                        (
                            owner,
                            next_epoch,
                            now,
                            key,
                            request_hash,
                            stored_owner,
                            int(stored_epoch),
                            float(updated_at),
                        ),
                    ).rowcount
                    connection.commit()
                    if updated == 1:
                        lease_epoch = next_epoch
                        break

            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for idempotency key: {key}")
            time.sleep(self.poll_interval_s)

        if lease_epoch is None:  # defensive control-flow invariant
            raise RuntimeError("idempotency claim loop exited without a lease")

        stop_heartbeat = Event()
        lease_lost = Event()
        heartbeat_errors: list[BaseException] = []
        heartbeat = Thread(
            target=self._renew_lease,
            kwargs={
                "key": key,
                "request_hash": request_hash,
                "owner": owner,
                "lease_epoch": lease_epoch,
                "stop": stop_heartbeat,
                "lease_lost": lease_lost,
                "errors": heartbeat_errors,
            },
            name=f"astrograph-idempotency-heartbeat-{owner[:8]}",
            daemon=True,
        )
        heartbeat.start()

        try:
            response = producer()
            if heartbeat_errors or lease_lost.is_set():
                cause = heartbeat_errors[0] if heartbeat_errors else None
                error = IdempotencyLeaseLostError(
                    "The idempotency lease heartbeat was lost before response commit.",
                    details={"idempotency_key": key, "lease_epoch": lease_epoch},
                )
                if cause is not None:
                    raise error from cause
                raise error

            response_json = json.dumps(response, ensure_ascii=False, separators=(",", ":"))
            response_hash = sha256_json(response)
            with sqlite3.connect(self.path, timeout=30.0) as connection:
                connection.execute("PRAGMA busy_timeout=30000")
                connection.execute("BEGIN IMMEDIATE")
                updated = connection.execute(
                    """
                    UPDATE idempotency_records
                    SET status = 'COMPLETED', response_hash = ?,
                        response_json = ?, updated_at = ?
                    WHERE idempotency_key = ?
                      AND request_hash = ?
                      AND status = 'RUNNING'
                      AND owner_token = ?
                      AND lease_epoch = ?
                    """,
                    (
                        response_hash,
                        response_json,
                        time.time(),
                        key,
                        request_hash,
                        owner,
                        lease_epoch,
                    ),
                ).rowcount
                if updated != 1:
                    connection.rollback()
                    raise IdempotencyLeaseLostError(
                        "The idempotency lease was lost before response commit.",
                        details={"idempotency_key": key, "lease_epoch": lease_epoch},
                    )
                connection.commit()
            return response, False
        except Exception:
            try:
                self._delete_owned_running_claim(
                    key=key,
                    request_hash=request_hash,
                    owner=owner,
                    lease_epoch=lease_epoch,
                )
            except sqlite3.Error:
                # Preserve the producer/commit error.  A surviving claim can be
                # reclaimed safely through the stale lease path.
                pass
            raise
        finally:
            stop_heartbeat.set()
            heartbeat.join(timeout=max(1.0, self.heartbeat_interval_s * 3.0))
