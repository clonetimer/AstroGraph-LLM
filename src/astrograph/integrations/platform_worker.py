"""Detached worker entry point for the durable platform job queue."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import time
import uuid

from astrograph.agents.mission.run_lock import task_lease
from .upload_report import UploadReportJobStore


def run_queue_worker(
    root: str | Path, slot: int, *,
    idle_timeout_s: float = 45.0, poll_interval_s: float = 0.25,
) -> int:
    """Drain durable work from one OS-leased worker slot."""
    platform_root = Path(root).expanduser().resolve()
    worker_lock = f"platform-queue-slot-{int(slot)}"
    try:
        lease = task_lease(platform_root, worker_lock)
        lease.__enter__()
    except RuntimeError as exc:
        if "TASK_ALREADY_RUNNING" in str(exc):
            return 0
        raise
    try:
        store = UploadReportJobStore(platform_root)
        queue = store.queue
        worker_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"
        try:
            max_workers = int(os.environ.get("ASTROGRAPH_PLATFORM_QUEUE_WORKERS", "2"))
        except ValueError:
            max_workers = 2
        max_workers = max(1, min(8, max_workers))
        try:
            lease_seconds = max(3.0, float(os.environ.get("ASTROGRAPH_PLATFORM_QUEUE_LEASE_SECONDS", "30")))
        except ValueError:
            lease_seconds = 30.0
        idle_since = time.monotonic()
        while True:
            expired = queue.recover_expired()
            for interrupted_id in expired:
                try:
                    store.queue_snapshot(interrupted_id)
                except (KeyError, OSError):
                    pass
            task = queue.claim_next(
                worker_id, lease_seconds=lease_seconds, max_running=max_workers,
            )
            if task is None:
                if time.monotonic() - idle_since >= idle_timeout_s:
                    return 0
                time.sleep(max(0.05, poll_interval_s))
                continue
            idle_since = time.monotonic()
            job_id = str(task["job_id"])
            try:
                store.run(job_id, worker_id=worker_id)
            except BaseException as exc:
                try:
                    job = store.get(job_id)
                    if job.get("status") not in {"SUCCEEDED", "FAILED", "CANCELLED"}:
                        job.update(
                            status="FAILED", stage="FAILED",
                            stage_detail="独立 Worker 异常退出",
                            completed_at=store._now() if hasattr(store, "_now") else None,
                            error={"type": type(exc).__name__, "message": str(exc)[:500]},
                        )
                        store._save(job)
                        store._append_event(job_id, f"[WORKER_FAILED] {type(exc).__name__}")
                    queue.finish(
                        job_id, worker_id, "FAILED",
                        error_code=type(exc).__name__[:120],
                    )
                except Exception:
                    # If the lease was lost, leave the durable row for expiry recovery.
                    pass
    finally:
        lease.__exit__(None, None, None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job-id")
    parser.add_argument("--queue-slot", type=int)
    parser.add_argument("--idle-timeout", type=float, default=45.0)
    args = parser.parse_args(argv)
    if args.queue_slot is not None:
        if args.job_id:
            parser.error("--job-id and --queue-slot are mutually exclusive")
        return run_queue_worker(args.root, args.queue_slot, idle_timeout_s=max(1.0, args.idle_timeout))
    if not args.job_id:
        parser.error("either --job-id or --queue-slot is required")
    result = UploadReportJobStore(args.root).run(args.job_id)
    return 0 if result["status"] == "SUCCEEDED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
