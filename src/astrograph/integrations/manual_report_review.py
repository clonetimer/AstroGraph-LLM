"""Append-only expert review of *material-analysis reports*, separate from knowledge approval.

A review acknowledges/assesses each upstream fault candidate; it never edits the
immutable generated report, confirms an on-orbit root cause, or releases any action.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from astrograph.audit.store import sha256_json

DECISIONS = {
    "FOLLOW_UP_HYPOTHESIS", "NOT_SUPPORTED", "EVIDENCE_INSUFFICIENT",
}
NO_CANDIDATE = "__REPORT_NO_CANDIDATE__"


class ManualReviewConflict(ValueError):
    """Stale report version or concurrent review write."""


def report_candidates(report: dict[str, Any]) -> list[dict[str, str]]:
    if report.get("artifact_kind") != "RESULT_LOG_REVIEW":
        raise ValueError("Manual report review is available only for result/log/reference tasks")
    candidates = report.get("analysis", {}).get("result_log_review", {}).get("candidates") or []
    found: dict[str, str] = {}
    for c in candidates:
        fault_id = str(c.get("fault_id") or "").strip()
        if fault_id and fault_id not in found:
            found[fault_id] = str(c.get("fault_name") or fault_id)[:300]
    return ([{"fault_id": fid, "fault_name": name} for fid, name in found.items()]
            if found else [{"fault_id": NO_CANDIDATE, "fault_name": "未形成故障候选／整体证据复核"}])


class ManualReportReviewStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS report_review_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                report_sha256 TEXT NOT NULL,
                fault_id TEXT NOT NULL,
                decision TEXT NOT NULL,
                evidence_ref TEXT NOT NULL,
                note TEXT NOT NULL,
                actor TEXT NOT NULL,
                recorded_at TEXT NOT NULL
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS idx_report_review_identity ON report_review_events(job_id, report_sha256, event_id)")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=20)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=20000")
        return db

    def detail(self, job_id: str, report: dict[str, Any], *, include_history: bool = False) -> dict[str, Any]:
        candidates = report_candidates(report)
        digest = sha256_json(report)
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM report_review_events WHERE job_id=? AND report_sha256=? ORDER BY event_id",
                (job_id, digest),
            ).fetchall()
        latest: dict[str, dict] = {}
        for row in rows:
            record = dict(row)
            latest[record["fault_id"]] = record
        items = [{**c, "review": latest.get(c["fault_id"])} for c in candidates]
        count = sum(x["review"] is not None for x in items)
        has_gaps = any(x["review"] and x["review"]["decision"] == "EVIDENCE_INSUFFICIENT" for x in items)
        status = ("PENDING" if count == 0 else "IN_PROGRESS" if count < len(items)
                  else "REVIEWED_WITH_GAPS" if has_gaps else "REVIEWED")
        payload = {
            "schema_version": "astrograph-manual-report-review.v1",
            "job_id": job_id, "report_sha256": digest,
            "status": status, "reviewed_count": count, "total_count": len(items),
            "revision": int(rows[-1]["event_id"]) if rows else 0,
            "current_root_cause_confirmed": False,
            "automatic_control_authorized": False,
        }
        if include_history:
            payload["candidates"] = items
            payload["history"] = [dict(row) for row in reversed(rows[-100:])]
        return payload

    def record(self, *, job_id: str, report: dict[str, Any], report_sha256: str,
               fault_id: str, decision: str, evidence_ref: str, note: str,
               actor: str, expected_revision: int) -> dict[str, Any]:
        known = {c["fault_id"] for c in report_candidates(report)}
        if fault_id not in known or decision not in DECISIONS:
            raise ValueError("Unknown report candidate or review decision")
        note = note.strip()
        evidence_ref = evidence_ref.strip()
        if len(note) < 8 or len(note) > 2000:
            raise ValueError("Independent review notes must be 8–2000 characters")
        if len(evidence_ref) > 500:
            raise ValueError("Evidence reference is too long")
        if decision != "EVIDENCE_INSUFFICIENT" and len(evidence_ref) < 8:
            raise ValueError("Accepting or rejecting a candidate needs independent evidence reference")
        digest = sha256_json(report)
        if report_sha256 != digest:
            raise ManualReviewConflict("Report changed; reload before recording review")
        when = datetime.now(timezone.utc).isoformat()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT COALESCE(MAX(event_id), 0) AS version FROM report_review_events WHERE job_id=? AND report_sha256=?",
                (job_id, digest),
            ).fetchone()
            if int(row["version"]) != expected_revision:
                raise ManualReviewConflict("Review version changed; reload before recording review")
            db.execute(
                "INSERT INTO report_review_events(job_id,report_sha256,fault_id,decision,evidence_ref,note,actor,recorded_at) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (job_id, digest, fault_id, decision, evidence_ref, note, actor, when),
            )
        return self.detail(job_id, report, include_history=True)
