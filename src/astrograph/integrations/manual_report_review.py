"""Append-only expert review of material-analysis reports and generated claims.

Expert opinions are immutable overlays bound to an immutable report hash. They never
rewrite the generated report, establish an on-orbit root cause, or authorize control.
Review rows carry a per-report SHA-256 chain so accidental or direct database edits
are detectable. This is tamper-evident, not a substitute for external backups.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from astrograph.audit.store import sha256_json

DECISIONS = {"FOLLOW_UP_HYPOTHESIS", "NOT_SUPPORTED", "EVIDENCE_INSUFFICIENT"}
CLAIM_DECISIONS = {"ACCEPT", "REJECT", "REVISE", "NEEDS_EVIDENCE"}
CLAIM_EPISTEMIC_STATUSES = {"REPORTED", "DERIVED", "INFERRED"}
NO_CANDIDATE = "__REPORT_NO_CANDIDATE__"
MAX_FEEDBACK_EVENTS = 10000
_HASH = re.compile(r"^[0-9a-f]{64}$")


class ManualReviewConflict(ValueError):
    """Stale report version, invalid audit chain, or concurrent review write."""


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _event_hash(payload: dict[str, Any]) -> str:
    return sha256_json(payload)


def _clean_text(value: str, *, maximum: int) -> str:
    value = value.strip()
    if any(ord(char) < 32 and char not in "\t\n\r" for char in value):
        raise ValueError("Review text contains control characters")
    if len(value) > maximum:
        raise ValueError("Review text exceeds the allowed length")
    return value


def _candidate_event_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": row["job_id"], "report_sha256": row["report_sha256"],
        "fault_id": row["fault_id"], "decision": row["decision"],
        "evidence_ref": row["evidence_ref"], "note": row["note"],
        "actor": row["actor"], "recorded_at": row["recorded_at"],
        "previous_event_sha256": row.get("previous_event_sha256") or "",
    }


def _claim_event_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": row["job_id"], "report_sha256": row["report_sha256"],
        "claim_id": row["claim_id"], "claim_snapshot_sha256": row.get("claim_snapshot_sha256") or "",
        "snapshot_binding_status": row.get("snapshot_binding_status") or "LEGACY_UNBOUND",
        "decision": row["decision"], "evidence_ref": row["evidence_ref"],
        "note": row["note"], "revised_text": row["revised_text"],
        "revised_epistemic_status": row["revised_epistemic_status"],
        "actor": row["actor"], "recorded_at": row["recorded_at"],
        "previous_event_sha256": row.get("previous_event_sha256") or "",
    }


def _verify_chain(rows: list[dict[str, Any]], *, claim: bool) -> tuple[bool, str | None]:
    previous_by_stream: dict[tuple[str, str], str] = {}
    for row in rows:
        stream = (str(row["job_id"]), str(row["report_sha256"]))
        expected_previous = previous_by_stream.get(stream, "")
        actual_previous = str(row.get("previous_event_sha256") or "")
        actual_hash = str(row.get("event_sha256") or "")
        if actual_previous != expected_previous or not _HASH.fullmatch(actual_hash):
            return False, f"EVENT_CHAIN_LINK_INVALID:{row.get('event_id')}"
        payload = _claim_event_payload(row) if claim else _candidate_event_payload(row)
        if _event_hash(payload) != actual_hash:
            return False, f"EVENT_HASH_MISMATCH:{row.get('event_id')}"
        previous_by_stream[stream] = actual_hash
    return True, None


def report_candidates(report: dict[str, Any]) -> list[dict[str, str]]:
    if report.get("artifact_kind") != "RESULT_LOG_REVIEW":
        raise ValueError("Manual report review is available only for result/log/reference tasks")
    candidates = report.get("analysis", {}).get("result_log_review", {}).get("candidates") or []
    found: dict[str, str] = {}
    for candidate in candidates:
        fault_id = str(candidate.get("fault_id") or "").strip()
        if fault_id and fault_id not in found:
            found[fault_id] = str(candidate.get("fault_name") or fault_id)[:300]
    return ([{"fault_id": fault_id, "fault_name": name} for fault_id, name in found.items()]
            if found else [{"fault_id": NO_CANDIDATE, "fault_name": "未形成故障候选／整体证据复核"}])


def report_claims(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve only host-generated claim-governance records from the frozen report."""
    if report.get("artifact_kind") != "RESULT_LOG_REVIEW":
        raise ValueError("Claim review is available only for result/log/reference tasks")
    analysis = report.get("analysis") or {}
    knowledge = analysis.get("knowledge_enrichment") or {}
    model = knowledge.get("model_explanation") or {}
    governance = model.get("claim_governance") or {}
    raw = governance.get("claims") or []
    if not isinstance(raw, list):
        raise ValueError("Invalid claim governance registry")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Invalid claim governance item")
        claim_id = str(item.get("claim_id") or "").strip()
        text = str(item.get("text") or "").strip()
        if not claim_id or claim_id in seen or not text:
            raise ValueError("Invalid or duplicate governed claim")
        seen.add(claim_id)
        result.append({
            "claim_id": claim_id,
            "claim_type": str(item.get("claim_type") or "CLAIM")[:80],
            "text": text[:4000],
            "generator_model": str(item.get("generator_model") or governance.get("generator_model") or "")[:200],
            "verifier_model": str(item.get("verifier_model") or (governance.get("verification") or {}).get("verifier_model") or "")[:200],
            "epistemic_status": str(item.get("epistemic_status") or "INFERRED"),
            "epistemic_status_basis": str(item.get("epistemic_status_basis") or "LEGACY_DEFAULT"),
            "evidence_refs": list(item.get("evidence_refs") or []),
            "opposing_evidence_refs": list(item.get("opposing_evidence_refs") or []),
            "verification_status": str(item.get("verification_status") or "NOT_EVALUATED"),
            "verification_rationale": str(item.get("verification_rationale") or "")[:4000],
            "verification_evidence_refs": list(item.get("verification_evidence_refs") or []),
            "verification_counter_evidence_refs": list(item.get("verification_counter_evidence_refs") or []),
            "verification_information_gaps": list(item.get("verification_information_gaps") or []),
        })
    return result


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
                recorded_at TEXT NOT NULL,
                previous_event_sha256 TEXT NOT NULL DEFAULT '',
                event_sha256 TEXT NOT NULL DEFAULT ''
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS report_claim_review_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                report_sha256 TEXT NOT NULL,
                claim_id TEXT NOT NULL,
                decision TEXT NOT NULL,
                evidence_ref TEXT NOT NULL,
                note TEXT NOT NULL,
                revised_text TEXT NOT NULL,
                revised_epistemic_status TEXT NOT NULL,
                actor TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                claim_snapshot_sha256 TEXT NOT NULL DEFAULT '',
                snapshot_binding_status TEXT NOT NULL DEFAULT 'LEGACY_UNBOUND',
                previous_event_sha256 TEXT NOT NULL DEFAULT '',
                event_sha256 TEXT NOT NULL DEFAULT ''
            )""")
            self._ensure_columns(db)
            self._backfill_legacy_chains(db)
            db.execute("CREATE INDEX IF NOT EXISTS idx_report_review_identity ON report_review_events(job_id, report_sha256, event_id)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_report_claim_review_identity ON report_claim_review_events(job_id, report_sha256, event_id)")
            db.execute("""CREATE TRIGGER IF NOT EXISTS report_review_events_no_update
                BEFORE UPDATE ON report_review_events BEGIN
                SELECT RAISE(ABORT, 'REPORT_REVIEW_EVENTS_APPEND_ONLY');
            END""")
            db.execute("""CREATE TRIGGER IF NOT EXISTS report_review_events_no_delete
                BEFORE DELETE ON report_review_events BEGIN
                SELECT RAISE(ABORT, 'REPORT_REVIEW_EVENTS_APPEND_ONLY');
            END""")
            db.execute("""CREATE TRIGGER IF NOT EXISTS report_claim_review_events_no_update
                BEFORE UPDATE ON report_claim_review_events BEGIN
                SELECT RAISE(ABORT, 'REPORT_CLAIM_REVIEW_EVENTS_APPEND_ONLY');
            END""")
            db.execute("""CREATE TRIGGER IF NOT EXISTS report_claim_review_events_no_delete
                BEFORE DELETE ON report_claim_review_events BEGIN
                SELECT RAISE(ABORT, 'REPORT_CLAIM_REVIEW_EVENTS_APPEND_ONLY');
            END""")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=20)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA busy_timeout=20000")
            db.execute("PRAGMA foreign_keys=ON")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _ensure_columns(db: sqlite3.Connection) -> None:
        expected = {
            "report_review_events": {
                "previous_event_sha256": "TEXT NOT NULL DEFAULT ''",
                "event_sha256": "TEXT NOT NULL DEFAULT ''",
            },
            "report_claim_review_events": {
                "claim_snapshot_sha256": "TEXT NOT NULL DEFAULT ''",
                "snapshot_binding_status": "TEXT NOT NULL DEFAULT 'LEGACY_UNBOUND'",
                "previous_event_sha256": "TEXT NOT NULL DEFAULT ''",
                "event_sha256": "TEXT NOT NULL DEFAULT ''",
            },
        }
        for table, columns in expected.items():
            existing = {str(row["name"]) for row in db.execute(f"PRAGMA table_info({table})")}
            for name, declaration in columns.items():
                if name not in existing:
                    db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")

    @staticmethod
    def _backfill_legacy_chains(db: sqlite3.Connection) -> None:
        candidate_rows = [dict(row) for row in db.execute(
            "SELECT * FROM report_review_events ORDER BY job_id,report_sha256,event_id"
        ).fetchall()]
        previous: dict[tuple[str, str], str] = {}
        for row in candidate_rows:
            stream = (row["job_id"], row["report_sha256"])
            prior = previous.get(stream, "")
            if not row.get("event_sha256"):
                row["previous_event_sha256"] = prior
                row["event_sha256"] = _event_hash(_candidate_event_payload(row))
                db.execute(
                    "UPDATE report_review_events SET previous_event_sha256=?,event_sha256=? WHERE event_id=?",
                    (row["previous_event_sha256"], row["event_sha256"], row["event_id"]),
                )
            previous[stream] = row.get("event_sha256") or ""

        claim_rows = [dict(row) for row in db.execute(
            "SELECT * FROM report_claim_review_events ORDER BY job_id,report_sha256,event_id"
        ).fetchall()]
        previous = {}
        for row in claim_rows:
            stream = (row["job_id"], row["report_sha256"])
            prior = previous.get(stream, "")
            if not row.get("event_sha256"):
                row["previous_event_sha256"] = prior
                row["snapshot_binding_status"] = row.get("snapshot_binding_status") or "LEGACY_UNBOUND"
                row["claim_snapshot_sha256"] = row.get("claim_snapshot_sha256") or ""
                row["event_sha256"] = _event_hash(_claim_event_payload(row))
                db.execute(
                    "UPDATE report_claim_review_events SET previous_event_sha256=?,event_sha256=?,"
                    "claim_snapshot_sha256=?,snapshot_binding_status=? WHERE event_id=?",
                    (row["previous_event_sha256"], row["event_sha256"], row["claim_snapshot_sha256"],
                     row["snapshot_binding_status"], row["event_id"]),
                )
            previous[stream] = row.get("event_sha256") or ""

    def _load_events(self, db: sqlite3.Connection, job_id: str, digest: str) -> tuple[list[dict], list[dict]]:
        rows = [dict(row) for row in db.execute(
            "SELECT * FROM report_review_events WHERE job_id=? AND report_sha256=? ORDER BY event_id",
            (job_id, digest),
        ).fetchall()]
        claim_rows = [dict(row) for row in db.execute(
            "SELECT * FROM report_claim_review_events WHERE job_id=? AND report_sha256=? ORDER BY event_id",
            (job_id, digest),
        ).fetchall()]
        return rows, claim_rows

    def detail(self, job_id: str, report: dict[str, Any], *, include_history: bool = False) -> dict[str, Any]:
        candidates = report_candidates(report)
        claims = report_claims(report)
        digest = sha256_json(report)
        with self._connect() as db:
            rows, claim_rows = self._load_events(db, job_id, digest)

        candidate_integrity, candidate_error = _verify_chain(rows, claim=False)
        claim_integrity, claim_error = _verify_chain(claim_rows, claim=True)
        latest: dict[str, dict] = {}
        if candidate_integrity:
            for row in rows:
                latest[row["fault_id"]] = row
        items = [{**candidate, "review": latest.get(candidate["fault_id"])} for candidate in candidates]
        count = sum(item["review"] is not None for item in items)
        has_gaps = any(item["review"] and item["review"]["decision"] == "EVIDENCE_INSUFFICIENT" for item in items)
        status = ("AUDIT_INTEGRITY_FAILED" if not candidate_integrity else
                  "PENDING" if count == 0 else "IN_PROGRESS" if count < len(items) else
                  "REVIEWED_WITH_GAPS" if has_gaps else "REVIEWED")

        latest_claims: dict[str, dict] = {}
        if claim_integrity:
            for row in claim_rows:
                latest_claims[row["claim_id"]] = row
        reviewed_claim_count = sum(claim["claim_id"] in latest_claims for claim in claims)
        claim_gaps = any(
            latest_claims[claim["claim_id"]]["decision"] == "NEEDS_EVIDENCE"
            for claim in claims if claim["claim_id"] in latest_claims
        )
        claim_status = (
            "AUDIT_INTEGRITY_FAILED" if not claim_integrity else
            "NOT_APPLICABLE" if not claims else
            "PENDING" if reviewed_claim_count == 0 else
            "IN_PROGRESS" if reviewed_claim_count < len(claims) else
            "REVIEWED_WITH_GAPS" if claim_gaps else "REVIEWED"
        )
        payload = {
            "schema_version": "astrograph-manual-report-review.v2",
            "job_id": job_id, "report_sha256": digest,
            "status": status, "reviewed_count": count, "total_count": len(items),
            "revision": int(rows[-1]["event_id"]) if rows else 0,
            "review_history_integrity": "VERIFIED" if candidate_integrity else "FAILED",
            "review_history_integrity_error": candidate_error,
            "claim_review_status": claim_status,
            "claim_reviewed_count": reviewed_claim_count,
            "claim_total_count": len(claims),
            "claim_revision": int(claim_rows[-1]["event_id"]) if claim_rows else 0,
            "claim_history_integrity": "VERIFIED" if claim_integrity else "FAILED",
            "claim_history_integrity_error": claim_error,
            "current_root_cause_confirmed": False,
            "automatic_control_authorized": False,
        }
        if include_history:
            payload["candidates"] = items
            payload["history"] = [dict(row) for row in reversed(rows[-100:])]
            payload["claims"] = [{**claim, "review": latest_claims.get(claim["claim_id"])} for claim in claims]
            payload["claim_history"] = [dict(row) for row in reversed(claim_rows[-200:])]
        return payload

    def record(self, *, job_id: str, report: dict[str, Any], report_sha256: str,
               fault_id: str, decision: str, evidence_ref: str, note: str,
               actor: str, expected_revision: int) -> dict[str, Any]:
        known = {candidate["fault_id"] for candidate in report_candidates(report)}
        if fault_id not in known or decision not in DECISIONS:
            raise ValueError("Unknown report candidate or review decision")
        note = _clean_text(note, maximum=2000)
        evidence_ref = _clean_text(evidence_ref, maximum=500)
        if len(note) < 8:
            raise ValueError("Independent review notes must be 8–2000 characters")
        if decision != "EVIDENCE_INSUFFICIENT" and len(evidence_ref) < 8:
            raise ValueError("Accepting or rejecting a candidate needs independent evidence reference")
        digest = sha256_json(report)
        if report_sha256 != digest:
            raise ManualReviewConflict("Report changed; reload before recording review")
        actor = _clean_text(actor, maximum=160)
        if not actor:
            raise ValueError("Authenticated reviewer identity is required")
        when = datetime.now(timezone.utc).isoformat()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows, _ = self._load_events(db, job_id, digest)
            valid, _ = _verify_chain(rows, claim=False)
            if not valid:
                raise ManualReviewConflict("Review audit chain integrity failed; writes are blocked")
            revision = int(rows[-1]["event_id"]) if rows else 0
            if revision != expected_revision:
                raise ManualReviewConflict("Review version changed; reload before recording review")
            previous = rows[-1]["event_sha256"] if rows else ""
            payload = {
                "job_id": job_id, "report_sha256": digest, "fault_id": fault_id,
                "decision": decision, "evidence_ref": evidence_ref, "note": note,
                "actor": actor, "recorded_at": when, "previous_event_sha256": previous,
            }
            db.execute(
                "INSERT INTO report_review_events(job_id,report_sha256,fault_id,decision,evidence_ref,note,actor,"
                "recorded_at,previous_event_sha256,event_sha256) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (job_id, digest, fault_id, decision, evidence_ref, note, actor, when, previous, _event_hash(payload)),
            )
        return self.detail(job_id, report, include_history=True)

    def record_claim(
        self, *, job_id: str, report: dict[str, Any], report_sha256: str,
        claim_id: str, decision: str, evidence_ref: str, note: str, actor: str,
        expected_revision: int, revised_text: str = "",
        revised_epistemic_status: str = "INFERRED",
    ) -> dict[str, Any]:
        claims = {item["claim_id"]: item for item in report_claims(report)}
        claim = claims.get(claim_id)
        if claim is None or decision not in CLAIM_DECISIONS:
            raise ValueError("Unknown governed claim or review decision")
        note = _clean_text(note, maximum=2000)
        evidence_ref = _clean_text(evidence_ref, maximum=500)
        revised_text = _clean_text(revised_text, maximum=4000)
        actor = _clean_text(actor, maximum=160)
        if not actor:
            raise ValueError("Authenticated reviewer identity is required")
        if len(note) < 8:
            raise ValueError("Expert claim review notes must be 8–2000 characters")
        if decision != "NEEDS_EVIDENCE" and len(evidence_ref) < 8:
            raise ValueError("Accepting, rejecting or revising a claim needs an independent evidence reference")
        if decision == "REVISE":
            if len(revised_text) < 8:
                raise ValueError("Revised claim text must be 8–4000 characters")
            if revised_epistemic_status not in CLAIM_EPISTEMIC_STATUSES:
                raise ValueError("Invalid revised epistemic status")
        elif revised_text or revised_epistemic_status != "INFERRED":
            raise ValueError("Revised text and epistemic status are only valid for REVISE")
        digest = sha256_json(report)
        if report_sha256 != digest:
            raise ManualReviewConflict("Report changed; reload before recording claim review")
        when = datetime.now(timezone.utc).isoformat()
        snapshot_digest = sha256_json(claim)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            _, rows = self._load_events(db, job_id, digest)
            valid, _ = _verify_chain(rows, claim=True)
            if not valid:
                raise ManualReviewConflict("Claim review audit chain integrity failed; writes are blocked")
            revision = int(rows[-1]["event_id"]) if rows else 0
            if revision != expected_revision:
                raise ManualReviewConflict("Claim review version changed; reload before recording review")
            previous = rows[-1]["event_sha256"] if rows else ""
            payload = {
                "job_id": job_id, "report_sha256": digest, "claim_id": claim_id,
                "claim_snapshot_sha256": snapshot_digest, "snapshot_binding_status": "BOUND",
                "decision": decision, "evidence_ref": evidence_ref, "note": note,
                "revised_text": revised_text,
                "revised_epistemic_status": revised_epistemic_status if decision == "REVISE" else "",
                "actor": actor, "recorded_at": when, "previous_event_sha256": previous,
            }
            db.execute(
                "INSERT INTO report_claim_review_events(job_id,report_sha256,claim_id,decision,evidence_ref,note,"
                "revised_text,revised_epistemic_status,actor,recorded_at,claim_snapshot_sha256,"
                "snapshot_binding_status,previous_event_sha256,event_sha256) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (job_id, digest, claim_id, decision, evidence_ref, note, revised_text,
                 revised_epistemic_status if decision == "REVISE" else "", actor, when,
                 snapshot_digest, "BOUND", previous, _event_hash(payload)),
            )
        return self.detail(job_id, report, include_history=True)

    def export_claim_feedback(self, *, job_id: str, report: dict[str, Any]) -> dict[str, Any]:
        """Export a source-bound, privacy-minimized review dataset for offline evaluation."""
        claims = report_claims(report)
        digest = sha256_json(report)
        with self._connect() as db:
            _, rows = self._load_events(db, job_id, digest)
        valid, error = _verify_chain(rows, claim=True)
        if not valid:
            raise ManualReviewConflict("Claim review audit chain integrity failed; export is blocked")
        if len(rows) > MAX_FEEDBACK_EVENTS:
            raise ManualReviewConflict("Claim review export exceeds the bounded event limit")
        by_id = {item["claim_id"]: item for item in claims}
        latest_by_id = {row["claim_id"]: row for row in rows}
        model_feedback: dict[str, dict[str, int | str]] = {}
        for claim in claims:
            model_id = str(claim.get("generator_model") or "UNATTRIBUTED")
            bucket = model_feedback.setdefault(model_id, {
                "generator_model": model_id, "claim_count": 0, "reviewed_claim_count": 0,
                "pending_claim_count": 0, "accept_count": 0, "reject_count": 0,
                "revise_count": 0, "needs_evidence_count": 0,
            })
            bucket["claim_count"] += 1
            review = latest_by_id.get(claim["claim_id"])
            if review is None:
                bucket["pending_claim_count"] += 1
                continue
            bucket["reviewed_claim_count"] += 1
            decision_key = {
                "ACCEPT": "accept_count", "REJECT": "reject_count",
                "REVISE": "revise_count", "NEEDS_EVIDENCE": "needs_evidence_count",
            }.get(str(review.get("decision")))
            if decision_key:
                bucket[decision_key] += 1
        model_feedback_summary = [model_feedback[key] for key in sorted(model_feedback)]
        return {
            "schema_version": "astrograph-expert-claim-feedback.v1",
            "job_id": job_id, "report_sha256": digest,
            "claim_snapshot_count": len(claims),
            "claim_snapshots": list(by_id.values()),
            "event_count": len(rows),
            "history_integrity": "VERIFIED",
            "history": rows,
            "model_feedback_summary": model_feedback_summary,
            "model_feedback_interpretation": (
                "Counts are expert feedback outcomes, not model accuracy. "
                "They require sample-size, reviewer and case-mix context."
            ),
            "fact_truth_certified": False,
            "automatic_training_authorized": False,
            "note": "Expert feedback is a review overlay for evaluation. It is not a training label until separately curated and approved.",
        }
