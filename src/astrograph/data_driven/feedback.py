"""Controlled diagnosis feedback cycle (AI-Native roadmap Phase 2).

Implements the governed loop: evaluation -> selection -> upgrade proposal that
stays PENDING_HUMAN_GATE. Nothing is promoted into the production registry
without explicit human approval; the loop is audit-only until then.

The frozen selection gate mirrors the P1 mechanism pipeline gates: a candidate
is proposed only when end-to-end Macro-F1 improves by at least E2E_MIN_IMPROVEMENT
and no task regresses beyond TASK_MAX_REGRESSION.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from astrograph.evidence.executable import stable_id

E2E_MIN_IMPROVEMENT = 0.010
TASK_MAX_REGRESSION = 0.005


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class FeedbackCycleStatus(str, Enum):
    OPEN = "OPEN"
    EVALUATED = "EVALUATED"
    SELECTION_PROPOSED = "SELECTION_PROPOSED"
    PENDING_HUMAN_GATE = "PENDING_HUMAN_GATE"
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"
    STOPPED = "STOPPED"


class SelectionDecision(str, Enum):
    PROPOSE = "PROPOSE"
    STOP = "STOP"
    REJECT = "REJECT"


class TaskMetrics(BaseModel):
    """Per-task Macro-F1 values plus optional false-alarm rate."""

    model_config = ConfigDict(extra="forbid")

    detection_macro_f1: float | None = None
    localization_macro_f1: float | None = None
    classification_macro_f1: float | None = None
    end_to_end_macro_f1: float | None = None
    false_alarm_rate: float | None = None


class FeedbackCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    family: str = "UNSPECIFIED"
    metrics: TaskMetrics
    rationale: str = ""
    metadata: dict = Field(default_factory=dict)


class FeedbackCycleRecord(BaseModel):
    """One audit-able feedback cycle snapshot (append-only ledger line)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-feedback-cycle.v1"
    cycle_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    split_audit_ref: str = ""
    baseline: FeedbackCandidate
    candidates: list[FeedbackCandidate] = Field(default_factory=list)
    status: FeedbackCycleStatus = FeedbackCycleStatus.OPEN
    selection: dict | None = None
    upgrade_proposal: dict | None = None
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
    evidence_refs: list[str] = Field(default_factory=list)
    claim_boundary: str = Field(default="Controlled feedback-cycle record; upgrade proposals require explicit human approval before any production Registry change.")

    @classmethod
    def new_cycle(cls, *, cycle_id: str, dataset_id: str, baseline: FeedbackCandidate,
                  split_audit_ref: str = "") -> "FeedbackCycleRecord":
        return cls(
            cycle_id=cycle_id,
            dataset_id=dataset_id,
            split_audit_ref=split_audit_ref,
            baseline=baseline,
            candidates=[],
            status=FeedbackCycleStatus.OPEN,
        )


def evaluate_candidate(
    baseline: TaskMetrics, candidate: TaskMetrics,
    *,
    e2e_min_improvement: float = E2E_MIN_IMPROVEMENT,
    task_max_regression: float = TASK_MAX_REGRESSION,
) -> dict:
    """Apply the frozen selection gate to one candidate vs baseline.

    Returns a selection record with per-task deltas and a decision of
    PROPOSE / STOP / REJECT plus the audit trail.
    """
    fields = (
        ("detection_macro_f1", "detection"),
        ("localization_macro_f1", "localization"),
        ("classification_macro_f1", "classification"),
        ("end_to_end_macro_f1", "end_to_end"),
    )
    deltas: dict[str, float | None] = {}
    for attr, key in fields:
        base = getattr(baseline, attr)
        cand = getattr(candidate, attr)
        deltas[key] = (cand - base) if (base is not None and cand is not None) else None
    e2e = deltas["end_to_end"]
    reasons: list[str] = []
    if e2e is None:
        return {
            "decision": SelectionDecision.REJECT.value,
            "deltas": deltas,
            "reasons": ["end_to_end Macro-F1 missing for gate evaluation"],
        }
    if e2e < e2e_min_improvement:
        return {
            "decision": SelectionDecision.STOP.value,
            "deltas": deltas,
            "reasons": [f"end_to_end delta {e2e:+.4f} below +{e2e_min_improvement}"],
        }
    regressions = {
        key: value for key, value in deltas.items()
        if value is not None and value < -task_max_regression
    }
    if regressions:
        return {
            "decision": SelectionDecision.REJECT.value,
            "deltas": deltas,
            "reasons": [
                f"{key} regressed {value:+.4f} beyond -{task_max_regression}"
                for key, value in sorted(regressions.items())
            ],
        }
    return {
        "decision": SelectionDecision.PROPOSE.value,
        "deltas": deltas,
        "reasons": [f"end_to_end delta {e2e:+.4f} meets gate"],
    }


def build_upgrade_proposal(
    *,
    cycle_id: str,
    dataset_id: str,
    candidate: FeedbackCandidate,
    selection: dict,
) -> dict:
    """Build the frozen, human-gated upgrade proposal (never auto-promotes)."""
    return {
        "proposal_id": stable_id(
            "UPG",
            {"cycle_id": cycle_id, "dataset_id": dataset_id, "candidate_id": candidate.candidate_id},
        ),
        "cycle_id": cycle_id,
        "dataset_id": dataset_id,
        "candidate_id": candidate.candidate_id,
        "model_id": candidate.model_id,
        "model_version": candidate.model_version,
        "selection": selection,
        "requires_human_gate": True,
        "promoted": False,
        "gate": "candidate promotion requires explicit human approval (DEV_GOAL manual gate)",
    }


def advance_cycle(
    record: FeedbackCycleRecord,
    *,
    candidate: FeedbackCandidate | None = None,
    selection: dict | None = None,
    upgrade_proposal: dict | None = None,
    status: FeedbackCycleStatus | None = None,
    evidence_refs: list[str] | None = None,
) -> FeedbackCycleRecord:
    """Return an advanced snapshot of the cycle record (append-only semantics)."""
    updated = record.model_copy(update={
        "candidates": [*record.candidates, candidate] if candidate is not None else record.candidates,
        "selection": selection if selection is not None else record.selection,
        "upgrade_proposal": upgrade_proposal if upgrade_proposal is not None else record.upgrade_proposal,
        "status": status if status is not None else record.status,
        "evidence_refs": list(dict.fromkeys([*record.evidence_refs, *(evidence_refs or [])])),
        "updated_at": _now(),
    })
    return updated


class FeedbackCycleLedger:
    """Append-only JSONL ledger under governance/evidence/ai_native/feedback_cycles."""

    schema_version = "astrograph-feedback-cycle-ledger.v1"

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, cycle_id: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in cycle_id)
        return self.root / f"{safe}.jsonl"

    def append(self, record: FeedbackCycleRecord) -> Path:
        path = self._path(record.cycle_id)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")
        return path

    def read(self, cycle_id: str) -> list[FeedbackCycleRecord]:
        path = self._path(cycle_id)
        if not path.is_file():
            return []
        return [
            FeedbackCycleRecord.model_validate(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def latest(self, cycle_id: str) -> FeedbackCycleRecord | None:
        records = self.read(cycle_id)
        return records[-1] if records else None


__all__ = [
    "E2E_MIN_IMPROVEMENT",
    "FeedbackCandidate",
    "FeedbackCycleLedger",
    "FeedbackCycleRecord",
    "FeedbackCycleStatus",
    "SelectionDecision",
    "TASK_MAX_REGRESSION",
    "TaskMetrics",
    "advance_cycle",
    "build_upgrade_proposal",
    "evaluate_candidate",
]
