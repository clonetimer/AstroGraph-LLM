from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.governance.ledger import AppendOnlyJsonlLedger
from astrograph.governance.models import utc_now


class ProblemPriority(str, Enum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class ProblemStatus(str, Enum):
    OPEN = "OPEN"
    TRIAGED = "TRIAGED"
    FIXED = "FIXED"
    VERIFIED = "VERIFIED"
    CLOSED = "CLOSED"


class ProblemRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-problem-record.v1"
    problem_id: str = Field(pattern=r"^P-[0-9]{3,}$")
    parent_task_id: str = Field(min_length=1)
    resume_task_id: str = Field(min_length=1)
    priority: ProblemPriority
    signature: str = Field(min_length=1)
    raw_evidence: list[str] = Field(min_length=1)
    impact: str = Field(min_length=1)
    root_cause: str = ""
    literature_queries: list[str] = Field(default_factory=list)
    candidate_solutions: list[str] = Field(default_factory=list)
    decision: str = ""
    fix_artifacts: list[str] = Field(default_factory=list)
    regressions: list[str] = Field(default_factory=list)
    status: ProblemStatus = ProblemStatus.OPEN
    opened_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    resumed_at: str = ""

    @model_validator(mode="after")
    def evidence_matches_state(self) -> "ProblemRecord":
        if self.status in {ProblemStatus.FIXED, ProblemStatus.VERIFIED, ProblemStatus.CLOSED}:
            if not self.root_cause or not self.decision or not self.fix_artifacts:
                raise ValueError("fixed problems require root cause, decision and fix artifacts")
        if self.status in {ProblemStatus.VERIFIED, ProblemStatus.CLOSED} and not self.regressions:
            raise ValueError("verified problems require regression evidence")
        return self


class ProblemStackEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-problem-stack-event.v1"
    action: str = Field(pattern=r"^(?:OPEN|TRIAGE|RETRIAGE|RESOLVE|VERIFY|CLOSE|RESUME)$")
    problem: ProblemRecord


class ProblemStackView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-problem-stack-view.v1"
    generated_at: str
    ledger_record_count: int = Field(ge=0)
    blocker_count: int = Field(ge=0)
    active: list[ProblemRecord] = Field(default_factory=list)
    closed: list[ProblemRecord] = Field(default_factory=list)


class ProblemStack:
    """Governed LIFO problem stack backed by the shared hash-chained ledger."""

    _TRANSITIONS = {
        "TRIAGE": (ProblemStatus.OPEN, ProblemStatus.TRIAGED),
        "RETRIAGE": (ProblemStatus.TRIAGED, ProblemStatus.TRIAGED),
        "RESOLVE": (ProblemStatus.TRIAGED, ProblemStatus.FIXED),
        "VERIFY": (ProblemStatus.FIXED, ProblemStatus.VERIFIED),
        "CLOSE": (ProblemStatus.VERIFIED, ProblemStatus.CLOSED),
        "RESUME": (ProblemStatus.CLOSED, ProblemStatus.CLOSED),
    }
    _PRIORITY_ORDER = {priority: index for index, priority in enumerate(ProblemPriority)}

    def __init__(self, events_path: str | Path, current_path: str | Path):
        self.ledger = AppendOnlyJsonlLedger(events_path)
        self.current_path = Path(current_path)

    def _latest(self) -> tuple[dict[str, ProblemRecord], dict[str, int]]:
        latest: dict[str, ProblemRecord] = {}
        sequences: dict[str, int] = {}
        if not self.ledger.path.exists():
            return latest, sequences
        audit = self.ledger.audit()
        if audit.status != "PASS":
            raise ValueError("problem stack ledger is invalid")
        for line in self.ledger.path.read_text(encoding="utf-8").splitlines():
            envelope = json.loads(line)
            if envelope["record_type"] != "problem_stack_event":
                raise ValueError("unexpected record type in problem stack ledger")
            event = ProblemStackEvent.model_validate(envelope["payload"])
            latest[event.problem.problem_id] = event.problem
            sequences[event.problem.problem_id] = int(envelope["sequence"])
        return latest, sequences

    def _write_view(self) -> ProblemStackView:
        latest, sequences = self._latest()
        active = [item for item in latest.values() if item.status != ProblemStatus.CLOSED]
        active.sort(key=lambda item: (self._PRIORITY_ORDER[item.priority], -sequences[item.problem_id]))
        closed = [item for item in latest.values() if item.status == ProblemStatus.CLOSED]
        closed.sort(key=lambda item: -sequences[item.problem_id])
        audit = self.ledger.audit()
        generated_at = ""
        if self.ledger.path.exists() and audit.record_count:
            generated_at = json.loads(self.ledger.path.read_text(encoding="utf-8").splitlines()[-1])["recorded_at"]
        view = ProblemStackView(
            generated_at=generated_at,
            ledger_record_count=audit.record_count,
            blocker_count=sum(item.priority in {ProblemPriority.P0, ProblemPriority.P1} for item in active),
            active=active,
            closed=closed,
        )
        self.current_path.parent.mkdir(parents=True, exist_ok=True)
        self.current_path.write_text(
            json.dumps(view.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return view

    def open(self, record: ProblemRecord) -> ProblemStackView:
        latest, _ = self._latest()
        if record.problem_id in latest:
            raise ValueError(f"problem already exists: {record.problem_id}")
        if record.status != ProblemStatus.OPEN:
            raise ValueError("new problems must start OPEN")
        self.ledger.append("problem_stack_event", ProblemStackEvent(action="OPEN", problem=record))
        return self._write_view()

    def transition(self, problem_id: str, action: str, patch: dict[str, Any] | None = None) -> ProblemStackView:
        action = action.upper()
        if action not in self._TRANSITIONS:
            raise ValueError(f"unsupported problem action: {action}")
        latest, _ = self._latest()
        if problem_id not in latest:
            raise ValueError(f"unknown problem: {problem_id}")
        expected, target = self._TRANSITIONS[action]
        current = latest[problem_id]
        if current.status != expected:
            raise ValueError(f"invalid transition: {current.status.value}->{action}")
        updates = dict(patch or {})
        protected = {"problem_id", "opened_at", "schema_version", "status"}
        if protected.intersection(updates):
            raise ValueError("transition patch contains protected fields")
        updates["status"] = target
        updates["updated_at"] = utc_now()
        if action == "RESUME":
            updates["resumed_at"] = updates["updated_at"]
        changed = ProblemRecord.model_validate({**current.model_dump(mode="json"), **updates})
        self.ledger.append("problem_stack_event", ProblemStackEvent(action=action, problem=changed))
        return self._write_view()

    def view(self) -> ProblemStackView:
        return self._write_view()


__all__ = [
    "ProblemPriority",
    "ProblemRecord",
    "ProblemStack",
    "ProblemStackEvent",
    "ProblemStackView",
    "ProblemStatus",
]
