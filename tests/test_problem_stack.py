from __future__ import annotations

import json
from pathlib import Path

import pytest

from astrograph.governance.problem_stack import ProblemRecord, ProblemStack


def _record(problem_id: str, priority: str = "P1") -> ProblemRecord:
    return ProblemRecord(
        problem_id=problem_id,
        parent_task_id="T-010",
        resume_task_id="T-001",
        priority=priority,
        signature=f"signature-{problem_id}",
        raw_evidence=["tests/evidence.txt"],
        impact="Blocks trustworthy release evidence.",
    )


def test_problem_stack_enforces_priority_lifo_and_lifecycle(tmp_path: Path) -> None:
    stack = ProblemStack(tmp_path / "events.jsonl", tmp_path / "current.json")
    stack.open(_record("P-100", "P1"))
    view = stack.open(_record("P-101", "P1"))
    view = stack.open(_record("P-102", "P0"))
    assert [item.problem_id for item in view.active] == ["P-102", "P-101", "P-100"]
    assert view.blocker_count == 3

    stack.transition("P-102", "TRIAGE", {"root_cause": "A candidate-controlled path bypassed isolation."})
    stack.transition(
        "P-102",
        "RESOLVE",
        {"decision": "Move the path outside the candidate boundary.", "fix_artifacts": ["src/fix.py"]},
    )
    stack.transition("P-102", "VERIFY", {"regressions": ["tests/test_fix.py"]})
    view = stack.transition("P-102", "CLOSE")
    assert [item.problem_id for item in view.active] == ["P-101", "P-100"]
    assert view.closed[0].problem_id == "P-102"
    resumed = stack.transition("P-102", "RESUME")
    assert resumed.closed[0].resumed_at
    assert stack.ledger.audit().status == "PASS"
    persisted = json.loads((tmp_path / "current.json").read_text(encoding="utf-8"))
    assert persisted["ledger_record_count"] == 8


def test_problem_stack_rejects_invalid_close_and_duplicate(tmp_path: Path) -> None:
    stack = ProblemStack(tmp_path / "events.jsonl", tmp_path / "current.json")
    stack.open(_record("P-200"))
    with pytest.raises(ValueError, match="already exists"):
        stack.open(_record("P-200"))
    with pytest.raises(ValueError, match="invalid transition"):
        stack.transition("P-200", "CLOSE")
    with pytest.raises(ValueError, match="invalid transition"):
        stack.transition("P-200", "RETRIAGE")


def test_problem_stack_retriage_preserves_history_and_triaged_status(tmp_path: Path) -> None:
    stack = ProblemStack(tmp_path / "events.jsonl", tmp_path / "current.json")
    opened = stack.open(_record("P-300", "P0")).active[0]
    stack.transition("P-300", "TRIAGE", {"root_cause": "Initial evidence boundary."})
    view = stack.transition(
        "P-300",
        "RETRIAGE",
        {
            "signature": "The active data source changed while the problem remained open.",
            "root_cause": "New evidence narrowed the active source and changed the next work unit.",
            "raw_evidence": ["governance/evidence/new-inventory.json"],
            "decision": "Continue zero-fit adapter qualification before requesting model work.",
            "fix_artifacts": ["src/adapter.py"],
        },
    )
    current = view.active[0]
    assert current.status.value == "TRIAGED"
    assert current.signature.startswith("The active data source changed")
    assert current.opened_at == opened.opened_at
    assert view.ledger_record_count == 3
    assert stack.ledger.audit().status == "PASS"
