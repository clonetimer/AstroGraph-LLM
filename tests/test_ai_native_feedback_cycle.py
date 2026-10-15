"""AI-Native Phase 2: controlled diagnosis feedback cycle tests.
Verifies the selection gate (PROPOSE/STOP/REJECT), the human-gated upgrade
proposal, and the append-only ledger semantics.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from astrograph.data_driven.feedback import (
    FeedbackCandidate,
    FeedbackCycleLedger,
    FeedbackCycleRecord,
    FeedbackCycleStatus,
    SelectionDecision,
    TaskMetrics,
    advance_cycle,
    build_upgrade_proposal,
    evaluate_candidate,
)

ROOT = Path(__file__).resolve().parents[1]


def _metrics(
    *,
    detection=0.9,
    localization=0.9,
    classification=0.9,
    e2e=0.8,
) -> TaskMetrics:
    return TaskMetrics(
        detection_macro_f1=detection,
        localization_macro_f1=localization,
        classification_macro_f1=classification,
        end_to_end_macro_f1=e2e,
    )


def test_gate_proposes_when_e2e_improves_and_no_regression():
    baseline = _metrics()
    candidate = _metrics(detection=0.91, localization=0.91, classification=0.91, e2e=0.83)
    selection = evaluate_candidate(baseline, candidate)
    assert selection["decision"] == SelectionDecision.PROPOSE.value
    assert selection["deltas"]["end_to_end"] == pytest.approx(0.03)


def test_gate_stops_when_e2e_gain_below_minimum():
    baseline = _metrics()
    candidate = _metrics(e2e=0.805)
    selection = evaluate_candidate(baseline, candidate)
    assert selection["decision"] == SelectionDecision.STOP.value
    assert selection["deltas"]["end_to_end"] == pytest.approx(0.005)


def test_gate_rejects_when_task_regresses_beyond_tolerance():
    baseline = _metrics()
    candidate = _metrics(detection=0.885, e2e=0.83)
    selection = evaluate_candidate(baseline, candidate)
    assert selection["decision"] == SelectionDecision.REJECT.value
    assert any("detection regressed" in reason for reason in selection["reasons"])


def test_gate_rejects_missing_e2e():
    baseline = _metrics(e2e=None)
    candidate = _metrics(e2e=None)
    selection = evaluate_candidate(baseline, candidate)
    assert selection["decision"] == SelectionDecision.REJECT.value


def test_upgrade_proposal_never_auto_promotes():
    baseline = FeedbackCandidate(candidate_id="b", model_id="m", model_version="1", metrics=_metrics())
    candidate = FeedbackCandidate(candidate_id="c", model_id="m", model_version="2", metrics=_metrics(e2e=0.83))
    selection = evaluate_candidate(baseline.metrics, candidate.metrics)
    proposal = build_upgrade_proposal(
        cycle_id="cycle-1", dataset_id="ds", candidate=candidate, selection=selection,
    )
    assert proposal["requires_human_gate"] is True
    assert proposal["promoted"] is False
    assert proposal["candidate_id"] == "c"


def test_ledger_is_append_only_and_round_trips(tmp_path: Path):
    ledger = FeedbackCycleLedger(tmp_path / "cycles")
    record = FeedbackCycleRecord.new_cycle(
        cycle_id="cycle-x", dataset_id="ds",
        baseline=FeedbackCandidate(candidate_id="b", model_id="m", model_version="1", metrics=_metrics()),
    )
    path = ledger.append(record)
    assert path.is_file()
    advanced = advance_cycle(
        record,
        candidate=FeedbackCandidate(candidate_id="c", model_id="m", model_version="2", metrics=_metrics(e2e=0.83)),
        selection=evaluate_candidate(record.baseline.metrics, _metrics(e2e=0.83)),
        status=FeedbackCycleStatus.PENDING_HUMAN_GATE,
    )
    ledger.append(advanced)
    records = ledger.read("cycle-x")
    assert len(records) == 2
    assert records[0].status == FeedbackCycleStatus.OPEN
    assert records[-1].status == FeedbackCycleStatus.PENDING_HUMAN_GATE
    assert ledger.latest("cycle-x").candidates[0].candidate_id == "c"


def test_full_simulated_cycle_ends_pending_human_gate(tmp_path: Path):
    baseline = FeedbackCandidate(
        candidate_id="baseline", model_id="residual-tcn", model_version="p2",
        metrics=_metrics(detection=0.9293, localization=0.9687, classification=0.9399, e2e=0.8974),
    )
    candidate = FeedbackCandidate(
        candidate_id="candidate", model_id="residual-tcn", model_version="research",
        metrics=_metrics(detection=0.9350, localization=0.9700, classification=0.9450, e2e=0.9274),
    )
    ledger = FeedbackCycleLedger(tmp_path / "cycles")
    record = FeedbackCycleRecord.new_cycle(cycle_id="cycle-1", dataset_id="adcs", baseline=baseline)
    selection = evaluate_candidate(baseline.metrics, candidate.metrics)
    assert selection["decision"] == SelectionDecision.PROPOSE.value
    proposal = build_upgrade_proposal(
        cycle_id="cycle-1", dataset_id="adcs", candidate=candidate, selection=selection,
    )
    advanced = advance_cycle(
        record, candidate=candidate, selection=selection, upgrade_proposal=proposal,
        status=FeedbackCycleStatus.PENDING_HUMAN_GATE,
    )
    ledger.append(advanced)
    latest = ledger.latest("cycle-1")
    assert latest is not None
    assert latest.status == FeedbackCycleStatus.PENDING_HUMAN_GATE
    assert latest.upgrade_proposal["promoted"] is False
    assert latest.upgrade_proposal["requires_human_gate"] is True


def test_committed_cycle_zero_baseline_is_valid():
    evidence_root = ROOT / "governance/evidence/ai_native/feedback_cycles"
    if not (evidence_root / "cycle-0-baseline.jsonl").is_file():
        pytest.skip("AI-native feedback-cycle evidence is not included in the source-only package")
    ledger = FeedbackCycleLedger(evidence_root)
    records = ledger.read("cycle-0-baseline")
    assert len(records) == 1
    record = records[0]
    assert record.status == FeedbackCycleStatus.EVALUATED
    assert record.baseline.metrics.end_to_end_macro_f1 == pytest.approx(0.8974)
    assert record.candidates == []
