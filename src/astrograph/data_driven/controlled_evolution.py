"""Stage-89 coordinator for automatic offline evaluation with a frozen production gate."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from astrograph.evidence.executable import stable_id

from .failure_regression import FailureRegressionSet
from .feedback import (
    FeedbackCandidate, FeedbackCycleLedger, FeedbackCycleRecord, FeedbackCycleStatus,
    TaskMetrics, advance_cycle, build_upgrade_proposal, evaluate_candidate,
)


def run_controlled_evolution_cycle(
    *,
    root: str | Path,
    dataset_id: str,
    baseline: FeedbackCandidate,
    challenger: FeedbackCandidate,
    failure_regression_root: str | Path,
    predictor: Callable[[Any], str],
    split_audit_ref: str = "",
    evidence_refs: list[str] | None = None,
) -> dict[str, Any]:
    """Run all offline gates automatically; never mutate the production registry."""
    cycle_id=stable_id("CYCLE",{"dataset_id":dataset_id,"baseline":baseline.candidate_id,"challenger":challenger.candidate_id})
    ledger=FeedbackCycleLedger(Path(root))
    record=FeedbackCycleRecord.new_cycle(cycle_id=cycle_id,dataset_id=dataset_id,baseline=baseline,split_audit_ref=split_audit_ref)
    ledger.append(record)
    regression=FailureRegressionSet(Path(failure_regression_root)).replay(predictor)
    selection=evaluate_candidate(baseline.metrics,challenger.metrics)
    selection={**selection,"failure_regression":regression}
    # A challenger cannot be proposed if it leaves any recorded expert-corrected case unfixed.
    if regression["case_count"] and regression["fixed_count"] < regression["case_count"]:
        selection={**selection,"decision":"REJECT","reasons":[*selection.get("reasons",[]),"recorded failure regression cases remain unfixed"]}
    record=advance_cycle(record,candidate=challenger,selection=selection,status=FeedbackCycleStatus.EVALUATED,
                         evidence_refs=evidence_refs or [])
    ledger.append(record)
    proposal=None
    if selection["decision"]=="PROPOSE":
        proposal=build_upgrade_proposal(cycle_id=cycle_id,dataset_id=dataset_id,candidate=challenger,selection=selection)
        record=advance_cycle(record,upgrade_proposal=proposal,status=FeedbackCycleStatus.PENDING_HUMAN_GATE)
    else:
        record=advance_cycle(record,status=FeedbackCycleStatus.REJECTED if selection["decision"]=="REJECT" else FeedbackCycleStatus.STOPPED)
    ledger.append(record)
    return {
        "schema_version":"astrograph-controlled-evolution-run.v1",
        "cycle_id":cycle_id,"status":record.status.value,"selection":selection,"upgrade_proposal":proposal,
        "failure_regression":regression,"offline_evaluation_completed":True,
        "production_modified":False,"production_promotion_allowed_without_gate":False,
        "next_step":"EXISTING_PRODUCTION_GATE" if proposal else "NONE",
        "claim_boundary":(
            "Feedback-to-challenger evaluation and regression are automatic. Production promotion remains behind the "
            "existing explicit gate; this coordinator never self-rewrites code or changes production registries."
        ),
    }


__all__=["run_controlled_evolution_cycle"]
