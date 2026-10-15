from __future__ import annotations

from astrograph.experiments.kg_coverage_protocol import (
    GroupOutcomeV2,
    NonInferiorityMarginsV2,
    select_development_policy_v2,
)

SEED = "e" * 64
MARGINS = NonInferiorityMarginsV2(
    max_accuracy_drop=0.02,
    max_coverage_drop=0.05,
    max_selective_risk_increase=0.02,
    max_nll_increase=0.02,
    max_brier_increase=0.02,
    max_ece_increase=0.02,
    max_aurc_increase=0.02,
    max_worst_group_accuracy_drop=0.02,
)


def row(
    candidate: str,
    group: str,
    *,
    accuracy: float,
    coverage: float = 1.0,
    risk: float = 0.1,
    nll: float = 0.2,
    brier: float = 0.1,
    ece: float = 0.05,
    aurc: float = 0.12,
) -> GroupOutcomeV2:
    return GroupOutcomeV2(
        candidate_id=candidate,
        group_id=group,
        observation_count=10,
        identification_accuracy=accuracy,
        accepted_coverage=coverage,
        selective_risk=risk,
        nll=nll,
        brier=brier,
        ece=ece,
        aurc=aurc,
    )


def test_paired_interval_not_point_accuracy_controls_feasibility() -> None:
    rows = [
        row("A0", "3", accuracy=0.90),
        row("A0", "8", accuracy=0.90),
        # Higher accuracy but clearly harmful selective risk in both groups.
        row("harmful", "3", accuracy=0.95, risk=0.30),
        row("harmful", "8", accuracy=0.95, risk=0.30),
        row("beneficial", "3", accuracy=0.93, risk=0.09),
        row("beneficial", "8", accuracy=0.93, risk=0.09),
    ]
    receipt = select_development_policy_v2(
        rows,
        baseline_candidate_id="A0",
        development_groups=("3", "8"),
        forbidden_groups=("4", "9"),
        margins=MARGINS,
        confidence_level=0.95,
        bootstrap_replicates=1000,
        bootstrap_seed_sha256=SEED,
    )
    assert receipt.selected_candidate_id == "beneficial"
    assert receipt.noninferiority_checks["harmful"]["selective_risk"] is False
    assert receipt.paired_inference["beneficial"]["identification_accuracy"].ci_low > 0
    assert receipt.receipt_sha256


def test_coverage_protection_rejects_abstention_collapse() -> None:
    rows = [
        row("A0", "3", accuracy=0.9),
        row("A0", "8", accuracy=0.9),
        row("empty", "3", accuracy=0.9, coverage=0.0, risk=1.0),
        row("empty", "8", accuracy=0.9, coverage=0.0, risk=1.0),
    ]
    receipt = select_development_policy_v2(
        rows,
        baseline_candidate_id="A0",
        development_groups=("3", "8"),
        forbidden_groups=("4", "9"),
        margins=MARGINS,
        confidence_level=0.95,
        bootstrap_replicates=1000,
        bootstrap_seed_sha256=SEED,
    )
    assert receipt.selected_candidate_id == "A0"
    assert receipt.noninferiority_checks["empty"]["accepted_coverage"] is False
