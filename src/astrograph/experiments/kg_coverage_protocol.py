"""Group-level development selection for evidence-aware fusion policies.

Version 1 functions are retained for backward compatibility.  The v2 protocol
adds paired group-bootstrap intervals, accepted-coverage protection, explicit
small-group warnings, and a content-addressed selection receipt.  It is a
*development-selection* procedure only; it cannot convert groups 4/9 or any
previously inspected dataset into confirmatory evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import random
import re
from statistics import fmean, mean, median, pstdev
from typing import Mapping, Sequence

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# Compatibility v1 API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupOutcome:
    policy_id: str
    group_id: int
    identification_accuracy: float
    selective_risk: float
    nll: float
    brier: float
    ece: float
    aurc: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name in {"policy_id", "group_id"}:
                continue
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class NonInferiorityMargins:
    selective_risk: float
    nll: float
    brier: float
    ece: float
    worst_group_accuracy: float

    def __post_init__(self) -> None:
        if any(value < 0 or not math.isfinite(value) for value in asdict(self).values()):
            raise ValueError("margins must be finite and non-negative")


@dataclass(frozen=True)
class PolicySummary:
    policy_id: str
    group_count: int
    mean_identification_accuracy: float
    worst_group_accuracy: float
    mean_selective_risk: float
    mean_nll: float
    mean_brier: float
    mean_ece: float
    mean_aurc: float


def summarize(records: Sequence[GroupOutcome]) -> PolicySummary:
    if not records:
        raise ValueError("records required")
    policy_ids = {record.policy_id for record in records}
    groups = [record.group_id for record in records]
    if len(policy_ids) != 1:
        raise ValueError("summary requires one policy")
    if len(groups) != len(set(groups)):
        raise ValueError("each independent group may appear only once per policy")
    return PolicySummary(
        policy_id=records[0].policy_id,
        group_count=len(records),
        mean_identification_accuracy=mean(
            record.identification_accuracy for record in records
        ),
        worst_group_accuracy=min(record.identification_accuracy for record in records),
        mean_selective_risk=mean(record.selective_risk for record in records),
        mean_nll=mean(record.nll for record in records),
        mean_brier=mean(record.brier for record in records),
        mean_ece=mean(record.ece for record in records),
        mean_aurc=mean(record.aurc for record in records),
    )


def select_development_policy(
    records: Sequence[GroupOutcome],
    *,
    baseline_policy_id: str,
    development_groups: Sequence[int],
    forbidden_groups: Sequence[int],
    margins: NonInferiorityMargins,
) -> tuple[str, dict[str, PolicySummary]]:
    """Point-estimate v1 selector retained for existing receipts/tests."""

    development = set(development_groups)
    forbidden = set(forbidden_groups)
    if not development or development & forbidden:
        raise ValueError(
            "development groups must be non-empty and disjoint from forbidden groups"
        )
    observed = {record.group_id for record in records}
    if observed != development:
        raise ValueError(
            "records must cover exactly development groups; "
            f"observed={sorted(observed)} expected={sorted(development)}"
        )
    by_policy: dict[str, list[GroupOutcome]] = {}
    for record in records:
        by_policy.setdefault(record.policy_id, []).append(record)
    if baseline_policy_id not in by_policy:
        raise ValueError("baseline policy missing")
    for policy_id, rows in by_policy.items():
        policy_groups = {record.group_id for record in rows}
        if policy_groups != development or len(rows) != len(development):
            raise ValueError(
                f"policy {policy_id} must contain exactly one record for each development group"
            )
    summaries = {policy_id: summarize(rows) for policy_id, rows in by_policy.items()}
    baseline = summaries[baseline_policy_id]
    feasible: list[PolicySummary] = []
    for policy_id, summary in summaries.items():
        if policy_id == baseline_policy_id:
            feasible.append(summary)
            continue
        if summary.mean_selective_risk > baseline.mean_selective_risk + margins.selective_risk:
            continue
        if summary.mean_nll > baseline.mean_nll + margins.nll:
            continue
        if summary.mean_brier > baseline.mean_brier + margins.brier:
            continue
        if summary.mean_ece > baseline.mean_ece + margins.ece:
            continue
        if (
            summary.worst_group_accuracy
            < baseline.worst_group_accuracy - margins.worst_group_accuracy
        ):
            continue
        feasible.append(summary)
    feasible.sort(
        key=lambda summary: (
            -summary.mean_identification_accuracy,
            summary.mean_selective_risk,
            summary.mean_nll,
            summary.mean_ece,
            0 if summary.policy_id == baseline_policy_id else 1,
            summary.policy_id,
        )
    )
    return feasible[0].policy_id, summaries


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def cluster_bootstrap_difference(
    candidate: Sequence[GroupOutcome],
    baseline: Sequence[GroupOutcome],
    *,
    metric: str,
    replicates: int,
    seed_sha256: str,
) -> dict[str, float]:
    """Compatibility paired group bootstrap for one endpoint."""

    if replicates < 100:
        raise ValueError("at least 100 bootstrap replicates required")
    candidate_by_group = {record.group_id: record for record in candidate}
    baseline_by_group = {record.group_id: record for record in baseline}
    if candidate_by_group.keys() != baseline_by_group.keys() or not candidate_by_group:
        raise ValueError("paired candidate/baseline independent groups required")
    groups = sorted(candidate_by_group)
    if metric not in {
        "identification_accuracy",
        "selective_risk",
        "nll",
        "brier",
        "ece",
        "aurc",
    }:
        raise ValueError("unsupported metric")
    if not _SHA256.fullmatch(seed_sha256):
        raise ValueError("seed_sha256 must be lowercase SHA-256")
    differences = {
        group_id: getattr(candidate_by_group[group_id], metric)
        - getattr(baseline_by_group[group_id], metric)
        for group_id in groups
    }
    rng = random.Random(int(seed_sha256, 16))
    bootstrap = []
    for _ in range(replicates):
        sampled = [rng.choice(groups) for _ in groups]
        bootstrap.append(mean(differences[group_id] for group_id in sampled))
    return {
        "mean_difference": mean(differences.values()),
        "ci95_low": _percentile(bootstrap, 0.025),
        "ci95_high": _percentile(bootstrap, 0.975),
        "replicates": float(replicates),
    }


# ---------------------------------------------------------------------------
# Uncertainty-aware v2 API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupOutcomeV2:
    """One policy variant evaluated once on one independent development group."""

    candidate_id: str
    group_id: str
    observation_count: int
    identification_accuracy: float
    accepted_coverage: float
    selective_risk: float
    nll: float
    brier: float
    ece: float
    aurc: float

    def __post_init__(self) -> None:
        if not self.candidate_id.strip() or not self.group_id.strip():
            raise ValueError("candidate_id and group_id are required")
        if self.observation_count < 1:
            raise ValueError("observation_count must be positive")
        bounded = {
            "identification_accuracy": self.identification_accuracy,
            "accepted_coverage": self.accepted_coverage,
            "selective_risk": self.selective_risk,
            "ece": self.ece,
            "aurc": self.aurc,
        }
        for name, value in bounded.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        for name in ("nll", "brier"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class NonInferiorityMarginsV2:
    max_accuracy_drop: float
    max_coverage_drop: float
    max_selective_risk_increase: float
    max_nll_increase: float
    max_brier_increase: float
    max_ece_increase: float
    max_aurc_increase: float
    max_worst_group_accuracy_drop: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in (
            "max_accuracy_drop",
            "max_coverage_drop",
            "max_selective_risk_increase",
            "max_ece_increase",
            "max_aurc_increase",
            "max_worst_group_accuracy_drop",
        ):
            if getattr(self, name) > 1.0:
                raise ValueError(f"{name} must be <= 1")


@dataclass(frozen=True)
class PairedBootstrapInference:
    metric: str
    direction: str
    group_count: int
    observed_mean_delta: float
    observed_median_delta: float
    paired_standardized_effect: float | None
    confidence_level: float
    ci_low: float
    ci_high: float
    bootstrap_replicates: int


@dataclass(frozen=True)
class PolicySummaryV2:
    candidate_id: str
    group_count: int
    observation_count: int
    group_macro_identification_accuracy: float
    worst_group_accuracy: float
    group_macro_accepted_coverage: float
    worst_group_accepted_coverage: float
    group_macro_selective_risk: float
    group_macro_nll: float
    group_macro_brier: float
    group_macro_ece: float
    group_macro_aurc: float


@dataclass(frozen=True)
class DevelopmentSelectionReceiptV2:
    schema_version: str
    baseline_candidate_id: str
    development_groups: tuple[str, ...]
    forbidden_groups: tuple[str, ...]
    confidence_level: float
    bootstrap_replicates: int
    bootstrap_seed_sha256: str
    margins: NonInferiorityMarginsV2
    summaries: Mapping[str, PolicySummaryV2]
    paired_inference: Mapping[str, Mapping[str, PairedBootstrapInference]]
    noninferiority_checks: Mapping[str, Mapping[str, bool]]
    feasible_candidate_ids: tuple[str, ...]
    selected_candidate_id: str
    small_group_warning: bool
    claim_boundary: str
    receipt_sha256: str


def summarize_v2(records: Sequence[GroupOutcomeV2]) -> PolicySummaryV2:
    if not records:
        raise ValueError("records required")
    candidate_ids = {record.candidate_id for record in records}
    groups = [record.group_id for record in records]
    if len(candidate_ids) != 1:
        raise ValueError("summary requires one candidate")
    if len(groups) != len(set(groups)):
        raise ValueError("each independent group may appear only once per candidate")
    return PolicySummaryV2(
        candidate_id=records[0].candidate_id,
        group_count=len(records),
        observation_count=sum(record.observation_count for record in records),
        group_macro_identification_accuracy=fmean(
            record.identification_accuracy for record in records
        ),
        worst_group_accuracy=min(record.identification_accuracy for record in records),
        group_macro_accepted_coverage=fmean(record.accepted_coverage for record in records),
        worst_group_accepted_coverage=min(record.accepted_coverage for record in records),
        group_macro_selective_risk=fmean(record.selective_risk for record in records),
        group_macro_nll=fmean(record.nll for record in records),
        group_macro_brier=fmean(record.brier for record in records),
        group_macro_ece=fmean(record.ece for record in records),
        group_macro_aurc=fmean(record.aurc for record in records),
    )


def _paired_bootstrap_inference(
    candidate: Mapping[str, GroupOutcomeV2],
    baseline: Mapping[str, GroupOutcomeV2],
    *,
    metric: str,
    direction: str,
    confidence_level: float,
    replicates: int,
    seed_sha256: str,
    seed_namespace: str,
) -> PairedBootstrapInference:
    if candidate.keys() != baseline.keys() or not candidate:
        raise ValueError("paired candidate/baseline independent groups required")
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0.5, 1)")
    if replicates < 1000:
        raise ValueError("v2 selection requires at least 1000 bootstrap replicates")
    if not _SHA256.fullmatch(seed_sha256):
        raise ValueError("seed_sha256 must be lowercase SHA-256")
    groups = sorted(candidate)
    differences = [
        float(getattr(candidate[group_id], metric) - getattr(baseline[group_id], metric))
        for group_id in groups
    ]
    namespace_seed = hashlib.sha256(
        f"{seed_sha256}:{seed_namespace}:{metric}".encode()
    ).hexdigest()
    rng = random.Random(int(namespace_seed, 16))
    bootstrap = []
    for _ in range(replicates):
        sampled = [rng.choice(differences) for _ in differences]
        bootstrap.append(fmean(sampled))
    alpha = 1.0 - confidence_level
    standard_deviation = pstdev(differences) if len(differences) > 1 else 0.0
    standardized = (
        None if standard_deviation == 0.0 else fmean(differences) / standard_deviation
    )
    return PairedBootstrapInference(
        metric=metric,
        direction=direction,
        group_count=len(groups),
        observed_mean_delta=fmean(differences),
        observed_median_delta=median(differences),
        paired_standardized_effect=standardized,
        confidence_level=confidence_level,
        ci_low=_percentile(bootstrap, alpha / 2.0),
        ci_high=_percentile(bootstrap, 1.0 - alpha / 2.0),
        bootstrap_replicates=replicates,
    )


def _receipt_payload(receipt: DevelopmentSelectionReceiptV2) -> dict[str, object]:
    return {
        "schema_version": receipt.schema_version,
        "baseline_candidate_id": receipt.baseline_candidate_id,
        "development_groups": list(receipt.development_groups),
        "forbidden_groups": list(receipt.forbidden_groups),
        "confidence_level": receipt.confidence_level,
        "bootstrap_replicates": receipt.bootstrap_replicates,
        "bootstrap_seed_sha256": receipt.bootstrap_seed_sha256,
        "margins": asdict(receipt.margins),
        "summaries": {
            candidate_id: asdict(summary)
            for candidate_id, summary in sorted(receipt.summaries.items())
        },
        "paired_inference": {
            candidate_id: {
                metric: asdict(inference)
                for metric, inference in sorted(metric_map.items())
            }
            for candidate_id, metric_map in sorted(receipt.paired_inference.items())
        },
        "noninferiority_checks": {
            candidate_id: dict(sorted(checks.items()))
            for candidate_id, checks in sorted(receipt.noninferiority_checks.items())
        },
        "feasible_candidate_ids": list(receipt.feasible_candidate_ids),
        "selected_candidate_id": receipt.selected_candidate_id,
        "small_group_warning": receipt.small_group_warning,
        "claim_boundary": receipt.claim_boundary,
    }


def select_development_policy_v2(
    records: Sequence[GroupOutcomeV2],
    *,
    baseline_candidate_id: str,
    development_groups: Sequence[str],
    forbidden_groups: Sequence[str],
    margins: NonInferiorityMarginsV2,
    confidence_level: float,
    bootstrap_replicates: int,
    bootstrap_seed_sha256: str,
    small_group_warning_threshold: int = 5,
) -> DevelopmentSelectionReceiptV2:
    """Select a Route A candidate using paired group-level non-inferiority.

    The confidence intervals are used only to choose a *development* policy.
    With very few independent groups (notably the historical 3/8 split), the
    receipt explicitly warns that the interval is low-resolution and cannot be
    interpreted as confirmatory inference.
    """

    development = tuple(sorted(set(str(group) for group in development_groups)))
    forbidden = tuple(sorted(set(str(group) for group in forbidden_groups)))
    if not development:
        raise ValueError("development_groups cannot be empty")
    if set(development) & set(forbidden):
        raise ValueError("development groups overlap forbidden post-hoc groups")
    if small_group_warning_threshold < 2:
        raise ValueError("small_group_warning_threshold must be >= 2")
    by_candidate: dict[str, dict[str, GroupOutcomeV2]] = {}
    for record in records:
        candidate_rows = by_candidate.setdefault(record.candidate_id, {})
        if record.group_id in candidate_rows:
            raise ValueError(
                f"candidate {record.candidate_id} repeats independent group {record.group_id}"
            )
        candidate_rows[record.group_id] = record
    if baseline_candidate_id not in by_candidate:
        raise ValueError("baseline candidate missing")
    expected_groups = set(development)
    for candidate_id, rows in by_candidate.items():
        observed_groups = set(rows)
        if observed_groups != expected_groups:
            raise ValueError(
                f"candidate {candidate_id} must contain exactly development groups; "
                f"observed={sorted(observed_groups)} expected={list(development)}"
            )
        if observed_groups & set(forbidden):
            raise ValueError("forbidden post-hoc group present in selection rows")

    summaries = {
        candidate_id: summarize_v2([rows[group_id] for group_id in development])
        for candidate_id, rows in sorted(by_candidate.items())
    }
    baseline_rows = by_candidate[baseline_candidate_id]
    paired: dict[str, dict[str, PairedBootstrapInference]] = {}
    checks: dict[str, dict[str, bool]] = {}
    feasible: list[str] = []
    metric_specs = {
        "identification_accuracy": ("HIGHER_IS_BETTER", margins.max_accuracy_drop),
        "accepted_coverage": ("HIGHER_IS_BETTER", margins.max_coverage_drop),
        "selective_risk": ("LOWER_IS_BETTER", margins.max_selective_risk_increase),
        "nll": ("LOWER_IS_BETTER", margins.max_nll_increase),
        "brier": ("LOWER_IS_BETTER", margins.max_brier_increase),
        "ece": ("LOWER_IS_BETTER", margins.max_ece_increase),
        "aurc": ("LOWER_IS_BETTER", margins.max_aurc_increase),
    }
    for candidate_id, candidate_rows in sorted(by_candidate.items()):
        if candidate_id == baseline_candidate_id:
            checks[candidate_id] = {
                **{metric: True for metric in metric_specs},
                "worst_group_accuracy": True,
            }
            paired[candidate_id] = {}
            feasible.append(candidate_id)
            continue
        candidate_inference: dict[str, PairedBootstrapInference] = {}
        candidate_checks: dict[str, bool] = {}
        for metric, (direction, margin) in metric_specs.items():
            inference = _paired_bootstrap_inference(
                candidate_rows,
                baseline_rows,
                metric=metric,
                direction=direction,
                confidence_level=confidence_level,
                replicates=bootstrap_replicates,
                seed_sha256=bootstrap_seed_sha256,
                seed_namespace=candidate_id,
            )
            candidate_inference[metric] = inference
            if direction == "HIGHER_IS_BETTER":
                candidate_checks[metric] = inference.ci_low >= -margin
            else:
                candidate_checks[metric] = inference.ci_high <= margin
        baseline_worst = summaries[baseline_candidate_id].worst_group_accuracy
        candidate_worst = summaries[candidate_id].worst_group_accuracy
        candidate_checks["worst_group_accuracy"] = (
            candidate_worst
            >= baseline_worst - margins.max_worst_group_accuracy_drop
        )
        paired[candidate_id] = candidate_inference
        checks[candidate_id] = candidate_checks
        if all(candidate_checks.values()):
            feasible.append(candidate_id)

    # Benefit seeking only after all non-inferiority constraints pass.  Coverage
    # is ranked before risk to prevent a near-empty abstaining policy from
    # winning through superficially low selective risk.
    feasible.sort(
        key=lambda candidate_id: (
            -summaries[candidate_id].group_macro_identification_accuracy,
            -summaries[candidate_id].group_macro_accepted_coverage,
            summaries[candidate_id].group_macro_selective_risk,
            summaries[candidate_id].group_macro_nll,
            summaries[candidate_id].group_macro_brier,
            summaries[candidate_id].group_macro_ece,
            summaries[candidate_id].group_macro_aurc,
            0 if candidate_id == baseline_candidate_id else 1,
            candidate_id,
        )
    )
    selected = feasible[0]
    provisional = DevelopmentSelectionReceiptV2(
        schema_version="astrograph-route-a-development-selection-receipt.v2",
        baseline_candidate_id=baseline_candidate_id,
        development_groups=development,
        forbidden_groups=forbidden,
        confidence_level=confidence_level,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed_sha256=bootstrap_seed_sha256,
        margins=margins,
        summaries=summaries,
        paired_inference=paired,
        noninferiority_checks=checks,
        feasible_candidate_ids=tuple(feasible),
        selected_candidate_id=selected,
        small_group_warning=len(development) < small_group_warning_threshold,
        claim_boundary=(
            "Development-only paired-group selection. Bootstrap resamples independent groups, "
            "not windows or telemetry rows. Small development-group counts limit interval "
            "resolution. The selected candidate requires a separate untouched sealed campaign "
            "before any confirmatory claim. Groups 4/9 remain post-hoc only."
        ),
        receipt_sha256="",
    )
    digest = hashlib.sha256(
        json.dumps(
            _receipt_payload(provisional), sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    return replace(provisional, receipt_sha256=digest)


__all__ = [
    "DevelopmentSelectionReceiptV2",
    "GroupOutcome",
    "GroupOutcomeV2",
    "NonInferiorityMargins",
    "NonInferiorityMarginsV2",
    "PairedBootstrapInference",
    "PolicySummary",
    "PolicySummaryV2",
    "cluster_bootstrap_difference",
    "select_development_policy",
    "select_development_policy_v2",
    "summarize",
    "summarize_v2",
]
