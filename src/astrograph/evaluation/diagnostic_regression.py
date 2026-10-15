"""R36 executable diagnostic regression contracts.

This module computes three task families from frozen case artifacts:
Detection, component Localization, and fault-mode Identification.  Metric
computation is deliberately separate from evidence classification.  A
DEVELOPMENT or fixture result can be complete, but it is never promotion
eligible.

Reconstruction note: the historical R36 implementation was not present in the
source-only baseline.  The contracts below implement the handoff requirements
without reconstructing missing predictions or qualification evidence.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from math import isfinite
from statistics import mean, median
from typing import Callable, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvidenceClass(str, Enum):
    QUALIFICATION = "QUALIFICATION"
    DEVELOPMENT = "DEVELOPMENT"
    FIXTURE_ONLY_NOT_QUALIFICATION = "FIXTURE_ONLY_NOT_QUALIFICATION"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class QualificationAuthorizationReceipt(_Strict):
    """Externally issued authorization bound to one exact benchmark artifact."""

    schema_version: Literal["astrograph-qualification-authorization.v1"] = (
        "astrograph-qualification-authorization.v1"
    )
    receipt_id: str = Field(min_length=1)
    issuer: str = Field(min_length=1)
    issued_at: datetime
    dataset_id: str = Field(min_length=1)
    protocol_id: str = Field(min_length=1)
    cohort_id: str = Field(min_length=1)
    source_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    signature: str = Field(min_length=1)

    @model_validator(mode="after")
    def timezone_aware(self) -> "QualificationAuthorizationReceipt":
        if self.issued_at.tzinfo is None or self.issued_at.utcoffset() is None:
            raise ValueError("qualification receipt issued_at must be timezone-aware")
        return self


class BenchmarkCase(_Strict):
    case_id: str = Field(min_length=1)
    true_detected: bool
    predicted_detected: bool
    duration_s: float = Field(gt=0.0)

    true_onset_s: float | None = Field(default=None, ge=0.0)
    first_alarm_time_s: float | None = Field(default=None, ge=0.0)
    detection_time_s: float | None = Field(default=None, ge=0.0)

    true_component_id: str | None = None
    predicted_component_id: str | None = None

    true_fault_mode_id: str | None = None
    predicted_fault_mode_id: str | None = None

    true_open_set: bool = False
    abstained: bool = False

    @model_validator(mode="after")
    def validate_case_semantics(self) -> "BenchmarkCase":
        if self.true_onset_s is not None and self.true_onset_s > self.duration_s:
            raise ValueError("true_onset_s cannot exceed duration_s")
        if self.detection_time_s is not None and self.detection_time_s > self.duration_s:
            raise ValueError("detection_time_s cannot exceed duration_s")
        if self.first_alarm_time_s is not None and self.first_alarm_time_s > self.duration_s:
            raise ValueError("first_alarm_time_s cannot exceed duration_s")
        if self.detection_time_s is not None and not self.predicted_detected:
            raise ValueError("detection_time_s requires predicted_detected=True")
        if (
            self.true_detected
            and self.predicted_detected
            and self.true_onset_s is not None
            and self.detection_time_s is not None
            and self.detection_time_s < self.true_onset_s
        ):
            raise ValueError("fault detection_time_s cannot precede true_onset_s")
        if not self.true_detected:
            if self.true_onset_s is not None:
                raise ValueError("normal case cannot carry true_onset_s")
            if self.true_component_id is not None or self.true_fault_mode_id is not None:
                raise ValueError("normal case cannot carry fault truth")
            if self.true_open_set:
                raise ValueError("normal case cannot be true_open_set")
        return self


class DetectionMetrics(_Strict):
    TP: int = Field(ge=0)
    FP: int = Field(ge=0)
    FN: int = Field(ge=0)
    TN: int = Field(ge=0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    F1: float = Field(ge=0.0, le=1.0)
    false_alarm_rate: float = Field(ge=0.0, le=1.0)
    false_alarms_per_hour: float = Field(ge=0.0)
    fault_pre_onset_alarm_count: int = Field(ge=0)
    fault_pre_onset_alarm_rate: float = Field(ge=0.0, le=1.0)
    latency_count: int = Field(ge=0)
    latency_mean_s: float | None = Field(default=None, ge=0.0)
    latency_median_s: float | None = Field(default=None, ge=0.0)
    latency_p95_s: float | None = Field(default=None, ge=0.0)
    latency_definition: Literal[
        "FIRST_POST_ONSET_ALARM_EPISODE_MINUS_TRUTH_ONSET"
    ] = "FIRST_POST_ONSET_ALARM_EPISODE_MINUS_TRUTH_ONSET"


class LocalizationMetrics(_Strict):
    evaluated_count: int = Field(ge=0)
    accuracy: float = Field(ge=0.0, le=1.0)
    macro_f1: float = Field(ge=0.0, le=1.0)
    abstention_rate: float = Field(ge=0.0, le=1.0)


class IdentificationMetrics(_Strict):
    evaluated_count: int = Field(ge=0)
    accuracy: float = Field(ge=0.0, le=1.0)
    macro_f1: float = Field(ge=0.0, le=1.0)
    abstention_rate: float = Field(ge=0.0, le=1.0)
    open_set_count: int = Field(ge=0)
    open_set_recall: float = Field(ge=0.0, le=1.0)


class DiagnosticBenchmarkMetrics(_Strict):
    detection: DetectionMetrics
    localization: LocalizationMetrics
    identification: IdentificationMetrics


class BenchmarkSnapshot(_Strict):
    schema_version: Literal["astrograph-r36-diagnostic-benchmark-snapshot.v1"] = (
        "astrograph-r36-diagnostic-benchmark-snapshot.v1"
    )
    astrograph_version: str = Field(min_length=1)
    benchmark_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    protocol_id: str = Field(min_length=1)
    cohort_id: str = Field(min_length=1)
    source_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_class: EvidenceClass
    qualification_receipt_id: str | None = None
    case_count: int = Field(gt=0)
    fault_case_count: int = Field(ge=0)
    normal_case_count: int = Field(ge=0)
    metrics: DiagnosticBenchmarkMetrics
    status: Literal["COMPLETE"] = "COMPLETE"
    promotion_eligible: bool
    claim_boundary: str = (
        "Executable three-task benchmark result bound to a frozen case artifact. "
        "Metric completeness does not change DEVELOPMENT or fixture evidence into qualification."
    )

    @model_validator(mode="after")
    def consistent_counts_and_class(self) -> "BenchmarkSnapshot":
        if self.fault_case_count + self.normal_case_count != self.case_count:
            raise ValueError("fault_case_count + normal_case_count must equal case_count")
        expected = self.evidence_class == EvidenceClass.QUALIFICATION
        if self.promotion_eligible != expected:
            raise ValueError("promotion_eligible must be true only for QUALIFICATION evidence")
        if expected != (self.qualification_receipt_id is not None):
            raise ValueError("QUALIFICATION requires a verified qualification receipt id")
        return self


class RegressionBoardRow(_Strict):
    benchmark_id: str
    status: Literal["COMPLETE", "MISSING"]
    evidence_class: EvidenceClass | None = None
    promotion_eligible: bool = False
    snapshot: BenchmarkSnapshot | None = None

    @model_validator(mode="after")
    def missing_has_no_snapshot(self) -> "RegressionBoardRow":
        if self.status == "MISSING":
            if self.snapshot is not None or self.evidence_class is not None or self.promotion_eligible:
                raise ValueError("MISSING board row cannot contain benchmark evidence")
        else:
            if self.snapshot is None or self.evidence_class is None:
                raise ValueError("COMPLETE board row requires a snapshot")
        return self


class DiagnosticRegressionBoard(_Strict):
    schema_version: Literal["astrograph-r36-diagnostic-regression-board.v1"] = (
        "astrograph-r36-diagnostic-regression-board.v1"
    )
    astrograph_version: str
    rows: tuple[RegressionBoardRow, ...]
    promotion_eligible: bool
    claim_boundary: str = (
        "Missing evidence stays MISSING. Historical prose, fixtures, and DEVELOPMENT "
        "results are not reconstructed or promoted to qualification."
    )


class SnapshotComparison(_Strict):
    schema_version: Literal["astrograph-r36-diagnostic-snapshot-comparison.v1"] = (
        "astrograph-r36-diagnostic-snapshot-comparison.v1"
    )
    benchmark_id: str
    baseline_version: str
    candidate_version: str
    metric_deltas: dict[str, float]
    comparable: bool
    noncomparability_reasons: tuple[str, ...] = ()


def _ratio(num: int | float, den: int | float) -> float:
    return float(num / den) if den else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    frac = position - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _macro_f1(truth: list[str], predicted: list[str | None]) -> float:
    if not truth:
        return 0.0
    labels = sorted(set(truth))
    values: list[float] = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(truth, predicted))
        fp = sum(t != label and p == label for t, p in zip(truth, predicted))
        fn = sum(t == label and p != label for t, p in zip(truth, predicted))
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, tp + fn)
        values.append(_f1(precision, recall))
    return float(mean(values)) if values else 0.0


def evaluate_cases(cases: Iterable[BenchmarkCase]) -> DiagnosticBenchmarkMetrics:
    values = list(cases)
    if not values:
        raise ValueError("benchmark requires at least one case")
    if len({item.case_id for item in values}) != len(values):
        raise ValueError("benchmark case_id values must be unique")

    tp = sum(item.true_detected and item.predicted_detected for item in values)
    fp = sum((not item.true_detected) and item.predicted_detected for item in values)
    fn = sum(item.true_detected and (not item.predicted_detected) for item in values)
    tn = sum((not item.true_detected) and (not item.predicted_detected) for item in values)
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    latencies = [
        float(item.detection_time_s - item.true_onset_s)
        for item in values
        if item.true_detected
        and item.predicted_detected
        and item.true_onset_s is not None
        and item.detection_time_s is not None
    ]
    normal_hours = sum(item.duration_s for item in values if not item.true_detected) / 3600.0
    fault_count = sum(item.true_detected for item in values)
    pre_onset_alarm_count = sum(
        item.true_detected
        and item.true_onset_s is not None
        and item.first_alarm_time_s is not None
        and item.first_alarm_time_s < item.true_onset_s
        for item in values
    )
    detection = DetectionMetrics(
        TP=tp,
        FP=fp,
        FN=fn,
        TN=tn,
        precision=precision,
        recall=recall,
        F1=_f1(precision, recall),
        false_alarm_rate=_ratio(fp, fp + tn),
        false_alarms_per_hour=float(fp / normal_hours) if normal_hours else 0.0,
        fault_pre_onset_alarm_count=pre_onset_alarm_count,
        fault_pre_onset_alarm_rate=_ratio(pre_onset_alarm_count, fault_count),
        latency_count=len(latencies),
        latency_mean_s=float(mean(latencies)) if latencies else None,
        latency_median_s=float(median(latencies)) if latencies else None,
        latency_p95_s=_percentile(latencies, 0.95),
    )

    loc_cases = [item for item in values if item.true_detected and item.true_component_id is not None]
    loc_truth = [str(item.true_component_id) for item in loc_cases]
    loc_pred: list[str | None] = [
        None
        if item.abstained or not item.predicted_detected or item.predicted_component_id is None
        else item.predicted_component_id
        for item in loc_cases
    ]
    loc_n = len(loc_cases)
    localization = LocalizationMetrics(
        evaluated_count=loc_n,
        accuracy=_ratio(sum(t == p for t, p in zip(loc_truth, loc_pred)), loc_n),
        macro_f1=_macro_f1(loc_truth, loc_pred),
        abstention_rate=_ratio(sum(p is None for p in loc_pred), loc_n),
    )

    id_cases = [item for item in values if item.true_detected and item.true_fault_mode_id is not None]
    id_truth = [str(item.true_fault_mode_id) for item in id_cases]
    id_pred: list[str | None] = [
        None
        if item.abstained or not item.predicted_detected or item.predicted_fault_mode_id is None
        else item.predicted_fault_mode_id
        for item in id_cases
    ]
    id_n = len(id_cases)
    open_cases = [item for item in id_cases if item.true_open_set]
    identification = IdentificationMetrics(
        evaluated_count=id_n,
        accuracy=_ratio(sum(t == p for t, p in zip(id_truth, id_pred)), id_n),
        macro_f1=_macro_f1(id_truth, id_pred),
        abstention_rate=_ratio(sum(p is None for p in id_pred), id_n),
        open_set_count=len(open_cases),
        # For the generic case artifact, explicit abstention is the only
        # unambiguous open-set success signal.
        open_set_recall=_ratio(sum(item.abstained for item in open_cases), len(open_cases)),
    )
    return DiagnosticBenchmarkMetrics(
        detection=detection,
        localization=localization,
        identification=identification,
    )


def evaluate_diagnostic_benchmark(
    *,
    astrograph_version: str,
    benchmark_id: str,
    dataset_id: str,
    protocol_id: str,
    cohort_id: str,
    source_artifact_sha256: str,
    evidence_class: EvidenceClass,
    cases: Iterable[BenchmarkCase],
    qualification_receipt: QualificationAuthorizationReceipt | None = None,
    qualification_receipt_validator: (
        Callable[[QualificationAuthorizationReceipt], bool] | None
    ) = None,
) -> BenchmarkSnapshot:
    if evidence_class == EvidenceClass.QUALIFICATION:
        if qualification_receipt is None or qualification_receipt_validator is None:
            raise ValueError("QUALIFICATION requires an externally verified authorization receipt")
        expected_identity = (
            dataset_id,
            protocol_id,
            cohort_id,
            source_artifact_sha256,
        )
        receipt_identity = (
            qualification_receipt.dataset_id,
            qualification_receipt.protocol_id,
            qualification_receipt.cohort_id,
            qualification_receipt.source_artifact_sha256,
        )
        if receipt_identity != expected_identity:
            raise ValueError("qualification receipt does not bind the benchmark identity")
        if not qualification_receipt_validator(qualification_receipt):
            raise ValueError("qualification receipt verification failed")
    elif qualification_receipt is not None:
        raise ValueError("qualification receipt is only valid for QUALIFICATION evidence")

    case_list = list(cases)
    metrics = evaluate_cases(case_list)
    fault_count = sum(item.true_detected for item in case_list)
    return BenchmarkSnapshot(
        astrograph_version=astrograph_version,
        benchmark_id=benchmark_id,
        dataset_id=dataset_id,
        protocol_id=protocol_id,
        cohort_id=cohort_id,
        source_artifact_sha256=source_artifact_sha256,
        evidence_class=evidence_class,
        qualification_receipt_id=(
            qualification_receipt.receipt_id
            if qualification_receipt is not None
            else None
        ),
        case_count=len(case_list),
        fault_case_count=fault_count,
        normal_case_count=len(case_list) - fault_count,
        metrics=metrics,
        promotion_eligible=evidence_class == EvidenceClass.QUALIFICATION,
    )


def build_regression_board(
    *,
    astrograph_version: str,
    benchmark_ids: Iterable[str],
    snapshots: Iterable[BenchmarkSnapshot],
) -> DiagnosticRegressionBoard:
    ids = list(benchmark_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("benchmark_ids must be unique")
    by_id: dict[str, BenchmarkSnapshot] = {}
    for snapshot in snapshots:
        if snapshot.benchmark_id in by_id:
            raise ValueError(f"duplicate snapshot for benchmark_id={snapshot.benchmark_id}")
        by_id[snapshot.benchmark_id] = snapshot
    unknown = sorted(set(by_id) - set(ids))
    if unknown:
        raise ValueError(f"snapshot benchmark ids not present in release configuration: {unknown}")
    rows: list[RegressionBoardRow] = []
    for benchmark_id in ids:
        snapshot = by_id.get(benchmark_id)
        if snapshot is None:
            rows.append(RegressionBoardRow(benchmark_id=benchmark_id, status="MISSING"))
        else:
            rows.append(
                RegressionBoardRow(
                    benchmark_id=benchmark_id,
                    status="COMPLETE",
                    evidence_class=snapshot.evidence_class,
                    promotion_eligible=snapshot.promotion_eligible,
                    snapshot=snapshot,
                )
            )
    return DiagnosticRegressionBoard(
        astrograph_version=astrograph_version,
        rows=tuple(rows),
        promotion_eligible=bool(rows) and all(row.promotion_eligible for row in rows),
    )


def compare_snapshots(
    baseline: BenchmarkSnapshot,
    candidate: BenchmarkSnapshot,
) -> SnapshotComparison:
    reasons: list[str] = []
    for field in ("benchmark_id", "dataset_id", "protocol_id", "cohort_id"):
        if getattr(baseline, field) != getattr(candidate, field):
            reasons.append(f"{field.upper()}_MISMATCH")
    comparable = not reasons
    deltas: dict[str, float] = {}
    if comparable:
        deltas = {
            "detection.F1": candidate.metrics.detection.F1 - baseline.metrics.detection.F1,
            "localization.accuracy": candidate.metrics.localization.accuracy - baseline.metrics.localization.accuracy,
            "localization.macro_f1": candidate.metrics.localization.macro_f1 - baseline.metrics.localization.macro_f1,
            "identification.accuracy": candidate.metrics.identification.accuracy - baseline.metrics.identification.accuracy,
            "identification.macro_f1": candidate.metrics.identification.macro_f1 - baseline.metrics.identification.macro_f1,
        }
        if any(not isfinite(value) for value in deltas.values()):
            raise ValueError("comparison metric delta must be finite")
    return SnapshotComparison(
        benchmark_id=candidate.benchmark_id,
        baseline_version=baseline.astrograph_version,
        candidate_version=candidate.astrograph_version,
        metric_deltas=deltas,
        comparable=comparable,
        noncomparability_reasons=tuple(reasons),
    )


__all__ = [
    "EvidenceClass",
    "QualificationAuthorizationReceipt",
    "BenchmarkCase",
    "DetectionMetrics",
    "LocalizationMetrics",
    "IdentificationMetrics",
    "DiagnosticBenchmarkMetrics",
    "BenchmarkSnapshot",
    "RegressionBoardRow",
    "DiagnosticRegressionBoard",
    "SnapshotComparison",
    "evaluate_cases",
    "evaluate_diagnostic_benchmark",
    "build_regression_board",
    "compare_snapshots",
]
