from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .credibility import (
    CalibrationMetrics,
    CredibilityStatus,
    IndependentRunAudit,
    IndependentRunRecord,
    SplitName,
    calibration_metrics,
)
from .models import DataDrivenPrediction


class BenchmarkStatus(str, Enum):
    PASS = "PASS"
    BLOCKED = "BLOCKED"


class BenchmarkReleaseStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    BLOCKED = "BLOCKED"


class BenchmarkClaimType(str, Enum):
    SEED_REGRESSION_RESULT = "SEED_REGRESSION_RESULT"
    CROSS_CONDITION_GENERALIZATION = "CROSS_CONDITION_GENERALIZATION"
    CROSS_SATELLITE_GENERALIZATION = "CROSS_SATELLITE_GENERALIZATION"
    FLIGHT_VALIDATION = "FLIGHT_VALIDATION"


class BenchmarkPredictionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    truth_class_id: str = Field(min_length=1)
    prediction: DataDrivenPrediction
    open_set_rejected: bool = False
    latency_ms: float | None = Field(default=None, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PerClassMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_id: str
    support: int = Field(ge=0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)


class ModelBenchmarkMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_count: int = Field(ge=0)
    known_sample_count: int = Field(ge=0)
    unknown_sample_count: int = Field(ge=0)
    accuracy: float = Field(ge=0.0, le=1.0)
    balanced_accuracy: float = Field(ge=0.0, le=1.0)
    macro_f1: float = Field(ge=0.0, le=1.0)
    top_k_recall: dict[str, float]
    calibration: CalibrationMetrics | None = None
    unknown_false_accept_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_latency_ms: float | None = Field(default=None, ge=0.0)
    p95_latency_ms: float | None = Field(default=None, ge=0.0)
    per_class: list[PerClassMetric] = Field(default_factory=list)


class ModelBenchmarkReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-model-benchmark-report.v1"
    status: BenchmarkStatus
    model_id: str
    model_version: str
    evaluated_split: SplitName
    run_ids: list[str]
    run_audit_status: CredibilityStatus
    metrics: ModelBenchmarkMetrics | None = None
    blocking_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    claim_boundary: str = (
        "Metrics are reportable only when the independent-run audit passes and every evaluated sample belongs "
        "to the declared held-out split. Results remain specific to the represented simulator, objects, faults, "
        "operating conditions, and label process."
    )

    @field_validator("run_ids")
    @classmethod
    def unique_run_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("benchmark run_ids must be unique")
        return value


class BenchmarkEvidenceScope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope_id: str = Field(min_length=1)
    dataset_manifest_id: str = Field(min_length=1)
    dataset_manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")
    supported_axes: list[str] = Field(default_factory=list)
    unsupported_axes: list[str] = Field(default_factory=list)
    seed_only: bool = True
    frozen_split: bool = True
    test_used_for_tuning: bool = False
    flight_data_present: bool = False
    satellite_instance_count: int = Field(default=1, ge=1)
    claim_boundary: str = (
        "Benchmark evidence scope limits which release claims may be made from a metric report. "
        "It is separate from metric computation."
    )

    @model_validator(mode="after")
    def validate_axes(self) -> "BenchmarkEvidenceScope":
        overlap = set(self.supported_axes) & set(self.unsupported_axes)
        if overlap:
            raise ValueError(f"benchmark axes cannot be both supported and unsupported: {sorted(overlap)}")
        return self


class BenchmarkReleaseDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-benchmark-release-decision.v1"
    status: BenchmarkReleaseStatus
    claim_type: BenchmarkClaimType
    report_model_id: str
    report_model_version: str
    blocking_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    allowed_claim_boundary: str


class BenchmarkReleaseGate:
    _CROSS_CONDITION_AXES = {
        "severity",
        "initial_state",
        "operating_condition",
        "fault_timing",
        "noise",
        "dropout",
    }

    def decide(
        self,
        report: ModelBenchmarkReport,
        scope: BenchmarkEvidenceScope,
        *,
        claim_type: BenchmarkClaimType,
    ) -> BenchmarkReleaseDecision:
        blocking: list[str] = []
        warnings: list[str] = []

        if report.status != BenchmarkStatus.PASS:
            blocking.append("BENCHMARK_REPORT_NOT_PASS")
        if report.metrics is None:
            blocking.append("BENCHMARK_METRICS_MISSING")
        if report.run_audit_status != CredibilityStatus.PASS:
            blocking.append("INDEPENDENT_RUN_AUDIT_NOT_PASS")
        if not scope.frozen_split:
            blocking.append("SPLIT_NOT_FROZEN")
        if scope.test_used_for_tuning:
            blocking.append("TEST_USED_FOR_TUNING")

        supported = set(scope.supported_axes)
        if claim_type == BenchmarkClaimType.SEED_REGRESSION_RESULT:
            if not scope.seed_only:
                warnings.append("scope is broader than seed-only; claim remains limited to seed regression")
            allowed = "Seed-level simulator regression result; not a cross-condition, cross-satellite, or flight claim."
        elif claim_type == BenchmarkClaimType.CROSS_CONDITION_GENERALIZATION:
            missing = sorted(self._CROSS_CONDITION_AXES - supported)
            if missing:
                blocking.append("MISSING_CROSS_CONDITION_AXES")
                warnings.append(f"missing_axes={','.join(missing)}")
            if scope.seed_only:
                blocking.append("SEED_ONLY_SCOPE_CANNOT_SUPPORT_GENERALIZATION")
            allowed = "Cross-condition claim only if all required activity axes are represented and split discipline holds."
        elif claim_type == BenchmarkClaimType.CROSS_SATELLITE_GENERALIZATION:
            if scope.satellite_instance_count < 2:
                blocking.append("SINGLE_SATELLITE_SCOPE_CANNOT_SUPPORT_CROSS_SATELLITE")
            if scope.seed_only:
                blocking.append("SEED_ONLY_SCOPE_CANNOT_SUPPORT_GENERALIZATION")
            allowed = "Cross-satellite claim only if multiple satellite instances are represented."
        elif claim_type == BenchmarkClaimType.FLIGHT_VALIDATION:
            if not scope.flight_data_present:
                blocking.append("NO_FLIGHT_DATA_PRESENT")
            allowed = "Flight validation claim only if flight data is explicitly present in the evidence scope."
        else:
            raise ValueError(f"unsupported benchmark claim type: {claim_type}")

        blocking = list(dict.fromkeys(blocking))
        return BenchmarkReleaseDecision(
            status=BenchmarkReleaseStatus.BLOCKED if blocking else BenchmarkReleaseStatus.ACCEPTED,
            claim_type=claim_type,
            report_model_id=report.model_id,
            report_model_version=report.model_version,
            blocking_reasons=blocking,
            warnings=warnings,
            allowed_claim_boundary=allowed,
        )


def _zero_metrics() -> ModelBenchmarkMetrics:
    return ModelBenchmarkMetrics(
        sample_count=0,
        known_sample_count=0,
        unknown_sample_count=0,
        accuracy=0.0,
        balanced_accuracy=0.0,
        macro_f1=0.0,
        top_k_recall={"1": 0.0, "3": 0.0, "5": 0.0},
    )


def evaluate_model_benchmark(
    records: list[BenchmarkPredictionRecord],
    inventory: list[IndependentRunRecord],
    run_audit: IndependentRunAudit,
    *,
    evaluated_split: SplitName = SplitName.TEST,
    known_class_ids: set[str] | None = None,
    top_k_values: tuple[int, ...] = (1, 3, 5),
) -> ModelBenchmarkReport:
    blocking: list[str] = []
    warnings: list[str] = []
    if not top_k_values or any(k < 1 for k in top_k_values) or len(set(top_k_values)) != len(top_k_values):
        raise ValueError("top_k_values must be nonempty, unique, positive integers")
    if run_audit.status != CredibilityStatus.PASS:
        blocking.append("INDEPENDENT_RUN_AUDIT_NOT_PASS")

    inventory_by_id = {item.run_id: item for item in inventory}
    for record in records:
        item = inventory_by_id.get(record.run_id)
        if item is None:
            blocking.append(f"BENCHMARK_RUN_NOT_IN_INVENTORY:{record.run_id}")
        elif item.split != evaluated_split:
            blocking.append(
                f"BENCHMARK_SAMPLE_NOT_IN_DECLARED_SPLIT:{record.run_id}:{item.split.value}!={evaluated_split.value}"
            )
    model_pairs = {(item.prediction.model_id, item.prediction.model_version) for item in records}
    if len(model_pairs) > 1:
        blocking.append("MULTIPLE_MODEL_IDENTITIES_IN_ONE_BENCHMARK")
    model_id, model_version = next(iter(model_pairs), ("UNKNOWN", "UNKNOWN"))
    if not records:
        blocking.append("NO_BENCHMARK_PREDICTIONS")

    blocking = list(dict.fromkeys(blocking))
    run_ids = sorted({item.run_id for item in records})
    if blocking:
        return ModelBenchmarkReport(
            status=BenchmarkStatus.BLOCKED,
            model_id=model_id,
            model_version=model_version,
            evaluated_split=evaluated_split,
            run_ids=run_ids,
            run_audit_status=run_audit.status,
            blocking_reasons=blocking,
            warnings=warnings,
        )

    if known_class_ids is None:
        known_class_ids = {
            class_id
            for item in inventory
            for class_id in item.class_ids
            if class_id not in {"UNKNOWN", "OOD"}
        }
    class_order = sorted(
        {
            row.class_id
            for record in records
            for row in record.prediction.classes
            if row.class_id in known_class_ids
        }
    )
    known_records = [item for item in records if item.truth_class_id in known_class_ids]
    unknown_records = [item for item in records if item.truth_class_id not in known_class_ids]

    truth: list[str] = []
    predicted: list[str] = []
    ranking_rows: list[list[str]] = []
    probability_rows: list[list[float]] = []
    truth_indices: list[int] = []
    class_index = {class_id: index for index, class_id in enumerate(class_order)}
    for item in known_records:
        ranked = sorted(item.prediction.classes, key=lambda row: row.rank)
        ranking_rows.append([row.class_id for row in ranked])
        top = ranked[0].class_id if ranked and not item.open_set_rejected else "__REJECTED__"
        truth.append(item.truth_class_id)
        predicted.append(top)
        if item.truth_class_id in class_index:
            prob_by_id = {row.class_id: row.probability for row in item.prediction.classes}
            probability_rows.append([prob_by_id.get(class_id, 0.0) for class_id in class_order])
            truth_indices.append(class_index[item.truth_class_id])

    correct = sum(a == b for a, b in zip(truth, predicted))
    accuracy = correct / len(truth) if truth else 0.0
    per_class: list[PerClassMetric] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for class_id in sorted(set(truth)):
        tp = sum(t == class_id and p == class_id for t, p in zip(truth, predicted))
        fp = sum(t != class_id and p == class_id for t, p in zip(truth, predicted))
        fn = sum(t == class_id and p != class_id for t, p in zip(truth, predicted))
        support = sum(t == class_id for t in truth)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1s.append(f1)
        per_class.append(
            PerClassMetric(
                class_id=class_id,
                support=support,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )
    top_k_recall = {
        str(k): (
            sum(t in ranking[:k] for t, ranking in zip(truth, ranking_rows)) / len(truth)
            if truth
            else 0.0
        )
        for k in top_k_values
    }
    calibration = None
    if probability_rows and len(probability_rows) == len(known_records):
        calibration = calibration_metrics(np.asarray(probability_rows), np.asarray(truth_indices))
    unknown_false_accept_rate = None
    if unknown_records:
        false_accepts = sum(not item.open_set_rejected for item in unknown_records)
        unknown_false_accept_rate = false_accepts / len(unknown_records)
    latencies = [item.latency_ms for item in records if item.latency_ms is not None]
    metrics = ModelBenchmarkMetrics(
        sample_count=len(records),
        known_sample_count=len(known_records),
        unknown_sample_count=len(unknown_records),
        accuracy=accuracy,
        balanced_accuracy=float(np.mean(recalls)) if recalls else 0.0,
        macro_f1=float(np.mean(f1s)) if f1s else 0.0,
        top_k_recall=top_k_recall,
        calibration=calibration,
        unknown_false_accept_rate=unknown_false_accept_rate,
        mean_latency_ms=float(np.mean(latencies)) if latencies else None,
        p95_latency_ms=float(np.percentile(latencies, 95)) if latencies else None,
        per_class=per_class,
    )
    return ModelBenchmarkReport(
        status=BenchmarkStatus.PASS,
        model_id=model_id,
        model_version=model_version,
        evaluated_split=evaluated_split,
        run_ids=run_ids,
        run_audit_status=run_audit.status,
        metrics=metrics,
        warnings=warnings,
    )
