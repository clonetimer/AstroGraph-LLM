"""Benchmark campaign pipeline (engineering scaffolding).

STATUS (recorded 2026-08-04, ADR-004): DEVELOPMENT / NOT WIRED.  This module is
engineering scaffolding for independent-run benchmarking; it is not a diagnosis
capability and is not part of the S1 data-driven mainline's production surface.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from astrograph.evidence.executable import stable_id

from .benchmark import (
    BenchmarkClaimType,
    BenchmarkEvidenceScope,
    BenchmarkReleaseGate,
    BenchmarkReleaseStatus,
    BenchmarkStatus,
    ModelBenchmarkReport,
    PerClassMetric,
)
from .fault_catalog import FaultModeCatalog
from .selection import FaultModelBinding, FaultModelBindingSet


class FaultModelRecommendationStatus(str, Enum):
    RECOMMENDED = "RECOMMENDED"
    AMBIGUOUS = "AMBIGUOUS"
    NO_ELIGIBLE_MODEL = "NO_ELIGIBLE_MODEL"


class BenchmarkComparisonPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minimum_class_support: int = Field(default=1, ge=1)
    minimum_class_recall: float = Field(default=0.0, ge=0.0, le=1.0)
    minimum_class_f1: float = Field(default=0.0, ge=0.0, le=1.0)
    minimum_score_gap: float = Field(default=0.02, ge=0.0)
    require_release_gate_acceptance: bool = True
    enable_generated_bindings: bool = False
    weight_precision: float = Field(default=0.15, ge=0.0)
    weight_recall: float = Field(default=0.25, ge=0.0)
    weight_class_f1: float = Field(default=0.35, ge=0.0)
    weight_balanced_accuracy: float = Field(default=0.10, ge=0.0)
    weight_macro_f1: float = Field(default=0.10, ge=0.0)
    weight_top1_recall: float = Field(default=0.05, ge=0.0)


class BenchmarkModelCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    model_version: str
    fault_id: str
    score: float = Field(ge=0.0)
    class_metric: PerClassMetric
    balanced_accuracy: float = Field(ge=0.0, le=1.0)
    macro_f1: float = Field(ge=0.0, le=1.0)
    top1_recall: float = Field(ge=0.0, le=1.0)
    release_status: BenchmarkReleaseStatus
    eligible: bool
    blocking_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class FaultModelRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fault_id: str
    status: FaultModelRecommendationStatus
    selected_model_id: str | None = None
    selected_model_version: str | None = None
    candidates: list[BenchmarkModelCandidate] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    requires_human_review: bool = True


class ModelBenchmarkComparisonReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-model-benchmark-comparison-report.v1"
    comparison_id: str
    catalog_id: str
    claim_type: BenchmarkClaimType
    recommendations: list[FaultModelRecommendation]
    proposed_binding_set: FaultModelBindingSet
    metadata: dict[str, Any] = Field(default_factory=dict)
    claim_boundary: str = (
        "A recommendation ranks models within the supplied audited benchmark evidence. "
        "It does not prove flight validity and generated bindings remain disabled unless explicitly authorized."
    )


def _metric_for_fault(report: ModelBenchmarkReport, fault_id: str) -> PerClassMetric | None:
    if report.metrics is None:
        return None
    return next((item for item in report.metrics.per_class if item.class_id == fault_id), None)


def _candidate_score(
    report: ModelBenchmarkReport,
    metric: PerClassMetric,
    policy: BenchmarkComparisonPolicy,
) -> float:
    if report.metrics is None:
        raise ValueError("candidate scoring requires benchmark metrics")
    top1 = float(report.metrics.top_k_recall.get("1", report.metrics.accuracy))
    return float(
        policy.weight_precision * metric.precision
        + policy.weight_recall * metric.recall
        + policy.weight_class_f1 * metric.f1
        + policy.weight_balanced_accuracy * report.metrics.balanced_accuracy
        + policy.weight_macro_f1 * report.metrics.macro_f1
        + policy.weight_top1_recall * top1
    )


def compare_model_benchmarks(
    *,
    catalog: FaultModeCatalog,
    reports: list[ModelBenchmarkReport],
    scopes: dict[str, BenchmarkEvidenceScope],
    claim_type: BenchmarkClaimType = BenchmarkClaimType.SEED_REGRESSION_RESULT,
    policy: BenchmarkComparisonPolicy | None = None,
) -> ModelBenchmarkComparisonReport:
    """Compare audited model reports and create conservative fault-model binding proposals.

    ``scopes`` is keyed by ``model_id@model_version``. A missing scope blocks that model
    from recommendation because claim boundaries cannot be evaluated without it.
    """

    policy = policy or BenchmarkComparisonPolicy()
    gate = BenchmarkReleaseGate()
    recommendations: list[FaultModelRecommendation] = []
    bindings: list[FaultModelBinding] = []

    for fault_id in catalog.fault_ids():
        candidates: list[BenchmarkModelCandidate] = []
        for report in reports:
            metric = _metric_for_fault(report, fault_id)
            if metric is None:
                continue
            key = f"{report.model_id}@{report.model_version}"
            scope = scopes.get(key)
            blocking: list[str] = []
            warnings: list[str] = []
            release_status = BenchmarkReleaseStatus.BLOCKED
            if scope is None:
                blocking.append("BENCHMARK_EVIDENCE_SCOPE_MISSING")
            else:
                release = gate.decide(report, scope, claim_type=claim_type)
                release_status = release.status
                blocking.extend(release.blocking_reasons)
                warnings.extend(release.warnings)
            if report.status != BenchmarkStatus.PASS:
                blocking.append("BENCHMARK_REPORT_NOT_PASS")
            required_support = max(
                policy.minimum_class_support,
                catalog.get(fault_id).validation.minimum_independent_runs,
            )
            if metric.support < required_support:
                blocking.append("INSUFFICIENT_CLASS_SUPPORT")
                warnings.append(f"required_support={required_support}; observed_support={metric.support}")
            if metric.recall < policy.minimum_class_recall:
                blocking.append("CLASS_RECALL_BELOW_POLICY")
            if metric.f1 < policy.minimum_class_f1:
                blocking.append("CLASS_F1_BELOW_POLICY")
            if policy.require_release_gate_acceptance and release_status != BenchmarkReleaseStatus.ACCEPTED:
                blocking.append("RELEASE_GATE_NOT_ACCEPTED")
            score = _candidate_score(report, metric, policy) if report.metrics is not None else 0.0
            candidates.append(
                BenchmarkModelCandidate(
                    model_id=report.model_id,
                    model_version=report.model_version,
                    fault_id=fault_id,
                    score=score,
                    class_metric=metric,
                    balanced_accuracy=(report.metrics.balanced_accuracy if report.metrics else 0.0),
                    macro_f1=(report.metrics.macro_f1 if report.metrics else 0.0),
                    top1_recall=(
                        float(report.metrics.top_k_recall.get("1", report.metrics.accuracy))
                        if report.metrics
                        else 0.0
                    ),
                    release_status=release_status,
                    eligible=not blocking,
                    blocking_reasons=list(dict.fromkeys(blocking)),
                    warnings=warnings,
                )
            )

        candidates.sort(key=lambda item: (-int(item.eligible), -item.score, item.model_id, item.model_version))
        eligible = [item for item in candidates if item.eligible]
        reasons: list[str] = []
        selected: BenchmarkModelCandidate | None = None
        status = FaultModelRecommendationStatus.NO_ELIGIBLE_MODEL
        if eligible:
            selected = eligible[0]
            status = FaultModelRecommendationStatus.RECOMMENDED
            if len(eligible) > 1 and selected.score - eligible[1].score < policy.minimum_score_gap:
                status = FaultModelRecommendationStatus.AMBIGUOUS
                reasons.append("TOP_MODEL_SCORE_GAP_BELOW_POLICY")
        else:
            reasons.append("NO_MODEL_PASSED_FAULT_SPECIFIC_AND_RELEASE_GATES")

        recommendation = FaultModelRecommendation(
            fault_id=fault_id,
            status=status,
            selected_model_id=selected.model_id if selected else None,
            selected_model_version=selected.model_version if selected else None,
            candidates=candidates,
            reasons=reasons,
            requires_human_review=True,
        )
        recommendations.append(recommendation)
        if selected is not None and status == FaultModelRecommendationStatus.RECOMMENDED:
            bindings.append(
                FaultModelBinding(
                    fault_id=fault_id,
                    model_id=selected.model_id,
                    model_version=selected.model_version,
                    priority=max(100, int(round(selected.score * 1000))),
                    enabled=(
                        policy.enable_generated_bindings
                        and status == FaultModelRecommendationStatus.RECOMMENDED
                    ),
                    rationale=(
                        f"Benchmark recommendation score={selected.score:.6f}; "
                        f"claim_type={claim_type.value}; status={status.value}."
                    ),
                    metadata={
                        "generated_by": "compare_model_benchmarks",
                        "comparison_requires_expert_review": True,
                        "release_status": selected.release_status.value,
                    },
                )
            )

    comparison_id = stable_id(
        "BMCOMP",
        {
            "catalog_id": catalog.catalog_id,
            "claim_type": claim_type.value,
            "reports": sorted(f"{item.model_id}@{item.model_version}" for item in reports),
            "policy": policy.model_dump(mode="json"),
            "recommendations": [
                {
                    "fault_id": item.fault_id,
                    "status": item.status.value,
                    "model_id": item.selected_model_id,
                    "model_version": item.selected_model_version,
                }
                for item in recommendations
            ],
        },
    )
    return ModelBenchmarkComparisonReport(
        comparison_id=comparison_id,
        catalog_id=catalog.catalog_id,
        claim_type=claim_type,
        recommendations=recommendations,
        proposed_binding_set=FaultModelBindingSet(
            binding_set_id=f"{comparison_id}-proposed-bindings",
            bindings=bindings,
            metadata={
                "generated_from_benchmark_comparison": comparison_id,
                "bindings_enabled_by_policy": policy.enable_generated_bindings,
                "expert_review_required": True,
            },
        ),
        metadata={
            "report_count": len(reports),
            "scope_count": len(scopes),
            "policy": policy.model_dump(mode="json"),
        },
    )
