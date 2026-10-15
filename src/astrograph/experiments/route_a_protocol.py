"""Executable development-freeze protocol for Route A.

The protocol closes the gap between scattered statistical helpers and a
content-addressed experiment receipt.  Runtime prediction bundles are frozen
without truth.  A separate offline truth table may be joined only after every
candidate bundle hash is fixed.  Policy and rejection-threshold candidates are
then compared on exactly the preregistered independent development groups.

This module does not generate model predictions and cannot replace the missing
ADCS P2/temporal-v2 evidence chain.  It makes that chain executable and
fail-closed once the required immutable inputs exist.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from astrograph.assurance.analysis_plan import PrimaryAnalysisPlanV2
from astrograph.experiments.kg_coverage_protocol import (
    DevelopmentSelectionReceiptV2,
    GroupOutcomeV2,
    NonInferiorityMarginsV2,
    select_development_policy_v2,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _validate_sha256(value: str, field_name: str) -> str:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{field_name} must be lowercase SHA-256")
    return value


class FrozenProbabilityPrediction(BaseModel):
    """One label-free runtime probability prediction fixed before truth join."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-frozen-probability-prediction.v2"
    sample_id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    probabilities: dict[str, float]
    predicted_id: str = Field(min_length=1)
    runtime_trace_sha256: str
    row_sha256: str | None = None

    @field_validator("runtime_trace_sha256")
    @classmethod
    def runtime_hash(cls, value: str) -> str:
        return _validate_sha256(value, "runtime_trace_sha256")

    @model_validator(mode="after")
    def validate_prediction(self) -> "FrozenProbabilityPrediction":
        if not self.probabilities:
            raise ValueError("probabilities cannot be empty")
        if len(self.probabilities) != len(set(self.probabilities)):
            raise ValueError("probability labels must be unique")
        values = list(self.probabilities.values())
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
            raise ValueError("probabilities must be finite and in [0, 1]")
        if abs(sum(values) - 1.0) > 1e-8:
            raise ValueError("probabilities must sum to one")
        if self.predicted_id not in self.probabilities:
            raise ValueError("predicted_id absent from probabilities")
        maximum = max(values)
        if abs(self.probabilities[self.predicted_id] - maximum) > 1e-12:
            raise ValueError("predicted_id must be a maximum-probability label")
        expected = self.compute_sha256()
        if self.row_sha256 is not None and self.row_sha256 != expected:
            raise ValueError("row_sha256 mismatch")
        return self

    def compute_sha256(self) -> str:
        return _canonical_sha256(
            self.model_dump(mode="json", exclude={"row_sha256"})
        )


class FrozenPolicyPredictionBundle(BaseModel):
    """One policy/threshold candidate's truth-free development predictions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-route-a-frozen-prediction-bundle.v1"
    candidate_id: str = Field(min_length=1)
    base_policy_id: str = Field(min_length=1)
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    source_tree_sha256: str
    model_package_sha256: str
    temporal_policy_sha256: str
    evidence_package_sha256: str
    fusion_policy_sha256: str
    development_split_sha256: str
    prediction_generation_code_sha256: str
    generated_without_truth: bool = True
    freeze_status: str = Field(default="FROZEN", pattern=r"^FROZEN$")
    predictions: tuple[FrozenProbabilityPrediction, ...]
    bundle_sha256: str | None = None
    claim_boundary: str = (
        "Label-free development prediction artifact. It is not an evaluator output and cannot "
        "contain ground truth, correctness, or policy-selection outcomes."
    )

    @model_validator(mode="after")
    def validate_bundle(self) -> "FrozenPolicyPredictionBundle":
        for field_name in (
            "source_tree_sha256",
            "model_package_sha256",
            "temporal_policy_sha256",
            "evidence_package_sha256",
            "fusion_policy_sha256",
            "development_split_sha256",
            "prediction_generation_code_sha256",
        ):
            _validate_sha256(getattr(self, field_name), field_name)
        if not self.generated_without_truth:
            raise ValueError("prediction bundle must be generated without truth")
        if not self.predictions:
            raise ValueError("prediction bundle cannot be empty")
        sample_ids = [row.sample_id for row in self.predictions]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("prediction bundle contains duplicate sample_id values")
        label_sets = {tuple(sorted(row.probabilities)) for row in self.predictions}
        if len(label_sets) != 1:
            raise ValueError("all predictions must share one declared label universe")
        expected = self.compute_sha256()
        if self.bundle_sha256 is not None and self.bundle_sha256 != expected:
            raise ValueError("bundle_sha256 mismatch")
        return self

    @property
    def group_ids(self) -> tuple[str, ...]:
        return tuple(sorted({row.group_id for row in self.predictions}))

    @property
    def label_universe(self) -> tuple[str, ...]:
        return tuple(sorted(self.predictions[0].probabilities))

    def compute_sha256(self) -> str:
        payload = self.model_dump(
            mode="json",
            exclude={"bundle_sha256"},
        )
        # Persist row content addresses in the canonical bundle even when an
        # input producer omitted the optional row_sha256 field.
        payload["predictions"] = [
            {
                **row.model_dump(mode="json", exclude={"row_sha256"}),
                "row_sha256": row.compute_sha256(),
            }
            for row in self.predictions
        ]
        return _canonical_sha256(payload)


class OfflineTruthRow(BaseModel):
    """Evaluator-side truth row, never available to runtime prediction code."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    true_id: str = Field(min_length=1)


class OfflineTruthTable(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-route-a-offline-truth-table.v1"
    dataset_id: str = Field(min_length=1)
    development_split_sha256: str
    joined_after_prediction_freeze: bool = True
    eligible_prediction_bundle_sha256s: tuple[str, ...]
    rows: tuple[OfflineTruthRow, ...]
    truth_table_sha256: str | None = None
    claim_boundary: str = (
        "Evaluator-only truth table. Runtime score traces, prediction generation, fusion, and "
        "window selection must complete and freeze before this artifact is read."
    )

    @model_validator(mode="after")
    def validate_truth_table(self) -> "OfflineTruthTable":
        _validate_sha256(self.development_split_sha256, "development_split_sha256")
        if not self.joined_after_prediction_freeze:
            raise ValueError("truth table must be joined after prediction freeze")
        if not self.eligible_prediction_bundle_sha256s:
            raise ValueError("eligible prediction bundle hashes are required")
        for digest in self.eligible_prediction_bundle_sha256s:
            _validate_sha256(digest, "eligible_prediction_bundle_sha256")
        if len(self.eligible_prediction_bundle_sha256s) != len(
            set(self.eligible_prediction_bundle_sha256s)
        ):
            raise ValueError("duplicate eligible prediction bundle hash")
        sample_ids = [row.sample_id for row in self.rows]
        if not self.rows or len(sample_ids) != len(set(sample_ids)):
            raise ValueError("truth rows must be nonempty with unique sample_id values")
        expected = self.compute_sha256()
        if self.truth_table_sha256 is not None and self.truth_table_sha256 != expected:
            raise ValueError("truth_table_sha256 mismatch")
        return self

    def compute_sha256(self) -> str:
        return _canonical_sha256(
            self.model_dump(mode="json", exclude={"truth_table_sha256"})
        )


class RouteADevelopmentFreezeReceipt(BaseModel):
    """Content-addressed development result; not a confirmatory result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-route-a-development-freeze-receipt.v1"
    analysis_plan_sha256: str
    truth_table_sha256: str
    prediction_bundle_sha256s: dict[str, str]
    group_outcomes: tuple[dict[str, Any], ...]
    selection: dict[str, Any]
    selected_candidate_id: str
    selected_base_policy_id: str
    selected_confidence_threshold: float
    development_groups: tuple[str, ...]
    forbidden_post_hoc_groups: tuple[str, ...]
    result_status: str = "DEVELOPMENT_SELECTED_NOT_CONFIRMATORY"
    receipt_sha256: str | None = None
    claim_boundary: str = (
        "This receipt selects a Route A fusion/rejection candidate using only the frozen "
        "development groups named in the analysis plan. It does not validate groups 4/9, an "
        "external simulator, HIL, flight, safety, or universal KG gain."
    )

    @model_validator(mode="after")
    def validate_receipt(self) -> "RouteADevelopmentFreezeReceipt":
        _validate_sha256(self.analysis_plan_sha256, "analysis_plan_sha256")
        _validate_sha256(self.truth_table_sha256, "truth_table_sha256")
        for candidate_id, digest in self.prediction_bundle_sha256s.items():
            if not candidate_id.strip():
                raise ValueError("empty candidate ID in prediction bundle map")
            _validate_sha256(digest, "prediction_bundle_sha256")
        if set(self.development_groups) & set(self.forbidden_post_hoc_groups):
            raise ValueError("development groups overlap forbidden post-hoc groups")
        expected = self.compute_sha256()
        if self.receipt_sha256 is not None and self.receipt_sha256 != expected:
            raise ValueError("receipt_sha256 mismatch")
        return self

    def compute_sha256(self) -> str:
        return _canonical_sha256(
            self.model_dump(mode="json", exclude={"receipt_sha256"})
        )


def _ece(rows: Sequence[tuple[float, bool]], bins: int = 10) -> float:
    if not rows:
        return 0.0
    total = len(rows)
    value = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        bucket = [
            (confidence, hit)
            for confidence, hit in rows
            if confidence >= low and (confidence < high or index == bins - 1)
        ]
        if not bucket:
            continue
        accuracy = sum(hit for _, hit in bucket) / len(bucket)
        mean_confidence = sum(confidence for confidence, _ in bucket) / len(bucket)
        value += len(bucket) / total * abs(accuracy - mean_confidence)
    return value


def _aurc(rows: Sequence[tuple[float, bool]]) -> float:
    """Area under risk-coverage with all observations retained in the curve."""

    if not rows:
        return 1.0
    ranked = sorted(rows, key=lambda item: item[0], reverse=True)
    errors = 0
    previous_coverage = 0.0
    previous_risk = 0.0
    area = 0.0
    total = len(ranked)
    for index, (_, hit) in enumerate(ranked, start=1):
        errors += int(not hit)
        coverage = index / total
        risk = errors / index
        area += (coverage - previous_coverage) * (previous_risk + risk) / 2.0
        previous_coverage = coverage
        previous_risk = risk
    return area


def evaluate_prediction_bundle(
    bundle: FrozenPolicyPredictionBundle,
    truth_table: OfflineTruthTable,
) -> tuple[GroupOutcomeV2, ...]:
    """Join truth offline and return exactly one outcome per independent group."""

    bundle_hash = bundle.compute_sha256()
    if bundle_hash not in set(truth_table.eligible_prediction_bundle_sha256s):
        raise ValueError("truth table was not authorized for this frozen prediction bundle")
    if bundle.development_split_sha256 != truth_table.development_split_sha256:
        raise ValueError("prediction/truth development split hash mismatch")
    prediction_by_id = {row.sample_id: row for row in bundle.predictions}
    truth_by_id = {row.sample_id: row for row in truth_table.rows}
    if set(prediction_by_id) != set(truth_by_id):
        raise ValueError("prediction and truth sample sets must match exactly")
    grouped: dict[str, list[tuple[FrozenProbabilityPrediction, OfflineTruthRow]]] = defaultdict(list)
    for sample_id in sorted(prediction_by_id):
        prediction = prediction_by_id[sample_id]
        truth = truth_by_id[sample_id]
        if prediction.group_id != truth.group_id:
            raise ValueError(f"prediction/truth group mismatch for {sample_id}")
        if truth.true_id not in prediction.probabilities:
            raise ValueError(f"truth label outside prediction universe for {sample_id}")
        grouped[prediction.group_id].append((prediction, truth))

    outcomes: list[GroupOutcomeV2] = []
    for group_id, rows in sorted(grouped.items()):
        hits: list[bool] = []
        confidence_hits: list[tuple[float, bool]] = []
        accepted_hits: list[bool] = []
        nll_values: list[float] = []
        brier_values: list[float] = []
        labels = bundle.label_universe
        for prediction, truth in rows:
            confidence = max(prediction.probabilities.values())
            hit = prediction.predicted_id == truth.true_id
            hits.append(hit)
            confidence_hits.append((confidence, hit))
            if confidence >= bundle.confidence_threshold:
                accepted_hits.append(hit)
            probability_true = max(prediction.probabilities[truth.true_id], 1e-15)
            nll_values.append(-math.log(probability_true))
            brier_values.append(
                sum(
                    (
                        prediction.probabilities[label]
                        - (1.0 if label == truth.true_id else 0.0)
                    )
                    ** 2
                    for label in labels
                )
            )
        accepted_coverage = len(accepted_hits) / len(rows)
        # Empty acceptance is treated as maximal risk. This prevents a candidate
        # from winning by rejecting every trajectory.
        selective_risk = (
            1.0
            if not accepted_hits
            else 1.0 - sum(accepted_hits) / len(accepted_hits)
        )
        outcomes.append(
            GroupOutcomeV2(
                candidate_id=bundle.candidate_id,
                group_id=group_id,
                observation_count=len(rows),
                identification_accuracy=sum(hits) / len(hits),
                accepted_coverage=accepted_coverage,
                selective_risk=selective_risk,
                nll=fmean(nll_values),
                brier=fmean(brier_values),
                ece=_ece(confidence_hits),
                aurc=_aurc(confidence_hits),
            )
        )
    return tuple(outcomes)


def _selection_to_dict(selection: DevelopmentSelectionReceiptV2) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(selection)


def freeze_route_a_development_selection(
    *,
    analysis_plan: PrimaryAnalysisPlanV2,
    prediction_bundles: Iterable[FrozenPolicyPredictionBundle],
    truth_table: OfflineTruthTable,
) -> RouteADevelopmentFreezeReceipt:
    """Evaluate and freeze one Route A development selection receipt."""

    if analysis_plan.status != "FROZEN":
        raise ValueError("Route A development selection requires a frozen analysis plan")
    bundles = tuple(prediction_bundles)
    if not bundles:
        raise ValueError("prediction bundles are required")
    by_candidate = {bundle.candidate_id: bundle for bundle in bundles}
    if len(by_candidate) != len(bundles):
        raise ValueError("duplicate candidate_id prediction bundle")
    if set(by_candidate) != set(analysis_plan.candidate_ids):
        raise ValueError(
            "prediction bundle candidate IDs must equal the frozen analysis-plan grid"
        )
    baseline_id = analysis_plan.baseline_candidate_id
    if baseline_id not in by_candidate:
        raise ValueError("baseline prediction bundle missing")
    reference = by_candidate[baseline_id]
    reference_samples = {row.sample_id for row in reference.predictions}
    reference_groups = set(reference.group_ids)
    plan_bindings = {
        "source_tree_sha256": analysis_plan.source_tree_sha256,
        "model_package_sha256": analysis_plan.model_package_sha256,
        "temporal_policy_sha256": analysis_plan.temporal_policy_sha256,
        "development_split_sha256": analysis_plan.development_split_sha256,
        "prediction_generation_code_sha256": analysis_plan.prediction_generation_code_sha256,
    }
    for bundle in bundles:
        candidate_id = bundle.candidate_id
        if bundle.confidence_threshold != analysis_plan.candidate_thresholds[candidate_id]:
            raise ValueError(
                f"candidate {candidate_id} threshold differs from frozen analysis plan"
            )
        if bundle.fusion_policy_sha256 != analysis_plan.candidate_fusion_policy_sha256s[candidate_id]:
            raise ValueError(
                f"candidate {candidate_id} fusion policy differs from frozen analysis plan"
            )
        if bundle.evidence_package_sha256 != analysis_plan.candidate_evidence_package_sha256s[candidate_id]:
            raise ValueError(
                f"candidate {candidate_id} evidence package differs from frozen analysis plan"
            )
        if {row.sample_id for row in bundle.predictions} != reference_samples:
            raise ValueError("all candidate bundles must contain the same frozen sample set")
        if set(bundle.group_ids) != reference_groups:
            raise ValueError("all candidate bundles must contain the same independent groups")
        for field_name, expected_digest in plan_bindings.items():
            if getattr(bundle, field_name) != expected_digest:
                raise ValueError(
                    f"candidate {candidate_id} differs from frozen plan on {field_name}"
                )
    expected_groups = set(analysis_plan.development_groups)
    if reference_groups != expected_groups:
        raise ValueError(
            f"prediction groups must equal frozen development groups; "
            f"observed={sorted(reference_groups)} expected={sorted(expected_groups)}"
        )
    if reference_groups & set(analysis_plan.forbidden_post_hoc_groups):
        raise ValueError("forbidden groups 4/9 cannot enter Route A development selection")

    bundle_hashes = {
        candidate_id: bundle.compute_sha256()
        for candidate_id, bundle in sorted(by_candidate.items())
    }
    if set(truth_table.eligible_prediction_bundle_sha256s) != set(bundle_hashes.values()):
        raise ValueError(
            "truth table must be bound to exactly the frozen candidate prediction bundles"
        )
    outcomes = tuple(
        outcome
        for candidate_id in analysis_plan.candidate_ids
        for outcome in evaluate_prediction_bundle(by_candidate[candidate_id], truth_table)
    )
    margins_plan = analysis_plan.noninferiority_margins
    if margins_plan is None:  # guarded by the frozen-plan validator, kept explicit
        raise ValueError("frozen analysis plan lacks non-inferiority margins")
    margins = NonInferiorityMarginsV2(
        max_accuracy_drop=margins_plan.max_accuracy_drop,
        max_coverage_drop=margins_plan.max_coverage_drop,
        max_selective_risk_increase=margins_plan.max_selective_risk_increase,
        max_nll_increase=margins_plan.max_nll_increase,
        max_brier_increase=margins_plan.max_brier_increase,
        max_ece_increase=margins_plan.max_ece_increase,
        max_aurc_increase=margins_plan.max_aurc_increase,
        max_worst_group_accuracy_drop=margins_plan.max_worst_group_accuracy_drop,
    )
    selection = select_development_policy_v2(
        outcomes,
        baseline_candidate_id=baseline_id,
        development_groups=analysis_plan.development_groups,
        forbidden_groups=analysis_plan.forbidden_post_hoc_groups,
        margins=margins,
        confidence_level=analysis_plan.confidence_level,
        bootstrap_replicates=analysis_plan.bootstrap_replicates,
        bootstrap_seed_sha256=analysis_plan.bootstrap_seed_sha256,
        small_group_warning_threshold=analysis_plan.small_group_warning_threshold,
    )
    selected_bundle = by_candidate[selection.selected_candidate_id]
    provisional = RouteADevelopmentFreezeReceipt(
        analysis_plan_sha256=analysis_plan.compute_sha256(),
        truth_table_sha256=truth_table.compute_sha256(),
        prediction_bundle_sha256s=bundle_hashes,
        group_outcomes=tuple(
            {
                "candidate_id": outcome.candidate_id,
                "group_id": outcome.group_id,
                "observation_count": outcome.observation_count,
                "identification_accuracy": outcome.identification_accuracy,
                "accepted_coverage": outcome.accepted_coverage,
                "selective_risk": outcome.selective_risk,
                "nll": outcome.nll,
                "brier": outcome.brier,
                "ece": outcome.ece,
                "aurc": outcome.aurc,
            }
            for outcome in outcomes
        ),
        selection=_selection_to_dict(selection),
        selected_candidate_id=selection.selected_candidate_id,
        selected_base_policy_id=selected_bundle.base_policy_id,
        selected_confidence_threshold=selected_bundle.confidence_threshold,
        development_groups=tuple(sorted(analysis_plan.development_groups)),
        forbidden_post_hoc_groups=tuple(
            sorted(analysis_plan.forbidden_post_hoc_groups)
        ),
    )
    return provisional.model_copy(
        update={"receipt_sha256": provisional.compute_sha256()}
    )


__all__ = [
    "FrozenPolicyPredictionBundle",
    "FrozenProbabilityPrediction",
    "OfflineTruthRow",
    "OfflineTruthTable",
    "RouteADevelopmentFreezeReceipt",
    "evaluate_prediction_bundle",
    "freeze_route_a_development_selection",
]
