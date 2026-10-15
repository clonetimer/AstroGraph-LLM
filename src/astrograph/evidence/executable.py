from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.investigation.models import ClaimType


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_id(prefix: str, payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(raw).hexdigest()[:20].upper()}"


class EvidenceRole(str, Enum):
    REQUIRED = "REQUIRED"
    SUPPORTING = "SUPPORTING"
    COUNTER = "COUNTER"
    EXCLUSION = "EXCLUSION"


class ComparisonOperator(str, Enum):
    GREATER_THAN = "GREATER_THAN"
    GREATER_OR_EQUAL = "GREATER_OR_EQUAL"
    LESS_THAN = "LESS_THAN"
    LESS_OR_EQUAL = "LESS_OR_EQUAL"
    EQUAL = "EQUAL"
    NOT_EQUAL = "NOT_EQUAL"


class EvidenceFactStatus(str, Enum):
    SATISFIED = "SATISFIED"
    NOT_SATISFIED = "NOT_SATISFIED"
    CONTRADICTED = "CONTRADICTED"
    MISSING_DATA = "MISSING_DATA"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    INVALID_DATA = "INVALID_DATA"
    UNKNOWN = "UNKNOWN"




class FeatureResultValidity(str, Enum):
    VALID = "VALID"
    MISSING_VALUE = "MISSING_VALUE"
    MISSING_WINDOW_DATA = "MISSING_WINDOW_DATA"
    INSUFFICIENT_SAMPLES = "INSUFFICIENT_SAMPLES"
    INVALID_NONFINITE = "INVALID_NONFINITE"
    INVALID_QUALITY_FLAG = "INVALID_QUALITY_FLAG"
    INVALID_SAMPLE_COUNT = "INVALID_SAMPLE_COUNT"


_MISSING_QUALITY_FLAGS = {
    "MISSING",
    "MISSING_WINDOW_DATA",
    "PERTURBED_MISSING_DATA",
    "INSUFFICIENT_SAMPLES",
    "EMPTY",
    "MISSING_REQUIRED_PORT",
}
_INVALID_QUALITY_FLAGS = {
    "INVALID",
    "NONFINITE_INPUT",
    "NONFINITE_CURRENT_INPUT",
    "NONFINITE_BASELINE_INPUT",
    "NO_FINITE_RESULT",
    "OPERATOR_ERROR",
}


def classify_feature_result_validity(feature_result: "FeatureResult | None") -> tuple[FeatureResultValidity, str]:
    """Classify whether a FeatureResult is admissible for formal threshold semantics.

    Formal evidence may only compare a finite observation backed by at least one sample.
    Known data-quality failure flags are converted into missing/invalid semantics before
    threshold comparison.  Audit-only perturbation flags are intentionally not blockers.
    """
    if feature_result is None or feature_result.value is None:
        return FeatureResultValidity.MISSING_VALUE, "feature result/value unavailable"
    normalized_flags = {str(flag).strip().upper() for flag in feature_result.quality_flags if str(flag).strip()}
    missing = sorted(normalized_flags & _MISSING_QUALITY_FLAGS)
    if missing:
        validity = (
            FeatureResultValidity.INSUFFICIENT_SAMPLES
            if "INSUFFICIENT_SAMPLES" in missing
            else FeatureResultValidity.MISSING_WINDOW_DATA
        )
        return validity, "blocking missing-data quality flag(s): " + ",".join(missing)
    invalid = sorted(normalized_flags & _INVALID_QUALITY_FLAGS)
    if invalid:
        return FeatureResultValidity.INVALID_QUALITY_FLAG, "blocking invalid-data quality flag(s): " + ",".join(invalid)
    try:
        value = float(feature_result.value)
    except (TypeError, ValueError):
        return FeatureResultValidity.INVALID_NONFINITE, "feature value is not numeric"
    if not math.isfinite(value):
        return FeatureResultValidity.INVALID_NONFINITE, "feature value is non-finite"
    if int(feature_result.sample_count) <= 0:
        return FeatureResultValidity.INVALID_SAMPLE_COUNT, "feature sample_count must be positive for formal evidence"
    return FeatureResultValidity.VALID, "finite admitted feature result"


def _validity_fact_status(validity: FeatureResultValidity) -> EvidenceFactStatus:
    if validity in {
        FeatureResultValidity.MISSING_VALUE,
        FeatureResultValidity.MISSING_WINDOW_DATA,
        FeatureResultValidity.INSUFFICIENT_SAMPLES,
    }:
        return EvidenceFactStatus.MISSING_DATA
    if validity == FeatureResultValidity.VALID:
        raise ValueError("VALID feature result has no fail-closed status")
    return EvidenceFactStatus.INVALID_DATA


class FeatureAggregation(str, Enum):
    MAX_ABS_PAIRED_RESIDUAL = "MAX_ABS_PAIRED_RESIDUAL"
    MEAN_ABS_PAIRED_RESIDUAL = "MEAN_ABS_PAIRED_RESIDUAL"
    MEAN_PAIRED_RESIDUAL = "MEAN_PAIRED_RESIDUAL"
    LINEAR_SLOPE = "LINEAR_SLOPE"
    LAST_PAIRED_RESIDUAL = "LAST_PAIRED_RESIDUAL"
    AVAILABILITY_DROP_FRACTION = "AVAILABILITY_DROP_FRACTION"
    MAX_ABS_STATUS_RESIDUAL = "MAX_ABS_STATUS_RESIDUAL"
    MAX_VALUE = "MAX_VALUE"
    MIN_VALUE = "MIN_VALUE"
    MEAN_VALUE = "MEAN_VALUE"
    TRUE_FRACTION = "TRUE_FRACTION"
    FALSE_FRACTION = "FALSE_FRACTION"


class EpisodeWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    paired_nominal_run_id: str = Field(min_length=1)
    start_s: float = Field(ge=0.0)
    end_s: float = Field(ge=0.0)
    interval: str = Field(default="closed", pattern=r"^closed$")
    source: str = Field(min_length=1)
    source_is_label_free: bool = True
    operating_condition: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_window(self) -> "EpisodeWindow":
        if self.end_s < self.start_s:
            raise ValueError("episode end_s must be >= start_s")
        return self


class FeatureDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-feature-definition.v1"
    feature_definition_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    source_field_template: str = Field(min_length=1)
    aggregation: FeatureAggregation
    unit: str = Field(min_length=1)
    window_scope: str = Field(default="episode", pattern=r"^(episode|full_run)$")
    algorithm_version: str = Field(min_length=1)
    required_semantics: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    claim_boundary: str = "E0 synthetic paired-run feature; not a flight-calibrated health metric."


class FeatureResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-feature-result.v1"
    feature_result_id: str = Field(min_length=1)
    feature_definition_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    paired_nominal_run_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    resolved_field: str = Field(min_length=1)
    component_instance: str = Field(min_length=1)
    value: float | None
    unit: str = Field(min_length=1)
    window_start_s: float = Field(ge=0.0)
    window_end_s: float = Field(ge=0.0)
    sample_count: int = Field(ge=0)
    algorithm_version: str = Field(min_length=1)
    input_data_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    quality_flags: list[str] = Field(default_factory=list)
    claim_type: ClaimType = ClaimType.DERIVED
    created_at: str = Field(default_factory=_now)


class ConjunctiveFeatureCondition(BaseModel):
    """An additional (AND) feature condition on the same evidence definition.

    Lets a single evidence definition express e.g. "star-gyro innovation in band
    AND RW command residual low", which is needed to disambiguate faults whose
    primary feature ranges overlap (GYRO_BIAS_DRIFT vs ADCS_RW_JAM). The main
    condition (comparison + threshold) must be satisfied first; every conjunct
    must then also be satisfied. Empty list keeps the historical single-feature
    behaviour exactly.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-conjunctive-feature-condition.v1"
    feature_definition_id: str = Field(min_length=1)
    comparison: ComparisonOperator
    threshold: float
    threshold_unit: str = Field(min_length=1)


class EvidenceDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-evidence-definition.v1"
    evidence_definition_id: str = Field(min_length=1)
    target_fault_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    feature_definition_id: str = Field(min_length=1)
    role: EvidenceRole
    comparison: ComparisonOperator
    threshold: float
    threshold_unit: str = Field(min_length=1)
    critical: bool = False
    evidence_level: str = Field(default="E0", pattern=r"^E[0-4]$")
    source_refs: list[str] = Field(default_factory=list)
    applicable_condition_ids: list[str] = Field(default_factory=list)
    and_feature_conditions: list[ConjunctiveFeatureCondition] = Field(default_factory=list)
    claim_boundary: str = "E0 synthetic engineering threshold; not a flight-certified limit."


class EvidenceFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-evidence-fact.v1"
    evidence_fact_id: str = Field(min_length=1)
    evidence_definition_id: str = Field(min_length=1)
    target_fault_id: str = Field(min_length=1)
    feature_result_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    role: EvidenceRole
    status: EvidenceFactStatus
    observed_value: float | None = None
    threshold_value: float
    unit: str = Field(min_length=1)
    comparison: ComparisonOperator
    explanation: str = ""
    source_refs: list[str] = Field(default_factory=list)
    claim_type: ClaimType = ClaimType.DERIVED
    created_at: str = Field(default_factory=_now)


class EpisodeEvidenceBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-episode-evidence-bundle.v1"
    episode: EpisodeWindow
    response_window: EpisodeWindow | None = None
    domain_id: str = Field(default="D3_RW", min_length=1)
    analysis_wheel_index: int | None = Field(default=None, ge=0, le=2)
    analysis_component_instance: str = Field(min_length=1)
    selected_wheel_index: int | None = Field(default=None, ge=0, le=2)
    selected_component_instance: str | None = None
    wheel_selection_status: str = Field(
        default="NOT_APPLICABLE",
        pattern=r"^(MECHANISM_DETECTED|NO_MECHANISM_DETECTED|NOT_APPLICABLE)$",
    )
    component_selection_status: str = Field(
        default="MECHANISM_DETECTED",
        pattern=r"^(MECHANISM_DETECTED|NO_MECHANISM_DETECTED|NOT_APPLICABLE)$",
    )
    analysis_metadata: dict[str, Any] = Field(default_factory=dict)
    feature_definitions: list[FeatureDefinition]
    feature_results: list[FeatureResult]
    evidence_definitions: list[EvidenceDefinition]
    evidence_facts: list[EvidenceFact]
    source_protocol_hashes: dict[str, str] = Field(default_factory=dict)
    evaluation_metadata: dict[str, Any] = Field(default_factory=dict)
    claim_boundary: str = (
        "Evidence bundle is generated from E0 synthetic telemetry and paired nominal runs; evaluation labels remain isolated."
    )


def compare_value(value: float, operator: ComparisonOperator, threshold: float) -> bool:
    if operator == ComparisonOperator.GREATER_THAN:
        return value > threshold
    if operator == ComparisonOperator.GREATER_OR_EQUAL:
        return value >= threshold
    if operator == ComparisonOperator.LESS_THAN:
        return value < threshold
    if operator == ComparisonOperator.LESS_OR_EQUAL:
        return value <= threshold
    if operator == ComparisonOperator.EQUAL:
        return value == threshold
    if operator == ComparisonOperator.NOT_EQUAL:
        return value != threshold
    raise ValueError(f"unsupported comparison operator: {operator}")


def evaluate_evidence(
    definition: EvidenceDefinition,
    feature_result: FeatureResult | None,
    *,
    all_feature_results: Sequence[FeatureResult] | None = None,
) -> EvidenceFact:
    validity, validity_reason = classify_feature_result_validity(feature_result)
    if validity != FeatureResultValidity.VALID:
        status = _validity_fact_status(validity)
        observed = None
        explanation = f"Formal threshold comparison not performed: {validity.value} ({validity_reason})."
        feature_result_id = feature_result.feature_result_id if feature_result is not None else "MISSING"
        episode_id = feature_result.episode_id if feature_result is not None else "UNKNOWN"
    else:
        assert feature_result is not None and feature_result.value is not None
        observed = float(feature_result.value)
        matched = compare_value(observed, definition.comparison, float(definition.threshold))
        conjunct_failure: tuple[EvidenceFactStatus, str] | None = None
        failed_conjunct: tuple[ConjunctiveFeatureCondition, float] | None = None
        if matched and definition.and_feature_conditions:
            results_by_feature = {
                item.feature_definition_id: item
                for item in (all_feature_results or [])
                if item.episode_id == feature_result.episode_id
            }
            for conjunct in definition.and_feature_conditions:
                conjunct_result = results_by_feature.get(conjunct.feature_definition_id)
                conjunct_validity, conjunct_reason = classify_feature_result_validity(conjunct_result)
                if conjunct_validity != FeatureResultValidity.VALID:
                    conjunct_failure = (
                        _validity_fact_status(conjunct_validity),
                        f"conjunct {conjunct.feature_definition_id}: {conjunct_validity.value} ({conjunct_reason})",
                    )
                    break
                assert conjunct_result is not None and conjunct_result.value is not None
                conjunct_value = float(conjunct_result.value)
                conjunct_matched = compare_value(
                    conjunct_value, conjunct.comparison, float(conjunct.threshold),
                )
                if not conjunct_matched:
                    matched = False
                    failed_conjunct = (conjunct, conjunct_value)
                    break
        if conjunct_failure is not None:
            status, detail = conjunct_failure
            explanation = (
                f"Observed {observed:g} {definition.threshold_unit}; main condition met but {detail}. "
                "Formal conjunctive evidence fails closed before threshold truth semantics."
            )
        elif definition.role in {EvidenceRole.COUNTER, EvidenceRole.EXCLUSION}:
            status = EvidenceFactStatus.CONTRADICTED if matched else EvidenceFactStatus.NOT_SATISFIED
        else:
            status = EvidenceFactStatus.SATISFIED if matched else EvidenceFactStatus.NOT_SATISFIED
        if failed_conjunct is not None:
            conjunct, conjunct_value = failed_conjunct
            explanation = (
                f"Observed {observed:g} {definition.threshold_unit}; main condition "
                f"{definition.comparison.value} {definition.threshold:g} met but conjunct "
                f"{conjunct.feature_definition_id} {conjunct.comparison.value} "
                f"{conjunct.threshold:g} ({conjunct_value:g} {conjunct.threshold_unit}) not met."
            )
        elif conjunct_failure is None:
            conjunct_count = len(definition.and_feature_conditions)
            conjunct_note = f" (with {conjunct_count} conjunct condition(s))" if conjunct_count else ""
            explanation = (
                f"Observed {observed:g} {definition.threshold_unit}; condition "
                f"{definition.comparison.value} {definition.threshold:g} is {'met' if matched else 'not met'}"
                f"{conjunct_note}."
            )
        feature_result_id = feature_result.feature_result_id
        episode_id = feature_result.episode_id
    fact_id = stable_id(
        "EF",
        {
            "evidence_definition_id": definition.evidence_definition_id,
            "feature_result_id": feature_result_id,
            "episode_id": episode_id,
            "status": status.value,
        },
    )
    return EvidenceFact(
        evidence_fact_id=fact_id,
        evidence_definition_id=definition.evidence_definition_id,
        target_fault_id=definition.target_fault_id,
        feature_result_id=feature_result_id,
        episode_id=episode_id,
        role=definition.role,
        status=status,
        observed_value=observed,
        threshold_value=definition.threshold,
        unit=definition.threshold_unit,
        comparison=definition.comparison,
        explanation=explanation,
        source_refs=definition.source_refs,
    )
