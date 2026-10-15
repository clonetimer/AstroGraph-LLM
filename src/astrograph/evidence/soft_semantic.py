"""Auditable quantitative-to-semantic evidence projection.

The projection is deliberately parallel to ``EvidenceFact``.  It cannot change
the hard evidence status or fusion result.  Without a development-selected,
content-addressed calibration policy it emits exact binary compatibility and
states that uncertainty is not quantified.  A complete policy may instead
produce fuzzy truth degrees and intervals for inequality predicates.
"""
from __future__ import annotations

import hashlib
import json
import math
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .executable import (
    ComparisonOperator,
    EpisodeEvidenceBundle,
    EvidenceFactStatus,
    compare_value,
)


_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_CALIBRATABLE_COMPARISONS = {
    ComparisonOperator.GREATER_THAN,
    ComparisonOperator.GREATER_OR_EQUAL,
    ComparisonOperator.LESS_THAN,
    ComparisonOperator.LESS_OR_EQUAL,
}


def _canonical_sha256(payload: Any) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        decay = math.exp(-value)
        return 1.0 / (1.0 + decay)
    growth = math.exp(value)
    return growth / (1.0 + growth)


def _signed_margin(
    observed: float,
    threshold: float,
    comparison: ComparisonOperator,
) -> float | None:
    if comparison in {
        ComparisonOperator.GREATER_THAN,
        ComparisonOperator.GREATER_OR_EQUAL,
    }:
        return observed - threshold
    if comparison in {
        ComparisonOperator.LESS_THAN,
        ComparisonOperator.LESS_OR_EQUAL,
    }:
        return threshold - observed
    return None


class SoftEvidencePolicyStatus(str, Enum):
    CANDIDATE = "CANDIDATE"
    DEVELOPMENT_SELECTED = "DEVELOPMENT_SELECTED"
    FROZEN = "FROZEN"


class SoftQuantificationMode(str, Enum):
    BINARY_COMPATIBILITY = "BINARY_COMPATIBILITY"
    LOGISTIC_INTERVAL = "LOGISTIC_INTERVAL"
    UNAVAILABLE = "UNAVAILABLE"


class SoftConditionCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature_definition_id: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    transition_width: float = Field(gt=0.0)
    observation_uncertainty: float = Field(default=0.0, ge=0.0)
    reference_uncertainty: float = Field(default=0.0, ge=0.0)
    threshold_uncertainty: float = Field(default=0.0, ge=0.0)
    interval_z: float = Field(default=1.96, gt=0.0)


class SoftEvidenceCalibrationEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_definition_id: str = Field(min_length=1)
    conditions: list[SoftConditionCalibration] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_features(self) -> "SoftEvidenceCalibrationEntry":
        feature_ids = [item.feature_definition_id for item in self.conditions]
        if len(feature_ids) != len(set(feature_ids)):
            raise ValueError("soft evidence calibration conditions must have unique feature IDs")
        return self


class SoftEvidenceCalibrationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["astrograph-soft-evidence-calibration-policy.v1"] = (
        "astrograph-soft-evidence-calibration-policy.v1"
    )
    policy_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    status: SoftEvidencePolicyStatus = SoftEvidencePolicyStatus.CANDIDATE
    entries: list[SoftEvidenceCalibrationEntry] = Field(default_factory=list)
    selected_on_groups: list[int] = Field(default_factory=list)
    forbidden_groups: list[int] = Field(default_factory=lambda: [4, 9])
    selection_receipt_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    policy_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    claim_boundary: str = (
        "Fuzzy truth calibrates telemetry predicates, not fault posterior probability, "
        "causal validity, flight readiness, or safety qualification."
    )

    @model_validator(mode="after")
    def validate_selection_and_hash(self) -> "SoftEvidenceCalibrationPolicy":
        definition_ids = [item.evidence_definition_id for item in self.entries]
        if len(definition_ids) != len(set(definition_ids)):
            raise ValueError("soft evidence calibration entries must have unique definition IDs")
        if set(self.selected_on_groups) & set(self.forbidden_groups):
            raise ValueError("forbidden post-hoc groups cannot select soft evidence calibration")
        if self.status is not SoftEvidencePolicyStatus.CANDIDATE:
            if not self.selected_on_groups or self.selection_receipt_sha256 is None:
                raise ValueError("selected/frozen soft evidence policy requires groups and receipt")
        if self.status is SoftEvidencePolicyStatus.FROZEN:
            if not self.entries:
                raise ValueError("frozen soft evidence policy requires calibration entries")
            unquantified = [
                f"{entry.evidence_definition_id}:{condition.feature_definition_id}"
                for entry in self.entries
                for condition in entry.conditions
                if (
                    condition.observation_uncertainty
                    + condition.reference_uncertainty
                    + condition.threshold_uncertainty
                )
                <= 0.0
            ]
            if unquantified:
                raise ValueError(
                    "frozen soft evidence policy requires a non-zero identified uncertainty "
                    f"component for every condition: {unquantified}"
                )
        expected = self.compute_sha256()
        if self.policy_sha256 is not None and self.policy_sha256 != expected:
            raise ValueError("soft evidence policy_sha256 mismatch")
        return self

    def canonical_payload(self) -> dict[str, Any]:
        return self.model_dump(
            mode="json",
            exclude={"policy_sha256"},
        )

    def compute_sha256(self) -> str:
        return _canonical_sha256(self.canonical_payload())


class SoftConditionAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature_definition_id: str
    comparison: ComparisonOperator
    observed_value: float | None = None
    threshold_value: float
    unit: str
    signed_margin: float | None = None
    normalized_margin: float | None = None
    truth_degree: float | None = Field(default=None, ge=0.0, le=1.0)
    truth_lower: float | None = Field(default=None, ge=0.0, le=1.0)
    truth_upper: float | None = Field(default=None, ge=0.0, le=1.0)
    quantification_mode: SoftQuantificationMode
    transition_width: float | None = Field(default=None, gt=0.0)
    observation_uncertainty: float | None = Field(default=None, ge=0.0)
    reference_uncertainty: float | None = Field(default=None, ge=0.0)
    threshold_uncertainty: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def ordered_interval(self) -> "SoftConditionAssessment":
        values = (self.truth_lower, self.truth_degree, self.truth_upper)
        if all(item is not None for item in values):
            lower, degree, upper = values
            if not float(lower) <= float(degree) <= float(upper):
                raise ValueError("soft truth interval must contain truth_degree")
        return self


class SoftEvidenceFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["astrograph-soft-evidence-fact.v1"] = (
        "astrograph-soft-evidence-fact.v1"
    )
    soft_evidence_fact_id: str = Field(min_length=1)
    evidence_fact_id: str = Field(min_length=1)
    evidence_definition_id: str = Field(min_length=1)
    target_fault_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    hard_status: EvidenceFactStatus
    predicate_truth_degree: float | None = Field(default=None, ge=0.0, le=1.0)
    predicate_truth_lower: float | None = Field(default=None, ge=0.0, le=1.0)
    predicate_truth_upper: float | None = Field(default=None, ge=0.0, le=1.0)
    quantification_mode: SoftQuantificationMode
    calibration_policy_id: str | None = None
    calibration_policy_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    conditions: list[SoftConditionAssessment] = Field(default_factory=list)
    uncertainty_quantified: bool = False
    claim_boundary: str = (
        "Predicate truth is not a fault posterior. Binary compatibility means no "
        "measurement, reference, or threshold uncertainty has been calibrated."
    )


class SoftEvidenceBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["astrograph-soft-evidence-bundle.v1"] = (
        "astrograph-soft-evidence-bundle.v1"
    )
    episode_id: str = Field(min_length=1)
    source_evidence_sha256: str = Field(pattern=_SHA256_PATTERN)
    facts: list[SoftEvidenceFact]
    calibration_policy_id: str | None = None
    calibration_policy_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    calibrated_fact_count: int = Field(ge=0)
    unavailable_fact_count: int = Field(ge=0)
    bundle_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    claim_boundary: str = (
        "Parallel audit artifact only. It does not modify EvidenceFact status, KG ranking, "
        "BiRRF, Governor disposition, active knowledge, or qualification state."
    )

    @model_validator(mode="after")
    def validate_counts_and_hash(self) -> "SoftEvidenceBundle":
        calibrated = sum(
            item.quantification_mode is SoftQuantificationMode.LOGISTIC_INTERVAL
            for item in self.facts
        )
        unavailable = sum(
            item.quantification_mode is SoftQuantificationMode.UNAVAILABLE
            for item in self.facts
        )
        if calibrated != self.calibrated_fact_count:
            raise ValueError("calibrated_fact_count mismatch")
        if unavailable != self.unavailable_fact_count:
            raise ValueError("unavailable_fact_count mismatch")
        expected = self.compute_sha256()
        if self.bundle_sha256 is not None and self.bundle_sha256 != expected:
            raise ValueError("soft evidence bundle_sha256 mismatch")
        return self

    def canonical_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"bundle_sha256"})

    def compute_sha256(self) -> str:
        return _canonical_sha256(self.canonical_payload())


def _source_evidence_sha256(bundle: EpisodeEvidenceBundle) -> str:
    return _canonical_sha256(
        {
            "episode_id": bundle.episode.episode_id,
            "feature_definitions": [item.model_dump(mode="json") for item in bundle.feature_definitions],
            "feature_results": [
                item.model_dump(mode="json", exclude={"created_at"})
                for item in bundle.feature_results
            ],
            "evidence_definitions": [item.model_dump(mode="json") for item in bundle.evidence_definitions],
            "evidence_facts": [
                item.model_dump(mode="json", exclude={"created_at"})
                for item in bundle.evidence_facts
            ],
            "source_protocol_hashes": dict(sorted(bundle.source_protocol_hashes.items())),
        }
    )


def _binary_degree(status: EvidenceFactStatus) -> float | None:
    if status in {EvidenceFactStatus.SATISFIED, EvidenceFactStatus.CONTRADICTED}:
        return 1.0
    if status is EvidenceFactStatus.NOT_SATISFIED:
        return 0.0
    return None


def _condition_assessment(
    *,
    feature_definition_id: str,
    comparison: ComparisonOperator,
    threshold: float,
    unit: str,
    observed: float | None,
    observed_unit: str | None,
    calibration: SoftConditionCalibration | None,
) -> SoftConditionAssessment:
    if observed_unit is not None and observed_unit != unit:
        raise ValueError(
            f"soft evidence feature/threshold unit mismatch for {feature_definition_id}: "
            f"{observed_unit} != {unit}"
        )
    if calibration is not None and calibration.unit != unit:
        raise ValueError(
            f"soft evidence calibration unit mismatch for {feature_definition_id}: "
            f"{calibration.unit} != {unit}"
        )
    if observed is None:
        return SoftConditionAssessment(
            feature_definition_id=feature_definition_id,
            comparison=comparison,
            threshold_value=threshold,
            unit=unit,
            quantification_mode=SoftQuantificationMode.UNAVAILABLE,
        )
    margin = _signed_margin(observed, threshold, comparison)
    if calibration is None or comparison not in _CALIBRATABLE_COMPARISONS or margin is None:
        degree = float(compare_value(observed, comparison, threshold))
        return SoftConditionAssessment(
            feature_definition_id=feature_definition_id,
            comparison=comparison,
            observed_value=observed,
            threshold_value=threshold,
            unit=unit,
            signed_margin=margin,
            truth_degree=degree,
            truth_lower=degree,
            truth_upper=degree,
            quantification_mode=SoftQuantificationMode.BINARY_COMPATIBILITY,
        )
    combined_uncertainty = math.sqrt(
        calibration.observation_uncertainty**2
        + calibration.reference_uncertainty**2
        + calibration.threshold_uncertainty**2
    )
    center = _sigmoid(margin / calibration.transition_width)
    delta = calibration.interval_z * combined_uncertainty
    lower = _sigmoid((margin - delta) / calibration.transition_width)
    upper = _sigmoid((margin + delta) / calibration.transition_width)
    return SoftConditionAssessment(
        feature_definition_id=feature_definition_id,
        comparison=comparison,
        observed_value=observed,
        threshold_value=threshold,
        unit=unit,
        signed_margin=margin,
        normalized_margin=margin / calibration.transition_width,
        truth_degree=center,
        truth_lower=lower,
        truth_upper=upper,
        quantification_mode=SoftQuantificationMode.LOGISTIC_INTERVAL,
        transition_width=calibration.transition_width,
        observation_uncertainty=calibration.observation_uncertainty,
        reference_uncertainty=calibration.reference_uncertainty,
        threshold_uncertainty=calibration.threshold_uncertainty,
    )


def build_soft_evidence_bundle(
    bundle: EpisodeEvidenceBundle,
    policy: SoftEvidenceCalibrationPolicy | None = None,
) -> SoftEvidenceBundle:
    """Project hard evidence into a non-voting quantitative semantic artifact."""

    definitions = {item.evidence_definition_id: item for item in bundle.evidence_definitions}
    results = {item.feature_definition_id: item for item in bundle.feature_results}
    policy_entries = {
        item.evidence_definition_id: item
        for item in (policy.entries if policy is not None else [])
    }
    unknown_policy_definitions = set(policy_entries) - set(definitions)
    if unknown_policy_definitions:
        raise ValueError(
            "soft evidence policy contains definitions outside the source bundle: "
            f"{sorted(unknown_policy_definitions)}"
        )
    policy_sha256 = policy.compute_sha256() if policy is not None else None
    source_evidence_sha256 = _source_evidence_sha256(bundle)
    soft_facts: list[SoftEvidenceFact] = []
    for fact in bundle.evidence_facts:
        definition = definitions.get(fact.evidence_definition_id)
        if definition is None:
            raise ValueError(
                f"evidence fact lacks definition for soft projection: {fact.evidence_fact_id}"
            )
        entry = policy_entries.get(definition.evidence_definition_id)
        calibrations = {
            item.feature_definition_id: item for item in (entry.conditions if entry else [])
        }
        expected_features = {
            definition.feature_definition_id,
            *(item.feature_definition_id for item in definition.and_feature_conditions),
        }
        calibration_complete = bool(entry) and set(calibrations) == expected_features
        if entry is not None and not calibration_complete:
            raise ValueError(
                "soft calibration must cover the primary and every conjunct feature exactly: "
                f"{definition.evidence_definition_id}"
            )

        condition_specs = [
            (
                definition.feature_definition_id,
                definition.comparison,
                float(definition.threshold),
                definition.threshold_unit,
            ),
            *(
                (
                    item.feature_definition_id,
                    item.comparison,
                    float(item.threshold),
                    item.threshold_unit,
                )
                for item in definition.and_feature_conditions
            ),
        ]
        assessments = []
        for feature_id, comparison, threshold, unit in condition_specs:
            result = results.get(feature_id)
            observed = None if result is None or result.value is None else float(result.value)
            assessments.append(
                _condition_assessment(
                    feature_definition_id=feature_id,
                    comparison=comparison,
                    threshold=threshold,
                    unit=unit,
                    observed=observed,
                    observed_unit=(result.unit if result is not None else None),
                    calibration=(calibrations.get(feature_id) if calibration_complete else None),
                )
            )

        if calibration_complete:
            if any(item.quantification_mode is SoftQuantificationMode.UNAVAILABLE for item in assessments):
                mode = SoftQuantificationMode.UNAVAILABLE
                degree = lower = upper = None
            elif all(
                item.quantification_mode is SoftQuantificationMode.LOGISTIC_INTERVAL
                for item in assessments
            ):
                mode = SoftQuantificationMode.LOGISTIC_INTERVAL
                degree = min(float(item.truth_degree) for item in assessments)
                lower = min(float(item.truth_lower) for item in assessments)
                upper = min(float(item.truth_upper) for item in assessments)
            else:
                # Equality/inequality mixtures cannot silently claim calibrated semantics.
                mode = SoftQuantificationMode.BINARY_COMPATIBILITY
                degree = _binary_degree(fact.status)
                lower = upper = degree
        else:
            degree = _binary_degree(fact.status)
            mode = (
                SoftQuantificationMode.UNAVAILABLE
                if degree is None
                else SoftQuantificationMode.BINARY_COMPATIBILITY
            )
            lower = upper = degree

        fact_payload = {
            "evidence_fact_id": fact.evidence_fact_id,
            "source_evidence_sha256": source_evidence_sha256,
            "policy_sha256": policy_sha256,
            "degree": degree,
            "lower": lower,
            "upper": upper,
            "mode": mode.value,
        }
        soft_facts.append(
            SoftEvidenceFact(
                soft_evidence_fact_id=f"SEF-{_canonical_sha256(fact_payload)[:20].upper()}",
                evidence_fact_id=fact.evidence_fact_id,
                evidence_definition_id=fact.evidence_definition_id,
                target_fault_id=fact.target_fault_id,
                role=fact.role.value,
                hard_status=fact.status,
                predicate_truth_degree=degree,
                predicate_truth_lower=lower,
                predicate_truth_upper=upper,
                quantification_mode=mode,
                calibration_policy_id=(
                    policy.policy_id
                    if mode is SoftQuantificationMode.LOGISTIC_INTERVAL and policy
                    else None
                ),
                calibration_policy_sha256=(
                    policy_sha256
                    if mode is SoftQuantificationMode.LOGISTIC_INTERVAL
                    else None
                ),
                conditions=assessments,
                uncertainty_quantified=(
                    mode is SoftQuantificationMode.LOGISTIC_INTERVAL
                    and all(
                        (
                            float(item.observation_uncertainty or 0.0)
                            + float(item.reference_uncertainty or 0.0)
                            + float(item.threshold_uncertainty or 0.0)
                        )
                        > 0.0
                        for item in assessments
                    )
                ),
            )
        )

    provisional = SoftEvidenceBundle(
        episode_id=bundle.episode.episode_id,
        source_evidence_sha256=source_evidence_sha256,
        facts=soft_facts,
        calibration_policy_id=(policy.policy_id if policy is not None else None),
        calibration_policy_sha256=policy_sha256,
        calibrated_fact_count=sum(
            item.quantification_mode is SoftQuantificationMode.LOGISTIC_INTERVAL
            for item in soft_facts
        ),
        unavailable_fact_count=sum(
            item.quantification_mode is SoftQuantificationMode.UNAVAILABLE
            for item in soft_facts
        ),
    )
    return provisional.model_copy(update={"bundle_sha256": provisional.compute_sha256()})


def load_frozen_soft_evidence_policy(
    path: str | Path,
    *,
    expected_sha256: str,
) -> SoftEvidenceCalibrationPolicy:
    """Load a content-addressed runtime policy and reject non-frozen candidates."""

    policy_path = Path(path).resolve()
    actual_sha256 = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("soft evidence calibration policy asset SHA-256 mismatch")
    policy = SoftEvidenceCalibrationPolicy.model_validate_json(
        policy_path.read_text(encoding="utf-8-sig")
    )
    if policy.status is not SoftEvidencePolicyStatus.FROZEN:
        raise ValueError("runtime soft evidence calibration policy must have FROZEN status")
    if policy.policy_sha256 is None:
        raise ValueError("runtime soft evidence calibration policy requires embedded policy_sha256")
    return policy


__all__ = [
    "SoftConditionAssessment",
    "SoftConditionCalibration",
    "SoftEvidenceBundle",
    "SoftEvidenceCalibrationEntry",
    "SoftEvidenceCalibrationPolicy",
    "SoftEvidenceFact",
    "SoftEvidencePolicyStatus",
    "SoftQuantificationMode",
    "build_soft_evidence_bundle",
    "load_frozen_soft_evidence_policy",
]
