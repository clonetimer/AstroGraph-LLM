"""Development-only selection for quantitative semantic evidence calibration.

This module deliberately stops at ``DEVELOPMENT_SELECTED``.  It scores a
predeclared transition-width grid on independent development groups, emits a
content-addressed receipt, and records why the result is not runtime eligible.
It never reads or promotes test/forbidden groups.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .executable import ComparisonOperator, EpisodeEvidenceBundle
from .soft_semantic import (
    SoftConditionCalibration,
    SoftEvidenceCalibrationEntry,
    SoftEvidenceCalibrationPolicy,
    SoftEvidencePolicyStatus,
    build_soft_evidence_bundle,
)


def _canonical_sha256(payload: Any) -> str:
    raw = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class SoftCalibrationEpisode(BaseModel):
    """One label-separated offline episode; truth must never enter runtime."""

    model_config = ConfigDict(extra="forbid")

    group_id: int
    true_fault_id: str = Field(min_length=1)
    bundle: EpisodeEvidenceBundle


class SoftCalibrationSelectionProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-soft-calibration-selection-protocol.v1"] = (
        "astrograph-soft-calibration-selection-protocol.v1"
    )
    independent_group_key: Literal["sample_index"] = "sample_index"
    development_group_ids: tuple[int, ...] = (3, 8)
    forbidden_group_ids: tuple[int, ...] = (4, 9)
    transition_width_multipliers: tuple[float, ...] = (0.02, 0.05, 0.1, 0.2, 0.4)
    brier_tie_tolerance: float = Field(default=1e-6, ge=0.0)
    selection_rule: Literal[
        "minimum_group_macro_brier_then_largest_width_within_tolerance"
    ] = "minimum_group_macro_brier_then_largest_width_within_tolerance"
    runtime_activation_allowed: Literal[False] = False

    @model_validator(mode="after")
    def validate_groups_and_grid(self) -> "SoftCalibrationSelectionProtocol":
        if not self.development_group_ids:
            raise ValueError("soft calibration selection requires development groups")
        if len(set(self.development_group_ids)) != len(self.development_group_ids):
            raise ValueError("development groups must be unique")
        if set(self.development_group_ids) & set(self.forbidden_group_ids):
            raise ValueError("development and forbidden groups must be disjoint")
        if not self.transition_width_multipliers or any(
            value <= 0.0 for value in self.transition_width_multipliers
        ):
            raise ValueError("transition-width multipliers must be positive")
        if len(set(self.transition_width_multipliers)) != len(
            self.transition_width_multipliers
        ):
            raise ValueError("transition-width multipliers must be unique")
        return self


class SoftCalibrationCandidateMetric(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    transition_width_multiplier: float = Field(gt=0.0)
    candidate_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observation_count: int = Field(gt=0)
    group_macro_brier: float = Field(ge=0.0, le=1.0)
    worst_group_brier: float = Field(ge=0.0, le=1.0)
    group_brier: dict[str, float]


class SoftCalibrationSelectionReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-soft-calibration-selection-receipt.v1"] = (
        "astrograph-soft-calibration-selection-receipt.v1"
    )
    receipt_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    runtime_evidence_package_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_hashes: dict[str, str]
    protocol: SoftCalibrationSelectionProtocol
    selected_evidence_definition_ids: list[str]
    group_episode_counts: dict[str, int]
    observation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_metrics: list[SoftCalibrationCandidateMetric] = Field(min_length=1)
    selected_candidate_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: Literal["DEVELOPMENT_SELECTED_NOT_FROZEN"] = (
        "DEVELOPMENT_SELECTED_NOT_FROZEN"
    )
    freeze_eligible: Literal[False] = False
    freeze_blockers: list[str] = Field(min_length=1)
    receipt_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    claim_boundary: str = (
        "Development-group predicate calibration only. It is not a fault posterior, "
        "does not quantify sensor/threshold uncertainty, and is not runtime eligible."
    )

    @model_validator(mode="after")
    def validate_receipt(self) -> "SoftCalibrationSelectionReceipt":
        if any(set(key) - set("0123456789abcdef") for key in self.source_hashes.values()):
            raise ValueError("source_hashes must contain lowercase SHA-256 values")
        if any(len(value) != 64 for value in self.source_hashes.values()):
            raise ValueError("source_hashes must contain SHA-256 values")
        if self.receipt_sha256 is not None and self.receipt_sha256 != self.compute_sha256():
            raise ValueError("soft calibration selection receipt SHA-256 mismatch")
        return self

    def canonical_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"receipt_sha256"})

    def compute_sha256(self) -> str:
        return _canonical_sha256(self.canonical_payload())


def _candidate_policy(
    *,
    policy_id: str,
    version: str,
    definitions: Sequence[Any],
    multiplier: float,
    forbidden_groups: Sequence[int],
) -> SoftEvidenceCalibrationPolicy:
    entries: list[SoftEvidenceCalibrationEntry] = []
    for definition in definitions:
        specs = [
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
        conditions: list[SoftConditionCalibration] = []
        for feature_id, comparison, threshold, unit in specs:
            if comparison not in {
                ComparisonOperator.GREATER_THAN,
                ComparisonOperator.GREATER_OR_EQUAL,
                ComparisonOperator.LESS_THAN,
                ComparisonOperator.LESS_OR_EQUAL,
            }:
                raise ValueError(
                    f"soft calibration selection does not support {comparison}: {feature_id}"
                )
            if threshold == 0.0:
                raise ValueError(
                    f"zero threshold requires a separately preregistered absolute grid: {feature_id}"
                )
            conditions.append(
                SoftConditionCalibration(
                    feature_definition_id=feature_id,
                    unit=unit,
                    transition_width=abs(threshold) * multiplier,
                )
            )
        entries.append(
            SoftEvidenceCalibrationEntry(
                evidence_definition_id=definition.evidence_definition_id,
                conditions=conditions,
            )
        )
    return SoftEvidenceCalibrationPolicy(
        policy_id=policy_id,
        version=version,
        entries=entries,
        forbidden_groups=list(forbidden_groups),
    )


def select_development_soft_calibration(
    episodes: Sequence[SoftCalibrationEpisode],
    *,
    dataset_id: str,
    runtime_evidence_package_sha256: str,
    source_hashes: dict[str, str],
    selected_evidence_definition_ids: Sequence[str],
    policy_id: str,
    policy_version: str,
    receipt_id: str,
    protocol: SoftCalibrationSelectionProtocol | None = None,
) -> tuple[SoftEvidenceCalibrationPolicy, SoftCalibrationSelectionReceipt]:
    """Select a development candidate without granting runtime eligibility."""

    protocol = protocol or SoftCalibrationSelectionProtocol()
    if not episodes:
        raise ValueError("soft calibration selection requires episodes")
    groups = {item.group_id for item in episodes}
    unexpected = groups - set(protocol.development_group_ids)
    if unexpected:
        raise ValueError(f"episodes contain non-development groups: {sorted(unexpected)}")
    missing = set(protocol.development_group_ids) - groups
    if missing:
        raise ValueError(f"episodes are missing development groups: {sorted(missing)}")
    if groups & set(protocol.forbidden_group_ids):
        raise ValueError("forbidden groups cannot enter soft calibration selection")

    first_definitions = {
        item.evidence_definition_id: item for item in episodes[0].bundle.evidence_definitions
    }
    selected_ids = list(dict.fromkeys(selected_evidence_definition_ids))
    if len(selected_ids) != len(selected_evidence_definition_ids):
        raise ValueError("selected evidence definition IDs must be unique")
    try:
        definitions = [first_definitions[item] for item in selected_ids]
    except KeyError as exc:
        raise ValueError(f"unknown selected evidence definition: {exc.args[0]}") from exc
    definition_signature = _canonical_sha256(
        [item.model_dump(mode="json") for item in definitions]
    )
    selected_feature_ids = {
        definition.feature_definition_id
        for definition in definitions
    } | {
        condition.feature_definition_id
        for definition in definitions
        for condition in definition.and_feature_conditions
    }
    observations: list[dict[str, Any]] = []
    for episode in episodes:
        current = {
            item.evidence_definition_id: item
            for item in episode.bundle.evidence_definitions
            if item.evidence_definition_id in selected_ids
        }
        if _canonical_sha256(
            [current[item].model_dump(mode="json") for item in selected_ids]
        ) != definition_signature:
            raise ValueError("evidence definitions differ across calibration episodes")
        values = {
            item.feature_definition_id: item.value
            for item in episode.bundle.feature_results
            if item.feature_definition_id in selected_feature_ids
        }
        observations.append(
            {
                "group_id": episode.group_id,
                "true_fault_id": episode.true_fault_id,
                "values": {key: values.get(key) for key in sorted(values)},
            }
        )
    observations.sort(
        key=lambda item: (
            item["group_id"],
            item["true_fault_id"],
            json.dumps(item["values"], sort_keys=True, separators=(",", ":")),
        )
    )
    observation_sha256 = _canonical_sha256(observations)

    candidates: list[tuple[SoftEvidenceCalibrationPolicy, SoftCalibrationCandidateMetric]] = []
    for multiplier in protocol.transition_width_multipliers:
        candidate = _candidate_policy(
            policy_id=policy_id,
            version=policy_version,
            definitions=definitions,
            multiplier=multiplier,
            forbidden_groups=protocol.forbidden_group_ids,
        )
        squared_by_group: dict[int, list[float]] = {
            item: [] for item in protocol.development_group_ids
        }
        for episode in episodes:
            facts = {
                item.evidence_definition_id: item
                for item in build_soft_evidence_bundle(episode.bundle, candidate).facts
            }
            for definition in definitions:
                degree = facts[definition.evidence_definition_id].predicate_truth_degree
                if degree is None:
                    raise ValueError("calibration episode contains unavailable selected evidence")
                truth = float(definition.target_fault_id == episode.true_fault_id)
                squared_by_group[episode.group_id].append((float(degree) - truth) ** 2)
        group_brier = {
            str(group): sum(values) / len(values)
            for group, values in sorted(squared_by_group.items())
            if values
        }
        macro = sum(group_brier.values()) / len(group_brier)
        metric = SoftCalibrationCandidateMetric(
            transition_width_multiplier=multiplier,
            candidate_policy_sha256=candidate.compute_sha256(),
            observation_count=sum(len(item) for item in squared_by_group.values()),
            group_macro_brier=macro,
            worst_group_brier=max(group_brier.values()),
            group_brier=group_brier,
        )
        candidates.append((candidate, metric))

    minimum = min(item.group_macro_brier for _, item in candidates)
    eligible = [
        item
        for item in candidates
        if item[1].group_macro_brier <= minimum + protocol.brier_tie_tolerance
    ]
    selected_candidate, selected_metric = max(
        eligible, key=lambda item: item[1].transition_width_multiplier
    )
    counts = {
        str(group): sum(item.group_id == group for item in episodes)
        for group in protocol.development_group_ids
    }
    provisional_receipt = SoftCalibrationSelectionReceipt(
        receipt_id=receipt_id,
        dataset_id=dataset_id,
        runtime_evidence_package_sha256=runtime_evidence_package_sha256,
        source_hashes=source_hashes,
        protocol=protocol,
        selected_evidence_definition_ids=selected_ids,
        group_episode_counts=counts,
        observation_sha256=observation_sha256,
        candidate_metrics=[item for _, item in candidates],
        selected_candidate_policy_sha256=selected_metric.candidate_policy_sha256,
        freeze_blockers=[
            "NO_INDEPENDENT_CALIBRATION_ROLE_IN_FROZEN_THREE_WAY_SPLIT",
            "OBSERVATION_REFERENCE_THRESHOLD_UNCERTAINTIES_NOT_IDENTIFIED",
            "NO_RUNTIME_ACTIVATION_REVIEW",
        ],
    )
    receipt = provisional_receipt.model_copy(
        update={"receipt_sha256": provisional_receipt.compute_sha256()}
    )
    selected = selected_candidate.model_copy(
        update={
            "status": SoftEvidencePolicyStatus.DEVELOPMENT_SELECTED,
            "selected_on_groups": list(protocol.development_group_ids),
            "selection_receipt_sha256": receipt.receipt_sha256,
        }
    )
    selected = selected.model_copy(update={"policy_sha256": selected.compute_sha256()})
    return selected, receipt


__all__ = [
    "SoftCalibrationCandidateMetric",
    "SoftCalibrationEpisode",
    "SoftCalibrationSelectionProtocol",
    "SoftCalibrationSelectionReceipt",
    "select_development_soft_calibration",
]
