"""Conservative, non-voting projection of soft evidence onto fault concepts.

The projection is deliberately a shadow diagnostic.  It preserves intervals,
counter-evidence, coverage, and declared provenance-independence groups instead
of manufacturing a fault posterior.  Runtime admission remains a separate,
future governed decision.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from enum import Enum
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.evidence import SoftEvidenceBundle


def _sha256(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class SoftGraphSuggestedAction(str, Enum):
    NO_OP = "NO_OP"
    ABSTAIN = "ABSTAIN"
    CANDIDATE_SET = "CANDIDATE_SET"
    SINGLE_CANDIDATE = "SINGLE_CANDIDATE"


class SoftGraphIndependenceDeclaration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    group_id: str = Field(min_length=1)
    evidence_definition_ids: list[str] = Field(min_length=1)
    basis_refs: list[str] = Field(min_length=1)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def unique_definitions(self) -> "SoftGraphIndependenceDeclaration":
        if len(self.evidence_definition_ids) != len(set(self.evidence_definition_ids)):
            raise ValueError("soft graph provenance group contains duplicate definitions")
        return self


class SoftGraphProvenancePolicy(BaseModel):
    """Reviewed declaration that different groups are independent evidence sources."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["astrograph-soft-graph-provenance-policy.v1"] = (
        "astrograph-soft-graph-provenance-policy.v1"
    )
    policy_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    status: Literal["FROZEN"] = "FROZEN"
    declarations: list[SoftGraphIndependenceDeclaration] = Field(min_length=2)
    policy_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    claim_boundary: str = (
        "Declares reviewed provenance independence only; it does not establish causal "
        "independence, a fault posterior, or Runtime voting authority."
    )

    @model_validator(mode="after")
    def validate_policy(self) -> "SoftGraphProvenancePolicy":
        group_ids = [item.group_id for item in self.declarations]
        definition_ids = [
            definition_id
            for item in self.declarations
            for definition_id in item.evidence_definition_ids
        ]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("soft graph provenance group IDs must be unique")
        if len(definition_ids) != len(set(definition_ids)):
            raise ValueError("an evidence definition may belong to only one provenance group")
        expected = self.compute_sha256()
        if self.policy_sha256 is not None and self.policy_sha256 != expected:
            raise ValueError("soft graph provenance policy_sha256 mismatch")
        return self

    def compute_sha256(self) -> str:
        return _sha256(self.model_dump(mode="json", exclude={"policy_sha256"}))

    def definition_groups(self) -> dict[str, str]:
        return {
            definition_id: item.group_id
            for item in self.declarations
            for definition_id in item.evidence_definition_ids
        }


def load_soft_graph_provenance_policy(
    path: str | Path, *, expected_sha256: str
) -> SoftGraphProvenancePolicy:
    asset = Path(path)
    actual = hashlib.sha256(asset.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError("soft graph provenance policy asset SHA-256 mismatch")
    policy = SoftGraphProvenancePolicy.model_validate_json(asset.read_text(encoding="utf-8"))
    if policy.policy_sha256 is None:
        raise ValueError("frozen soft graph provenance policy requires embedded policy_sha256")
    return policy


class SoftGraphObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-soft-graph-observation.v1"] = (
        "astrograph-soft-graph-observation.v1"
    )
    observation_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)
    predicate_id: Literal["HAS_SOFT_EVIDENCE"] = "HAS_SOFT_EVIDENCE"
    object_fault_id: str = Field(min_length=1)
    evidence_fact_id: str = Field(min_length=1)
    evidence_definition_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    truth_lower: float | None = Field(default=None, ge=0.0, le=1.0)
    truth_point: float | None = Field(default=None, ge=0.0, le=1.0)
    truth_upper: float | None = Field(default=None, ge=0.0, le=1.0)
    uncertainty_quantified: bool = False
    quantification_mode: str = Field(min_length=1)
    provenance_group_id: str = Field(min_length=1)
    independence_declared: bool = False
    source_evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_policy_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_interval(self) -> "SoftGraphObservation":
        values = (self.truth_lower, self.truth_point, self.truth_upper)
        if any(value is None for value in values):
            if not all(value is None for value in values):
                raise ValueError("soft graph truth interval must be complete or unavailable")
        elif not (self.truth_lower <= self.truth_point <= self.truth_upper):
            raise ValueError("soft graph truth interval must be ordered")
        return self


class SoftGraphFaultAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    fault_id: str = Field(min_length=1)
    support_lower: float = Field(ge=0.0, le=1.0)
    support_point: float = Field(ge=0.0, le=1.0)
    support_upper: float = Field(ge=0.0, le=1.0)
    counter_lower: float = Field(ge=0.0, le=1.0)
    counter_point: float = Field(ge=0.0, le=1.0)
    counter_upper: float = Field(ge=0.0, le=1.0)
    net_support_lower: float = Field(ge=0.0, le=1.0)
    net_support_point: float = Field(ge=0.0, le=1.0)
    net_support_upper: float = Field(ge=0.0, le=1.0)
    evidence_coverage: float = Field(ge=0.0, le=1.0)
    available_observation_count: int = Field(ge=0)
    total_observation_count: int = Field(ge=0)
    calibrated_observation_count: int = Field(ge=0)
    provenance_group_ids: list[str] = Field(default_factory=list)
    declared_independent_group_count: int = Field(ge=0)
    supporting_observation_ids: list[str] = Field(default_factory=list)
    counter_observation_ids: list[str] = Field(default_factory=list)
    conflict: bool = False


class SoftGraphInferenceResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-soft-graph-inference.v1"] = (
        "astrograph-soft-graph-inference.v1"
    )
    inference_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    knowledge_release_id: str = Field(min_length=1)
    provenance_policy_id: str | None = None
    provenance_policy_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_soft_evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observations: list[SoftGraphObservation]
    fault_assessments: list[SoftGraphFaultAssessment]
    suggested_action: SoftGraphSuggestedAction
    candidate_fault_ids: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    minimum_evidence_coverage: float = Field(ge=0.0, le=1.0)
    minimum_declared_independent_groups: int = Field(ge=1)
    maximum_counter_support: float = Field(ge=0.0, le=1.0)
    runtime_effect_applied: Literal[False] = False
    result_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    claim_boundary: str = (
        "Shadow interval reasoning only. Values are predicate support, not fault posteriors; "
        "the result cannot vote in BiRRF, alter Governor disposition, or modify active knowledge."
    )

    @model_validator(mode="after")
    def validate_hash(self) -> "SoftGraphInferenceResult":
        expected = self.compute_sha256()
        if self.result_sha256 is not None and self.result_sha256 != expected:
            raise ValueError("soft graph inference result_sha256 mismatch")
        return self

    def compute_sha256(self) -> str:
        return _sha256(self.model_dump(mode="json", exclude={"result_sha256"}))


def _mean(values: list[float]) -> float:
    return float(fmean(values)) if values else 0.0


def _positive_interval(observations: list[SoftGraphObservation]) -> tuple[float, float, float]:
    required = [item for item in observations if item.role == "REQUIRED"]
    supporting = [item for item in observations if item.role == "SUPPORTING"]
    selected = required or supporting
    if not selected:
        return (0.0, 0.0, 0.0)
    if required:
        return (
            min(float(item.truth_lower) for item in selected),
            min(float(item.truth_point) for item in selected),
            min(float(item.truth_upper) for item in selected),
        )
    return (
        _mean([float(item.truth_lower) for item in selected]),
        _mean([float(item.truth_point) for item in selected]),
        _mean([float(item.truth_upper) for item in selected]),
    )


def build_soft_graph_inference(
    soft_evidence: SoftEvidenceBundle,
    *,
    knowledge_release_id: str | None = None,
    candidate_fault_ids: list[str] | None = None,
    independence_groups_by_definition: dict[str, str] | None = None,
    provenance_policy_id: str | None = None,
    provenance_policy_sha256: str | None = None,
    minimum_evidence_coverage: float = 0.5,
    minimum_declared_independent_groups: int = 2,
    maximum_counter_support: float = 0.5,
) -> SoftGraphInferenceResult:
    """Build a conservative, deterministic shadow inference artifact."""

    declared_groups = independence_groups_by_definition or {}
    observations: list[SoftGraphObservation] = []
    for fact in soft_evidence.facts:
        declared = fact.evidence_definition_id in declared_groups
        group_id = declared_groups.get(
            fact.evidence_definition_id, f"UNDECLARED:{fact.evidence_definition_id}"
        )
        payload = {
            "episode_id": soft_evidence.episode_id,
            "soft_evidence_fact_id": fact.soft_evidence_fact_id,
            "group_id": group_id,
        }
        observations.append(
            SoftGraphObservation(
                observation_id="SGO-" + _sha256(payload)[:20].upper(),
                episode_id=soft_evidence.episode_id,
                subject_id=fact.soft_evidence_fact_id,
                object_fault_id=fact.target_fault_id,
                evidence_fact_id=fact.evidence_fact_id,
                evidence_definition_id=fact.evidence_definition_id,
                role=fact.role,
                truth_lower=fact.predicate_truth_lower,
                truth_point=fact.predicate_truth_degree,
                truth_upper=fact.predicate_truth_upper,
                uncertainty_quantified=fact.uncertainty_quantified,
                quantification_mode=fact.quantification_mode.value,
                provenance_group_id=group_id,
                independence_declared=declared,
                source_evidence_sha256=soft_evidence.source_evidence_sha256,
                calibration_policy_sha256=fact.calibration_policy_sha256,
            )
        )

    by_fault: dict[str, list[SoftGraphObservation]] = defaultdict(list)
    for item in observations:
        by_fault[item.object_fault_id].append(item)
    faults = sorted(set(candidate_fault_ids or []) | set(by_fault))
    assessments: list[SoftGraphFaultAssessment] = []
    for fault_id in faults:
        all_items = by_fault.get(fault_id, [])
        available = [item for item in all_items if item.truth_point is not None]
        calibrated = [
            item for item in available
            if item.uncertainty_quantified and item.calibration_policy_sha256 is not None
        ]
        positive = [item for item in calibrated if item.role in {"REQUIRED", "SUPPORTING"}]
        counter = [item for item in calibrated if item.role in {"COUNTER", "EXCLUSION"}]
        support_lower, support_point, support_upper = _positive_interval(positive)
        counter_lower = max((float(item.truth_lower) for item in counter), default=0.0)
        counter_point = max((float(item.truth_point) for item in counter), default=0.0)
        counter_upper = max((float(item.truth_upper) for item in counter), default=0.0)
        groups = sorted({item.provenance_group_id for item in calibrated})
        declared_count = len({
            item.provenance_group_id for item in calibrated if item.independence_declared
        })
        assessments.append(
            SoftGraphFaultAssessment(
                fault_id=fault_id,
                support_lower=support_lower,
                support_point=support_point,
                support_upper=support_upper,
                counter_lower=counter_lower,
                counter_point=counter_point,
                counter_upper=counter_upper,
                net_support_lower=max(0.0, support_lower - counter_upper),
                net_support_point=max(0.0, support_point - counter_point),
                net_support_upper=max(0.0, support_upper - counter_lower),
                evidence_coverage=(len(calibrated) / len(all_items) if all_items else 0.0),
                available_observation_count=len(available),
                total_observation_count=len(all_items),
                calibrated_observation_count=len(calibrated),
                provenance_group_ids=groups,
                declared_independent_group_count=declared_count,
                supporting_observation_ids=[item.observation_id for item in positive],
                counter_observation_ids=[item.observation_id for item in counter],
                conflict=support_lower > 0.5 and counter_lower > 0.5,
            )
        )
    assessments.sort(key=lambda item: (-item.net_support_point, item.fault_id))

    action = SoftGraphSuggestedAction.NO_OP
    selected: list[str] = []
    reasons: list[str] = []
    if soft_evidence.calibrated_fact_count == 0:
        reasons.append("NO_CALIBRATED_SOFT_EVIDENCE")
    elif not assessments or assessments[0].calibrated_observation_count == 0:
        action = SoftGraphSuggestedAction.ABSTAIN
        reasons.append("NO_AVAILABLE_SOFT_GRAPH_OBSERVATIONS")
    else:
        top = assessments[0]
        if top.evidence_coverage < minimum_evidence_coverage:
            action = SoftGraphSuggestedAction.ABSTAIN
            reasons.append("INSUFFICIENT_SOFT_GRAPH_COVERAGE")
        elif top.counter_upper > maximum_counter_support or top.conflict:
            action = SoftGraphSuggestedAction.ABSTAIN
            reasons.append("SOFT_GRAPH_COUNTER_EVIDENCE_OR_CONFLICT")
        elif top.declared_independent_group_count < minimum_declared_independent_groups:
            action = SoftGraphSuggestedAction.ABSTAIN
            reasons.append("INSUFFICIENT_DECLARED_INDEPENDENT_PROVENANCE")
        else:
            overlapping = [
                item.fault_id
                for item in assessments
                if item.net_support_upper >= top.net_support_lower
            ]
            selected = overlapping
            action = (
                SoftGraphSuggestedAction.SINGLE_CANDIDATE
                if len(overlapping) == 1
                else SoftGraphSuggestedAction.CANDIDATE_SET
            )
            reasons.append("INTERVAL_SEPARATED" if len(overlapping) == 1 else "OVERLAPPING_INTERVALS")

    provisional = SoftGraphInferenceResult(
        inference_id="SGI-" + _sha256({
            "episode_id": soft_evidence.episode_id,
            "soft_sha256": soft_evidence.bundle_sha256,
            "knowledge_release_id": knowledge_release_id or "NO_ACTIVE_RELEASE",
            "groups": declared_groups,
            "provenance_policy_id": provenance_policy_id,
            "provenance_policy_sha256": provenance_policy_sha256,
        })[:20].upper(),
        episode_id=soft_evidence.episode_id,
        knowledge_release_id=knowledge_release_id or "NO_ACTIVE_RELEASE",
        provenance_policy_id=provenance_policy_id,
        provenance_policy_sha256=provenance_policy_sha256,
        source_soft_evidence_sha256=(
            soft_evidence.bundle_sha256 or soft_evidence.compute_sha256()
        ),
        observations=observations,
        fault_assessments=assessments,
        suggested_action=action,
        candidate_fault_ids=selected,
        reason_codes=reasons,
        minimum_evidence_coverage=minimum_evidence_coverage,
        minimum_declared_independent_groups=minimum_declared_independent_groups,
        maximum_counter_support=maximum_counter_support,
    )
    return provisional.model_copy(update={"result_sha256": provisional.compute_sha256()})


__all__ = [
    "SoftGraphFaultAssessment",
    "SoftGraphInferenceResult",
    "SoftGraphObservation",
    "SoftGraphIndependenceDeclaration",
    "SoftGraphProvenancePolicy",
    "SoftGraphSuggestedAction",
    "build_soft_graph_inference",
    "load_soft_graph_provenance_policy",
]
