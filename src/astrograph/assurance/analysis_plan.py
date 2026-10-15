"""Pre-registered group-level analysis plans for Route A experiments.

The original :class:`PrimaryAnalysisPlan` remains available for compatibility.
``PrimaryAnalysisPlanV2`` adds the policy-selection details that must be fixed
before any outcome-bearing development table is read: paired-bootstrap seed,
confidence level, candidate/threshold grid, coverage protection, and explicit
non-inferiority margins for every primary safety endpoint.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class MultiplicityMethod(str, Enum):
    HIERARCHICAL = "HIERARCHICAL"
    HOLM = "HOLM"
    NONE_PRIMARY_ONLY = "NONE_PRIMARY_ONLY"


class PrimaryAnalysisPlan(BaseModel):
    """Compatibility model for the first Route A planning contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: str = "astrograph-primary-analysis-plan.v1"
    plan_id: str = Field(min_length=1)
    primary_endpoint: str = Field(min_length=1)
    primary_contrast: str = Field(min_length=1)
    independent_unit: str = Field(
        pattern=r"^(TRAJECTORY_GROUP|MONTE_CARLO_GROUP|EXPERIMENT_ID|EVENT_CLUSTER)$"
    )
    development_groups: list[str] = Field(min_length=1)
    forbidden_post_hoc_groups: list[str] = Field(default_factory=lambda: ["4", "9"])
    status: str = Field(pattern=r"^(DRAFT|FROZEN)$")
    noninferiority_margin: float | None = Field(default=None, ge=0)
    secondary_endpoints: list[str] = Field(default_factory=list)
    multiplicity_method: MultiplicityMethod
    bootstrap_replicates: int = Field(default=2000, ge=200)
    plan_sha256: str | None = None

    @model_validator(mode="after")
    def valid(self) -> "PrimaryAnalysisPlan":
        if set(self.development_groups) & set(self.forbidden_post_hoc_groups):
            raise ValueError("post-hoc groups cannot enter analysis-plan selection")
        if self.status == "FROZEN" and self.noninferiority_margin is None:
            raise ValueError("frozen analysis plan requires a preregistered noninferiority margin")
        if len(set(self.development_groups)) != len(self.development_groups):
            raise ValueError("duplicate development groups")
        expected = self.compute_sha256()
        if self.plan_sha256 is not None and self.plan_sha256 != expected:
            raise ValueError("plan_sha256 mismatch")
        return self

    def compute_sha256(self) -> str:
        payload = self.model_dump(mode="json", exclude={"plan_sha256"})
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class RouteAEndpoint(str, Enum):
    IDENTIFICATION_ACCURACY = "IDENTIFICATION_ACCURACY"
    COVERAGE = "COVERAGE"
    SELECTIVE_RISK = "SELECTIVE_RISK"
    NLL = "NLL"
    BRIER = "BRIER"
    ECE = "ECE"
    AURC = "AURC"


class RouteANonInferiorityMargins(BaseModel):
    """Maximum permitted candidate-minus-A0 degradation on development groups.

    Accuracy and accepted coverage are higher-is-better, so their fields are
    maximum allowed *drops*.  The remaining endpoints are lower-is-better and
    their fields are maximum allowed *increases*.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_accuracy_drop: float = Field(ge=0.0, le=1.0)
    max_coverage_drop: float = Field(ge=0.0, le=1.0)
    max_selective_risk_increase: float = Field(ge=0.0, le=1.0)
    max_nll_increase: float = Field(ge=0.0)
    max_brier_increase: float = Field(ge=0.0)
    max_ece_increase: float = Field(ge=0.0, le=1.0)
    max_aurc_increase: float = Field(ge=0.0, le=1.0)
    max_worst_group_accuracy_drop: float = Field(ge=0.0, le=1.0)


class PrimaryAnalysisPlanV2(BaseModel):
    """Frozen development-selection plan for coverage-aware Route A policies.

    This is a development protocol, not a confirmatory analysis plan.  It fixes
    every degree of freedom used to choose a fusion/rejection candidate while
    permanently excluding groups 4/9.  A fresh confirmatory campaign requires a
    separate sealed-campaign receipt after the selected policy is frozen.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-route-a-primary-analysis-plan.v2"
    plan_id: str = Field(min_length=1)
    status: str = Field(pattern=r"^(DRAFT|FROZEN)$")
    primary_endpoint: RouteAEndpoint = RouteAEndpoint.SELECTIVE_RISK
    primary_contrast: str = "SELECTED_COVERAGE_AWARE_POLICY_VS_A0"
    independent_unit: str = Field(
        pattern=r"^(TRAJECTORY_GROUP|MONTE_CARLO_GROUP|EXPERIMENT_ID|EVENT_CLUSTER)$"
    )
    development_groups: tuple[str, ...]
    forbidden_post_hoc_groups: tuple[str, ...] = ("4", "9")
    candidate_ids: tuple[str, ...]
    baseline_candidate_id: str = "A0"
    candidate_thresholds: dict[str, float]
    candidate_fusion_policy_sha256s: dict[str, str]
    candidate_evidence_package_sha256s: dict[str, str]
    source_tree_sha256: str
    model_package_sha256: str
    temporal_policy_sha256: str
    development_split_sha256: str
    prediction_generation_code_sha256: str
    noninferiority_margins: RouteANonInferiorityMargins | None = None
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    bootstrap_replicates: int = Field(default=5000, ge=1000)
    bootstrap_seed_sha256: str
    multiplicity_method: MultiplicityMethod = MultiplicityMethod.HIERARCHICAL
    secondary_endpoints: tuple[RouteAEndpoint, ...] = (
        RouteAEndpoint.IDENTIFICATION_ACCURACY,
        RouteAEndpoint.COVERAGE,
        RouteAEndpoint.NLL,
        RouteAEndpoint.BRIER,
        RouteAEndpoint.ECE,
        RouteAEndpoint.AURC,
    )
    small_group_warning_threshold: int = Field(default=5, ge=2)
    plan_sha256: str | None = None
    claim_boundary: str = (
        "Development-only policy and rejection-threshold selection. Groups 4/9 are permanently "
        "post-hoc and cannot supply margins, thresholds, weights, evidence applicability, or any "
        "other policy value. A frozen plan does not create confirmatory evidence."
    )

    @field_validator("development_groups", "forbidden_post_hoc_groups", "candidate_ids")
    @classmethod
    def unique_nonempty_values(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(value) != len(set(value)) or any(not item.strip() for item in value):
            raise ValueError("values must be unique and nonempty")
        return value

    @model_validator(mode="after")
    def valid_plan(self) -> "PrimaryAnalysisPlanV2":
        development = set(self.development_groups)
        forbidden = set(self.forbidden_post_hoc_groups)
        if development & forbidden:
            raise ValueError("post-hoc groups cannot enter Route A development selection")
        candidate_set = set(self.candidate_ids)
        if self.baseline_candidate_id not in candidate_set:
            raise ValueError("baseline_candidate_id must be present in candidate_ids")
        for mapping_name in (
            "candidate_thresholds",
            "candidate_fusion_policy_sha256s",
            "candidate_evidence_package_sha256s",
        ):
            mapping = getattr(self, mapping_name)
            if set(mapping) != candidate_set:
                raise ValueError(f"{mapping_name} keys must equal candidate_ids")
        for candidate_id, threshold in self.candidate_thresholds.items():
            if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                raise ValueError(f"candidate threshold for {candidate_id} must be in [0, 1]")
        for mapping_name in (
            "candidate_fusion_policy_sha256s",
            "candidate_evidence_package_sha256s",
        ):
            for candidate_id, digest in getattr(self, mapping_name).items():
                if not _SHA256.fullmatch(digest):
                    raise ValueError(f"{mapping_name}[{candidate_id}] must be lowercase SHA-256")
        for field_name in (
            "source_tree_sha256",
            "model_package_sha256",
            "temporal_policy_sha256",
            "development_split_sha256",
            "prediction_generation_code_sha256",
            "bootstrap_seed_sha256",
        ):
            if not _SHA256.fullmatch(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be lowercase SHA-256")
        if self.status == "FROZEN" and self.noninferiority_margins is None:
            raise ValueError("frozen Route A plan requires all non-inferiority margins")
        expected = self.compute_sha256()
        if self.plan_sha256 is not None and self.plan_sha256 != expected:
            raise ValueError("plan_sha256 mismatch")
        return self

    def compute_sha256(self) -> str:
        payload = self.model_dump(mode="json", exclude={"plan_sha256"})
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def paired_power_simulation(
    *,
    group_count: int,
    effect: float,
    sd: float,
    alpha: float = 0.05,
    simulations: int = 5000,
    seed: int = 1,
) -> float:
    if group_count < 2 or sd <= 0 or simulations < 100:
        raise ValueError("invalid power simulation inputs")
    rng = random.Random(seed)
    hits = 0
    # Normal approximation to a two-sided paired test; used for planning only.
    zcrit = 1.959963984540054
    for _ in range(simulations):
        sample_mean = sum(rng.gauss(effect, sd) for _ in range(group_count)) / group_count
        z = abs(sample_mean / (sd / math.sqrt(group_count)))
        hits += z >= zcrit
    return hits / simulations


__all__ = [
    "MultiplicityMethod",
    "PrimaryAnalysisPlan",
    "PrimaryAnalysisPlanV2",
    "RouteAEndpoint",
    "RouteANonInferiorityMargins",
    "paired_power_simulation",
]
