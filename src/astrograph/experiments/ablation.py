from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from astrograph.assurance.artifact_paths import ArtifactRoot
from astrograph.assurance.feature_admission import canonical_sha256
from astrograph.fusion import BiRRFConfig, BiRRFResult, FaultRanking, RankingSource, fuse_rankings


class AblationVariant(str, Enum):
    A0_DATA_ONLY = "A0_DATA_ONLY"
    A1_FORWARD_ONLY = "A1_FORWARD_ONLY"
    A2_BACKWARD_ONLY = "A2_BACKWARD_ONLY"
    A3_DATA_FORWARD = "A3_DATA_FORWARD"
    A4_DATA_BACKWARD = "A4_DATA_BACKWARD"
    A5_FORWARD_BACKWARD = "A5_FORWARD_BACKWARD"
    A6_ALL_UNADJUSTED = "A6_ALL_UNADJUSTED"
    A7_ALL_LINEAGE_ADJUSTED = "A7_ALL_LINEAGE_ADJUSTED"


_VARIANT_BRANCHES: dict[AblationVariant, tuple[bool, bool, bool, bool]] = {
    AblationVariant.A0_DATA_ONLY: (True, False, False, False),
    AblationVariant.A1_FORWARD_ONLY: (False, True, False, False),
    AblationVariant.A2_BACKWARD_ONLY: (False, False, True, False),
    AblationVariant.A3_DATA_FORWARD: (True, True, False, False),
    AblationVariant.A4_DATA_BACKWARD: (True, False, True, False),
    AblationVariant.A5_FORWARD_BACKWARD: (False, True, True, False),
    AblationVariant.A6_ALL_UNADJUSTED: (True, True, True, False),
    AblationVariant.A7_ALL_LINEAGE_ADJUSTED: (True, True, True, True),
}


class FrozenAblationPlan(BaseModel):
    """Identity of one comparable A0–A7 experiment family.

    All fields that can change data membership, calibration, candidate universe,
    policy, or branch implementation are content-addressed.  The runner refuses to
    compare outputs from plans with different identities.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-frozen-ablation-plan.v1"
    plan_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_id: str = Field(min_length=1)
    split_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_id: str = Field(min_length=1)
    calibration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fault_universe: tuple[str, ...] = Field(min_length=1)
    policy_id: str = Field(min_length=1)
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    code_version: str = Field(min_length=1)
    data_branch_id: str = Field(min_length=1)
    forward_branch_id: str = Field(min_length=1)
    backward_branch_id: str = Field(min_length=1)
    base_config: BiRRFConfig = Field(default_factory=BiRRFConfig)
    metadata: dict[str, Any] = Field(default_factory=dict)
    claim_boundary: str = (
        "Frozen execution identity only. The plan does not claim statistical significance, "
        "external validity, calibration quality, or flight qualification."
    )

    @field_validator("fault_universe")
    @classmethod
    def unique_faults(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("ablation fault universe must be unique")
        if any(not item for item in value):
            raise ValueError("ablation fault IDs must be nonempty")
        return value

    @model_validator(mode="after")
    def all_branch_weights_available(self) -> "FrozenAblationPlan":
        if min(
            self.base_config.data_weight,
            self.base_config.forward_weight,
            self.base_config.backward_weight,
        ) <= 0.0:
            raise ValueError("A0–A7 plan requires positive configured weight for every branch")
        return self

    @property
    def plan_sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json"))


class AblationBranchOutputs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-ablation-branch-outputs.v1"
    episode_id: str = Field(min_length=1)
    data_ranking: FaultRanking
    forward_ranking: FaultRanking
    backward_ranking: FaultRanking
    ranking_sha256_by_source: dict[str, str]


class AblationVariantReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-ablation-variant-receipt.v1"
    plan_id: str
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    variant: AblationVariant
    branch_ranking_sha256: dict[str, str]
    enabled_sources: tuple[str, ...]
    effective_ranking_sha256: dict[str, str]
    effective_config: BiRRFConfig
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    top_fault_id: str | None = None
    dependency_adjusted_fault_ids: tuple[str, ...] = ()


class AblationRunBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-ablation-run-bundle.v1"
    plan: FrozenAblationPlan
    branch_outputs: AblationBranchOutputs
    results: dict[str, BiRRFResult]
    receipts: dict[str, AblationVariantReceipt]
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


RankingFactory = Callable[[], FaultRanking]


def _effective_config(plan: FrozenAblationPlan, variant: AblationVariant) -> BiRRFConfig:
    data_on, forward_on, backward_on, lineage_on = _VARIANT_BRANCHES[variant]
    payload = plan.base_config.model_dump(mode="python")
    payload.update(
        {
            # A0-A7 receipts explicitly migrate to the v2 coverage contract. This
            # avoids Pydantic model_copy bypass and forbids silent v1 reinterpretation.
            "schema_version": "astrograph-birrf-config.v2",
            "data_weight": plan.base_config.data_weight if data_on else 0.0,
            "forward_weight": plan.base_config.forward_weight if forward_on else 0.0,
            "backward_weight": plan.base_config.backward_weight if backward_on else 0.0,
            "collapse_identical_kg_rankings": lineage_on,
            "collapse_highly_correlated_kg_rankings": lineage_on,
            "collapse_shared_correlation_groups": lineage_on,
            "enforce_kg_fault_universe_coverage": lineage_on,
            "expected_fault_universe": plan.fault_universe if lineage_on else (),
            "minimum_kg_fault_universe_coverage": 1.0,
        }
    )
    return BiRRFConfig.model_validate(payload)


def _validate_branch(
    ranking: FaultRanking,
    *,
    expected_source: RankingSource,
    fault_universe: set[str],
) -> None:
    if ranking.source != expected_source:
        raise ValueError(
            f"ablation branch source mismatch: expected {expected_source.value}, got {ranking.source.value}"
        )
    unknown = {item.fault_id for item in ranking.candidates} - fault_universe
    if unknown:
        raise ValueError(f"ablation ranking contains faults outside frozen universe: {sorted(unknown)}")


def _neutralize_disabled_branch(ranking: FaultRanking, *, variant: AblationVariant) -> FaultRanking:
    """Return an empty ranking so a disabled branch cannot affect diagnostics.

    Setting a BiRRF weight to zero is insufficient because diagnostics inspect
    source rankings independently of contribution weights.  A0–A7 therefore use
    an explicit empty ranking for disabled branches while retaining the original
    content hash in the experiment receipt.
    """

    return ranking.model_copy(
        update={
            "ranking_id": (
                f"{ranking.ranking_id}:disabled:{variant.value.lower()}"
            ),
            "candidates": [],
            "metadata": {
                **ranking.metadata,
                "ablation_disabled": True,
                "ablation_variant": variant.value,
                "original_ranking_id": ranking.ranking_id,
            },
        }
    )


class FrozenAblationRunner:
    """Evaluate each branch once and reuse exact outputs across A0–A7."""

    def __init__(self, plan: FrozenAblationPlan, *, artifact_root: ArtifactRoot | None = None):
        self.plan = plan
        self.artifact_root = artifact_root

    def run(
        self,
        *,
        data_factory: RankingFactory,
        forward_factory: RankingFactory,
        backward_factory: RankingFactory,
        normal_probability: float | None = None,
        normal_rank: int | None = None,
    ) -> AblationRunBundle:
        # Deliberately evaluate each potentially expensive branch exactly once.
        data = data_factory()
        forward = forward_factory()
        backward = backward_factory()
        fault_universe = set(self.plan.fault_universe)
        _validate_branch(data, expected_source=RankingSource.DATA_DRIVEN, fault_universe=fault_universe)
        _validate_branch(forward, expected_source=RankingSource.KG_FORWARD, fault_universe=fault_universe)
        _validate_branch(backward, expected_source=RankingSource.KG_BACKWARD, fault_universe=fault_universe)
        episode_ids = {data.episode_id, forward.episode_id, backward.episode_id}
        if len(episode_ids) != 1:
            raise ValueError("all ablation branches must refer to the same episode")

        ranking_hashes = {
            RankingSource.DATA_DRIVEN.value: canonical_sha256(data.model_dump(mode="json")),
            RankingSource.KG_FORWARD.value: canonical_sha256(forward.model_dump(mode="json")),
            RankingSource.KG_BACKWARD.value: canonical_sha256(backward.model_dump(mode="json")),
        }
        branches = AblationBranchOutputs(
            episode_id=data.episode_id,
            data_ranking=data,
            forward_ranking=forward,
            backward_ranking=backward,
            ranking_sha256_by_source=ranking_hashes,
        )

        results: dict[str, BiRRFResult] = {}
        receipts: dict[str, AblationVariantReceipt] = {}
        for variant in AblationVariant:
            config = _effective_config(self.plan, variant)
            data_on, forward_on, backward_on, _ = _VARIANT_BRANCHES[variant]
            effective_rankings = {
                RankingSource.DATA_DRIVEN.value: (
                    data if data_on else _neutralize_disabled_branch(data, variant=variant)
                ),
                RankingSource.KG_FORWARD.value: (
                    forward
                    if forward_on
                    else _neutralize_disabled_branch(forward, variant=variant)
                ),
                RankingSource.KG_BACKWARD.value: (
                    backward
                    if backward_on
                    else _neutralize_disabled_branch(backward, variant=variant)
                ),
            }
            result = fuse_rankings(
                effective_rankings[RankingSource.DATA_DRIVEN.value],
                effective_rankings[RankingSource.KG_FORWARD.value],
                effective_rankings[RankingSource.KG_BACKWARD.value],
                config=config,
                normal_probability=normal_probability,
                normal_rank=normal_rank,
            )
            result_sha = canonical_sha256(result.model_dump(mode="json"))
            key = variant.value
            results[key] = result
            receipts[key] = AblationVariantReceipt(
                plan_id=self.plan.plan_id,
                plan_sha256=self.plan.plan_sha256,
                variant=variant,
                branch_ranking_sha256=ranking_hashes,
                enabled_sources=tuple(
                    source
                    for source, enabled in (
                        (RankingSource.DATA_DRIVEN.value, data_on),
                        (RankingSource.KG_FORWARD.value, forward_on),
                        (RankingSource.KG_BACKWARD.value, backward_on),
                    )
                    if enabled
                ),
                effective_ranking_sha256={
                    source: canonical_sha256(ranking.model_dump(mode="json"))
                    for source, ranking in effective_rankings.items()
                },
                effective_config=config,
                result_sha256=result_sha,
                top_fault_id=result.top_fault_id,
                dependency_adjusted_fault_ids=tuple(
                    result.metadata.get("evidence_lineage_dependency_adjusted_fault_ids", [])
                ),
            )

        unhashed = {
            "schema_version": "astrograph-ablation-run-bundle.v1",
            "plan": self.plan.model_dump(mode="json"),
            "branch_outputs": branches.model_dump(mode="json"),
            "results": {key: value.model_dump(mode="json") for key, value in results.items()},
            "receipts": {key: value.model_dump(mode="json") for key, value in receipts.items()},
        }
        bundle = AblationRunBundle(
            plan=self.plan,
            branch_outputs=branches,
            results=results,
            receipts=receipts,
            bundle_sha256=canonical_sha256(unhashed),
        )
        if self.artifact_root is not None:
            self.artifact_root.write_json(
                f"ablation/{self.plan.plan_id}/{branches.episode_id}.json",
                bundle.model_dump(mode="json"),
            )
        return bundle


__all__ = [
    "AblationBranchOutputs",
    "AblationRunBundle",
    "AblationVariant",
    "AblationVariantReceipt",
    "FrozenAblationPlan",
    "FrozenAblationRunner",
]
