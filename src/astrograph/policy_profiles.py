from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.fusion import BiRRFConfig
from astrograph.investigation import GraphQueryPlan
from astrograph.knowledge import (
    CompiledGraphQuery,
    GraphQueryKind,
    ReadOnlyCypherCompiler,
    ReadOnlyGraphQueryPolicy,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class GraphQueryPolicyProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-graph-query-policy-profile.v1"] = (
        "astrograph-graph-query-policy-profile.v1"
    )
    profile_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    policy: ReadOnlyGraphQueryPolicy
    allowed_query_kinds: list[GraphQueryKind] = Field(min_length=1)
    claim_boundary: str = Field(min_length=1)


class FusionCalibrationReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1)
    calibration_split_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class FusionPolicyProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-fusion-policy-profile.v1"] = (
        "astrograph-fusion-policy-profile.v1"
    )
    profile_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    qualification_status: Literal["ENGINEERING_DEFAULT", "PRODUCTION_STRICT", "CALIBRATED"]
    config: BiRRFConfig
    calibration_receipt: FusionCalibrationReceipt | None = None
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def calibrated_requires_receipt(self) -> "FusionPolicyProfile":
        if self.qualification_status == "CALIBRATED" and self.calibration_receipt is None:
            raise ValueError("CALIBRATED fusion profile requires a calibration receipt")
        if self.qualification_status in {"ENGINEERING_DEFAULT", "PRODUCTION_STRICT"} and self.calibration_receipt is not None:
            raise ValueError(f"{self.qualification_status} fusion profile cannot carry a calibration receipt")
        if self.qualification_status == "PRODUCTION_STRICT":
            if not self.config.require_complete_evidence_lineage:
                raise ValueError("PRODUCTION_STRICT fusion profile requires complete evidence lineage")
            if not self.config.enforce_kg_fault_universe_coverage:
                raise ValueError("PRODUCTION_STRICT fusion profile requires KG fault-universe coverage enforcement")
            if not self.config.expected_fault_universe:
                raise ValueError("PRODUCTION_STRICT fusion profile requires a frozen expected fault universe")
            if self.config.minimum_kg_fault_universe_coverage != 1.0:
                raise ValueError("PRODUCTION_STRICT fusion profile requires 100% KG fault-universe coverage")
            if self.config.kg_partial_coverage_action.value != "SUPPRESS_BRANCH":
                raise ValueError("PRODUCTION_STRICT fusion profile must suppress partially covered KG branches")
        return self


class RuntimePolicyReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_type: Literal["GRAPH_QUERY", "FUSION"]
    profile_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RuntimePolicyRegistryIndex(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-runtime-policy-registry.v1"] = (
        "astrograph-runtime-policy-registry.v1"
    )
    registry_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    asset_root: str = Field(min_length=1)
    profiles: list[RuntimePolicyReference] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_profiles(self) -> "RuntimePolicyRegistryIndex":
        keys = [(item.profile_type, item.profile_id) for item in self.profiles]
        if len(keys) != len(set(keys)):
            raise ValueError("runtime policy registry contains duplicate profile identities")
        return self


class RuntimePolicyReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-runtime-policy-receipt.v1"
    registry_id: str
    registry_version: str
    profile_type: Literal["GRAPH_QUERY", "FUSION"]
    profile_id: str
    profile_version: str
    profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    qualification_status: str
    claim_boundary: str


class ResolvedRuntimePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: GraphQueryPolicyProfile | FusionPolicyProfile
    receipt: RuntimePolicyReceipt


class RuntimePolicyRegistry:
    def __init__(self, index_path: str | Path) -> None:
        self.index_path = Path(index_path).resolve()
        self.index = RuntimePolicyRegistryIndex.model_validate_json(
            self.index_path.read_text(encoding="utf-8")
        )
        asset_root = Path(self.index.asset_root)
        self.asset_root = (
            asset_root.resolve()
            if asset_root.is_absolute()
            else (self.index_path.parent / asset_root).resolve()
        )

    def _safe_path(self, raw: str) -> Path:
        candidate = (self.asset_root / raw).resolve()
        if not candidate.is_relative_to(self.asset_root):
            raise ValueError("runtime policy path escapes asset root")
        return candidate

    def resolve(
        self,
        *,
        profile_type: Literal["GRAPH_QUERY", "FUSION"],
        profile_id: str,
    ) -> ResolvedRuntimePolicy:
        matches = [
            item
            for item in self.index.profiles
            if item.profile_type == profile_type and item.profile_id == profile_id
        ]
        if len(matches) != 1:
            raise KeyError(f"unknown runtime policy profile: {profile_type}/{profile_id}")
        reference = matches[0]
        path = self._safe_path(reference.path)
        actual_hash = sha256_file(path)
        if actual_hash != reference.sha256:
            raise ValueError("runtime policy profile hash mismatch")
        if profile_type == "GRAPH_QUERY":
            profile: GraphQueryPolicyProfile | FusionPolicyProfile = (
                GraphQueryPolicyProfile.model_validate_json(path.read_text(encoding="utf-8"))
            )
            qualification = "READ_ONLY_POLICY"
        else:
            profile = FusionPolicyProfile.model_validate_json(path.read_text(encoding="utf-8"))
            qualification = profile.qualification_status
        if profile.profile_id != profile_id:
            raise ValueError("runtime policy profile ID does not match registry reference")
        return ResolvedRuntimePolicy(
            profile=profile,
            receipt=RuntimePolicyReceipt(
                registry_id=self.index.registry_id,
                registry_version=self.index.version,
                profile_type=profile_type,
                profile_id=profile.profile_id,
                profile_version=profile.version,
                profile_sha256=actual_hash,
                qualification_status=qualification,
                claim_boundary=profile.claim_boundary,
            ),
        )


class GovernedGraphQueryCompiler:
    """Compile only fixed read-only templates allowed by one resolved policy profile."""

    def __init__(self, resolved: ResolvedRuntimePolicy) -> None:
        if not isinstance(resolved.profile, GraphQueryPolicyProfile):
            raise TypeError("GovernedGraphQueryCompiler requires a GRAPH_QUERY profile")
        self.profile = resolved.profile
        self.receipt = resolved.receipt
        self.compiler = ReadOnlyCypherCompiler(self.profile.policy)

    def compile(
        self,
        plan: GraphQueryPlan,
        *,
        query_kind: GraphQueryKind = GraphQueryKind.AUTO,
        knowledge_version: str | None = None,
    ) -> CompiledGraphQuery:
        compiled = self.compiler.compile(
            plan,
            query_kind=query_kind,
            knowledge_version=knowledge_version,
        )
        if compiled.query_kind not in self.profile.allowed_query_kinds:
            raise ValueError(
                f"query kind is not allowed by runtime policy profile: {compiled.query_kind.value}"
            )
        return compiled


__all__ = [
    "FusionCalibrationReceipt",
    "FusionPolicyProfile",
    "GovernedGraphQueryCompiler",
    "GraphQueryPolicyProfile",
    "ResolvedRuntimePolicy",
    "RuntimePolicyReceipt",
    "RuntimePolicyRegistry",
]
