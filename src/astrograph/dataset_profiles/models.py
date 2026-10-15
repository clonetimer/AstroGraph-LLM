from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator



REQUIRED_PROFILE_ASSET_ROLES = frozenset(
    {
        "runtime_evidence",
        "candidate_universe",
        "fault_class_map",
        "diagnosis_hierarchy",
        "fault_ontology_bindings",
        "generic_fault_ontology",
        "component_archetypes",
        "semantic_schema",
        "protocol_contract",
    }
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DatasetProfileAsset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DatasetSourceLayout(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_adapter_id: Literal["simulation.completed_run.v1"] = "simulation.completed_run.v1"
    telemetry_filename: str = "telemetry.csv"
    nominal_required: bool = True
    time_axis_field: str = "time_s"


class DatasetModelBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    selection_mode: Literal["caller_prediction", "registered_provider"] = "caller_prediction"
    model_id: str | None = None
    model_version: str | None = None

    @model_validator(mode="after")
    def require_registered_model_identity(self) -> "DatasetModelBinding":
        if self.selection_mode == "registered_provider" and not self.model_id:
            raise ValueError("registered_provider model binding requires model_id")
        return self


class DatasetReferenceScope(BaseModel):
    """Reviewed identity boundary for calibration/reference reuse."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["ASSET_LOCAL", "FLEET_GROUP"]
    identity_basis: Literal["SYNTHETIC_DATASET_DOMAIN", "PHYSICAL_ASSET_REGISTRY"]
    asset_id: str | None = None
    fleet_group_id: str | None = None
    member_asset_ids: tuple[str, ...] = ()
    commissioning_receipt_asset_role: str | None = None
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_scope(self) -> "DatasetReferenceScope":
        if len(self.member_asset_ids) != len(set(self.member_asset_ids)):
            raise ValueError("reference scope member_asset_ids must be unique")
        if self.kind == "ASSET_LOCAL":
            if not self.asset_id:
                raise ValueError("ASSET_LOCAL reference scope requires asset_id")
            if self.fleet_group_id or self.member_asset_ids or self.commissioning_receipt_asset_role:
                raise ValueError("ASSET_LOCAL reference scope forbids fleet fields")
        else:
            if self.asset_id:
                raise ValueError("FLEET_GROUP reference scope forbids asset_id")
            if not self.fleet_group_id or len(self.member_asset_ids) < 2:
                raise ValueError(
                    "FLEET_GROUP reference scope requires fleet_group_id and at least two members"
                )
            if not self.commissioning_receipt_asset_role:
                raise ValueError(
                    "FLEET_GROUP reference scope requires a commissioning receipt asset role"
                )
        return self

    def to_dor_manifest(self) -> dict[str, object]:
        if self.kind == "ASSET_LOCAL":
            return {"kind": self.kind, "asset_id": self.asset_id}
        return {
            "kind": self.kind,
            "fleet_group_id": self.fleet_group_id,
            "member_asset_ids": list(self.member_asset_ids),
        }


class DatasetProfile(BaseModel):
    """Reviewed binding of one dataset family to AstroGraph runtime assets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-dataset-profile.v3"] = "astrograph-dataset-profile.v3"
    profile_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    source_layout: DatasetSourceLayout = Field(default_factory=DatasetSourceLayout)
    protocol_root_path: str = Field(min_length=1)
    runtime_evidence_builder_id: Literal["configured.package.v1"] = "configured.package.v1"
    assets: dict[str, DatasetProfileAsset]
    reference_scope: DatasetReferenceScope
    model_binding: DatasetModelBinding = Field(default_factory=DatasetModelBinding)
    fusion_policy_profile_id: str = Field(default="birrf.engineering-default.v1", min_length=1)
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_profile_assets(self) -> "DatasetProfile":
        missing = sorted(REQUIRED_PROFILE_ASSET_ROLES - set(self.assets))
        if missing:
            raise ValueError(f"dataset profile is missing required asset roles: {missing}")
        receipt_role = self.reference_scope.commissioning_receipt_asset_role
        if receipt_role and receipt_role not in self.assets:
            raise ValueError(
                "reference scope commissioning receipt asset role is not bound by the profile"
            )
        if self.runtime_evidence_builder_id == "configured.package.v1" and (
            self.source_layout.telemetry_filename != "telemetry.csv"
            or not self.source_layout.nominal_required
            or self.source_layout.time_axis_field != "time_s"
        ):
            raise ValueError(
                "configured.package.v1 currently supports telemetry.csv, a paired nominal, "
                "and time_s only"
            )
        return self


class DatasetProfileReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DatasetProfileRegistryIndex(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-dataset-profile-registry.v1"] = (
        "astrograph-dataset-profile-registry.v1"
    )
    registry_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    asset_root: str = Field(min_length=1)
    profiles: list[DatasetProfileReference] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_profile_ids(self) -> "DatasetProfileRegistryIndex":
        ids = [item.profile_id for item in self.profiles]
        if len(ids) != len(set(ids)):
            raise ValueError("dataset profile registry contains duplicate profile IDs")
        return self


class DatasetProfileReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-dataset-profile-receipt.v3"] = (
        "astrograph-dataset-profile-receipt.v3"
    )
    registry_id: str
    registry_version: str
    profile_id: str
    profile_version: str
    dataset_id: str
    reference_scope: DatasetReferenceScope
    fusion_policy_profile_id: str
    profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verified_asset_sha256: dict[str, str]
    claim_boundary: str


class ResolvedDatasetProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    profile: DatasetProfile
    protocol_root: Path
    asset_paths: dict[str, Path]
    receipt: DatasetProfileReceipt


__all__ = [
    "DatasetModelBinding",
    "DatasetProfile",
    "DatasetProfileAsset",
    "DatasetProfileReceipt",
    "DatasetProfileReference",
    "DatasetReferenceScope",
    "DatasetProfileRegistryIndex",
    "DatasetSourceLayout",
    "REQUIRED_PROFILE_ASSET_ROLES",
    "ResolvedDatasetProfile",
    "sha256_file",
]
