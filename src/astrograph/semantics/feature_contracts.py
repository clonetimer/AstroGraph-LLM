from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_IDENTITY_OR_TRUTH_TOKENS = (
    "case",
    "class",
    "dataset",
    "effect",
    "event",
    "fault",
    "file",
    "health",
    "identity",
    "label",
    "mode",
    "modifier",
    "nominal",
    "object",
    "onset",
    "path",
    "provenance",
    "run_id",
    "scenario",
    "seed",
    "severity",
    "split",
    "status",
    "target",
    "truth",
    "anomaly",
)


class FeaturePrimitive(BaseModel):
    """Truth-blind operation over declared semantic port roles.

    A primitive names an operation family, not a fitted threshold, coefficient, or
    output dimension.  Exact windows and hyperparameters remain a later frozen
    protocol concern.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    primitive_id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_.-]*$")
    operator_family: str = Field(min_length=1, pattern=r"^[A-Z][A-Z0-9_]*$")
    input_port_role_ids: list[str] = Field(min_length=1)
    output_token_type: str = Field(min_length=1, pattern=r"^[A-Z][A-Z0-9_]*$")
    availability_policy: Literal["REQUIRED", "OPTIONAL_SKIP_WITH_MASK"]
    description: str = ""

    @field_validator("input_port_role_ids")
    @classmethod
    def truth_blind_unique_roles(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("input_port_role_ids must be unique")
        for role in value:
            lowered = role.lower()
            if any(token in lowered for token in _IDENTITY_OR_TRUTH_TOKENS):
                raise ValueError(f"feature primitive input role is prohibited: {role}")
        return value


class ArchetypeFeatureContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_id: str = Field(min_length=1)
    component_archetype_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    required_port_role_ids: list[str] = Field(min_length=1)
    optional_port_role_ids: list[str] = Field(default_factory=list)
    primitives: list[FeaturePrimitive] = Field(min_length=1)
    output_contract_id: str = Field(min_length=1)
    output_shape_policy: Literal["VARIABLE_BY_ARCHETYPE_FIXED_ONLY_AFTER_S1_FREEZE"]
    fitted: bool = False

    @model_validator(mode="after")
    def internally_consistent(self) -> "ArchetypeFeatureContract":
        declared = self.required_port_role_ids + self.optional_port_role_ids
        if len(declared) != len(set(declared)):
            raise ValueError("required and optional port roles must be disjoint and unique")
        primitive_ids = [item.primitive_id for item in self.primitives]
        if len(primitive_ids) != len(set(primitive_ids)):
            raise ValueError("primitive_id values must be unique within an archetype contract")
        unknown_roles = sorted(
            {
                role
                for primitive in self.primitives
                for role in primitive.input_port_role_ids
                if role not in set(declared)
            }
        )
        if unknown_roles:
            raise ValueError(f"primitives reference undeclared port roles: {unknown_roles}")
        return self


class EvidenceTokenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_id: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    model_input_fields: list[str] = Field(min_length=1)
    audit_sidecar_fields: list[str] = Field(default_factory=list)
    variable_token_count: Literal[True] = True
    labels_used: Literal[False] = False
    truth_used: Literal[False] = False
    identity_fields_allowed: Literal[False] = False

    @field_validator("model_input_fields")
    @classmethod
    def unique_truth_blind_fields(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("evidence-token fields must be unique")
        for field in value:
            lowered = field.lower()
            if any(token in lowered for token in _IDENTITY_OR_TRUTH_TOKENS):
                raise ValueError(f"evidence-token field is prohibited: {field}")
        return value

    @model_validator(mode="after")
    def sidecar_is_not_model_input(self) -> "EvidenceTokenContract":
        if len(self.audit_sidecar_fields) != len(set(self.audit_sidecar_fields)):
            raise ValueError("audit-sidecar fields must be unique")
        overlap = sorted(set(self.model_input_fields) & set(self.audit_sidecar_fields))
        if overlap:
            raise ValueError(f"audit-sidecar fields cannot be model inputs: {overlap}")
        return self


class ArchetypeFeatureCatalog(BaseModel):
    """Differentiated feature design; records the S1-authorized fit state when
    materialized.  Stage ``P018_S0_ZERO_FIT`` (pre-S1) is the frozen historical
    state; ``P018_S1_FIT_AUTHORIZED`` records that the project owner authorized
    fit on 2026-08-04 and one grouped-holdout baseline has been materialized."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-archetype-feature-catalog.v1"
    catalog_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    stage: Literal["P018_S0_ZERO_FIT", "P018_S1_FIT_AUTHORIZED"]
    source_id: str = Field(min_length=1, pattern=r"^[A-Z][A-Z0-9_]*$")
    source_adapter_id: str = Field(min_length=1)
    source_count: Literal[1] = 1
    evidence_token_contract: EvidenceTokenContract
    archetype_contracts: list[ArchetypeFeatureContract] = Field(min_length=1)
    feature_selection_evidence_used: Literal[False] = False
    fitted_model_count: int = Field(ge=0, default=0)
    s1_fit_authorized: bool = False
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def unique_and_differentiated(self) -> "ArchetypeFeatureCatalog":
        ids = [item.contract_id for item in self.archetype_contracts]
        archetypes = [item.component_archetype_id for item in self.archetype_contracts]
        if len(ids) != len(set(ids)):
            raise ValueError("feature contract IDs must be unique")
        if len(archetypes) != len(set(archetypes)):
            raise ValueError("component archetypes must be unique")
        if any(
            item.output_contract_id != self.evidence_token_contract.contract_id
            for item in self.archetype_contracts
        ):
            raise ValueError("all archetype contracts must emit the shared evidence-token contract")
        primitive_signatures = {
            tuple(
                sorted(
                    (
                        item.operator_family,
                        tuple(sorted(item.input_port_role_ids)),
                        item.output_token_type,
                        item.availability_policy,
                    )
                    for item in contract.primitives
                )
            )
            for contract in self.archetype_contracts
        }
        if len(primitive_signatures) != len(self.archetype_contracts):
            raise ValueError("archetype contracts must not reuse one identical feature recipe")
        return self


__all__ = [
    "ArchetypeFeatureCatalog",
    "ArchetypeFeatureContract",
    "EvidenceTokenContract",
    "FeaturePrimitive",
]
