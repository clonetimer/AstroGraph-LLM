from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelContract(_FrozenContract):
    """Stable runtime boundary for any diagnostic/state-estimation model provider.

    The contract deliberately describes the artifact and callable semantics rather than
    a concrete model family. Runtime code must not depend on concrete ML frameworks or source-specific
    implementation details.
    """

    schema_version: Literal["astrograph-model-contract.v1"] = "astrograph-model-contract.v1"
    contract_id: str = "astrograph.model.v1"
    input_protocol: str = "astrograph-model-inference-request.v1"
    output_protocol: str = "astrograph-model-prediction.v1"
    artifact_manifest_required: bool = True
    artifact_hash_required: bool = True
    raw_pickle_allowed: bool = False
    task_axes: tuple[str, ...] = ("detection", "localization", "classification")
    required_metadata: tuple[str, ...] = (
        "model_id",
        "model_version",
        "input_schema_version",
        "output_schema_version",
    )


class EvidenceContract(_FrozenContract):
    """Stable boundary between numerical/physical providers and reasoning layers."""

    schema_version: Literal["astrograph-evidence-contract.v1"] = "astrograph-evidence-contract.v1"
    contract_id: str = "astrograph.evidence.v1"
    bundle_protocol: str = "astrograph-episode-evidence-bundle.v1"
    facts_must_be_typed: bool = True
    provenance_required: bool = True
    calibration_semantics_required: bool = True
    truth_fields_forbidden: bool = True
    required_fact_fields: tuple[str, ...] = (
        "evidence_fact_id",
        "evidence_definition_id",
        "value",
        "status",
    )


class KnowledgeContract(_FrozenContract):
    """Stable runtime boundary for knowledge graph reads and governed releases."""

    schema_version: Literal["astrograph-knowledge-contract.v1"] = "astrograph-knowledge-contract.v1"
    contract_id: str = "astrograph.knowledge.v1"
    query_protocol: str = "astrograph-read-only-graph-query.v1"
    release_manifest_required: bool = True
    content_hash_required: bool = True
    read_only_by_default: bool = True
    write_requires_governance: bool = True
    allowed_runtime_operations: tuple[str, ...] = (
        "forward_evidence_query",
        "backward_fault_query",
        "candidate_association_read",
    )


class PolicyContract(_FrozenContract):
    """Single ownership boundary for runtime decision policy.

    Dataset/source profiles describe data only. They may reference a policy profile ID,
    but they must not silently redefine thresholds/fusion/abstention semantics.
    """

    schema_version: Literal["astrograph-policy-contract.v1"] = "astrograph-policy-contract.v1"
    contract_id: str = "astrograph.policy.v1"
    owns: tuple[str, ...] = (
        "fusion",
        "thresholds",
        "abstention",
        "open_set",
        "human_review",
        "release_gates",
    )
    dataset_profile_may_override: bool = False
    explicit_profile_receipt_required: bool = True


class StableContractBundle(_FrozenContract):
    schema_version: Literal["astrograph-stable-contract-bundle.v1"] = (
        "astrograph-stable-contract-bundle.v1"
    )
    model: ModelContract = Field(default_factory=ModelContract)
    evidence: EvidenceContract = Field(default_factory=EvidenceContract)
    knowledge: KnowledgeContract = Field(default_factory=KnowledgeContract)
    policy: PolicyContract = Field(default_factory=PolicyContract)

    @model_validator(mode="after")
    def unique_contract_ids(self) -> "StableContractBundle":
        ids = [self.model.contract_id, self.evidence.contract_id, self.knowledge.contract_id, self.policy.contract_id]
        if len(ids) != len(set(ids)):
            raise ValueError("stable contract IDs must be unique")
        return self


STABLE_CONTRACTS = StableContractBundle()
