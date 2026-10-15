from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChangeImpact(str, Enum):
    NO_CORE_CHANGE = "NO_CORE_CHANGE"
    CONFIG_OR_ADAPTER_EXTENSION = "CONFIG_OR_ADAPTER_EXTENSION"
    BREAKING_ADAPTER_CHANGE = "BREAKING_ADAPTER_CHANGE"
    CORE_SCHEMA_REVIEW = "CORE_SCHEMA_REVIEW"


class SimulationInterfaceSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_contract_version: str
    telemetry_dictionary_version: str = ""
    fault_catalog_version: str = ""
    streams: list[str] = Field(default_factory=list)
    channel_ids: list[str] = Field(default_factory=list)
    fault_ids: list[str] = Field(default_factory=list)
    degradation_ids: list[str] = Field(default_factory=list)
    required_identity_fields: list[str] = Field(
        default_factory=lambda: ["dataset_id", "run_id", "record_id", "time_s"]
    )
    raw_labels_separated: bool = True
    paired_nominal_supported: bool = True


class CompatibilityAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    impact: ChangeImpact
    reasons: list[str]
    added_streams: list[str] = Field(default_factory=list)
    removed_streams: list[str] = Field(default_factory=list)
    added_channels: list[str] = Field(default_factory=list)
    removed_channels: list[str] = Field(default_factory=list)
    added_fault_ids: list[str] = Field(default_factory=list)
    removed_fault_ids: list[str] = Field(default_factory=list)
    added_degradation_ids: list[str] = Field(default_factory=list)
    removed_degradation_ids: list[str] = Field(default_factory=list)
    required_actions: list[str] = Field(default_factory=list)


def _major(version: str) -> str:
    digits = "".join(character if character.isdigit() or character == "." else " " for character in version)
    token = next((part for part in digits.split() if part), "")
    return token.split(".")[0] if token else version


def assess_simulation_interface_change(
    current: SimulationInterfaceSnapshot,
    candidate: SimulationInterfaceSnapshot,
) -> CompatibilityAssessment:
    added_streams = sorted(set(candidate.streams) - set(current.streams))
    removed_streams = sorted(set(current.streams) - set(candidate.streams))
    added_channels = sorted(set(candidate.channel_ids) - set(current.channel_ids))
    removed_channels = sorted(set(current.channel_ids) - set(candidate.channel_ids))
    added_faults = sorted(set(candidate.fault_ids) - set(current.fault_ids))
    removed_faults = sorted(set(current.fault_ids) - set(candidate.fault_ids))
    added_degradations = sorted(set(candidate.degradation_ids) - set(current.degradation_ids))
    removed_degradations = sorted(set(current.degradation_ids) - set(candidate.degradation_ids))

    reasons: list[str] = []
    actions: list[str] = []
    impact = ChangeImpact.NO_CORE_CHANGE

    if _major(current.dataset_contract_version) != _major(candidate.dataset_contract_version):
        impact = ChangeImpact.CORE_SCHEMA_REVIEW
        reasons.append("Dataset protocol major version changed.")
        actions.append("Review Pydantic protocols, data gate, registry, and API compatibility before ingestion.")
    if not candidate.raw_labels_separated:
        impact = ChangeImpact.CORE_SCHEMA_REVIEW
        reasons.append("Candidate dataset no longer separates raw telemetry and labels.")
        actions.append("Reject dataset or restore raw/label/evidence isolation.")
    if not candidate.paired_nominal_supported:
        impact = max(impact, ChangeImpact.BREAKING_ADAPTER_CHANGE, key=list(ChangeImpact).index)
        reasons.append("Paired nominal baseline is no longer supported.")
        actions.append("Provide a new baseline detector protocol before using paired residual detection.")
    missing_identity = sorted(set(current.required_identity_fields) - set(candidate.required_identity_fields))
    if missing_identity:
        impact = ChangeImpact.CORE_SCHEMA_REVIEW
        reasons.append(f"Required identity fields were removed: {missing_identity}")
        actions.append("Restore stable dataset/run/record/time identities.")

    if removed_streams or removed_channels or removed_faults or removed_degradations:
        if impact == ChangeImpact.NO_CORE_CHANGE:
            impact = ChangeImpact.BREAKING_ADAPTER_CHANGE
        reasons.append("Published stable identifiers or streams were removed.")
        actions.append("Add compatibility aliases or a versioned migration adapter; do not silently rename IDs.")

    if added_streams or added_channels or added_faults or added_degradations:
        if impact == ChangeImpact.NO_CORE_CHANGE:
            impact = ChangeImpact.CONFIG_OR_ADAPTER_EXTENSION
        reasons.append("New streams, channels, faults, or degradations were added without removing existing protocols.")
        actions.extend([
            "Update telemetry dictionary and stream adapter configuration.",
            "Add anomaly rules and ontology/knowledge entries for new diagnostic concepts.",
            "Run the external data gate and regression tests.",
        ])

    if current.telemetry_dictionary_version != candidate.telemetry_dictionary_version:
        reasons.append("Telemetry dictionary version changed.")
        if impact == ChangeImpact.NO_CORE_CHANGE:
            impact = ChangeImpact.CONFIG_OR_ADAPTER_EXTENSION
        actions.append("Rebuild window indexes and validate units, data types, and sampling periods.")
    if current.fault_catalog_version != candidate.fault_catalog_version:
        reasons.append("Fault catalog version changed.")
        if impact == ChangeImpact.NO_CORE_CHANGE:
            impact = ChangeImpact.CONFIG_OR_ADAPTER_EXTENSION
        actions.append("Update fault protocols, labels, ontology and knowledge release; diagnosis core remains unchanged.")

    if not reasons:
        reasons.append("No protocol-level difference detected.")
        actions.append("Run the standard data gate and regression suite.")

    return CompatibilityAssessment(
        impact=impact,
        reasons=reasons,
        added_streams=added_streams,
        removed_streams=removed_streams,
        added_channels=added_channels,
        removed_channels=removed_channels,
        added_fault_ids=added_faults,
        removed_fault_ids=removed_faults,
        added_degradation_ids=added_degradations,
        removed_degradation_ids=removed_degradations,
        required_actions=sorted(set(actions)),
    )
