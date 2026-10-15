"""Conservative identifiability screening for generic telemetry channels.

Knowledge uniqueness is not empirical separability.  This module deliberately
keeps those two notions separate so a one-to-one catalog mapping can never be
mistaken for a validated single-channel classifier.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SignalIdentifiability(_Strict):
    channel: str
    catalog_candidate_fault_ids: tuple[str, ...] = ()
    primary_fault_ids: tuple[str, ...] = ()
    supporting_fault_ids: tuple[str, ...] = ()
    context_fault_ids: tuple[str, ...] = ()
    knowledge_status: Literal[
        "NO_CATALOG_MAPPING",
        "KNOWLEDGE_UNIQUE",
        "KNOWLEDGE_AMBIGUOUS",
    ]
    empirical_profile_id: str | None = None
    empirical_classification_validated: bool = False
    formal_single_signal_classification_eligible: bool = False
    reason: str


class IdentifiabilityReport(_Strict):
    schema_version: Literal["astrograph-identifiability-report.v1"] = "astrograph-identifiability-report.v1"
    catalog_id: str
    catalog_version: str
    channels: tuple[SignalIdentifiability, ...]
    eligible_channel_count: int = Field(ge=0)
    claim_boundary: str = (
        "Catalog mapping describes engineering candidate relationships only.  A unique fault mapping is not "
        "evidence of empirical class separability. Formal single-signal classification requires an independently "
        "validated labelled profile."
    )


def screen_single_signal_identifiability(
    channel_names: list[str] | tuple[str, ...],
    *,
    fault_catalog_path: str | Path,
    empirical_profiles: dict[str, dict[str, Any]] | None = None,
) -> IdentifiabilityReport:
    payload = json.loads(Path(fault_catalog_path).read_text(encoding="utf-8"))
    by_channel: dict[str, list[tuple[str, str]]] = {}
    for fault in payload.get("fault_modes", []):
        fault_id = str(fault.get("fault_id", "")).strip()
        if not fault_id:
            continue
        for row in fault.get("telemetry_channels", []):
            channel = str(row.get("channel", "")).strip()
            if channel:
                by_channel.setdefault(channel, []).append((fault_id, str(row.get("role", "SUPPORTING")).upper()))
    profiles = empirical_profiles or {}
    results: list[SignalIdentifiability] = []
    for channel in channel_names:
        rows = by_channel.get(str(channel), [])
        fault_ids = tuple(sorted({fid for fid, _ in rows}))
        primary = tuple(sorted({fid for fid, role in rows if role == "PRIMARY"}))
        supporting = tuple(sorted({fid for fid, role in rows if role == "SUPPORTING"}))
        context = tuple(sorted({fid for fid, role in rows if role == "CONTEXT"}))
        if not fault_ids:
            knowledge_status = "NO_CATALOG_MAPPING"
        elif len(fault_ids) == 1:
            knowledge_status = "KNOWLEDGE_UNIQUE"
        else:
            knowledge_status = "KNOWLEDGE_AMBIGUOUS"
        profile = profiles.get(str(channel), {})
        profile_id = str(profile.get("profile_id", "")).strip() or None
        validated = bool(profile.get("independently_validated", False))
        labelled_faults = set(map(str, profile.get("fault_ids", [])))
        # A valid empirical profile must identify at least two classes (normal can be external)
        # and must not claim faults outside the engineering candidate universe when that universe exists.
        empirical_ok = bool(profile_id and validated and labelled_faults and (not fault_ids or labelled_faults.issubset(set(fault_ids))))
        eligible = empirical_ok and knowledge_status != "NO_CATALOG_MAPPING"
        if eligible:
            reason = "INDEPENDENT_LABELLED_PROFILE_VALIDATED"
        elif knowledge_status == "KNOWLEDGE_AMBIGUOUS":
            reason = "MULTIPLE_CATALOG_FAULTS_REQUIRE_EMPIRICAL_SEPARABILITY_EVIDENCE"
        elif knowledge_status == "KNOWLEDGE_UNIQUE":
            reason = "UNIQUE_KNOWLEDGE_MAPPING_IS_NOT_EMPIRICAL_CLASSIFICATION_EVIDENCE"
        else:
            reason = "NO_ENGINEERING_FAULT_MAPPING_FOR_CHANNEL"
        results.append(SignalIdentifiability(
            channel=str(channel), catalog_candidate_fault_ids=fault_ids, primary_fault_ids=primary,
            supporting_fault_ids=supporting, context_fault_ids=context, knowledge_status=knowledge_status,
            empirical_profile_id=profile_id, empirical_classification_validated=empirical_ok,
            formal_single_signal_classification_eligible=eligible, reason=reason,
        ))
    return IdentifiabilityReport(
        catalog_id=str(payload.get("catalog_id", "unknown")),
        catalog_version=str(payload.get("catalog_version", "unknown")),
        channels=tuple(results),
        eligible_channel_count=sum(int(x.formal_single_signal_classification_eligible) for x in results),
    )


__all__ = ["SignalIdentifiability", "IdentifiabilityReport", "screen_single_signal_identifiability"]
