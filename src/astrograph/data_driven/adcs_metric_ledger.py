"""Typed ADCS metric/protocol ledger.

This module prevents a recurring class of evaluation error: reporting a metric
without the protocol, cohort reuse status, denominator, and model binding that
make the number meaningful.  It is intentionally read-only with respect to
models, knowledge, memory, and production registries.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


EvidenceRole = Literal[
    "RESEARCH_GROUP_CV_BASELINE",
    "HISTORICAL_PRODUCTION_QUALIFICATION",
    "USED_GROUP_RETROSPECTIVE_REPLAY",
    "CURRENT_MISSION_BINDING",
]
ReuseStatus = Literal[
    "TRAIN_TEST_GROUP_CV",
    "HISTORICAL_FROZEN_QUALIFICATION",
    "PREVIOUSLY_USED_TEST_GROUPS",
    "CONFIGURATION_BINDING_ONLY",
]


class ADCSMetricRecord(_Frozen):
    record_id: str
    protocol_id: str
    evidence_role: EvidenceRole
    evidence_date: str
    cohort: str
    reuse_status: ReuseStatus
    model_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    metrics: dict[str, float | None] = {}
    denominators: dict[str, int | float | str] = {}
    source_path: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    independent_confirmation: bool = False
    current_operational_replay_evidence: bool = False
    claim_boundary: str


class ADCSMetricProtocolLedger(_Frozen):
    schema_version: str = "astrograph-adcs-metric-protocol-ledger.v1"
    version: str
    current_mission_model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    records: tuple[ADCSMetricRecord, ...]
    reporting_rules: tuple[str, ...]
    correction_note: str

    @model_validator(mode="after")
    def validate_semantics(self):
        ids = [record.record_id for record in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate ADCS metric record_id")
        roles = {record.evidence_role for record in self.records}
        required = {
            "RESEARCH_GROUP_CV_BASELINE",
            "HISTORICAL_PRODUCTION_QUALIFICATION",
            "USED_GROUP_RETROSPECTIVE_REPLAY",
            "CURRENT_MISSION_BINDING",
        }
        if not required <= roles:
            raise ValueError(f"missing required evidence roles: {sorted(required - roles)}")
        replay = [r for r in self.records if r.evidence_role == "USED_GROUP_RETROSPECTIVE_REPLAY"]
        if any(r.independent_confirmation for r in replay):
            raise ValueError("previously used replay groups cannot be relabeled independent confirmation")
        bindings = [r for r in self.records if r.evidence_role == "CURRENT_MISSION_BINDING"]
        if len(bindings) != 1 or bindings[0].model_sha256 != self.current_mission_model_sha256:
            raise ValueError("current Mission model binding is missing or inconsistent")
        return self


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_ledger(path: Path) -> ADCSMetricProtocolLedger:
    return ADCSMetricProtocolLedger.model_validate_json(path.read_text(encoding="utf-8"))


def verify_ledger_sources(root: Path, ledger: ADCSMetricProtocolLedger) -> None:
    root = root.resolve()
    for record in ledger.records:
        source = (root / record.source_path).resolve()
        if not source.is_relative_to(root):
            raise ValueError(f"ledger source escaped project root: {record.source_path}")
        if not source.is_file():
            raise FileNotFoundError(source)
        actual = file_sha256(source)
        if actual != record.source_sha256:
            raise ValueError(f"ledger source hash changed: {record.record_id}: {actual} != {record.source_sha256}")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
