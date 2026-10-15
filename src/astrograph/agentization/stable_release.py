"""Software-stability gate with research claims explicitly separated.

A software baseline may be stable while a scientific route is blocked, but only if
all shipped software tests/reconstruction checks pass and blocked research claims are
disabled and disclosed.  This module never upgrades scientific qualification.
"""
from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ReleaseScope(str, Enum):
    SOFTWARE_STABLE_RESEARCH_BLOCKED = "SOFTWARE_STABLE_RESEARCH_BLOCKED"
    RESEARCH_RELEASE = "RESEARCH_RELEASE"


class CheckStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    BLOCKED = "BLOCKED"


class ReleaseCheck(BaseModel):
    model_config=ConfigDict(extra="forbid", frozen=True)
    check_id: str
    required_for_software: bool
    required_for_research: bool
    status: CheckStatus
    evidence_sha256: str | None = None
    reason: str

    @field_validator("evidence_sha256")
    @classmethod
    def _digest(cls, v):
        if v is None: return v
        v=v.lower()
        if len(v)!=64 or any(c not in "0123456789abcdef" for c in v):
            raise ValueError("invalid SHA-256")
        return v


class StabilityInput(BaseModel):
    model_config=ConfigDict(extra="forbid", frozen=True)
    schema_version: str = "astrograph-stability-input.v1"
    candidate_id: str
    scope: ReleaseScope
    project_status_before: str
    checks: tuple[ReleaseCheck, ...]
    permanently_post_hoc_groups: tuple[int, ...]=(4,9)
    qualification_change_requested: bool=False
    confirmatory_evidence_present: bool=False
    disabled_research_capabilities: tuple[str, ...]=()
    disclosed_research_blockers: tuple[str, ...]=()

    @model_validator(mode="after")
    def _safety(self):
        if self.qualification_change_requested:
            raise ValueError("software stability cannot raise G0-G4/flight/safety qualification")
        if self.scope is ReleaseScope.SOFTWARE_STABLE_RESEARCH_BLOCKED and not self.disclosed_research_blockers:
            raise ValueError("research-blocked scope requires disclosed blockers")
        return self


class StabilityReceipt(BaseModel):
    model_config=ConfigDict(extra="forbid", frozen=True)
    schema_version: str = "astrograph-stability-receipt.v1"
    candidate_id: str
    scope: ReleaseScope
    status: str
    software_stable: bool
    research_release_allowed: bool
    failed_checks: tuple[str, ...]
    blocked_checks: tuple[str, ...]
    skipped_checks: tuple[str, ...]
    unchanged_boundaries: tuple[str, ...]
    receipt_sha256: str


def _digest(payload: Mapping[str,Any]) -> str:
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()


def assess_stability(inp: StabilityInput) -> StabilityReceipt:
    failed=tuple(c.check_id for c in inp.checks if c.status is CheckStatus.FAIL)
    blocked=tuple(c.check_id for c in inp.checks if c.status is CheckStatus.BLOCKED)
    skipped=tuple(c.check_id for c in inp.checks if c.status is CheckStatus.SKIP)
    software_required_bad=tuple(c.check_id for c in inp.checks if c.required_for_software and c.status is not CheckStatus.PASS)
    research_required_bad=tuple(c.check_id for c in inp.checks if c.required_for_research and c.status is not CheckStatus.PASS)
    software_stable=not software_required_bad
    research_allowed=software_stable and not research_required_bad and inp.confirmatory_evidence_present and inp.scope is ReleaseScope.RESEARCH_RELEASE
    if research_allowed:
        status="STABLE_RESEARCH_RELEASE"
    elif software_stable and inp.scope is ReleaseScope.SOFTWARE_STABLE_RESEARCH_BLOCKED:
        status="STABLE_SOFTWARE_BASELINE_RESEARCH_BLOCKED"
    else:
        status="STABLE_CANDIDATE_BLOCKED"
    boundaries=(
        "B005 remains the last historically validated stable research baseline.",
        "Groups 4/9 remain permanently post-hoc and cannot select any parameter or policy.",
        "No new confirmatory, flight, safety, G0-G4, or universal-KG-gain claim is created.",
        "A software-stable status does not imply research validation of disabled or blocked capabilities.",
    )
    payload={"schema_version":"astrograph-stability-receipt.v1","candidate_id":inp.candidate_id,"scope":inp.scope.value,"status":status,"software_stable":software_stable,"research_release_allowed":research_allowed,"failed_checks":list(failed),"blocked_checks":list(blocked),"skipped_checks":list(skipped),"unchanged_boundaries":list(boundaries)}
    return StabilityReceipt(**payload, receipt_sha256=_digest(payload))
