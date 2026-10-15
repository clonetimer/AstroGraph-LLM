"""Independent nominal-audit contract for calibration transfer.

The reference distribution and the calibration-audit segment are deliberately
separate.  This module never updates a reference from deployment samples.  It
only asks whether a frozen reference + frozen alarm policy remains quiet on an
independent segment that is *claimed* to be nominal by the caller's provenance
contract.
"""
from __future__ import annotations

from enum import Enum
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field

from .trusted_reference import (
    CalibrationTransferState,
    ReferenceQualification,
    TrustedReferenceRecord,
    qualify_reference,
)


class CalibrationAuditState(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    INSUFFICIENT = "INSUFFICIENT"


class CalibrationTransferAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-calibration-transfer-audit.v1"
    reference_id: str = Field(min_length=1)
    audit_id: str = Field(min_length=1)
    audit_sample_count: int = Field(ge=0)
    on_threshold: float = Field(ge=0.0, le=1.0)
    minimum_consecutive: int = Field(ge=1)
    alarm_episode_count: int = Field(ge=0)
    active_sample_fraction: float = Field(ge=0.0, le=1.0)
    false_alarm_episodes_per_1000_nominal_samples: float | None = Field(default=None, ge=0.0)
    maximum_allowed_false_alarm_episodes_per_1000: float = Field(ge=0.0)
    maximum_allowed_active_sample_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    minimum_required_samples: int = Field(ge=1)
    state: CalibrationAuditState
    reference_updated_from_audit: bool = False
    claim_boundary: str = (
        "A passing audit supports transfer of the frozen calibration only on the declared nominal audit regime; "
        "it does not prove reference nominality and it never authorizes automatic reference adaptation."
    )


def _alarm_mask(scores: Sequence[float], *, threshold: float, minimum_consecutive: int) -> tuple[int, ...]:
    output: list[int] = []
    active = False
    run = 0
    for raw in scores:
        value = float(raw)
        if not 0.0 <= value <= 1.0:
            raise ValueError("calibration audit scores must lie in [0, 1]")
        if active:
            if value < threshold:
                active = False
                run = 0
        elif value >= threshold:
            run += 1
            if run >= minimum_consecutive:
                active = True
        else:
            run = 0
        output.append(int(active))
    return tuple(output)


def _episode_count(mask: Sequence[int]) -> int:
    episodes = 0
    previous = 0
    for value in mask:
        current = int(bool(value))
        if current and not previous:
            episodes += 1
        previous = current
    return episodes


def audit_nominal_calibration_transfer(
    scores: Sequence[float],
    *,
    reference_id: str,
    audit_id: str,
    on_threshold: float = 0.999,
    minimum_consecutive: int = 3,
    maximum_false_alarm_episodes_per_1000: float = 0.5,
    maximum_active_sample_fraction: float | None = None,
    minimum_samples: int = 512,
) -> CalibrationTransferAudit:
    values = tuple(float(value) for value in scores)
    if len(values) < minimum_samples:
        return CalibrationTransferAudit(
            reference_id=reference_id,
            audit_id=audit_id,
            audit_sample_count=len(values),
            on_threshold=on_threshold,
            minimum_consecutive=minimum_consecutive,
            alarm_episode_count=0,
            active_sample_fraction=0.0,
            false_alarm_episodes_per_1000_nominal_samples=None,
            maximum_allowed_false_alarm_episodes_per_1000=maximum_false_alarm_episodes_per_1000,
            maximum_allowed_active_sample_fraction=maximum_active_sample_fraction,
            minimum_required_samples=minimum_samples,
            state=CalibrationAuditState.INSUFFICIENT,
        )
    mask = _alarm_mask(values, threshold=on_threshold, minimum_consecutive=minimum_consecutive)
    episodes = _episode_count(mask)
    rate = 1000.0 * episodes / len(values)
    active_fraction = sum(mask) / len(mask)
    episode_rate_passed = rate <= maximum_false_alarm_episodes_per_1000
    active_fraction_passed = (
        maximum_active_sample_fraction is None
        or active_fraction <= maximum_active_sample_fraction
    )
    state = (
        CalibrationAuditState.PASSED
        if episode_rate_passed and active_fraction_passed
        else CalibrationAuditState.FAILED
    )
    return CalibrationTransferAudit(
        reference_id=reference_id,
        audit_id=audit_id,
        audit_sample_count=len(values),
        on_threshold=on_threshold,
        minimum_consecutive=minimum_consecutive,
        alarm_episode_count=episodes,
        active_sample_fraction=active_fraction,
        false_alarm_episodes_per_1000_nominal_samples=rate,
        maximum_allowed_false_alarm_episodes_per_1000=maximum_false_alarm_episodes_per_1000,
        maximum_allowed_active_sample_fraction=maximum_active_sample_fraction,
        minimum_required_samples=minimum_samples,
        state=state,
    )


def attach_transfer_audit(
    record: TrustedReferenceRecord,
    audit: CalibrationTransferAudit,
) -> TrustedReferenceRecord:
    if audit.reference_id != record.reference_id:
        raise ValueError("calibration audit reference_id does not match TrustedReferenceRecord")
    if audit.reference_updated_from_audit:
        raise ValueError("calibration audit must not update the frozen reference")
    if audit.state is CalibrationAuditState.PASSED:
        transfer_state = CalibrationTransferState.PASSED
    elif audit.state is CalibrationAuditState.FAILED:
        transfer_state = CalibrationTransferState.FAILED
    else:
        transfer_state = CalibrationTransferState.NOT_ASSESSED
    return record.model_copy(update={"transfer_state": transfer_state})


def qualify_reference_with_transfer_audit(
    record: TrustedReferenceRecord,
    audit: CalibrationTransferAudit,
    *,
    minimum_reference_samples: int = 1000,
) -> ReferenceQualification:
    updated = attach_transfer_audit(record, audit)
    return qualify_reference(
        updated,
        minimum_samples=minimum_reference_samples,
        require_transfer_pass=True,
    )
