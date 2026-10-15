"""Canonical R36 review lifecycle."""
from __future__ import annotations

from enum import IntEnum
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class ReviewLifecyclePhase(IntEnum):
    PENDING = 0
    OBSERVED = 1
    ANALYZED = 2
    EXPLAINED = 3
    REVIEWED = 4
    HANDOFF_READY = 5


class ReviewLifecycleInputs(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    observed: bool
    analyzed: bool
    explained: bool
    reviewed: bool
    handoff_ready: bool

    projection_blockers: tuple[str, ...] = ()

    @model_validator(mode="after")
    def monotonic(self) -> "ReviewLifecycleInputs":
        values = [
            self.observed,
            self.analyzed,
            self.explained,
            self.reviewed,
            self.handoff_ready,
        ]

        if any(
            values[i] and not values[i - 1]
            for i in range(1, len(values))
        ):
            raise ValueError(
                "review lifecycle inputs must be monotonic"
            )

        return self


class ReviewLifecycleProjection(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[
        "astrograph-review-lifecycle-projection.v1"
    ] = "astrograph-review-lifecycle-projection.v1"

    phase: str

    phase_rank: int = Field(
        ge=0,
        le=5,
    )

    blockers: tuple[str, ...] = ()

    claim_boundary: str = (
        "Single derived lifecycle only. R35 closure, coverage, "
        "replay and rereview remain read-only projections; "
        "this lifecycle is not an approval or release gate."
    )


def derive_review_lifecycle(
    inputs: ReviewLifecycleInputs,
) -> ReviewLifecycleProjection:

    if inputs.handoff_ready:
        phase = ReviewLifecyclePhase.HANDOFF_READY

    elif inputs.reviewed:
        phase = ReviewLifecyclePhase.REVIEWED

    elif inputs.explained:
        phase = ReviewLifecyclePhase.EXPLAINED

    elif inputs.analyzed:
        phase = ReviewLifecyclePhase.ANALYZED

    elif inputs.observed:
        phase = ReviewLifecyclePhase.OBSERVED

    else:
        phase = ReviewLifecyclePhase.PENDING

    blockers = list(
        dict.fromkeys(inputs.projection_blockers)
    )

    next_requirements = {
        ReviewLifecyclePhase.PENDING:
            "OBSERVATION_NOT_COMPLETE",

        ReviewLifecyclePhase.OBSERVED:
            "ANALYSIS_NOT_COMPLETE",

        ReviewLifecyclePhase.ANALYZED:
            "EXPLANATION_NOT_COMPLETE",

        ReviewLifecyclePhase.EXPLAINED:
            "HUMAN_REVIEW_NOT_COMPLETE",

        ReviewLifecyclePhase.REVIEWED:
            "HANDOFF_NOT_READY",

        ReviewLifecyclePhase.HANDOFF_READY:
            None,
    }

    requirement = next_requirements[phase]

    if requirement:
        blockers.insert(0, requirement)

    return ReviewLifecycleProjection(
        phase=phase.name,
        phase_rank=int(phase),
        blockers=tuple(
            dict.fromkeys(blockers)
        ),
    )


__all__ = [
    "ReviewLifecyclePhase",
    "ReviewLifecycleInputs",
    "ReviewLifecycleProjection",
    "derive_review_lifecycle",
]
