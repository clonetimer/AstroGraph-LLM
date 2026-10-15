"""Reconstructed R36.4 contract tests from the source handoff."""
import pytest
from pydantic import ValidationError

from astrograph.integrations.review_lifecycle import (
    ReviewLifecycleInputs,
    ReviewLifecyclePhase,
    derive_review_lifecycle,
)


def test_reviewed_projects_to_reviewed_with_handoff_blocker():
    result = derive_review_lifecycle(
        ReviewLifecycleInputs(
            observed=True, analyzed=True, explained=True, reviewed=True, handoff_ready=False,
            projection_blockers=("EXTERNAL_BLOCKER",),
        )
    )
    assert result.phase == ReviewLifecyclePhase.REVIEWED.name
    assert result.phase_rank == int(ReviewLifecyclePhase.REVIEWED)
    assert result.blockers[0] == "HANDOFF_NOT_READY"
    assert "EXTERNAL_BLOCKER" in result.blockers


def test_handoff_ready_is_terminal_projection():
    result = derive_review_lifecycle(
        ReviewLifecycleInputs(observed=True, analyzed=True, explained=True, reviewed=True, handoff_ready=True)
    )
    assert result.phase == "HANDOFF_READY"
    assert result.blockers == ()


def test_lifecycle_inputs_are_monotonic():
    with pytest.raises(ValidationError):
        ReviewLifecycleInputs(observed=True, analyzed=False, explained=True, reviewed=False, handoff_ready=False)


def test_empty_lifecycle_is_pending_not_observed():
    result = derive_review_lifecycle(
        ReviewLifecycleInputs(
            observed=False, analyzed=False, explained=False,
            reviewed=False, handoff_ready=False,
        )
    )
    assert result.phase == ReviewLifecyclePhase.PENDING.name
    assert result.phase_rank == int(ReviewLifecyclePhase.PENDING)
    assert result.blockers[0] == "OBSERVATION_NOT_COMPLETE"
