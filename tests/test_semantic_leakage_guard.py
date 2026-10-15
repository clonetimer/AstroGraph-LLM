from __future__ import annotations

import pytest

from astrograph.semantics import is_prohibited_field


@pytest.mark.parametrize(
    "field",
    [
        "fault_type",
        "fault-type",
        "Fault Type",
        "condition_id",
        "diagnostic_class",
        "event_start_s",
        "event.active",
        "injected_fault",
        "case_id",
        "health_state",
        "modifier.injected_fault",
    ],
)
def test_answer_proxy_field_variants_are_prohibited(field: str) -> None:
    assert is_prohibited_field(field) is True


@pytest.mark.parametrize(
    "field",
    [
        "operating_condition_id",
        "battery_health_indicator",
        "fault_tolerant_mode_enabled",
        "event_counter",
        "reaction_wheel_speed_rad_s",
        "payload_mode_commanded",
    ],
)
def test_physical_or_command_observations_are_not_rejected_by_substring_only(field: str) -> None:
    assert is_prohibited_field(field) is False
