from __future__ import annotations

from pathlib import Path

import pytest

from astrograph.data_driven.mechanism_graph import load_progressive_hierarchy
from scripts.integration import run_adcs_v522_regression as adcs


ROOT = Path(__file__).resolve().parents[1]


def test_adcs_progressive_contract_routes_all_six_levels() -> None:
    contract = load_progressive_hierarchy(
        ROOT / "configs/data_driven/progressive_hierarchies.json",
        "adcs_v5_2_2",
        list(adcs.FEATURE_COLUMNS),
        list(adcs.COMPONENT_NAMES),
    )
    assert contract["components"] == list(adcs.COMPONENT_NAMES)
    assert len(contract["parameters"]) == 11
    assert len(contract["parts"]) == 8
    assert contract["functional_modules"] == [
        "attitude_control", "inertial_sensing", "actuation", "optical_sensing"
    ]
    assert contract["subsystems"] == ["adcs"]
    assert contract["spacecraft"] == "whole_spacecraft"
    assert all(contract["parameter_groups"])
    assert contract["topology_fit_scope"].startswith("configuration_only")


def test_progressive_contract_fails_closed_when_required_channel_is_absent() -> None:
    with pytest.raises(ValueError, match="no selected telemetry"):
        load_progressive_hierarchy(
            ROOT / "configs/data_driven/progressive_hierarchies.json",
            "adcs_v5_2_2",
            ["adcs.pointing_error_deg"],
            list(adcs.COMPONENT_NAMES),
        )
