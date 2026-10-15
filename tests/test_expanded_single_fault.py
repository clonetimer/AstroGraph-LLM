from __future__ import annotations

import json
from importlib.resources import files

import numpy as np
import pandas as pd

from astrograph.diagnosis.expanded_single_fault import (
    ExpandedSingleFaultEngine,
    extract_paired_residual_features,
)


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    profile = {
        "imu_gyro_x_rad_s": [0.0, 0.0, 0.0],
        "star_tracker_q0": [1.0, 1.0, 1.0],
        "sun_sensor_output": [0.5, 0.5, 0.5],
        "magnetometer_x_t": [1e-5, 1e-5, 1e-5],
        "thermal_electronics_temp_c": [20.0, 20.0, 20.0],
        "radiator_electronics_on": [1.0, 1.0, 1.0],
    }
    nominal = pd.DataFrame({"time_s": [0.0, 1.0, 2.0], "signal": [1.0, 1.0, 1.0], **profile})
    effect = pd.DataFrame({"time_s": [0.0, 1.0, 2.0], "signal": [2.0, 2.0, 2.0], **profile})
    return nominal, effect


def _catalog(*, grouped: bool) -> dict:
    nominal, effect = _frames()
    features = extract_paired_residual_features(nominal, effect)
    names = sorted(features)
    vector = np.asarray([features[name] for name in names], dtype=float)
    norm = float(np.linalg.norm(vector)) or 1.0
    prototype = (vector / norm).tolist()
    groups = {}
    observation_class = "mode_a"
    prototypes = [
        {
            "mode": "mode_a",
            "component": "component_a",
            "component_display": "ComponentA",
            "subsystem": "TEST",
            "observation_class": "mode_a",
            "vector": prototype,
        }
    ]
    if grouped:
        observation_class = "GROUP_A"
        groups = {
            "GROUP_A": {
                "members": ["mode_a", "mode_b"],
                "subsystem": "TEST",
                "component": None,
                "reason": "same observable response",
            }
        }
        prototypes[0]["observation_class"] = observation_class
    return {
        "schema_version": "test",
        "feature_names": names,
        "center": [0.0] * len(names),
        "scale": [1.0] * len(names),
        "prototypes": prototypes,
        "equivalence_groups": groups,
        "claim_boundary": "test boundary",
    }


def test_packaged_catalog_covers_all_expanded_modes() -> None:
    payload = json.loads(
        files("astrograph.resources").joinpath("expanded_single_fault_catalog.json").read_text(encoding="utf-8")
    )
    assert len(payload["prototypes"]) == 69
    assert len({item["mode"] for item in payload["prototypes"]}) == 69
    assert len(payload["equivalence_groups"]) == 7
    assert payload["training_split"] == "development"


def test_exact_mode_is_reported_without_case_identifier_feature() -> None:
    nominal, effect = _frames()
    engine = ExpandedSingleFaultEngine(_catalog(grouped=False))
    result = engine.diagnose_frames(nominal=nominal, effect=effect)
    assert result.status == "DIAGNOSED"
    assert result.primary_mode == "mode_a"
    assert result.candidate_modes == ["mode_a"]
    assert all("case" not in name.lower() for name in engine.feature_names)


def test_observational_equivalence_fails_closed_with_candidates() -> None:
    nominal, effect = _frames()
    result = ExpandedSingleFaultEngine(_catalog(grouped=True)).diagnose_frames(nominal=nominal, effect=effect)
    assert result.status == "OBSERVATIONALLY_EQUIVALENT"
    assert result.primary_mode is None
    assert result.observation_class == "GROUP_A"
    assert result.candidate_modes == ["mode_a", "mode_b"]
    assert "SOURCE_NOT_IDENTIFIABLE_FROM_AVAILABLE_TELEMETRY" in result.reason_codes


def test_no_residual_is_insufficient_evidence() -> None:
    nominal, _ = _frames()
    result = ExpandedSingleFaultEngine(_catalog(grouped=False)).diagnose_frames(
        nominal=nominal, effect=nominal.copy()
    )
    assert result.status == "INSUFFICIENT_EVIDENCE"
    assert result.candidate_modes == []
