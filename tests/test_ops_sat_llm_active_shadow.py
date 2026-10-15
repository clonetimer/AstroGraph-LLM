from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ops_sat_llm_active_shadow", ROOT / "scripts/experiments/run_ops_sat_llm_active_shadow.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _points(values):
    return [
        {"channel": "CADC0001", "timestamp": f"2026-01-01T00:00:{index:02d}+00:00",
         "value": value, "sampling": "1", "segment": "7"}
        for index, value in enumerate(values)
    ]


@pytest.mark.parametrize("action", MODULE.ACTIVE_ACTIONS)
def test_active_measurements_are_finite_source_bound_and_non_voting(action: str) -> None:
    result = MODULE.measure_action(
        _points([0.0, 0.1, -0.1, 0.0, 1.0, 1.1, 0.9, 1.0]), action,
        source_sha256="a" * 64,
    )
    assert result["observation_id"].startswith("OPSOBS-")
    assert result["source_sha256"] == "a" * 64
    assert result["creates_new_raw_observation"] is True
    assert result["formal_evidence_generated"] is False


def test_baseline_action_cannot_masquerade_as_new_raw_observation() -> None:
    with pytest.raises(ValueError, match="not a raw telemetry"):
        MODULE.measure_action(_points([0, 0, 1, 1]), "inspect_baseline", source_sha256="a" * 64)


def test_unknown_measurement_fails_closed() -> None:
    with pytest.raises(ValueError, match="unregistered"):
        MODULE.measure_action(_points([0, 0, 1, 1]), "read_truth", source_sha256="a" * 64)
