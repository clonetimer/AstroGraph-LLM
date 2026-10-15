from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from astrograph.diagnosis.sat_sim_multifault import diagnose_paired_multifault
from astrograph.diagnosis.unified import UnifiedDiagnosisService
from astrograph.diagnosis.unified_models import UnifiedDiagnosisStatus


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "trusted_single_fault"
EXPECTED = json.loads((FIXTURE_ROOT / "expected.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case_id", sorted(EXPECTED))
def test_holdout_single_faults_are_exactly_diagnosed_and_localized(case_id: str) -> None:
    expected = EXPECTED[case_id]
    root = FIXTURE_ROOT / case_id
    result = UnifiedDiagnosisService().diagnose_paths(
        case_id="anonymous-case",
        telemetry_path=root / "effect.csv",
        nominal_path=root / "nominal.csv",
        backend="composite",
    )
    assert result.status == UnifiedDiagnosisStatus.DIAGNOSED
    assert result.primary_effect_id == expected["effect_id"]
    assert result.primary_subsystem == expected["subsystem"]
    assert result.primary_component == expected["component"]
    assert result.labels_used is False
    assert result.truth_used is False
    assert result.estimated_onset_time_s == pytest.approx(expected["onset"], abs=1e-6)
    assert result.severity_score == pytest.approx(expected["severity"], abs=0.04)
    assert result.diagnostic_evidence


def test_payload_degradation_is_not_promoted_to_payload_off() -> None:
    root = FIXTURE_ROOT / "payload_degraded_4"
    result = UnifiedDiagnosisService().diagnose_paths(
        case_id="no-semantic-case-id",
        telemetry_path=root / "effect.csv",
        nominal_path=root / "nominal.csv",
        backend="composite",
    )
    assert result.primary_effect_id == "payload_instrument_degraded_rate"
    assert "payload_instrument_off" not in result.conclusion_effect_ids


def test_rw_root_suppresses_downstream_payload_and_comm_as_independent_faults(tmp_path: Path) -> None:
    root = FIXTURE_ROOT / "rw_bearing_4"
    result = diagnose_paired_multifault(
        case_id="anonymous",
        nominal_csv=root / "nominal.csv",
        effect_csv=root / "effect.csv",
    )
    assert result.detected_effect_ids == ["adcs_rw_bearing_seizure"]
    assert result.primary_effect_id == "adcs_rw_bearing_seizure"
    assert "payload_instrument_off" not in result.detected_effect_ids
    assert "comm_data_downlink_link_loss" not in result.detected_effect_ids


def test_case_identifier_does_not_change_diagnosis() -> None:
    root = FIXTURE_ROOT / "thermal_heater_4"
    service = UnifiedDiagnosisService()
    first = service.diagnose_paths(
        case_id="thermal_heater_stuck_on",
        telemetry_path=root / "effect.csv",
        nominal_path=root / "nominal.csv",
        backend="composite",
    )
    second = service.diagnose_paths(
        case_id="opaque-9172",
        telemetry_path=root / "effect.csv",
        nominal_path=root / "nominal.csv",
        backend="composite",
    )
    def strip_case_id(value):
        if isinstance(value, dict):
            return {key: strip_case_id(item) for key, item in value.items() if key != "case_id"}
        if isinstance(value, list):
            return [strip_case_id(item) for item in value]
        return value

    assert strip_case_id(first.model_dump()) == strip_case_id(second.model_dump())


def test_truth_columns_still_fail_closed(tmp_path: Path) -> None:
    root = FIXTURE_ROOT / "eps_capacity_4"
    effect = pd.read_csv(root / "effect.csv")
    effect["truth.effect_id"] = "eps_battery_capacity_loss"
    contaminated = tmp_path / "effect.csv"
    effect.to_csv(contaminated, index=False)
    result = UnifiedDiagnosisService().diagnose_paths(
        case_id="anonymous",
        telemetry_path=contaminated,
        nominal_path=root / "nominal.csv",
        backend="composite",
    )
    assert result.status == UnifiedDiagnosisStatus.DATA_ERROR
    assert result.truth_used is False
