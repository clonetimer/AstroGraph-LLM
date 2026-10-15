from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from astrograph.diagnosis.sat_sim_multifault_single import (
    SingleTraceMultiFaultStatus,
    diagnose_single_trace_multifault,
)

DATA_ROOT = Path(__file__).resolve().parents[1] / "tests/fixtures/multifault"


def _cases(campaign: str):
    root = DATA_ROOT / campaign
    truth = json.loads((root / 'hidden_truth.json').read_text(encoding='utf-8'))
    for case_id, expected in truth.items():
        case_dir = next((root / 'public').glob(f'*/{case_id}'))
        yield case_id, expected, case_dir


@pytest.mark.parametrize('campaign', ['development', 'holdout'])
def test_single_trace_multifault_exact_on_development_and_holdout(campaign: str) -> None:
    for case_id, expected, case_dir in _cases(campaign):
        nominal = diagnose_single_trace_multifault(case_id=f'{case_id}-n', telemetry_csv=case_dir/'nominal.csv')
        effect = diagnose_single_trace_multifault(case_id=f'{case_id}-e', telemetry_csv=case_dir/'effect.csv')
        assert nominal.status is SingleTraceMultiFaultStatus.NORMAL
        assert nominal.detected_effect_ids == []
        assert effect.status is SingleTraceMultiFaultStatus.KNOWN_FAULT
        assert set(effect.detected_effect_ids) == set(expected['effect_ids'])
        assert effect.truth_used is False
        assert effect.labels_used is False


def test_single_trace_multifault_rejects_prohibited_columns(tmp_path: Path) -> None:
    frame = pd.DataFrame({'battery_soc':[0.9], 'label.answer':['x']})
    path = tmp_path/'bad.csv'; frame.to_csv(path,index=False)
    result = diagnose_single_trace_multifault(case_id='bad', telemetry_csv=path)
    assert result.status is SingleTraceMultiFaultStatus.DATA_ERROR
    assert result.prohibited_columns_used == ['label.answer']


def test_single_trace_multifault_requires_telemetry_coverage(tmp_path: Path) -> None:
    path=tmp_path/'thin.csv'; pd.DataFrame({'battery_soc':[0.8]}).to_csv(path,index=False)
    result=diagnose_single_trace_multifault(case_id='thin',telemetry_csv=path)
    assert result.status is SingleTraceMultiFaultStatus.INSUFFICIENT_EVIDENCE


def test_single_trace_rw_rule_does_not_flag_high_maneuver_nominal(tmp_path: Path) -> None:
    frame = pd.DataFrame({
        "rw_motor_torque_nm_0": [0.7769, 0.7769],
        "rw_motor_torque_nm_1": [1.0, 1.0],
        "rw_motor_torque_nm_2": [1.0, 1.0],
        "rw_speed_rad_s_0": [400.0, 410.0],
        "battery_soc": [0.60, 0.60],
        "net_power_w": [-165.0, -165.0],
        "thermal_temp_c": [23.0, 24.0],
        "heater_eps_load_w": [0.0, 0.0],
        "payload_generated_bps": [2_500_000.0, 2_500_000.0],
        "fuel_mass_flow_kg_s": [0.00041, 0.00041],
        "heater_active_count": [0.0, 0.0],
        "downlink_requested_rate_bps": [1_500_000.0, 1_500_000.0],
        "data_storage_bits": [110_000_000.0, 110_000_000.0],
    })
    path=tmp_path/"nominal.csv"; frame.to_csv(path,index=False)
    result=diagnose_single_trace_multifault(case_id="nominal",telemetry_csv=path)
    assert "adcs_rw_jamming" not in result.detected_effect_ids


def test_single_trace_primary_is_selected_by_physical_strength() -> None:
    case_id, expected, case_dir = next(
        item for item in _cases('holdout') if item[1]['scenario_id'] == 'comm_storage_downlink_late'
    )
    result = diagnose_single_trace_multifault(case_id=case_id, telemetry_csv=case_dir/'effect.csv')
    assert result.primary_selection_semantics == 'STRONGEST_NORMALIZED_PHYSICAL_SIGNATURE_NOT_CAUSAL_ROOT'
    assert result.primary_effect_id in result.detected_effect_ids
    assert result.secondary_effect_ids == [effect for effect in result.detected_effect_ids if effect != result.primary_effect_id]
    assert set(result.effect_scores) == set(result.detected_effect_ids)


def test_single_trace_power_bus_collapse_is_root_not_three_independent_faults() -> None:
    case_id,expected,case_dir=next(
        item for item in _cases('holdout') if item[1]['scenario_id']=='eps_open_to_payload_comm'
    )
    result=diagnose_single_trace_multifault(case_id=case_id,telemetry_csv=case_dir/'effect.csv')
    assert result.detected_effect_ids == ['eps_battery_open_circuit']
    assert set(result.propagation_paths)=={
        'eps_open->power_bus->payload','eps_open->power_bus->downlink'
    }
    assert {'EPS','PAYLOAD','COMM_DATA'} <= set(result.affected_subsystems)
