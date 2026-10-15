from __future__ import annotations

import pandas as pd

from astrograph.diagnosis.sat_sim_paired import (
    PairedDiagnosisStatus,
    SatSimPairedDiagnosisEngine,
)


def _frames(**effect_values: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    length = len(next(iter(effect_values.values())))
    nominal = {"time_s": list(range(length))}
    effect = {"time_s": list(range(length))}
    for field, values in effect_values.items():
        nominal[field] = [0.0] * length
        effect[field] = values
    return pd.DataFrame(nominal), pd.DataFrame(effect)


def test_rejects_label_or_truth_columns() -> None:
    nominal, effect = _frames(**{"payload.generated_bps": [0.0, 0.0]})
    nominal["label.answer"] = [0, 0]
    effect["label.answer"] = [1, 1]
    result = SatSimPairedDiagnosisEngine().diagnose_frames(
        case_id="opaque", backend="unified", nominal=nominal, effect=effect
    )
    assert result.status == PairedDiagnosisStatus.DATA_ERROR
    assert result.prohibited_columns_used == ["label.answer"]
    assert result.labels_used is False
    assert result.truth_used is False


def test_identical_pair_is_not_diagnosed() -> None:
    nominal, _ = _frames(**{
        "battery_storage_j": [100.0, 90.0],
        "power_bus_available": [1.0, 1.0],
        "payload_generated_bps": [1_000_000.0, 1_000_000.0],
    })
    result = SatSimPairedDiagnosisEngine().diagnose_frames(
        case_id="opaque", backend="composite", nominal=nominal, effect=nominal.copy()
    )
    assert result.status != PairedDiagnosisStatus.DIAGNOSED
    assert result.conclusion_effect_id is None


def test_compound_signature_dominates_nested_component_signatures() -> None:
    nominal = pd.DataFrame({
        "time_s": [0.0, 1.0],
        "battery_storage_j": [500_000.0, 500_000.0],
        "net_power_w": [100.0, 100.0],
        "fuel_mass_flow_kg_s": [0.0002, 0.0002],
        "propellant_remaining_kg": [10.0, 9.99],
    })
    effect = pd.DataFrame({
        "time_s": [0.0, 1.0],
        "battery_storage_j": [500_000.0, 380_000.0],
        "net_power_w": [80.0, 80.0],
        "fuel_mass_flow_kg_s": [0.00012, 0.00012],
        "propellant_remaining_kg": [10.0, 9.995],
    })
    result = SatSimPairedDiagnosisEngine().diagnose_frames(
        case_id="opaque", backend="composite", nominal=nominal, effect=effect
    )
    assert result.status == PairedDiagnosisStatus.DIAGNOSED
    assert result.conclusion_effect_id == "combined_degradation"
    assert result.resolution_basis == "UNIQUE_COMPOUND_SIGNATURE"


def test_case_id_does_not_influence_prediction() -> None:
    nominal, effect = _frames(**{
        "payload.generated_bps": [0.0, -2_000_000.0],
        "payload.active": [0.0, -1.0],
        "comm.active": [0.0, 0.0],
    })
    engine = SatSimPairedDiagnosisEngine()
    first = engine.diagnose_frames(case_id="CASE-001", backend="unified", nominal=nominal, effect=effect)
    second = engine.diagnose_frames(case_id="totally-random-name", backend="unified", nominal=nominal, effect=effect)
    assert first.conclusion_effect_id == second.conclusion_effect_id
    assert first.status == second.status
    assert first.candidates == second.candidates


def test_cli_function_writes_paired_diagnosis(tmp_path) -> None:
    import argparse
    from astrograph.cli import _diagnosis_sat_sim_paired

    nominal = pd.DataFrame({
        "time_s": [0.0, 1.0],
        "payload.generated_bps": [2_000_000.0, 2_000_000.0],
        "payload.active": [1.0, 1.0],
        "comm.active": [1.0, 1.0],
    })
    effect = pd.DataFrame({
        "time_s": [0.0, 1.0],
        "payload.generated_bps": [0.0, 0.0],
        "payload.active": [0.0, 0.0],
        "comm.active": [1.0, 1.0],
    })
    nominal_path = tmp_path / "nominal.csv"
    effect_path = tmp_path / "effect.csv"
    output_path = tmp_path / "diagnosis.json"
    nominal.to_csv(nominal_path, index=False)
    effect.to_csv(effect_path, index=False)
    rc = _diagnosis_sat_sim_paired(argparse.Namespace(
        case_id="opaque", backend="unified", nominal=nominal_path,
        telemetry=effect_path, output=output_path,
    ))
    assert rc == 0
    assert output_path.exists()
    payload = __import__("json").loads(output_path.read_text(encoding="utf-8"))
    assert payload["conclusion_effect_id"] == "payload_instrument_off"
