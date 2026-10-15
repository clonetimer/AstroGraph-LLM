from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

from astrograph.diagnosis.unified import (
    CATALOG_ENGINE_ID,
    OPEN_SET_ENGINE_ID,
    _fuse_single,
)
from astrograph.diagnosis import (
    EngineExecutionStatus,
    EngineRunRecord,
    UnifiedDiagnosisService,
    UnifiedDiagnosisStatus,
    engine_capabilities,
)


FIXTURES = Path(__file__).parent / "fixtures" / "unified_runtime"


def _model_record(engine_id: str, status: str) -> EngineRunRecord:
    return EngineRunRecord(
        engine_id=engine_id,
        eligible=True,
        execution_status=EngineExecutionStatus.EXECUTED,
        status=status,
    )


def test_single_trace_requires_both_models_to_agree_on_normal() -> None:
    status, conclusions, basis = _fuse_single([
        _model_record(CATALOG_ENGINE_ID, "NORMAL"),
        _model_record(OPEN_SET_ENGINE_ID, "NORMAL"),
    ])
    assert status == UnifiedDiagnosisStatus.NORMAL
    assert conclusions == []
    assert basis == "CATALOG_AND_OPEN_SET_NORMAL_AGREEMENT"


def test_single_trace_model_contract_failure_is_explicitly_insufficient() -> None:
    status, conclusions, basis = _fuse_single([
        _model_record(CATALOG_ENGINE_ID, "DATA_ERROR"),
        _model_record(OPEN_SET_ENGINE_ID, "DATA_ERROR"),
    ])
    assert status == UnifiedDiagnosisStatus.INSUFFICIENT_DATA
    assert conclusions == []
    assert basis == "MODEL_INPUT_CONTRACT_NOT_MET"


def _pair(case_id: str):
    root = FIXTURES / case_id
    return UnifiedDiagnosisService().diagnose_paths(
        case_id=case_id,
        nominal_path=root / "nominal.csv",
        telemetry_path=root / "effect.csv",
        backend="auto",
    )


def test_unified_capabilities_are_explicit_and_unique() -> None:
    capabilities = engine_capabilities()
    ids = [item.engine_id for item in capabilities]
    assert len(ids) == len(set(ids))
    assert "paired_physical_signatures" in ids
    assert "single_trace_open_set_model" in ids
    assert all(item.claim_boundary for item in capabilities)
    assert any(item.requires_model for item in capabilities)


def test_trusted_eps_pair_is_cross_engine_diagnosed() -> None:
    result = _pair("eps_capacity_a")
    assert result.backend == "composite"
    assert result.status == UnifiedDiagnosisStatus.DIAGNOSED
    assert result.primary_effect_id == "eps_battery_capacity_loss"
    assert result.resolution_basis == "CROSS_ENGINE_CORROBORATION"
    assert result.labels_used is False
    assert result.truth_used is False


def test_trusted_heater_pair_is_cross_engine_diagnosed() -> None:
    result = _pair("thermal_heater_a")
    assert result.status == UnifiedDiagnosisStatus.DIAGNOSED
    assert result.primary_effect_id == "thermal_heater_stuck_on"


def test_rw_pair_exposes_conflict_instead_of_false_diagnosis() -> None:
    result = _pair("rw_bearing_a")
    assert result.status == UnifiedDiagnosisStatus.CONFLICTING_EVIDENCE
    assert result.primary_effect_id is None
    assert result.conclusion_effect_ids == []
    assert any(
        run.engine_id == "paired_multifault_signatures"
        and run.detected_effect_ids == ["comm_data_downlink_link_loss", "payload_instrument_off"]
        for run in result.engine_runs
    )


def test_single_trace_physical_screening_is_not_promoted_without_corroboration() -> None:
    root = FIXTURES / "eps_capacity_a"
    result = UnifiedDiagnosisService().diagnose_paths(
        case_id="single",
        telemetry_path=root / "effect.csv",
        backend="auto",
    )
    assert result.status == UnifiedDiagnosisStatus.AMBIGUOUS
    assert result.primary_effect_id is None
    assert result.resolution_basis == "SINGLE_TRACE_SCREENING_REQUIRES_CORROBORATION"


def test_prohibited_truth_column_fails_closed(tmp_path: Path) -> None:
    frame = pd.DataFrame({"time_s": [0.0, 1.0], "battery_soc": [0.5, 0.4], "label.fault": [0, 1]})
    path = tmp_path / "leaky.csv"
    frame.to_csv(path, index=False)
    result = UnifiedDiagnosisService().diagnose_paths(case_id="leaky", telemetry_path=path)
    assert result.status == UnifiedDiagnosisStatus.DATA_ERROR
    assert result.resolution_basis == "DATA_QUALITY_GATE_FAILED"
    assert result.engine_runs == []
    assert result.data_quality[0].prohibited_columns == ["label.fault"]


def test_alias_normalization_and_nonmonotonic_time_gate(tmp_path: Path) -> None:
    valid = tmp_path / "valid.csv"
    pd.DataFrame({"timestamp": [0.0, 1.0], "battery_soc": [0.5, 0.4]}).to_csv(valid, index=False)
    valid_result = UnifiedDiagnosisService().diagnose_paths(case_id="alias", telemetry_path=valid)
    assert valid_result.data_quality[0].alias_mapping == {"timestamp": "time_s"}
    assert valid_result.status != UnifiedDiagnosisStatus.DATA_ERROR

    invalid = tmp_path / "invalid.csv"
    pd.DataFrame({"time_s": [1.0, 0.0], "battery_soc": [0.5, 0.4]}).to_csv(invalid, index=False)
    invalid_result = UnifiedDiagnosisService().diagnose_paths(case_id="time", telemetry_path=invalid)
    assert invalid_result.status == UnifiedDiagnosisStatus.DATA_ERROR
    assert any(item.code == "NON_MONOTONIC_TIME" for item in invalid_result.data_quality[0].findings)


def test_repeated_python_api_result_is_deterministic() -> None:
    first = _pair("eps_capacity_a").model_dump(mode="json")
    second = _pair("eps_capacity_a").model_dump(mode="json")
    assert first == second


def test_unified_cli_and_python_api_are_identical(tmp_path: Path) -> None:
    case_id = "eps_capacity_a"
    root = FIXTURES / case_id
    output = tmp_path / "result.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    env["ASTROGRAPH_STATE_DIR"] = str(tmp_path / "state")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "astrograph.cli",
            "diagnose",
            "--case",
            case_id,
            "--nominal",
            str(root / "nominal.csv"),
            "--telemetry",
            str(root / "effect.csv"),
            "--backend",
            "auto",
            "--output-json",
            str(output),
        ],
        cwd=Path(__file__).parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    cli_payload = json.loads(output.read_text(encoding="utf-8"))
    api_payload = _pair(case_id).model_dump(mode="json")
    assert cli_payload == api_payload


def test_batch_manifest_resolves_relative_paths_and_matches_api(tmp_path: Path) -> None:
    case_root = FIXTURES / "thermal_heater_a"
    nominal = tmp_path / "nominal.csv"
    effect = tmp_path / "effect.csv"
    nominal.write_bytes((case_root / "nominal.csv").read_bytes())
    effect.write_bytes((case_root / "effect.csv").read_bytes())
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": "astrograph-unified-batch-manifest.v1",
        "cases": [{
            "case_id": "thermal",
            "nominal": "nominal.csv",
            "telemetry": "effect.csv",
            "backend": "auto",
        }],
    }), encoding="utf-8")
    result = UnifiedDiagnosisService().diagnose_batch_path(manifest)
    direct = UnifiedDiagnosisService().diagnose_paths(
        case_id="thermal", nominal_path=nominal, telemetry_path=effect, backend="auto"
    )
    assert result.results == [direct]
    assert result.status_counts == {"DIAGNOSED": 1}


def test_frozen_unified_runtime_gate_is_pass() -> None:
    gate = json.loads((Path(__file__).parents[1] / "governance/evidence/0.7.0/unified_runtime_gate_decision.json").read_text(encoding="utf-8"))
    evaluation = json.loads((Path(__file__).parents[1] / "governance/evidence/0.7.0/unified_runtime_evaluation.json").read_text(encoding="utf-8"))
    assert gate["status"] == "PASS"
    assert gate["blockers"] == []
    assert evaluation["case_count"] == 6
    assert evaluation["exact_diagnosis_count"] == 4
    assert evaluation["safe_non_conclusion_count"] == 2
    assert evaluation["false_diagnosis_count"] == 0
    assert evaluation["data_error_count"] == 0
    assert evaluation["cli_api_mismatch_count"] == 0
    assert evaluation["batch_api_mismatch_count"] == 0
    assert evaluation["deterministic_replay_mismatch_count"] == 0
