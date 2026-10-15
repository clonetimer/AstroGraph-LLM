from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from astrograph.data_sources import InMemoryDataSource
from astrograph.diagnosis import (
    DatasetDiagnosisRequest,
    DatasetDiagnosisRunner,
    JsonDiagnosisRunStateStore,
    RW4_SINGLE_FAULT_DETECTOR_ID,
    detect_rw4_single_fault,
)
from astrograph.knowledge.repository import CsvKnowledgeRepository


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EDGES = PROJECT_ROOT / "configs" / "knowledge" / "reaction_wheel_edges.csv"


def _frame(fault_wheel: int | None, *, canonical: bool = False) -> pd.DataFrame:
    rows: dict[str, list[float | int | bool]] = {
        "time_s": list(range(50)),
        "label.fault_active": [False] * 15 + [fault_wheel is not None] * 35,
    }
    for wheel in range(1, 5):
        current: list[float] = []
        speed: list[float] = []
        for index in range(50):
            if index < 15:
                current_value = -(0.040 + 0.004 * index) + 0.002 * (wheel - 1)
                speed_value = 50.0 + wheel + 0.03 * index
            elif fault_wheel == wheel:
                current_value = 0.002 if index % 2 == 0 else -0.002
                speed_value = 50.0 + wheel + 0.03 * 14 + 0.005 * (index - 14)
            else:
                current_value = -(0.100 - 0.001 * (index - 15)) + 0.002 * (wheel - 1)
                speed_value = 50.0 + wheel + 0.03 * index
            current.append(current_value)
            speed.append(speed_value)
        if canonical:
            rows[f"sensor.adcs.rw.speed_rad_s_{wheel - 1}"] = speed
            rows[f"sensor.adcs.rw.motor_current_a_{wheel - 1}"] = current
        else:
            rows[f"rw{wheel}_speed_rad_s"] = speed
            rows[f"rw{wheel}_current_a"] = current
    return pd.DataFrame(rows)


def _runner(tmp_path: Path, frame: pd.DataFrame) -> DatasetDiagnosisRunner:
    source = InMemoryDataSource({"case": frame}, timestamp_column="time_s")
    repository = CsvKnowledgeRepository(EDGES, knowledge_version="rw4-single-fault-e0-v1")
    return DatasetDiagnosisRunner(
        source,
        repository,
        state_store=JsonDiagnosisRunStateStore(tmp_path / "state"),
        detector=detect_rw4_single_fault,
        detector_id=RW4_SINGLE_FAULT_DETECTOR_ID,
    )


@pytest.mark.parametrize("fault_wheel", [1, 2, 3, 4])
def test_rw4_detector_localizes_each_wheel_without_labels(
    tmp_path: Path, fault_wheel: int
) -> None:
    result = _runner(tmp_path, _frame(fault_wheel)).run(
        DatasetDiagnosisRequest(
            request_id=f"RW4-{fault_wheel}",
            entity_id="case",
        )
    )

    assert result.conclusion.level.value == "probable"
    assert result.conclusion.fault_id == f"ADCS_RW{fault_wheel}_TORQUE_AUTHORITY_LOSS"
    assert len(result.detections) == 1
    detection = result.detections[0]
    assert detection.observed_at == 15
    assert detection.metadata["wheel_ranking"][0]["wheel_id"] == f"RW{fault_wheel}"
    assert detection.metadata["gate_checks"] == {
        "current_step": True,
        "current_collapse": True,
        "step_to_pre_activity": True,
        "cross_wheel_dominance": True,
    }
    assert result.provenance["labels_used_by_detector"] is False
    assert result.provenance["truth_channels_used_by_detector"] is False
    assert result.provenance["detector_id"] == RW4_SINGLE_FAULT_DETECTOR_ID


def test_rw4_detector_supports_canonical_zero_based_field_names(tmp_path: Path) -> None:
    result = _runner(tmp_path, _frame(4, canonical=True)).run(
        DatasetDiagnosisRequest(request_id="RW4-CANONICAL", entity_id="case")
    )
    assert result.conclusion.fault_id == "ADCS_RW4_TORQUE_AUTHORITY_LOSS"
    assert result.detections[0].signal == "sensor.adcs.rw.motor_current_a_3"


def test_rw4_detector_caps_warmup_even_when_shared_minimum_is_large(tmp_path: Path) -> None:
    result = _runner(tmp_path, _frame(2)).run(
        DatasetDiagnosisRequest(
            request_id="RW4-WARMUP-CAP",
            entity_id="case",
            baseline_fraction=0.7,
            minimum_baseline_points=40,
        )
    )
    assert result.conclusion.fault_id == "ADCS_RW2_TORQUE_AUTHORITY_LOSS"
    assert result.detections[0].observed_at == 15
    assert result.observations[0].metadata["maximum_warmup_points"] == 5
    assert result.observations[0].metadata["warmup_points"] == 5


@pytest.mark.parametrize("case_index", range(12))
def test_rw4_nominal_controls_fail_closed_without_false_positive(
    tmp_path: Path, case_index: int
) -> None:
    frame = _frame(None)
    # Deterministic bounded perturbations preserve nominal continuity while
    # exercising different wheel/channel combinations.
    for wheel in range(1, 5):
        signal = f"rw{wheel}_current_a"
        frame[signal] = frame[signal] + [
            0.001 * (((index + wheel + case_index) % 5) - 2)
            for index in range(len(frame))
        ]
    result = _runner(tmp_path, frame).run(
        DatasetDiagnosisRequest(request_id=f"RW4-NOMINAL-{case_index}", entity_id="case")
    )
    assert result.conclusion.level.value == "unknown"
    assert result.conclusion.fault_id is None
    assert result.detections == []


def test_rw4_detector_rejects_incomplete_four_current_schema() -> None:
    frame = _frame(1).drop(columns=["rw4_current_a"])
    channels = [
        column
        for column in frame.columns
        if column != "time_s" and not column.startswith("label.")
    ]
    observations, detections, anomalies = detect_rw4_single_fault(
        frame,
        timestamp_column="time_s",
        channels=channels,
        baseline_fraction=0.2,
        z_threshold=6.0,
        minimum_baseline_points=10,
    )
    assert detections == []
    assert anomalies == []
    assert observations[0].metadata["status"] == "UNSUPPORTED_SCHEMA"
    assert observations[0].metadata["fail_closed"] is True
    assert observations[0].metadata["missing_current_wheels"] == ["RW4"]


def test_rw4_detector_rejects_duplicate_semantic_aliases() -> None:
    frame = _frame(1)
    frame["sensor.adcs.rw.motor_current_a_0"] = frame["rw1_current_a"]
    channels = [
        column
        for column in frame.columns
        if column != "time_s" and not column.startswith("label.")
    ]
    observations, detections, anomalies = detect_rw4_single_fault(
        frame,
        timestamp_column="time_s",
        channels=channels,
        baseline_fraction=0.2,
        z_threshold=6.0,
        minimum_baseline_points=10,
    )
    assert detections == []
    assert anomalies == []
    assert observations[0].metadata["status"] == "UNSUPPORTED_SCHEMA"
    assert observations[0].metadata["duplicate_aliases"] == [
        {
            "wheel_id": "RW1",
            "kind": "current_a",
            "selected": "rw1_current_a",
            "ignored": "sensor.adcs.rw.motor_current_a_0",
        }
    ]


def test_custom_detector_run_identity_is_isolated_from_generic_runner(tmp_path: Path) -> None:
    frame = _frame(2)
    source_a = InMemoryDataSource({"case": frame}, timestamp_column="time_s")
    source_b = InMemoryDataSource({"case": frame}, timestamp_column="time_s")
    repository = CsvKnowledgeRepository(EDGES, knowledge_version="rw4-single-fault-e0-v1")
    request = DatasetDiagnosisRequest(request_id="SAME-REQUEST", entity_id="case")
    generic = DatasetDiagnosisRunner(
        source_a,
        repository,
        state_store=JsonDiagnosisRunStateStore(tmp_path / "generic"),
    )
    specialized = DatasetDiagnosisRunner(
        source_b,
        repository,
        state_store=JsonDiagnosisRunStateStore(tmp_path / "specialized"),
        detector=detect_rw4_single_fault,
        detector_id=RW4_SINGLE_FAULT_DETECTOR_ID,
    )
    assert generic.identity(request)[0] != specialized.identity(request)[0]


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    current = env.get("PYTHONPATH", "")
    source = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = source if not current else source + os.pathsep + current
    return env


def test_cli_selects_rw4_detector_and_resume_auto_recovers_it(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _frame(3).to_csv(dataset / "case.csv", index=False)
    state_root = tmp_path / "state"
    output = tmp_path / "output"
    command = [
        sys.executable,
        "-m",
        "astrograph.cli",
        "diagnosis",
        "run",
        "--source-type",
        "local_file",
        "--root",
        str(dataset),
        "--entity",
        "case",
        "--request-id",
        "CLI-RW4",
        "--detector",
        "rw4-single-fault",
        "--edges",
        str(EDGES),
        "--policy",
        str(PROJECT_ROOT / "configs" / "diagnosis.json"),
        "--state-root",
        str(state_root),
        "--audit",
        str(tmp_path / "audit.jsonl"),
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=_subprocess_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["conclusion"]["fault_id"] == "ADCS_RW3_TORQUE_AUTHORITY_LOSS"

    resumed = subprocess.run(
        [
            sys.executable,
            "-m",
            "astrograph.cli",
            "diagnosis",
            "resume",
            "--run-id",
            payload["run_id"],
            "--source-type",
            "local_file",
            "--root",
            str(dataset),
            "--edges",
            str(EDGES),
            "--policy",
            str(PROJECT_ROOT / "configs" / "diagnosis.json"),
            "--state-root",
            str(state_root),
            "--audit",
            str(tmp_path / "audit.jsonl"),
            "--output",
            str(tmp_path / "replayed"),
        ],
        cwd=PROJECT_ROOT,
        env=_subprocess_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert resumed.returncode == 0, resumed.stderr
    replay = json.loads(resumed.stdout)
    assert replay["conclusion"]["fault_id"] == "ADCS_RW3_TORQUE_AUTHORITY_LOSS"
    assert replay["replayed"] is True


    wrong_detector = subprocess.run(
        [
            sys.executable,
            "-m",
            "astrograph.cli",
            "diagnosis",
            "resume",
            "--run-id",
            payload["run_id"],
            "--source-type",
            "local_file",
            "--root",
            str(dataset),
            "--detector",
            "robust-baseline",
            "--edges",
            str(EDGES),
            "--policy",
            str(PROJECT_ROOT / "configs" / "diagnosis.json"),
            "--state-root",
            str(state_root),
            "--audit",
            str(tmp_path / "audit.jsonl"),
            "--output",
            str(tmp_path / "wrong_detector"),
        ],
        cwd=PROJECT_ROOT,
        env=_subprocess_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert wrong_detector.returncode != 0
    assert "identity is inconsistent" in wrong_detector.stderr
