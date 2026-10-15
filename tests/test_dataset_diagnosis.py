from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from astrograph.data_sources import InMemoryDataSource
from astrograph.diagnosis import (
    DatasetDiagnosisRequest,
    DatasetDiagnosisRunner,
    JsonDiagnosisRunStateStore,
)
from astrograph.knowledge.repository import CsvKnowledgeRepository


def _repository(tmp_path: Path) -> CsvKnowledgeRepository:
    path = tmp_path / "edges.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "fault_id",
                "fault_name",
                "anomaly_id",
                "anomaly_name",
                "weight",
                "edge_role",
                "evidence_level",
                "provisional",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "fault_id": "EPS_BATTERY_VOLTAGE_FAULT",
                "fault_name": "Battery voltage fault",
                "anomaly_id": "ANOM_BATTERY_VOLTAGE_DEVIATION",
                "anomaly_name": "Battery voltage deviation",
                "weight": "1.0",
                "edge_role": "direct",
                "evidence_level": "E0",
                "provisional": "true",
            }
        )
    return CsvKnowledgeRepository(path)


def _frame(signal: str = "sensor.eps.battery_voltage_v") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": list(range(100)),
            signal: [0.0] * 50 + [100.0] * 50,
            "label.fault_active": [False] * 50 + [True] * 50,
        }
    )


def test_dataset_diagnosis_is_layered_idempotent_and_does_not_use_labels(tmp_path: Path) -> None:
    source = InMemoryDataSource({"case": _frame()}, timestamp_column="time_s")
    store = JsonDiagnosisRunStateStore(tmp_path / "state")
    runner = DatasetDiagnosisRunner(source, _repository(tmp_path), state_store=store)
    request = DatasetDiagnosisRequest(
        request_id="REQ-1",
        entity_id="case",
        baseline_fraction=0.3,
        z_threshold=6.0,
    )

    first = runner.run(request, output_dir=tmp_path / "output")
    second = runner.run(request, output_dir=tmp_path / "output")
    replay_output = tmp_path / "replayed_output"
    replayed_to_new_output = runner.run(request, output_dir=replay_output)

    assert first.conclusion.level.value == "probable"
    assert first.conclusion.fault_id == "EPS_BATTERY_VOLTAGE_FAULT"
    assert len(first.observations) == 1
    assert len(first.detections) == 1
    assert first.detections[0].mapped is True
    assert first.provenance["labels_used_by_detector"] is False
    assert first.provenance["truth_channels_used_by_detector"] is False
    assert second.replayed is True
    assert second.diagnosis_id == first.diagnosis_id
    assert replayed_to_new_output.replayed is True
    assert (replay_output / "diagnosis.json").is_file()
    assert (replay_output / "report.md").is_file()
    assert (tmp_path / "output" / "diagnosis.json").is_file()
    assert (tmp_path / "output" / "run_state.json").is_file()

    state_text = store.path_for(first.run_id).read_text(encoding="utf-8")
    assert '"label.fault_active": [false' not in state_text.lower()
    state = store.get(first.run_id)
    assert state is not None and state.status == "COMPLETED"
    assert state.completed_stages == ["DATA_READ", "DETECTED", "DIAGNOSED", "COMPLETED"]


def test_unmapped_signal_fails_closed_as_unknown(tmp_path: Path) -> None:
    source = InMemoryDataSource(
        {"case": _frame("unknown_channel")}, timestamp_column="time_s"
    )
    runner = DatasetDiagnosisRunner(
        source,
        _repository(tmp_path),
        state_store=JsonDiagnosisRunStateStore(tmp_path / "state"),
    )
    result = runner.run(
        DatasetDiagnosisRequest(request_id="REQ-UNKNOWN", entity_id="case")
    )

    assert result.detections
    assert result.detections[0].mapped is False
    assert result.conclusion.level.value == "unknown"
    assert result.conclusion.fault_id is None


class _FlakySource(InMemoryDataSource):
    def __init__(self, frames):
        super().__init__(frames, timestamp_column="time_s")
        self.fail_once = True

    def read_range(self, *args, **kwargs):
        if self.fail_once:
            self.fail_once = False
            raise OSError("transient local read failure")
        return super().read_range(*args, **kwargs)


def test_failed_data_read_can_resume_without_duplicate_diagnosis(tmp_path: Path) -> None:
    source = _FlakySource({"case": _frame()})
    store = JsonDiagnosisRunStateStore(tmp_path / "state")
    runner = DatasetDiagnosisRunner(source, _repository(tmp_path), state_store=store)
    request = DatasetDiagnosisRequest(request_id="REQ-RESUME", entity_id="case")
    run_id, _, _ = runner.identity(request)

    with pytest.raises(OSError, match="transient"):
        runner.run(request)
    failed = store.get(run_id)
    assert failed is not None
    assert failed.status == "FAILED"
    assert failed.failed_stage == "DATA_READ"

    result = runner.resume(run_id)
    assert result.conclusion.level.value == "probable"
    completed = store.get(run_id)
    assert completed is not None and completed.status == "COMPLETED"


def test_state_store_rejects_path_traversal_run_ids(tmp_path: Path) -> None:
    store = JsonDiagnosisRunStateStore(tmp_path / "state")
    with pytest.raises(ValueError, match="unsafe run_id"):
        store.get("../outside")
