from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from astrograph.onboarding import OnboardingStatus, build_dataset_onboarding_report


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols" / "phm_project_a"
FAULT_CATALOG = ROOT / "configs/fault_modes/lab_fault_catalog.json"


def _registered_fault_ids() -> list[str]:
    payload = json.loads(FAULT_CATALOG.read_text(encoding="utf-8"))
    return sorted(item["fault_id"] for item in payload["fault_modes"])


def _required_columns() -> list[str]:
    import csv

    with (PROTOCOL / "dataset_contract.json").open("r", encoding="utf-8-sig") as handle:
        return list(json.load(handle)["raw_telemetry_fields"])


def _write_dataset(root: Path, frame: pd.DataFrame) -> None:
    root.mkdir(parents=True)
    frame.to_csv(root / "telemetry.csv", index=False)


def test_onboarding_allows_only_gap_report_for_unlabeled_mappable_dataset(tmp_path: Path) -> None:
    columns = _required_columns()
    frame = pd.DataFrame({column: [0.0, 1.0, 2.0] for column in columns})
    frame["time_s"] = [0.0, 10.0, 20.0]
    dataset = tmp_path / "unlabeled"
    _write_dataset(dataset, frame)

    report = build_dataset_onboarding_report(
        dataset_root=dataset,
        protocol_root=PROTOCOL,
        registered_fault_ids=_registered_fault_ids(),
    )

    assert report.status == OnboardingStatus.ONBOARDING_ONLY
    assert report.schema_mapping is not None
    assert report.schema_mapping.required_field_coverage == 1.0
    assert report.label_coverage is not None
    assert report.label_coverage.status.value == "ABSENT"
    assert report.capability.diagnosis_permission == "CANDIDATE_EXPLANATION_AND_GAP_REPORT_ONLY"
    assert "LABELS_ABSENT_ACCURACY_NOT_EVALUABLE" in report.warnings


def test_onboarding_blocks_unknown_label_values(tmp_path: Path) -> None:
    columns = _required_columns()
    frame = pd.DataFrame({column: [0.0, 1.0] for column in columns})
    frame["time_s"] = [0.0, 10.0]
    frame["fault_id"] = ["NOT_REGISTERED_FAULT", "NORMAL"]
    dataset = tmp_path / "unknown-label"
    _write_dataset(dataset, frame)

    report = build_dataset_onboarding_report(
        dataset_root=dataset,
        protocol_root=PROTOCOL,
        registered_fault_ids=_registered_fault_ids(),
    )

    assert report.status == OnboardingStatus.UNKNOWN
    assert "LABELS_REFERENCE_UNREGISTERED_FAULTS" in report.blocking_reasons
    assert report.capability.diagnosis_permission == "UNKNOWN_ONLY"
    assert report.label_coverage is not None
    assert report.label_coverage.unknown_label_values == ["NOT_REGISTERED_FAULT"]


def test_onboarding_blocks_low_required_field_coverage(tmp_path: Path) -> None:
    frame = pd.DataFrame({"time_s": [0.0, 10.0], "some.unmapped.field": [1.0, 2.0]})
    dataset = tmp_path / "low-coverage"
    _write_dataset(dataset, frame)

    report = build_dataset_onboarding_report(
        dataset_root=dataset,
        protocol_root=PROTOCOL,
        registered_fault_ids=_registered_fault_ids(),
    )

    assert report.status == OnboardingStatus.UNKNOWN
    assert "REQUIRED_FIELD_COVERAGE_BELOW_THRESHOLD" in report.blocking_reasons
    assert "NO_MAPPED_MODEL_INPUT_FIELDS" in report.blocking_reasons
    assert "EXTRA_COLUMNS_NOT_IN_PROTOCOL" in report.capability.ood_risk_flags


def test_onboarding_ready_for_seed_only_replay_when_labels_are_known(tmp_path: Path) -> None:
    columns = _required_columns()
    frame = pd.DataFrame({column: [0.0, 1.0] for column in columns})
    frame["time_s"] = [0.0, 10.0]
    frame["fault_id"] = ["THERMAL_HEATER_STUCK_ON", "NORMAL"]
    dataset = tmp_path / "known-labels"
    _write_dataset(dataset, frame)

    report = build_dataset_onboarding_report(
        dataset_root=dataset,
        protocol_root=PROTOCOL,
        registered_fault_ids=_registered_fault_ids(),
    )

    assert report.status == OnboardingStatus.READY_FOR_SEED_ONLY_REPLAY
    assert report.capability.diagnosis_permission == "SEED_ONLY_REPLAY_ALLOWED"
    assert report.capability.kg_covered_fault_ids == ["THERMAL_HEATER_STUCK_ON"]


def test_onboarding_marks_flight_data_axis_when_flight_metadata_are_present(tmp_path: Path) -> None:
    columns = _required_columns()
    frame = pd.DataFrame({column: [0.0, 1.0] for column in columns})
    frame["time_s"] = [0.0, 10.0]
    frame["mission_elapsed_time"] = [0.0, 10.0]
    frame["flight_id"] = ["FLIGHT-A", "FLIGHT-A"]
    frame["spacecraft_id"] = ["SC-A", "SC-A"]
    dataset = tmp_path / "flight-metadata"
    _write_dataset(dataset, frame)

    report = build_dataset_onboarding_report(
        dataset_root=dataset,
        protocol_root=PROTOCOL,
        registered_fault_ids=_registered_fault_ids(),
    )

    assert report.status == OnboardingStatus.ONBOARDING_ONLY
    assert "flight_data" in report.capability.supported_axes
    assert "flight_data" not in report.capability.unsupported_axes
