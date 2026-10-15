from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from astrograph.data_sources import (
    DataSourceProbeError,
    DataSourceType,
    DatasetSourceConfig,
    InMemoryDataSource,
    LocalFileDataSource,
    ProbeOptions,
    SimulationDataSource,
    probe_data_source,
)
from astrograph.simulation import (
    FaultPatternSpec,
    FaultPatternType,
    SyntheticDatasetSpec,
    generate_synthetic_dataset,
)


def _source(tmp_path: Path) -> tuple[Path, LocalFileDataSource]:
    root = tmp_path / "private_esa_adb"
    root.mkdir()
    pd.DataFrame(
        {
            "timestamp": [0, 1, 2],
            "channel_name": ["SECRET_RAW_A", "SECRET_RAW_B", "SECRET_RAW_C"],
            "numeric": [1.0, 2.0, 3.0],
            "label": ["normal", "normal", "fault"],
        }
    ).to_csv(root / "series.csv", index=False)
    source = LocalFileDataSource(
        DatasetSourceConfig(
            source_type=DataSourceType.LOCAL_FILE,
            root=str(root),
            timestamp_column="timestamp",
            label_columns=["label"],
        )
    )
    return root, source


def test_probe_emits_only_redacted_profiles_by_default(tmp_path: Path) -> None:
    root, source = _source(tmp_path)
    output = tmp_path / "evidence"
    artifacts = probe_data_source(
        source,
        output,
        options=ProbeOptions(max_rows_per_entity=3, hash_mode="metadata"),
    )
    assert artifacts.output_dir == "<local_evidence>"
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in output.iterdir()
        if path.is_file()
    )
    assert "SECRET_RAW_A" not in combined
    assert str(root.resolve()) not in combined
    manifest = json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["raw_data_included"] is False
    assert manifest["absolute_paths_included"] is False
    assert manifest["root"] == "${ASTROGRAPH_ESA_ADB_ROOT}"


def test_probe_refuses_to_write_into_dataset_root(tmp_path: Path) -> None:
    root, source = _source(tmp_path)
    with pytest.raises(ValueError, match="must not be written"):
        probe_data_source(source, root / "probe_output")


def test_probe_marks_an_explicit_in_memory_source_as_virtual(tmp_path: Path) -> None:
    source = InMemoryDataSource(
        {"sample": pd.DataFrame({"timestamp": [0, 1], "value": [1.0, 2.0]})},
        timestamp_column="timestamp",
    )

    probe_data_source(source, tmp_path / "evidence")

    profile = json.loads((tmp_path / "evidence" / "file_profile.json").read_text(encoding="utf-8"))
    assert profile["files"][0]["fingerprint_status"] == "NOT_APPLICABLE_VIRTUAL_SOURCE"


def test_file_fingerprint_failure_is_not_downgraded_to_virtual(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, source = _source(tmp_path)

    def denied(*args, **kwargs):
        raise PermissionError("simulated permission denial")

    monkeypatch.setattr(source, "resolve_under", denied)
    with pytest.raises(DataSourceProbeError) as captured:
        probe_data_source(source, tmp_path / "evidence")

    assert captured.value.entity_id == "series"
    assert isinstance(captured.value.__cause__, PermissionError)


def test_profile_driven_generator_is_deterministic_and_injects_faults(tmp_path: Path) -> None:
    profile = {
        "schema_version": "astrograph-dataset-schema-profile.v1",
        "timestamp_column": "timestamp",
        "label_columns": ["label"],
        "entities": [
            {
                "entity_id": "sample",
                "timestamp_column": "timestamp",
                "label_columns": ["label"],
                "columns": {
                    "timestamp": {"dtype": "float64"},
                    "sensor.eps.battery_voltage_v": {"dtype": "float64"},
                    "sensor.eps.battery_current_a": {"dtype": "float64"},
                    "label": {"dtype": "bool"},
                },
            }
        ],
    }
    spec = SyntheticDatasetSpec(
        dataset_id="R4F3-SYNTH",
        entity_id="case_a",
        row_count=100,
        seed=42,
        patterns=[
            FaultPatternSpec(
                pattern=FaultPatternType.SHIFT,
                channel="sensor.eps.battery_voltage_v",
                start_fraction=0.6,
                end_fraction=0.9,
                magnitude=20.0,
            ),
            FaultPatternSpec(
                pattern=FaultPatternType.DUPLICATE_TIMESTAMP,
                start_fraction=0.7,
            ),
        ],
    )
    first = generate_synthetic_dataset(spec, tmp_path / "out1", schema_profile=profile)
    second = generate_synthetic_dataset(spec, tmp_path / "out2", schema_profile=profile)
    assert first["telemetry_sha256"] == second["telemetry_sha256"]
    assert first["source_profile_sha256"] is not None
    assert len(first["patterns"]) == 2

    source = SimulationDataSource(tmp_path / "out1")
    with source:
        assert [item.entity_id for item in source.list_entities()] == ["case_a"]
        frame = source.read_range("case_a")
        labels = source.read_labels("case_a")
        assert len(frame) == 100
        assert bool(labels["label"].any())
        assert frame["timestamp"].duplicated().any()


def test_generator_can_create_missing_channel_boundary_case(tmp_path: Path) -> None:
    spec = SyntheticDatasetSpec(
        dataset_id="MISSING",
        row_count=20,
        channels=["channel_a", "channel_b"],
        patterns=[
            FaultPatternSpec(
                pattern=FaultPatternType.MISSING_CHANNEL,
                channel="channel_b",
            )
        ],
    )
    manifest = generate_synthetic_dataset(spec, tmp_path / "out")
    assert "channel_b" not in manifest["channels"]
    frame = pd.read_csv(tmp_path / "out" / "synthetic_case" / "telemetry.csv")
    assert "channel_b" not in frame.columns
