from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from astrograph.data_sources import (
    DataSourceType,
    DatasetSourceConfig,
    ESAADBDataSource,
    LocalFileDataSource,
    SimulationDataSource,
    validate_data_source,
)


def _simulation_root(tmp_path: Path) -> Path:
    root = tmp_path / "simulation"
    run = root / "fault_case"
    run.mkdir(parents=True)
    pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0, 3.0],
            "sensor.eps.battery_voltage_v": [10.0, 10.1, 9.9, 7.0],
            "truth.eps.battery_voltage_v": [10.0, 10.0, 10.0, 7.0],
            "label.fault_active": [False, False, False, True],
        }
    ).to_csv(run / "telemetry.csv", index=False)
    (run / "run_manifest.json").write_text(
        json.dumps(
            {
                "scenario": "fault_case",
                "row_count": 4,
                "duration_s": 3.0,
                "seed": 7,
                "claim_boundary": "test only",
            }
        ),
        encoding="utf-8",
    )
    return root


def test_simulation_source_excludes_truth_and_labels_from_model_channels(tmp_path: Path) -> None:
    source = SimulationDataSource(_simulation_root(tmp_path))
    with source:
        entities = source.list_entities()
        assert [item.entity_id for item in entities] == ["fault_case"]
        assert entities[0].row_count == 4
        assert source.list_channels("fault_case") == ["sensor.eps.battery_voltage_v"]

        frame = source.read_range("fault_case", 1.0, 2.0)
        assert list(frame.columns) == ["time_s", "sensor.eps.battery_voltage_v"]
        assert frame["time_s"].tolist() == [1.0, 2.0]

        labels = source.read_labels("fault_case")
        assert list(labels.columns) == [
            "time_s",
            "truth.eps.battery_voltage_v",
            "label.fault_active",
        ]
        assert labels["label.fault_active"].tolist() == [False, False, False, True]

    with pytest.raises(RuntimeError, match="not open"):
        source.list_entities()


def test_file_source_rejects_write_mode_and_root_escape(tmp_path: Path) -> None:
    root = _simulation_root(tmp_path)
    config = DatasetSourceConfig(
        source_type=DataSourceType.LOCAL_FILE,
        root=str(root),
        read_only=False,
    )
    with pytest.raises(ValueError, match="read-only"):
        LocalFileDataSource(config).open()

    with pytest.raises(ValueError, match="escapes"):
        LocalFileDataSource.resolve_under(root.resolve(), "../outside")


def test_csv_iteration_is_chunked_and_bounded(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    pd.DataFrame({"time_s": range(10), "signal": range(10)}).to_csv(
        root / "data.csv", index=False
    )
    source = LocalFileDataSource(
        DatasetSourceConfig(
            source_type=DataSourceType.LOCAL_FILE,
            root=str(root),
            timestamp_column="time_s",
        )
    )
    with source:
        entity_id = source.list_entities()[0].entity_id
        assert len(source.read_range(entity_id, limit=3)) == 3
        chunks = list(source.iter_range(entity_id, 2, 7, chunk_size=3))
        assert [len(item) for item in chunks] == [1, 3, 2]
        assert pd.concat(chunks)["time_s"].tolist() == [2, 3, 4, 5, 6, 7]


def test_esa_adapter_is_explicitly_local_and_structural(tmp_path: Path) -> None:
    root = tmp_path / "esa"
    root.mkdir()
    pd.DataFrame({"timestamp": [1, 2], "channel_a": [0.1, 0.2]}).to_csv(
        root / "test.csv", index=False
    )
    # Metadata JSON must not be mistaken for a telemetry entity.
    (root / "metadata.json").write_text('{"name":"dataset"}', encoding="utf-8")
    source = ESAADBDataSource(root)
    with source:
        assert len(source.list_entities()) == 1
        metadata = source.get_metadata()
        assert metadata.source_type == DataSourceType.ESA_ADB
        assert metadata.read_only is True
        assert metadata.metadata["local_acceptance_required"] is True
        assert metadata.root_token == "${ASTROGRAPH_ESA_ADB_ROOT}"


def test_validation_reports_duplicate_time_and_missing_values_without_crashing(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    pd.DataFrame(
        {"time_s": [0, 1, 1, 0], "signal": [1.0, None, 2.0, 3.0]}
    ).to_csv(root / "data.csv", index=False)
    source = LocalFileDataSource(
        DatasetSourceConfig(
            source_type=DataSourceType.LOCAL_FILE,
            root=str(root),
            timestamp_column="time_s",
        )
    )
    report = validate_data_source(source)
    codes = {item.code for item in report.issues}
    assert report.status == "WARNING"
    assert {"MISSING_VALUES", "DUPLICATE_TIMESTAMPS", "TIMESTAMPS_NOT_MONOTONIC"} <= codes


def test_probe_profile_applies_timestamp_and_labels_per_entity(tmp_path: Path) -> None:
    root = tmp_path / "heterogeneous"
    root.mkdir()
    pd.DataFrame(
        {"clock_a": [1, 2], "signal_a": [0.1, 0.2], "class_a": [0, 1]}
    ).to_csv(root / "a.csv", index=False)
    pd.DataFrame(
        {"clock_b": [3, 4], "signal_b": [0.3, 0.4], "class_b": ["N", "F"]}
    ).to_csv(root / "b.csv", index=False)
    profile = tmp_path / "schema_profile.json"
    profile.write_text(
        json.dumps(
            {
                "schema_version": "astrograph-dataset-schema-profile.v1",
                "entities": [
                    {
                        "entity_id": "a",
                        "timestamp_column": "clock_a",
                        "label_columns": ["class_a"],
                    },
                    {
                        "entity_id": "b",
                        "timestamp_column": "clock_b",
                        "label_columns": ["class_b"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    source = ESAADBDataSource(root, profile_path=profile)
    with source:
        entities = {item.entity_id: item for item in source.list_entities()}
        assert entities["a"].timestamp_column == "clock_a"
        assert entities["b"].timestamp_column == "clock_b"
        assert entities["a"].label_columns == ["class_a"]
        assert entities["b"].label_columns == ["class_b"]
        assert source.list_channels("a") == ["signal_a"]
        assert source.list_channels("b") == ["signal_b"]


def test_fault_named_telemetry_files_are_not_misclassified_as_label_files(tmp_path: Path) -> None:
    root = tmp_path / "mixed"
    root.mkdir()
    pd.DataFrame({"time_s": [0, 1], "normal_signal": [1.0, 1.1]}).to_csv(
        root / "normal.csv", index=False
    )
    pd.DataFrame({"time_s": [0, 1], "fault_signal": [2.0, 2.1]}).to_csv(
        root / "reaction_wheel_fault.csv", index=False
    )
    source = ESAADBDataSource(root)
    with source:
        assert [item.entity_id for item in source.list_entities()] == [
            "normal",
            "reaction_wheel_fault",
        ]


def _raw_esa_mission_root(tmp_path: Path) -> Path:
    root = tmp_path / "ESA-Mission1"
    (root / "channels").mkdir(parents=True)
    (root / "telecommands").mkdir()
    pd.DataFrame(
        [
            {
                "Channel": "channel_1",
                "Subsystem": "EPS",
                "Physical Unit": "V",
                "Group": "power",
                "Target": "battery",
            },
            {
                "Channel": "channel_2",
                "Subsystem": "AOCS",
                "Physical Unit": "deg",
                "Group": "attitude",
                "Target": "sensor",
            },
        ]
    ).to_csv(root / "channels.csv", index=False)
    pd.DataFrame(
        [{"Telecommand": "command_1", "Priority": 3}]
    ).to_csv(root / "telecommands.csv", index=False)
    pd.DataFrame(
        [
            {
                "ID": "A-1",
                "Channel": "channel_1",
                "StartTime": "2020-01-01T00:01:00Z",
                "EndTime": "2020-01-01T00:02:00Z",
            }
        ]
    ).to_csv(root / "labels.csv", index=False)
    pd.DataFrame(
        [
            {
                "ID": "A-1",
                "Class": "Point",
                "Subclass": "Spike",
                "Category": "Anomaly",
                "Dimensionality": "Univariate",
                "Locality": "Local",
                "Length": "Short",
            }
        ]
    ).to_csv(root / "anomaly_types.csv", index=False)
    index = pd.to_datetime(
        [
            "2020-01-01T00:00:00Z",
            "2020-01-01T00:01:00Z",
            "2020-01-01T00:02:00Z",
        ]
    )
    pd.DataFrame({"channel_1": [1.0, 2.0, 3.0]}, index=index).to_pickle(
        root / "channels" / "channel_1.zip"
    )
    pd.DataFrame({"channel_2": [4.0, 5.0, 6.0]}, index=index).to_pickle(
        root / "channels" / "channel_2.zip"
    )
    pd.DataFrame({"command_1": [0, 1, 0]}, index=index).to_pickle(
        root / "telecommands" / "command_1.zip"
    )
    return root


def test_raw_esa_mission_requires_explicit_pickle_trust(tmp_path: Path) -> None:
    root = _raw_esa_mission_root(tmp_path)
    with pytest.raises(ValueError, match="trusted_pickle"):
        ESAADBDataSource(root).open()


def test_raw_esa_mission_reads_pickle_zip_and_interval_labels(tmp_path: Path) -> None:
    root = _raw_esa_mission_root(tmp_path)
    source = ESAADBDataSource(root, trusted_pickle=True)
    with source:
        entity_list = source.list_entities()
        assert [item.entity_id for item in entity_list] == [
            "channel_1",
            "channel_2",
            "command_1",
        ]
        entities = {item.entity_id: item for item in entity_list}
        assert entities["channel_1"].format == "pickle_zip"
        assert entities["channel_1"].metadata["entity_kind"] == "telemetry"
        assert entities["command_1"].metadata["entity_kind"] == "telecommand"
        assert source.list_channels("channel_1") == ["channel_1"]

        frame = source.read_range(
            "channel_1",
            "2020-01-01T00:01:00Z",
            "2020-01-01T00:02:00Z",
        )
        assert list(frame.columns) == ["timestamp", "channel_1"]
        assert frame["channel_1"].tolist() == [2.0, 3.0]

        labels = source.read_labels(
            "channel_1",
            "2020-01-01T00:01:30Z",
            "2020-01-01T00:01:45Z",
        )
        assert labels["ID"].tolist() == ["A-1"]
        assert labels["Category"].tolist() == ["Anomaly"]

        metadata = source.get_metadata()
        assert metadata.metadata["mission_layout_detected"] is True
        assert metadata.metadata["telemetry_archive_entity_count"] == 2
        assert metadata.metadata["telecommand_archive_entity_count"] == 1
        assert metadata.metadata["partial_pickle_reads_supported"] is False


def test_raw_esa_mission_ignores_non_allowlisted_pickle_archive(tmp_path: Path) -> None:
    root = _raw_esa_mission_root(tmp_path)
    pd.DataFrame({"rogue": [1]}, index=pd.to_datetime(["2020-01-01T00:00:00Z"])).to_pickle(
        root / "channels" / "rogue.zip"
    )
    source = ESAADBDataSource(root, trusted_pickle=True)
    with source:
        assert "rogue" not in {item.entity_id for item in source.list_entities()}
        metadata = source.get_metadata()
        assert metadata.metadata["extra_channel_archives_not_allowlisted"] == ["rogue"]
