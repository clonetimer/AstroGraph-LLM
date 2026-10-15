from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from astrograph.data_sources import (
    DataSourceType,
    DatasetSourceConfig,
    SatAgentPlatformDataSource,
    build_data_source,
    is_sat_agent_model_channel_safe,
)


def _write_manifest(
    case_root: Path,
    *,
    capability_id: str,
    level: str,
    target: str,
    rows: int,
) -> None:
    (case_root / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_version": "0.1.0",
                "task": {
                    "task_id": case_root.name,
                    "task_type": level,
                    "seed": 1234,
                    "capability_id": capability_id,
                },
                "run": {"duration_s": 60.0, "sample_s": 1.0},
                "labels": {
                    "target_level": level,
                    "target_name": target,
                    "fault_labels": ["DO_NOT_EXPOSE"],
                },
                "quality": {"trace_rows": rows},
                "capability": {"source_binding": {"mode": "source_native"}},
            }
        ),
        encoding="utf-8",
    )


def _sat_agent_root(tmp_path: Path) -> Path:
    root = tmp_path / "sat_agent_platform"

    contracted = root / "reports" / "rw_campaign" / "cases" / "rw_fault"
    (contracted / "astrograph").mkdir(parents=True)
    _write_manifest(
        contracted,
        capability_id="subsystem.adcs_unified_native.v1",
        level="subsystem",
        target="reaction_wheel",
        rows=3,
    )
    # The source trace deliberately contains a channel not allowed by the
    # projection contract.  Discovery must prefer astrograph/telemetry.csv.
    pd.DataFrame({"time_s": [0, 1, 2], "rogue.source_trace": [1, 2, 3]}).to_csv(
        contracted / "trace.csv", index=False
    )
    pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0],
            "sensor.adcs.rw.speed_rad_s_0": [1.0, 1.1, 1.2],
            "command.adcs.rw.motor_torque_nm_0": [0.1, 0.2, 0.3],
            "estimate.adcs.rw.effective_max_torque_nm_0": [0.4, 0.0, 0.0],
            "status.adcs.rw.available_0": [True, False, False],
            "label.class_id": ["ADCS_RW_JAM"] * 3,
            "label.case_role": ["fault"] * 3,
        }
    ).to_csv(contracted / "astrograph" / "telemetry.csv", index=False)
    (contracted / "astrograph" / "dataset_contract.json").write_text(
        json.dumps(
            {
                "schema_version": "sat-sim.astrograph-dataset.v2",
                "simulation": {
                    "capability_id": "subsystem.adcs_unified_native.v1",
                    "backend": "basilisk",
                    "duration_s": 60.0,
                    "sample_s": 1.0,
                    "fidelity": {
                        "fidelity_level": "A_ENGINEERING_BASILISK_NATIVE",
                        "flight_validated": False,
                    },
                },
                "telemetry": {
                    "row_count": 3,
                    "channel_contract": {
                        "training_feature_policy": {
                            "allowed_model_input_channels": [
                                "sensor.adcs.rw.speed_rad_s_0",
                                "command.adcs.rw.motor_torque_nm_0",
                            ]
                        }
                    },
                },
                "ground_truth": {"class_id": "DO_NOT_EXPOSE"},
            }
        ),
        encoding="utf-8",
    )

    battery = root / "reports" / "battery_campaign" / "cases" / "battery_nominal"
    battery.mkdir(parents=True)
    _write_manifest(
        battery,
        capability_id="component.battery.v1",
        level="component",
        target="battery",
        rows=2,
    )
    pd.DataFrame(
        {
            "task_id": ["battery_nominal"] * 2,
            "case_id": ["case_000"] * 2,
            "time_s": [0.0, 10.0],
            "sample_index": [0, 1],
            "target_level": ["component"] * 2,
            "target_name": ["battery"] * 2,
            "mode": ["nominal"] * 2,
            "eps.battery.soc": [0.6, 0.59],
            "eps.battery.net_power_w": [-20.0, -20.0],
            "eps.battery.effective_capacity_wh": [120.0, 80.0],
            "eps.battery.nominal_capacity_wh": [120.0, 120.0],
            "eps.battery.degradation_factor": [1.0, 0.8],
            "eps.battery.bias_estimate": [0.0, 0.1],
            "eps.solar.fault_multiplier": [1.0, 0.2],
            "status.battery.direct_fault_proxy": [0, 1],
            "label.health_state": ["nominal", "fault"],
            "label.fault_active": [False, True],
        }
    ).to_csv(battery / "trace.csv", index=False)

    thermal = root / "reports" / "thermal_campaign" / "cases" / "thermal_nominal"
    thermal.mkdir(parents=True)
    _write_manifest(
        thermal,
        capability_id="subsystem.thermal.source_native.v1",
        level="subsystem",
        target="thermal",
        rows=2,
    )
    pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "thermal.battery_temp_k": [290.0, 291.0],
            "thermal.heater_power_w": [0.0, 5.0],
            "label.health_state": ["nominal", "nominal"],
        }
    ).to_csv(thermal / "trace.csv", index=False)

    unreadable = root / "reports" / "historical" / "cases" / "empty_run"
    unreadable.mkdir(parents=True)
    _write_manifest(
        unreadable,
        capability_id="component.unknown.v1",
        level="component",
        target="unknown",
        rows=0,
    )
    (unreadable / "trace.csv").write_text("", encoding="utf-8")

    invalid = root / "reports" / "invalid_contract" / "cases" / "invalid_projection"
    (invalid / "astrograph").mkdir(parents=True)
    _write_manifest(
        invalid,
        capability_id="component.invalid.v1",
        level="component",
        target="invalid",
        rows=2,
    )
    pd.DataFrame(
        {"time_s": [0.0, 1.0], "physical.signal": [1.0, 2.0]}
    ).to_csv(invalid / "astrograph" / "telemetry.csv", index=False)
    (invalid / "astrograph" / "dataset_contract.json").write_text(
        "{invalid-json", encoding="utf-8"
    )

    invalid_policy = root / "reports" / "invalid_policy" / "cases" / "projection"
    (invalid_policy / "astrograph").mkdir(parents=True)
    _write_manifest(
        invalid_policy,
        capability_id="component.invalid-policy.v1",
        level="component",
        target="invalid",
        rows=2,
    )
    pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "physical.signal": [1.0, 2.0],
            "label.class_id": ["secret", "secret"],
        }
    ).to_csv(invalid_policy / "astrograph" / "telemetry.csv", index=False)
    (invalid_policy / "astrograph" / "dataset_contract.json").write_text(
        json.dumps(
            {
                "telemetry": {
                    "channel_contract": {
                        "training_feature_policy": {
                            "allowed_model_input_channels": [
                                "physical.signal",
                                "label.class_id",
                                "missing.signal",
                            ]
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return root


def test_contract_projection_wins_and_is_an_exact_model_allowlist(tmp_path: Path) -> None:
    source = SatAgentPlatformDataSource(_sat_agent_root(tmp_path))
    with source:
        rw = next(
            item
            for item in source.list_entities()
            if item.metadata["model_channel_policy"] == "dataset_contract_allowlist"
        )
        assert rw.entity_id.startswith("sat-entity-")
        assert rw.relative_path == f"${{ASTROGRAPH_SAT_AGENT_ROOT}}/{rw.entity_id}"
        assert "fault" not in rw.entity_id
        assert "fault" not in rw.relative_path
        assert source.list_channels(rw.entity_id) == [
            "sensor.adcs.rw.speed_rad_s_0",
            "command.adcs.rw.motor_torque_nm_0",
        ]
        assert rw.channel_count == 2
        assert rw.row_count == 3
        assert rw.metadata["model_channel_policy"] == "dataset_contract_allowlist"
        assert rw.metadata["contract_missing_model_channels"] == []
        assert rw.metadata["contract_rejected_prohibited_model_channels"] == []
        assert rw.metadata["time_axis_contract_status"] == "EXPECTED_EXPLICIT_TIME_PRESENT"
        assert rw.metadata["explicit_time_axis_candidates"] == ["time_s"]
        assert rw.metadata["index_surrogate_candidates"] == []

        frame = source.read_range(rw.entity_id)
        assert list(frame.columns) == [
            "time_s",
            "sensor.adcs.rw.speed_rad_s_0",
            "command.adcs.rw.motor_torque_nm_0",
        ]
        assert "rogue.source_trace" not in frame
        assert not any(column.startswith("label.") for column in frame)
        assert not any(column.startswith("status.") for column in frame)


def test_uncontracted_cases_preserve_heterogeneous_schemas_but_remove_proxies(
    tmp_path: Path,
) -> None:
    source = SatAgentPlatformDataSource(_sat_agent_root(tmp_path))
    with source:
        entities = {item.entity_id: item for item in source.list_entities()}
        battery_id = next(
            key
            for key, item in entities.items()
            if item.metadata["capability_id"] == "component.battery.v1"
        )
        thermal_id = next(
            key
            for key, item in entities.items()
            if item.metadata["capability_id"] == "subsystem.thermal.source_native.v1"
        )

        assert source.list_channels(battery_id) == [
            "eps.battery.soc",
            "eps.battery.net_power_w",
        ]
        assert source.list_channels(thermal_id) == [
            "thermal.battery_temp_k",
            "thermal.heater_power_w",
        ]
        assert source.list_channels(battery_id) != source.list_channels(thermal_id)
        assert entities[battery_id].metadata["metadata_is_model_input"] is False
        assert entities[battery_id].metadata["truth_access"] == "DENIED_IN_SOURCE_ADAPTER"
        assert entities[battery_id].metadata["component_archetype"] is None
        assert (
            entities[battery_id].metadata["time_axis_contract_status"]
            == "EXPECTED_EXPLICIT_TIME_PRESENT"
        )
        assert entities[battery_id].metadata["index_surrogate_candidates"] == [
            "sample_index"
        ]
        assert "sample_index" not in source.list_channels(battery_id)
        invalid_entity = next(
            item
            for item in entities.values()
            if item.metadata["capability_id"] == "component.invalid.v1"
        )
        assert source.list_channels(invalid_entity.entity_id) == []
        assert invalid_entity.metadata["model_channel_policy"] == "invalid_contract_fail_closed"
        assert invalid_entity.metadata["contract_policy_error_codes"] == [
            "CONTRACT_PARSE_FAILED"
        ]
        invalid_policy_entity = next(
            item
            for item in entities.values()
            if item.metadata["capability_id"] == "component.invalid-policy.v1"
        )
        assert source.list_channels(invalid_policy_entity.entity_id) == []
        assert invalid_policy_entity.metadata["contract_policy_error_codes"] == [
            "PROHIBITED_OR_TIMESTAMP_MODEL_CHANNEL",
            "MISSING_MODEL_CHANNEL",
        ]


def test_truth_is_fail_closed_and_source_is_explicitly_single_source(tmp_path: Path) -> None:
    root = _sat_agent_root(tmp_path)
    config = DatasetSourceConfig(
        source_type=DataSourceType.SAT_AGENT_PLATFORM,
        root=str(root),
        read_only=True,
    )
    source = build_data_source(config)
    assert isinstance(source, SatAgentPlatformDataSource)
    with source:
        entity_id = source.list_entities()[0].entity_id
        with pytest.raises(PermissionError, match="evaluator-only"):
            source.read_labels(entity_id)

        metadata = source.get_metadata()
        assert metadata.source_type == DataSourceType.SAT_AGENT_PLATFORM
        assert metadata.root_token == "${ASTROGRAPH_SAT_AGENT_ROOT}"
        assert metadata.metadata["source_count"] == 1
        assert metadata.metadata["heterogeneous_component_schemas_preserved"] is True
        assert metadata.metadata["candidate_file_count"] == 6
        assert metadata.metadata["skipped_unreadable_candidate_count"] == 1
        assert metadata.metadata["skipped_unreadable_candidate_error_counts"] == {
            "EmptyDataError": 1
        }
        assert metadata.metadata["invalid_contract_fail_closed_entity_count"] == 2
        assert "not independent data sources" in metadata.claim_boundary
        assert "flight" in metadata.claim_boundary


def test_sat_agent_adapter_rejects_write_mode(tmp_path: Path) -> None:
    config = DatasetSourceConfig(
        source_type=DataSourceType.SAT_AGENT_PLATFORM,
        root=str(_sat_agent_root(tmp_path)),
        read_only=False,
    )
    with pytest.raises(ValueError, match="read-only"):
        SatAgentPlatformDataSource(config).open()


@pytest.mark.parametrize(
    "channel",
    [
        "adcs.star_tracker.drift_norm",
        "adcs.magnetometer.effective_bias_x_t",
        "eps.battery.nominal_capacity_wh",
        "eps.battery.degradation_factor",
        "thermal.heater.stuck_state",
        "sim.injected_modifier",
        "adcs.event.active_effects",
        "adcs.target.mode",
        "eps.load_shed_active",
        "fault",
        "label",
        "target",
        "anomaly",
        "is_anomaly",
        "truth",
        "status",
        "time_s",
        "time",
        "timestamp",
        "sample_index",
    ],
)
def test_uncontracted_simulator_answer_proxies_are_fail_closed(channel: str) -> None:
    assert is_sat_agent_model_channel_safe(channel) is False
