from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from jsonschema import Draft202012Validator, ValidationError


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "datasets" / "validate_sat_agent_semantic_contracts.py"
SPEC = importlib.util.spec_from_file_location(
    "validate_sat_agent_semantic_contracts", SCRIPT
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

REPAIR_SCRIPT = ROOT / "scripts" / "datasets" / "repair_sat_agent_time_axes.py"
REPAIR_SPEC = importlib.util.spec_from_file_location(
    "repair_sat_agent_time_axes", REPAIR_SCRIPT
)
assert REPAIR_SPEC and REPAIR_SPEC.loader
REPAIR_MODULE = importlib.util.module_from_spec(REPAIR_SPEC)
sys.modules[REPAIR_SPEC.name] = REPAIR_MODULE
REPAIR_SPEC.loader.exec_module(REPAIR_MODULE)


def _battery_case(
    root: Path,
    name: str,
    *,
    with_time: bool,
    capability_id: str = "component.battery.v1",
    task_type: str = "component",
) -> None:
    case = root / "reports" / "battery" / "cases" / name
    case.mkdir(parents=True)
    (case / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_version": "0.1.0",
                "task": {
                    "task_type": task_type,
                    "capability_id": capability_id,
                },
                "run": {"duration_s": 1.0, "sample_s": 1.0},
                "quality": {"trace_rows": 2},
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "eps.battery.soc": [0.8, 0.79],
        "eps.battery.net_power_w": [-10.0, -10.0],
        "label.health_state": ["secret", "secret"],
    }
    if with_time:
        payload = {"time_s": [0.0, 1.0], **payload}
    pd.DataFrame(payload).to_csv(case / "trace.csv", index=False)


def test_semantic_contract_validation_keeps_missing_time_in_failure_denominator(
    tmp_path: Path,
) -> None:
    _battery_case(tmp_path, "with-time", with_time=True)
    _battery_case(tmp_path, "without-time", with_time=False)
    _battery_case(
        tmp_path,
        "cross-profile-with-time",
        with_time=True,
        capability_id="subsystem.eps.basic.v1",
        task_type="subsystem",
    )
    result = MODULE.build_validation(
        tmp_path,
        semantic_schema_path=ROOT / "configs/semantics/core_semantic_schema.json",
        archetype_catalog_path=ROOT / "configs/semantics/core_component_archetypes.json",
        channel_map_path=(
            ROOT / "configs/semantics/sat_agent_platform_channel_maps.s0.v1.json"
        ),
        schema_profile_path=(
            ROOT
            / "configs/data_sources/sat_agent_platform_schema_profiles.s0.v1.json"
        ),
        row_limit=16,
    )
    battery = next(
        item
        for item in result["mapping_results"]
        if item["mapping_id"] == "sat-agent-platform.component.battery.v1"
    )
    assert battery["matched_entity_count"] == 3
    assert battery["passed_entity_count"] == 2
    assert battery["failed_entity_count"] == 1
    assert battery["declared_profile_matched_entity_count"] == 2
    assert battery["declared_profile_passed_entity_count"] == 1
    assert battery["declared_profile_failed_entity_count"] == 1
    assert battery["cross_profile_compatible_entity_count"] == 1
    assert battery["cross_profile_passed_entity_count"] == 1
    assert battery["cross_profile_failed_entity_count"] == 0
    assert battery["signal_schema_match_missing_time_axis_count"] == 1
    assert battery["error_counts"] == {"MISSING_TIME_AXIS": 1}
    assert battery["missing_time_axis_contract_status_counts"] == {
        "MISSING_EXPLICIT_TIME_AXIS": 1
    }
    assert battery["missing_time_row_count_convention_counts"] == {
        "ROW_COUNT_EQUALS_DURATION_DIV_SAMPLE_PLUS_ONE": 1
    }
    summary = result["summary"]
    assert summary["field_compatible_entity_denominator"] == 3
    assert summary["declared_profile_entity_denominator"] == 2
    assert summary["cross_profile_compatible_entity_denominator"] == 1
    assert (
        summary["declared_profile_entity_denominator"]
        + summary["cross_profile_compatible_entity_denominator"]
        == summary["field_compatible_entity_denominator"]
    )
    assert result["summary"]["structural_contract_status"] == "BLOCKED"
    assert result["time_axis_audit"]["missing_time_axis_entity_count"] == 1
    assert (
        result["time_axis_audit"]["consumer_side_time_synthesis_authorized"]
        is False
    )
    assert result["time_axis_audit"]["source_export_repair_required"] is True
    assert (
        result["execution_receipt"][
            "manifest_annotation_fields_used_for_selection_or_mapping"
        ]
        is False
    )
    assert result["execution_receipt"]["truth_values_read"] is False
    assert result["source_count"] == 1


def _valid_time_axis_receipt() -> dict[str, object]:
    return {
        "schema_version": "sat-agent-public-time-axis.v1",
        "source_id": "SAT_AGENT_PLATFORM",
        "source_count": 1,
        "export_receipt_id": "thermal.case-001",
        "telemetry_sha256": "0" * 64,
        "row_count": 2,
        "time_axis": {
            "field": "time_s",
            "unit": "s",
            "storage_policy": "EXPLICIT_COLUMN_PER_TELEMETRY_ROW",
            "monotonicity": "STRICTLY_INCREASING",
            "finite_values_required": True,
            "row_index_derivation_allowed": False,
            "manifest_sample_interval_substitution_allowed": False,
            "endpoint_policy": "INCLUDE_START_AND_END",
        },
        "truth_or_label_fields_present": False,
    }


def test_sat_agent_public_time_axis_schema_forbids_surrogate_reconstruction() -> None:
    schema_path = (
        ROOT / "protocols" / "integration" / "sat_agent_public_time_axis.v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    validator.validate(_valid_time_axis_receipt())

    invalid = _valid_time_axis_receipt()
    invalid["time_axis"] = {
        **invalid["time_axis"],  # type: ignore[arg-type]
        "row_index_derivation_allowed": True,
    }
    with pytest.raises(ValidationError):
        validator.validate(invalid)


def test_time_axis_repair_protocol_binds_audit_denominator_and_authority() -> None:
    evidence = json.loads(
        (
            ROOT
            / "governance"
            / "evidence"
            / "0.7.1-extension"
            / "p018_sat_agent_semantic_contract_validation.v1.json"
        ).read_text(encoding="utf-8")
    )
    repair = json.loads(
        (
            ROOT
            / "protocols"
            / "integration"
            / "p018_sat_agent_time_axis_export_repair.v1.json"
        ).read_text(encoding="utf-8")
    )
    summary = evidence["summary"]
    time_audit = evidence["time_axis_audit"]
    problem = repair["problem"]

    assert repair["status"] == "PROPOSED_SOURCE_WRITE_NOT_AUTHORIZED"
    assert repair["source_id"] == evidence["source_id"] == "SAT_AGENT_PLATFORM"
    assert repair["source_count"] == evidence["source_count"] == 1
    assert problem["field_compatible_entity_denominator"] == summary[
        "field_compatible_entity_denominator"
    ]
    assert problem["missing_explicit_time_axis_entity_count"] == time_audit[
        "missing_time_axis_entity_count"
    ]
    assert problem["declared_profile_missing_entity_count"] == summary[
        "declared_profile_failed_entity_count"
    ]
    assert problem["cross_profile_compatible_missing_entity_count"] == summary[
        "cross_profile_failed_entity_count"
    ]
    assert problem["alternative_explicit_time_field_entity_count"] == time_audit[
        "explicit_alternative_time_field_entity_count"
    ]
    assert sum(problem["affected_capability_counts"].values()) == time_audit[
        "missing_time_axis_entity_count"
    ]
    assert problem["consumer_side_time_synthesis_authorized"] is False
    assert repair["authorization_boundary"]["sat_agent_source_write_authorized"] is False
    assert repair["authorization_boundary"]["model_fit_or_calibration_authorized"] is False
    assert repair["authorization_boundary"]["new_seed_authorized"] is False


def test_time_axis_byte_repair_is_exactly_reversible_and_endpoint_bound() -> None:
    original = (
        b"\xef\xbb\xbfeps.battery.soc,eps.battery.net_power_w\r\n"
        b"0.8,-10.0\r\n"
        b"0.79,-10.0"
    )
    assert REPAIR_MODULE.endpoint_policy(2, 1.0, 1.0) == "INCLUDE_START_AND_END"
    assert REPAIR_MODULE.endpoint_policy(2, 2.0, 1.0) == "INCLUDE_START_EXCLUDE_END"
    with pytest.raises(ValueError, match="ROW_COUNT_CONVENTION_OTHER"):
        REPAIR_MODULE.endpoint_policy(2, 1.5, 1.0)

    repaired = REPAIR_MODULE.prepend_explicit_time_axis(
        original,
        row_count=2,
        sample_s=1.0,
        endpoint="INCLUDE_START_AND_END",
    )
    assert repaired.startswith(b"\xef\xbb\xbftime_s,eps.battery.soc")
    assert REPAIR_MODULE.recover_original_bytes(repaired) == original

    single_column = b"thermal.node.temperature_k\n293.15\n294.2\n"
    repaired_single = REPAIR_MODULE.prepend_explicit_time_axis(
        single_column,
        row_count=2,
        sample_s=60.0,
        endpoint="INCLUDE_START_EXCLUDE_END",
    )
    assert repaired_single.startswith(b"time_s,thermal.node.temperature_k\n")
    assert REPAIR_MODULE.recover_original_bytes(repaired_single) == single_column


def test_time_axis_repair_discovery_matches_missing_public_field_contract(
    tmp_path: Path,
) -> None:
    _battery_case(tmp_path, "missing-time", with_time=False)
    _battery_case(tmp_path, "has-time", with_time=True)
    targets = REPAIR_MODULE.discover_targets(
        tmp_path,
        ROOT / "configs/semantics/sat_agent_platform_channel_maps.s0.v1.json",
    )
    assert len(targets) == 1
    assert targets[0].capability_id == "component.battery.v1"
    assert targets[0].endpoint_policy == "INCLUDE_START_AND_END"
    assert targets[0].row_count == 2


def test_applied_time_axis_receipt_preserves_denominator_and_prior_failure() -> None:
    evidence_root = ROOT / "governance" / "evidence" / "0.7.1-extension"
    before = json.loads(
        (evidence_root / "p018_sat_agent_semantic_contract_validation.v1.json").read_text(
            encoding="utf-8"
        )
    )
    after = json.loads(
        (
            evidence_root
            / "p018_sat_agent_semantic_contract_validation.post_repair.v1.json"
        ).read_text(encoding="utf-8")
    )
    receipt = json.loads(
        (evidence_root / "p018_sat_agent_time_axis_repair_receipt.v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert before["summary"]["field_compatible_entity_denominator"] == 237
    assert before["summary"]["passed_entity_count"] == 180
    assert before["summary"]["failed_entity_count"] == 57
    assert after["summary"]["field_compatible_entity_denominator"] == 237
    assert after["summary"]["passed_entity_count"] == 237
    assert after["summary"]["failed_entity_count"] == 0
    assert after["summary"]["structural_contract_status"] == "PASS"
    assert after["summary"]["blockers"] == []
    assert receipt["status"] == "APPLIED_PASS"
    assert receipt["authorized_target_count"] == 57
    assert receipt["changed_source_file_count"] == 57
    assert len(receipt["receipts"]) == 57
    assert receipt["endpoint_policy_counts"] == {
        "INCLUDE_START_AND_END": 21,
        "INCLUDE_START_EXCLUDE_END": 36,
    }
    assert receipt["execution_boundary"]["non_time_bytes_preserved_exactly"] is True
    assert receipt["execution_boundary"]["truth_or_label_values_parsed"] is False
    assert receipt["execution_boundary"]["case_run_paths_emitted"] is False
    assert all(
        item["opaque_original_bytes_recovered"] for item in receipt["receipts"]
    )
