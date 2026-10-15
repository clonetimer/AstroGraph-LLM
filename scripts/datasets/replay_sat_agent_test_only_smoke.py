#!/usr/bin/env python3
"""Replay seven Sat-Agent nominal TEST_ONLY cases twice without writing datasets.

This is an S0 source-side determinism smoke check.  It reads only an explicit
allowlist of public physical quantities, never reads label/truth values, creates
no seed, and permanently disqualifies the cases from every modeling or gate
split.  It is not a diagnosis-performance evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


REPLAY_CASES = (
    {
        "replay_id": "R01",
        "component_archetype_id": "storage.energy_buffer",
        "capability_id": "component.battery.v1",
        "template_id": "component_battery_cycle",
        "public_allowlist": (
            "time_s",
            "eps.battery.soc",
            "eps.battery.net_power_w",
        ),
    },
    {
        "replay_id": "R02",
        "component_archetype_id": "thermal.lumped_capacity",
        "capability_id": "component.thermal_node.v1",
        "template_id": "component_thermal_node_response",
        "public_allowlist": (
            "time_s",
            "thermal.node.temp_k",
            "thermal.node.power_w",
        ),
    },
    {
        "replay_id": "R03",
        "component_archetype_id": "actuator.controlled_effort",
        "capability_id": "component.reaction_wheel.v1",
        "template_id": "component_reaction_wheel_nominal",
        "public_allowlist": (
            "time_s",
            "adcs.reaction_wheel.command_torque_nm_0",
            "adcs.reaction_wheel.speed_rad_s_0",
        ),
    },
    {
        "replay_id": "R04",
        "component_archetype_id": "sensor.measurement",
        "capability_id": "component.imu.v1",
        "template_id": "component_imu_nominal",
        "public_allowlist": (
            "time_s",
            "adcs.imu.gyro_x_rad_s",
            "adcs.imu.gyro_y_rad_s",
            "adcs.imu.gyro_z_rad_s",
        ),
    },
    {
        "replay_id": "R05",
        "component_archetype_id": "converter.power",
        "capability_id": "component.pdu.v1",
        "template_id": "component_pdu_nominal",
        "public_allowlist": (
            "time_s",
            "eps.pdu.requested_power_w",
            "eps.pdu.delivered_power_w",
        ),
    },
    {
        "replay_id": "R06",
        "component_archetype_id": "link.transport",
        "capability_id": "component.data_queue.v1",
        "template_id": "component_data_queue",
        "public_allowlist": (
            "time_s",
            "comm.data_queue.generated_bps",
            "comm.data_queue.downlink_bps",
        ),
    },
    {
        "replay_id": "R07",
        "component_archetype_id": "queue.bounded_inventory",
        "capability_id": "component.onboard_storage.v1",
        "template_id": "component_onboard_storage_fill",
        "public_allowlist": (
            "time_s",
            "comm.storage.fill_ratio",
            "comm.storage.generated_bps",
            "comm.storage.downlink_bps",
        ),
    },
)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _contains_seed_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).lower() == "seed" or _contains_seed_key(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_seed_key(item) for item in value)
    return False


def _capability_id(spec: Mapping[str, Any]) -> str | None:
    direct = spec.get("capability_id")
    if isinstance(direct, str):
        return direct
    model = spec.get("model")
    if isinstance(model, Mapping) and isinstance(model.get("capability_id"), str):
        return str(model["capability_id"])
    return None


def _validate_allowlist(fields: tuple[str, ...]) -> None:
    if not fields or fields[0] != "time_s" or len(fields) != len(set(fields)):
        raise ValueError("public allowlist must be unique and begin with time_s")
    denied_fragments = (
        "label",
        "truth",
        "fault",
        "effect",
        "case_id",
        "task_id",
        "target_name",
        "sample_index",
        "seed",
        "split",
    )
    for field in fields:
        lowered = field.lower()
        if any(fragment in lowered for fragment in denied_fragments):
            raise ValueError(f"prohibited public replay field: {field}")


def _project_public_rows(
    rows: list[Mapping[str, Any]], fields: tuple[str, ...]
) -> list[list[float]]:
    _validate_allowlist(fields)
    if not rows:
        raise ValueError("replay returned no telemetry rows")
    projected: list[list[float]] = []
    for row in rows:
        missing = [field for field in fields if field not in row]
        if missing:
            raise ValueError("replay is missing an allowlisted public field")
        values = [float(row[field]) for field in fields]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("replay returned a non-finite public value")
        projected.append(values)
    times = [row[0] for row in projected]
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("replay time axis is not strictly increasing")
    return projected


def _run_once(
    case: Mapping[str, Any],
    instantiate: Callable[[str], dict[str, Any]],
    compile_task: Callable[[dict[str, Any]], Any],
    run_task: Callable[..., Any],
) -> tuple[str, int, str]:
    spec = instantiate(str(case["template_id"]))
    if _contains_seed_key(spec):
        raise ValueError("TEST_ONLY replay template unexpectedly contains a seed")
    if _capability_id(spec) != case["capability_id"]:
        raise ValueError("TEST_ONLY replay capability binding changed")
    result = run_task(compile_task(spec), task_spec=spec, write_dataset=False)
    if result.summary.get("status") != "complete":
        raise RuntimeError("TEST_ONLY replay did not complete")
    projected = _project_public_rows(
        list(result.trace_rows), tuple(case["public_allowlist"])
    )
    return _canonical_sha256(projected), len(projected), _canonical_sha256(spec)


def build_receipt() -> dict[str, Any]:
    from sat_sim.scenario_templates import instantiate_scenario_template
    from sat_sim.task_compiler import compile_task_spec
    from sat_sim.task_runner import run_compiled_task

    results = []
    failures = []
    simulator_execution_count = 0

    def counted_run_task(*args: Any, **kwargs: Any) -> Any:
        nonlocal simulator_execution_count
        simulator_execution_count += 1
        return run_compiled_task(*args, **kwargs)

    for case in REPLAY_CASES:
        try:
            first_hash, first_rows, input_hash = _run_once(
                case,
                instantiate_scenario_template,
                compile_task_spec,
                counted_run_task,
            )
            second_hash, second_rows, second_input_hash = _run_once(
                case,
                instantiate_scenario_template,
                compile_task_spec,
                counted_run_task,
            )
            equal = (
                first_hash == second_hash
                and first_rows == second_rows
                and input_hash == second_input_hash
            )
            if not equal:
                failures.append(f"{case['replay_id']}_NONDETERMINISTIC")
            results.append(
                {
                    "replay_id": case["replay_id"],
                    "component_archetype_id": case["component_archetype_id"],
                    "capability_id": case["capability_id"],
                    "test_fixture_input_sha256": input_hash,
                    "allowlisted_public_field_count": len(case["public_allowlist"]),
                    "public_row_count": first_rows,
                    "first_public_projection_sha256": first_hash,
                    "second_public_projection_sha256": second_hash,
                    "exact_public_projection_match": equal,
                }
            )
        except Exception as exc:  # preserve every scheduled replay as a failure
            failures.append(f"{case['replay_id']}_{type(exc).__name__}")
            results.append(
                {
                    "replay_id": case["replay_id"],
                    "component_archetype_id": case["component_archetype_id"],
                    "capability_id": case["capability_id"],
                    "exact_public_projection_match": False,
                    "error_type": type(exc).__name__,
                }
            )

    return {
        "schema_version": "astrograph.p018.sat-agent-test-only-replay.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage": "P018_S0_ZERO_FIT",
        "status": "PASS" if not failures else "FAIL",
        "source_id": "SAT_AGENT_PLATFORM",
        "source_count": 1,
        "environment": {
            "interpreter_token": "${SAT_AGENT_PYTHON}",
            "python_version": platform.python_version(),
            "sat_agent_distribution_version": importlib.metadata.version(
                "satellite-simulation-platform"
            ),
        },
        "replays": results,
        "summary": {
            "scheduled_replay_count": len(REPLAY_CASES),
            "completed_exact_match_count": sum(
                item["exact_public_projection_match"] for item in results
            ),
            "planned_simulator_execution_count": 2 * len(REPLAY_CASES),
            "actual_simulator_execution_count": simulator_execution_count,
            "component_archetype_count": len(
                {case["component_archetype_id"] for case in REPLAY_CASES}
            ),
            "dataset_write_count": 0,
            "campaign_generation_count": 0,
            "seed_present_in_test_fixture_count": 0,
            "seed_creation_or_reservation_count": 0,
            "model_fit_count": 0,
            "threshold_selection_count": 0,
            "split_creation_count": 0,
            "truth_or_label_value_read_count": 0,
            "raw_public_values_emitted": False,
            "failure_codes": failures,
        },
        "usage_policy": {
            "grade": "TEST_ONLY_PERMANENTLY_DISQUALIFIED",
            "allowed_use": "SOURCE_SIDE_DETERMINISM_SMOKE_ONLY",
            "prohibited_uses": [
                "training",
                "pretraining",
                "feature selection",
                "calibration",
                "threshold selection",
                "model comparison",
                "technical validation",
                "acceptance",
            ],
        },
        "readiness": {
            "sampled_source_side_replay_complete": not failures,
            "formal_campaign_reproducibility_established": False,
            "diagnosis_task_suitability_established": False,
            "stage_s1_fit_authorized": False,
            "stage_b_authorized": False,
            "g2_pass": False,
            "g3_pass": False,
        },
        "claim_boundary": (
            "Seven archetype-specific nominal TEST_ONLY smoke cases were replayed twice "
            "inside the independent Sat-Agent interpreter and compared only on explicit "
            "public physical allowlists. This establishes sampled source-side determinism, "
            "not campaign, diagnosis, generalization, Stage S1, Stage B, G2 or G3 readiness."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_receipt()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": receipt["status"], "output": str(output)}))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
