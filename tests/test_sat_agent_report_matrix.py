from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "datasets" / "inventory_sat_agent_report_matrix.py"
SPEC = importlib.util.spec_from_file_location("inventory_sat_agent_report_matrix", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _case(root: Path, name: str, *, task_id: str) -> None:
    dataset = root / "reports" / name / "results" / "dataset"
    dataset.mkdir(parents=True)
    (dataset / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": task_id,
                "status": "complete",
                "task": {"capability_id": "component.battery.v1", "seed": 9},
                "labels": {
                    "target_level": "component",
                    "target_name": "battery",
                    "mode": "fault",
                },
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "task_id": [task_id, task_id],
            "case_id": ["secret", "secret"],
            "time_s": [0.0, 1.0],
            "eps.battery.soc": [0.8, 0.7],
            "label.fault_active": [False, True],
            "modifier.active_ids": ["", "hidden-effect"],
        }
    ).to_csv(dataset / "trace.csv", index=False)


def test_report_matrix_deduplicates_physics_and_never_emits_identity_values(
    tmp_path: Path,
) -> None:
    _case(tmp_path, "first", task_id="first-secret")
    _case(tmp_path, "retry", task_id="retry-secret")
    invalid = tmp_path / "reports" / "bad" / "results" / "dataset"
    invalid.mkdir(parents=True)
    (invalid / "manifest.json").write_text("{broken", encoding="utf-8")

    result = MODULE.build_matrix(tmp_path)
    assert result["source_count"] == 1
    assert result["summary"]["manifest_file_count"] == 3
    assert result["summary"]["valid_manifest_count"] == 2
    assert result["summary"]["invalid_manifest_count"] == 1
    assert result["summary"]["readable_trace_count"] == 2
    assert result["summary"]["unique_physical_trace_count"] == 1
    assert result["summary"]["duplicate_physical_trace_count"] == 1
    assert result["summary"]["trace_error_counts"] == {}
    assert result["matrix"][0]["safe_channel_names"] == ["eps.battery.soc"]
    assert set(result["matrix"][0]["prohibited_field_names_present"]) == {
        "case_id",
        "label.fault_active",
        "modifier.active_ids",
        "task_id",
    }
    serialized = json.dumps(result)
    for secret in ("first-secret", "retry-secret", "hidden-effect"):
        assert secret not in serialized
    assert result["execution_receipt"]["manifest_aggregate_label_metadata_loaded"] is True
    assert result["execution_receipt"]["manifest_aggregate_label_metadata_emitted"] is True
    assert result["execution_receipt"]["trace_label_truth_or_proxy_values_loaded"] is False
