from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "datasets" / "inventory_sat_agent_exports.py"
SPEC = importlib.util.spec_from_file_location("inventory_sat_agent_exports", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_export(path: Path, *, kind: str) -> None:
    if kind == "RW_D3":
        root = "rw/02_dataset"
        manifest = {
            "dataset_id": "rw-test",
            "campaign_stage": "D3_RW_PRODUCTION",
            "status": "COMPLETE",
            "planned_run_count": 3,
            "complete_run_count": 3,
            "independent_run_counts": {"nominal": 1, "friction": 1, "authority": 1},
            "field_contract_sha256": "a" * 64,
            "claim_boundary": "engineering only",
        }
        split = {
            "split_unit": "run_id",
            "seed": 7,
            "train_run_ids": ["secret-a"],
            "validation_run_ids": ["secret-b"],
            "test_run_ids": ["secret-c"],
            "zero_shot_validation_run_ids": [],
        }
        reproduction = {
            "status": "COMPLETE",
            "passed": True,
            "comparison": "exact hash",
            "runs": [{"run_id": "secret-a", "equal": True}],
        }
    else:
        root = "thermal"
        manifest = {
            "dataset_id": "thermal-test",
            "campaign_stage": "D3_SINGLE_FAULT_PRODUCTION",
            "status": "COMPLETE",
            "run_counts": {"nominal": 2, "heater": 1, "star": 1},
            "claim_boundary": "engineering only",
        }
        split = {
            "split_unit": "run_id",
            "train_run_ids": ["secret-1", "secret-2"],
            "validation_run_ids": ["secret-3"],
            "test_run_ids": ["secret-4"],
            "zero_shot_validation_run_ids": [],
        }
        reproduction = None

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{root}/dataset_manifest.json", json.dumps(manifest))
        archive.writestr(f"{root}/dataset_split_manifest.json", json.dumps(split))
        if reproduction is not None:
            archive.writestr(
                f"{root}/reproducibility_report.json", json.dumps(reproduction)
            )
        archive.writestr(f"{root}/telemetry_raw.csv", "time_s,signal\n0,1\n")
        archive.writestr(f"{root}/telemetry_labels.csv", "secret-label\n")
        run_ids = {
            run_id
            for key, values in split.items()
            if key.endswith("_run_ids")
            for run_id in values
        }
        for run_id in sorted(run_ids):
            if kind == "RW_D3":
                prefix = f"rw/03_run_audit/{run_id}"
            else:
                prefix = f"thermal/runs/{run_id}"
            archive.writestr(f"{prefix}/run_manifest.json", "{}")


def test_export_registry_is_aggregate_single_source_and_truth_blind(tmp_path: Path) -> None:
    rw = tmp_path / "rw.zip"
    thermal = tmp_path / "thermal.zip"
    _write_export(rw, kind="RW_D3")
    _write_export(thermal, kind="THERMAL_STAR_D3")

    registry = MODULE.build_registry(rw, thermal)
    assert registry["source_count"] == 1
    assert registry["aggregate"] == {
        "asset_count": 2,
        "complete_run_count": 7,
        "model_fit_count": 0,
        "truth_or_label_row_read_count": 0,
        "raw_telemetry_row_read_count": 0,
        "aggregate_manifest_class_counts_read": True,
        "split_run_ids_read_for_integrity_but_not_emitted": True,
        "seed_created_or_reserved_count": 0,
    }
    assert registry["assets"][0]["aggregate_split_counts"] == {
        "test": 1,
        "train": 1,
        "validation": 1,
        "zero_shot_validation": 0,
    }
    assert registry["assets"][0]["historical_split_seed_present"] is True
    assert registry["assets"][0]["historical_split_seed_value_emitted"] is False
    assert registry["assets"][0]["materialized_run_manifest_count"] == 3
    assert registry["assets"][0]["complete_run_count_verified_by_run_manifests"] is True
    assert registry["assets"][0]["reproducibility_summary"]["comparison_count"] == 1
    serialized = json.dumps(registry)
    for secret in ("secret-a", "secret-b", "secret-c", "secret-label"):
        assert secret not in serialized


def test_split_ids_must_be_unique_and_disjoint() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        MODULE._split_id_sets({"train_run_ids": ["a", "a"]})
    with pytest.raises(ValueError, match="overlap"):
        MODULE._split_id_sets(
            {"train_run_ids": ["a"], "validation_run_ids": ["a"]}
        )


def test_same_archive_cannot_be_double_counted(tmp_path: Path) -> None:
    archive = tmp_path / "rw.zip"
    _write_export(archive, kind="RW_D3")
    with pytest.raises(ValueError, match="distinct archives"):
        MODULE.build_registry(archive, archive)
