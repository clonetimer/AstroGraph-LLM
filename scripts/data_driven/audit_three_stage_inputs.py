#!/usr/bin/env python3
"""Read-only inventory for detection/localization/classification experiments.

This audit deliberately does not fit a model.  It checks whether the apparent
sample count represents independent simulation conditions, inventories the
different dataset generations, and separates top-level validation from the
source simulator's own execution/physics status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
import os
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SAT_ROOT = Path(os.environ.get("ASTROGRAPH_DATA_ROOT", str(Path(__file__).resolve().parents[2] / "data" / "external")))
DEFAULT_DATASETS = (
    "whole_spacecraft_adcs_paired_dataset_v5_2_2",
    "whole_spacecraft_phm_dataset_v6_0",
)
TELEMETRY_NAMES = (
    "telemetry_fused_1hz.csv",
    "telemetry.csv",
    "telemetry_native.csv",
)
DENY_COLUMN_TOKENS = ("fault", "label", "target", "scenario", "seed", "split", "phase")


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _metadata_inventory(root: Path) -> dict[str, Any]:
    metadata_paths = sorted(root.rglob("metadata.json"))
    profile_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    validation_status: Counter[str] = Counter()
    source_status: Counter[str] = Counter()
    source_execution_status: Counter[str] = Counter()
    source_physics_status: Counter[str] = Counter()
    source_mission_status: Counter[str] = Counter()
    fault_cardinality: Counter[int] = Counter()
    source_execution_core_pass: Counter[str] = Counter()
    initial_signatures: set[str] = set()
    case_counts: Counter[str] = Counter()

    for path in metadata_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        case = payload.get("case", {})
        profile = (
            payload.get("profile_name")
            or payload.get("profile", {}).get("name")
            or case.get("profile")
            or "UNKNOWN"
        )
        profile_counts[str(profile)] += 1
        mode_counts[str(payload.get("mode", "fault_case"))] += 1
        validation_status[str(payload.get("validation", {}).get("status", "UNDECLARED"))] += 1
        source_execution_core_pass[
            str(payload.get("validation", {}).get("checks", {}).get("source_execution_core_pass", "UNDECLARED"))
        ] += 1
        summary = payload.get("source_summary", {})
        source_status[str(summary.get("status", "UNDECLARED"))] += 1
        source_execution_status[str(summary.get("execution_status", "UNDECLARED"))] += 1
        source_physics_status[str(summary.get("physics_status", "UNDECLARED"))] += 1
        source_mission_status[str(summary.get("mission_status", "UNDECLARED"))] += 1
        fault_cardinality[len(payload.get("faults", []))] += 1
        initial_signatures.add(_sha256_json(payload.get("initial_overrides", {})))
        if case.get("id"):
            case_counts[str(case["id"])] += 1

    return {
        "metadata_count": len(metadata_paths),
        "profile_counts": dict(sorted(profile_counts.items())),
        "mode_counts": dict(sorted(mode_counts.items())),
        "validation_status": dict(sorted(validation_status.items())),
        "source_execution_core_pass": dict(sorted(source_execution_core_pass.items())),
        "source_status": dict(sorted(source_status.items())),
        "source_execution_status": dict(sorted(source_execution_status.items())),
        "source_physics_status": dict(sorted(source_physics_status.items())),
        "source_mission_status": dict(sorted(source_mission_status.items())),
        "fault_cardinality": {str(key): value for key, value in sorted(fault_cardinality.items())},
        "unique_initial_override_signatures": len(initial_signatures),
        "unique_case_ids": len(case_counts),
        "samples_per_case_distribution": {
            str(key): value for key, value in sorted(Counter(case_counts.values()).items())
        },
    }


def _sample_plan_inventory(root: Path) -> dict[str, Any] | None:
    path = root / "sample_plan.json"
    if not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"sample plan must be a list: {path}")
    if rows and "case_id" not in rows[0]:
        return {
            "planned_rows": len(rows),
            "schema_status": "NON_MULTISAMPLE_PLAN",
            "row_keys": sorted(rows[0]),
        }
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[str(row["case_id"])].append(row)

    def distribution(key: str) -> dict[str, int]:
        unique_per_case = Counter(len({row.get(key) for row in case_rows}) for case_rows in by_case.values())
        return {str(count): cases for count, cases in sorted(unique_per_case.items())}

    return {
        "planned_rows": len(rows),
        "case_count": len(by_case),
        "samples_per_case": {
            str(count): cases for count, cases in sorted(Counter(map(len, by_case.values())).items())
        },
        "unique_magnitude_count_per_case": distribution("magnitude"),
        "unique_onset_count_per_case": distribution("onset_s"),
        "unique_duration_count_per_case": distribution("fault_duration_s"),
        "declared_seed_count": len({row.get("sample_seed") for row in rows}),
        "seed_consumed_true_count": sum(row.get("seed_consumed") is True for row in rows),
        "seed_consumption_attestation_present_count": sum("seed_consumed" in row for row in rows),
    }


def _generator_seed_binding(generator_path: Path) -> dict[str, Any]:
    if not generator_path.exists():
        return {"generator_path": str(generator_path), "status": "MISSING"}
    source = generator_path.read_text(encoding="utf-8")
    run_one = source[source.find("def run_one"):source.find("def write_manifest")]
    return {
        "generator_path": str(generator_path),
        "sample_seed_declared": '"sample_seed"' in source,
        "sample_seed_referenced_by_run_one": "sample_seed" in run_one,
        "seed_cli_forwarded_by_run_one": any(
            token in run_one
            for token in ('"--seed"', '"--random-seed"', '"--sample-seed"', '"--simulation-seed"')
        ),
    }


def _prefault_identity_probe(dataset_root: Path, rows: int) -> dict[str, Any]:
    plan_path = dataset_root / "sample_plan.json"
    if not plan_path.exists():
        return {"status": "NOT_APPLICABLE"}
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in plan:
        by_case[str(item["case_id"])].append(item)
    candidate = next((items for items in by_case.values() if len(items) >= 2), None)
    if candidate is None:
        return {"status": "NO_MULTI_SAMPLE_CASE"}
    safe_rows = min(rows, int(min(float(item["onset_s"]) for item in candidate)))
    frames: list[pd.DataFrame] = []
    sample_ids: list[int] = []
    compared_columns: list[str] | None = None
    for item in sorted(candidate, key=lambda row: int(row["sample_index"])):
        csv_path = Path(item["output_dir"]) / "telemetry_fused_1hz.csv"
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path, nrows=safe_rows)
        columns = [
            column
            for column in frame.columns
            if column not in {"time_s", "run_id"}
            and not any(token in column.lower() for token in DENY_COLUMN_TOKENS)
        ]
        compared_columns = columns
        frames.append(frame[columns].fillna("<NA>").astype(str))
        sample_ids.append(int(item["sample_index"]))
    if len(frames) < 2:
        return {"status": "INSUFFICIENT_EXISTING_FILES"}
    reference = frames[0]
    equality = [reference.equals(frame) for frame in frames[1:]]
    return {
        "status": "PASS",
        "case_id": candidate[0]["case_id"],
        "prefault_rows_compared": safe_rows,
        "telemetry_columns_compared": len(compared_columns or []),
        "sample_indices": sample_ids,
        "all_non_identity_prefault_values_equal": all(equality),
        "pairwise_equal_to_first": equality,
    }


def audit(sat_root: Path, dataset_names: list[str], prefault_rows: int) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for name in dataset_names:
        root = sat_root / name
        telemetry_counts = {
            filename: sum(1 for _ in root.rglob(filename)) if root.exists() else 0
            for filename in TELEMETRY_NAMES
        }
        datasets[name] = {
            "exists": root.exists(),
            "telemetry_counts": telemetry_counts,
            **(_metadata_inventory(root) if root.exists() else {}),
        }
        plan = _sample_plan_inventory(root) if root.exists() else None
        if plan is not None:
            datasets[name]["sample_plan"] = plan

    generator = _generator_seed_binding(sat_root / "generate_multi_sample_dataset.py")
    prefault = _prefault_identity_probe(sat_root / "dataset", prefault_rows)
    hard_findings: list[str] = []
    plan = datasets.get("dataset", {}).get("sample_plan", {})
    if plan.get("unique_magnitude_count_per_case") == {"1": plan.get("case_count")}:
        hard_findings.append("FULL_DATASET_ONE_MAGNITUDE_PER_CASE")
    if plan.get("declared_seed_count", 0) and plan.get("seed_consumed_true_count", 0) == 0:
        hard_findings.append("EXISTING_DATASET_HAS_NO_CONSUMED_SEED_ATTESTATION")
    if generator.get("sample_seed_declared") and not generator.get("sample_seed_referenced_by_run_one"):
        hard_findings.append("CURRENT_GENERATOR_DOES_NOT_CONSUME_DECLARED_SAMPLE_SEED")
    if prefault.get("all_non_identity_prefault_values_equal"):
        hard_findings.append("MULTI_SAMPLE_PREFAULT_TRAJECTORIES_IDENTICAL_IN_PROBE")
    supplement = datasets.get("dataset_supplements", {})
    if supplement.get("validation_status", {}).get("PASS", 0) > supplement.get("source_status", {}).get("PASS", 0):
        hard_findings.append("SUPPLEMENT_TOP_LEVEL_PASS_MASKS_SOURCE_FAILURES")
    return {
        "schema_version": "astrograph-three-stage-input-audit.v1",
        "read_only": True,
        "sat_root": str(sat_root),
        "datasets": datasets,
        "generator_seed_binding": generator,
        "prefault_identity_probe": prefault,
        "hard_findings": hard_findings,
        "claim_boundary": (
            "Inventory and deterministic identity checks only; no model fit, threshold selection, "
            "simulation, accuracy recomputation, or production capability claim."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sat-root", type=Path, default=DEFAULT_SAT_ROOT)
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--prefault-rows", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.sat_root.resolve(), args.datasets or list(DEFAULT_DATASETS), args.prefault_rows)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
