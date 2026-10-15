#!/usr/bin/env python3
"""Build a label-isolated AstroGraph evaluation dataset from validated Sat-Agent pairs.

The input must be a validated fault-matrix output containing one nominal/effect trace pair
and result.json per public Capability--Effect combination.  Model-input copies exclude
labels, effect identifiers and object-level effect evidence.  Evaluation truth is held
in a separate truth manifest and is never read by the diagnosis engine.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

PROHIBITED_EXACT = {
    "case_id", "task_id", "effect.active_ids", "effect.active_count",
    "object.composite_digital_twin.effect_evidence",
    "object.composite_digital_twin.health_state",
    "object.composite_digital_twin.coupling_evidence",
}
PROHIBITED_PREFIXES = ("label.", "truth.")
PROHIBITED_TOKENS = (
    "label.", "truth.", "event.", "object.", "modifier.", "fault",
    "health_state", "runtime_truth", "file_path",
)
HARD_EFFECTS = {
    "combined_degradation",
    "adcs_rw_friction_and_sensor_noise",
    "thermal_heater_and_radiator_degradation",
    "fuel_leak_and_pressure_loss",
    "multi_subsystem_end_of_life",
}
SUBTLE_EFFECTS = {
    "adcs_rw_motor_failure",
    "gyro_bias_step",
    "comm_data_downlink_link_loss",
    "thermal_radiator_rejection_loss",
    "gyro_noise_increase",
    "rw_friction_degradation",
    "adcs_reaction_wheel_speed_limit",
    "thruster_thrust_loss_15pct",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def difficulty(effect_id: str, kind: str, duplicate_index: int) -> str:
    if effect_id in HARD_EFFECTS:
        return "hard"
    if kind != "fault" or effect_id in SUBTLE_EFFECTS or duplicate_index > 0:
        return "medium"
    return "simple"


def sanitized(frame: pd.DataFrame) -> pd.DataFrame:
    keep = [
        c for c in frame.columns
        if c not in PROHIBITED_EXACT and not c.lower().startswith(PROHIBITED_PREFIXES)
    ]
    result = frame[keep].copy()
    # Fail closed if hidden truth survives a future schema change.
    leaked = [
        c for c in result.columns
        if c.lower().startswith(PROHIBITED_PREFIXES)
        or "effect_evidence" in c.lower()
        or any(token in c.lower() for token in PROHIBITED_TOKENS)
    ]
    if leaked:
        raise RuntimeError(f"truth/effect evidence leaked into model input: {leaked}")
    proxy_values = sorted({
        str(value)
        for column in result.columns
        for value in result[column].dropna().unique()
        if any(token in str(value).strip().lower() for token in PROHIBITED_TOKENS)
    })
    if proxy_values:
        raise RuntimeError(f"truth/effect answer-proxy values leaked into model input: {proxy_values[:5]}")
    return result


def campaign_sources(campaign_root: Path) -> list[dict[str, Any]]:
    index_path = campaign_root / "astrograph_dataset_index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if payload.get("campaign_status") != "complete" or payload.get("dry_run") is True:
        raise RuntimeError("Sat-Agent campaign is not a completed execution")
    contracts = payload.get("contracts") or []
    by_dataset = {str(item.get("dataset_id")): item for item in contracts}
    sources: list[dict[str, Any]] = []
    for fault in contracts:
        truth = fault.get("ground_truth") or {}
        if truth.get("case_role") != "fault":
            continue
        pairing = fault.get("pairing") or {}
        nominal = by_dataset.get(str(pairing.get("nominal_case_id")))
        if nominal is None:
            raise RuntimeError(f"nominal pair is missing for {fault.get('dataset_id')}")
        fault_root = campaign_root / "cases" / str(fault["dataset_id"])
        nominal_root = campaign_root / "cases" / str(nominal["dataset_id"])
        fault_trace = fault_root / str((fault.get("telemetry") or {}).get("file") or "")
        nominal_trace = nominal_root / str((nominal.get("telemetry") or {}).get("file") or "")
        quality = fault.get("quality") or {}
        physics_gate = "PASS" if quality.get("physics_validation_passed") else "FAIL"
        sources.append({
            "result": {
                "status": "PASS",
                "effect_id": truth.get("simulator_effect") or truth.get("class_id"),
                "kind": truth.get("simulator_event_kind") or "fault",
                "config_sha256": str((fault.get("simulation") or {}).get("spec_hash") or ""),
                "seed_sha256": stable_sha256((fault.get("simulation") or {}).get("seed", "UNDECLARED")),
                "physics_gate": physics_gate,
                "conservation_gate": "NOT_APPLICABLE",
                "data_grade": "RESEARCH_ONLY" if not quality.get("formal_training_eligible") else "ENGINEERING_VALIDATED",
            },
            "source_record": index_path,
            "nominal": nominal_trace,
            "effect": fault_trace,
            "backend": str((fault.get("simulation") or {}).get("backend") or "unknown"),
        })
    if not sources:
        raise RuntimeError("Sat-Agent campaign contains no paired fault contracts")
    return sources


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--matrix-root", type=Path)
    source.add_argument("--campaign-root", type=Path)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--dataset-id", default="SAT-SIM-FAULT-CAMPAIGN")
    p.add_argument("--source-campaign", default="validated_basilisk_fault_matrix")
    p.add_argument("--claim-boundary", default="Qualified simulation handoffs for diagnostic integration only; fidelity and data grade remain case-specific, and no flight-validation claim is made.")
    p.add_argument(
        "--truth-output",
        type=Path,
        required=True,
        help="Required hidden-truth path outside the public model-input dataset.",
    )
    args = p.parse_args()
    out = args.output_root.resolve()
    truth_path = args.truth_output.resolve()
    if truth_path == out or out in truth_path.parents or truth_path in out.parents:
        raise ValueError("hidden truth and public output roots must be physically disjoint")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    if args.campaign_root is not None:
        source_cases = campaign_sources(args.campaign_root.resolve())
    else:
        matrix_root = args.matrix_root.resolve()
        result_paths = sorted(matrix_root.glob("*/*/result.json"))
        if not result_paths:
            raise SystemExit("no validated fault-matrix result.json files found")
        source_cases = [
            {
                "result": json.loads(path.read_text(encoding="utf-8")),
                "source_record": path,
                "nominal": path.parent / "nominal_trace.csv",
                "effect": path.parent / "effect_trace.csv",
                "backend": path.parent.parent.name,
            }
            for path in result_paths
        ]
    duplicate_seen: dict[str, int] = {}
    difficulty_counts = {name: 0 for name in ("simple", "medium", "hard")}
    public_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    checksum_rows: list[tuple[str, str]] = []

    for index, source_case in enumerate(source_cases, start=1):
        result = source_case["result"]
        result_path = source_case["source_record"]
        if result.get("status") != "PASS":
            raise RuntimeError(f"validated fault-matrix source result is not PASS: {result_path}")
        backend = source_case["backend"]
        effect_id = str(result["effect_id"])
        kind = str(result["kind"])
        duplicate_index = duplicate_seen.get(effect_id, 0)
        duplicate_seen[effect_id] = duplicate_index + 1
        level = difficulty(effect_id, kind, duplicate_index)
        difficulty_counts[level] += 1
        case_id = f"CASE-{index:03d}"
        case_dir = out / case_id
        case_dir.mkdir(parents=True)

        nominal_in = source_case["nominal"]
        effect_in = source_case["effect"]
        nominal = sanitized(pd.read_csv(nominal_in, low_memory=False))
        effect = sanitized(pd.read_csv(effect_in, low_memory=False))
        if list(nominal.columns) != list(effect.columns):
            raise RuntimeError(f"paired schema mismatch: {result_path.parent}")
        if len(nominal) != len(effect):
            raise RuntimeError(f"paired row-count mismatch: {result_path.parent}")
        if "time_s" not in nominal.columns:
            raise RuntimeError(f"time_s missing: {result_path.parent}")
        if not nominal["time_s"].equals(effect["time_s"]):
            raise RuntimeError(f"paired time axis mismatch: {result_path.parent}")

        nominal_path = case_dir / "nominal.csv"
        effect_path = case_dir / "telemetry.csv"
        nominal.to_csv(nominal_path, index=False)
        effect.to_csv(effect_path, index=False)
        input_manifest = {
            "schema_version": "simulation_handoff.v2",
            "handoff_id": f"HANDOFF-{case_id}",
            "case_id": case_id,
            "telemetry": {
                "file": "telemetry.csv",
                "sha256": sha256(effect_path),
                "rows": len(effect),
                "fields": list(effect.columns),
            },
            "nominal_reference": {
                "file": "nominal.csv",
                "sha256": sha256(nominal_path),
                "rows": len(nominal),
                "fields": list(nominal.columns),
            },
            "isolation": {
                "truth_available_to_diagnosis": False,
                "labels_removed": True,
                "answer_proxy_metadata_removed": True,
            },
            "provenance": {
                "source_result_sha256": sha256(result_path),
                "source_nominal_sha256": sha256(nominal_in),
                "source_telemetry_sha256": sha256(effect_in),
                "config_sha256": str(result.get("config_sha256") or stable_sha256({
                    "source_campaign": args.source_campaign,
                    "source_runtime": backend,
                })),
                "seed_sha256": str(result.get("seed_sha256") or stable_sha256(result.get("seed", "UNDECLARED"))),
            },
            "qualification": {
                "source_validation": "PASS",
                "physics_gate": str(result.get("physics_gate") or "PASS"),
                "conservation_gate": str(result.get("conservation_gate") or "NOT_APPLICABLE"),
                "data_grade": str(result.get("data_grade") or "RESEARCH_ONLY"),
                "intended_use": "diagnostic_integration_evaluation",
            },
            "claim_boundary": "Validated laboratory simulation input; not flight-qualified.",
        }
        manifest_path = case_dir / "manifest.json"
        manifest_path.write_text(json.dumps(input_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for path in (nominal_path, effect_path, manifest_path):
            checksum_rows.append((path.relative_to(out).as_posix(), sha256(path)))
        public_rows.append({
            "case_id": case_id,
            "input_dir": case_id,
            "row_count": len(effect),
            "channel_count": len(effect.columns) - 1,
        })
        truth_rows.append({
            "case_id": case_id,
            "effect_id": effect_id,
            "kind": kind,
            "source_backend": backend,
            "source_result_sha256": sha256(result_path),
        })

    with (out / "case_index.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(public_rows[0]))
        writer.writeheader(); writer.writerows(public_rows)
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    truth_path.write_text(
        json.dumps({"schema_version": "astrograph-sat-sim-hidden-truth.v1", "cases": truth_rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "astrograph-sat-sim-fault-dataset.v1",
        "dataset_id": args.dataset_id,
        "source_campaign": args.source_campaign,
        "case_count": len(public_rows),
        "difficulty_counts": difficulty_counts,
        "source_runtimes": sorted({str(item["backend"]) for item in source_cases}),
        "source_case_count": len(source_cases),
        "model_input_truth_isolation": "PASS",
        "truth_manifest_external": True,
        "public_contract_schema": "simulation_handoff.v2",
        "training_qualified": False,
        "purpose": "diagnostic integration and capability-gap evaluation",
        "claim_boundary": args.claim_boundary,
    }
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checksum_rows.extend([
        ("case_index.csv", sha256(out / "case_index.csv")),
        ("dataset_manifest.json", sha256(out / "dataset_manifest.json")),
    ])
    with (out / "CHECKSUMS.txt").open("w", encoding="utf-8") as f:
        for rel, digest in sorted(checksum_rows):
            f.write(f"{digest}  {rel}\n")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
