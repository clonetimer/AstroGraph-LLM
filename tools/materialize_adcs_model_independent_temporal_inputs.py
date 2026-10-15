#!/usr/bin/env python3
"""Materialize label-isolated ADCS temporal inputs from a fixed dataset archive.

This command is the only stage that sees source metadata/labels.  It writes a
neutral runtime manifest and opaque telemetry files separately from an
evaluator-only truth table.  The trace generator receives only the runtime
manifest and therefore cannot infer fault identity from filenames or paths.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
from pathlib import Path
import re
import zipfile

import pandas as pd

from astrograph.assurance import canonical_sha256
from astrograph.experiments.model_independent_temporal import (
    ModelIndependentTruthRow,
    ModelIndependentTruthTable,
)

DATASET_PREFIX = "whole_spacecraft_adcs_paired_dataset_v5_2_2/"
DEVELOPMENT_GROUPS = (3, 8)
SCALE_FIT_GROUPS = (0, 1, 2, 5, 6, 7)
EXCLUDED_GROUPS = (4, 9)
FORBIDDEN_COLUMNS = ("fault", "label", "truth", "event", "condition", "target", "y_true")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def opaque(prefix: str, value: str, length: int) -> str:
    return prefix + hashlib.sha256(value.encode("utf-8")).hexdigest()[:length].upper()


def read_json_member(archive: zipfile.ZipFile, name: str) -> tuple[dict, bytes]:
    raw = archive.read(name)
    return json.loads(raw), raw


def filtered_telemetry(raw: bytes) -> tuple[pd.DataFrame, tuple[str, ...]]:
    frame = pd.read_csv(BytesIO(raw))
    channels = tuple(column for column in frame.columns if column.startswith("adcs."))
    if not channels:
        raise ValueError("telemetry contains no adcs.* channels")
    if any(any(token in column.lower() for token in FORBIDDEN_COLUMNS) for column in channels):
        raise ValueError("truth-like telemetry channel detected")
    result = frame.loc[:, ["time_s", *channels]].copy()
    if result["time_s"].duplicated().any() or not result["time_s"].is_monotonic_increasing:
        raise ValueError("telemetry time_s must be unique and monotonic")
    return result, channels


def write_csv(path: Path, frame: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")
    return sha256_file(path)


def label_truth(raw: bytes, metadata: dict, trajectory_id: str, group_id: str) -> ModelIndependentTruthRow:
    labels = pd.read_csv(BytesIO(raw))
    if len(labels) == 0:
        raise ValueError("labels table is empty")
    if set(labels["sample_id"].astype(str)) != {str(metadata["sample_id"])}:
        raise ValueError("metadata/labels sample binding differs")
    is_nominal_values = set(bool(int(value)) for value in labels["is_nominal"])
    if len(is_nominal_values) != 1:
        raise ValueError("is_nominal changes within a trajectory")
    is_nominal = next(iter(is_nominal_values))
    diagnostic = set(labels["diagnostic_class"].astype(str))
    if len(diagnostic) != 1:
        raise ValueError("diagnostic class changes within a trajectory")
    diagnostic_class = next(iter(diagnostic))
    active = labels[labels["event_active"].astype(float) > 0.5]
    if is_nominal:
        start = end = None
        if len(active):
            raise ValueError("nominal trajectory contains active fault labels")
    else:
        if len(active) == 0:
            raise ValueError("fault trajectory has no active interval")
        start = float(active["time_s"].iloc[0])
        end = float(active["time_s"].iloc[-1])
    return ModelIndependentTruthRow(
        opaque_trajectory_id=trajectory_id,
        independent_group_id=group_id,
        development_group=int(metadata["sample_index"]),
        is_nominal=is_nominal,
        diagnostic_class=diagnostic_class,
        active_start_s=start,
        active_end_s=end,
        source_metadata_sha256="0" * 64,  # replaced by caller
        source_labels_sha256=sha256_bytes(raw),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--runtime-output", type=Path, required=True)
    parser.add_argument("--evaluator-output", type=Path, required=True)
    args = parser.parse_args()

    actual_sha = sha256_file(args.archive)
    if actual_sha != args.expected_archive_sha256.lower():
        raise SystemExit(f"archive SHA-256 mismatch: {actual_sha}")
    if zipfile.ZipFile(args.archive).testzip() is not None:
        raise SystemExit("archive CRC validation failed")

    runtime_root = args.runtime_output.resolve()
    evaluator_root = args.evaluator_output.resolve()
    runtime_root.mkdir(parents=True, exist_ok=True)
    evaluator_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.archive) as archive:
        metadata_members = sorted(name for name in archive.namelist() if name.endswith("/sample_metadata.json"))
        metadata_records: list[tuple[str, dict, bytes]] = []
        for name in metadata_members:
            metadata, metadata_raw = read_json_member(archive, name)
            metadata_records.append((name, metadata, metadata_raw))

        nominal: dict[tuple[str, int], tuple[str, dict, bytes]] = {}
        development: list[tuple[str, dict, bytes]] = []
        for record in metadata_records:
            name, metadata, _ = record
            group = int(metadata["sample_index"])
            if metadata["condition_id"] == "nominal":
                nominal[(str(metadata["mission_id"]), group)] = record
                if group in DEVELOPMENT_GROUPS:
                    development.append(record)
            elif group in DEVELOPMENT_GROUPS:
                development.append(record)

        missions = sorted({mission for mission, _ in nominal})
        expected_nominal_keys = {(mission, group) for mission in missions for group in range(10)}
        if not expected_nominal_keys <= set(nominal):
            raise ValueError("nominal reference inventory is incomplete")

        telemetry_dir = runtime_root / "telemetry"
        reference_files: dict[tuple[str, int], dict] = {}
        scale_fit_entries: list[dict] = []
        channel_order: tuple[str, ...] | None = None

        required_reference_keys = {(mission, group) for mission in missions for group in (*SCALE_FIT_GROUPS, *DEVELOPMENT_GROUPS)}
        for mission, group in sorted(required_reference_keys):
            metadata_name, metadata, _ = nominal[(mission, group)]
            telemetry_member = metadata_name.removesuffix("sample_metadata.json") + "telemetry.csv"
            telemetry_raw = archive.read(telemetry_member)
            frame, channels = filtered_telemetry(telemetry_raw)
            if channel_order is None:
                channel_order = channels
            elif channel_order != channels:
                raise ValueError("ADCS channel order differs across reference telemetry")
            reference_id = opaque("REF-", f"{actual_sha}:{mission}:{group}", 20)
            output_path = telemetry_dir / f"{reference_id}.csv"
            output_sha = write_csv(output_path, frame)
            family_id = opaque("FAM-", mission, 16)
            record = {
                "reference_id": reference_id,
                "reference_family_id": family_id,
                "group": group,
                "relative_path": output_path.relative_to(runtime_root).as_posix(),
                "sha256": output_sha,
                "row_count": len(frame),
                "channel_count": len(channels),
            }
            reference_files[(mission, group)] = record
            if group in SCALE_FIT_GROUPS:
                scale_fit_entries.append(record)

        if channel_order is None:
            raise ValueError("no reference telemetry was materialized")

        truth_rows: list[ModelIndependentTruthRow] = []
        runtime_entries: list[dict] = []
        mapping_entries: list[dict] = []
        for metadata_name, metadata, metadata_raw in sorted(development, key=lambda item: item[0]):
            group = int(metadata["sample_index"])
            if group not in DEVELOPMENT_GROUPS:
                raise ValueError("materializer selected a non-development trajectory")
            mission = str(metadata["mission_id"])
            source_key = metadata_name.removesuffix("sample_metadata.json")
            telemetry_member = source_key + "telemetry.csv"
            labels_member = source_key + "labels.csv"
            telemetry_raw = archive.read(telemetry_member)
            labels_raw = archive.read(labels_member)
            frame, channels = filtered_telemetry(telemetry_raw)
            if channel_order != channels:
                raise ValueError("ADCS channel order differs across current telemetry")
            trajectory_id = opaque("TRJ-", f"{actual_sha}:{metadata_name}", 20)
            group_id = opaque("GRP-", str(metadata["trajectory_pair_id"]), 16)
            current_path = telemetry_dir / f"{trajectory_id}.csv"
            current_sha = write_csv(current_path, frame)
            reference = reference_files[(mission, group)]
            alternate_group = 8 if group == 3 else 3
            alternate = reference_files[(mission, alternate_group)]
            runtime_entries.append(
                {
                    "opaque_trajectory_id": trajectory_id,
                    "independent_group_id": group_id,
                    "development_group": group,
                    "current_relative_path": current_path.relative_to(runtime_root).as_posix(),
                    "current_sha256": current_sha,
                    "reference_relative_path": reference["relative_path"],
                    "reference_sha256": reference["sha256"],
                    "alternate_reference_relative_path": alternate["relative_path"],
                    "alternate_reference_sha256": alternate["sha256"],
                }
            )
            row = label_truth(labels_raw, metadata, trajectory_id, group_id)
            row = row.model_copy(update={"source_metadata_sha256": sha256_bytes(metadata_raw)})
            truth_rows.append(row)
            mapping_entries.append(
                {
                    "opaque_trajectory_id": trajectory_id,
                    "source_metadata_member": metadata_name,
                    "source_telemetry_member": telemetry_member,
                    "source_labels_member": labels_member,
                    "source_metadata_sha256": sha256_bytes(metadata_raw),
                    "source_telemetry_sha256": sha256_bytes(telemetry_raw),
                    "source_labels_sha256": sha256_bytes(labels_raw),
                }
            )

    scale_fit_manifest = {
        "schema_version": "astrograph-model-independent-scale-fit-manifest.v1",
        "dataset_archive_sha256": actual_sha,
        "scale_fit_groups": list(SCALE_FIT_GROUPS),
        "excluded_groups": list(EXCLUDED_GROUPS),
        "channel_order": list(channel_order),
        "entries": scale_fit_entries,
        "ground_truth_fields_present": False,
    }
    scale_fit_manifest["manifest_sha256"] = canonical_sha256(scale_fit_manifest)
    write_json(runtime_root / "scale_fit_manifest.json", scale_fit_manifest)

    runtime_manifest = {
        "schema_version": "astrograph-model-independent-runtime-manifest.v1",
        "dataset_archive_sha256": actual_sha,
        "development_groups": list(DEVELOPMENT_GROUPS),
        "excluded_groups": list(EXCLUDED_GROUPS),
        "channel_order": list(channel_order),
        "entries": runtime_entries,
        "ground_truth_fields_present": False,
        "source_paths_exposed": False,
        "claim_boundary": [
            "Runtime filenames and trajectory identifiers are opaque.",
            "No diagnostic class, fault label, event onset, or active interval is present.",
            "Groups 4/9 are absent and cannot be used for policy selection.",
        ],
    }
    runtime_manifest["manifest_sha256"] = canonical_sha256(runtime_manifest)
    write_json(runtime_root / "runtime_manifest.json", runtime_manifest)

    truth_table = ModelIndependentTruthTable(
        dataset_archive_sha256=actual_sha,
        rows=tuple(sorted(truth_rows, key=lambda row: row.opaque_trajectory_id)),
    )
    write_json(evaluator_root / "truth_table.json", truth_table.model_dump(mode="json"))
    mapping = {
        "schema_version": "astrograph-model-independent-materialization-receipt.v1",
        "dataset_archive_sha256": actual_sha,
        "runtime_manifest_sha256": runtime_manifest["manifest_sha256"],
        "scale_fit_manifest_sha256": scale_fit_manifest["manifest_sha256"],
        "truth_table_sha256": truth_table.truth_table_sha256,
        "runtime_output": str(runtime_root),
        "evaluator_output": str(evaluator_root),
        "entries": mapping_entries,
        "runtime_truth_isolation": "PASS",
        "groups_4_9_materialized": False,
    }
    mapping["receipt_sha256"] = canonical_sha256(mapping)
    write_json(evaluator_root / "materialization_receipt.json", mapping)
    print(json.dumps({
        "runtime_manifest": str(runtime_root / "runtime_manifest.json"),
        "runtime_manifest_sha256": runtime_manifest["manifest_sha256"],
        "truth_table": str(evaluator_root / "truth_table.json"),
        "truth_table_sha256": truth_table.truth_table_sha256,
        "trajectory_count": len(runtime_entries),
        "channel_count": len(channel_order),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
