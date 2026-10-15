#!/usr/bin/env python3
"""Register Sat-Agent D3 ZIP exports without extracting rows or reading labels.

The utility reads only aggregate dataset/split/reproducibility JSON members and
hashes each archive.  Run IDs, raw telemetry, per-run labels and simulator
internal evidence are never emitted.  Both archives remain one Sat-Agent source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


MAX_CONTROL_MEMBER_BYTES = 4 * 1024 * 1024
MAX_CONTROL_COMPRESSION_RATIO = 200.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_json_member(archive: zipfile.ZipFile, member_name: str) -> dict[str, Any]:
    _safe_member_parts(member_name)
    info = archive.getinfo(member_name)
    if info.file_size > MAX_CONTROL_MEMBER_BYTES:
        raise ValueError(f"control member exceeds size limit: {member_name}")
    ratio = info.file_size / max(info.compress_size, 1)
    if ratio > MAX_CONTROL_COMPRESSION_RATIO:
        raise ValueError(f"control member exceeds compression-ratio limit: {member_name}")
    payload = json.loads(archive.read(info).decode("utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"control member root must be an object: {member_name}")
    return payload


def _unique_suffix(names: list[str], suffix: str, *, required: bool = True) -> str | None:
    matches = [name for name in names if name.lower().endswith(suffix.lower())]
    if not matches and not required:
        return None
    if len(matches) != 1:
        raise ValueError(f"expected exactly one *{suffix}, found {len(matches)}")
    return matches[0]


def _split_counts(payload: dict[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for key, value in payload.items():
        if key.endswith("_run_ids") and isinstance(value, list):
            result[key.removesuffix("_run_ids")] = len(value)
    return dict(sorted(result.items()))


def _split_id_sets(payload: dict[str, Any]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    seen: set[str] = set()
    for key, value in payload.items():
        if not key.endswith("_run_ids"):
            continue
        if not isinstance(value, list) or not all(
            isinstance(item, str) and bool(item) for item in value
        ):
            raise ValueError(f"split member {key} must be a list of non-empty run IDs")
        if len(value) != len(set(value)):
            raise ValueError(f"split member {key} contains duplicate run IDs")
        overlap = seen & set(value)
        if overlap:
            raise ValueError("run IDs overlap across split partitions")
        partition = key.removesuffix("_run_ids")
        result[partition] = set(value)
        seen.update(value)
    if not result:
        raise ValueError("split manifest contains no run-ID partitions")
    return result


def _safe_member_parts(name: str) -> tuple[str, ...]:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or any(":" in part for part in path.parts):
        raise ValueError(f"unsafe archive member path: {name}")
    return tuple(part for part in path.parts if part not in {"", "."})


def _materialized_run_manifest_ids(names: list[str], *, kind: str) -> set[str]:
    marker = "03_run_audit" if kind == "RW_D3" else "runs"
    run_ids: set[str] = set()
    for name in names:
        parts = _safe_member_parts(name)
        if not parts or parts[-1].lower() != "run_manifest.json":
            continue
        try:
            marker_index = next(
                index for index, part in enumerate(parts) if part.lower() == marker.lower()
            )
        except StopIteration:
            continue
        if marker_index + 2 != len(parts) - 1:
            continue
        run_id = parts[marker_index + 1]
        if not run_id or run_id in run_ids:
            raise ValueError("duplicate or empty materialized run-manifest identity")
        run_ids.add(run_id)
    return run_ids


def _archive_record(path: Path, *, kind: str) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        manifest_name = _unique_suffix(names, "dataset_manifest.json")
        split_name = _unique_suffix(names, "dataset_split_manifest.json")
        manifest = _safe_json_member(archive, str(manifest_name))
        split = _safe_json_member(archive, str(split_name))
        reproduction_name = _unique_suffix(
            names, "reproducibility_report.json", required=False
        )
        reproduction = (
            _safe_json_member(archive, reproduction_name)
            if reproduction_name is not None
            else None
        )

        if kind == "RW_D3":
            class_counts = manifest.get("independent_run_counts", {})
            planned = manifest.get("planned_run_count")
            complete = manifest.get("complete_run_count")
            replayability = "BITWISE_EVIDENCE_PRESENT"
            expected_reproduction = True
        elif kind == "THERMAL_STAR_D3":
            class_counts = manifest.get("run_counts", {})
            planned = sum(class_counts.values()) if isinstance(class_counts, dict) else None
            complete = planned
            replayability = "CONDITIONAL_SOURCE_AND_ENVIRONMENT_MATCH_REQUIRED"
            expected_reproduction = False
        else:
            raise ValueError(f"unsupported Sat-Agent export kind: {kind}")

        if not isinstance(class_counts, dict) or not all(
            isinstance(key, str) and isinstance(value, int)
            for key, value in class_counts.items()
        ):
            raise ValueError("dataset manifest has invalid aggregate class counts")
        split_counts = _split_counts(split)
        split_id_sets = _split_id_sets(split)
        split_run_ids = set().union(*split_id_sets.values())
        materialized_run_ids = _materialized_run_manifest_ids(names, kind=kind)
        if sum(split_counts.values()) != int(complete or 0):
            raise ValueError("split counts do not equal complete run count")
        if sum(class_counts.values()) != int(complete or 0):
            raise ValueError("class counts do not equal complete run count")
        if len(materialized_run_ids) != int(complete or 0):
            raise ValueError("materialized run-manifest count does not equal complete run count")
        if split_run_ids != materialized_run_ids:
            raise ValueError("split run IDs do not exactly match materialized run manifests")
        if expected_reproduction and reproduction is None:
            raise ValueError("RW D3 export is missing its reproducibility report")

        reproduction_summary: dict[str, Any]
        if reproduction is None:
            reproduction_summary = {
                "report_present": False,
                "status": "NOT_MATERIALIZED_IN_ARCHIVE",
                "passed": False,
                "comparison_count": 0,
            }
        else:
            comparisons = reproduction.get("runs", [])
            reproduction_summary = {
                "report_present": True,
                "status": reproduction.get("status"),
                "passed": bool(reproduction.get("passed", False)),
                "comparison": reproduction.get("comparison"),
                "comparison_count": len(comparisons) if isinstance(comparisons, list) else 0,
                "all_recorded_comparisons_equal": bool(
                    isinstance(comparisons, list)
                    and comparisons
                    and all(
                        isinstance(item, dict) and item.get("equal") is True
                        for item in comparisons
                    )
                ),
            }

        return {
            "asset_kind": kind,
            "source_group_id": "SAT_AGENT_PLATFORM",
            "archive_token": f"${{ASTROGRAPH_SAT_AGENT_EXPORT_ROOT}}/{path.name}",
            "archive_size_bytes": path.stat().st_size,
            "archive_sha256": _sha256(path),
            "zip_member_count": len(names),
            "dataset_id": manifest.get("dataset_id"),
            "campaign_stage": manifest.get("campaign_stage"),
            "dataset_status": manifest.get("status"),
            "planned_run_count": planned,
            "complete_run_count": complete,
            "materialized_run_manifest_count": len(materialized_run_ids),
            "complete_run_count_verified_by_run_manifests": True,
            "aggregate_class_counts": dict(sorted(class_counts.items())),
            "aggregate_split_counts": split_counts,
            "split_unit": split.get("split_unit"),
            "historical_split_seed_present": "seed" in split,
            "historical_split_seed_value_emitted": False,
            "field_contract_sha256": manifest.get("field_contract_sha256"),
            "replayability": replayability,
            "reproducibility_summary": reproduction_summary,
            "source_claim_boundary": manifest.get("claim_boundary", ""),
            "model_input_policy": {
                "raw_telemetry_only": True,
                "labels_are_evaluator_only": True,
                "simulation_internal_evidence_prohibited": True,
                "run_or_split_ids_prohibited": True,
                "method_use_now": "INVENTORY_ONLY_NO_FIT",
            },
        }


def build_registry(rw_archive: Path, thermal_star_archive: Path) -> dict[str, Any]:
    if rw_archive.resolve() == thermal_star_archive.resolve():
        raise ValueError("RW and thermal/star exports must be distinct archives")
    assets = [
        _archive_record(rw_archive, kind="RW_D3"),
        _archive_record(thermal_star_archive, kind="THERMAL_STAR_D3"),
    ]
    if assets[0]["archive_sha256"] == assets[1]["archive_sha256"]:
        raise ValueError("RW and thermal/star exports have identical archive content")
    return {
        "schema_version": "astrograph.p018.sat-agent-export-registry.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage": "P018_S0_ZERO_FIT",
        "source_id": "SAT_AGENT_PLATFORM",
        "source_count": 1,
        "assets": assets,
        "aggregate": {
            "asset_count": len(assets),
            "complete_run_count": sum(int(item["complete_run_count"]) for item in assets),
            "model_fit_count": 0,
            "truth_or_label_row_read_count": 0,
            "raw_telemetry_row_read_count": 0,
            "aggregate_manifest_class_counts_read": True,
            "split_run_ids_read_for_integrity_but_not_emitted": True,
            "seed_created_or_reserved_count": 0,
        },
        "readiness": {
            "tier": "A_HISTORICAL_MULTI_RUN_ENGINEERING_EXPORT",
            "stage_s1_fit_authorized": False,
            "stage_b_authorized": False,
            "taxonomy_mapping_frozen": False,
            "fresh_split_claimed": False,
            "reason": (
                "The exports provide repeated engineering runs but do not by themselves freeze "
                "their mapping to the G1.1 taxonomy or authorize reuse of historical splits for S1."
            ),
        },
        "claim_boundary": (
            "Both exports were generated by one Sat-Agent platform source. They are not two "
            "independent sources and provide no flight, hardware, cross-simulator, or real-world evidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rw-archive", type=Path, required=True)
    parser.add_argument("--thermal-star-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path in (args.rw_archive, args.thermal_star_archive):
        if not path.is_file():
            raise FileNotFoundError(path)
    rw_archive = args.rw_archive.resolve()
    thermal_archive = args.thermal_star_archive.resolve()
    output = args.output.resolve()
    if output in {rw_archive, thermal_archive}:
        raise ValueError("registry output must not overwrite an input archive")
    payload = build_registry(rw_archive, thermal_archive)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "aggregate": payload["aggregate"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
