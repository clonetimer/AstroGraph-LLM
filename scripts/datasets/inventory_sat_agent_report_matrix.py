#!/usr/bin/env python3
"""Audit broad Sat-Agent report-derived datasets as Tier-C design evidence.

Only ``reports/**/results/dataset/manifest.json`` bundles are considered.  The
script reads aggregate manifest metadata and safe physical columns from sibling
``trace.csv`` files to detect duplicates.  It never loads label/truth/proxy column
values and emits no case, run, dataset, seed, path, or raw-row identifiers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from astrograph.data_sources import is_sat_agent_model_channel_safe  # noqa: E402


def _value(payload: dict[str, Any], *keys: str, fallback: str = "UNDECLARED") -> str:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return fallback
        value = value.get(key)
    return fallback if value in (None, "") else str(value)


def _contained_file(root: Path, path: Path) -> bool:
    try:
        path.resolve(strict=True).relative_to(root.resolve())
    except (FileNotFoundError, OSError, ValueError):
        return False
    return path.resolve().is_file()


def _physical_trace_fingerprint(
    path: Path, source_root: Path
) -> tuple[str | None, int, int, list[str], list[str], str]:
    if not _contained_file(source_root, path):
        return None, 0, 0, [], [], "TRACE_PATH_ESCAPE_OR_NOT_REGULAR"
    try:
        columns = [str(item) for item in pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns]
    except EmptyDataError:
        return None, 0, 0, [], [], "TRACE_EMPTY"
    except UnicodeDecodeError:
        return None, 0, 0, [], [], "TRACE_DECODE_ERROR"
    except OSError:
        return None, 0, 0, [], [], "TRACE_IO_ERROR"
    except ValueError:
        return None, 0, 0, [], [], "TRACE_HEADER_VALUE_ERROR"
    time_fields = [item for item in columns if item.lower() in {"time_s", "timestamp", "time"}]
    safe_signal_fields = [
        item
        for item in columns
        if item not in time_fields and is_sat_agent_model_channel_safe(item)
    ]
    usecols = [*time_fields[:1], *safe_signal_fields]
    prohibited = sorted(set(columns) - set(usecols))
    if not usecols or not safe_signal_fields:
        return None, 0, len(columns), prohibited, safe_signal_fields, "NO_SAFE_SIGNAL_FIELDS"
    try:
        frame = pd.read_csv(path, usecols=usecols, encoding="utf-8-sig")
    except EmptyDataError:
        return None, 0, len(columns), prohibited, safe_signal_fields, "TRACE_EMPTY"
    except UnicodeDecodeError:
        return None, 0, len(columns), prohibited, safe_signal_fields, "TRACE_DECODE_ERROR"
    except OSError:
        return None, 0, len(columns), prohibited, safe_signal_fields, "TRACE_IO_ERROR"
    except ValueError:
        return None, 0, len(columns), prohibited, safe_signal_fields, "TRACE_BODY_VALUE_ERROR"
    canonical = frame[sorted(frame.columns)].to_csv(
        index=False, lineterminator="\n", float_format="%.17g"
    )
    return (
        hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        len(frame),
        len(columns),
        prohibited,
        safe_signal_fields,
        "OK",
    )


def build_matrix(source_root: Path) -> dict[str, Any]:
    manifests = sorted(
        (source_root / "reports").glob("**/results/dataset/manifest.json")
    )
    valid_manifest_count = 0
    invalid_errors: Counter[str] = Counter()
    dataset_ids: set[str] = set()
    physical_hashes: set[str] = set()
    total_safe_rows_read = 0
    trace_readable_count = 0
    trace_missing_or_unreadable_count = 0
    trace_error_counts: Counter[str] = Counter()
    all_prohibited_names: Counter[str] = Counter()
    fingerprint_annotations: dict[str, set[tuple[str, str, str, str]]] = defaultdict(set)
    groups: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "manifest_count": 0,
            "readable_trace_count": 0,
            "safe_physical_row_count": 0,
            "physical_trace_hashes": set(),
            "mode_counts": Counter(),
            "status_counts": Counter(),
            "safe_channel_names": set(),
            "prohibited_field_names": set(),
        }
    )

    for manifest_path in manifests:
        if not _contained_file(source_root, manifest_path):
            invalid_errors["MANIFEST_PATH_ESCAPE_OR_NOT_REGULAR"] += 1
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            if not isinstance(manifest, dict):
                raise ValueError("root is not an object")
        except Exception as exc:
            invalid_errors[type(exc).__name__] += 1
            continue
        valid_manifest_count += 1
        dataset_id = manifest.get("dataset_id")
        if isinstance(dataset_id, str) and dataset_id:
            dataset_ids.add(dataset_id)

        level = _value(manifest, "labels", "target_level")
        target = _value(manifest, "labels", "target_name")
        capability = _value(manifest, "task", "capability_id")
        mode = _value(manifest, "labels", "mode")
        status = _value(manifest, "status")
        group = groups[(level, target, capability)]
        group["manifest_count"] += 1
        group["mode_counts"][mode] += 1
        group["status_counts"][status] += 1

        trace = manifest_path.parent / "trace.csv"
        if not trace.is_file():
            trace_missing_or_unreadable_count += 1
            trace_error_counts["TRACE_MISSING"] += 1
            continue
        fingerprint, row_count, _column_count, prohibited, safe_fields, trace_status = (
            _physical_trace_fingerprint(trace, source_root)
        )
        group["prohibited_field_names"].update(prohibited)
        all_prohibited_names.update(prohibited)
        if fingerprint is None:
            trace_missing_or_unreadable_count += 1
            trace_error_counts[trace_status] += 1
            continue
        trace_readable_count += 1
        total_safe_rows_read += row_count
        physical_hashes.add(fingerprint)
        fingerprint_annotations[fingerprint].add((level, target, capability, mode))
        group["readable_trace_count"] += 1
        group["safe_physical_row_count"] += row_count
        group["physical_trace_hashes"].add(fingerprint)

        group["safe_channel_names"].update(safe_fields)

    matrix = []
    for (level, target, capability), group in sorted(groups.items()):
        matrix.append(
            {
                "hierarchy_level": level,
                "target_name": target,
                "capability_id": capability,
                "usage_tier": "C_DESIGN_ONLY_ENGINEERING_PREVIEW",
                "manifest_count": group["manifest_count"],
                "readable_trace_count": group["readable_trace_count"],
                "unique_physical_trace_count": len(group["physical_trace_hashes"]),
                "safe_physical_row_count": group["safe_physical_row_count"],
                "mode_counts": dict(sorted(group["mode_counts"].items())),
                "status_counts": dict(sorted(group["status_counts"].items())),
                "safe_channel_names": sorted(group["safe_channel_names"]),
                "prohibited_field_names_present": sorted(group["prohibited_field_names"]),
                "training_eligible": False,
            }
        )

    level_counts = Counter(item["hierarchy_level"] for item in matrix)
    target_counts = Counter(item["target_name"] for item in matrix)
    return {
        "schema_version": "astrograph.p018.sat-agent-report-matrix.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage": "P018_S0_ZERO_FIT",
        "source_id": "SAT_AGENT_PLATFORM",
        "adapter_key": "sat_agent_platform",
        "source_family_alias": "sat-agent-platform",
        "source_count": 1,
        "scope": "reports/**/results/dataset only",
        "summary": {
            "manifest_file_count": len(manifests),
            "valid_manifest_count": valid_manifest_count,
            "invalid_manifest_count": len(manifests) - valid_manifest_count,
            "invalid_manifest_error_counts": dict(sorted(invalid_errors.items())),
            "declared_unique_dataset_id_count": len(dataset_ids),
            "readable_trace_count": trace_readable_count,
            "missing_or_unreadable_trace_count": trace_missing_or_unreadable_count,
            "trace_error_counts": dict(sorted(trace_error_counts.items())),
            "unique_physical_trace_count": len(physical_hashes),
            "duplicate_physical_trace_count": trace_readable_count - len(physical_hashes),
            "physical_trace_cross_annotation_conflict_count": sum(
                1 for annotations in fingerprint_annotations.values() if len(annotations) > 1
            ),
            "safe_physical_row_count_read_for_hashing": total_safe_rows_read,
            "matrix_row_count": len(matrix),
            "hierarchy_level_target_counts": dict(sorted(level_counts.items())),
            "distinct_target_name_count": len(target_counts),
            "prohibited_field_name_counts": dict(sorted(all_prohibited_names.items())),
        },
        "matrix": matrix,
        "execution_receipt": {
            "read_only": True,
            "raw_rows_emitted": False,
            "trace_label_truth_or_proxy_values_loaded": False,
            "trace_label_truth_or_proxy_values_emitted": False,
            "manifest_aggregate_label_metadata_loaded": True,
            "manifest_aggregate_label_metadata_emitted": True,
            "formal_hidden_truth_reader_invoked": False,
            "case_dataset_run_path_or_seed_values_emitted": False,
            "model_fit_count": 0,
            "threshold_selection_count": 0,
            "seed_creation_count": 0,
        },
        "readiness": {
            "stage_s1_training_eligible": False,
            "reason": (
                "Short report-derived runs are duplicated, contain prohibited fields, and lack a "
                "frozen multi-seed development/calibration campaign. They support schema and "
                "archetype design only."
            ),
        },
        "claim_boundary": (
            "Broad component coverage is Tier-C same-source design evidence only. Counts are not "
            "independent datasets or acceptance samples and establish no cross-source or flight generalization."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_root = args.root.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    payload = build_matrix(source_root)
    output = args.output.expanduser().resolve()
    try:
        output.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise ValueError("report-matrix output must not be placed inside the source tree")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "summary": payload["summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
