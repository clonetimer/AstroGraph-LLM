from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import shutil
import stat
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from astrograph.filesystem import publish_staged_directory

ARM_ORDER = [
    "A0_ALGORITHM_ONLY",
    "A1_LATE_LLM",
    "A2_CONTROLLED_FULL",
    "A3_NO_KG_OBSERVATION",
    "A4_NO_RAG_OAG_CONTEXT",
    "A5_NO_EXPERIENCE_CONTEXT",
    "A6_NO_OBSERVATION_FEEDBACK",
    "A7_NO_ACTIVE_PROBE",
]
TEST_SAMPLE_INDICES = {4, 9}
DESIGN_RELATIVE_PATH = Path("configs/evaluation/llm_method_acceptance.v4.json")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_confirmatory_design(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    design = _read_json(root / DESIGN_RELATIVE_PATH)
    if design.get("schema_version") != "astrograph-llm-method-acceptance-campaign.v4":
        raise ValueError("unsupported LLM method acceptance campaign schema")
    if design.get("arm_order") != ARM_ORDER:
        raise ValueError("confirmatory arm order differs from frozen A0-A7 order")
    if design.get("qualification_status") != "PREEXECUTION_GATES_FROZEN":
        raise ValueError("confirmatory design is not in PREEXECUTION_GATES_FROZEN state")
    thresholds = design.get("promotion_thresholds")
    probe_thresholds = design.get("probe_promotion_thresholds")
    statistics = design.get("statistics")
    if not isinstance(thresholds, dict) or not thresholds:
        raise ValueError("promotion thresholds must be frozen before confirmatory execution")
    if not isinstance(probe_thresholds, dict) or not probe_thresholds:
        raise ValueError("probe-specific promotion thresholds must be frozen before confirmatory execution")
    if not isinstance(statistics, dict) or statistics.get("inferential_unit") != "group_id":
        raise ValueError("confirmatory inference must be preregistered on group_id")
    if statistics.get("primary_test") != "EXACT_TWO_SIDED_GROUP_DIRECTION_SIGN":
        raise ValueError("unsupported confirmatory group-aware primary test")
    inference_profile_path = design.get("inference_profile_path")
    if not isinstance(inference_profile_path, str) or not inference_profile_path.strip():
        raise ValueError("confirmatory inference profile path must be frozen")
    if not (root / inference_profile_path).is_file():
        raise ValueError("confirmatory inference profile is missing")
    reproducibility = design.get("reproducibility_identity")
    if not isinstance(reproducibility, dict) or not reproducibility.get("runtime_attestation_required"):
        raise ValueError("R16 runtime identity policy must be frozen")
    if not reproducibility.get("post_execution_identity_stable_required"):
        raise ValueError("R16 post-execution runtime identity recheck must be frozen")
    return design


def default_dataset_roots(root: str | Path) -> dict[str, Path]:
    root = Path(root).resolve()
    return {
        "adcs": Path(os.environ.get(
            "ASTROGRAPH_ADCS_V522_DATA_ROOT",
            root / "data/external/whole_spacecraft_adcs_paired_dataset_v5_2_2",
        )),
        "phm": Path(os.environ.get(
            "ASTROGRAPH_PHM_V60_DATA_ROOT",
            root / "data/external/whole_spacecraft_phm_dataset_v6_0",
        )),
    }


def _mount_access_status(path: Path) -> dict[str, Any]:
    """Best-effort Linux mount access audit for an experiment dataset root.

    This is deliberately an admission/audit signal, not a portability claim.  On platforms
    without ``/proc/self/mountinfo`` the result is UNKNOWN rather than guessing from
    ``os.access`` (which is unreliable for privileged users).
    """
    path = path.resolve()
    mountinfo = Path("/proc/self/mountinfo")
    if not mountinfo.is_file():
        return {"status": "UNKNOWN", "mount_point": None, "options": [], "detail": "mountinfo unavailable"}
    best: tuple[int, str, list[str]] | None = None
    try:
        for line in mountinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split()
            if len(fields) < 7 or "-" not in fields:
                continue
            mount_point = fields[4].replace("\\040", " ")
            options = fields[5].split(",")
            try:
                mp = Path(mount_point).resolve()
                if path != mp and mp not in path.parents:
                    continue
                score = len(str(mp))
            except OSError:
                continue
            if best is None or score > best[0]:
                best = (score, str(mp), options)
    except OSError as exc:
        return {"status": "UNKNOWN", "mount_point": None, "options": [], "detail": f"{type(exc).__name__}:{exc}"}
    if best is None:
        return {"status": "UNKNOWN", "mount_point": None, "options": [], "detail": "no covering mount found"}
    _, mount_point, options = best
    status = "READ_ONLY" if "ro" in options else ("READ_WRITE" if "rw" in options else "UNKNOWN")
    return {"status": status, "mount_point": mount_point, "options": options, "detail": None}


def _dataset_contract_status(
    path: Path,
    *,
    dataset: str,
    project_root: Path,
    expected_fault_case_count: int,
    manifest_cases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate that a mounted confirmatory dataset is actually runnable, not just named correctly.

    Admission checks stay metadata-only: sample-index schema/identity, exact mapped test rows,
    paired nominal identities, and the regular files referenced by those rows.  Telemetry content
    is not loaded into prompts and no inference is executed here.
    """
    path = path.resolve()
    sample_index = path / "sample_index.csv"
    sample_sha256 = _sha256_file(sample_index) if sample_index.is_file() else None
    result: dict[str, Any] = {
        "root": str(path),
        "root_exists": path.is_dir(),
        "sample_index_exists": sample_index.is_file(),
        "sample_index_sha256": sample_sha256,
        "sample_index_size_bytes": (sample_index.stat().st_size if sample_index.is_file() else None),
        "mount_access": _mount_access_status(path),
        "ready": False,
        "mount_audit_status": "NOT_CHECKED",
        "selected_fault_row_count": 0,
        "selected_nominal_sample_count": 0,
        "checked_file_count": 0,
        "payload_file_count": 0,
        "payload_identity_sha256": None,
        "payload_entries": [],
        "errors": [],
        "missing_files": [],
        "empty_files": [],
        "path_escape_files": [],
    }
    if not path.is_dir() or not sample_index.is_file():
        result["errors"] = ["DATASET_ROOT_OR_SAMPLE_INDEX_MISSING"]
        return result

    rows = _read_csv(sample_index)
    if not rows:
        result["errors"] = ["SAMPLE_INDEX_EMPTY"]
        result["mount_audit_status"] = "INVALID"
        return result
    if dataset == "adcs":
        required = {"sample_id", "trajectory_pair_id", "sample_index", "scenario_id", "diagnostic_class", "telemetry_file", "labels_file"}
        class_key = "diagnostic_class"
        class_map = _load_class_map(project_root / "configs/datasets/adcs_v5_2_2/fault_class_map.json")
    elif dataset == "phm":
        required = {
            "sample_id", "sample_index", "subsystem", "condition_id", "activation_time_s",
            "event_start_s", "fault_manifestation_time_s", "fault_injection_phase", "telemetry_file",
        }
        class_key = "condition_id"
        class_map = _load_class_map(project_root / "configs/datasets/phm_v6_0/fault_class_map.json")
    else:  # pragma: no cover - internal caller is frozen
        raise ValueError(f"unknown confirmatory dataset: {dataset}")

    errors: list[str] = []
    headers = set(rows[0])
    missing_headers = sorted(required - headers)
    if missing_headers:
        errors.append("SAMPLE_INDEX_SCHEMA_MISMATCH:" + ",".join(missing_headers))

    sample_ids = [str(row.get("sample_id") or "") for row in rows]
    if any(not value for value in sample_ids) or len(sample_ids) != len(set(sample_ids)):
        errors.append("SAMPLE_INDEX_SAMPLE_ID_INVALID")
    by_sample = {str(row.get("sample_id") or ""): row for row in rows if str(row.get("sample_id") or "")}

    selected: list[dict[str, str]] = []
    for row in rows:
        try:
            sample_index_value = int(row.get("sample_index") or -1)
        except (TypeError, ValueError):
            errors.append("SAMPLE_INDEX_VALUE_INVALID")
            continue
        if str(row.get(class_key) or "") in class_map and sample_index_value in TEST_SAMPLE_INDICES:
            selected.append(row)
    result["selected_fault_row_count"] = len(selected)
    if len(selected) != expected_fault_case_count:
        errors.append(f"FROZEN_TEST_SELECTION_COUNT_MISMATCH:{len(selected)}!={expected_fault_case_count}")

    selected_nominals: dict[str, dict[str, str]] = {}
    for row in selected:
        if dataset == "adcs":
            pair = str(row.get("trajectory_pair_id") or "")
            candidates = [x for x in rows if x.get("diagnostic_class") == "nominal" and str(x.get("trajectory_pair_id") or "") == pair]
        else:
            subsystem = str(row.get("subsystem") or "")
            try:
                sample_index_value = int(row.get("sample_index") or -1)
            except (TypeError, ValueError):
                sample_index_value = -1
            candidates = [
                x for x in rows
                if x.get("condition_id") == "nominal"
                and str(x.get("subsystem") or "") == subsystem
                and int(x.get("sample_index") or -1) == sample_index_value
            ]
        if len(candidates) != 1:
            errors.append(f"PAIRED_NOMINAL_IDENTITY_INVALID:{row.get('sample_id')}")
            continue
        selected_nominals[str(candidates[0].get("sample_id"))] = candidates[0]
    result["selected_nominal_sample_count"] = len(selected_nominals)

    if manifest_cases is not None:
        manifest_dataset_cases = [item for item in manifest_cases if item.get("dataset") == dataset]
        manifest_fault_ids = {str(item.get("sample_id") or "") for item in manifest_dataset_cases}
        manifest_nominal_ids = {str(item.get("nominal_sample_id") or "") for item in manifest_dataset_cases}
        selected_ids = {str(item.get("sample_id") or "") for item in selected}
        nominal_ids = set(selected_nominals)
        if manifest_fault_ids != selected_ids:
            errors.append("CASE_MANIFEST_FAULT_SAMPLE_IDENTITY_MISMATCH")
        if manifest_nominal_ids != nominal_ids:
            errors.append("CASE_MANIFEST_NOMINAL_SAMPLE_IDENTITY_MISMATCH")
        for item in manifest_dataset_cases:
            row = by_sample.get(str(item.get("sample_id") or ""))
            if row is None:
                continue
            mapped = class_map.get(str(row.get(class_key) or ""))
            if mapped != str(item.get("true_fault_id") or ""):
                errors.append(f"CASE_MANIFEST_FAULT_MAPPING_MISMATCH:{item.get('case_id')}")

    rows_to_check = {str(row.get("sample_id")): row for row in [*selected, *selected_nominals.values()] if row.get("sample_id")}
    missing_files: list[str] = []
    empty_files: list[str] = []
    path_escape_files: list[str] = []
    checked = 0
    payload_identities: list[dict[str, Any]] = []
    payload_hash_cache: dict[Path, str] = {}
    for sample_id, row in sorted(rows_to_check.items()):
        fields = ["telemetry_file"] + (["labels_file"] if dataset == "adcs" else [])
        for field in fields:
            rel = str(row.get(field) or "").strip()
            if not rel:
                missing_files.append(f"{sample_id}:{field}:<empty>")
                continue
            candidate = (path / rel).resolve()
            if not candidate.is_relative_to(path):
                path_escape_files.append(f"{sample_id}:{field}:{rel}")
                continue
            checked += 1
            if not candidate.is_file():
                missing_files.append(f"{sample_id}:{field}:{rel}")
            elif candidate.stat().st_size <= 0:
                empty_files.append(f"{sample_id}:{field}:{rel}")
            else:
                payload_sha256 = payload_hash_cache.get(candidate)
                if payload_sha256 is None:
                    payload_sha256 = _sha256_file(candidate)
                    payload_hash_cache[candidate] = payload_sha256
                payload_identities.append({
                    "sample_id": sample_id,
                    "field": field,
                    "relative_path": candidate.relative_to(path).as_posix(),
                    "size_bytes": candidate.stat().st_size,
                    "sha256": payload_sha256,
                })
    result["checked_file_count"] = checked
    result["payload_file_count"] = len(payload_identities)
    result["payload_entries"] = payload_identities
    result["payload_identity_sha256"] = (
        _canonical_sha256(payload_identities) if payload_identities else None
    )
    result["missing_files"] = missing_files[:200]
    result["empty_files"] = empty_files[:200]
    result["path_escape_files"] = path_escape_files[:200]
    if missing_files:
        errors.append(f"REFERENCED_FILES_MISSING:{len(missing_files)}")
    if empty_files:
        errors.append(f"REFERENCED_FILES_EMPTY:{len(empty_files)}")
    if path_escape_files:
        errors.append(f"REFERENCED_FILE_PATH_ESCAPE:{len(path_escape_files)}")

    result["errors"] = sorted(set(errors))
    result["mount_audit_status"] = "VALID" if not errors else "INVALID"
    result["ready"] = not errors
    return result


def validate_confirmatory_case_manifest(
    root: str | Path,
    *,
    case_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the frozen case denominator without loading truth into runtime prompts.

    This is an execution-admission check only. It verifies campaign identity, denominator,
    uniqueness, dataset identity and the presence of the paired nominal identity declared in
    the preregistered manifest. It does not execute inference.
    """
    root = Path(root).resolve()
    design = load_confirmatory_design(root)
    path = Path(case_manifest_path) if case_manifest_path is not None else root / str(design["case_manifest_path"])
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        return {
            "status": "MISSING",
            "path": str(path),
            "sha256": None,
            "case_count": 0,
            "errors": ["CASE_MANIFEST_NOT_FROZEN"],
        }
    payload = _read_json(path)
    errors: list[str] = []
    if payload.get("schema_version") != "astrograph-llm-method-case-manifest.v1":
        errors.append("CASE_MANIFEST_SCHEMA_MISMATCH")
    if payload.get("campaign_id") != design["campaign_id"]:
        errors.append("CASE_MANIFEST_CAMPAIGN_MISMATCH")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        cases = []
        errors.append("CASE_MANIFEST_CASES_INVALID")
    expected_count = int(design["case_selection"]["expected_case_count"])
    if payload.get("case_count") != len(cases) or len(cases) != expected_count:
        errors.append("CASE_MANIFEST_DENOMINATOR_MISMATCH")
    case_ids = [str(item.get("case_id") or "") for item in cases if isinstance(item, dict)]
    if len(case_ids) != len(set(case_ids)) or any(not value for value in case_ids):
        errors.append("CASE_MANIFEST_CASE_ID_INVALID")
    for item in cases:
        if not isinstance(item, dict):
            continue
        required = ("dataset", "dataset_id", "dataset_class", "true_fault_id", "sample_id", "nominal_sample_id", "group_id")
        if not all(str(item.get(key) or "").strip() for key in required):
            errors.append("CASE_MANIFEST_IDENTITY_INCOMPLETE")
            break
        if item.get("dataset") not in {"adcs", "phm"}:
            errors.append("CASE_MANIFEST_DATASET_INVALID")
            break
    group_ids = [str(item.get("group_id") or "") for item in cases if isinstance(item, dict)]
    group_sizes: dict[str, int] = {}
    for group_id in group_ids:
        if group_id:
            group_sizes[group_id] = group_sizes.get(group_id, 0) + 1
    min_groups = int((design.get("statistics") or {}).get("minimum_independent_group_count") or 0)
    if min_groups and len(group_sizes) < min_groups:
        errors.append("CASE_MANIFEST_INSUFFICIENT_INDEPENDENT_GROUPS")
    return {
        "status": "VALID" if not errors else "INVALID",
        "path": str(path),
        "sha256": _sha256_file(path),
        "case_count": len(cases),
        "independent_group_count": len(group_sizes),
        "group_sizes": dict(sorted(group_sizes.items())),
        "case_identity_sha256": _canonical_sha256([
            {
                "case_id": item.get("case_id"),
                "dataset": item.get("dataset"),
                "dataset_id": item.get("dataset_id"),
                "sample_id": item.get("sample_id"),
                "nominal_sample_id": item.get("nominal_sample_id"),
                "group_id": item.get("group_id"),
            }
            for item in cases if isinstance(item, dict)
        ]),
        "errors": sorted(set(errors)),
    }


def _dataset_snapshot_identity(
    *, campaign_id: str, manifest_status: dict[str, Any], datasets: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    dataset_snapshots: dict[str, dict[str, Any]] = {}
    for key, status in sorted(datasets.items()):
        sample_index_entry = {
            "relative_path": "sample_index.csv",
            "size_bytes": status.get("sample_index_size_bytes"),
            "sha256": status.get("sample_index_sha256"),
        }
        payload_entries = sorted(
            [dict(item) for item in (status.get("payload_entries") or []) if isinstance(item, dict)],
            key=lambda item: (str(item.get("relative_path") or ""), str(item.get("sample_id") or ""), str(item.get("field") or "")),
        )
        identity_payload = {
            "dataset": key,
            "sample_index": sample_index_entry,
            "payload_entries": payload_entries,
        }
        dataset_snapshots[key] = {
            "snapshot_sha256": _canonical_sha256(identity_payload),
            "sample_index": sample_index_entry,
            "payload_file_count": len(payload_entries),
            "payload_identity_sha256": status.get("payload_identity_sha256"),
            "payload_entries": payload_entries,
            "mount_access": status.get("mount_access") or {"status": "UNKNOWN"},
        }
    content_identity = {
        "campaign_id": campaign_id,
        "case_manifest_sha256": manifest_status.get("sha256"),
        "case_identity_sha256": manifest_status.get("case_identity_sha256"),
        "datasets": {key: value["snapshot_sha256"] for key, value in sorted(dataset_snapshots.items())},
    }
    return {
        "snapshot_sha256": _canonical_sha256(content_identity),
        "content_identity": content_identity,
        "datasets": dataset_snapshots,
    }


def build_confirmatory_dataset_snapshot_receipt(
    *, readiness: dict[str, Any], require_readonly_datasets: bool = False
) -> dict[str, Any]:
    """Build a content-addressed receipt for the exact confirmatory dataset payloads.

    The snapshot hash is independent of absolute mount paths.  Absolute roots and mount
    access are retained only as audit metadata.  A writable mount can be recorded without
    invalidating ordinary engineering admission; callers can opt into a strict read-only
    filesystem gate with ``require_readonly_datasets=True``.
    """
    datasets = readiness.get("datasets") or {}
    manifest = readiness.get("case_manifest") or {}
    identity = _dataset_snapshot_identity(
        campaign_id=str(readiness.get("campaign_id") or ""),
        manifest_status=manifest,
        datasets=datasets,
    )
    errors: list[str] = []
    if manifest.get("status") != "VALID":
        errors.append("CASE_MANIFEST_NOT_VALID")
    for key, status in sorted(datasets.items()):
        if not status.get("ready"):
            errors.append(f"DATASET_NOT_READY:{key}")
        mount_status = str((status.get("mount_access") or {}).get("status") or "UNKNOWN")
        if require_readonly_datasets and mount_status != "READ_ONLY":
            errors.append(f"DATASET_NOT_READ_ONLY:{key}")
    receipt = {
        "schema_version": "astrograph-llm-method-confirmatory-dataset-snapshot-receipt.v1",
        "campaign_id": readiness.get("campaign_id"),
        "status": "VALID" if not errors else "INVALID",
        "snapshot_sha256": identity["snapshot_sha256"],
        "case_manifest_sha256": manifest.get("sha256"),
        "case_identity_sha256": manifest.get("case_identity_sha256"),
        "require_readonly_datasets": bool(require_readonly_datasets),
        "datasets": {
            key: {
                "root": status.get("root"),
                **identity["datasets"][key],
            }
            for key, status in sorted(datasets.items())
        },
        "errors": errors,
        "claim_boundary": (
            "Content-addressed confirmatory input snapshot only. The snapshot binds the frozen case denominator, sample indices and selected fault/nominal payload bytes; "
            "it is not a diagnostic-performance, safety or qualification result. READ_ONLY refers only to the observed filesystem mount access at admission time."
        ),
    }
    receipt["receipt_sha256"] = _canonical_sha256({k: v for k, v in receipt.items() if k != "receipt_sha256"})
    return receipt


def verify_confirmatory_dataset_snapshot_receipt(
    *, snapshot_receipt: dict[str, Any], current_readiness: dict[str, Any], require_readonly_datasets: bool | None = None
) -> dict[str, Any]:
    expected_readonly = (
        bool(snapshot_receipt.get("require_readonly_datasets"))
        if require_readonly_datasets is None else bool(require_readonly_datasets)
    )
    observed = build_confirmatory_dataset_snapshot_receipt(
        readiness=current_readiness, require_readonly_datasets=expected_readonly
    )
    drifts: list[str] = []
    expected_receipt_sha256 = _canonical_sha256({k: v for k, v in snapshot_receipt.items() if k != "receipt_sha256"})
    if snapshot_receipt.get("receipt_sha256") != expected_receipt_sha256:
        drifts.append("SNAPSHOT_RECEIPT_SHA256_INVALID")
    if snapshot_receipt.get("schema_version") != "astrograph-llm-method-confirmatory-dataset-snapshot-receipt.v1":
        drifts.append("SNAPSHOT_RECEIPT_SCHEMA_MISMATCH")
    if snapshot_receipt.get("campaign_id") != current_readiness.get("campaign_id"):
        drifts.append("SNAPSHOT_CAMPAIGN_ID_DRIFT")
    if snapshot_receipt.get("status") != "VALID":
        drifts.append("FROZEN_SNAPSHOT_RECEIPT_INVALID")
    if observed.get("status") != "VALID":
        drifts.append("CURRENT_DATASET_SNAPSHOT_INVALID")
        drifts.extend(f"CURRENT_SNAPSHOT_ERROR:{item}" for item in (observed.get("errors") or []))
    if snapshot_receipt.get("snapshot_sha256") != observed.get("snapshot_sha256"):
        drifts.append("DATASET_SNAPSHOT_SHA256_DRIFT")
    if snapshot_receipt.get("case_manifest_sha256") != observed.get("case_manifest_sha256"):
        drifts.append("SNAPSHOT_CASE_MANIFEST_DRIFT")
    if snapshot_receipt.get("case_identity_sha256") != observed.get("case_identity_sha256"):
        drifts.append("SNAPSHOT_CASE_IDENTITY_DRIFT")
    for key, frozen in sorted((snapshot_receipt.get("datasets") or {}).items()):
        current = (observed.get("datasets") or {}).get(key) or {}
        if frozen.get("snapshot_sha256") != current.get("snapshot_sha256"):
            drifts.append(f"DATASET_CONTENT_SNAPSHOT_DRIFT:{key}")
        if expected_readonly and str((current.get("mount_access") or {}).get("status")) != "READ_ONLY":
            drifts.append(f"DATASET_READONLY_POLICY_DRIFT:{key}")
    return {
        "schema_version": "astrograph-llm-method-confirmatory-dataset-snapshot-check.v1",
        "status": "STABLE" if not drifts else "DRIFTED",
        "frozen_snapshot_sha256": snapshot_receipt.get("snapshot_sha256"),
        "observed_snapshot_sha256": observed.get("snapshot_sha256"),
        "drift_codes": sorted(set(drifts)),
        "observed_receipt_sha256": observed.get("receipt_sha256"),
        "claim_boundary": "Dataset snapshot identity check only; STABLE is not a diagnostic-performance or qualification result.",
    }



def _assert_snapshot_receipt_integrity(snapshot_receipt: dict[str, Any]) -> None:
    expected = _canonical_sha256({k: v for k, v in snapshot_receipt.items() if k != "receipt_sha256"})
    if snapshot_receipt.get("schema_version") != "astrograph-llm-method-confirmatory-dataset-snapshot-receipt.v1":
        raise ValueError("unsupported confirmatory dataset snapshot receipt schema")
    if snapshot_receipt.get("receipt_sha256") != expected:
        raise ValueError("confirmatory dataset snapshot receipt SHA256 is invalid")
    if snapshot_receipt.get("status") != "VALID":
        raise ValueError("confirmatory dataset snapshot receipt is not VALID")


def _readonly_permission_status(root: Path) -> dict[str, Any]:
    writable: list[str] = []
    checked = 0
    if not root.exists():
        return {"status": "MISSING", "checked_path_count": 0, "writable_mode_paths": []}
    for path in [root, *sorted(root.rglob("*"))]:
        try:
            mode = path.stat().st_mode
        except OSError:
            writable.append(str(path))
            continue
        checked += 1
        if mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            writable.append(str(path.relative_to(root)) if path != root else ".")
    return {
        "status": "READ_ONLY_MODE_BITS" if not writable else "WRITABLE_MODE_BITS_PRESENT",
        "checked_path_count": checked,
        "writable_mode_paths": writable[:200],
    }


def _harden_tree_readonly_permissions(root: Path) -> dict[str, Any]:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    root.chmod(0o555)
    return _readonly_permission_status(root)


def materialize_confirmatory_dataset_snapshot(
    root: str | Path,
    *,
    snapshot_receipt: dict[str, Any],
    case_manifest_path: str | Path,
    output_root: str | Path,
    dataset_roots: dict[str, str | Path] | None = None,
    harden_readonly_permissions: bool = True,
) -> dict[str, Any]:
    """Copy the admitted confirmatory denominator into a content-addressed self-contained snapshot.

    Only the frozen case manifest, each dataset sample index, and payload files referenced by the
    selected fault/nominal rows are copied.  The output is re-admitted and must reproduce the exact
    path-independent dataset snapshot SHA256 before the materialization is committed.  Permission
    hardening removes write mode bits as an accidental-mutation guard; it is explicitly not a
    substitute for a true read-only filesystem mount.
    """
    project_root = Path(root).resolve()
    output_root = Path(output_root).resolve()
    case_manifest_path = Path(case_manifest_path).resolve()
    _assert_snapshot_receipt_integrity(snapshot_receipt)
    if output_root == project_root or output_root.is_relative_to(project_root):
        raise ValueError("materialized confirmatory snapshot must be outside the source tree")
    if output_root.exists():
        raise FileExistsError(f"materialized snapshot output already exists: {output_root}")
    if not case_manifest_path.is_file():
        raise FileNotFoundError(f"case manifest unavailable: {case_manifest_path}")
    if _sha256_file(case_manifest_path) != snapshot_receipt.get("case_manifest_sha256"):
        raise ValueError("case manifest SHA256 does not match frozen dataset snapshot receipt")

    source_roots = {
        str(key): Path(str(value.get("root") or "")).resolve()
        for key, value in sorted((snapshot_receipt.get("datasets") or {}).items())
        if isinstance(value, dict)
    }
    if dataset_roots:
        source_roots.update({str(key): Path(value).resolve() for key, value in dataset_roots.items()})
    if set(source_roots) != {"adcs", "phm"}:
        raise ValueError("snapshot receipt must bind exactly adcs and phm datasets")
    for key, source_root in source_roots.items():
        if output_root == source_root or output_root.is_relative_to(source_root):
            raise ValueError(f"materialized snapshot output cannot be nested inside source dataset: {key}")
    source_readiness = confirmatory_readiness(
        project_root,
        dataset_roots=source_roots,
        live_model_available=True,
        live_model_detail="snapshot-materializer-source-verification",
        case_manifest_path=case_manifest_path,
        require_readonly_datasets=False,
    )
    source_check = verify_confirmatory_dataset_snapshot_receipt(
        snapshot_receipt=snapshot_receipt, current_readiness=source_readiness, require_readonly_datasets=False
    )
    if source_check.get("status") != "STABLE":
        raise ValueError("source dataset bytes no longer match frozen snapshot receipt")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=str(output_root.parent)))
    committed = False
    try:
        materialized_manifest = staging / "case_manifest.json"
        shutil.copyfile(case_manifest_path, materialized_manifest)
        dataset_roots: dict[str, Path] = {}
        copied_files: dict[str, list[dict[str, Any]]] = {}
        for key, frozen in sorted((snapshot_receipt.get("datasets") or {}).items()):
            source_root = source_roots[str(key)]
            destination_root = staging / "datasets" / str(key)
            destination_root.mkdir(parents=True, exist_ok=True)
            dataset_roots[str(key)] = destination_root
            entries = [
                {
                    "relative_path": "sample_index.csv",
                    "size_bytes": (frozen.get("sample_index") or {}).get("size_bytes"),
                    "sha256": (frozen.get("sample_index") or {}).get("sha256"),
                    "kind": "sample_index",
                },
                *[
                    {**dict(item), "kind": "payload"}
                    for item in (frozen.get("payload_entries") or [])
                    if isinstance(item, dict)
                ],
            ]
            unique: dict[str, dict[str, Any]] = {}
            for entry in entries:
                rel = str(entry.get("relative_path") or "").strip()
                if not rel:
                    raise ValueError(f"snapshot receipt contains empty relative path for dataset {key}")
                unique.setdefault(rel, entry)
            copied_files[str(key)] = []
            for rel, entry in sorted(unique.items()):
                src = (source_root / rel).resolve()
                if not src.is_relative_to(source_root):
                    raise ValueError(f"snapshot source path escapes dataset root: {key}:{rel}")
                if not src.is_file():
                    raise FileNotFoundError(f"snapshot source file missing: {key}:{rel}")
                expected_size = entry.get("size_bytes")
                expected_sha = entry.get("sha256")
                if expected_size is not None and src.stat().st_size != int(expected_size):
                    raise ValueError(f"snapshot source size drift: {key}:{rel}")
                if expected_sha and _sha256_file(src) != str(expected_sha):
                    raise ValueError(f"snapshot source SHA256 drift: {key}:{rel}")
                dst = destination_root / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                copied_files[str(key)].append({
                    "relative_path": rel,
                    "size_bytes": dst.stat().st_size,
                    "sha256": _sha256_file(dst),
                    "kind": entry.get("kind"),
                })

        copied_readiness = confirmatory_readiness(
            project_root,
            dataset_roots=dataset_roots,
            live_model_available=True,
            live_model_detail="snapshot-materializer-content-verification",
            case_manifest_path=materialized_manifest,
            require_readonly_datasets=False,
        )
        copied_snapshot = build_confirmatory_dataset_snapshot_receipt(
            readiness=copied_readiness, require_readonly_datasets=False
        )
        if copied_snapshot.get("status") != "VALID":
            raise ValueError(f"materialized dataset snapshot is invalid: {copied_snapshot.get('errors')}")
        if copied_snapshot.get("snapshot_sha256") != snapshot_receipt.get("snapshot_sha256"):
            raise ValueError("materialized dataset snapshot identity differs from frozen source snapshot")

        governance_dir = staging / "snapshot_metadata"
        governance_dir.mkdir(parents=True, exist_ok=True)
        (governance_dir / "source_dataset_snapshot_receipt.json").write_text(
            json.dumps(snapshot_receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        # Rebind only audit roots to the final materialized location.  snapshot_sha256 is path-independent.
        final_snapshot = json.loads(json.dumps(copied_snapshot))
        for key in sorted(final_snapshot.get("datasets") or {}):
            final_snapshot["datasets"][key]["root"] = str(output_root / "datasets" / key)
        final_snapshot["receipt_sha256"] = _canonical_sha256(
            {k: v for k, v in final_snapshot.items() if k != "receipt_sha256"}
        )
        (governance_dir / "materialized_dataset_snapshot_receipt.json").write_text(
            json.dumps(final_snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        receipt = {
            "schema_version": "astrograph-llm-method-confirmatory-snapshot-materialization.v1",
            "campaign_id": snapshot_receipt.get("campaign_id"),
            "status": "VALID",
            "materialization_root": str(output_root),
            "case_manifest_relative_path": "case_manifest.json",
            "case_manifest_sha256": _sha256_file(materialized_manifest),
            "source_snapshot_sha256": snapshot_receipt.get("snapshot_sha256"),
            "materialized_snapshot_sha256": final_snapshot.get("snapshot_sha256"),
            "source_snapshot_receipt_sha256": snapshot_receipt.get("receipt_sha256"),
            "materialized_snapshot_receipt_sha256": final_snapshot.get("receipt_sha256"),
            "datasets": {
                key: {
                    "relative_root": f"datasets/{key}",
                    "copied_file_count": len(items),
                    "copied_files_sha256": _canonical_sha256(items),
                }
                for key, items in sorted(copied_files.items())
            },
            "permission_hardening_requested": bool(harden_readonly_permissions),
            "permission_hardening": {"status": "PENDING" if harden_readonly_permissions else "NOT_REQUESTED"},
            "claim_boundary": (
                "Self-contained content materialization for the frozen confirmatory denominator only. "
                "Removing write permission bits reduces accidental mutation but is not equivalent to a kernel-enforced read-only mount; "
                "strict confirmatory execution must still satisfy the independent read-only mount admission gate when required."
            ),
        }
        (governance_dir / "materialization_receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        publish_staged_directory(staging, output_root)
        committed = True
        if harden_readonly_permissions:
            hardening = _harden_tree_readonly_permissions(output_root)
        else:
            hardening = _readonly_permission_status(output_root)
        receipt["permission_hardening"] = hardening
        receipt["materialization_receipt_sha256"] = _canonical_sha256(
            {k: v for k, v in receipt.items() if k != "materialization_receipt_sha256"}
        )
        receipt_path = output_root / "snapshot_metadata" / "materialization_receipt.json"
        if harden_readonly_permissions:
            # Temporarily reopen only this metadata path to persist the observed hardening result.
            (output_root / "snapshot_metadata").chmod(0o755)
            receipt_path.chmod(0o644)
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if harden_readonly_permissions:
            receipt_path.chmod(0o444)
            (output_root / "snapshot_metadata").chmod(0o555)
            output_root.chmod(0o555)
        return receipt
    except Exception:
        target = output_root if committed else staging
        if target.exists():
            for path in [target, *target.rglob("*")]:
                try:
                    if path.is_dir():
                        path.chmod(0o755)
                    else:
                        path.chmod(0o644)
                except OSError:
                    pass
            shutil.rmtree(target, ignore_errors=True)
        raise

def confirmatory_readiness(
    root: str | Path,
    *,
    dataset_roots: dict[str, str | Path] | None = None,
    live_model_available: bool | None = None,
    live_model_detail: str | None = None,
    requested_model_id: str | None = None,
    observed_model_ids: list[str] | None = None,
    case_manifest_path: str | Path | None = None,
    require_readonly_datasets: bool = False,
    runtime_identity_admission: dict[str, Any] | None = None,
    enforce_runtime_identity: bool = False,
) -> dict[str, Any]:
    root = Path(root).resolve()
    design = load_confirmatory_design(root)
    roots = default_dataset_roots(root)
    if dataset_roots:
        roots.update({key: Path(value) for key, value in dataset_roots.items()})
    manifest_status = validate_confirmatory_case_manifest(root, case_manifest_path=case_manifest_path)
    resolved_case_manifest_path = Path(manifest_status["path"])
    manifest_cases: list[dict[str, Any]] | None = None
    if manifest_status["status"] == "VALID":
        payload = _read_json(resolved_case_manifest_path)
        manifest_cases = [item for item in (payload.get("cases") or []) if isinstance(item, dict)]
    expected_counts = {
        "adcs": int(design["case_selection"]["expected_adcs_case_count"]),
        "phm": int(design["case_selection"]["expected_phm_case_count"]),
    }
    datasets = {
        key: _dataset_contract_status(
            value, dataset=key, project_root=root, expected_fault_case_count=expected_counts[key],
            manifest_cases=manifest_cases,
        )
        for key, value in roots.items()
    }
    blockers: list[str] = []
    for key, status in datasets.items():
        if not status["root_exists"] or not status["sample_index_exists"]:
            blockers.append(f"DATASET_NOT_READY:{key}")
        elif not status["ready"]:
            blockers.append(f"DATASET_MOUNT_INVALID:{key}")
        if require_readonly_datasets and str((status.get("mount_access") or {}).get("status") or "UNKNOWN") != "READ_ONLY":
            blockers.append(f"DATASET_NOT_READ_ONLY:{key}")
    if manifest_status["status"] == "MISSING":
        blockers.append("CASE_MANIFEST_NOT_FROZEN")
    elif manifest_status["status"] != "VALID":
        blockers.extend(manifest_status["errors"])
    if live_model_available is False:
        blockers.append("LIVE_MODEL_UNAVAILABLE")
    elif live_model_available is None:
        blockers.append("LIVE_MODEL_NOT_PROBED")
    observed_model_ids = sorted(set(str(item) for item in (observed_model_ids or []) if str(item).strip()))
    if requested_model_id and live_model_available and requested_model_id not in observed_model_ids:
        blockers.append("REQUESTED_MODEL_NOT_ADVERTISED")
    if enforce_runtime_identity:
        if runtime_identity_admission is None:
            blockers.append("RUNTIME_IDENTITY_ADMISSION_MISSING")
        elif runtime_identity_admission.get("status") != "VALID":
            blockers.append("RUNTIME_IDENTITY_ADMISSION_BLOCKED")
            blockers.extend(f"RUNTIME_IDENTITY:{item}" for item in (runtime_identity_admission.get("blockers") or []))
    status = "READY" if not blockers else "BLOCKED"
    design_path = root / DESIGN_RELATIVE_PATH
    inference_profile_path = root / str(design["inference_profile_path"])
    return {
        "schema_version": "astrograph-llm-method-confirmatory-readiness.v6",
        "campaign_id": design["campaign_id"],
        "status": status,
        "design": {"path": str(DESIGN_RELATIVE_PATH), "sha256": _sha256_file(design_path)},
        "inference_profile": {"path": str(design["inference_profile_path"]), "sha256": _sha256_file(inference_profile_path)},
        "datasets": datasets,
        "declared_case_manifest_path": str(design["case_manifest_path"]),
        "case_manifest_path": str(resolved_case_manifest_path),
        "case_manifest_present": resolved_case_manifest_path.is_file(),
        "case_manifest": manifest_status,
        "live_model_available": live_model_available,
        "live_model_detail": live_model_detail,
        "requested_model_id": requested_model_id,
        "observed_model_ids": observed_model_ids,
        "readonly_dataset_policy": {
            "required": bool(require_readonly_datasets),
            "status": (
                "SATISFIED" if require_readonly_datasets and all(str((item.get("mount_access") or {}).get("status")) == "READ_ONLY" for item in datasets.values())
                else "NOT_REQUIRED" if not require_readonly_datasets else "UNSATISFIED"
            ),
        },
        "runtime_identity_policy": {
            "enforced": bool(enforce_runtime_identity),
            "admission_status": (runtime_identity_admission or {}).get("status") if enforce_runtime_identity else "NOT_ENFORCED",
            "runtime_identity_sha256": (runtime_identity_admission or {}).get("runtime_identity_sha256"),
            "model_runtime_attestation_sha256": (runtime_identity_admission or {}).get("model_runtime_attestation_sha256"),
            "admission": runtime_identity_admission if enforce_runtime_identity else None,
        },
        "dataset_snapshot": _dataset_snapshot_identity(campaign_id=str(design["campaign_id"]), manifest_status=manifest_status, datasets=datasets),
        "blockers": sorted(set(blockers)),
        "claim_boundary": (
            "READY means only that preregistered data/model availability and, when formally enforced, the R16 source/environment/model-runtime identity admission are satisfied. "
            "It is not an acceptance result and does not qualify the model or diagnostic runtime."
        ),
    }


EXPORT_MANIFEST_RELATIVE_PATH = Path("snapshot_metadata/export_manifest.json")
EXPORT_VERIFICATION_RECEIPT_RELATIVE_PATH = Path("snapshot_metadata/export_verification_receipt.json")


def _snapshot_export_entries(snapshot_root: Path) -> list[dict[str, Any]]:
    snapshot_root = snapshot_root.resolve()
    ignored = {EXPORT_MANIFEST_RELATIVE_PATH.as_posix(), EXPORT_VERIFICATION_RECEIPT_RELATIVE_PATH.as_posix()}
    entries: list[dict[str, Any]] = []
    for path in sorted(snapshot_root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"snapshot export cannot contain symlinks: {path.relative_to(snapshot_root)}")
        if not path.is_file():
            continue
        rel = path.relative_to(snapshot_root).as_posix()
        if rel in ignored:
            continue
        entries.append({
            "relative_path": rel,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        })
    return entries


def build_confirmatory_snapshot_export_manifest(snapshot_root: str | Path) -> dict[str, Any]:
    """Build a strict, path-independent file manifest for a materialized snapshot.

    The manifest is deliberately separate from the dataset snapshot receipt: the
    latter freezes the experimental denominator, while this export manifest freezes
    every file that must survive copying/archiving the self-contained snapshot tree.
    """
    snapshot_root = Path(snapshot_root).resolve()
    metadata_root = snapshot_root / "snapshot_metadata"
    materialization_path = metadata_root / "materialization_receipt.json"
    snapshot_receipt_path = metadata_root / "materialized_dataset_snapshot_receipt.json"
    case_manifest_path = snapshot_root / "case_manifest.json"
    required = [materialization_path, snapshot_receipt_path, case_manifest_path]
    missing = [str(path.relative_to(snapshot_root)) for path in required if not path.is_file()]
    if missing:
        raise ValueError("materialized snapshot is missing required export inputs: " + ",".join(missing))
    materialization = _read_json(materialization_path)
    dataset_snapshot = _read_json(snapshot_receipt_path)
    if materialization.get("status") != "VALID":
        raise ValueError("materialization receipt is not VALID")
    _assert_snapshot_receipt_integrity(dataset_snapshot)
    if materialization.get("materialized_snapshot_sha256") != dataset_snapshot.get("snapshot_sha256"):
        raise ValueError("materialization receipt and dataset snapshot receipt disagree")
    if materialization.get("case_manifest_sha256") != _sha256_file(case_manifest_path):
        raise ValueError("materialization receipt case manifest SHA256 does not match bytes on disk")

    entries = _snapshot_export_entries(snapshot_root)
    core = {
        "schema_version": "astrograph-llm-method-confirmatory-snapshot-export-manifest.v1",
        "status": "VALID",
        "campaign_id": materialization.get("campaign_id"),
        "dataset_snapshot_sha256": dataset_snapshot.get("snapshot_sha256"),
        "dataset_snapshot_receipt_sha256": dataset_snapshot.get("receipt_sha256"),
        "materialization_receipt_file_sha256": _sha256_file(materialization_path),
        "case_manifest_sha256": _sha256_file(case_manifest_path),
        "file_count": len(entries),
        "total_size_bytes": sum(int(item["size_bytes"]) for item in entries),
        "files": entries,
        "content_manifest_sha256": _canonical_sha256(entries),
        "claim_boundary": (
            "Path-independent export manifest for the self-contained confirmatory snapshot. "
            "It proves archive/copy byte identity; it does not replace the dataset snapshot admission, read-only mount gate, or model identity gate."
        ),
    }
    manifest = dict(core)
    manifest["manifest_sha256"] = _canonical_sha256(core)
    return manifest


def write_confirmatory_snapshot_export_manifest(snapshot_root: str | Path) -> dict[str, Any]:
    snapshot_root = Path(snapshot_root).resolve()
    manifest = build_confirmatory_snapshot_export_manifest(snapshot_root)
    path = snapshot_root / EXPORT_MANIFEST_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    # Materialized trees may have permission hardening enabled. Reopen only the
    # metadata directory/file long enough to persist the manifest, then restore it.
    parent_mode = stat.S_IMODE(path.parent.stat().st_mode) if path.parent.exists() else 0o755
    root_mode = stat.S_IMODE(snapshot_root.stat().st_mode)
    try:
        snapshot_root.chmod(root_mode | stat.S_IWUSR | stat.S_IXUSR)
        path.parent.chmod(parent_mode | stat.S_IWUSR | stat.S_IXUSR)
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        path.chmod(0o444)
    finally:
        try:
            path.parent.chmod(parent_mode)
            snapshot_root.chmod(root_mode)
        except OSError:
            pass
    return manifest


def verify_confirmatory_snapshot_export_manifest(
    snapshot_root: str | Path,
    *,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    snapshot_root = Path(snapshot_root).resolve()
    manifest_path = snapshot_root / EXPORT_MANIFEST_RELATIVE_PATH
    if manifest is None:
        if not manifest_path.is_file():
            return {
                "schema_version": "astrograph-llm-method-confirmatory-snapshot-export-verification.v1",
                "status": "DRIFTED",
                "drift_codes": ["EXPORT_MANIFEST_MISSING"],
            }
        manifest = _read_json(manifest_path)
    drift: list[str] = []
    if manifest.get("schema_version") != "astrograph-llm-method-confirmatory-snapshot-export-manifest.v1":
        drift.append("EXPORT_MANIFEST_SCHEMA_MISMATCH")
    expected_manifest_sha = manifest.get("manifest_sha256")
    core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if expected_manifest_sha != _canonical_sha256(core):
        drift.append("EXPORT_MANIFEST_SELF_HASH_MISMATCH")

    try:
        current = _snapshot_export_entries(snapshot_root)
    except ValueError:
        current = []
        drift.append("EXPORT_TREE_CONTAINS_SYMLINK")
    expected_entries = manifest.get("files") if isinstance(manifest.get("files"), list) else []
    expected_by_path = {str(item.get("relative_path")): item for item in expected_entries if isinstance(item, dict)}
    current_by_path = {str(item.get("relative_path")): item for item in current}
    missing = sorted(set(expected_by_path) - set(current_by_path))
    unexpected = sorted(set(current_by_path) - set(expected_by_path))
    changed = sorted(
        rel for rel in set(expected_by_path) & set(current_by_path)
        if int(expected_by_path[rel].get("size_bytes", -1)) != int(current_by_path[rel].get("size_bytes", -2))
        or str(expected_by_path[rel].get("sha256")) != str(current_by_path[rel].get("sha256"))
    )
    if missing:
        drift.append("EXPORT_FILE_MISSING")
    if unexpected:
        drift.append("EXPORT_UNEXPECTED_FILE")
    if changed:
        drift.append("EXPORT_FILE_CONTENT_DRIFT")
    if manifest.get("content_manifest_sha256") != _canonical_sha256(current):
        drift.append("EXPORT_CONTENT_MANIFEST_SHA256_DRIFT")

    result = {
        "schema_version": "astrograph-llm-method-confirmatory-snapshot-export-verification.v1",
        "status": "STABLE" if not drift else "DRIFTED",
        "manifest_sha256": expected_manifest_sha,
        "observed_content_manifest_sha256": _canonical_sha256(current),
        "drift_codes": sorted(set(drift)),
        "missing_files": missing,
        "unexpected_files": unexpected,
        "changed_files": changed,
        "file_count": len(current),
        "claim_boundary": (
            "Export verification checks the self-contained snapshot tree bytes only. Confirmatory execution must independently re-run dataset snapshot, model identity and read-only admission gates."
        ),
    }
    return result


def build_confirmatory_execution_receipt(
    root: str | Path,
    *,
    readiness: dict[str, Any],
    vllm_url: str,
    vllm_model: str,
    dataset_snapshot_receipt: dict[str, Any] | None = None,
    runtime_identity_admission: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze the exact pre-execution dataset and R16 runtime identities."""
    if readiness.get("status") != "READY":
        raise ValueError("confirmatory execution receipt requires READY admission")
    if runtime_identity_admission is None or runtime_identity_admission.get("status") != "VALID":
        raise ValueError("confirmatory execution receipt requires VALID R16 runtime identity admission")
    if runtime_identity_admission.get("schema_version") != "astrograph-confirmatory-runtime-identity-admission.v1":
        raise ValueError("confirmatory execution receipt received unsupported runtime identity schema")
    root = Path(root).resolve()
    design = load_confirmatory_design(root)
    if dataset_snapshot_receipt is not None:
        snapshot_check = verify_confirmatory_dataset_snapshot_receipt(
            snapshot_receipt=dataset_snapshot_receipt, current_readiness=readiness
        )
        if snapshot_check["status"] != "STABLE":
            raise ValueError("confirmatory dataset snapshot receipt does not match READY admission")
    dataset_fingerprints = {
        key: {
            "root": value.get("root"),
            "sample_index_sha256": value.get("sample_index_sha256"),
            "payload_identity_sha256": value.get("payload_identity_sha256"),
            "payload_file_count": value.get("payload_file_count"),
        }
        for key, value in sorted((readiness.get("datasets") or {}).items())
    }
    runtime_identity = runtime_identity_admission.get("runtime_identity") or {}
    client = runtime_identity_admission.get("client") or {}
    client_identity = client.get("identity") or {}
    payload = {
        "schema_version": "astrograph-llm-method-confirmatory-execution-receipt.v4",
        "admitted_at": datetime.now(timezone.utc).isoformat(),
        "campaign_id": design["campaign_id"],
        "design_sha256": (readiness.get("design") or {}).get("sha256"),
        "case_manifest_path": readiness.get("case_manifest_path"),
        "case_manifest_sha256": (readiness.get("case_manifest") or {}).get("sha256"),
        "case_identity_sha256": (readiness.get("case_manifest") or {}).get("case_identity_sha256"),
        "case_count": (readiness.get("case_manifest") or {}).get("case_count"),
        "dataset_fingerprints": dataset_fingerprints,
        "dataset_snapshot_sha256": (
            dataset_snapshot_receipt.get("snapshot_sha256") if dataset_snapshot_receipt is not None
            else (readiness.get("dataset_snapshot") or {}).get("snapshot_sha256")
        ),
        "dataset_snapshot_receipt_sha256": (
            dataset_snapshot_receipt.get("receipt_sha256") if dataset_snapshot_receipt is not None else None
        ),
        "readonly_dataset_policy_required": bool((readiness.get("readonly_dataset_policy") or {}).get("required")),
        "vllm_url": vllm_url.rstrip("/"),
        "vllm_model": vllm_model,
        "observed_model_ids": readiness.get("observed_model_ids") or [],
        "runtime_identity_sha256": runtime_identity_admission.get("runtime_identity_sha256"),
        "client_runtime_identity_sha256": runtime_identity.get("client_identity_sha256"),
        "model_runtime_identity_sha256": runtime_identity.get("model_runtime_identity_sha256"),
        "inference_config_sha256": runtime_identity.get("inference_config_sha256"),
        "client_environment_lock_sha256": client_identity.get("environment_lock_sha256"),
        "client_environment_packages_sha256": client_identity.get("environment_packages_sha256"),
        "client_cyclonedx_sbom_sha256": client_identity.get("environment_cyclonedx_sbom_sha256") or client_identity.get("cyclonedx_sbom_sha256"),
        "source_manifest_sha256": client_identity.get("source_manifest_sha256"),
        "source_checksums_sha256": client_identity.get("source_checksums_sha256"),
        "prompt_implementation_sha256": client_identity.get("prompt_implementation_sha256"),
        "inference_profile_file_sha256": client_identity.get("inference_profile_file_sha256"),
        "inference_profile_canonical_sha256": client_identity.get("inference_profile_canonical_sha256"),
        "model_runtime_attestation_sha256": runtime_identity_admission.get("model_runtime_attestation_sha256"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "arm_order": design["arm_order"],
        "promotion_thresholds": design["promotion_thresholds"],
        "probe_promotion_thresholds": design["probe_promotion_thresholds"],
        "claim_boundary": (
            "Pre-execution identity receipt only. R16 freezes dataset bytes, AstroGraph source/checksum identity, exact Python environment/SBOM, "
            "frozen inference/prompt configuration and content-bound model/tokenizer/server runtime identity. It is not a performance result, "
            "remote producer-authenticity proof, safety certification or qualification statement."
        ),
    }
    payload["receipt_sha256"] = _canonical_sha256(payload)
    return payload


def verify_confirmatory_execution_identity(
    *,
    execution_receipt: dict[str, Any],
    current_readiness: dict[str, Any],
    current_runtime_identity_admission: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare frozen pre-execution dataset/runtime identities with fresh post-run observations."""
    drifts: list[str] = []
    frozen_receipt_sha256 = str(execution_receipt.get("receipt_sha256") or "")
    receipt_core = dict(execution_receipt)
    receipt_core.pop("receipt_sha256", None)
    if not frozen_receipt_sha256 or frozen_receipt_sha256 != _canonical_sha256(receipt_core):
        drifts.append("EXECUTION_RECEIPT_SHA256_INVALID")
    if current_readiness.get("status") != "READY":
        drifts.append("POST_EXECUTION_READINESS_BLOCKED")
        drifts.extend(f"POST_BLOCKER:{item}" for item in (current_readiness.get("blockers") or []))
    if execution_receipt.get("campaign_id") != current_readiness.get("campaign_id"):
        drifts.append("CAMPAIGN_ID_DRIFT")
    if execution_receipt.get("design_sha256") != (current_readiness.get("design") or {}).get("sha256"):
        drifts.append("DESIGN_SHA256_DRIFT")
    manifest = current_readiness.get("case_manifest") or {}
    if execution_receipt.get("case_manifest_sha256") != manifest.get("sha256"):
        drifts.append("CASE_MANIFEST_SHA256_DRIFT")
    if execution_receipt.get("case_identity_sha256") != manifest.get("case_identity_sha256"):
        drifts.append("CASE_IDENTITY_DRIFT")
    if execution_receipt.get("case_count") != manifest.get("case_count"):
        drifts.append("CASE_COUNT_DRIFT")
    current_datasets = current_readiness.get("datasets") or {}
    for key, frozen in sorted((execution_receipt.get("dataset_fingerprints") or {}).items()):
        observed = current_datasets.get(key) or {}
        if frozen.get("sample_index_sha256") != observed.get("sample_index_sha256"):
            drifts.append(f"DATASET_SAMPLE_INDEX_DRIFT:{key}")
        if frozen.get("payload_identity_sha256") != observed.get("payload_identity_sha256"):
            drifts.append(f"DATASET_PAYLOAD_IDENTITY_DRIFT:{key}")
        if frozen.get("payload_file_count") != observed.get("payload_file_count"):
            drifts.append(f"DATASET_PAYLOAD_FILE_COUNT_DRIFT:{key}")
    current_snapshot_sha256 = (current_readiness.get("dataset_snapshot") or {}).get("snapshot_sha256")
    if execution_receipt.get("dataset_snapshot_sha256") and execution_receipt.get("dataset_snapshot_sha256") != current_snapshot_sha256:
        drifts.append("DATASET_CONTENT_SNAPSHOT_DRIFT")
    if execution_receipt.get("readonly_dataset_policy_required"):
        for key, value in sorted(current_datasets.items()):
            if str((value.get("mount_access") or {}).get("status") or "UNKNOWN") != "READ_ONLY":
                drifts.append(f"DATASET_READONLY_POLICY_DRIFT:{key}")
    requested_model = str(execution_receipt.get("vllm_model") or "")
    observed_models = {str(item) for item in (current_readiness.get("observed_model_ids") or [])}
    if requested_model and requested_model not in observed_models:
        drifts.append("MODEL_ENDPOINT_ADVERTISEMENT_DRIFT")

    current_runtime = current_runtime_identity_admission
    if current_runtime is None:
        current_runtime = ((current_readiness.get("runtime_identity_policy") or {}).get("admission") or {})
    if current_runtime.get("status") != "VALID":
        drifts.append("RUNTIME_IDENTITY_ADMISSION_DRIFT")
        drifts.extend(f"RUNTIME_POST_BLOCKER:{item}" for item in (current_runtime.get("blockers") or []))
    observed_runtime_identity = current_runtime.get("runtime_identity") or {}
    runtime_comparisons = [
        ("runtime_identity_sha256", current_runtime.get("runtime_identity_sha256"), "RUNTIME_IDENTITY_SHA256_DRIFT"),
        ("client_runtime_identity_sha256", observed_runtime_identity.get("client_identity_sha256"), "CLIENT_RUNTIME_IDENTITY_DRIFT"),
        ("model_runtime_identity_sha256", observed_runtime_identity.get("model_runtime_identity_sha256"), "MODEL_RUNTIME_IDENTITY_DRIFT"),
        ("inference_config_sha256", observed_runtime_identity.get("inference_config_sha256"), "INFERENCE_CONFIG_IDENTITY_DRIFT"),
    ]
    for frozen_key, observed_value, code in runtime_comparisons:
        if execution_receipt.get(frozen_key) != observed_value:
            drifts.append(code)

    return {
        "schema_version": "astrograph-llm-method-confirmatory-identity-check.v4",
        "status": "STABLE" if not drifts else "DRIFTED",
        "campaign_id": execution_receipt.get("campaign_id"),
        "execution_receipt_sha256": execution_receipt.get("receipt_sha256"),
        "drift_codes": sorted(set(drifts)),
        "observed": {
            "design_sha256": (current_readiness.get("design") or {}).get("sha256"),
            "case_manifest_sha256": manifest.get("sha256"),
            "case_identity_sha256": manifest.get("case_identity_sha256"),
            "case_count": manifest.get("case_count"),
            "dataset_snapshot_sha256": current_snapshot_sha256,
            "dataset_fingerprints": {
                key: {
                    "sample_index_sha256": value.get("sample_index_sha256"),
                    "payload_identity_sha256": value.get("payload_identity_sha256"),
                    "payload_file_count": value.get("payload_file_count"),
                }
                for key, value in sorted(current_datasets.items())
            },
            "runtime_identity_sha256": current_runtime.get("runtime_identity_sha256"),
            "client_runtime_identity_sha256": observed_runtime_identity.get("client_identity_sha256"),
            "model_runtime_identity_sha256": observed_runtime_identity.get("model_runtime_identity_sha256"),
            "inference_config_sha256": observed_runtime_identity.get("inference_config_sha256"),
            "model_runtime_attestation_sha256": current_runtime.get("model_runtime_attestation_sha256"),
            "observed_model_ids": sorted(observed_models),
        },
        "claim_boundary": (
            "R16 identity stability check only. STABLE means preregistered dataset/source/environment/inference/model-runtime identities matched at post-run re-observation. "
            "It is not a performance, safety, producer-authenticity or flight-qualification result."
        ),
    }


def _load_class_map(path: Path) -> dict[str, str]:
    payload = _read_json(path)
    mappings = payload.get("mappings")
    if not isinstance(mappings, dict) or not mappings:
        raise ValueError(f"fault class map has no mappings: {path}")
    return {str(k): str(v) for k, v in mappings.items()}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def freeze_confirmatory_case_manifest(
    root: str | Path,
    *,
    output_path: str | Path,
    dataset_roots: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    root = Path(root).resolve()
    design = load_confirmatory_design(root)
    roots = default_dataset_roots(root)
    if dataset_roots:
        roots.update({key: Path(value) for key, value in dataset_roots.items()})
    for key, dataset_root in roots.items():
        if not (dataset_root / "sample_index.csv").is_file():
            raise FileNotFoundError(f"{key} sample_index.csv unavailable: {dataset_root}")

    adcs_map = _load_class_map(root / "configs/datasets/adcs_v5_2_2/fault_class_map.json")
    phm_map = _load_class_map(root / "configs/datasets/phm_v6_0/fault_class_map.json")
    cases: list[dict[str, Any]] = []

    adcs_rows = _read_csv(roots["adcs"] / "sample_index.csv")
    for row in adcs_rows:
        dataset_class = row.get("diagnostic_class", "")
        sample_index = int(row["sample_index"])
        if dataset_class not in adcs_map or sample_index not in TEST_SAMPLE_INDICES:
            continue
        pair = row["trajectory_pair_id"]
        nominal = next((x for x in adcs_rows if x.get("diagnostic_class") == "nominal" and x.get("trajectory_pair_id") == pair), None)
        if nominal is None:
            raise ValueError(f"ADCS nominal pair missing for {row['sample_id']}")
        cases.append({
            "case_id": f"ADCS-{dataset_class}-{row['sample_id']}",
            "dataset": "adcs",
            "dataset_id": "whole_spacecraft_adcs_paired_dataset_v5_2_2",
            "dataset_class": dataset_class,
            "true_fault_id": adcs_map[dataset_class],
            "sample_id": row["sample_id"],
            "sample_index": sample_index,
            "group_id": pair,
            "nominal_sample_id": nominal["sample_id"],
        })

    phm_rows = _read_csv(roots["phm"] / "sample_index.csv")
    for row in phm_rows:
        dataset_class = row.get("condition_id", "")
        sample_index = int(row["sample_index"])
        if dataset_class not in phm_map or sample_index not in TEST_SAMPLE_INDICES:
            continue
        subsystem = row["subsystem"]
        nominal = next((x for x in phm_rows if x.get("condition_id") == "nominal" and x.get("subsystem") == subsystem and int(x["sample_index"]) == sample_index), None)
        if nominal is None:
            raise ValueError(f"PHM nominal pair missing for {row['sample_id']}")
        cases.append({
            "case_id": f"PHM-{dataset_class}-{row['sample_id']}",
            "dataset": "phm",
            "dataset_id": "whole_spacecraft_phm_dataset_v6_0",
            "dataset_class": dataset_class,
            "true_fault_id": phm_map[dataset_class],
            "sample_id": row["sample_id"],
            "sample_index": sample_index,
            "group_id": f"{subsystem}:{sample_index}",
            "nominal_sample_id": nominal["sample_id"],
        })

    cases.sort(key=lambda x: (x["dataset"], x["dataset_class"], x["sample_index"], x["sample_id"]))
    if len({x["case_id"] for x in cases}) != len(cases):
        raise ValueError("confirmatory case IDs are not unique")
    observed_classes = {(x["dataset"], x["dataset_class"]) for x in cases}
    expected_classes = {("adcs", k) for k in adcs_map} | {("phm", k) for k in phm_map}
    missing_classes = sorted(expected_classes - observed_classes)
    if missing_classes:
        raise ValueError(f"mapped classes absent from frozen test selection: {missing_classes}")
    expected_count = int(design["case_selection"]["expected_case_count"])
    if len(cases) != expected_count:
        raise ValueError(f"case count {len(cases)} differs from frozen expected_case_count={expected_count}")

    manifest = {
        "schema_version": "astrograph-llm-method-case-manifest.v1",
        "campaign_id": design["campaign_id"],
        "selection_split": "test",
        "selection_sample_indices": sorted(TEST_SAMPLE_INDICES),
        "case_count": len(cases),
        "cases": cases,
        "claim_boundary": (
            "Frozen paired case identities only. Ground truth is used for offline evaluation after inference and must not enter runtime prompts or candidate generation."
        ),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def exact_mcnemar_two_sided(*, corrected: int, induced: int) -> float:
    """Legacy case-level exact McNemar calculation retained for descriptive compatibility."""
    if corrected < 0 or induced < 0:
        raise ValueError("paired counts must be nonnegative")
    n = corrected + induced
    if n == 0:
        return 1.0
    k = min(corrected, induced)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def exact_group_direction_sign_two_sided(*, positive_groups: int, negative_groups: int) -> float:
    """Exact two-sided sign test over independent group directions.

    Each ``group_id`` contributes at most one direction regardless of how many cases it
    contains.  Tied groups are excluded from the binomial direction test but remain in
    the effect-size estimate.  This is intentionally conservative and avoids treating
    correlated cases inside one group as independent McNemar pairs.
    """
    return exact_mcnemar_two_sided(corrected=positive_groups, induced=negative_groups)


_T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
    9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
    23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045,
    30: 2.042, 31: 2.040, 32: 2.037, 33: 2.035, 34: 2.032, 35: 2.030, 36: 2.028,
    37: 2.026, 38: 2.024, 39: 2.023, 40: 2.021, 50: 2.009, 60: 2.000, 80: 1.990,
    100: 1.984, 120: 1.980,
}


def _student_t_critical_975(df: int) -> float:
    if df < 1:
        raise ValueError("degrees of freedom must be positive")
    for key in sorted(_T975):
        if df <= key:
            return _T975[key]
    return 1.96


def _group_mean_t_interval95(values: list[float]) -> tuple[float, float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"), float("inf"))
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    standard_error = math.sqrt(max(variance, 0.0) / len(values))
    half_width = _student_t_critical_975(len(values) - 1) * standard_error
    return (max(-1.0, mean - half_width), min(1.0, mean + half_width), half_width)


def paired_arm_effect(rows: Iterable[Any], *, baseline_arm_id: str, challenger_arm_id: str) -> dict[str, Any]:
    by_arm: dict[str, dict[str, Any]] = {baseline_arm_id: {}, challenger_arm_id: {}}
    for row in rows:
        arm = getattr(row, "arm_id")
        if arm in by_arm:
            case_id = getattr(row, "case_id")
            if case_id in by_arm[arm]:
                raise ValueError("duplicate case in paired arm comparison")
            by_arm[arm][case_id] = row
    if set(by_arm[baseline_arm_id]) != set(by_arm[challenger_arm_id]) or not by_arm[baseline_arm_id]:
        raise ValueError("paired arm comparison requires identical nonempty case IDs")
    corrected = 0
    induced = 0
    both_correct = 0
    both_noncorrect = 0
    groups: dict[str, dict[str, Any]] = {}
    for case_id in sorted(by_arm[baseline_arm_id]):
        baseline = by_arm[baseline_arm_id][case_id]
        challenger = by_arm[challenger_arm_id][case_id]
        baseline_group = str(getattr(baseline, "group_id", "") or "")
        challenger_group = str(getattr(challenger, "group_id", "") or "")
        if not baseline_group or baseline_group != challenger_group:
            raise ValueError("paired arm comparison requires identical nonblank group_id per case")
        b_ok = getattr(baseline, "predicted_fault_id") == getattr(baseline, "true_fault_id")
        c_ok = getattr(challenger, "predicted_fault_id") == getattr(challenger, "true_fault_id")
        group = groups.setdefault(baseline_group, {
            "group_id": baseline_group, "case_count": 0, "corrected": 0, "induced": 0,
            "probe_exposed": False,
        })
        group["case_count"] += 1
        if int(getattr(challenger, "probe_execution_count", 0) or 0) > int(getattr(baseline, "probe_execution_count", 0) or 0):
            group["probe_exposed"] = True
        if not b_ok and c_ok:
            corrected += 1
            group["corrected"] += 1
        elif b_ok and not c_ok:
            induced += 1
            group["induced"] += 1
        elif b_ok and c_ok:
            both_correct += 1
        else:
            both_noncorrect += 1
    group_rows: list[dict[str, Any]] = []
    group_deltas: list[float] = []
    positive_groups = negative_groups = tie_groups = 0
    for group_id in sorted(groups):
        group = groups[group_id]
        net = int(group["corrected"]) - int(group["induced"])
        delta = net / int(group["case_count"])
        if net > 0:
            positive_groups += 1
        elif net < 0:
            negative_groups += 1
        else:
            tie_groups += 1
        group_deltas.append(delta)
        group_rows.append({
            "group_id": group_id,
            "case_count": int(group["case_count"]),
            "corrected_case_count": int(group["corrected"]),
            "induced_error_count": int(group["induced"]),
            "net_diagnostic_gain": net,
            "paired_accuracy_delta": round(delta, 8),
            "probe_exposed": bool(group["probe_exposed"]),
        })
    group_mean = sum(group_deltas) / len(group_deltas)
    ci_low, ci_high, ci_half = _group_mean_t_interval95(group_deltas)
    return {
        "baseline_arm_id": baseline_arm_id,
        "challenger_arm_id": challenger_arm_id,
        "case_count": len(by_arm[baseline_arm_id]),
        "corrected_case_count": corrected,
        "induced_error_count": induced,
        "net_diagnostic_gain": corrected - induced,
        "both_correct_count": both_correct,
        "both_noncorrect_count": both_noncorrect,
        "mcnemar_exact_two_sided_p": round(exact_mcnemar_two_sided(corrected=corrected, induced=induced), 8),
        "case_level_mcnemar_inference_status": "DESCRIPTIVE_ONLY_CLUSTERING_IGNORED",
        "independent_group_count": len(group_rows),
        "positive_group_count": positive_groups,
        "negative_group_count": negative_groups,
        "tie_group_count": tie_groups,
        "probe_exposed_group_count": sum(bool(item["probe_exposed"]) for item in group_rows),
        "group_mean_accuracy_delta": round(group_mean, 8),
        "group_mean_accuracy_delta_ci95_lower": (None if not math.isfinite(ci_low) else round(ci_low, 8)),
        "group_mean_accuracy_delta_ci95_upper": (None if not math.isfinite(ci_high) else round(ci_high, 8)),
        "group_mean_accuracy_delta_ci95_half_width": (None if not math.isfinite(ci_half) else round(ci_half, 8)),
        "group_direction_exact_two_sided_p": round(
            exact_group_direction_sign_two_sided(positive_groups=positive_groups, negative_groups=negative_groups), 8
        ),
        "group_effects": group_rows,
        "inferential_unit": "group_id",
        "claim_boundary": (
            "Group-aware engineering inference. Case-level McNemar is descriptive only; the exact sign test treats each "
            "independent group as one directional unit, and the t interval summarizes equal-group-weighted paired accuracy deltas."
        ),
    }


def _comparison_gate(
    paired: dict[str, Any] | None,
    thresholds: dict[str, Any],
    *,
    require_probe_exposure: bool = False,
) -> dict[str, bool]:
    paired = paired or {}
    lower = paired.get("group_mean_accuracy_delta_ci95_lower")
    half = paired.get("group_mean_accuracy_delta_ci95_half_width")
    checks = {
        "no_induced_errors": int(paired.get("induced_error_count", 10**9)) <= int(thresholds["max_induced_error_count"]),
        "independent_group_count": int(paired.get("independent_group_count", 0)) >= int(thresholds["min_independent_group_count"]),
        "minimum_practical_effect": float(paired.get("group_mean_accuracy_delta", -1.0)) >= float(thresholds["min_group_mean_accuracy_delta"]),
        "effect_ci_positive": lower is not None and float(lower) > float(thresholds["min_group_mean_accuracy_delta_ci95_lower"]),
        "effect_precision": half is not None and float(half) <= float(thresholds["max_group_mean_accuracy_delta_ci95_half_width"]),
        "group_direction_significance": float(paired.get("group_direction_exact_two_sided_p", 1.0)) <= float(thresholds["max_group_direction_exact_p"]),
    }
    if require_probe_exposure:
        checks["probe_exposed_group_count"] = int(paired.get("probe_exposed_group_count", 0)) >= int(thresholds["min_probe_exposed_group_count"])
    return checks


def evaluate_confirmatory_promotion(report: dict[str, Any], design: dict[str, Any]) -> dict[str, Any]:
    thresholds = design["promotion_thresholds"]
    metrics = report["arm_metrics"]["A2_CONTROLLED_FULL"]
    comparisons = report.get("paired_comparisons", {})
    full_key = "A0_ALGORITHM_ONLY__vs__A2_CONTROLLED_FULL"
    full_paired = comparisons.get(full_key)
    checks = _comparison_gate(full_paired, thresholds)
    checks.update({
        "candidate_recall": metrics["true_fault_in_candidate_set_rate"] is not None and metrics["true_fault_in_candidate_set_rate"] >= thresholds["min_true_fault_in_candidate_set_rate"],
        "evidence_citation_valid": metrics["evidence_citation_valid_rate"] is not None and metrics["evidence_citation_valid_rate"] >= thresholds["min_evidence_citation_valid_rate"],
        "case_net_gain": bool(full_paired and full_paired["net_diagnostic_gain"] >= thresholds["min_case_net_diagnostic_gain"]),
    })

    probe_thresholds = design["probe_promotion_thresholds"]
    probe_key = "A7_NO_ACTIVE_PROBE__vs__A2_CONTROLLED_FULL"
    probe_paired = comparisons.get(probe_key)
    probe_checks = _comparison_gate(probe_paired, probe_thresholds, require_probe_exposure=True)
    probe_status = "ELIGIBLE_FOR_SEPARATE_FORMAL_EVIDENCE_REVIEW" if all(probe_checks.values()) else "NOT_ELIGIBLE_FOR_FORMAL_EVIDENCE_REVIEW"
    return {
        "schema_version": "astrograph-llm-method-confirmatory-promotion-decision.v2",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "thresholds": thresholds,
        "primary_estimand": {
            "comparison": design["primary_comparison"],
            "paired_comparison_key": full_key,
            "inferential_unit": "group_id",
            "effect": (full_paired or {}).get("group_mean_accuracy_delta"),
            "ci95": [
                (full_paired or {}).get("group_mean_accuracy_delta_ci95_lower"),
                (full_paired or {}).get("group_mean_accuracy_delta_ci95_upper"),
            ],
        },
        "probe_promotion_review": {
            "status": probe_status,
            "comparison": design["secondary_probe_comparison"],
            "paired_comparison_key": probe_key,
            "checks": probe_checks,
            "thresholds": probe_thresholds,
            "claim_boundary": (
                "This result addresses the marginal contribution of active Probe via A2 vs A7 only. Even ELIGIBLE status "
                "does not itself promote Probe output to formal EvidenceFact production; an independent governance review is required."
            ),
        },
        "claim_boundary": (
            "A full-stack PASS authorizes engineering promotion review only. The separately preregistered A2-vs-A7 result "
            "is the only comparison allowed to inform a future Probe formal-evidence-producer review. Neither is flight or safety qualification."
        ),
    }
