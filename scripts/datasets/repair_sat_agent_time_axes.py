#!/usr/bin/env python3
"""Repair only the P-018 S0 Sat-Agent exports missing explicit ``time_s``.

Selection repeats the frozen zero-fit field-presence contract.  Row payloads are
handled as opaque bytes: the utility prepends one ASCII time value to each CSV
row and verifies that removing that prefix recovers the original file exactly.
It never parses, emits, or selects from label/truth values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from astrograph.data_sources import (  # noqa: E402
    DataSourceType,
    DatasetSourceConfig,
    build_data_source,
)
from astrograph.semantics import ChannelMapCatalog  # noqa: E402


UTF8_BOM = b"\xef\xbb\xbf"
EXPECTED_SOURCE_ID = "SAT_AGENT_PLATFORM"
EXPECTED_TIME_FIELD = "time_s"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path.name}")
    return payload


def _decimal(value: object, name: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid {name}") from exc
    if not parsed.is_finite():
        raise ValueError(f"non-finite {name}")
    return parsed


def _decimal_text(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def endpoint_policy(row_count: int, duration_s: object, sample_s: object) -> str:
    """Return the declared endpoint policy; never guess an unmatched convention."""

    if row_count < 1:
        raise ValueError("row_count must be positive")
    duration = _decimal(duration_s, "duration_s")
    sample = _decimal(sample_s, "sample_s")
    if duration < 0 or sample <= 0:
        raise ValueError("duration_s/sample_s outside physical timing bounds")
    exclude_end = sample * Decimal(row_count)
    include_end = sample * Decimal(max(row_count - 1, 0))
    tolerance = max(Decimal("1e-9"), abs(duration) * Decimal("1e-9"))
    if abs(duration - exclude_end) <= tolerance:
        return "INCLUDE_START_EXCLUDE_END"
    if abs(duration - include_end) <= tolerance:
        return "INCLUDE_START_AND_END"
    raise ValueError("ROW_COUNT_CONVENTION_OTHER")


def _split_physical_lines(raw: bytes) -> tuple[bytes, list[bytes], bool]:
    has_bom = raw.startswith(UTF8_BOM)
    body = raw[len(UTF8_BOM) :] if has_bom else raw
    lines = body.splitlines(keepends=True)
    if len(lines) < 2:
        raise ValueError("telemetry CSV must contain a header and at least one row")
    if any(not line.rstrip(b"\r\n") for line in lines):
        raise ValueError("blank physical CSV lines are not repairable")
    header = lines[0].rstrip(b"\r\n")
    if b"\x00" in raw or not header.strip():
        raise ValueError("telemetry must be a non-binary CSV with a non-empty header")
    return lines[0], lines[1:], has_bom


def prepend_explicit_time_axis(
    raw: bytes,
    *,
    row_count: int,
    sample_s: object,
    endpoint: str,
) -> bytes:
    """Prepend time bytes while preserving every original byte as a row suffix."""

    header, rows, has_bom = _split_physical_lines(raw)
    if len(rows) != row_count:
        raise ValueError("manifest row_count does not equal physical CSV row count")
    header_without_bom = header
    first_name = header_without_bom.split(b",", 1)[0].strip().strip(b'"').lower()
    if first_name == EXPECTED_TIME_FIELD.encode("ascii"):
        raise ValueError("time_s already present")
    sample = _decimal(sample_s, "sample_s")
    if endpoint not in {"INCLUDE_START_EXCLUDE_END", "INCLUDE_START_AND_END"}:
        raise ValueError("unsupported endpoint policy")
    prefix = UTF8_BOM if has_bom else b""
    repaired = [prefix + b"time_s," + header_without_bom]
    repaired.extend(
        _decimal_text(sample * Decimal(index)).encode("ascii") + b"," + row
        for index, row in enumerate(rows)
    )
    return b"".join(repaired)


def recover_original_bytes(repaired: bytes) -> bytes:
    """Remove the newly prepended field and recover the original bytes exactly."""

    has_bom = repaired.startswith(UTF8_BOM)
    body = repaired[len(UTF8_BOM) :] if has_bom else repaired
    lines = body.splitlines(keepends=True)
    if not lines or not lines[0].startswith(b"time_s,"):
        raise ValueError("repaired telemetry does not start with time_s")
    recovered = [lines[0].split(b",", 1)[1]]
    for line in lines[1:]:
        if b"," not in line:
            raise ValueError("repaired data row has no time prefix")
        recovered.append(line.split(b",", 1)[1])
    return (UTF8_BOM if has_bom else b"") + b"".join(recovered)


@dataclass(frozen=True)
class RepairTarget:
    entity_id: str
    path: Path
    row_count: int
    duration_s: Decimal
    sample_s: Decimal
    endpoint_policy: str
    mapping_id: str
    capability_id: str


def discover_targets(source_root: Path, channel_map_path: Path) -> list[RepairTarget]:
    """Discover the same missing-time field-compatible cohort as S0 validation."""

    catalog = ChannelMapCatalog.model_validate_json(
        channel_map_path.read_text(encoding="utf-8-sig")
    )
    source = build_data_source(
        DatasetSourceConfig(
            source_type=DataSourceType.SAT_AGENT_PLATFORM,
            root=str(source_root),
            read_only=True,
            lazy=True,
            recursive=True,
            timestamp_column=EXPECTED_TIME_FIELD,
            metadata={"stage": "P018_S0_TIME_AXIS_SOURCE_REPAIR", "source_count": 1},
        )
    )
    selected: dict[str, RepairTarget] = {}
    with source:
        entities = source.list_entities()
        entity_paths = getattr(source, "_entity_paths", None)
        if not isinstance(entity_paths, dict):
            raise ValueError("Sat-Agent adapter does not expose governed source paths")
        for mapping in catalog.maps:
            required = {
                entry.source_field for entry in mapping.entries if entry.required
            }
            for entity in entities:
                available = set(source.list_channels(entity.entity_id))
                if not required <= available:
                    continue
                if entity.metadata.get("time_axis_contract_status") != (
                    "MISSING_EXPLICIT_TIME_AXIS"
                ):
                    continue
                if entity.entity_id in selected:
                    raise ValueError("one entity matched multiple repair mappings")
                row_count = entity.row_count
                if not isinstance(row_count, int) or row_count < 1:
                    raise ValueError("repair target lacks a valid manifest row count")
                duration = _decimal(entity.metadata.get("duration_s"), "duration_s")
                sample = _decimal(entity.metadata.get("sample_s"), "sample_s")
                policy = endpoint_policy(row_count, duration, sample)
                path = Path(entity_paths[entity.entity_id]).resolve(strict=True)
                path.relative_to(source_root.resolve(strict=True))
                selected[entity.entity_id] = RepairTarget(
                    entity_id=entity.entity_id,
                    path=path,
                    row_count=row_count,
                    duration_s=duration,
                    sample_s=sample,
                    endpoint_policy=policy,
                    mapping_id=mapping.mapping_id,
                    capability_id=str(entity.metadata.get("capability_id", "UNDECLARED")),
                )
    return [selected[key] for key in sorted(selected)]


def _validate_authorization(
    authorization: dict[str, Any],
    *,
    schema_path: Path,
    repair_protocol_path: Path,
    pre_repair_evidence_path: Path,
) -> int:
    if authorization.get("status") != "APPROVED":
        raise ValueError("project-owner repair authorization is not APPROVED")
    permissions = authorization.get("permissions", {})
    if permissions.get("sat_agent_source_write") is not True:
        raise ValueError("Sat-Agent source write is not authorized")
    for prohibited in (
        "model_fit_or_calibration",
        "new_seed",
        "stage_s1",
        "stage_b",
        "formal_technical_validation",
    ):
        if permissions.get(prohibited) is not False:
            raise ValueError(f"authorization must explicitly deny {prohibited}")
    bindings = authorization.get("sha256_bindings", {})
    expected = {
        "time_axis_schema": _sha256_file(schema_path),
        "repair_protocol": _sha256_file(repair_protocol_path),
        "pre_repair_semantic_evidence": _sha256_file(pre_repair_evidence_path),
    }
    if bindings != expected:
        raise ValueError("authorization artifact hash bindings do not match")
    count = authorization.get("authorized_target_count")
    if not isinstance(count, int) or count < 1:
        raise ValueError("authorized_target_count must be positive")
    return count


def build_repair(
    *,
    source_root: Path,
    channel_map_path: Path,
    schema_path: Path,
    repair_protocol_path: Path,
    pre_repair_evidence_path: Path,
    authorization_path: Path,
    backup_root: Path,
    apply: bool,
) -> dict[str, Any]:
    authorization = _load_object(authorization_path)
    expected_count = _validate_authorization(
        authorization,
        schema_path=schema_path,
        repair_protocol_path=repair_protocol_path,
        pre_repair_evidence_path=pre_repair_evidence_path,
    )
    targets = discover_targets(source_root, channel_map_path)
    if len(targets) != expected_count:
        raise ValueError(
            f"repair target count changed: expected {expected_count}, found {len(targets)}"
        )

    prepared: list[tuple[RepairTarget, bytes, bytes]] = []
    mapping_counts: Counter[str] = Counter()
    capability_counts: Counter[str] = Counter()
    endpoint_counts: Counter[str] = Counter()
    for target in targets:
        raw = target.path.read_bytes()
        repaired = prepend_explicit_time_axis(
            raw,
            row_count=target.row_count,
            sample_s=target.sample_s,
            endpoint=target.endpoint_policy,
        )
        if recover_original_bytes(repaired) != raw:
            raise ValueError("non-time byte preservation check failed")
        prepared.append((target, raw, repaired))
        mapping_counts[target.mapping_id] += 1
        capability_counts[target.capability_id] += 1
        endpoint_counts[target.endpoint_policy] += 1

    receipts: list[dict[str, Any]] = []
    written: list[tuple[Path, Path]] = []
    backup_root_resolved = backup_root.resolve()
    if apply:
        backup_root.mkdir(parents=True, exist_ok=False)
        backup_root_resolved = backup_root.resolve(strict=True)
        try:
            for target, raw, repaired in prepared:
                backup_path = backup_root_resolved / f"{target.entity_id}.csv"
                backup_path.write_bytes(raw)
                if backup_path.read_bytes() != raw:
                    raise ValueError("backup verification failed")
            for target, raw, repaired in prepared:
                backup_path = backup_root_resolved / f"{target.entity_id}.csv"
                temporary = target.path.with_name(
                    f".{target.path.name}.{target.entity_id}.p018.tmp"
                )
                temporary.write_bytes(repaired)
                if temporary.read_bytes() != repaired:
                    raise ValueError("staged repaired file verification failed")
                os.replace(temporary, target.path)
                written.append((target.path, backup_path))
                if target.path.read_bytes() != repaired:
                    raise ValueError("repaired source file verification failed")
        except Exception:
            for target_path, backup_path in reversed(written):
                shutil.copyfile(backup_path, target_path)
            raise

    for target, raw, repaired in prepared:
        first_time = "0"
        last_time = _decimal_text(target.sample_s * Decimal(target.row_count - 1))
        receipts.append(
            {
                "entity_id": target.entity_id,
                "mapping_id": target.mapping_id,
                "capability_id": target.capability_id,
                "row_count": target.row_count,
                "endpoint_policy": target.endpoint_policy,
                "first_time_s": first_time,
                "last_time_s": last_time,
                "original_telemetry_sha256": _sha256_bytes(raw),
                "repaired_telemetry_sha256": _sha256_bytes(repaired),
                "non_time_payload_sha256": _sha256_bytes(raw),
                "opaque_original_bytes_recovered": recover_original_bytes(repaired)
                == raw,
            }
        )

    return {
        "schema_version": "astrograph.p018.sat-agent-time-axis-repair-receipt.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "APPLIED_PASS" if apply else "DRY_RUN_READY",
        "mode": "APPLY" if apply else "DRY_RUN",
        "stage": "P018_S0_ZERO_FIT_SOURCE_REPAIR",
        "source_id": EXPECTED_SOURCE_ID,
        "source_count": 1,
        "authorized_target_count": expected_count,
        "discovered_target_count": len(targets),
        "changed_source_file_count": len(written),
        "added_field": EXPECTED_TIME_FIELD,
        "mapping_counts": dict(sorted(mapping_counts.items())),
        "capability_counts": dict(sorted(capability_counts.items())),
        "endpoint_policy_counts": dict(sorted(endpoint_counts.items())),
        "receipts": receipts,
        "backup": {
            "created": apply,
            "root_token": "${P018_EXTERNAL_BACKUP_ROOT}",
            "file_count": len(written),
            "naming_policy": "OPAQUE_ENTITY_ID",
        },
        "execution_boundary": {
            "model_fit_count": 0,
            "calibration_count": 0,
            "seed_created": False,
            "truth_or_label_values_parsed": False,
            "truth_or_label_fields_added": False,
            "raw_row_bytes_copied_opaquely": True,
            "case_run_paths_emitted": False,
            "consumer_side_time_synthesis": False,
            "non_time_bytes_preserved_exactly": all(
                item["opaque_original_bytes_recovered"] for item in receipts
            ),
        },
        "control_sha256": {
            "authorization": _sha256_file(authorization_path),
            "time_axis_schema": _sha256_file(schema_path),
            "repair_protocol": _sha256_file(repair_protocol_path),
            "pre_repair_semantic_evidence": _sha256_file(
                pre_repair_evidence_path
            ),
            "channel_map_catalog": _sha256_file(channel_map_path),
        },
        "claim_boundary": (
            "P-018 S0 source-export repair only. APPLIED_PASS confirms byte-"
            "preserving explicit time-axis insertion, not data qualification, model "
            "performance, Stage S1, Stage B or formal acceptance."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--channel-map",
        type=Path,
        default=ROOT
        / "configs"
        / "semantics"
        / "sat_agent_platform_channel_maps.s0.v1.json",
    )
    parser.add_argument(
        "--time-axis-schema",
        type=Path,
        default=ROOT
        / "protocols"
        / "integration"
        / "sat_agent_public_time_axis.v1.schema.json",
    )
    parser.add_argument(
        "--repair-protocol",
        type=Path,
        default=ROOT
        / "protocols"
        / "integration"
        / "p018_sat_agent_time_axis_export_repair.v1.json",
    )
    parser.add_argument(
        "--pre-repair-evidence",
        type=Path,
        default=ROOT
        / "governance"
        / "evidence"
        / "0.7.1-extension"
        / "p018_sat_agent_semantic_contract_validation.v1.json",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = build_repair(
        source_root=args.root,
        channel_map_path=args.channel_map,
        schema_path=args.time_axis_schema,
        repair_protocol_path=args.repair_protocol,
        pre_repair_evidence_path=args.pre_repair_evidence,
        authorization_path=args.authorization,
        backup_root=args.backup_root,
        apply=args.apply,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "target_count": result["discovered_target_count"],
                "changed_source_file_count": result["changed_source_file_count"],
                "endpoint_policy_counts": result["endpoint_policy_counts"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
