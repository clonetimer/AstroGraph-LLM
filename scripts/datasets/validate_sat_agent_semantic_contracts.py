#!/usr/bin/env python3
"""Exercise P-018 S0 semantic mappings against Sat-Agent telemetry, without fit.

Entities are selected only by the presence of declared public input fields.  The
utility never reads manifest labels, truth, case identities, or simulator answer
proxies and emits aggregate counts only.  It is a structural contract test, not
a diagnostic performance experiment or a data-qualification gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from astrograph.data_sources import (  # noqa: E402
    DataSourceType,
    DatasetSourceConfig,
    build_data_source,
)
from astrograph.semantics import (  # noqa: E402
    ChannelMapCatalog,
    ChannelMapper,
    ComponentArchetypeCatalog,
    SemanticMappingError,
    SemanticSchema,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_profiles(path: Path) -> tuple[dict[str, set[str]], dict[str, dict[str, str]]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or not isinstance(payload.get("profiles"), list):
        raise ValueError("Sat-Agent schema-profile catalog is invalid")
    fields: dict[str, set[str]] = {}
    metadata: dict[str, dict[str, str]] = {}
    for item in payload["profiles"]:
        if not isinstance(item, dict):
            raise ValueError("schema profile must be an object")
        profile_id = str(item.get("profile_id", ""))
        if not profile_id or profile_id in fields:
            raise ValueError("schema profile IDs must be non-empty and unique")
        time_field = item.get("time_field", {})
        observations = item.get("public_observation_fields", [])
        if not isinstance(time_field, dict) or not isinstance(observations, list):
            raise ValueError(f"invalid fields for schema profile: {profile_id}")
        names = {str(time_field.get("name", ""))}
        names.update(
            str(field.get("name", ""))
            for field in observations
            if isinstance(field, dict)
        )
        if "" in names:
            raise ValueError(f"empty field name in schema profile: {profile_id}")
        fields[profile_id] = names
        metadata[profile_id] = {
            "capability_id": str(item.get("capability_id", "UNDECLARED")),
            "hierarchy_level": str(item.get("hierarchy_level", "UNDECLARED")),
            "usage_grade": str(item.get("usage_grade", "UNDECLARED")),
            "time_field_name": str(time_field.get("name", "")),
        }
    return fields, metadata


def _profile_for_mapping(
    mapping_fields: set[str], profile_fields: dict[str, set[str]]
) -> str:
    matches = sorted(
        profile_id
        for profile_id, declared in profile_fields.items()
        if mapping_fields <= declared
    )
    if len(matches) != 1:
        raise ValueError(
            f"mapping fields must resolve to exactly one schema profile; found {len(matches)}"
        )
    return matches[0]


def _row_count_convention(
    row_count: int | None, duration_s: object, sample_s: object
) -> str:
    """Classify timing metadata without constructing a consumer-side time axis."""

    if row_count is None or duration_s is None or sample_s is None:
        return "INCOMPLETE_TIMING_METADATA"
    try:
        duration = float(duration_s)
        sample = float(sample_s)
    except (TypeError, ValueError):
        return "INVALID_TIMING_METADATA"
    if sample <= 0 or duration < 0:
        return "INVALID_TIMING_METADATA"
    interval_count = duration / sample
    tolerance = max(1e-9, abs(interval_count) * 1e-9)
    if abs(float(row_count) - interval_count) <= tolerance:
        return "ROW_COUNT_EQUALS_DURATION_DIV_SAMPLE"
    if abs(float(row_count) - (interval_count + 1.0)) <= tolerance:
        return "ROW_COUNT_EQUALS_DURATION_DIV_SAMPLE_PLUS_ONE"
    return "ROW_COUNT_CONVENTION_OTHER"


def build_validation(
    source_root: Path,
    *,
    semantic_schema_path: Path,
    archetype_catalog_path: Path,
    channel_map_path: Path,
    schema_profile_path: Path,
    row_limit: int = 2048,
) -> dict[str, Any]:
    semantic_schema = SemanticSchema.model_validate_json(
        semantic_schema_path.read_text(encoding="utf-8-sig")
    )
    archetypes = ComponentArchetypeCatalog.model_validate_json(
        archetype_catalog_path.read_text(encoding="utf-8-sig")
    )
    channel_maps = ChannelMapCatalog.model_validate_json(
        channel_map_path.read_text(encoding="utf-8-sig")
    )
    archetypes.validate_quantity_kinds(
        {item.quantity_kind_id for item in semantic_schema.quantity_kinds}
    )
    archetypes.validate_channel_maps(channel_maps)
    profile_fields, profile_metadata = _load_profiles(schema_profile_path)
    profile_sha256 = _sha256(schema_profile_path)
    if {item.source_schema_sha256 for item in channel_maps.maps} != {profile_sha256}:
        raise ValueError("channel maps do not bind the supplied schema-profile hash")

    config = DatasetSourceConfig(
        source_type=DataSourceType.SAT_AGENT_PLATFORM,
        root=str(source_root),
        read_only=True,
        lazy=True,
        recursive=True,
        timestamp_column="time_s",
        metadata={"stage": "P018_S0_ZERO_FIT", "source_count": 1},
    )
    source = build_data_source(config)
    rows: list[dict[str, Any]] = []
    configured_archetypes: set[str] = set()
    exercised_archetypes: set[str] = set()
    exercised_levels: set[str] = set()
    total_passed = 0
    total_failed = 0
    total_rows = 0
    total_declared_profile_matched = 0
    total_declared_profile_passed = 0
    total_declared_profile_failed = 0
    total_cross_profile_matched = 0
    total_cross_profile_passed = 0
    total_cross_profile_failed = 0
    total_missing_explicit_alternative = 0
    total_missing_index_surrogate = 0
    global_missing_time_statuses: Counter[str] = Counter()
    global_row_count_conventions: Counter[str] = Counter()

    with source:
        entities = source.list_entities()
        source_metadata = source.get_metadata()
        if source_metadata.metadata.get("source_count") != 1:
            raise ValueError("Sat-Agent adapter must report source_count=1")

        for mapping in channel_maps.maps:
            mapping_field_set = {
                mapping.time_axis.source_field,
                *(entry.source_field for entry in mapping.entries),
            }
            profile_id = _profile_for_mapping(mapping_field_set, profile_fields)
            profile = profile_metadata[profile_id]
            required_signal_fields = {
                entry.source_field for entry in mapping.entries if entry.required
            }
            mapped_signal_fields = {entry.source_field for entry in mapping.entries}
            mapper = ChannelMapper(
                semantic_schema=semantic_schema,
                channel_map=mapping,
            )
            matched = 0
            passed = 0
            failed = 0
            mapped_rows = 0
            signal_counts: list[int] = []
            semantic_hashes: set[str] = set()
            errors: Counter[str] = Counter()
            signal_schema_match_missing_time_axis_count = 0
            actual_capability_counts: Counter[str] = Counter()
            actual_hierarchy_level_counts: Counter[str] = Counter()
            declared_profile_matched = 0
            declared_profile_passed = 0
            declared_profile_failed = 0
            cross_profile_matched = 0
            cross_profile_passed = 0
            cross_profile_failed = 0
            missing_time_status_counts: Counter[str] = Counter()
            missing_time_sample_s_counts: Counter[str] = Counter()
            missing_time_duration_s_counts: Counter[str] = Counter()
            missing_time_row_count_conventions: Counter[str] = Counter()
            missing_time_explicit_alternative_count = 0
            missing_time_index_surrogate_count = 0

            for entity in entities:
                available = set(source.list_channels(entity.entity_id))
                if not required_signal_fields <= available:
                    continue
                matched += 1
                actual_capability = str(
                    entity.metadata.get("capability_id", "UNDECLARED")
                )
                actual_hierarchy_level = str(
                    entity.metadata.get("hierarchy_level", "UNDECLARED")
                )
                actual_capability_counts[actual_capability] += 1
                actual_hierarchy_level_counts[actual_hierarchy_level] += 1
                declared_profile_match = (
                    actual_capability == profile["capability_id"]
                    and actual_hierarchy_level == profile["hierarchy_level"]
                )
                if declared_profile_match:
                    declared_profile_matched += 1
                else:
                    cross_profile_matched += 1
                available_with_time = set(available)
                if entity.timestamp_column:
                    available_with_time.add(entity.timestamp_column)
                if mapping.time_axis.source_field not in available_with_time:
                    signal_schema_match_missing_time_axis_count += 1
                    failed += 1
                    if declared_profile_match:
                        declared_profile_failed += 1
                    else:
                        cross_profile_failed += 1
                    errors["MISSING_TIME_AXIS"] += 1
                    time_status = str(
                        entity.metadata.get(
                            "time_axis_contract_status", "UNDECLARED_TIME_AXIS_STATUS"
                        )
                    )
                    missing_time_status_counts[time_status] += 1
                    explicit_alternatives = list(
                        entity.metadata.get("explicit_time_axis_candidates", [])
                    )
                    index_surrogates = list(
                        entity.metadata.get("index_surrogate_candidates", [])
                    )
                    if explicit_alternatives:
                        missing_time_explicit_alternative_count += 1
                    if index_surrogates:
                        missing_time_index_surrogate_count += 1
                    sample_s = entity.metadata.get("sample_s")
                    duration_s = entity.metadata.get("duration_s")
                    if sample_s is not None:
                        missing_time_sample_s_counts[str(sample_s)] += 1
                    if duration_s is not None:
                        missing_time_duration_s_counts[str(duration_s)] += 1
                    convention = _row_count_convention(
                        entity.row_count, duration_s, sample_s
                    )
                    missing_time_row_count_conventions[convention] += 1
                    continue
                selected = sorted(mapped_signal_fields & available)
                try:
                    frame = source.read_range(
                        entity.entity_id,
                        channels=selected,
                        limit=row_limit,
                    )
                    result = mapper.map_frame(frame)
                except SemanticMappingError as exc:
                    failed += 1
                    if declared_profile_match:
                        declared_profile_failed += 1
                    else:
                        cross_profile_failed += 1
                    errors[exc.code] += 1
                    continue
                except Exception as exc:  # keep complete failure denominator
                    failed += 1
                    if declared_profile_match:
                        declared_profile_failed += 1
                    else:
                        cross_profile_failed += 1
                    errors[type(exc).__name__] += 1
                    continue
                passed += 1
                if declared_profile_match:
                    declared_profile_passed += 1
                else:
                    cross_profile_passed += 1
                mapped_rows += result.frame.sampling.row_count
                signal_counts.append(len(result.frame.signals))
                semantic_hashes.add(result.frame.semantic_data_sha256)
                if result.frame.labels_used or result.frame.truth_used:
                    raise ValueError("semantic mapping unexpectedly used labels or truth")

            mapping_archetypes = sorted(
                {entry.component_archetype_id for entry in mapping.entries}
            )
            configured_archetypes.update(mapping_archetypes)
            if passed:
                exercised_archetypes.update(mapping_archetypes)
                exercised_levels.add(profile["hierarchy_level"])
            total_passed += passed
            total_failed += failed
            total_rows += mapped_rows
            total_declared_profile_matched += declared_profile_matched
            total_declared_profile_passed += declared_profile_passed
            total_declared_profile_failed += declared_profile_failed
            total_cross_profile_matched += cross_profile_matched
            total_cross_profile_passed += cross_profile_passed
            total_cross_profile_failed += cross_profile_failed
            total_missing_explicit_alternative += missing_time_explicit_alternative_count
            total_missing_index_surrogate += missing_time_index_surrogate_count
            global_missing_time_statuses.update(missing_time_status_counts)
            global_row_count_conventions.update(missing_time_row_count_conventions)
            rows.append(
                {
                    "mapping_id": mapping.mapping_id,
                    "profile_id": profile_id,
                    "declared_profile_capability_id": profile["capability_id"],
                    "declared_profile_hierarchy_level": profile["hierarchy_level"],
                    "hierarchy_level": profile["hierarchy_level"],
                    "component_archetype_ids": mapping_archetypes,
                    "usage_grade": profile["usage_grade"],
                    "matched_entity_count": matched,
                    "passed_entity_count": passed,
                    "failed_entity_count": failed,
                    "declared_profile_matched_entity_count": declared_profile_matched,
                    "declared_profile_passed_entity_count": declared_profile_passed,
                    "declared_profile_failed_entity_count": declared_profile_failed,
                    "cross_profile_compatible_entity_count": cross_profile_matched,
                    "cross_profile_passed_entity_count": cross_profile_passed,
                    "cross_profile_failed_entity_count": cross_profile_failed,
                    "actual_capability_counts": dict(sorted(actual_capability_counts.items())),
                    "actual_hierarchy_level_counts": dict(
                        sorted(actual_hierarchy_level_counts.items())
                    ),
                    "signal_schema_match_missing_time_axis_count": (
                        signal_schema_match_missing_time_axis_count
                    ),
                    "missing_time_axis_contract_status_counts": dict(
                        sorted(missing_time_status_counts.items())
                    ),
                    "missing_time_explicit_alternative_entity_count": (
                        missing_time_explicit_alternative_count
                    ),
                    "missing_time_index_surrogate_entity_count": (
                        missing_time_index_surrogate_count
                    ),
                    "missing_time_manifest_sample_s_counts": dict(
                        sorted(missing_time_sample_s_counts.items())
                    ),
                    "missing_time_manifest_duration_s_counts": dict(
                        sorted(missing_time_duration_s_counts.items())
                    ),
                    "missing_time_row_count_convention_counts": dict(
                        sorted(missing_time_row_count_conventions.items())
                    ),
                    "mapped_row_count": mapped_rows,
                    "unique_semantic_frame_count": len(semantic_hashes),
                    "mapped_signal_count_min": min(signal_counts, default=0),
                    "mapped_signal_count_max": max(signal_counts, default=0),
                    "error_counts": dict(sorted(errors.items())),
                }
            )

    blockers = []
    for row in rows:
        if row["matched_entity_count"] == 0:
            blockers.append(f"NO_MATCHING_ENTITY::{row['mapping_id']}")
        if row["signal_schema_match_missing_time_axis_count"]:
            blockers.append(f"MISSING_TIME_AXIS::{row['mapping_id']}")
        if row["failed_entity_count"]:
            blockers.append(f"MAPPING_FAILURE::{row['mapping_id']}")
    if global_missing_time_statuses:
        blockers.append("SOURCE_EXPORT_TIME_AXIS_REPAIR_REQUIRED")
    if len(exercised_archetypes) < 7:
        blockers.append("FEWER_THAN_SEVEN_ARCHETYPES_MAPPED")

    return {
        "schema_version": "astrograph.p018.sat-agent-semantic-contract-validation.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage": "P018_S0_ZERO_FIT",
        "source_id": "SAT_AGENT_PLATFORM",
        "adapter_key": "sat_agent_platform",
        "source_family_alias": "sat-agent-platform",
        "source_count": 1,
        "control_sha256": {
            "semantic_schema": _sha256(semantic_schema_path),
            "archetype_catalog": _sha256(archetype_catalog_path),
            "channel_map_catalog": _sha256(channel_map_path),
            "schema_profile_catalog": profile_sha256,
        },
        "summary": {
            "mapping_count": len(rows),
            "configured_component_archetype_count": len(configured_archetypes),
            "exercised_component_archetype_count": len(exercised_archetypes),
            "exercised_component_archetype_ids": sorted(exercised_archetypes),
            "exercised_hierarchy_level_count": len(exercised_levels),
            "exercised_hierarchy_levels": sorted(exercised_levels),
            "passed_entity_count": total_passed,
            "failed_entity_count": total_failed,
            "field_compatible_entity_denominator": total_passed + total_failed,
            "declared_profile_entity_denominator": total_declared_profile_matched,
            "declared_profile_passed_entity_count": total_declared_profile_passed,
            "declared_profile_failed_entity_count": total_declared_profile_failed,
            "cross_profile_compatible_entity_denominator": total_cross_profile_matched,
            "cross_profile_passed_entity_count": total_cross_profile_passed,
            "cross_profile_failed_entity_count": total_cross_profile_failed,
            "mapped_row_count": total_rows,
            "structural_contract_status": "PASS" if not blockers else "BLOCKED",
            "blockers": blockers,
        },
        "time_axis_audit": {
            "expected_source_field": "time_s",
            "missing_time_axis_entity_count": sum(global_missing_time_statuses.values()),
            "missing_time_axis_contract_status_counts": dict(
                sorted(global_missing_time_statuses.items())
            ),
            "explicit_alternative_time_field_entity_count": (
                total_missing_explicit_alternative
            ),
            "index_surrogate_entity_count": total_missing_index_surrogate,
            "row_count_convention_counts": dict(
                sorted(global_row_count_conventions.items())
            ),
            "consumer_side_time_synthesis_authorized": False,
            "source_export_repair_required": bool(global_missing_time_statuses),
            "interpretation": (
                "Manifest duration/sample metadata is structural audit evidence only. "
                "It cannot replace an explicit source-native time column because endpoint "
                "and row-count conventions differ across export families."
            ),
        },
        "mapping_results": rows,
        "execution_receipt": {
            "read_only": True,
            "selection_basis": "DECLARED_PUBLIC_FIELD_PRESENCE_ONLY",
            "manifest_files_may_contain_annotation_values_and_are_parsed": True,
            "manifest_annotation_fields_used_for_selection_or_mapping": False,
            "manifest_annotation_values_emitted": False,
            "manifest_structural_capability_level_and_timing_metadata_used_for_audit": True,
            "manifest_structural_metadata_is_model_input": False,
            "truth_values_read": False,
            "simulator_answer_proxy_values_read": False,
            "entity_case_run_path_or_seed_values_emitted": False,
            "raw_rows_emitted": False,
            "model_fit_count": 0,
            "threshold_selection_count": 0,
            "split_creation_count": 0,
            "seed_creation_or_reservation_count": 0,
        },
        "readiness": {
            "common_semantic_interface_exercised": not blockers,
            "training_qualified": False,
            "stage_s1_fit_authorized": False,
            "stage_b_authorized": False,
            "g2_pass": False,
            "g3_pass": False,
        },
        "claim_boundary": (
            "A PASS proves only that actual telemetry from one Sat-Agent source can be projected "
            "through declared, truth-blind semantic contracts. It is not data qualification, "
            "diagnostic accuracy, cross-source generalization, flight evidence, or Stage B authorization."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--semantic-schema",
        type=Path,
        default=ROOT / "configs/semantics/core_semantic_schema.json",
    )
    parser.add_argument(
        "--archetype-catalog",
        type=Path,
        default=ROOT / "configs/semantics/core_component_archetypes.json",
    )
    parser.add_argument(
        "--channel-maps",
        type=Path,
        default=ROOT / "configs/semantics/sat_agent_platform_channel_maps.s0.v1.json",
    )
    parser.add_argument(
        "--schema-profiles",
        type=Path,
        default=ROOT / "configs/data_sources/sat_agent_platform_schema_profiles.s0.v1.json",
    )
    parser.add_argument("--row-limit", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.row_limit < 1:
        raise ValueError("row limit must be positive")
    source_root = args.root.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    output = args.output.expanduser().resolve()
    try:
        output.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise ValueError("validation output must not be placed inside the source tree")
    payload = build_validation(
        source_root,
        semantic_schema_path=args.semantic_schema.resolve(),
        archetype_catalog_path=args.archetype_catalog.resolve(),
        channel_map_path=args.channel_maps.resolve(),
        schema_profile_path=args.schema_profiles.resolve(),
        row_limit=args.row_limit,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "summary": payload["summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
