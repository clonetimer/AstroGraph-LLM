#!/usr/bin/env python3
"""Inventory truth-blind public channel support for missing formal profiles.

This S0 utility reads only adapter metadata and public CSV headers.  It does not
read telemetry rows or labels and does not create semantic mappings by guessing
from names.  Its output separates profiles that already exist from capabilities
that have enough declared public structure for a later manually reviewed map.
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

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from astrograph.data_sources import (  # noqa: E402
    DataSourceType,
    DatasetSourceConfig,
    build_data_source,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def _declared_output_units(contract_path: Path) -> dict[str, str]:
    payload = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    outputs = payload.get("outputs", [])
    units: dict[str, str] = {}
    if isinstance(outputs, list):
        for item in outputs:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                units[str(item["name"])] = str(item.get("unit") or "UNDECLARED")
    elif isinstance(outputs, dict):
        for name, item in outputs.items():
            if isinstance(item, dict):
                units[str(name)] = str(item.get("unit") or "UNDECLARED")
            else:
                units[str(name)] = "UNDECLARED"
    return units


def _row_count_convention(entity: Any) -> str:
    row_count = entity.row_count
    duration = entity.metadata.get("duration_s")
    sample = entity.metadata.get("sample_s")
    if row_count is None or duration is None or sample is None:
        return "INCOMPLETE_TIMING_METADATA"
    try:
        interval_count = float(duration) / float(sample)
    except (TypeError, ValueError, ZeroDivisionError):
        return "INVALID_TIMING_METADATA"
    tolerance = max(1e-9, abs(interval_count) * 1e-9)
    if abs(float(row_count) - interval_count) <= tolerance:
        return "ROW_COUNT_EQUALS_DURATION_DIV_SAMPLE"
    if abs(float(row_count) - interval_count - 1.0) <= tolerance:
        return "ROW_COUNT_EQUALS_DURATION_DIV_SAMPLE_PLUS_ONE"
    return "ROW_COUNT_CONVENTION_OTHER"


def build_audit(
    source_root: Path,
    diagnosis_scope_path: Path,
    schema_profile_path: Path,
    source_environment_receipt_path: Path,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    scope = _load(diagnosis_scope_path)
    profile_catalog = _load(schema_profile_path)
    formal_objects = scope.get("objects", [])
    profiles = profile_catalog.get("profiles", [])
    if not isinstance(formal_objects, list) or not isinstance(profiles, list):
        raise ValueError("scope and profile catalog must contain list-valued records")

    configured_capabilities = {
        str(item.get("capability_id"))
        for item in profiles
        if isinstance(item, dict)
    }
    formal_capabilities = {
        str(item.get("capability_id")): item
        for item in formal_objects
        if isinstance(item, dict)
    }
    capability_dir = source_root / "src" / "sat_sim" / "capabilities"
    contract_manifest: list[dict[str, str]] = []
    declared_units: dict[str, dict[str, str]] = {}
    for capability_id in sorted(formal_capabilities):
        contract_path = capability_dir / f"{capability_id}.yaml"
        if not contract_path.is_file():
            raise FileNotFoundError(contract_path)
        contract_manifest.append(
            {"capability_id": capability_id, "sha256": _sha256(contract_path)}
        )
        declared_units[capability_id] = _declared_output_units(contract_path)

    config = DatasetSourceConfig(
        source_type=DataSourceType.SAT_AGENT_PLATFORM,
        root=str(source_root),
        read_only=True,
        lazy=True,
        recursive=True,
        timestamp_column="time_s",
        metadata={"stage": "P018_S0_ZERO_FIT", "source_count": 1},
    )
    grouped: dict[str, list[Any]] = {capability: [] for capability in formal_capabilities}
    with build_data_source(config) as source:
        metadata = source.get_metadata()
        if metadata.metadata.get("source_count") != 1:
            raise ValueError("Sat-Agent adapter must remain source_count=1")
        for entity in source.list_entities():
            capability = str(entity.metadata.get("capability_id") or "UNDECLARED")
            if capability in grouped:
                grouped[capability].append(entity)

        capability_rows: list[dict[str, Any]] = []
        missing_time_conventions: Counter[str] = Counter()
        missing_time_targets: list[dict[str, Any]] = []
        for capability_id, formal in sorted(formal_capabilities.items()):
            entities = grouped[capability_id]
            channel_sets = [set(source.list_channels(entity.entity_id)) for entity in entities]
            common = set.intersection(*channel_sets) if channel_sets else set()
            union = set.union(*channel_sets) if channel_sets else set()
            contract_units = declared_units[capability_id]
            common_fields = [
                {
                    "source_field": field,
                    "declared_unit": contract_units.get(field, "UNDECLARED"),
                }
                for field in sorted(common)
            ]
            explicit_time_count = sum(
                entity.timestamp_column == "time_s" for entity in entities
            )
            missing_time_entities = [
                entity for entity in entities if entity.timestamp_column != "time_s"
            ]
            for entity in missing_time_entities:
                convention = _row_count_convention(entity)
                missing_time_conventions[convention] += 1
                missing_time_targets.append(
                    {
                        "entity_id": entity.entity_id,
                        "capability_id": capability_id,
                        "row_count": entity.row_count,
                        "duration_s": entity.metadata.get("duration_s"),
                        "sample_s": entity.metadata.get("sample_s"),
                        "row_count_convention": convention,
                    }
                )
            exact_level_count = sum(
                str(entity.metadata.get("hierarchy_level")) == str(formal.get("level"))
                for entity in entities
            )
            profile_exists = capability_id in configured_capabilities
            if profile_exists:
                status = "EXISTING_PROFILE"
            elif not entities:
                status = "NO_DISCOVERED_PUBLIC_ENTITY"
            elif not common:
                status = "NO_COMMON_SAFE_PUBLIC_CHANNEL"
            elif explicit_time_count != len(entities):
                status = "TIME_AXIS_INCOMPLETE"
            elif exact_level_count != len(entities):
                status = "HIERARCHY_BINDING_INCOMPLETE"
            else:
                status = "PROFILE_FEASIBLE_REQUIRES_MANUAL_SEMANTIC_MAP"
            capability_rows.append(
                {
                    "capability_id": capability_id,
                    "profile_status": status,
                    "formal_level": formal.get("level"),
                    "discovered_entity_count": len(entities),
                    "explicit_time_entity_count": explicit_time_count,
                    "missing_explicit_time_entity_count": len(missing_time_entities),
                    "exact_hierarchy_level_entity_count": exact_level_count,
                    "common_safe_public_channel_count": len(common),
                    "union_safe_public_channel_count": len(union),
                    "common_safe_public_channels": common_fields,
                    "public_channel_names_are_model_inputs": False,
                    "semantic_quantity_or_port_role_inferred": False,
                }
            )

    status_counts: dict[str, int] = {}
    for row in capability_rows:
        key = str(row["profile_status"])
        status_counts[key] = status_counts.get(key, 0) + 1
    feasible_missing_count = status_counts.get(
        "PROFILE_FEASIBLE_REQUIRES_MANUAL_SEMANTIC_MAP", 0
    )
    return {
        "schema_version": "astrograph.p018.sat-agent-semantic-profile-gap-audit.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage": "P018_S0_ZERO_FIT",
        "status": "PROFILE_GAPS_AUDITED_NO_PROFILE_MUTATION",
        "source_id": "SAT_AGENT_PLATFORM",
        "source_count": 1,
        "input_bindings": {
            "diagnosis_scope": {
                "path": "protocols/acceptance/diagnosis_scope.json",
                "sha256": _sha256(diagnosis_scope_path),
            },
            "schema_profile_catalog": {
                "path": "configs/data_sources/sat_agent_platform_schema_profiles.s0.v1.json",
                "sha256": _sha256(schema_profile_path),
            },
            "source_environment_receipt": {
                "path": "governance/evidence/0.7.1-extension/p018_sat_agent_source_environment_receipt.v1.json",
                "sha256": _sha256(source_environment_receipt_path),
            },
            "formal_capability_contract_manifest": {
                "path": "${SAT_AGENT_PLATFORM_ROOT}/src/sat_sim/capabilities/<formal-capability>.yaml",
                "file_count": len(contract_manifest),
                "aggregate_sha256": _canonical_sha256(contract_manifest),
                "individual_source_paths_emitted": False,
            },
        },
        "summary": {
            "formal_object_capability_denominator": len(formal_capabilities),
            "configured_exact_capability_profile_count": len(
                set(formal_capabilities) & configured_capabilities
            ),
            "missing_exact_capability_profile_count": len(
                set(formal_capabilities) - configured_capabilities
            ),
            "feasible_missing_profile_count": feasible_missing_count,
            "missing_explicit_time_entity_count": len(missing_time_targets),
            "missing_time_capability_count": sum(
                row["missing_explicit_time_entity_count"] > 0
                for row in capability_rows
            ),
            "missing_time_row_count_convention_counts": dict(
                sorted(missing_time_conventions.items())
            ),
            "missing_time_target_manifest_sha256": _canonical_sha256(
                sorted(missing_time_targets, key=lambda item: str(item["entity_id"]))
            ),
            "individual_missing_time_target_identities_emitted": False,
            "status_counts": dict(sorted(status_counts.items())),
            "profile_or_channel_map_created_count": 0,
        },
        "capability_results": capability_rows,
        "manual_review_contract": {
            "required_before_profile_addition": [
                "Choose component-specific public fields from the observed common set; do not copy one common feature list across objects.",
                "Bind every selected field to an existing reviewed quantity kind, port role and component archetype; do not infer these identities from field names in this audit.",
                "Keep source, path, case, scenario, seed, split, label, truth, fault and answer-proxy fields denied.",
                "Rerun the full 237-case structural denominator and the exact formal-capability binding audit after any profile change.",
            ],
            "automatic_profile_generation_authorized": False,
            "model_fit_or_feature_selection_authorized": False,
        },
        "execution_receipt": {
            "public_header_and_structural_metadata_only": True,
            "telemetry_row_read_count": 0,
            "label_or_truth_value_read_count": 0,
            "entity_or_case_id_emitted": False,
            "profile_write_count": 0,
            "channel_map_write_count": 0,
            "model_fit_count": 0,
            "split_count": 0,
            "seed_creation_or_reservation_count": 0,
            "formal_hidden_acceptance_truth_accessed": False,
        },
        "claim_boundary": (
            "S0 public-header gap audit only. Feasible means safe common public fields "
            "exist; it does not approve an automatically inferred semantic map, a model "
            "feature, an episode denominator, diagnosis performance, Stage S1 or Stage B."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--diagnosis-scope", type=Path, required=True)
    parser.add_argument("--schema-profiles", type=Path, required=True)
    parser.add_argument("--source-environment-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {
        args.diagnosis_scope.resolve(),
        args.schema_profiles.resolve(),
        args.source_environment_receipt.resolve(),
    }
    output = args.output.resolve()
    if output in inputs:
        raise ValueError("output must not overwrite an input")
    result = build_audit(
        args.root,
        args.diagnosis_scope,
        args.schema_profiles,
        args.source_environment_receipt,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
