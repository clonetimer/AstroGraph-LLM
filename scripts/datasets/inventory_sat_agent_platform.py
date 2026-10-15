#!/usr/bin/env python3
"""Write a redacted, aggregate inventory of one local Sat-Agent platform tree.

This is a Stage-S0 read-only utility.  It discovers only canonical ``trace.csv``
bundles and contracted ``astrograph/telemetry.csv`` projections, emits no raw
rows or entity/case identifiers, never calls ``read_labels``, and performs no
model fit, threshold selection, split creation, or campaign generation.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from astrograph.data_sources import (  # noqa: E402
    DataSourceType,
    DatasetSourceConfig,
    build_data_source,
)


def _text(value: object, fallback: str = "UNDECLARED") -> str:
    if value is None or value == "":
        return fallback
    return str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _control_hashes(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in ("pyproject.toml", "spec.md", "task.md", "checklist.md"):
        path = root / relative
        if path.is_file():
            result[relative] = _sha256(path)
    return result


def _astrograph_control_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in (
        "task.md",
        "checklist.md",
        "protocols/acceptance/p018_sat_agent_only_s0_scope.json",
        "configs/data_sources/sat_agent_platform.example.json",
        "configs/data_sources/sat_agent_platform_schema_profiles.s0.v1.json",
        "configs/semantics/sat_agent_platform_channel_maps.s0.v1.json",
        "configs/semantics/sat_agent_archetype_feature_contracts.s0.v1.json",
        "src/astrograph/data_sources/sat_agent_platform.py",
        "src/astrograph/semantics/feature_contracts.py",
    ):
        path = ROOT / relative
        if path.is_file():
            result[relative] = _sha256(path)
    return result


def _sorted_counts(values: Counter[str]) -> dict[str, int]:
    return {key: values[key] for key in sorted(values)}


def build_inventory(root: Path, include_globs: list[str]) -> dict[str, Any]:
    config = DatasetSourceConfig(
        source_type=DataSourceType.SAT_AGENT_PLATFORM,
        root=str(root),
        read_only=True,
        lazy=True,
        recursive=True,
        timestamp_column="time_s",
        include_globs=include_globs,
        metadata={"stage": "P018_S0_ZERO_FIT", "source_count": 1},
    )
    source = build_data_source(config)
    with source:
        entities = source.list_entities()
        source_metadata = source.get_metadata()

        levels: Counter[str] = Counter()
        archetypes: Counter[str] = Counter()
        capabilities: Counter[str] = Counter()
        backends: Counter[str] = Counter()
        policies: Counter[str] = Counter()
        fidelity_levels: Counter[str] = Counter()
        matrix: dict[tuple[str, str, str, str], dict[str, Any]] = defaultdict(
            lambda: {
                "discovered_entity_count": 0,
                "contracted_entity_count": 0,
                "declared_row_count_known": 0,
                "declared_row_count_total": 0,
                "model_channel_count_min": None,
                "model_channel_count_max": None,
                "contract_missing_model_channel_count": 0,
                "contract_rejected_prohibited_model_channel_count": 0,
            }
        )
        channel_counts: list[int] = []
        parse_issue_count = 0
        empty_model_channel_entity_count = 0

        for entity in entities:
            metadata = entity.metadata
            level = _text(metadata.get("hierarchy_level"))
            archetype = _text(metadata.get("component_archetype"))
            capability = _text(metadata.get("capability_id"))
            backend = _text(metadata.get("backend_family"))
            policy = _text(metadata.get("model_channel_policy"))
            fidelity = _text(metadata.get("fidelity_level"))
            levels[level] += 1
            archetypes[archetype] += 1
            capabilities[capability] += 1
            backends[backend] += 1
            policies[policy] += 1
            fidelity_levels[fidelity] += 1

            channel_count = int(entity.channel_count or 0)
            channel_counts.append(channel_count)
            empty_model_channel_entity_count += int(channel_count == 0)
            parse_issue_count += int(metadata.get("manifest_parse_status") != "ok")
            parse_issue_count += int(
                bool(metadata.get("contract_present"))
                and metadata.get("contract_parse_status") != "ok"
            )

            row = matrix[(level, archetype, capability, backend)]
            row["discovered_entity_count"] += 1
            row["contracted_entity_count"] += int(
                policy == "dataset_contract_allowlist"
            )
            row["contract_missing_model_channel_count"] += len(
                metadata.get("contract_missing_model_channels", [])
            )
            row["contract_rejected_prohibited_model_channel_count"] += len(
                metadata.get("contract_rejected_prohibited_model_channels", [])
            )
            if entity.row_count is not None:
                row["declared_row_count_known"] += 1
                row["declared_row_count_total"] += int(entity.row_count)
            current_min = row["model_channel_count_min"]
            current_max = row["model_channel_count_max"]
            row["model_channel_count_min"] = (
                channel_count if current_min is None else min(current_min, channel_count)
            )
            row["model_channel_count_max"] = (
                channel_count if current_max is None else max(current_max, channel_count)
            )

        matrix_rows = []
        for (level, archetype, capability, backend), values in sorted(matrix.items()):
            matrix_rows.append(
                {
                    "hierarchy_level": level,
                    "component_archetype": archetype,
                    "capability_id": capability,
                    "backend_family": backend,
                    **values,
                }
            )

        contracted_count = int(source_metadata.metadata["contracted_entity_count"])
        invalid_contract_count = int(
            source_metadata.metadata.get("invalid_contract_fail_closed_entity_count", 0)
        )
        skipped_unreadable = int(
            source_metadata.metadata.get("skipped_unreadable_candidate_count", 0)
        )
        return {
            "schema_version": "astrograph.p018.sat-agent-inventory.v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "stage": "P018_S0_ZERO_FIT",
            "source_id": "SAT_AGENT_PLATFORM",
            "source_count": 1,
            "adapter_key": "sat_agent_platform",
            "source_family_alias": "sat-agent-platform",
            "source": {
                "source_group_id": "SAT_AGENT_PLATFORM",
                "root_token": "${ASTROGRAPH_SAT_AGENT_ROOT}",
                "nature": "E0_ENGINEERING_SIMULATION",
                "source_control_file_sha256": _control_hashes(root),
            },
            "astrograph_control_sha256": _astrograph_control_hashes(),
            "execution_receipt": {
                "read_only": True,
                "raw_rows_emitted": False,
                "entity_or_case_ids_emitted": False,
                "model_fit_count": 0,
                "threshold_selection_count": 0,
                "split_or_seed_creation_count": 0,
                "campaign_generation_count": 0,
                "formal_hidden_acceptance_truth_accessed": False,
                "model_adapter_truth_access": "DENIED",
                "manifest_files_may_contain_annotation_values_and_are_parsed": True,
                "manifest_annotation_fields_used_in_inventory": False,
                "manifest_annotation_values_emitted": False,
                "note": (
                    "Source manifests/contracts can contain simulator annotations; this utility "
                    "uses only structural metadata and never emits or requests their truth values."
                ),
            },
            "discovery_scope": {
                "include_globs": include_globs or ["<all canonical Sat-Agent bundles>"],
                "canonical_layouts": [
                    "<case>/astrograph/telemetry.csv + dataset_contract.json",
                    "<case>/trace.csv + manifest.json",
                ],
                "canonical_projection_precedence": True,
            },
            "summary": {
                "discovered_readable_entity_count": len(entities),
                "contracted_entity_count": contracted_count,
                "invalid_contract_fail_closed_entity_count": invalid_contract_count,
                "conservatively_guarded_uncontracted_entity_count": (
                    len(entities) - contracted_count - invalid_contract_count
                ),
                "empty_model_channel_entity_count": empty_model_channel_entity_count,
                "parse_issue_count": parse_issue_count,
                "candidate_file_count": int(
                    source_metadata.metadata.get("candidate_file_count", len(entities))
                ),
                "skipped_unreadable_candidate_count": skipped_unreadable,
                "skipped_unreadable_candidate_error_counts": source_metadata.metadata.get(
                    "skipped_unreadable_candidate_error_counts", {}
                ),
                "model_channel_count_min": min(channel_counts, default=0),
                "model_channel_count_max": max(channel_counts, default=0),
                "content_deduplication_performed": False,
                "qualified_independent_observation_count": None,
                "hierarchy_level_counts": _sorted_counts(levels),
                "component_archetype_counts": _sorted_counts(archetypes),
                "capability_counts": _sorted_counts(capabilities),
                "backend_family_counts": _sorted_counts(backends),
                "model_channel_policy_counts": _sorted_counts(policies),
                "fidelity_level_counts": _sorted_counts(fidelity_levels),
            },
            "coverage_matrix": matrix_rows,
            "readiness": {
                "adapter_contract_tests_allowed": True,
                "model_fitting_authorized": False,
                "stage_s1_timer_freeze_required_before_fit": True,
                "stage_b_authorized": False,
                "g3_pass": False,
                "qualified_cross_component_training_matrix_ready": False,
                "reason": (
                    "Only contracted entities have an explicit model-input allowlist; remaining "
                    "heterogeneous bundles are inventory candidates pending semantic and fidelity qualification. "
                    "Unreadable historical candidates remain counted as readiness failures, and discovered "
                    "entity/declared-row counts are not treated as independent observations without content deduplication."
                ),
            },
            "claim_boundary": (
                "All rows come from one Sat-Agent platform source. Coverage across campaigns, "
                "components, levels, configurations, or backends supports only same-source study; "
                "it is not cross-source, flight, hardware, or real-world generalization evidence."
            ),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-glob",
        action="append",
        default=[],
        help="Optional relative allowlist glob; repeat for multiple campaign families",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Sat-Agent root is not a directory: {root}")
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("inventory output must not be placed inside the source tree")

    payload = build_inventory(root, list(args.include_glob))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": payload["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
