#!/usr/bin/env python3
"""Audit the frozen Sat-Agent catalog join and zero-fit holdout prerequisites.

The audit is deliberately aggregate and truth-blind.  It verifies the frozen
31-object/93-scenario/145-effect catalog against Sat-Agent's public engineering
coverage report, then asks whether existing governed evidence materializes the
episode lineage required by the preregistered grouped holdouts.  It never reads
telemetry rows, creates a split or seed, or fits a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


EXPECTED_HOLDOUT_AXES = {
    "LEAVE_ONE_OBJECT_INSTANCE_OUT",
    "LEAVE_ONE_CONFIGURATION_FAMILY_OUT",
    "LEAVE_ONE_SCENARIO_OR_CAMPAIGN_FAMILY_OUT",
    "OPERATING_CONDITION_OUT",
    "GROUPED_SEVERITY_ONSET_DURATION_TIME_AND_SEED",
    "LEAVE_ONE_ARCHETYPE_OUT_REJECTION_OR_ADAPTATION_ONLY",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path.name}")
    return value


def _counter_difference_size(left: Counter[Any], right: Counter[Any]) -> int:
    return sum((left - right).values())


def _normalized_source_effects(
    scenario_type: str, effect_ids: Iterable[Any]
) -> tuple[str, ...]:
    values = tuple(str(value) for value in effect_ids)
    if scenario_type == "nominal":
        if values not in ((), ("nominal",)):
            return values
        return ()
    return values


def audit_formal_catalog(
    diagnosis_scope: dict[str, Any],
    fault_taxonomy: dict[str, Any],
    coverage_matrix: dict[str, Any],
    source_report: dict[str, Any],
) -> dict[str, Any]:
    """Return aggregate, multiplicity-preserving catalog comparison results."""

    scope_objects = diagnosis_scope.get("objects", [])
    taxonomy_rows = fault_taxonomy.get("faults", [])
    coverage_rows = coverage_matrix.get("rows", [])
    source_objects = source_report.get("objects", [])
    if not all(
        isinstance(rows, list)
        for rows in (scope_objects, taxonomy_rows, coverage_rows, source_objects)
    ):
        raise ValueError("catalog inputs must contain list-valued records")

    frozen_object_records = Counter(
        (
            row.get("object_id"),
            row.get("level"),
            row.get("domain"),
            row.get("capability_id"),
        )
        for row in scope_objects
    )
    source_object_records = Counter(
        (
            row.get("object_id"),
            row.get("level"),
            row.get("domain"),
            row.get("primary_capability_id"),
        )
        for row in source_objects
    )

    frozen_scenarios = Counter(
        (
            row.get("object_id"),
            row.get("capability_id"),
            row.get("scenario_type"),
            row.get("scenario_id"),
            tuple(str(value) for value in row.get("effect_ids", [])),
        )
        for row in coverage_rows
    )
    source_scenarios: Counter[Any] = Counter()
    for source_object in source_objects:
        modes = source_object.get("modes", {})
        if not isinstance(modes, dict):
            raise ValueError("source object modes must be an object")
        for scenario_type, mode in modes.items():
            if not isinstance(mode, dict):
                raise ValueError("source mode must be an object")
            effects = _normalized_source_effects(
                str(scenario_type), mode.get("effect_ids", [])
            )
            for scenario_id in mode.get("scenario_ids", []):
                source_scenarios[
                    (
                        source_object.get("object_id"),
                        source_object.get("primary_capability_id"),
                        scenario_type,
                        scenario_id,
                        effects,
                    )
                ] += 1

    taxonomy_occurrences = Counter(
        (row.get("object_id"), row.get("scenario_type"), row.get("effect_id"))
        for row in taxonomy_rows
    )
    coverage_occurrences: Counter[Any] = Counter()
    for row in coverage_rows:
        for effect_id in row.get("effect_ids", []):
            coverage_occurrences[
                (row.get("object_id"), row.get("scenario_type"), effect_id)
            ] += 1
    source_occurrences: Counter[Any] = Counter()
    for source_object in source_objects:
        for scenario_type, mode in source_object.get("modes", {}).items():
            if scenario_type == "nominal":
                continue
            for effect_id in mode.get("effect_ids", []):
                source_occurrences[
                    (source_object.get("object_id"), scenario_type, effect_id)
                ] += 1

    duplicate_checks = {
        "frozen_object_id_duplicate_count": len(scope_objects)
        - len({row.get("object_id") for row in scope_objects}),
        "source_object_id_duplicate_count": len(source_objects)
        - len({row.get("object_id") for row in source_objects}),
        "frozen_scenario_key_duplicate_count": len(coverage_rows)
        - len(
            {
                (row.get("object_id"), row.get("scenario_type"), row.get("scenario_id"))
                for row in coverage_rows
            }
        ),
        "source_scenario_record_duplicate_count": sum(
            count - 1 for count in source_scenarios.values() if count > 1
        ),
        "fault_id_duplicate_count": len(taxonomy_rows)
        - len({row.get("fault_id") for row in taxonomy_rows}),
    }
    comparison = {
        "object_record_missing_from_source_count": _counter_difference_size(
            frozen_object_records, source_object_records
        ),
        "object_record_extra_in_source_count": _counter_difference_size(
            source_object_records, frozen_object_records
        ),
        "scenario_record_missing_from_source_count": _counter_difference_size(
            frozen_scenarios, source_scenarios
        ),
        "scenario_record_extra_in_source_count": _counter_difference_size(
            source_scenarios, frozen_scenarios
        ),
        "taxonomy_vs_coverage_missing_count": _counter_difference_size(
            taxonomy_occurrences, coverage_occurrences
        ),
        "taxonomy_vs_coverage_extra_count": _counter_difference_size(
            coverage_occurrences, taxonomy_occurrences
        ),
        "taxonomy_vs_source_missing_count": _counter_difference_size(
            taxonomy_occurrences, source_occurrences
        ),
        "taxonomy_vs_source_extra_count": _counter_difference_size(
            source_occurrences, taxonomy_occurrences
        ),
    }
    report_issue_count = len(source_report.get("issues", []))
    report_blocker_count = len(source_report.get("blockers", []))
    failures = [key for key, value in {**duplicate_checks, **comparison}.items() if value]
    if source_report.get("gate_status") != "PASS":
        failures.append("source_report_gate_status")
    if report_issue_count:
        failures.append("source_report_issue_count")
    if report_blocker_count:
        failures.append("source_report_blocker_count")

    return {
        "status": "PASS" if not failures else "FAIL",
        "denominators": {
            "frozen_object_count": len(scope_objects),
            "frozen_scenario_row_count": len(coverage_rows),
            "frozen_fault_effect_occurrence_count": len(taxonomy_rows),
            "frozen_unique_effect_id_count": len(
                {row.get("effect_id") for row in taxonomy_rows}
            ),
            "source_object_count": len(source_objects),
            "source_scenario_record_count": sum(source_scenarios.values()),
            "source_non_nominal_effect_occurrence_count": sum(
                source_occurrences.values()
            ),
        },
        "duplicate_checks": duplicate_checks,
        "exact_comparison": comparison,
        "source_report_controls": {
            "gate_status": source_report.get("gate_status"),
            "technical_baseline_status": source_report.get(
                "technical_baseline_status"
            ),
            "issue_count": report_issue_count,
            "blocker_count": report_blocker_count,
        },
        "failure_fields": sorted(failures),
    }


def audit_semantic_binding(
    diagnosis_scope: dict[str, Any],
    fault_taxonomy: dict[str, Any],
    coverage_matrix: dict[str, Any],
    semantic_validation: dict[str, Any],
) -> dict[str, Any]:
    """Measure exact primary-capability profile binding without inferring episodes."""

    scope_objects = diagnosis_scope.get("objects", [])
    taxonomy_rows = fault_taxonomy.get("faults", [])
    coverage_rows = coverage_matrix.get("rows", [])
    mapping_results = semantic_validation.get("mapping_results", [])

    passing_profile_capabilities: set[str] = set()
    for mapping in mapping_results:
        matched = int(mapping.get("declared_profile_matched_entity_count", 0))
        passed = int(mapping.get("declared_profile_passed_entity_count", 0))
        failed = int(mapping.get("declared_profile_failed_entity_count", 0))
        capability = mapping.get("declared_profile_capability_id")
        if isinstance(capability, str) and matched > 0 and passed == matched and failed == 0:
            passing_profile_capabilities.add(capability)

    frozen_capabilities = {
        str(row.get("capability_id")) for row in scope_objects
    }
    exact_capabilities = frozen_capabilities & passing_profile_capabilities
    bound_objects = {
        row.get("object_id")
        for row in scope_objects
        if row.get("capability_id") in exact_capabilities
    }
    associated_rows = [
        row for row in coverage_rows if row.get("object_id") in bound_objects
    ]
    associated_effect_occurrences = sum(
        len(row.get("effect_ids", [])) for row in associated_rows
    )
    mechanism_fields_present = sum(
        int(bool(str(row.get("mechanism_class", "")).strip()))
        for row in taxonomy_rows
    )
    object_count = len(scope_objects)
    status = "PASS" if len(bound_objects) == object_count else "PARTIAL"

    return {
        "status": status,
        "binding_kind": "EXACT_FROZEN_PRIMARY_CAPABILITY_TO_PASSING_PUBLIC_SEMANTIC_PROFILE",
        "semantic_profile_count": len(mapping_results),
        "passing_semantic_profile_count": len(passing_profile_capabilities),
        "exact_primary_capability_profile_count": len(exact_capabilities),
        "non_primary_or_unmatched_profile_count": len(
            passing_profile_capabilities - frozen_capabilities
        ),
        "formal_object_denominator": object_count,
        "exactly_bound_formal_object_count": len(bound_objects),
        "unbound_formal_object_count": object_count - len(bound_objects),
        "formal_scenario_row_denominator": len(coverage_rows),
        "associated_formal_scenario_row_count": len(associated_rows),
        "formal_fault_effect_occurrence_denominator": len(taxonomy_rows),
        "associated_formal_fault_effect_occurrence_count": (
            associated_effect_occurrences
        ),
        "materialized_formal_episode_count": 0,
        "materialized_formal_episode_binding_status": "NOT_ESTABLISHED",
        "g1_mechanism_class_field_present_row_count": mechanism_fields_present,
        "g1_mechanism_class_mapping_status": (
            "PRESENT_FOR_ALL_ROWS"
            if taxonomy_rows and mechanism_fields_present == len(taxonomy_rows)
            else "NOT_ESTABLISHED_DO_NOT_INFER"
        ),
        "interpretation": (
            "Associated formal rows are catalog rows for objects with an exact passing "
            "public semantic profile. They are not materialized scenario episodes, do not "
            "establish diagnosis labels, and do not map effects to G1 mechanism classes."
        ),
    }


def build_lineage_receipt(
    *,
    paths: dict[str, Path],
    diagnosis_scope: dict[str, Any],
    fault_taxonomy: dict[str, Any],
    coverage_matrix: dict[str, Any],
    source_report: dict[str, Any],
    semantic_validation: dict[str, Any],
) -> dict[str, Any]:
    catalog = audit_formal_catalog(
        diagnosis_scope, fault_taxonomy, coverage_matrix, source_report
    )
    semantic = audit_semantic_binding(
        diagnosis_scope, fault_taxonomy, coverage_matrix, semantic_validation
    )
    if catalog["status"] == "FAIL":
        overall = "FORMAL_CATALOG_FAIL"
    elif semantic["status"] == "PASS":
        overall = "FORMAL_CATALOG_AND_STRUCTURAL_SEMANTIC_BINDING_PASS"
    else:
        overall = "FORMAL_CATALOG_PASS_DIAGNOSIS_LINEAGE_BLOCKED"

    binding_tokens = {
        "diagnosis_scope": "protocols/acceptance/diagnosis_scope.json",
        "fault_taxonomy": "protocols/acceptance/fault_taxonomy.json",
        "coverage_matrix": "protocols/acceptance/coverage_matrix.json",
        "source_catalog_report": (
            "${SAT_AGENT_PLATFORM_ROOT}/reports/object_scenario_coverage.json"
        ),
        "semantic_validation": (
            "governance/evidence/0.7.1-extension/"
            "p018_sat_agent_semantic_contract_validation.post_repair.v1.json"
        ),
    }
    return {
        "schema_version": "astrograph.p018.sat-agent-formal-scope-lineage-audit.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage": "P018_S0_ZERO_FIT",
        "status": overall,
        "source_id": "SAT_AGENT_PLATFORM",
        "source_count": 1,
        "input_bindings": {
            key: {"path": binding_tokens[key], "sha256": _sha256_file(path)}
            for key, path in paths.items()
            if key in binding_tokens
        },
        "formal_catalog_join": catalog,
        "public_semantic_binding": semantic,
        "denominator_identity_guard": {
            "formal_fault_effect_denominator": 145,
            "report_matrix_unique_physical_trace_fingerprint_count": 145,
            "same_numeric_value_is_coincidental": True,
            "interchangeable_or_joinable": False,
            "interpretation": (
                "Fault/effect occurrences are catalog identities; physical trace "
                "fingerprints are content-deduplication results. Neither is a diagnosis "
                "performance denominator at S0."
            ),
        },
        "execution_receipt": {
            "read_only_aggregate_metadata_audit": True,
            "telemetry_row_read_count": 0,
            "label_or_truth_value_read_count": 0,
            "model_fit_count": 0,
            "threshold_selection_count": 0,
            "split_creation_count": 0,
            "seed_creation_or_reservation_count": 0,
            "formal_hidden_acceptance_truth_accessed": False,
            "formal_denominator_changed": False,
        },
        "decision": {
            "formal_catalog_join": catalog["status"],
            "formal_semantic_binding": semantic["status"],
            "diagnosis_episode_lineage_qualified": False,
            "stage_s1_fit_authorized": False,
            "stage_b_authorized": False,
            "g2_pass": False,
            "g3_pass": False,
        },
        "claim_boundary": (
            "S0 catalog and structural semantic-binding audit only. A PASS catalog "
            "join proves identifier/association consistency, not episode materialization, "
            "identifiability, diagnosis performance, G2, G3, Stage S1 or Stage B."
        ),
    }


def build_holdout_receipt(
    *,
    paths: dict[str, Path],
    inventory: dict[str, Any],
    report_matrix: dict[str, Any],
    semantic_validation: dict[str, Any],
    split_preregistration: dict[str, Any],
    lineage_receipt: dict[str, Any],
    lineage_path: Path,
) -> dict[str, Any]:
    preregistered_axes = {
        row.get("axis_id")
        for row in split_preregistration.get("preregistered_holdout_axes", [])
    }
    if preregistered_axes != EXPECTED_HOLDOUT_AXES:
        raise ValueError("grouped split preregistration axes changed")

    inventory_summary = inventory.get("summary", {})
    matrix_summary = report_matrix.get("summary", {})
    semantic_summary = semantic_validation.get("summary", {})
    axis_results = [
        {
            "axis_id": "LEAVE_ONE_OBJECT_INSTANCE_OUT",
            "status": "NOT_ESTABLISHED_FROM_CURRENT_GOVERNED_EVIDENCE",
            "observed": "Aggregate entity counts only; entity/case IDs are intentionally not emitted.",
            "missing_material_prerequisite": "A stable component_instance lineage field with at least two materially distinct instances in every evaluated group.",
        },
        {
            "axis_id": "LEAVE_ONE_CONFIGURATION_FAMILY_OUT",
            "status": "NOT_ESTABLISHED_FROM_CURRENT_GOVERNED_EVIDENCE",
            "observed": "No governed satellite_configuration_family assignment or topology provenance is emitted.",
            "missing_material_prerequisite": "Frozen configuration-family identity backed by materially different parameters or topology, not an alias or signal value.",
        },
        {
            "axis_id": "LEAVE_ONE_SCENARIO_OR_CAMPAIGN_FAMILY_OUT",
            "status": "CATALOG_IDS_PRESENT_MATERIAL_EPISODE_LINEAGE_NOT_ESTABLISHED",
            "observed": "The formal 93-row scenario catalog joins exactly, but no governed campaign_template_family-to-episode assignment exists.",
            "missing_material_prerequisite": "Frozen episode-level campaign template family and parent lineage for every materialized run.",
        },
        {
            "axis_id": "OPERATING_CONDITION_OUT",
            "status": "NOT_ESTABLISHED_FROM_CURRENT_GOVERNED_EVIDENCE",
            "observed": "Physical traces exist, but no frozen operating_condition_trajectory_id grouping or coverage denominator is emitted.",
            "missing_material_prerequisite": "Auditable operating-condition trajectory groups with non-overlapping material ranges and complete denominators.",
        },
        {
            "axis_id": "GROUPED_SEVERITY_ONSET_DURATION_TIME_AND_SEED",
            "status": "NOT_MATERIALIZED_NOT_AUTHORIZED",
            "observed": "The preregistration contains no assignment or seed and current evidence emits no run seed values.",
            "missing_material_prerequisite": "Timer-frozen lineage groups and authorized development/calibration seeds before any split or fit.",
        },
        {
            "axis_id": "LEAVE_ONE_ARCHETYPE_OUT_REJECTION_OR_ADAPTATION_ONLY",
            "status": "STRUCTURAL_PROFILE_CATALOG_ONLY_DENOMINATORS_NOT_MATERIALIZED",
            "observed": "Seven archetype profiles are structurally exercised, while the 689-entity inventory declares archetype as UNDECLARED and no episode denominator exists.",
            "missing_material_prerequisite": "Episode-level archetype assignments and frozen rejection/adaptation denominators; this axis cannot become default exact zero-shot diagnosis.",
        },
    ]
    active_count = sum(row["status"] == "ACTIVE" for row in axis_results)

    token_paths = {
        "post_repair_inventory": (
            "governance/evidence/0.7.1-extension/"
            "p018_sat_agent_inventory.post_repair.v1.json"
        ),
        "report_matrix": (
            "governance/evidence/0.7.1-extension/"
            "p018_sat_agent_report_matrix.v1.json"
        ),
        "semantic_validation": (
            "governance/evidence/0.7.1-extension/"
            "p018_sat_agent_semantic_contract_validation.post_repair.v1.json"
        ),
        "split_preregistration": (
            "protocols/acceptance/"
            "p018_sat_agent_s1_grouped_split_preregistration.json"
        ),
    }
    bindings = {
        key: {"path": token_paths[key], "sha256": _sha256_file(path)}
        for key, path in paths.items()
        if key in token_paths
    }
    bindings["formal_scope_lineage_audit"] = {
        "path": (
            "governance/evidence/0.7.1-extension/"
            "p018_sat_agent_formal_scope_lineage_audit.v1.json"
        ),
        "sha256": _sha256_file(lineage_path),
    }
    return {
        "schema_version": "astrograph.p018.sat-agent-holdout-materialization-audit.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage": "P018_S0_ZERO_FIT",
        "status": "HOLDOUT_PREREQUISITES_NOT_ESTABLISHED",
        "source_id": "SAT_AGENT_PLATFORM",
        "source_count": 1,
        "input_bindings": bindings,
        "observed_aggregate_evidence": {
            "discovered_readable_entity_count": inventory_summary.get(
                "discovered_readable_entity_count"
            ),
            "qualified_independent_observation_count": inventory_summary.get(
                "qualified_independent_observation_count"
            ),
            "inventory_component_archetype_counts": inventory_summary.get(
                "component_archetype_counts"
            ),
            "inventory_backend_family_count": len(
                inventory_summary.get("backend_family_counts", {})
            ),
            "inventory_hierarchy_level_count": len(
                inventory_summary.get("hierarchy_level_counts", {})
            ),
            "report_manifest_count": matrix_summary.get("manifest_file_count"),
            "report_unique_physical_trace_count": matrix_summary.get(
                "unique_physical_trace_count"
            ),
            "report_duplicate_physical_trace_count": matrix_summary.get(
                "duplicate_physical_trace_count"
            ),
            "report_cross_annotation_conflict_count": matrix_summary.get(
                "physical_trace_cross_annotation_conflict_count"
            ),
            "structurally_exercised_archetype_count": semantic_summary.get(
                "exercised_component_archetype_count"
            ),
            "formal_catalog_join_status": lineage_receipt.get(
                "formal_catalog_join", {}
            ).get("status"),
        },
        "holdout_axis_denominator": len(EXPECTED_HOLDOUT_AXES),
        "active_holdout_axis_count": active_count,
        "axis_results": axis_results,
        "supplemental_non_substitutable_axes": [
            {
                "axis_id": "BACKEND_FAMILY",
                "status": "DISTINCT_VALUES_OBSERVED_PAIRED_LINEAGE_NOT_ESTABLISHED",
                "rule": "Multiple backends remain one Sat-Agent source and cannot substitute for instance, configuration, campaign or operating-condition holdouts.",
            },
            {
                "axis_id": "HIERARCHY_LEVEL",
                "status": "DISTINCT_LEVELS_OBSERVED_CAUSAL_EPISODE_LINEAGE_NOT_ESTABLISHED",
                "rule": "Component/subsystem/spacecraft aggregates cannot establish localization until their views are tied to one causal episode lineage.",
            },
        ],
        "decision": {
            "material_episode_inventory_frozen": False,
            "any_preregistered_holdout_axis_active": bool(active_count),
            "split_materialization_authorized": False,
            "stage_s1_fit_ready": False,
            "timer_freeze_still_required": True,
            "next_source_write_authorized": False,
        },
        "execution_receipt": {
            "aggregate_governed_evidence_only": True,
            "telemetry_row_read_count": 0,
            "label_or_truth_value_read_count": 0,
            "episode_assignment_count": 0,
            "split_count": 0,
            "seed_creation_or_reservation_count": 0,
            "model_fit_count": 0,
            "threshold_selection_count": 0,
            "formal_hidden_acceptance_truth_accessed": False,
            "formal_denominator_changed": False,
        },
        "claim_boundary": (
            "S0 materialization-prerequisite audit only. NOT_ESTABLISHED means the "
            "current governed evidence cannot prove an active holdout; it does not claim "
            "the Sat-Agent source can never provide that axis. No new Sat-Agent source "
            "write, split, seed, fit, Stage S1 or Stage B is authorized."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnosis-scope", type=Path, required=True)
    parser.add_argument("--fault-taxonomy", type=Path, required=True)
    parser.add_argument("--coverage-matrix", type=Path, required=True)
    parser.add_argument("--source-catalog-report", type=Path, required=True)
    parser.add_argument("--semantic-validation", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--report-matrix", type=Path, required=True)
    parser.add_argument("--split-preregistration", type=Path, required=True)
    parser.add_argument("--lineage-output", type=Path, required=True)
    parser.add_argument("--holdout-output", type=Path, required=True)
    parser.add_argument(
        "--reuse-lineage",
        action="store_true",
        help="Load the existing lineage receipt instead of regenerating it.",
    )
    args = parser.parse_args()

    paths = {
        "diagnosis_scope": args.diagnosis_scope.resolve(),
        "fault_taxonomy": args.fault_taxonomy.resolve(),
        "coverage_matrix": args.coverage_matrix.resolve(),
        "source_catalog_report": args.source_catalog_report.resolve(),
        "semantic_validation": args.semantic_validation.resolve(),
        "post_repair_inventory": args.inventory.resolve(),
        "report_matrix": args.report_matrix.resolve(),
        "split_preregistration": args.split_preregistration.resolve(),
    }
    lineage_output = args.lineage_output.resolve()
    holdout_output = args.holdout_output.resolve()
    if lineage_output == holdout_output or lineage_output in paths.values() or holdout_output in paths.values():
        raise ValueError("outputs must be distinct and must not overwrite inputs")

    loaded = {key: _load_json(path) for key, path in paths.items()}
    if args.reuse_lineage:
        lineage = _load_json(lineage_output)
    else:
        lineage = build_lineage_receipt(
            paths=paths,
            diagnosis_scope=loaded["diagnosis_scope"],
            fault_taxonomy=loaded["fault_taxonomy"],
            coverage_matrix=loaded["coverage_matrix"],
            source_report=loaded["source_catalog_report"],
            semantic_validation=loaded["semantic_validation"],
        )
        lineage_output.parent.mkdir(parents=True, exist_ok=True)
        lineage_output.write_text(
            json.dumps(lineage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    holdout = build_holdout_receipt(
        paths=paths,
        inventory=loaded["post_repair_inventory"],
        report_matrix=loaded["report_matrix"],
        semantic_validation=loaded["semantic_validation"],
        split_preregistration=loaded["split_preregistration"],
        lineage_receipt=lineage,
        lineage_path=lineage_output,
    )
    holdout_output.parent.mkdir(parents=True, exist_ok=True)
    holdout_output.write_text(
        json.dumps(holdout, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "formal_catalog_join": lineage["formal_catalog_join"]["status"],
                "semantic_binding": lineage["public_semantic_binding"]["status"],
                "active_holdout_axis_count": holdout["active_holdout_axis_count"],
            }
        )
    )
    return 0 if lineage["formal_catalog_join"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
