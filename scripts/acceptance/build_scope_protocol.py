#!/usr/bin/env python3
"""Build the frozen AstroGraph G1 scope and G2 coverage protocol from Sat-Agent contracts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


CASE_TYPES = [
    "normal", "known_single_fault", "known_multifault", "unknown_fault",
    "data_quality", "operating_condition_shift",
]


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root: Path, sat_root: Path) -> dict:
    output = root / "protocols" / "acceptance"
    scope_source = _read(sat_root / "configs/acceptance/object_scope.json")
    scenario_source = _read(sat_root / "configs/acceptance/scenario_baseline.json")
    objects = scope_source["objects"]
    scenarios = {item["object_id"]: item for item in scenario_source["objects"]}

    diagnosis_scope = {
        "schema_version": "astrograph-diagnosis-scope.v1",
        "protocol_version": "G1.1",
        "status": "FROZEN_TECHNICAL",
        "claim_boundary": "Internal engineering simulation diagnosis only; no flight or safety qualification.",
        "object_count": len(objects),
        "objects": [
            {
                "object_id": item["object_id"], "level": item["level"], "domain": item["domain"],
                "capability_id": item["primary_capability_id"],
            }
            for item in objects
        ],
    }
    taxonomy_rows = []
    coverage_rows = []
    for index, item in enumerate(objects):
        object_id = item["object_id"]
        capability_id = item["primary_capability_id"]
        capability = yaml.safe_load((sat_root / "src/sat_sim/capabilities" / f"{capability_id}.yaml").read_text(encoding="utf-8"))
        effects = capability.get("operator", {}).get("effects", [])
        by_kind = {
            kind: [effect["effect_id"] for effect in effects if effect.get("kind") == kind and effect.get("effect_id") != "nominal"]
            for kind in ("fault", "degradation")
        }
        for scenario_type in ("nominal", "fault", "degradation"):
            scenario = scenarios[object_id]["scenarios"][scenario_type][0]
            effect_ids = [] if scenario_type == "nominal" else by_kind[scenario_type]
            if scenario_type != "nominal" and not effect_ids:
                raise ValueError(f"{capability_id} has no {scenario_type} effects")
            coverage_rows.append({
                "scenario_id": scenario["scenario_id"],
                "object_id": object_id,
                "scenario_type": scenario_type,
                "capability_id": capability_id,
                "effect_ids": effect_ids,
                "support_status": "SUPPORTED",
                "split_group": ("development", "validation", "holdout")[index % 3],
                "required_case_types": CASE_TYPES,
            })
            for effect_id in effect_ids:
                taxonomy_rows.append({
                    "fault_id": f"{object_id}:{effect_id}", "object_id": object_id,
                    "scenario_type": scenario_type, "effect_id": effect_id,
                })

    coverage = {
        "schema_version": "astrograph-coverage-matrix.v1",
        "protocol_version": "G1.1",
        "status": "FROZEN_TECHNICAL",
        "expansion_policy": {"required_case_types": CASE_TYPES, "formal_generation_requires_independent_approval": True},
        "rows": coverage_rows,
    }
    taxonomy = {
        "schema_version": "astrograph-fault-taxonomy.v1", "protocol_version": "G1.1",
        "status": "FROZEN_TECHNICAL", "fault_count": len(taxonomy_rows), "faults": taxonomy_rows,
    }
    metric_policy = {
        "schema_version": "astrograph-metric-policy.v1", "protocol_version": "G1.1",
        "formal_case_status": "ACCEPTANCE_VALID", "research_case_status": "RESEARCH_ONLY",
        "thresholds": {"detection_accuracy": 0.98, "localization_accuracy": 0.95, "classification_accuracy": 0.90, "reasoning_correctness": 0.90},
        "rules": {"macro_and_micro_required": True, "unknown_and_data_quality_included": True, "truth_blind_runtime": True},
    }
    rubric = {
        "schema_version": "astrograph-reasoning-rubric.v1", "protocol_version": "G1.1",
        "minimum_score": 0.90,
        "dimensions": [
            {"id": "evidence_grounding", "weight": 0.30}, {"id": "causal_consistency", "weight": 0.30},
            {"id": "uncertainty_calibration", "weight": 0.20}, {"id": "claim_boundary", "weight": 0.20},
        ],
        "independent_review_required": True,
    }
    schema_common = {
        "$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
        "additionalProperties": True, "required": ["schema_version", "protocol_version"],
        "properties": {"schema_version": {"type": "string"}, "protocol_version": {"type": "string"}},
    }
    coverage_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object", "additionalProperties": False,
        "required": ["schema_version", "protocol_version", "status", "expansion_policy", "rows"],
        "properties": {
            "schema_version": {"const": "astrograph-coverage-matrix.v1"}, "protocol_version": {"type": "string"},
            "status": {"type": "string"},
            "expansion_policy": {"type": "object", "required": ["required_case_types"], "additionalProperties": True},
            "rows": {"type": "array", "minItems": 93, "maxItems": 93, "items": {
                "type": "object", "additionalProperties": False,
                "required": ["scenario_id", "object_id", "scenario_type", "capability_id", "effect_ids", "support_status", "split_group", "required_case_types"],
                "properties": {
                    "scenario_id": {"type": "string"}, "object_id": {"type": "string"},
                    "scenario_type": {"enum": ["nominal", "fault", "degradation"]}, "capability_id": {"type": "string"},
                    "effect_ids": {"type": "array", "items": {"type": "string"}}, "support_status": {"const": "SUPPORTED"},
                    "split_group": {"enum": ["development", "validation", "holdout"]},
                    "required_case_types": {"type": "array", "items": {"type": "string"}},
                },
            }},
        },
    }
    files = {
        "diagnosis_scope.json": diagnosis_scope, "fault_taxonomy.json": taxonomy,
        "coverage_matrix.json": coverage, "metric_policy.json": metric_policy,
        "reasoning_rubric.json": rubric,
        "diagnosis_scope.schema.json": schema_common, "fault_taxonomy.schema.json": schema_common,
        "coverage_matrix.schema.json": coverage_schema, "metric_policy.schema.json": schema_common,
        "reasoning_rubric.schema.json": schema_common,
    }
    for name, value in files.items():
        _write(output / name, value)
    approvals = {
        "schema_version": "astrograph-scope-approvals.v1", "protocol_version": "G1.1",
        "project_owner": {"status": "APPROVED", "reviewer": "project_owner", "reviewed_at_utc": "2026-07-29T00:00:00Z"},
        "independent_evaluator": {"status": "PENDING", "reviewer": None, "reviewed_at_utc": None},
        "artifacts": {name: _sha(output / name) for name in sorted(files)},
    }
    _write(output / "approvals.json", approvals)
    return {"object_count": len(objects), "coverage_row_count": len(coverage_rows), "variant_count": 31 + len(taxonomy_rows), "status": "BLOCKED_APPROVAL"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--sat-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.root.resolve(), args.sat_root.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
