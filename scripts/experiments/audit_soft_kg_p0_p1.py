#!/usr/bin/env python3
"""Freeze P0 semantics and exercise P1 hard/soft same-source perturbations."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from astrograph.evidence import (  # noqa: E402
    EpisodeEvidenceBundle, EpisodeWindow, EvidenceDefinition, FeatureDefinition,
    FeatureResult, SoftConditionCalibration, SoftEvidenceCalibrationEntry,
    SoftEvidenceCalibrationPolicy, build_soft_evidence_bundle, evaluate_evidence,
)

PACKAGES = {
    "ADCS_V5_2_2": ROOT / "configs/datasets/adcs_v5_2_2/runtime_evidence.json",
    "PHM_V6_0": ROOT / "configs/datasets/phm_v6_0/runtime_evidence.json",
}
OUTPUT = ROOT / "research/evidence/soft_kg_p0_p4_r001"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write(name: str, payload: dict) -> None:
    payload["artifact_sha256"] = canonical(payload)
    (OUTPUT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def bundle(domain: str, features: dict[str, FeatureDefinition], definition: EvidenceDefinition, value: float | None):
    feature = features[definition.feature_definition_id]
    result = FeatureResult(
        feature_result_id=f"FR-{definition.evidence_definition_id}-{value}",
        feature_definition_id=feature.feature_definition_id, run_id="P1-RUN",
        paired_nominal_run_id="P1-REFERENCE", episode_id="P1-EPISODE",
        resolved_field=feature.source_field_template, component_instance="P1-COMPONENT",
        value=value, unit=feature.unit, window_start_s=0.0, window_end_s=1.0,
        sample_count=2, algorithm_version=feature.algorithm_version,
        input_data_sha256=canonical({"definition": definition.evidence_definition_id, "value": value}),
    )
    results = [result]
    for conjunct in definition.and_feature_conditions:
        conjunct_feature = features[conjunct.feature_definition_id]
        direction = -1.0 if conjunct.comparison.value.startswith("LESS") else 1.0
        results.append(FeatureResult(
            feature_result_id=f"FR-{definition.evidence_definition_id}-{conjunct.feature_definition_id}",
            feature_definition_id=conjunct.feature_definition_id, run_id="P1-RUN",
            paired_nominal_run_id="P1-REFERENCE", episode_id="P1-EPISODE",
            resolved_field=conjunct_feature.source_field_template, component_instance="P1-COMPONENT",
            value=float(conjunct.threshold) + direction * max(abs(float(conjunct.threshold)) * 0.1, 1e-6),
            unit=conjunct_feature.unit, window_start_s=0, window_end_s=1, sample_count=2,
            algorithm_version=conjunct_feature.algorithm_version,
            input_data_sha256=canonical({"conjunct": conjunct.feature_definition_id}),
        ))
    fact = evaluate_evidence(definition, result, all_feature_results=results)
    return EpisodeEvidenceBundle(
        episode=EpisodeWindow(episode_id="P1-EPISODE", run_id="P1-RUN", paired_nominal_run_id="P1-REFERENCE", start_s=0, end_s=1, source="P1_FEATURE_LEVEL_PERTURBATION"),
        domain_id=domain, analysis_component_instance="P1-COMPONENT",
        feature_definitions=[features[item.feature_definition_id] for item in results], feature_results=results,
        evidence_definitions=[definition], evidence_facts=[fact],
        evaluation_metadata={"labels_used": False, "same_feature_result_for_hard_and_soft": True},
    )


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    p0_rows, p1_rows = [], []
    for domain, path in PACKAGES.items():
        raw = json.loads(path.read_text(encoding="utf-8"))
        features = {item["feature_definition_id"]: FeatureDefinition.model_validate(item) for item in raw["feature_definitions"]}
        for item in raw["evidence_definitions"]:
            definition = EvidenceDefinition.model_validate(item)
            feature = features[definition.feature_definition_id]
            paired = "PAIRED" in feature.aggregation.value
            p0_rows.append({
                "domain_id": domain, "target_fault_id": definition.target_fault_id,
                "evidence_definition_id": definition.evidence_definition_id,
                "feature_definition_id": feature.feature_definition_id,
                "source_field_template": feature.source_field_template,
                "aggregation": feature.aggregation.value, "window_scope": feature.window_scope,
                "unit": feature.unit, "threshold": definition.threshold,
                "comparison": definition.comparison.value,
                "reference_generation": "MATCHED_NOMINAL_PAIRED_RESIDUAL" if paired else "TELEMETRY_NATIVE_RESIDUAL_OR_STATUS",
                "applicable_condition_ids": definition.applicable_condition_ids,
                "threshold_source_refs": definition.source_refs,
                "feature_source_refs": feature.source_refs,
                "development_groups": [3, 8], "forbidden_posthoc_groups": [4, 9],
                "evidence_level": definition.evidence_level,
            })
            width = max(abs(float(definition.threshold)) * 0.05, 1e-6)
            conditions = [SoftConditionCalibration(
                feature_definition_id=feature.feature_definition_id,
                unit=feature.unit, transition_width=width,
                observation_uncertainty=width * 0.2,
                reference_uncertainty=width * 0.2 if paired else 0.0,
                threshold_uncertainty=width * 0.1,
            )]
            for conjunct in definition.and_feature_conditions:
                conjunct_feature = features[conjunct.feature_definition_id]
                conjunct_width = max(abs(float(conjunct.threshold)) * 0.05, 1e-6)
                conditions.append(SoftConditionCalibration(
                    feature_definition_id=conjunct.feature_definition_id,
                    unit=conjunct_feature.unit, transition_width=conjunct_width,
                    observation_uncertainty=conjunct_width * 0.2,
                    threshold_uncertainty=conjunct_width * 0.1,
                ))
            policy = SoftEvidenceCalibrationPolicy(
                policy_id="p1-perturbation-only", version="r001",
                entries=[SoftEvidenceCalibrationEntry(
                    evidence_definition_id=definition.evidence_definition_id,
                    conditions=conditions,
                )],
            )
            direction = -1.0 if definition.comparison.value.startswith("LESS") else 1.0
            scenarios = {
                "boundary_below": float(definition.threshold) - direction * width,
                "boundary_exact": float(definition.threshold),
                "boundary_above": float(definition.threshold) + direction * width,
                "noise_negative_half_width": float(definition.threshold) - direction * width * 0.5,
                "noise_positive_half_width": float(definition.threshold) + direction * width * 0.5,
                ("reference_mismatch_one_width" if paired else "telemetry_bias_one_width"):
                    float(definition.threshold) - direction * width,
                "missing": None,
            }
            for scenario, value in scenarios.items():
                source = bundle(domain, features, definition, value)
                hard = source.evidence_facts[0]
                soft = build_soft_evidence_bundle(source, policy).facts[0]
                p1_rows.append({
                    "domain_id": domain, "evidence_definition_id": definition.evidence_definition_id,
                    "scenario": scenario, "feature_result_id": hard.feature_result_id,
                    "observed_value": value, "hard_status": hard.status.value,
                    "soft_truth_degree": soft.predicate_truth_degree,
                    "soft_truth_interval": [soft.predicate_truth_lower, soft.predicate_truth_upper],
                    "same_source_feature_result": soft.evidence_fact_id == hard.evidence_fact_id,
                })
    faults = sorted({(row["domain_id"], row["target_fault_id"]) for row in p0_rows})
    write("p0_semantic_reference_audit.json", {
        "schema_version": "astrograph-p0-semantic-reference-audit.v1",
        "package_hashes": {domain: sha(path) for domain, path in PACKAGES.items()},
        "fault_count_by_domain": {domain: len({f for d, f in faults if d == domain}) for domain in PACKAGES},
        "rows": p0_rows,
        "claim_boundary": "E0 synthetic engineering semantics; no flight-certified threshold claim.",
    })
    write("p1_same_source_perturbation_audit.json", {
        "schema_version": "astrograph-p1-same-source-perturbation-audit.v1",
        "row_count": len(p1_rows), "rows": p1_rows,
        "acceptance": {
            "all_same_source": all(row["same_source_feature_result"] for row in p1_rows),
            "missing_fail_closed": all(row["hard_status"] == "MISSING_DATA" and row["soft_truth_degree"] is None for row in p1_rows if row["scenario"] == "missing"),
        },
        "claim_boundary": "FeatureResult-level sensitivity audit; raw sensor-noise and independent-reference uncertainty remain P4 evidence requirements.",
    })
    print(json.dumps({"output": str(OUTPUT), "p0_rows": len(p0_rows), "p1_rows": len(p1_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
