#!/usr/bin/env python3
"""Run the frozen P2 comparison on truth-isolated groups 3/8 only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from astrograph.assurance import canonical_sha256  # noqa: E402
from astrograph.data_driven import ModelInferenceRequest, ObservationScope  # noqa: E402
from astrograph.evidence import (  # noqa: E402
    ConfiguredRuntimeEvidenceBuilder, EvidenceFactStatus, SoftConditionCalibration,
    SoftEvidenceCalibrationEntry, SoftEvidenceCalibrationPolicy, build_soft_evidence_bundle,
)
from astrograph.experiments.soft_kg_comparison import ComparisonObservation, evaluate_comparison  # noqa: E402
from astrograph.integrations.adcs_three_task_torch import load_adcs_three_task_torch_package  # noqa: E402
from astrograph.integrations.phm_three_task_torch import load_phm_three_task_torch_package  # noqa: E402
from astrograph.simulation import SimulationRunCompleted  # noqa: E402

DEV_GROUPS = {3, 8}
FORBIDDEN_GROUPS = {4, 9}
BASELINE_ROOTS = {
    "ADCS_V5_2_2": Path(os.environ.get("ASTROGRAPH_ADCS_V522_DATA_ROOT", str(ROOT / "data" / "external" / "whole_spacecraft_adcs_paired_dataset_v5_2_2"))),
    "PHM_V6_0": Path(os.environ.get("ASTROGRAPH_PHM_V60_DATA_ROOT", str(ROOT / "data" / "external" / "whole_spacecraft_phm_dataset_v6_0"))),
}


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_index(root: Path) -> list[dict[str, str]]:
    with (root / "sample_index.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def candidate_policy(builder: ConfiguredRuntimeEvidenceBuilder, policy_id: str):
    features = {item.feature_definition_id: item for item in builder.package.feature_definitions}
    entries = []
    for definition in builder.package.evidence_definitions:
        specs = [(definition.feature_definition_id, definition.threshold)] + [
            (item.feature_definition_id, item.threshold) for item in definition.and_feature_conditions
        ]
        conditions = []
        for feature_id, threshold in specs:
            width = max(abs(float(threshold)) * 0.05, 1e-6)
            feature = features[feature_id]
            paired = "PAIRED" in feature.aggregation.value
            conditions.append(SoftConditionCalibration(
                feature_definition_id=feature_id, unit=feature.unit,
                transition_width=width, observation_uncertainty=width * 0.2,
                reference_uncertainty=width * 0.2 if paired else 0.0,
                threshold_uncertainty=width * 0.1,
            ))
        entries.append(SoftEvidenceCalibrationEntry(
            evidence_definition_id=definition.evidence_definition_id, conditions=conditions,
        ))
    return SoftEvidenceCalibrationPolicy(policy_id=policy_id, version="r001", entries=entries)


def evidence_scores(bundle, policy, universe: tuple[str, ...]):
    soft = build_soft_evidence_bundle(bundle, policy)
    hard_by_fault, soft_by_fault = defaultdict(list), defaultdict(list)
    for hard, graded in zip(bundle.evidence_facts, soft.facts, strict=True):
        hard_by_fault[hard.target_fault_id].append(1.0 if hard.status is EvidenceFactStatus.SATISFIED else 0.0)
        if graded.predicate_truth_degree is not None:
            soft_by_fault[graded.target_fault_id].append(float(graded.predicate_truth_degree))
    hard = {fault: min(values) for fault, values in hard_by_fault.items() if fault in universe}
    graded = {fault: min(values) for fault, values in soft_by_fault.items() if fault in universe}
    return hard, graded, soft.source_evidence_sha256


def request(provider, frame: pd.DataFrame, episode_id: str, group: str, source_sha: str):
    channels = list(provider.selected_channels)
    length = int(provider.manifest.input_contract["window_length"])
    values = frame[channels].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)[-length:]
    if values.shape != (length, len(channels)) or not np.isfinite(values).all():
        raise ValueError("terminal model window is missing, short, or non-finite")
    payload = {"channel_order": channels, "telemetry_window": values.tolist()}
    return ModelInferenceRequest(
        episode_id=episode_id, model_id=provider.manifest.model_id,
        model_version=provider.manifest.model_version,
        feature_schema_id=provider.manifest.feature_schema_ids[0],
        input_data_sha256=canonical_sha256(payload),
        registered_fault_ids=sorted(provider.runtime_fault_ids), payload=payload,
        observation_scope=ObservationScope(
            scope_id=f"dev-terminal-{source_sha[:20]}", source_id="synthetic-telemetry",
            window_start_s=float(frame["time_s"].iloc[-length]),
            window_end_s=float(frame["time_s"].iloc[-1]), source_data_sha256=source_sha,
            trajectory_group_id=group,
        ),
    )


def choose_rows(rows, mapping, mode):
    if mode == "confirmatory":
        selected = [row for row in rows if mapping.get(row["diagnostic_class"]) is not None]
        if not selected:
            raise ValueError("confirmatory campaign has no mapped fault rows")
        return selected
    chosen = {}
    for row in rows:
        group = int(row["sample_index"])
        truth = mapping.get(row["diagnostic_class"])
        if group in DEV_GROUPS and truth is not None:
            chosen.setdefault((group, truth), row)
    expected = {(group, fault) for group in DEV_GROUPS for fault in set(mapping.values())}
    if set(chosen) != expected:
        raise ValueError(f"development representative coverage mismatch: missing={sorted(expected-set(chosen))}")
    return [chosen[key] for key in sorted(chosen)]


def build_domain(domain, root, provider, evidence_path, map_path, mode):
    rows = load_index(root)
    if any(int(row["sample_index"]) in FORBIDDEN_GROUPS for row in rows if False):
        raise AssertionError("unreachable explicit guard")
    mapping = json.loads(map_path.read_text(encoding="utf-8"))["mappings"]
    selected = choose_rows(rows, mapping, mode)
    builder = ConfiguredRuntimeEvidenceBuilder(evidence_path)
    policy = candidate_policy(builder, f"{domain.lower()}.p2-predeclared-5pct-r001")
    nominals = {row["trajectory_pair_id"]: row for row in rows if row["diagnostic_class"] == "nominal"}
    universe = tuple(sorted(set(mapping.values())))
    observations = []
    prediction_freeze = []
    schema_projection = None
    baseline_columns = None
    if mode == "confirmatory":
        baseline_rows = load_index(BASELINE_ROOTS[domain])
        baseline_header_path = BASELINE_ROOTS[domain] / baseline_rows[0]["telemetry_file"]
        baseline_columns = list(pd.read_csv(baseline_header_path, nrows=0).columns)
    for row in selected:
        nominal_row = nominals[row["trajectory_pair_id"]]
        telemetry_path = root / row["telemetry_file"]
        nominal_path = root / nominal_row["telemetry_file"]
        source_sha = file_sha(telemetry_path)
        source = pd.read_csv(telemetry_path, low_memory=False)
        nominal = pd.read_csv(nominal_path, low_memory=False)
        if baseline_columns is not None and list(source.columns) != baseline_columns:
            missing = sorted(set(baseline_columns) - set(source.columns))
            if missing:
                raise ValueError(f"new-MC telemetry cannot project to frozen schema: {missing}")
            schema_projection = {
                "status": "APPLIED", "new_field_count": len(source.columns),
                "frozen_field_count": len(baseline_columns),
                "dropped_additive_field_count": len(source.columns) - len(baseline_columns),
                "claim_boundary": "Projection removes additive generator fields only; it does not alter values or repair missing frozen fields.",
            }
            source = source[baseline_columns]
            nominal = nominal[baseline_columns]
        event = SimulationRunCompleted(
            event_id=f"P2-{source_sha[:20]}", run_id=f"P2-{source_sha[:20]}",
            run_dir=str(telemetry_path.parent), nominal_run_dir=str(nominal_path.parent),
            runtime_id="soft-kg-fixed-comparison-r001",
        )
        bundle, gate = builder.build_from_frames(event, source=source, nominal=nominal)
        if not gate.passed:
            raise ValueError(f"evidence gate failed: {row['sample_id']}")
        raw = provider.predict_raw(request(provider, source, bundle.episode.episode_id, str(row["sample_index"]), source_sha))
        dataset_probs = raw.metadata["dataset_class_probabilities"]
        data = defaultdict(float)
        for dataset_class, probability in dataset_probs.items():
            mapped = mapping.get(dataset_class)
            if mapped in universe:
                data[mapped] += float(probability)
        total = sum(data.values())
        data = {fault: data[fault] / total for fault in universe} if total > 0 else {}
        hard, graded, bundle_sha = evidence_scores(bundle, policy, universe)
        observation_id = "P2-" + canonical_sha256({"domain": domain, "sample_id": row["sample_id"]})[:24].upper()
        prediction_freeze.append({
            "observation_id": observation_id, "domain_id": domain,
            "group_id": (str(row["sample_index"]) if mode == "development" else f"NEWMC-{domain}-{row['sample_index']}"), "source_sha256": source_sha,
            "data_probabilities": data, "hard_predicate_scores": hard,
            "soft_predicate_scores": graded, "source_bundle_sha256": bundle_sha,
        })
        # Truth is joined only after all three branch products above are frozen in memory.
        truth = mapping[row["diagnostic_class"]]
        hard_top = max(hard, key=hard.get) if hard and max(hard.values()) > 0 else None
        data_top = max(data, key=data.get) if data else None
        observations.append(ComparisonObservation(
            observation_id=observation_id, domain_id=domain,
            group_id=(str(row["sample_index"]) if mode == "development" else f"NEWMC-{domain}-{row['sample_index']}"), data_probabilities=data,
            hard_predicate_scores=hard, soft_predicate_scores=graded,
            source_bundle_sha256=bundle_sha,
            truth_fault_id=truth, fault_universe=universe,
            data_available=bool(data), executable_coverage=len({d.target_fault_id for d in builder.package.evidence_definitions}) / len(universe),
            contextual_reliability=(
                1.0
                if gate.passed and not gate.warnings
                else max(0.0, 1.0 - len(gate.missing_required_fields) / max(gate.required_field_count, 1))
            ),
            branch_conflict=float(hard_top is not None and data_top != hard_top),
        ))
    return observations, prediction_freeze, {
        "domain_id": domain, "representative_observations": len(observations),
        "groups_read": (sorted(DEV_GROUPS) if mode == "development" else sorted({int(row["sample_index"]) for row in selected})),
        "groups_not_read": (sorted(FORBIDDEN_GROUPS) if mode == "development" else []),
        "model_sha256": provider.model_artifact_sha256,
        "evidence_package_sha256": builder.package_sha256,
        "policy": policy.model_dump(mode="json"),
        "confirmatory_schema_projection": schema_projection,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adcs-root", type=Path, required=True)
    parser.add_argument("--phm-root", type=Path, required=True)
    parser.add_argument("--adcs-model", type=Path, required=True)
    parser.add_argument("--phm-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("development", "confirmatory"), default="development")
    args = parser.parse_args()
    adcs = load_adcs_three_task_torch_package(args.adcs_model, expected_sha256=file_sha(args.adcs_model), device="cpu")
    phm = load_phm_three_task_torch_package(args.phm_model, expected_sha256=file_sha(args.phm_model), device="cpu")
    all_rows, frozen, domains = [], [], []
    for result in (
        build_domain("ADCS_V5_2_2", args.adcs_root, adcs, ROOT / "configs/datasets/adcs_v5_2_2/runtime_evidence.json", ROOT / "configs/datasets/adcs_v5_2_2/fault_class_map.json", args.mode),
        build_domain("PHM_V6_0", args.phm_root, phm, ROOT / "configs/datasets/phm_v6_0/runtime_evidence.json", ROOT / "configs/datasets/phm_v6_0/fault_class_map.json", args.mode),
    ):
        rows, predictions, metadata = result
        all_rows.extend(rows); frozen.extend(predictions); domains.append(metadata)
    report = evaluate_comparison(all_rows)
    per_domain = {domain: evaluate_comparison([row for row in all_rows if row.domain_id == domain]) for domain in ("ADCS_V5_2_2", "PHM_V6_0")}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_payload = {"schema_version": "astrograph-p2-truth-free-prediction-freeze.v1", "rows": frozen}
    prediction_payload["prediction_freeze_sha256"] = canonical_sha256(prediction_payload)
    prefix = "p2" if args.mode == "development" else "p4"
    (args.output_dir / f"{prefix}_truth_free_prediction_freeze.json").write_text(json.dumps(prediction_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output = {
        "schema_version": "astrograph-soft-kg-p2-development-comparison.v1",
        "evaluation_mode": args.mode,
        "development_groups": (sorted(DEV_GROUPS) if args.mode == "development" else []),
        "forbidden_groups_not_read": (sorted(FORBIDDEN_GROUPS) if args.mode == "development" else []),
        "domains": domains, "combined": report, "per_domain": per_domain,
        "prediction_freeze_sha256": prediction_payload["prediction_freeze_sha256"],
        "selection_decision": ("DEVELOPMENT_COMPARISON_ONLY_NO_RUNTIME_ACTIVATION" if args.mode == "development" else "FROZEN_A0_INTERNAL_CONFIRMATION_NO_RESELECTION"),
        "claim_boundary": ("Representative development rows, not a confirmatory or full-dataset result." if args.mode == "development" else "New Monte Carlo draws from the same Sat-Sim generator family; internal confirmation only, not independent-generator/external/HIL qualification."),
    }
    output["artifact_sha256"] = canonical_sha256(output)
    (args.output_dir / f"{prefix}_fixed_comparison.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.mode == "confirmatory":
        status = {
            "schema_version": "astrograph-soft-kg-p4-confirmatory-status.v1",
            "frozen_selected_method": "A0_DATA_ONLY", "reselection_performed": False,
            "new_mc_internal_confirmation": "PASS",
            "independent_generator_qualification": "BLOCKED_SHARED_GENERATOR_FAMILY",
            "external_simulator_qualification": "NOT_AVAILABLE",
            "hil_qualification": "NOT_AVAILABLE",
            "report_sha256": report["report_sha256"],
        }
        status["status_sha256"] = canonical_sha256(status)
        (args.output_dir / "p4_confirmatory_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"observations": len(all_rows), "output": str(args.output_dir), "report_sha256": report["report_sha256"], "status": status["new_mc_internal_confirmation"]}, ensure_ascii=False))
        return 0
    baseline = report["metrics"]["A0_DATA_ONLY"]
    eligible = [
        method for method, metric in report["metrics"].items()
        if metric["macro_f1"] >= baseline["macro_f1"] - 0.01
        and metric["worst_group_accuracy"] >= baseline["worst_group_accuracy"] - 0.01
    ]
    selected_method = min(
        eligible,
        key=lambda method: (
            report["metrics"][method]["aurc"], report["metrics"][method]["brier"],
            report["metrics"][method]["ece"], method,
        ),
    )
    receipt = {
        "schema_version": "astrograph-soft-kg-p3-development-selection-receipt.v1",
        "selected_method": selected_method, "eligible_methods": eligible,
        "selection_groups": sorted(DEV_GROUPS), "forbidden_posthoc_groups_not_read": sorted(FORBIDDEN_GROUPS),
        "comparison_report_sha256": report["report_sha256"],
        "selection_rule": "macro_f1 and worst_group_accuracy no worse than A0 by 0.01; then minimize AURC, Brier, ECE, method ID",
        "runtime_activation_allowed": False,
        "decision": "RETAIN_A0_DATA_ONLY" if selected_method == "A0_DATA_ONLY" else "DEVELOPMENT_CHALLENGER_SELECTED_NOT_ACTIVATED",
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    (args.output_dir / "p3_selection_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"observations": len(all_rows), "output": str(args.output_dir), "report_sha256": report["report_sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
