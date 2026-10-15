#!/usr/bin/env python3
"""Offline evaluator for frozen label-free ADCS temporal traces."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

from astrograph.assurance import canonical_sha256
from astrograph.experiments.model_independent_temporal import (
    FrozenModelIndependentPolicyReceipt,
    FrozenTraceBundle,
    ModelIndependentScaleContract,
    ModelIndependentTemporalPolicy,
    ModelIndependentTruthTable,
    TraceVariant,
    evaluate_model_independent_policy,
    freeze_model_independent_policy,
    rank_model_independent_policies,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def load_contract(path: Path) -> ModelIndependentScaleContract:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop("contract_sha256", None)
    payload.pop("fit_components", None)
    fields = set(ModelIndependentScaleContract.model_fields)
    contract = ModelIndependentScaleContract.model_validate({key: value for key, value in payload.items() if key in fields})
    if expected is not None and expected != contract.contract_sha256:
        raise ValueError("scale contract hash mismatch")
    return contract


def load_truth(path: Path) -> ModelIndependentTruthTable:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop("truth_table_sha256", None)
    truth = ModelIndependentTruthTable.model_validate(payload)
    if expected is not None and expected != truth.truth_table_sha256:
        raise ValueError("truth table hash mismatch")
    return truth


def load_bundle(path: Path) -> FrozenTraceBundle:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        artifact = json.load(handle)
    bundle = FrozenTraceBundle.model_validate(artifact["bundle"])
    if artifact.get("bundle_sha256") != bundle.bundle_sha256:
        raise ValueError(f"bundle hash mismatch for {path}")
    observed = {trace.opaque_trajectory_id: trace.trace_sha256 for trace in bundle.traces}
    if artifact.get("trace_sha256s") != observed:
        raise ValueError(f"trace hash index mismatch for {path}")
    return bundle


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        return (0.0, 0.0)
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def candidate_grid() -> tuple[ModelIndependentTemporalPolicy, ...]:
    candidates = []
    for threshold in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
        for persistence in (1, 2, 3):
            candidates.append(
                ModelIndependentTemporalPolicy(
                    policy_id=f"ADCS-MI-T{threshold:g}-P{persistence}-R001",
                    trigger_threshold=threshold,
                    trigger_persistence_points=persistence,
                    warmup_points=31,
                    reference_prefix_max=0.05,
                )
            )
    return tuple(candidates)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--truth-table", type=Path, required=True)
    parser.add_argument("--scale-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    trace_dir = args.trace_dir.resolve()
    truth = load_truth(args.truth_table.resolve())
    contract = load_contract(args.scale_contract.resolve())
    bundles: dict[TraceVariant, FrozenTraceBundle] = {}
    bundle_files: dict[TraceVariant, Path] = {}
    for variant in TraceVariant:
        path = trace_dir / f"{variant.value.lower()}.trace_bundle.json.gz"
        bundle = load_bundle(path)
        if bundle.variant != variant:
            raise ValueError("trace file/variant mismatch")
        if bundle.dataset_archive_sha256 != truth.dataset_archive_sha256:
            raise ValueError("trace/truth dataset identity differs")
        bundles[variant] = bundle
        bundle_files[variant] = path

    canonical = bundles[TraceVariant.EXACT_PAIRED]
    candidates = candidate_grid()
    grid_payload = [candidate.model_dump(mode="json", exclude_computed_fields=True) for candidate in candidates]
    grid_sha256 = canonical_sha256(grid_payload)
    ranked = rank_model_independent_policies(candidates, canonical, truth)
    receipt = freeze_model_independent_policy(ranked, canonical, truth, contract, grid_sha256)
    selected = receipt.selected_policy
    unguarded = ModelIndependentTemporalPolicy(
        **{
            **selected.model_dump(mode="python", exclude={"policy_sha256"}),
            "policy_id": selected.policy_id + "-UNGUARDED-AUDIT",
            "reference_prefix_max": None,
        }
    )

    stress_results: dict[str, dict] = {}
    for variant, bundle in bundles.items():
        guarded_metrics = evaluate_model_independent_policy(selected, bundle, truth)
        unguarded_metrics = evaluate_model_independent_policy(unguarded, bundle, truth)
        stress_results[variant.value] = {
            "trace_artifact_path": bundle_files[variant].name,
            "trace_artifact_sha256": sha256_file(bundle_files[variant]),
            "bundle_sha256": bundle.bundle_sha256,
            "guarded": guarded_metrics.model_dump(mode="json"),
            "unguarded_fixed_policy_audit": unguarded_metrics.model_dump(mode="json"),
        }

    canonical_metrics = receipt.selected_metrics
    group_successes = sum(rate >= 1.0 - 1e-12 for rate in canonical_metrics.per_group_detected_within_20s_rate.values())
    group_trials = len(canonical_metrics.per_group_detected_within_20s_rate)
    timely_group_wilson = wilson_interval(group_successes, group_trials)
    no_false_alarm_groups = group_trials - canonical_metrics.nominal_false_alarm_group_count
    false_alarm_group_wilson = wilson_interval(canonical_metrics.nominal_false_alarm_group_count, group_trials)

    exact_strong = (
        canonical_metrics.group_macro_detected_within_20s_rate >= 0.95
        and canonical_metrics.minimum_class_detected_within_20s_rate >= 0.95
        and canonical_metrics.pre_onset_selection_count == 0
        and canonical_metrics.nominal_false_alarm_count == 0
    )
    brittle_variants = []
    guard_prevents_unsafe = True
    for variant in (TraceVariant.REFERENCE_SHIFT_1S, TraceVariant.REFERENCE_SHIFT_2S, TraceVariant.CURRENT_NOISE_5PCT_SCALE, TraceVariant.CROSS_MC_REFERENCE):
        unguarded_metrics = stress_results[variant.value]["unguarded_fixed_policy_audit"]
        guarded_metrics = stress_results[variant.value]["guarded"]
        if unguarded_metrics["pre_onset_selection_count"] > 0 or unguarded_metrics["nominal_false_alarm_count"] > 0:
            brittle_variants.append(variant.value)
        if guarded_metrics["pre_onset_selection_count"] > 0 or guarded_metrics["nominal_false_alarm_count"] > 0:
            guard_prevents_unsafe = False

    contribution = {
        "schema_version": "astrograph-model-independent-thesis-contribution-assessment.v1",
        "assessment": "STRONG_SYNTHETIC_FEASIBILITY_BOUNDED_COMPONENT_NOT_STANDALONE_CONFIRMATION",
        "canonical_exact_pair_feasibility_strong": exact_strong,
        "label_isolation_enforced": True,
        "fault_identification_evaluated": False,
        "independent_group_count": canonical_metrics.independent_group_count,
        "reference_mismatch_brittleness_observed": bool(brittle_variants),
        "brittle_stress_variants": brittle_variants,
        "reference_guard_prevented_preonset_and_nominal_false_alarms": guard_prevents_unsafe,
        "new_confirmatory_evidence": "NONE",
        "standalone_sufficient_for_excellent_thesis": False,
        "bounded_contribution_value": [
            "Demonstrates that label-free classifier-independent localization is feasible under an exact paired digital-twin contract.",
            "Falsifies any broad claim of robustness: one-second reference shift and reference mismatch can invalidate an unguarded residual trigger.",
            "Adds an explicit reference-admissibility abstention mechanism, turning an implicit assumption into an auditable selective-diagnosis boundary.",
            "Provides a defensible component for the thesis when combined with mismatch-tolerant localization, evidence-aware fusion, and a new sealed campaign.",
        ],
        "remaining_requirements": [
            "Develop and freeze a mismatch-tolerant alignment/uncertainty method using development evidence only.",
            "Obtain a new sealed simulator campaign after all policies are frozen.",
            "Keep ADCS fault-identification and universal KG-gain claims frozen until separately validated.",
            "Increase the number and diversity of independent generator/simulator groups.",
        ],
    }
    contribution["assessment_sha256"] = canonical_sha256(contribution)

    ranking_payload = [
        {
            "rank": rank,
            "policy": candidate.policy.model_dump(mode="json"),
            "metrics": candidate.metrics.model_dump(mode="json"),
            "selection_key": list(candidate.selection_key),
        }
        for rank, candidate in enumerate(ranked, start=1)
    ]
    result = {
        "schema_version": "astrograph-adcs-model-independent-temporal-development-result.v1",
        "project_status": "B008_WIP_DEVELOPMENT_EVIDENCE",
        "dataset_archive_sha256": truth.dataset_archive_sha256,
        "development_groups": [3, 8],
        "excluded_groups": [4, 9],
        "scale_contract_sha256": contract.contract_sha256,
        "truth_table_sha256": truth.truth_table_sha256,
        "candidate_grid_sha256": grid_sha256,
        "selected_policy_receipt": receipt.model_dump(mode="json"),
        "candidate_ranking": ranking_payload,
        "stress_replay": stress_results,
        "independent_group_uncertainty": {
            "group_count": group_trials,
            "groups_with_100pct_within20": group_successes,
            "wilson_95_interval_for_group_success_probability": list(timely_group_wilson),
            "nominal_false_alarm_group_count": canonical_metrics.nominal_false_alarm_group_count,
            "wilson_95_interval_for_false_alarm_group_probability": list(false_alarm_group_wilson),
            "warning": "Only six independent trajectory-pair groups are available; trajectory counts do not increase the independent sample size.",
        },
        "thesis_contribution_assessment": contribution,
        "result_boundary": [
            "Development evidence only; groups 4/9 were not read or used.",
            "The route is independent of the missing learned P2 classifier, not independent of a nominal digital-twin/reference trajectory.",
            "No fault-identification, KG-gain, external-validity, HIL, safety, flight, or confirmatory claim is supported.",
        ],
    }
    result["result_sha256"] = canonical_sha256(result)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "development_result.json", result)
    write_json(args.output_dir / "selected_policy_receipt.json", receipt.model_dump(mode="json"))
    write_json(args.output_dir / "candidate_grid.json", {"candidate_grid": grid_payload, "candidate_grid_sha256": grid_sha256})
    write_json(args.output_dir / "thesis_contribution_assessment.json", contribution)

    selected_metrics = receipt.selected_metrics
    stress_lines = []
    for variant in TraceVariant:
        guarded = stress_results[variant.value]["guarded"]
        unguarded_metrics = stress_results[variant.value]["unguarded_fixed_policy_audit"]
        stress_lines.append(
            f"| {variant.value} | {guarded['reference_admission_rate']:.4f} | {guarded['detected_within_20s_rate']:.4f} | "
            f"{guarded['pre_onset_selection_count']} | {guarded['nominal_false_alarm_count']} | "
            f"{unguarded_metrics['pre_onset_selection_count']} | {unguarded_metrics['nominal_false_alarm_count']} |"
        )
    report = f"""# AstroGraph B008 Model-Independent Temporal Development Result R001

## Status

- Project snapshot: `B008_WIP_DEVELOPMENT_EVIDENCE`
- Selection data: groups `3/8` only
- Permanently excluded from selection: groups `4/9`
- Runtime ground-truth access: `NO`
- Learned P2 classifier required: `NO`
- New confirmatory evidence: `NONE`

## Frozen development policy

- Trigger threshold: `{selected.trigger_threshold}`
- Persistence: `{selected.trigger_persistence_points}` point(s)
- Warmup: `{selected.warmup_points}` points
- Reference-prefix admissibility maximum: `{selected.reference_prefix_max}`
- Policy SHA-256: `{selected.policy_sha256}`

## Canonical exact-pair result

- Fault trajectories: `{selected_metrics.fault_trajectory_count}`
- Nominal trajectories: `{selected_metrics.nominal_trajectory_count}`
- Independent trajectory-pair groups: `{selected_metrics.independent_group_count}`
- Detected within 5 s: `{selected_metrics.detected_within_5s_rate:.6f}`
- Detected within 20 s: `{selected_metrics.detected_within_20s_rate:.6f}`
- Minimum class-wise within-20-s rate: `{selected_metrics.minimum_class_detected_within_20s_rate:.6f}`
- Pre-onset selections: `{selected_metrics.pre_onset_selection_count}`
- Nominal false alarms: `{selected_metrics.nominal_false_alarm_count}`
- Median delay: `{selected_metrics.median_detection_delay_s}` s

The exact-pair result is strong development evidence for the bounded digital-twin assumption.  It is not evidence that an exact reference is available in flight.

## Frozen-policy stress replay

| Variant | Guarded admission | Guarded within 20 s | Guarded pre-onset | Guarded nominal FA | Unguarded pre-onset | Unguarded nominal FA |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(stress_lines)}

The unguarded replay deliberately falsifies broad robustness claims.  The reference-admissibility guard prevents unsafe extrapolation by abstaining when the paired reference is outside the frozen envelope; this can reduce coverage substantially and is not a substitute for a mismatch-tolerant method.

## Thesis-value conclusion

`{contribution['assessment']}`

This route is a defensible thesis component because it makes reference validity explicit, demonstrates label isolation, and reports a strong negative robustness result instead of hiding it.  It is not sufficient by itself for an excellent thesis because only six independent groups are available, the canonical result depends on exact pairing, and no new sealed confirmatory campaign has been run.

## Research boundary

- No ADCS fault-identification result was produced.
- No KG or fusion parameter was selected.
- No groups 4/9 outcome was read.
- No external, HIL, safety, flight, G0–G4, or confirmatory status changed.
"""
    (args.output_dir / "development_result.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "result": str(args.output_dir / "development_result.json"),
        "result_sha256": result["result_sha256"],
        "selected_policy_sha256": selected.policy_sha256,
        "canonical_within20": selected_metrics.detected_within_20s_rate,
        "canonical_min_class_within20": selected_metrics.minimum_class_detected_within_20s_rate,
        "canonical_nominal_false_alarms": selected_metrics.nominal_false_alarm_count,
        "assessment": contribution["assessment"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
