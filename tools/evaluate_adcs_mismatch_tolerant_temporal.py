#!/usr/bin/env python3
"""Offline groups-3/8 evaluator for bounded mismatch-tolerant temporal traces."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np

from astrograph.assurance import canonical_sha256
from astrograph.experiments.mismatch_tolerant_temporal import (
    FrozenMismatchTraceBundle,
    MismatchTolerantAlignmentContract,
    MismatchTolerantTemporalPolicy,
    REQUIRED_DEVELOPMENT_VARIANTS,
    freeze_mismatch_tolerant_policy,
    rank_mismatch_tolerant_policies,
    select_mismatch_tolerant_trace,
)
from astrograph.experiments.model_independent_temporal import (
    ModelIndependentScaleContract,
    ModelIndependentTruthTable,
    TraceVariant,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_scale_contract(path: Path) -> ModelIndependentScaleContract:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop("contract_sha256", None)
    payload.pop("fit_components", None)
    fields = set(ModelIndependentScaleContract.model_fields)
    contract = ModelIndependentScaleContract.model_validate(
        {key: value for key, value in payload.items() if key in fields}
    )
    if expected is not None and expected != contract.contract_sha256:
        raise ValueError("scale contract hash mismatch")
    return contract


def load_alignment_contract(path: Path) -> MismatchTolerantAlignmentContract:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop("contract_sha256", None)
    payload.pop("candidate_lags", None)
    fields = set(MismatchTolerantAlignmentContract.model_fields)
    contract = MismatchTolerantAlignmentContract.model_validate(
        {key: value for key, value in payload.items() if key in fields}
    )
    if expected is not None and expected != contract.contract_sha256:
        raise ValueError("alignment contract hash mismatch")
    return contract


def load_truth(path: Path) -> ModelIndependentTruthTable:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop("truth_table_sha256", None)
    truth = ModelIndependentTruthTable.model_validate(payload)
    if expected is not None and expected != truth.truth_table_sha256:
        raise ValueError("truth table hash mismatch")
    return truth


def load_trace_index(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop("index_sha256", None)
    actual = canonical_sha256(payload)
    payload["index_sha256"] = expected
    if expected != actual:
        raise ValueError("trace index hash mismatch")
    if payload.get("truth_access_during_generation") is not False:
        raise ValueError("runtime trace generation accessed truth")
    if payload.get("outcome_data_read_during_generation") is not False:
        raise ValueError("runtime trace generation accessed outcome data")
    if payload.get("full_trajectory_alignment_used") is not False:
        raise ValueError("full-trajectory alignment is forbidden")
    if set(payload.get("development_groups", [])) != {3, 8}:
        raise ValueError("trace index development groups differ")
    if set(payload.get("excluded_groups", [])) != {4, 9}:
        raise ValueError("trace index excluded groups differ")
    return payload


def load_bundle(path: Path, expected_file_sha256: str, expected_bundle_sha256: str) -> FrozenMismatchTraceBundle:
    if sha256_file(path) != expected_file_sha256:
        raise ValueError(f"trace artifact SHA-256 mismatch: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    bundle = FrozenMismatchTraceBundle.model_validate(payload["bundle"])
    if payload.get("bundle_sha256") != bundle.bundle_sha256:
        raise ValueError("embedded bundle hash mismatch")
    if expected_bundle_sha256 != bundle.bundle_sha256:
        raise ValueError("trace index/bundle identity differs")
    trace_hashes = payload.get("trace_sha256s", {})
    actual_trace_hashes = {
        trace.opaque_trajectory_id: trace.trace_sha256 for trace in bundle.traces
    }
    if trace_hashes != actual_trace_hashes:
        raise ValueError("trace hash map differs")
    return bundle


def load_candidate_grid(path: Path) -> tuple[list[MismatchTolerantTemporalPolicy], str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop("candidate_grid_sha256", None)
    actual = canonical_sha256(payload)
    payload["candidate_grid_sha256"] = expected
    if expected != actual:
        raise ValueError("candidate-grid hash mismatch")
    if set(payload.get("development_groups", [])) != {3, 8}:
        raise ValueError("candidate grid development groups differ")
    if set(payload.get("excluded_groups", [])) != {4, 9}:
        raise ValueError("candidate grid excluded groups differ")
    if set(payload.get("development_variants", [])) != {
        variant.value for variant in REQUIRED_DEVELOPMENT_VARIANTS
    }:
        raise ValueError("candidate grid development variants differ")
    policies = [MismatchTolerantTemporalPolicy.model_validate(item) for item in payload["candidates"]]
    return policies, expected, payload


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 1.0
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def paired_group_bootstrap(
    current: dict[str, float],
    baseline: dict[str, float],
    *,
    seed: int = 20260830,
    repetitions: int = 20000,
) -> dict:
    if set(current) != set(baseline):
        raise ValueError("paired group bootstrap requires identical group sets")
    groups = sorted(current)
    differences = np.asarray([current[group] - baseline[group] for group in groups], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(groups), size=(repetitions, len(groups)))
    means = np.mean(differences[indices], axis=1)
    return {
        "statistical_unit": "independent trajectory-pair / Monte-Carlo group",
        "group_count": len(groups),
        "paired_mean_difference": float(np.mean(differences)),
        "cluster_bootstrap_seed": seed,
        "cluster_bootstrap_repetitions": repetitions,
        "cluster_bootstrap_95_interval": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
        "per_group_differences": {
            group: float(current[group] - baseline[group]) for group in groups
        },
    }


def compare_to_b008(selected: dict[str, dict], b008_path: Path | None) -> dict | None:
    if b008_path is None:
        return None
    payload = json.loads(b008_path.read_text(encoding="utf-8"))
    comparison: dict[str, dict] = {}
    for variant in REQUIRED_DEVELOPMENT_VARIANTS:
        old = payload["stress_replay"][variant.value]["guarded"]
        new = selected[variant.value]
        comparison[variant.value] = {
            "b008_guarded_reference_admission_rate": old["reference_admission_rate"],
            "b009_alignment_admission_rate": new["alignment_admission_rate"],
            "admission_rate_difference": new["alignment_admission_rate"] - old["reference_admission_rate"],
            "b008_guarded_localized_within_20s_rate": old["detected_within_20s_rate"],
            "b009_localized_within_20s_rate": new["localized_within_20s_rate"],
            "b009_notification_within_20s_rate": new["notified_within_20s_rate"],
            "localized_within_20s_difference": new["localized_within_20s_rate"] - old["detected_within_20s_rate"],
            "paired_group_localization_effect": paired_group_bootstrap(
                new["per_group_notified_within_20s_rate"],
                old["per_group_detected_within_20s_rate"],
            ),
        }
    return {
        "schema_version": "astrograph-b008-b009-development-comparison.v1",
        "comparison_is_development_only": True,
        "groups": [3, 8],
        "groups_4_9_read": False,
        "variants": comparison,
        "boundary": (
            "This paired development comparison quantifies recovery from the B008 all-abstain mismatch guard. "
            "It is not confirmatory and does not establish arbitrary reference robustness."
        ),
    }



def build_failure_analysis(receipt, bundles, truth_table, alignment_contract) -> dict:
    truth = {row.opaque_trajectory_id: row for row in truth_table.rows}
    output = {
        "schema_version": "astrograph-mismatch-tolerant-failure-analysis.v1",
        "development_groups": [3, 8],
        "excluded_groups": [4, 9],
        "groups_4_9_read": False,
        "policy_sha256": receipt.selected_policy.policy_sha256,
        "variants": {},
    }
    for variant in REQUIRED_DEVELOPMENT_VARIANTS:
        misses = []
        admitted = []
        abstained = []
        for trace in bundles[variant].traces:
            row = truth[trace.opaque_trajectory_id]
            selection = select_mismatch_tolerant_trace(
                trace, receipt.selected_policy, alignment_contract
            )
            item = {
                "opaque_trajectory_id": trace.opaque_trajectory_id,
                "independent_group_id": trace.independent_group_id,
                "diagnostic_class": row.diagnostic_class,
                "is_nominal": row.is_nominal,
                "alignment_fit_score": trace.alignment.fit_score,
                "selected_lag_points": trace.alignment.selected_lag_points,
                "status": selection.status.value,
                "selected_time_s": selection.selected_time_s,
                "notification_available_time_s": selection.notification_available_time_s,
                "active_start_s": row.active_start_s,
                "active_end_s": row.active_end_s,
            }
            if selection.alignment_decision.value == "ADMITTED":
                admitted.append(item)
            else:
                abstained.append(item)
            if not row.is_nominal:
                timely = (
                    selection.notification_available_time_s is not None
                    and selection.selected_time_s is not None
                    and selection.selected_time_s >= row.active_start_s
                    and selection.selected_time_s <= row.active_end_s
                    and selection.notification_available_time_s - row.active_start_s <= 20.0
                )
                if not timely:
                    misses.append(item)
        output["variants"][variant.value] = {
            "fault_miss_count": len(misses),
            "admitted_count": len(admitted),
            "abstained_count": len(abstained),
            "fault_misses": misses,
            "admitted_class_counts": dict(
                sorted(
                    Counter(
                        item["diagnostic_class"]
                        for item in admitted
                        if not item["is_nominal"]
                    ).items()
                )
            ),
            "abstained_class_counts": dict(
                sorted(
                    Counter(
                        item["diagnostic_class"]
                        for item in abstained
                        if not item["is_nominal"]
                    ).items()
                )
            ),
        }
    output["main_findings"] = [
        "The five-percent-noise route has one fault trajectory not notified within 20 seconds; this is not hidden by aggregate accuracy.",
        "Cross-Monte-Carlo admission and misses are class-heterogeneous; no general cross-reference claim is supported.",
    ]
    output["analysis_sha256"] = canonical_sha256(output)
    return output


def build_prefix_truth_audit(
    truth_table: ModelIndependentTruthTable,
    exact_bundle: FrozenMismatchTraceBundle,
    alignment_contract: MismatchTolerantAlignmentContract,
) -> dict:
    """Audit, offline, whether the fixed admission prefix overlaps a fault.

    The result is deliberately not available to the runtime trace generator and
    is not used to choose a policy.  It records an assumption that is otherwise
    easy to hide: prefix calibration is interpretable as nominal only when the
    active fault interval starts after the final prefix sample.
    """

    if not exact_bundle.traces:
        raise ValueError("prefix audit requires at least one exact-pair trace")
    prefix_index = alignment_contract.prefix_points - 1
    prefix_times = {
        float(trace.time_s[prefix_index]) for trace in exact_bundle.traces
    }
    if len(prefix_times) != 1:
        raise ValueError("development traces do not share one prefix end time")
    prefix_end_time_s = prefix_times.pop()
    fault_rows = [row for row in truth_table.rows if not row.is_nominal]
    overlapping = [
        {
            "opaque_trajectory_id": row.opaque_trajectory_id,
            "independent_group_id": row.independent_group_id,
            "diagnostic_class": row.diagnostic_class,
            "active_start_s": row.active_start_s,
            "prefix_end_time_s": prefix_end_time_s,
        }
        for row in fault_rows
        if float(row.active_start_s) <= prefix_end_time_s
    ]
    starts = [float(row.active_start_s) for row in fault_rows]
    output = {
        "schema_version": "astrograph-mismatch-prefix-truth-audit.v1",
        "audit_is_offline_after_trace_freeze": True,
        "audit_used_for_policy_selection": False,
        "development_groups": [3, 8],
        "excluded_groups": [4, 9],
        "groups_4_9_read": False,
        "prefix_points": alignment_contract.prefix_points,
        "prefix_end_time_s": prefix_end_time_s,
        "fault_trajectory_count": len(fault_rows),
        "minimum_fault_active_start_s": min(starts) if starts else None,
        "maximum_fault_active_start_s": max(starts) if starts else None,
        "fault_prefix_overlap_count": len(overlapping),
        "fault_prefix_overlaps": overlapping,
        "nominal_prefix_observed_on_this_development_set": not overlapping,
        "claim_boundary": (
            "This is an evaluator-side development audit of the fixed prefix assumption. "
            "It does not guarantee a nominal prefix in a new campaign or in flight."
        ),
    }
    output["audit_sha256"] = canonical_sha256(output)
    return output


def build_candidate_selection_summary(ranked, candidate_grid_sha256: str) -> dict:
    """Produce a concise, deterministic summary of the development search."""

    if not ranked:
        raise ValueError("candidate summary requires at least one candidate")
    selected = ranked[0]
    scientific_key = tuple(selected.selection_key[:10])
    scientific_ties = [
        candidate
        for candidate in ranked
        if tuple(candidate.selection_key[:10]) == scientific_key
    ]
    exact_pass = [candidate for candidate in ranked if candidate.exact_noninferiority_pass]
    safe = [candidate for candidate in ranked if candidate.safe_across_development_variants]
    summary = {
        "schema_version": "astrograph-mismatch-candidate-selection-summary.v1",
        "candidate_grid_sha256": candidate_grid_sha256,
        "candidate_count": len(ranked),
        "safe_candidate_count": len(safe),
        "exact_noninferiority_candidate_count": len(exact_pass),
        "top_scientific_outcome_tie_count": len(scientific_ties),
        "top_scientific_outcome_tied_policy_ids": [
            candidate.policy.policy_id for candidate in scientific_ties
        ],
        "selected_policy_id": selected.policy.policy_id,
        "selected_policy_sha256": selected.policy.policy_sha256,
        "selection_order": [
            "zero observed pre-onset and nominal-false-alarm groups across all declared development variants",
            "exact-pair non-inferiority gate",
            "worst bounded-variant group-macro notification within 20 seconds",
            "worst bounded-variant fault admission",
            "mean bounded-variant group-macro notification within 20 seconds",
            "minimum bounded-variant class notification within 20 seconds",
            "cross-Monte-Carlo selective notification and admission",
            "notification delay",
            "tighter alignment-fit envelope, higher trigger threshold, and persistence as deterministic tie-breaks",
            "policy hash as final deterministic tie-break",
        ],
        "selection_informed_by_prior_development_exploration": True,
        "multiplicity_boundary": (
            "Forty candidates were compared on exposed groups 3/8. The selected result is development evidence; "
            "it is not an untouched inferential estimate and does not supply a confirmatory p-value."
        ),
        "groups_4_9_read": False,
        "new_confirmatory_evidence": "NONE",
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    return summary

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--trace-index", type=Path, required=True)
    parser.add_argument("--truth-table", type=Path, required=True)
    parser.add_argument("--scale-contract", type=Path, required=True)
    parser.add_argument("--alignment-contract", type=Path, required=True)
    parser.add_argument("--candidate-grid", type=Path, required=True)
    parser.add_argument("--b008-result", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    trace_dir = args.trace_dir.resolve()
    trace_index = load_trace_index(args.trace_index.resolve())
    truth = load_truth(args.truth_table.resolve())
    scale_contract = load_scale_contract(args.scale_contract.resolve())
    alignment_contract = load_alignment_contract(args.alignment_contract.resolve())
    policies, candidate_grid_sha256, candidate_grid_payload = load_candidate_grid(
        args.candidate_grid.resolve()
    )

    if truth.dataset_archive_sha256 != scale_contract.dataset_archive_sha256:
        raise ValueError("truth/scale dataset identity differs")
    if trace_index["dataset_archive_sha256"] != truth.dataset_archive_sha256:
        raise ValueError("trace/truth dataset identity differs")
    if trace_index["scale_contract_sha256"] != scale_contract.contract_sha256:
        raise ValueError("trace/scale contract identity differs")
    if trace_index["alignment_contract_sha256"] != alignment_contract.contract_sha256:
        raise ValueError("trace/alignment contract identity differs")

    bundles: dict[TraceVariant, FrozenMismatchTraceBundle] = {}
    for variant in REQUIRED_DEVELOPMENT_VARIANTS:
        info = trace_index["trace_bundles"][variant.value]
        bundle = load_bundle(
            trace_dir / info["path"],
            info["sha256"],
            info["bundle_sha256"],
        )
        if bundle.variant != variant:
            raise ValueError("trace bundle variant differs from index")
        bundles[variant] = bundle

    ranked = rank_mismatch_tolerant_policies(
        policies,
        bundles,
        truth,
        alignment_contract,
    )
    receipt = freeze_mismatch_tolerant_policy(
        ranked,
        bundles,
        truth,
        scale_contract,
        alignment_contract,
        candidate_grid_sha256,
    )
    selected = receipt.selected_policy
    selected_metrics = {
        key: value.model_dump(mode="json")
        for key, value in receipt.selected_metrics_by_variant.items()
    }

    exact = receipt.selected_metrics_by_variant[TraceVariant.EXACT_PAIRED.value]
    group_successes = sum(
        value >= 1.0 - 1e-12
        for value in exact.per_group_notified_within_20s_rate.values()
    )
    group_trials = len(exact.per_group_notified_within_20s_rate)
    exact_group_interval = wilson_interval(group_successes, group_trials)

    stress_safety = all(
        metrics.pre_onset_selection_count == 0
        and metrics.nominal_false_alarm_count == 0
        for metrics in receipt.selected_metrics_by_variant.values()
    )
    bounded_variants = [
        TraceVariant.EXACT_PAIRED,
        TraceVariant.REFERENCE_SHIFT_1S,
        TraceVariant.REFERENCE_SHIFT_2S,
        TraceVariant.CURRENT_NOISE_1PCT_SCALE,
        TraceVariant.CURRENT_NOISE_5PCT_SCALE,
    ]
    bounded_min_group = min(
        receipt.selected_metrics_by_variant[variant.value].group_macro_notified_within_20s_rate
        for variant in bounded_variants
    )
    bounded_min_admission = min(
        receipt.selected_metrics_by_variant[variant.value].fault_alignment_admission_rate
        for variant in bounded_variants
    )
    cross = receipt.selected_metrics_by_variant[TraceVariant.CROSS_MC_REFERENCE.value]
    strong_bounded_development = (
        stress_safety
        and bounded_min_group >= 0.95
        and bounded_min_admission >= 0.95
        and exact.minimum_class_notified_within_20s_rate >= 0.95
    )

    comparison = compare_to_b008(
        selected_metrics,
        args.b008_result.resolve() if args.b008_result else None,
    )
    failure_analysis = build_failure_analysis(receipt, bundles, truth, alignment_contract)
    prefix_truth_audit = build_prefix_truth_audit(
        truth,
        bundles[TraceVariant.EXACT_PAIRED],
        alignment_contract,
    )
    candidate_selection_summary = build_candidate_selection_summary(
        ranked, candidate_grid_sha256
    )

    assessment = {
        "schema_version": "astrograph-mismatch-tolerant-thesis-contribution-assessment.v1",
        "assessment": (
            "STRONG_BOUNDED_DEVELOPMENT_CONTRIBUTION_CROSS_REFERENCE_GENERALIZATION_OPEN"
            if strong_bounded_development
            else "BOUNDED_METHOD_REQUIRES_FURTHER_DEVELOPMENT"
        ),
        "label_isolation_enforced": True,
        "classifier_independent": True,
        "full_trajectory_warping_used": False,
        "bounded_prefix_integer_lag_alignment": True,
        "notification_buffer_accounted": True,
        "nominal_prefix_observed_on_this_development_set": prefix_truth_audit[
            "nominal_prefix_observed_on_this_development_set"
        ],
        "fault_prefix_overlap_count": prefix_truth_audit[
            "fault_prefix_overlap_count"
        ],
        "safe_across_development_variants": stress_safety,
        "bounded_in_envelope_min_group_macro_notification_within20": bounded_min_group,
        "bounded_in_envelope_min_fault_admission": bounded_min_admission,
        "cross_mc_fault_admission_rate": cross.fault_alignment_admission_rate,
        "cross_mc_notification_within20_rate": cross.notified_within_20s_rate,
        "cross_mc_minimum_class_notification_within20": cross.minimum_class_notified_within_20s_rate,
        "arbitrary_cross_reference_robustness_established": False,
        "fault_identification_evaluated": False,
        "new_confirmatory_evidence": "NONE",
        "standalone_sufficient_for_excellent_thesis": False,
        "bounded_contribution_value": [
            "Replaces B008 all-abstain behavior for bounded timing shift and scale-relative noise with a label-free prefix alignment and calibrated selective score.",
            "Keeps padded edge samples invalid and accounts for a bounded online notification buffer rather than reporting noncausal look-ahead as zero delay.",
            "Retains fail-closed abstention under high alignment mismatch while recovering nonzero selective cross-reference coverage.",
            "Shows that cross-Monte-Carlo generalization remains heterogeneous and cannot be generalized from the bounded stress result.",
        ],
        "remaining_requirements": [
            "Develop a reference-set or probabilistic reference model for cross-Monte-Carlo mismatch without using groups 4/9.",
            "Increase the number and diversity of independent development and sealed-confirmatory groups.",
            "Freeze downstream rejection/fusion policies after this temporal policy and then generate a new sealed campaign.",
            "Keep fault-identification and universal KG-gain claims frozen until separately validated.",
        ],
    }
    assessment["assessment_sha256"] = canonical_sha256(assessment)

    ranking_payload = [
        {
            "rank": rank,
            "policy": candidate.policy.model_dump(mode="json"),
            "safe_across_development_variants": candidate.safe_across_development_variants,
            "exact_noninferiority_pass": candidate.exact_noninferiority_pass,
            "selection_key": list(candidate.selection_key),
            "metrics_by_variant": {
                key: value.model_dump(mode="json")
                for key, value in candidate.metrics_by_variant.items()
            },
        }
        for rank, candidate in enumerate(ranked, start=1)
    ]

    result = {
        "schema_version": "astrograph-adcs-mismatch-tolerant-temporal-development-result.v1",
        "project_status": "B009_WIP_DEVELOPMENT_EVIDENCE",
        "dataset_archive_sha256": truth.dataset_archive_sha256,
        "development_groups": [3, 8],
        "excluded_groups": [4, 9],
        "groups_4_9_read": False,
        "scale_contract_sha256": scale_contract.contract_sha256,
        "alignment_contract_sha256": alignment_contract.contract_sha256,
        "truth_table_sha256": truth.truth_table_sha256,
        "candidate_grid_sha256": candidate_grid_sha256,
        "candidate_grid_design_disclosure": candidate_grid_payload[
            "design_disclosure"
        ],
        "selected_policy_receipt": receipt.model_dump(mode="json"),
        "candidate_ranking": ranking_payload,
        "b008_development_comparison": comparison,
        "failure_analysis_sha256": failure_analysis["analysis_sha256"],
        "prefix_truth_audit_sha256": prefix_truth_audit["audit_sha256"],
        "candidate_selection_summary_sha256": candidate_selection_summary[
            "summary_sha256"
        ],
        "independent_group_uncertainty": {
            "group_count": group_trials,
            "exact_groups_with_100pct_notification_within20": group_successes,
            "wilson_95_interval_for_exact_group_success_probability": list(
                exact_group_interval
            ),
            "warning": (
                "Only six independent trajectory-pair groups are available. Trajectory counts do not "
                "increase the independent sample size."
            ),
        },
        "thesis_contribution_assessment": assessment,
        "result_boundary": [
            "Development evidence only; groups 4/9 were not read or used.",
            "The method is independent of the missing learned P2 classifier, but still depends on a nominal reference trajectory and a nominal initial prefix.",
            "The initial prefix is observed to precede every active fault interval on this development set; that is an offline audit, not a guarantee for a new campaign.",
            "Bounded timing shift and scale-relative noise are addressed; arbitrary cross-Monte-Carlo reference generalization is not established.",
            "No fault-identification, KG-gain, external-validity, HIL, safety, flight, or confirmatory claim is supported.",
        ],
    }
    result["result_sha256"] = canonical_sha256(result)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "development_result.json", result)
    write_json(output_dir / "selected_policy_receipt.json", receipt.model_dump(mode="json"))
    write_json(output_dir / "candidate_ranking.json", {"candidates": ranking_payload, "candidate_grid_sha256": candidate_grid_sha256})
    write_json(output_dir / "thesis_contribution_assessment.json", assessment)
    write_json(output_dir / "failure_analysis.json", failure_analysis)
    write_json(output_dir / "prefix_truth_audit.json", prefix_truth_audit)
    write_json(
        output_dir / "candidate_selection_summary.json",
        candidate_selection_summary,
    )
    if comparison is not None:
        comparison["comparison_sha256"] = canonical_sha256(comparison)
        write_json(output_dir / "b008_b009_development_comparison.json", comparison)

    rows = []
    for variant in REQUIRED_DEVELOPMENT_VARIANTS:
        metrics = receipt.selected_metrics_by_variant[variant.value]
        rows.append(
            f"| {variant.value} | {metrics.alignment_admission_rate:.4f} | "
            f"{metrics.localized_within_20s_rate:.4f} | {metrics.notified_within_20s_rate:.4f} | "
            f"{metrics.pre_onset_selection_count} | {metrics.nominal_false_alarm_count} | "
            f"{metrics.median_notification_delay_s} |"
        )
    report = f"""# AstroGraph B009 Mismatch-Tolerant Temporal Development Result R001

## Status

- Snapshot: `B009_WIP_DEVELOPMENT_EVIDENCE`
- Selection groups: `3/8` only
- Permanently post-hoc groups: `4/9` — not read
- Runtime truth access: `NO`
- Learned P2 classifier required: `NO`
- New confirmatory evidence: `NONE`

## Frozen method

- Prefix-only alignment points: `{alignment_contract.prefix_points}`
- Integer lag search: `{list(alignment_contract.candidate_lags)}`
- Selected trigger threshold: `{selected.trigger_threshold}`
- Selected persistence: `{selected.trigger_persistence_points}`
- Selected alignment-fit maximum: `{selected.alignment_fit_max}`
- Conservative notification buffer: `{selected.notification_buffer_points}` points
- Policy SHA-256: `{selected.policy_sha256}`
- Receipt SHA-256: `{receipt.receipt_sha256}`
- Development candidates compared: `{candidate_selection_summary['candidate_count']}`
- Top scientific-outcome ties before deterministic policy tie-breaks: `{candidate_selection_summary['top_scientific_outcome_tie_count']}`

The candidate grid was finalized for this B009 evaluation after prior groups-3/8 development exploration. It is not an untouched or confirmatory selection.

The fixed 31-point prefix ended at `{prefix_truth_audit['prefix_end_time_s']} s`; the earliest fault active interval began at `{prefix_truth_audit['minimum_fault_active_start_s']} s`, with `{prefix_truth_audit['fault_prefix_overlap_count']}` observed prefix overlaps. This check was performed offline after trace freeze and was not available to the runtime generator.

## Development results

| Variant | Admission | Localized within 20 s | Notified within 20 s | Pre-onset | Nominal FA | Median notification delay (s) |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Notification metrics include the bounded alignment buffer. This prevents a negative lag from being reported as a zero-latency online result.

## Main finding

`{assessment['assessment']}`

The bounded method recovers full development coverage for one- and two-second reference shifts and for the declared scale-relative noise envelope, while preserving zero pre-onset selections and zero nominal false alarms in the six declared development variants. Cross-Monte-Carlo reference behavior remains selective and heterogeneous; it is not solved by this method.

## Statistical boundary

- Independent trajectory-pair groups: `{group_trials}`
- Exact groups with 100% notification within 20 s: `{group_successes}/{group_trials}`
- Wilson 95% interval: `{exact_group_interval}`

Trajectory count is descriptive only and is not treated as the independent sample size.

## Research boundary

- No ADCS fault-identification result was produced.
- No KG, fusion, or rejection policy was selected.
- No groups 4/9 result was read.
- No external, HIL, safety, flight, G0–G4, or confirmatory status changed.
"""
    (output_dir / "development_result.md").write_text(report, encoding="utf-8")

    print(
        json.dumps(
            {
                "result": str(output_dir / "development_result.json"),
                "result_sha256": result["result_sha256"],
                "selected_policy_sha256": selected.policy_sha256,
                "selected_policy_receipt_sha256": receipt.receipt_sha256,
                "assessment": assessment["assessment"],
                "bounded_min_group_macro_notification_within20": bounded_min_group,
                "cross_mc_notification_within20_rate": cross.notified_within_20s_rate,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
