#!/usr/bin/env python3
"""A14 compact LLM revision over A13 length-free features and OOF calibration."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from astrograph.evaluation.data_exposure import dataset_use
from astrograph.evaluation.ops_sat_adapter import build_partitions, load_feature_track, load_raw_track
from astrograph.evaluation.ops_sat_probe_utility import measure_ops_sat_length_free_feature_bundle
from scripts.experiments.run_ops_sat_compact_bundle_comparator import (
    CompactBundleRevision,
    _candidate_states,
    _citations,
    summarize_attributable_outcomes,
)
from scripts.experiments.run_ops_sat_deterministic_bundle_comparator import exact_mcnemar_two_sided
from scripts.experiments.run_ops_sat_llm_active_shadow import _backend, _digest, _models, _sha_file, _usage


def _validated_report(path: Path, schema: str) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != schema:
        raise ValueError(f"unsupported report schema: {path}")
    content = dict(document)
    claimed = content.pop("report_sha256", None)
    if not isinstance(claimed, str) or _digest(content) != claimed:
        raise ValueError(f"report content digest mismatch: {path}")
    return document


def run_feature_llm_arm(
    *, feature_row: dict, feature_source_sha256: str, a13_case: dict,
    frozen_prediction: dict, a13_report_sha256: str, url: str, model: str,
) -> dict:
    feature_observation = measure_ops_sat_length_free_feature_bundle(
        feature_row, source_sha256=feature_source_sha256,
    )
    calibration = a13_case["length_free_policy"]
    calibration_binding = {
        "feature_observation_id": feature_observation["observation_id"],
        "a13_report_sha256": a13_report_sha256,
        "out_of_fold_probability": calibration["out_of_fold_probability"],
        "fold_selected_threshold": calibration["fold_selected_threshold"],
        "fold_selected_candidate_id": calibration["fold_selected_candidate_id"],
        "predicted_state": calibration["predicted_state"],
    }
    active = {
        **feature_observation,
        "calibration": calibration_binding,
        "observation_id": "OPSFEATURESCORE-" + _digest(calibration_binding)[:16].upper(),
        "calibration_semantics": (
            "Strict out-of-fold development-only logistic score; not a probability qualified for production."
        ),
    }
    baseline = {
        "observation_id": "OPSBASE-" + _digest({
            "segment": feature_row["segment"], "frozen": frozen_prediction,
        })[:16].upper(),
        "score": frozen_prediction["score"],
        "threshold": frozen_prediction["threshold"],
        "predicted_state": "ANOMALOUS" if frozen_prediction["predicted_anomaly"] else "NOMINAL",
        "score_semantics": frozen_prediction["score_semantics"],
    }
    backend = _backend(
        url, model, max_tokens=4096, response_format_strategy="plain_json",
        repair_attempts=0, max_http_attempts=1,
    )
    started = time.perf_counter()
    revision = backend.generate_structured(
        agent_id="ops-sat-length-free-feature-state-reviser",
        output_model=CompactBundleRevision,
        payload={
            "task": "Return a two-field shadow decision using the frozen baseline and derived-feature observation.",
            "baseline_candidate_states": ["NOMINAL", "ANOMALOUS"],
            "baseline_observation": baseline,
            "active_derived_feature_observation": active,
            "decision_meanings": {
                "KEEP_BOTH": "retain both candidates",
                "NOMINAL": "retain only NOMINAL",
                "ANOMALOUS": "retain only ANOMALOUS",
                "ABSTAIN": "recommend abstention and retain both candidates",
            },
            "rules": [
                "Truth labels are unavailable.",
                "The out-of-fold score is development-only and may be wrong.",
                "Do not infer root cause or invent observations.",
                "Use ACTIVE_BUNDLE_ONLY or BOTH only when derived features affect the decision.",
                "Return only the two schema fields.",
            ],
        },
    )
    candidates = _candidate_states(revision.decision)
    citations = _citations(revision.citation_scope, baseline["observation_id"], active["observation_id"])
    active_cited = active["observation_id"] in citations
    changed = set(candidates) != {"NOMINAL", "ANOMALOUS"}
    return {
        "arm": "A14_LENGTH_FREE_FEATURE_LLM",
        "baseline_observation": baseline,
        "observation": active,
        "revision": {
            **revision.model_dump(mode="json"),
            "candidate_states": candidates,
            "abstain_recommended": revision.decision == "ABSTAIN",
            "cited_observation_ids": citations,
        },
        "validated_candidate_states": candidates,
        "candidate_set_changed": changed,
        "active_observation_exposed": True,
        "active_observation_cited": active_cited,
        "active_observation_attributable": active_cited and changed,
        "elapsed_s": time.perf_counter() - started,
        "usage": _usage(backend),
        "backend_call_history": backend.call_history,
    }


def _summary(rows: list[dict]) -> dict:
    decided = [
        row for row in rows
        if len(row["arm"]["validated_candidate_states"]) == 1
        and not row["arm"]["revision"]["abstain_recommended"]
    ]
    nominal = [row for row in rows if row["offline_truth"] == "NOMINAL"]
    correct = sum(row["arm"]["validated_candidate_states"][0] == row["offline_truth"] for row in decided)
    return {
        "case_count": len(rows),
        "decided_count": len(decided),
        "abstain_or_ambiguous_count": len(rows) - len(decided),
        "correct_decided_count": correct,
        "accuracy_with_abstention_as_incorrect": correct / len(rows),
        "healthy_false_alarm_count": sum(
            row["arm"]["validated_candidate_states"] == ["ANOMALOUS"]
            and not row["arm"]["revision"]["abstain_recommended"] for row in nominal
        ),
        "healthy_case_count": len(nominal),
        "active_observation_exposed_count": len(rows),
        "active_observation_cited_count": sum(row["arm"]["active_observation_cited"] for row in rows),
        "active_observation_attributable_count": sum(row["arm"]["active_observation_attributable"] for row in rows),
        "elapsed_s": sum(row["arm"]["elapsed_s"] for row in rows),
        "prompt_tokens": sum(row["arm"]["usage"]["prompt_tokens"] for row in rows),
        "completion_tokens": sum(row["arm"]["usage"]["completion_tokens"] for row in rows),
        "llm_call_count": sum(row["arm"]["usage"]["call_count"] for row in rows),
    }


def _a13_paired_summary(rows: list[dict], a13_by_group: dict[str, dict]) -> dict:
    predicted = [a13_by_group[row["group_id"]]["length_free_policy"]["predicted_state"] for row in rows]
    truth = [row["offline_truth"] for row in rows]
    true_positive = sum(p == t == "ANOMALOUS" for p, t in zip(predicted, truth))
    false_positive = sum(p == "ANOMALOUS" and t == "NOMINAL" for p, t in zip(predicted, truth))
    false_negative = sum(p == "NOMINAL" and t == "ANOMALOUS" for p, t in zip(predicted, truth))
    true_negative = sum(p == t == "NOMINAL" for p, t in zip(predicted, truth))
    positive = true_positive + false_negative
    negative = true_negative + false_positive
    precision_denominator = true_positive + false_positive
    recall = true_positive / positive if positive else None
    specificity = true_negative / negative if negative else None
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if recall is not None and precision + recall > 0.0 else 0.0
    )
    return {
        "case_count": len(rows),
        "accuracy": sum(p == t for p, t in zip(predicted, truth)) / len(rows),
        "balanced_accuracy": (
            (recall + specificity) / 2.0
            if recall is not None and specificity is not None else None
        ),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "healthy_false_alarm_count": false_positive,
        "healthy_case_count": negative,
        "abstain_or_ambiguous_count": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--a10-report", type=Path, required=True)
    parser.add_argument("--a13-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--model", default="qwen/qwen3.5-9b")
    parser.add_argument("--case-limit", type=int)
    args = parser.parse_args()
    if args.case_limit is not None and not 1 <= args.case_limit <= 24:
        parser.error("case-limit must be between 1 and 24")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output-dir must be new or empty")
    a10 = _validated_report(args.a10_report, "astrograph-ops-sat-compact-bundle-comparator.v1")
    a13 = _validated_report(args.a13_report, "astrograph-ops-sat-official-feature-observation-audit.v1")
    if a13["length_free_policy"]["promotion_recommendation"] != "ELIGIBLE_FOR_LLM_SHADOW_EXPERIMENT_ONLY":
        raise ValueError("A13 length-free observation did not pass the shadow-experiment gate")
    if args.model != a10["preflight_receipt"]["model"]:
        raise ValueError("A14 model identity differs from paired A10")

    dataset_path = args.data_root / "dataset.csv"
    segments_path = args.data_root / "segments.csv"
    receipts = {
        "dataset.csv": {"sha256": _sha_file(dataset_path), "size": dataset_path.stat().st_size},
        "segments.csv": {"sha256": _sha_file(segments_path), "size": segments_path.stat().st_size},
    }
    if any(receipts[name] != a13["preflight_receipt"]["source_receipts"][name] for name in receipts):
        raise ValueError("A14 source identity differs from A13")
    if args.model not in _models(args.url):
        raise ValueError("A14 requested model is unavailable")

    feature, truth = load_feature_track(dataset_path)
    raw, raw_truth = load_raw_track(segments_path)
    if truth != raw_truth or set(feature) != set(raw):
        raise ValueError("OPS raw/feature identities differ")
    config = json.loads((ROOT / "configs/evaluation/r20_ops_sat.json").read_text(encoding="utf-8"))
    partition_by = {row["group_id"]: row["partition"] for row in build_partitions(truth, config)["groups"]}
    selected = list(a10["preflight_receipt"]["selected_group_ids"])
    if args.case_limit is not None:
        selected = selected[:args.case_limit]
    if any(partition_by.get(group_id) != "DEVELOPMENT_SELECTION" for group_id in selected):
        raise ValueError("A14 groups are outside development selection")
    a10_by_group = {row["group_id"]: row for row in a10["cases"]}
    a13_by_group = {row["group_id"]: row for row in a13["cases"]}
    preflight = {
        "status": "PASS",
        "arm": "A14_LENGTH_FREE_FEATURE_LLM",
        "selected_group_ids": selected,
        "source_receipts": receipts,
        "a10_report_file_sha256": _sha_file(args.a10_report),
        "a10_report_content_sha256": a10["report_sha256"],
        "a13_report_file_sha256": _sha_file(args.a13_report),
        "a13_report_content_sha256": a13["report_sha256"],
        "model": args.model,
        "external_validation_groups_consumed": False,
        "configured_max_tokens": 4096,
        "maximum_http_attempts_per_case": 1,
        "repair_attempts": 0,
        "feature_count": 12,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with dataset_use("OPS_SAT", "DEVELOPMENT_SELECTION", "AUDIT", group_ids=selected) as admission:
        for index, group_id in enumerate(selected, start=1):
            print(f"case {index}/{len(selected)} group={group_id} arm=A14_LENGTH_FREE_FEATURE_LLM", flush=True)
            try:
                arm = run_feature_llm_arm(
                    feature_row=feature[group_id],
                    feature_source_sha256=receipts["dataset.csv"]["sha256"],
                    a13_case=a13_by_group[group_id],
                    frozen_prediction=a10_by_group[group_id]["frozen_prediction"],
                    a13_report_sha256=a13["report_sha256"],
                    url=args.url,
                    model=args.model,
                )
            except Exception as exc:
                blocked = {
                    "schema_version": "astrograph-ops-sat-feature-llm-blocked.v1",
                    "status": "BLOCKED_MODEL_REVISION_UNAVAILABLE",
                    "preflight_receipt": preflight,
                    "failed_group_id": group_id,
                    "completed_case_count": len(rows),
                    "error": f"{type(exc).__name__}: {exc}",
                    "truth_joined_for_failed_case": False,
                    "claim_boundary": "A14 stopped before scoring the failed case; not a diagnostic result.",
                }
                blocked["report_sha256"] = _digest(blocked)
                (args.output_dir / "blocked_report.json").write_text(
                    json.dumps(blocked, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                raise
            rows.append({
                "group_id": group_id,
                "channel": feature[group_id]["channel"],
                "arm": arm,
                "offline_truth": "ANOMALOUS" if truth[group_id]["anomaly"] else "NOMINAL",
            })
    summary = _summary(rows)
    paired = _a13_paired_summary(rows, a13_by_group)
    a14_correct = {
        row["group_id"]: (
            len(row["arm"]["validated_candidate_states"]) == 1
            and not row["arm"]["revision"]["abstain_recommended"]
            and row["arm"]["validated_candidate_states"][0] == row["offline_truth"]
        )
        for row in rows
    }
    a13_correct = {
        row["group_id"]: a13_by_group[row["group_id"]]["length_free_policy"]["predicted_state"] == row["offline_truth"]
        for row in rows
    }
    a14_only_correct = sum(a14_correct[group_id] and not a13_correct[group_id] for group_id in a14_correct)
    a13_only_correct = sum(a13_correct[group_id] and not a14_correct[group_id] for group_id in a14_correct)
    report = {
        "schema_version": "astrograph-ops-sat-feature-llm-comparator.v1",
        "status": "EXPLORATORY_DEVELOPMENT_RESULT",
        "preflight_receipt": preflight,
        "dataset_admission": admission,
        "a14_feature_llm": summary,
        "attributable_outcome_audit": summarize_attributable_outcomes(rows),
        "a13_deterministic_paired_metrics": paired,
        "cases": rows,
        "comparison_to_a13": {
            "accuracy_delta_a14_minus_a13": summary["accuracy_with_abstention_as_incorrect"] - paired["accuracy"],
            "healthy_false_alarm_delta_a14_minus_a13": summary["healthy_false_alarm_count"] - paired["healthy_false_alarm_count"],
            "abstain_or_ambiguous_delta_a14_minus_a13": summary["abstain_or_ambiguous_count"] - paired["abstain_or_ambiguous_count"],
            "a14_only_correct_count": a14_only_correct,
            "a13_only_correct_count": a13_only_correct,
            "paired_exact_mcnemar_two_sided_p": exact_mcnemar_two_sided(
                a14_only_correct, a13_only_correct,
            ),
            "same_underlying_length_free_feature_values": True,
            "llm_additional_call_count": summary["llm_call_count"],
        },
        "claim_boundary": (
            "Development-only LLM meta-decision over retrospective derived features and an OOF calibration summary. "
            "Not an equal-algorithm comparison, external validation, early-warning result, formal evidence, or qualification."
        ),
    }
    report["report_sha256"] = _digest(report)
    output = args.output_dir / "report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": summary, "comparison_to_a13": report["comparison_to_a13"], "report_sha256": report["report_sha256"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
