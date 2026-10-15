#!/usr/bin/env python3
"""A10 compact-LLM comparator over one six-component OPS-SAT observation."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from astrograph.evaluation.data_exposure import dataset_use
from astrograph.evaluation.ops_sat_adapter import build_partitions, load_feature_track, load_raw_track
from astrograph.evaluation.ops_sat_probe_utility import measure_ops_sat_probe_bundle
from astrograph.evaluation.replication_r22 import ops_r21_frozen_remediation_prediction
from scripts.experiments.run_ops_sat_llm_active_shadow import (
    _backend,
    _digest,
    _models,
    _sha_file,
    _usage,
)


class CompactBundleRevision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["KEEP_BOTH", "NOMINAL", "ANOMALOUS", "ABSTAIN"]
    citation_scope: Literal["BASELINE_ONLY", "ACTIVE_BUNDLE_ONLY", "BOTH"]


def _candidate_states(decision: str) -> list[str]:
    if decision == "NOMINAL":
        return ["NOMINAL"]
    if decision == "ANOMALOUS":
        return ["ANOMALOUS"]
    return ["ANOMALOUS", "NOMINAL"]


def _citations(scope: str, baseline_id: str, bundle_id: str) -> list[str]:
    if scope == "BASELINE_ONLY":
        return [baseline_id]
    if scope == "ACTIVE_BUNDLE_ONLY":
        return [bundle_id]
    return [baseline_id, bundle_id]


def summarize_attributable_outcomes(rows: list[dict]) -> dict:
    attributable = [row for row in rows if row["arm"]["active_observation_attributable"]]
    correct = [
        row for row in attributable
        if len(row["arm"]["validated_candidate_states"]) == 1
        and row["arm"]["validated_candidate_states"][0] == row["offline_truth"]
        and not row["arm"]["revision"]["abstain_recommended"]
    ]
    incorrect = [
        row for row in attributable
        if len(row["arm"]["validated_candidate_states"]) == 1
        and row["arm"]["validated_candidate_states"][0] != row["offline_truth"]
        and not row["arm"]["revision"]["abstain_recommended"]
    ]
    if not attributable:
        verdict = "NO_ATTRIBUTABLE_CANDIDATE_CHANGE"
    elif incorrect and not correct:
        verdict = "ATTRIBUTABLE_HARM_OBSERVED_NO_BENEFIT"
    elif correct and not incorrect:
        verdict = "ATTRIBUTABLE_BENEFIT_OBSERVED_EXPLORATORY_ONLY"
    else:
        verdict = "MIXED_ATTRIBUTABLE_OUTCOMES"
    return {
        "attributable_case_count": len(attributable),
        "attributable_correct_count": len(correct),
        "attributable_incorrect_count": len(incorrect),
        "attributable_group_ids": [row["group_id"] for row in attributable],
        "attributable_correct_group_ids": [row["group_id"] for row in correct],
        "attributable_incorrect_group_ids": [row["group_id"] for row in incorrect],
        "verdict": verdict,
    }


def run_compact_bundle_arm(
    *, points: list[dict], frozen: dict, source_sha256: str, url: str, model: str
) -> dict:
    backend = _backend(
        url,
        model,
        max_tokens=4096,
        response_format_strategy="plain_json",
        repair_attempts=0,
        max_http_attempts=1,
    )
    baseline = {
        "observation_id": "OPSBASE-" + _digest({"segment": points[0]["segment"], "frozen": frozen})[:16].upper(),
        "score": frozen["score"],
        "threshold": frozen["threshold"],
        "predicted_state": "ANOMALOUS" if frozen["predicted_anomaly"] else "NOMINAL",
        "score_semantics": frozen["score_semantics"],
    }
    bundle = measure_ops_sat_probe_bundle(points, source_sha256=source_sha256)
    started = time.perf_counter()
    revision = backend.generate_structured(
        agent_id="ops-sat-compact-bundle-state-reviser",
        output_model=CompactBundleRevision,
        payload={
            "task": "Return a two-field shadow decision using only the supplied observations.",
            "baseline_candidate_states": ["NOMINAL", "ANOMALOUS"],
            "baseline_observation": baseline,
            "active_bundle_observation": bundle,
            "decision_meanings": {
                "KEEP_BOTH": "retain both candidates",
                "NOMINAL": "retain only NOMINAL",
                "ANOMALOUS": "retain only ANOMALOUS",
                "ABSTAIN": "recommend abstention and retain both candidates",
            },
            "rules": [
                "Truth labels are unavailable.",
                "Do not infer root cause or invent observations.",
                "Use ACTIVE_BUNDLE_ONLY or BOTH only when the bundle affects the decision.",
                "Return only the two schema fields.",
            ],
        },
    )
    candidates = _candidate_states(revision.decision)
    citations = _citations(revision.citation_scope, baseline["observation_id"], bundle["observation_id"])
    active_cited = bundle["observation_id"] in citations
    changed = set(candidates) != {"NOMINAL", "ANOMALOUS"}
    return {
        "arm": "A10_COMPACT_BUNDLE",
        "baseline_observation": baseline,
        "observation": bundle,
        "revision": {
            **revision.model_dump(mode="json"),
            "candidate_states": candidates,
            "abstain_recommended": revision.decision == "ABSTAIN",
            "cited_observation_ids": citations,
        },
        "revision_available": True,
        "validated_candidate_states": candidates,
        "candidate_set_changed": changed,
        "active_observation_exposed": True,
        "active_observation_cited": active_cited,
        "active_observation_attributable": active_cited and changed,
        "elapsed_s": time.perf_counter() - started,
        "usage": _usage(backend),
        "backend_call_history": backend.call_history,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--utility-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--model", default="qwen/qwen3.5-9b")
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.case_limit is not None and not 1 <= args.case_limit <= 24:
        parser.error("case-limit must be between 1 and 24")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output-dir must be new or empty")

    baseline = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    utility = json.loads(args.utility_report.read_text(encoding="utf-8"))
    if baseline.get("schema_version") != "astrograph-ops-sat-llm-active-shadow-report.v1":
        parser.error("baseline-report schema is unsupported")
    if utility.get("schema_version") != "astrograph-ops-sat-probe-utility-audit.v1":
        parser.error("utility-report schema is unsupported")
    if utility.get("best_candidate_id") != "ALL_REGISTERED_PROBES":
        raise ValueError("utility audit did not select the registered probe bundle")
    if utility.get("promotion_recommendation") != "ELIGIBLE_FOR_FURTHER_DEVELOPMENT_ONLY":
        raise ValueError("probe bundle did not pass the development-only utility gate")

    dataset_path = args.data_root / "dataset.csv"
    segments_path = args.data_root / "segments.csv"
    receipts = {
        "dataset.csv": {"sha256": _sha_file(dataset_path), "size": dataset_path.stat().st_size},
        "segments.csv": {"sha256": _sha_file(segments_path), "size": segments_path.stat().st_size},
    }
    if receipts != baseline["preflight_receipt"]["source_receipts"]:
        raise ValueError("OPS bundle comparator source identity differs from baseline")
    utility_receipts = utility["preflight_receipt"]["source_receipts"]
    if any(receipts[name] != utility_receipts[name] for name in receipts):
        raise ValueError("OPS bundle comparator source identity differs from utility audit")
    models = _models(args.url)
    if args.model != baseline["preflight_receipt"]["model"] or args.model not in models:
        raise ValueError("OPS bundle comparator model identity differs or is unavailable")

    feature, truth = load_feature_track(dataset_path)
    raw, raw_truth = load_raw_track(segments_path)
    if truth != raw_truth or set(feature) != set(raw):
        raise ValueError("OPS raw/feature identities differ")
    config = json.loads((ROOT / "configs/evaluation/r20_ops_sat.json").read_text(encoding="utf-8"))
    partition_by = {row["group_id"]: row["partition"] for row in build_partitions(truth, config)["groups"]}
    selected = list(baseline["preflight_receipt"]["selected_group_ids"])
    if args.case_limit is not None:
        selected = selected[:args.case_limit]
    if any(partition_by.get(group_id) != "DEVELOPMENT_SELECTION" for group_id in selected):
        raise ValueError("bundle comparator groups are outside development selection")

    preflight = {
        "status": "PASS",
        "arm": "A10_COMPACT_BUNDLE",
        "selected_group_ids": selected,
        "source_receipts": receipts,
        "baseline_report_file_sha256": _sha_file(args.baseline_report),
        "utility_report_file_sha256": _sha_file(args.utility_report),
        "utility_report_content_sha256": utility["report_sha256"],
        "model": args.model,
        "models": models,
        "external_validation_groups_consumed": False,
        "configured_max_tokens": 4096,
        "response_format_strategy": "plain_json",
        "repair_attempts": 0,
        "maximum_http_attempts_per_case": 1,
        "llm_calls_per_case": 1,
        "tool_actions_per_case": 1,
        "measurement_components_per_action": 6,
        "cost_comparability": "MECHANISM_ONLY_NOT_EQUAL_COMPUTE_OR_LLM_BUDGET",
    }
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with dataset_use("OPS_SAT", "DEVELOPMENT_SELECTION", "AUDIT", group_ids=selected) as admission:
        for index, group_id in enumerate(selected, start=1):
            print(f"case {index}/{len(selected)} group={group_id} arm=A10_COMPACT_BUNDLE", flush=True)
            frozen = ops_r21_frozen_remediation_prediction(raw[group_id])
            segment_sha = _digest(raw[group_id])
            try:
                arm = run_compact_bundle_arm(
                    points=raw[group_id], frozen=frozen, source_sha256=segment_sha,
                    url=args.url, model=args.model,
                )
            except Exception as exc:
                blocked = {
                    "schema_version": "astrograph-ops-sat-compact-bundle-blocked.v1",
                    "status": "BLOCKED_MODEL_REVISION_UNAVAILABLE",
                    "preflight_receipt": preflight,
                    "failed_group_id": group_id,
                    "completed_case_count": len(rows),
                    "error": f"{type(exc).__name__}: {exc}",
                    "truth_joined_for_failed_case": False,
                    "claim_boundary": "A10 stopped before scoring the failed case; this is not a diagnostic result.",
                }
                blocked["report_sha256"] = _digest(blocked)
                (args.output_dir / "blocked_report.json").write_text(
                    json.dumps(blocked, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
                raise
            rows.append({
                "group_id": group_id,
                "channel": raw[group_id][0]["channel"],
                "sample_count": len(raw[group_id]),
                "segment_sha256": segment_sha,
                "frozen_prediction": frozen,
                "arm": arm,
                "offline_truth": "ANOMALOUS" if truth[group_id]["anomaly"] else "NOMINAL",
            })

    decided = [
        row for row in rows
        if len(row["arm"]["validated_candidate_states"]) == 1
        and not row["arm"]["revision"]["abstain_recommended"]
    ]
    nominal = [row for row in rows if row["offline_truth"] == "NOMINAL"]
    correct = sum(row["arm"]["validated_candidate_states"][0] == row["offline_truth"] for row in decided)
    summary = {
        "case_count": len(rows),
        "decided_count": len(decided),
        "abstain_or_ambiguous_count": len(rows) - len(decided),
        "correct_decided_count": correct,
        "accuracy_with_abstention_as_incorrect": correct / len(rows),
        "healthy_false_alarm_count": sum(
            row["arm"]["validated_candidate_states"] == ["ANOMALOUS"]
            and not row["arm"]["revision"]["abstain_recommended"]
            for row in nominal
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
    report = {
        "schema_version": "astrograph-ops-sat-compact-bundle-comparator.v1",
        "status": "EXPLORATORY_DEVELOPMENT_RESULT",
        "preflight_receipt": preflight,
        "dataset_admission": admission,
        "a10_compact_bundle": summary,
        "baseline_arms": baseline["arms"],
        "cases": rows,
        "attributable_outcome_audit": summarize_attributable_outcomes(rows),
        "comparison": {
            "active_effect_identifiable": summary["active_observation_exposed_count"] > 0,
            "equal_tool_action_count_to_single_probe": True,
            "equal_measurement_compute_to_single_probe": False,
            "accuracy_delta_a10_minus_a8": (
                summary["accuracy_with_abstention_as_incorrect"]
                - baseline["arms"]["A8_ACTIVE_TELEMETRY"]["accuracy_with_abstention_as_incorrect"]
            ),
            "healthy_false_alarm_delta_a10_minus_a8": (
                summary["healthy_false_alarm_count"]
                - baseline["arms"]["A8_ACTIVE_TELEMETRY"]["healthy_false_alarm_count"]
            ),
            "abstain_or_ambiguous_delta_a10_minus_a8": (
                summary["abstain_or_ambiguous_count"]
                - baseline["arms"]["A8_ACTIVE_TELEMETRY"]["abstain_or_ambiguous_count"]
            ),
            "false_alarm_per_runtime_hour": None,
            "false_alarm_rate_reason": "NO_RELIABLE_CONTINUOUS_RUNTIME_DENOMINATOR",
        },
        "claim_boundary": (
            "Development-only compact bundle mechanism comparator on previously seen groups. "
            "Not equal compute, independent efficacy evidence, root-cause diagnosis, formal evidence, or qualification."
        ),
    }
    report["report_sha256"] = _digest(report)
    output = args.output_dir / "report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": summary, "report_sha256": report["report_sha256"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
