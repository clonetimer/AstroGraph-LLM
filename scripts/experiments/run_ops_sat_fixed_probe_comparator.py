#!/usr/bin/env python3
"""Fixed-probe comparator for the real OPS-SAT A2/A8 shadow experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from astrograph.evaluation.data_exposure import dataset_use
from astrograph.evaluation.ops_sat_adapter import build_partitions, load_feature_track, load_raw_track
from astrograph.evaluation.replication_r22 import ops_r21_frozen_remediation_prediction
from scripts.experiments.run_ops_sat_llm_active_shadow import (
    ShadowStateRevision,
    _backend,
    _digest,
    _models,
    _sha_file,
    _usage,
    measure_action,
)


FIXED_ACTION = "inspect_early_late_shift"


def run_fixed_arm(*, points: list[dict], frozen: dict, source_sha256: str,
                  url: str, model: str) -> dict:
    backend = _backend(url, model, max_tokens=1800, response_format_strategy="plain_json")
    baseline = {
        "observation_id": "OPSBASE-" + _digest({"segment": points[0]["segment"], "frozen": frozen})[:16].upper(),
        "detector": "R21_FROZEN_LENGTH_AWARE_FALLBACK_FEATURE_SET",
        "score": frozen["score"], "threshold": frozen["threshold"],
        "predicted_state": "ANOMALOUS" if frozen["predicted_anomaly"] else "NOMINAL",
        "score_semantics": frozen["score_semantics"], "formal_diagnosis_unchanged": True,
    }
    observation = measure_action(points, FIXED_ACTION, source_sha256=source_sha256)
    started = time.perf_counter()
    allowed_ids = [baseline["observation_id"], observation["observation_id"]]
    revision = backend.generate_structured(
        agent_id="ops-sat-fixed-probe-state-reviser",
        output_model=ShadowStateRevision,
        payload={
            "task": "Revise the non-voting binary candidate set or abstain using only supplied observations.",
            "baseline_candidate_states": ["NOMINAL", "ANOMALOUS"],
            "baseline_observation": baseline,
            "selected_observation": observation,
            "fixed_action_policy": {
                "action": FIXED_ACTION,
                "selection_owner": "DETERMINISTIC_HOST_COMPARATOR",
                "llm_action_selection_calls": 0,
            },
            "allowed_observation_ids": allowed_ids,
            "rules": [
                "candidate_states may contain only NOMINAL and ANOMALOUS.",
                "cited_observation_ids must come from allowed_observation_ids.",
                "Cite the telemetry observation if it affects the revision.",
                "OPS-SAT has no root-cause labels; do not produce a fault identity.",
                "This cannot alter the frozen detector or formal evidence.",
            ],
        },
    )
    cited_valid = set(revision.cited_observation_ids) <= set(allowed_ids)
    available = cited_valid
    candidates = sorted(revision.candidate_states) if available else ["ANOMALOUS", "NOMINAL"]
    active_cited = observation["observation_id"] in revision.cited_observation_ids
    return {
        "arm": "A9_FIXED_PROBE_POLICY",
        "fixed_action": FIXED_ACTION,
        "assessment": {
            "action": FIXED_ACTION,
            "selection_owner": "DETERMINISTIC_HOST_COMPARATOR",
            "llm_action_selection_calls": 0,
        },
        "assessment_valid": True,
        "observation": observation,
        "revision": revision.model_dump(mode="json"),
        "revision_available": available,
        "validated_candidate_states": candidates,
        "candidate_set_changed": set(candidates) != {"NOMINAL", "ANOMALOUS"},
        "active_observation_exposed": True,
        "active_observation_cited": active_cited,
        "active_observation_attributable": active_cited and set(candidates) != {"NOMINAL", "ANOMALOUS"},
        "elapsed_s": time.perf_counter() - started,
        "usage": _usage(backend),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--model", default="qwen/qwen3.5-9b")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output-dir must be new or empty")
    baseline = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    if baseline.get("schema_version") != "astrograph-ops-sat-llm-active-shadow-report.v1":
        parser.error("baseline-report schema is unsupported")
    dataset_path = args.data_root / "dataset.csv"; segments_path = args.data_root / "segments.csv"
    current_receipts = {
        "dataset.csv": {"sha256": _sha_file(dataset_path), "size": dataset_path.stat().st_size},
        "segments.csv": {"sha256": _sha_file(segments_path), "size": segments_path.stat().st_size},
    }
    if current_receipts != baseline["preflight_receipt"]["source_receipts"]:
        raise ValueError("OPS comparator source identity differs from baseline")
    if args.model != baseline["preflight_receipt"]["model"] or args.model not in _models(args.url):
        raise ValueError("OPS comparator model identity differs or is unavailable")
    feature, truth = load_feature_track(dataset_path)
    raw, raw_truth = load_raw_track(segments_path)
    if truth != raw_truth or set(feature) != set(raw):
        raise ValueError("OPS raw/feature identities differ")
    config = json.loads((ROOT / "configs/evaluation/r20_ops_sat.json").read_text(encoding="utf-8"))
    partition_by = {item["group_id"]: item["partition"] for item in build_partitions(truth, config)["groups"]}
    selected = list(baseline["preflight_receipt"]["selected_group_ids"])
    if any(partition_by.get(group_id) != "DEVELOPMENT_SELECTION" for group_id in selected):
        raise ValueError("baseline groups are outside development selection")
    preflight = {
        "status": "PASS", "fixed_action": FIXED_ACTION, "model": args.model,
        "selected_group_ids": selected, "source_receipts": current_receipts,
        "baseline_report_file_sha256": _sha_file(args.baseline_report),
        "baseline_report_content_sha256": baseline["report_sha256"],
        "external_validation_groups_consumed": False,
        "maximum_tool_steps_per_case": 2,
        "llm_calls_per_case": 1,
        "configured_max_tokens": 1800,
        "response_format_strategy": "plain_json",
        "cost_comparability": "MECHANISM_ONLY_NOT_EQUAL_LLM_CALL_OUTPUT_TOKEN_OR_TRANSPORT_STRATEGY",
    }
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2)); return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with dataset_use("OPS_SAT", "DEVELOPMENT_SELECTION", "AUDIT", group_ids=selected) as admission:
        for index, group_id in enumerate(selected, start=1):
            print(f"case {index}/{len(selected)} group={group_id} arm=A9_FIXED_PROBE_POLICY", flush=True)
            frozen = ops_r21_frozen_remediation_prediction(raw[group_id])
            segment_sha = _digest(raw[group_id])
            arm = run_fixed_arm(points=raw[group_id], frozen=frozen, source_sha256=segment_sha,
                                url=args.url, model=args.model)
            rows.append({
                "group_id": group_id, "channel": raw[group_id][0]["channel"],
                "sample_count": len(raw[group_id]), "segment_sha256": segment_sha,
                "frozen_prediction": frozen, "arm": arm,
                "offline_truth": "ANOMALOUS" if truth[group_id]["anomaly"] else "NOMINAL",
            })
    decided = [row for row in rows if len(row["arm"]["validated_candidate_states"]) == 1
               and not row["arm"]["revision"]["abstain_recommended"]]
    correct = sum(row["arm"]["validated_candidate_states"][0] == row["offline_truth"] for row in decided)
    nominal = [row for row in rows if row["offline_truth"] == "NOMINAL"]
    a9 = {
        "case_count": len(rows), "decided_count": len(decided),
        "abstain_or_ambiguous_count": len(rows) - len(decided), "correct_decided_count": correct,
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
    a8 = baseline["arms"]["A8_ACTIVE_TELEMETRY"]
    report = {
        "schema_version": "astrograph-ops-sat-fixed-probe-comparator-report.v1",
        "status": "EXPLORATORY_DEVELOPMENT_RESULT", "preflight_receipt": preflight,
        "dataset_admission": admission, "baseline_arms": baseline["arms"],
        "a9_fixed_probe": a9, "cases": rows,
        "comparison": {
            "accuracy_delta_a9_minus_a8": a9["accuracy_with_abstention_as_incorrect"] - a8["accuracy_with_abstention_as_incorrect"],
            "healthy_false_alarm_delta_a9_minus_a8": a9["healthy_false_alarm_count"] - a8["healthy_false_alarm_count"],
            "exposure_delta_a9_minus_a8": a9["active_observation_exposed_count"] - a8["active_observation_exposed_count"],
            "equal_llm_call_budget": a9["llm_call_count"] == a8["llm_call_count"],
            "false_alarm_per_runtime_hour": None,
            "false_alarm_rate_reason": "NO_RELIABLE_CONTINUOUS_RUNTIME_DENOMINATOR",
        },
        "claim_boundary": (
            "Fixed-action development comparator on previously seen OPS-SAT segments. It isolates action exposure, "
            "not independent diagnostic efficacy; no root-cause, formal-evidence, production, safety, or flight claim."
        ),
    }
    report["report_sha256"] = _digest(report)
    output = args.output_dir / "report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "a9_fixed_probe": a9,
                      "comparison": report["comparison"], "report_sha256": report["report_sha256"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
