#!/usr/bin/env python3
"""Compare LLM, information-gain and random active-observation policies on OPS-SAT.

This is a development-only binary anomaly study.  The action contracts are fit
only on MODEL_TRAIN groups, evaluation cases come only from DEVELOPMENT_SELECTION,
and truth is joined after every online arm returns.  No observation is promoted
to EvidenceFact and no root-cause identity is produced.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time
from typing import Literal
import urllib.error
import urllib.request

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from astrograph.evaluation.data_exposure import dataset_use
from astrograph.evaluation.ops_sat_adapter import build_partitions, load_feature_track, load_raw_track
from astrograph.evaluation.replication_r22 import ops_r21_frozen_remediation_prediction
from scripts.experiments.run_ops_sat_llm_active_shadow import (
    ACTIVE_ACTIONS,
    ShadowStateRevision,
    _backend,
    _digest,
    _models,
    _sha_file,
    _usage,
    measure_action,
)


class PolicyActionChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal[
        "inspect_early_late_shift",
        "inspect_volatility_ratio",
        "inspect_peak_concentration",
        "STOP_NO_NEW_OBSERVATION",
    ]
    reason: str = Field(min_length=1, max_length=400)


def server_runtime_receipt(url: str) -> dict:
    """Capture the OpenAI-compatible server version without making it a trust claim."""
    base = url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    endpoint = base + "/version"
    try:
        with urllib.request.urlopen(endpoint, timeout=10) as response:
            body = response.read(64 * 1024)
        payload = json.loads(body.decode("utf-8"))
        return {
            "status": "RECORDED",
            "endpoint": endpoint,
            "response": payload,
            "response_sha256": _digest(payload),
        }
    except (OSError, ValueError, urllib.error.URLError) as exc:
        return {
            "status": "UNAVAILABLE",
            "endpoint": endpoint,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _binary_entropy(positive: int, total: int) -> float:
    if total <= 0 or positive <= 0 or positive >= total:
        return 0.0
    probability = positive / total
    return -probability * math.log2(probability) - (1.0 - probability) * math.log2(1.0 - probability)


def _balanced_accuracy(truth: np.ndarray, predicted: np.ndarray) -> float:
    positive = truth
    negative = ~truth
    recall = float(np.mean(predicted[positive])) if np.any(positive) else 0.0
    specificity = float(np.mean(~predicted[negative])) if np.any(negative) else 0.0
    return (recall + specificity) / 2.0


def _information_gain(truth: np.ndarray, branch: np.ndarray) -> float:
    total = len(truth)
    before = _binary_entropy(int(np.sum(truth)), total)
    after = 0.0
    for selected in (branch, ~branch):
        count = int(np.sum(selected))
        if count:
            after += count / total * _binary_entropy(int(np.sum(truth[selected])), count)
    return before - after


def fit_action_contracts(
    raw: dict[str, list[dict]],
    truth: dict[str, dict],
    partition_by_group: dict[str, str],
) -> dict:
    """Fit one-dimensional action contracts using MODEL_TRAIN groups only."""
    training = sorted(
        (group_id for group_id in raw if partition_by_group[group_id] == "MODEL_TRAIN"),
        key=lambda value: int(value),
    )
    if not training:
        raise ValueError("no MODEL_TRAIN groups are available for action-policy fitting")
    labels = np.asarray([bool(truth[group_id]["anomaly"]) for group_id in training], dtype=bool)
    if len(set(labels.tolist())) != 2:
        raise ValueError("MODEL_TRAIN action-policy fitting requires both binary classes")
    contracts: dict[str, dict] = {}
    for action in ACTIVE_ACTIONS:
        values = np.asarray([
            float(measure_action(raw[group_id], action, source_sha256=_digest(raw[group_id]))["value"])
            for group_id in training
        ])
        unique = np.unique(values)
        if len(unique) == 1:
            thresholds = [float(unique[0])]
        else:
            thresholds = ((unique[:-1] + unique[1:]) / 2.0).tolist()
        candidates: list[tuple[float, float, float, str]] = []
        for threshold in thresholds:
            for direction in ("GE", "LT"):
                predicted = values >= threshold if direction == "GE" else values < threshold
                candidates.append((
                    _information_gain(labels, predicted),
                    _balanced_accuracy(labels, predicted),
                    -abs(float(threshold)),
                    direction,
                ))
        best = max(candidates)
        direction = best[3]
        # Recover the deterministic threshold selected by the complete tie-break key.
        selected_threshold = next(
            float(threshold)
            for threshold in thresholds
            if (
                _information_gain(labels, values >= threshold if direction == "GE" else values < threshold),
                _balanced_accuracy(labels, values >= threshold if direction == "GE" else values < threshold),
                -abs(float(threshold)),
                direction,
            ) == best
        )
        contracts[action] = {
            "action": action,
            "threshold": selected_threshold,
            "anomalous_when": direction,
            "training_information_gain_bits": best[0],
            "training_balanced_accuracy": best[1],
            "training_nominal_median": float(np.median(values[~labels])),
            "training_anomalous_median": float(np.median(values[labels])),
            "training_group_count": len(training),
        }
    selected_action = max(
        ACTIVE_ACTIONS,
        key=lambda action: (
            contracts[action]["training_information_gain_bits"],
            contracts[action]["training_balanced_accuracy"],
            -ACTIVE_ACTIONS.index(action),
        ),
    )
    return {
        "schema_version": "astrograph-ops-sat-action-contracts.v1",
        "fit_partition": "MODEL_TRAIN",
        "training_group_count": len(training),
        "training_group_ids_sha256": _digest(training),
        "contracts": contracts,
        "selected_action": selected_action,
        "selection_rule": "max information gain, then balanced accuracy, then registered action order",
        "evaluation_truth_used_for_fit": False,
    }


def contract_prediction(contract: dict, value: float) -> str:
    anomalous = (
        value >= float(contract["threshold"])
        if contract["anomalous_when"] == "GE"
        else value < float(contract["threshold"])
    )
    return "ANOMALOUS" if anomalous else "NOMINAL"


def _baseline_observation(points: list[dict], frozen: dict) -> dict:
    return {
        "observation_id": "OPSBASE-" + _digest({"segment": points[0]["segment"], "frozen": frozen})[:16].upper(),
        "score": frozen["score"],
        "threshold": frozen["threshold"],
        "predicted_state": "ANOMALOUS" if frozen["predicted_anomaly"] else "NOMINAL",
        "score_semantics": frozen["score_semantics"],
        "formal_diagnosis_unchanged": True,
    }


def run_deterministic_arm(
    *, arm: str, group_id: str, points: list[dict], frozen: dict,
    action: str | None, contracts: dict,
) -> dict:
    started = time.perf_counter()
    baseline = _baseline_observation(points, frozen)
    if action is None:
        return {
            "arm": arm,
            "selected_action": "NO_NEW_OBSERVATION",
            "observation": baseline,
            "validated_candidate_states": [baseline["predicted_state"]],
            "abstain_recommended": False,
            "active_observation_exposed": False,
            "active_observation_cited": False,
            "active_observation_attributable": False,
            "elapsed_s": time.perf_counter() - started,
            "usage": {"call_count": 0, "prompt_tokens": 0, "completion_tokens": 0},
        }
    observation = measure_action(points, action, source_sha256=_digest(points))
    state = contract_prediction(contracts["contracts"][action], float(observation["value"]))
    return {
        "arm": arm,
        "selected_action": action,
        "observation": observation,
        "validated_candidate_states": [state],
        "abstain_recommended": False,
        "active_observation_exposed": True,
        "active_observation_cited": True,
        "active_observation_attributable": True,
        "elapsed_s": time.perf_counter() - started,
        "usage": {"call_count": 0, "prompt_tokens": 0, "completion_tokens": 0},
        "group_id_binding": group_id,
    }


def stable_random_action(group_id: str) -> str:
    index = int(_digest({"policy": "stable-random-action.v1", "group_id": group_id})[:8], 16)
    return ACTIVE_ACTIONS[index % len(ACTIVE_ACTIONS)]


def run_llm_arm(
    *, group_id: str, points: list[dict], frozen: dict, contracts: dict,
    url: str, model: str,
) -> dict:
    backend = _backend(
        url, model, max_tokens=1200, response_format_strategy="json_schema",
        repair_attempts=0, max_http_attempts=2,
        provider="user-openai-compatible",
    )
    baseline = _baseline_observation(points, frozen)
    started = time.perf_counter()
    choice = backend.generate_structured(
        agent_id="ops-sat-action-policy-selector",
        output_model=PolicyActionChoice,
        payload={
            "task": "Choose one new observation or stop for a binary anomaly shadow investigation.",
            "operator_report": (
                "An automated ticket suspects an anomaly. Verify the premise; a healthy/no-anomaly "
                "conclusion and STOP_NO_NEW_OBSERVATION are valid."
            ),
            "baseline_observation": baseline,
            "allowed_actions": [*ACTIVE_ACTIONS, "STOP_NO_NEW_OBSERVATION"],
            "training_only_action_contracts": contracts["contracts"],
            "tool_budget": 1,
            "rules": [
                "Evaluation truth is unavailable.",
                "Choose at most one registered action.",
                "Prefer an action with expected discrimination; stop when no new observation is justified.",
                "Do not infer a root cause or invent an observation.",
            ],
        },
    )
    action_valid = choice.action in {*ACTIVE_ACTIONS, "STOP_NO_NEW_OBSERVATION"}
    selected_action = choice.action if action_valid else "STOP_NO_NEW_OBSERVATION"
    if selected_action == "STOP_NO_NEW_OBSERVATION":
        observation = baseline
        allowed_ids = [baseline["observation_id"]]
    else:
        observation = measure_action(points, selected_action, source_sha256=_digest(points))
        allowed_ids = [baseline["observation_id"], observation["observation_id"]]
    revision = backend.generate_structured(
        agent_id="ops-sat-action-policy-reviser",
        output_model=ShadowStateRevision,
        payload={
            "task": "Revise the shadow candidates or abstain using only the supplied observations.",
            "operator_report_is_unverified": True,
            "baseline_candidate_states": ["NOMINAL", "ANOMALOUS"],
            "baseline_observation": baseline,
            "selected_observation": observation,
            "selected_action_contract": (
                contracts["contracts"].get(selected_action)
                if selected_action != "STOP_NO_NEW_OBSERVATION" else None
            ),
            "allowed_observation_ids": allowed_ids,
            "rules": [
                "Evaluation truth is unavailable.",
                "Citations must be drawn from allowed_observation_ids.",
                "Abstention retains both candidates.",
                "A healthy/no-anomaly conclusion is valid.",
                "This shadow result cannot alter formal evidence or the frozen detector.",
            ],
        },
    )
    cited_valid = set(revision.cited_observation_ids) <= set(allowed_ids)
    revision_valid = revision.abstain_recommended or bool(revision.candidate_states)
    available = action_valid and cited_valid and revision_valid
    if not available or revision.abstain_recommended:
        candidates = ["ANOMALOUS", "NOMINAL"]
    else:
        candidates = sorted(revision.candidate_states)
    active_exposed = bool(observation.get("creates_new_raw_observation"))
    active_cited = active_exposed and observation["observation_id"] in revision.cited_observation_ids
    return {
        "arm": "LLM_ACTION_POLICY",
        "group_id_binding": group_id,
        "action": choice.model_dump(mode="json"),
        "action_valid": action_valid,
        "selected_action": selected_action,
        "observation": observation,
        "revision": revision.model_dump(mode="json"),
        "revision_available": available,
        "validated_candidate_states": candidates,
        "abstain_recommended": bool(revision.abstain_recommended),
        "active_observation_exposed": active_exposed,
        "active_observation_cited": active_cited,
        "active_observation_attributable": active_cited and set(candidates) != {"NOMINAL", "ANOMALOUS"},
        "elapsed_s": time.perf_counter() - started,
        "usage": _usage(backend),
        "backend_call_history": backend.call_history,
    }


def summarize(rows: list[dict], arm: str) -> dict:
    decided = [
        row for row in rows
        if len(row["arms"][arm]["validated_candidate_states"]) == 1
        and not row["arms"][arm]["abstain_recommended"]
    ]
    correct = sum(
        row["arms"][arm]["validated_candidate_states"][0] == row["offline_truth"]
        for row in decided
    )
    healthy = [row for row in rows if row["premise_status"] == "FALSE_PREMISE_HEALTHY"]
    return {
        "case_count": len(rows),
        "decided_count": len(decided),
        "correct_decided_count": correct,
        "accuracy_with_abstention_as_incorrect": correct / len(rows),
        "abstain_or_ambiguous_count": len(rows) - len(decided),
        "healthy_false_premise_case_count": len(healthy),
        "healthy_false_alarm_count": sum(
            row["arms"][arm]["validated_candidate_states"] == ["ANOMALOUS"]
            and not row["arms"][arm]["abstain_recommended"]
            for row in healthy
        ),
        "active_observation_exposed_count": sum(
            row["arms"][arm]["active_observation_exposed"] for row in rows
        ),
        "active_observation_attributable_count": sum(
            row["arms"][arm]["active_observation_attributable"] for row in rows
        ),
        "action_counts": dict(sorted(Counter(
            row["arms"][arm]["selected_action"] for row in rows
        ).items())),
        "elapsed_s": sum(row["arms"][arm]["elapsed_s"] for row in rows),
        "llm_call_count": sum(row["arms"][arm]["usage"]["call_count"] for row in rows),
        "prompt_tokens": sum(row["arms"][arm]["usage"]["prompt_tokens"] for row in rows),
        "completion_tokens": sum(row["arms"][arm]["usage"]["completion_tokens"] for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8081/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases-per-class", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.cases_per_class <= 12:
        parser.error("cases-per-class must be between 1 and 12")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output-dir must be new or empty")
    dataset_path = args.data_root / "dataset.csv"
    segments_path = args.data_root / "segments.csv"
    if not dataset_path.is_file() or not segments_path.is_file():
        parser.error("data-root must contain dataset.csv and segments.csv")
    models = _models(args.url)
    if args.model not in models:
        parser.error("requested vLLM model is not listed by /models")
    feature, truth = load_feature_track(dataset_path)
    raw, raw_truth = load_raw_track(segments_path)
    if truth != raw_truth or set(feature) != set(raw):
        raise ValueError("OPS raw/feature identities differ")
    config = json.loads((ROOT / "configs/evaluation/r20_ops_sat.json").read_text(encoding="utf-8"))
    partition_by = {
        row["group_id"]: row["partition"] for row in build_partitions(truth, config)["groups"]
    }
    contracts = fit_action_contracts(raw, truth, partition_by)
    eligible = sorted(
        (group_id for group_id in raw if partition_by[group_id] == "DEVELOPMENT_SELECTION"),
        key=lambda value: int(value),
    )
    selected: list[str] = []
    for label in (0, 1):
        selected.extend([
            group_id for group_id in eligible if int(bool(truth[group_id]["anomaly"])) == label
        ][:args.cases_per_class])
    if len(selected) != 2 * args.cases_per_class:
        raise ValueError("insufficient development-selection class coverage")
    preflight = {
        "status": "PASS",
        "dataset_family": "OPS-SAT",
        "policy_fit_partition": "MODEL_TRAIN",
        "evaluation_partition": "DEVELOPMENT_SELECTION",
        "selected_group_ids": selected,
        "source_receipts": {
            "dataset.csv": {"sha256": _sha_file(dataset_path), "size": dataset_path.stat().st_size},
            "segments.csv": {"sha256": _sha_file(segments_path), "size": segments_path.stat().st_size},
        },
        "model": args.model,
        "models": models,
        "server_runtime_receipt": server_runtime_receipt(args.url),
        "external_validation_groups_consumed": False,
        "evaluation_truth_visible_to_online_arms": False,
        "maximum_new_observations_per_case": 1,
        "action_contracts": contracts,
    }
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with dataset_use("OPS_SAT", "DEVELOPMENT_SELECTION", "AUDIT", group_ids=selected) as admission:
        for index, group_id in enumerate(selected, start=1):
            print(f"case {index}/{len(selected)} group={group_id}", flush=True)
            frozen = ops_r21_frozen_remediation_prediction(raw[group_id])
            deterministic_action = contracts["selected_action"]
            arms = {
                "FROZEN_NO_NEW_OBSERVATION": run_deterministic_arm(
                    arm="FROZEN_NO_NEW_OBSERVATION", group_id=group_id, points=raw[group_id],
                    frozen=frozen, action=None, contracts=contracts,
                ),
                "TRAIN_INFORMATION_GAIN_POLICY": run_deterministic_arm(
                    arm="TRAIN_INFORMATION_GAIN_POLICY", group_id=group_id, points=raw[group_id],
                    frozen=frozen, action=deterministic_action, contracts=contracts,
                ),
                "STABLE_RANDOM_LEGAL_ACTION": run_deterministic_arm(
                    arm="STABLE_RANDOM_LEGAL_ACTION", group_id=group_id, points=raw[group_id],
                    frozen=frozen, action=stable_random_action(group_id), contracts=contracts,
                ),
            }
            try:
                arms["LLM_ACTION_POLICY"] = run_llm_arm(
                    group_id=group_id, points=raw[group_id], frozen=frozen,
                    contracts=contracts, url=args.url, model=args.model,
                )
            except Exception as exc:
                blocked = {
                    "schema_version": "astrograph-ops-sat-action-policy-blocked.v1",
                    "status": "BLOCKED_MODEL_POLICY_UNAVAILABLE",
                    "preflight_receipt": preflight,
                    "failed_group_id": group_id,
                    "completed_case_count": len(rows),
                    "error": f"{type(exc).__name__}: {exc}",
                    "truth_joined_for_failed_case": False,
                    "claim_boundary": "The failed case was not scored; this is not a diagnostic result.",
                }
                blocked["report_sha256"] = _digest(blocked)
                (args.output_dir / "blocked_report.json").write_text(
                    json.dumps(blocked, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
                )
                raise
            # Truth is joined only after every online arm returns.
            offline_truth = "ANOMALOUS" if truth[group_id]["anomaly"] else "NOMINAL"
            rows.append({
                "group_id": group_id,
                "channel": raw[group_id][0]["channel"],
                "operator_report": "AUTOMATED_TICKET_SUSPECTS_ANOMALY",
                "premise_status": (
                    "SUPPORTED_PREMISE_ANOMALOUS" if offline_truth == "ANOMALOUS"
                    else "FALSE_PREMISE_HEALTHY"
                ),
                "arms": arms,
                "offline_truth": offline_truth,
            })
    arm_ids = [
        "FROZEN_NO_NEW_OBSERVATION", "TRAIN_INFORMATION_GAIN_POLICY",
        "STABLE_RANDOM_LEGAL_ACTION", "LLM_ACTION_POLICY",
    ]
    summaries = {arm: summarize(rows, arm) for arm in arm_ids}
    report = {
        "schema_version": "astrograph-ops-sat-action-policy-comparator.v1",
        "status": "EXPLORATORY_DEVELOPMENT_RESULT",
        "preflight_receipt": preflight,
        "dataset_admission": admission,
        "arm_metrics": summaries,
        "cases": rows,
        "comparison": {
            "same_evaluation_cases": True,
            "same_maximum_new_observation_budget": True,
            "llm_vs_information_gain_accuracy_delta": (
                summaries["LLM_ACTION_POLICY"]["accuracy_with_abstention_as_incorrect"]
                - summaries["TRAIN_INFORMATION_GAIN_POLICY"]["accuracy_with_abstention_as_incorrect"]
            ),
            "llm_vs_information_gain_healthy_false_alarm_delta": (
                summaries["LLM_ACTION_POLICY"]["healthy_false_alarm_count"]
                - summaries["TRAIN_INFORMATION_GAIN_POLICY"]["healthy_false_alarm_count"]
            ),
            "llm_active_effect_identifiable": (
                summaries["LLM_ACTION_POLICY"]["active_observation_exposed_count"] > 0
            ),
            "false_alarm_per_runtime_hour": None,
            "false_alarm_rate_reason": "NO_RELIABLE_CONTINUOUS_RUNTIME_DENOMINATOR",
        },
        "claim_boundary": (
            "Development-selection binary anomaly experiment with action contracts fit on MODEL_TRAIN. "
            "Not root-cause diagnosis, external validation, formal evidence, production qualification or a flight claim."
        ),
    }
    report["report_sha256"] = _digest(report)
    output = args.output_dir / "report.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output),
        "arm_metrics": summaries,
        "comparison": report["comparison"],
        "report_sha256": report["report_sha256"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
