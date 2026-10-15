#!/usr/bin/env python3
"""Development-only A2/A8 shadow investigation on real OPS-SAT CSV data.

The frozen detector and its prediction are unchanged.  Truth is joined only after
both LLM arms finish.  This script never emits a formal EvidenceFact or a root-
cause claim because OPS-SAT supplies binary anomaly labels, not fault identities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import urllib.request
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from astrograph.evaluation.data_exposure import dataset_use
from astrograph.evaluation.ops_sat_adapter import build_partitions, load_feature_track, load_raw_track
from astrograph.evaluation.replication_r22 import ops_r21_frozen_remediation_prediction
from astrograph.llm import OpenAICompatibleConfig, OpenAICompatibleStructuredBackend


ACTIVE_ACTIONS = (
    "inspect_early_late_shift",
    "inspect_volatility_ratio",
    "inspect_peak_concentration",
)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class ActionChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal[
        "inspect_baseline",
        "inspect_early_late_shift",
        "inspect_volatility_ratio",
        "inspect_peak_concentration",
    ]
    reason: str = Field(min_length=1, max_length=400)


class ShadowStateRevision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_states: list[Literal["NOMINAL", "ANOMALOUS"]] = Field(min_length=0, max_length=2)
    abstain_recommended: bool
    cited_observation_ids: list[str] = Field(min_length=1, max_length=4)
    rationale: str = Field(min_length=1, max_length=600)

    @model_validator(mode="after")
    def unique_values(self) -> "ShadowStateRevision":
        if len(self.candidate_states) != len(set(self.candidate_states)):
            raise ValueError("candidate states must be unique")
        if len(self.cited_observation_ids) != len(set(self.cited_observation_ids)):
            raise ValueError("observation IDs must be unique")
        return self


def measure_action(points: list[dict], action: str, *, source_sha256: str) -> dict:
    """Execute one bounded, truth-free measurement over an admitted raw segment."""
    values = np.asarray([float(item["value"]) for item in points], dtype=float)
    if values.size < 4 or not np.isfinite(values).all():
        raise ValueError("OPS_ACTIVE_MEASUREMENT_REQUIRES_FINITE_SEGMENT")
    split = values.size // 2
    early, late = values[:split], values[split:]
    scale = max(float(np.std(values, ddof=1)), abs(float(np.median(values))) * 1e-9, 1e-12)
    if action == "inspect_early_late_shift":
        value = (float(np.mean(late)) - float(np.mean(early))) / scale
        semantics = "late-minus-early mean divided by full-segment sample standard deviation"
    elif action == "inspect_volatility_ratio":
        value = float(np.std(late, ddof=1)) / max(float(np.std(early, ddof=1)), 1e-12)
        semantics = "late-half sample standard deviation divided by early-half sample standard deviation"
    elif action == "inspect_peak_concentration":
        center = float(np.median(values))
        mad = float(np.median(np.abs(values - center)))
        robust_scale = max(1.4826 * mad, scale * 1e-6, 1e-12)
        early_rate = float(np.mean(np.abs(early - center) / robust_scale >= 3.0))
        late_rate = float(np.mean(np.abs(late - center) / robust_scale >= 3.0))
        value = late_rate - early_rate
        semantics = "late-minus-early fraction of samples beyond three robust standard deviations"
    elif action == "inspect_baseline":
        raise ValueError("baseline inspection is supplied by the host and is not a raw telemetry measurement")
    else:
        raise ValueError("unregistered OPS active measurement")
    binding = {
        "action": action,
        "segment": str(points[0]["segment"]),
        "channel": str(points[0]["channel"]),
        "source_sha256": source_sha256,
        "sample_count": int(values.size),
        "value": value,
        "semantics": semantics,
    }
    return {
        "observation_id": "OPSOBS-" + _digest(binding)[:16].upper(),
        **binding,
        "creates_new_raw_observation": True,
        "formal_evidence_generated": False,
    }


def _backend(
    url: str, model: str, *, max_tokens: int = 700,
    response_format_strategy: str = "json_schema",
    repair_attempts: int = 1, max_http_attempts: int = 6,
    provider: str = "lmstudio",
) -> OpenAICompatibleStructuredBackend:
    return OpenAICompatibleStructuredBackend(OpenAICompatibleConfig(
        base_url=url,
        provider=provider,
        model=model,
        temperature=0.0,
        max_tokens=max_tokens,
        timeout_s=180.0,
        response_format_strategy=response_format_strategy,
        include_reference_hypothesis_set=False,
        prompt_profile="evidence_protocol_v059_compact",
        repair_attempts=repair_attempts,
        max_http_attempts=max_http_attempts,
        seed=42,
        budget_preflight=False,
        tokenizer_mode="disabled",
    ))


def _usage(backend: OpenAICompatibleStructuredBackend) -> dict:
    prompt = completion = 0
    for call in backend.call_history:
        usage = ((call.get("output_budget") or {}).get("server_usage") or {})
        prompt += int(usage.get("prompt_tokens", 0)); completion += int(usage.get("completion_tokens", 0))
    return {"call_count": len(backend.call_history), "prompt_tokens": prompt, "completion_tokens": completion}


def _run_arm(*, arm: str, points: list[dict], frozen: dict, source_sha256: str,
             url: str, model: str) -> dict:
    backend = _backend(url, model)
    baseline = {
        "observation_id": "OPSBASE-" + _digest({"segment": points[0]["segment"], "frozen": frozen})[:16].upper(),
        "detector": "R21_FROZEN_LENGTH_AWARE_FALLBACK_FEATURE_SET",
        "score": frozen["score"], "threshold": frozen["threshold"],
        "predicted_state": "ANOMALOUS" if frozen["predicted_anomaly"] else "NOMINAL",
        "score_semantics": frozen["score_semantics"],
        "formal_diagnosis_unchanged": True,
    }
    allowed = ["inspect_baseline"] if arm == "A2_CONTROLLED_FULL" else ["inspect_baseline", *ACTIVE_ACTIONS]
    started = time.perf_counter()
    choice = backend.generate_structured(
        agent_id="ops-sat-shadow-action-selector",
        output_model=ActionChoice,
        payload={
            "task": "Choose exactly one allowed next inspection for a binary anomaly shadow investigation.",
            "arm": arm,
            "allowed_actions": allowed,
            "baseline_observation": baseline,
            "channel": points[0]["channel"],
            "sample_count": len(points),
            "rules": ["Do not infer a root cause.", "Do not invent observations.", "Truth labels are unavailable."],
        },
    )
    action_valid = choice.action in allowed
    selected_action = choice.action if action_valid else "inspect_baseline"
    observation = baseline if selected_action == "inspect_baseline" else measure_action(
        points, selected_action, source_sha256=source_sha256,
    )
    allowed_ids = [baseline["observation_id"], observation["observation_id"]]
    revision = backend.generate_structured(
        agent_id="ops-sat-shadow-state-reviser",
        output_model=ShadowStateRevision,
        payload={
            "task": "Revise the non-voting binary candidate set or abstain using only supplied observations.",
            "baseline_candidate_states": ["NOMINAL", "ANOMALOUS"],
            "baseline_observation": baseline,
            "selected_observation": observation,
            "allowed_observation_ids": allowed_ids,
            "rules": [
                "candidate_states may contain only NOMINAL and ANOMALOUS.",
                "cited_observation_ids must come from allowed_observation_ids.",
                "This cannot alter the frozen detector or formal evidence.",
                "OPS-SAT has no root-cause labels; do not produce a fault identity.",
            ],
        },
    )
    cited_valid = set(revision.cited_observation_ids) <= set(allowed_ids)
    available = action_valid and cited_valid
    candidates = sorted(revision.candidate_states) if available else ["ANOMALOUS", "NOMINAL"]
    active_exposed = bool(observation.get("creates_new_raw_observation"))
    active_cited = active_exposed and observation["observation_id"] in revision.cited_observation_ids
    return {
        "arm": arm,
        "action": choice.model_dump(mode="json"),
        "action_valid": action_valid,
        "observation": observation,
        "revision": revision.model_dump(mode="json"),
        "revision_available": available,
        "validated_candidate_states": candidates,
        "candidate_set_changed": set(candidates) != {"NOMINAL", "ANOMALOUS"},
        "active_observation_exposed": active_exposed,
        "active_observation_cited": active_cited,
        "active_observation_attributable": active_cited and set(candidates) != {"NOMINAL", "ANOMALOUS"},
        "elapsed_s": time.perf_counter() - started,
        "usage": _usage(backend),
    }


def _models(url: str) -> list[str]:
    with urllib.request.urlopen(url.rstrip("/") + "/models", timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return sorted(str(item["id"]) for item in payload.get("data", []))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--model", default="qwen/qwen3.5-9b")
    parser.add_argument("--cases-per-class", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.cases_per_class < 1 or args.cases_per_class > 12:
        parser.error("cases-per-class must be between 1 and 12")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output-dir must be new or empty")
    dataset_path = args.data_root / "dataset.csv"; segments_path = args.data_root / "segments.csv"
    if not dataset_path.is_file() or not segments_path.is_file():
        parser.error("data-root must contain dataset.csv and segments.csv")
    models = _models(args.url)
    if args.model not in models:
        parser.error("requested LM Studio model is not listed by /models")
    feature, truth = load_feature_track(dataset_path)
    raw, raw_truth = load_raw_track(segments_path)
    if truth != raw_truth or set(feature) != set(raw):
        raise ValueError("OPS raw/feature identities differ")
    config = json.loads((ROOT / "configs/evaluation/r20_ops_sat.json").read_text(encoding="utf-8"))
    partitions = build_partitions(truth, config)
    partition_by = {item["group_id"]: item["partition"] for item in partitions["groups"]}
    eligible = [sid for sid in sorted(raw, key=lambda value: int(value)) if partition_by[sid] == "DEVELOPMENT_SELECTION"]
    selected: list[str] = []
    for label in (0, 1):
        selected.extend([sid for sid in eligible if truth[sid]["anomaly"] == label][:args.cases_per_class])
    if len(selected) != 2 * args.cases_per_class:
        raise ValueError("insufficient development-selection class coverage")
    preflight = {
        "status": "PASS", "dataset_family": "OPS-SAT", "partition": "DEVELOPMENT_SELECTION",
        "purpose": "AUDIT", "selected_group_ids": selected, "models": models, "model": args.model,
        "source_receipts": {
            "dataset.csv": {"sha256": _sha_file(dataset_path), "size": dataset_path.stat().st_size},
            "segments.csv": {"sha256": _sha_file(segments_path), "size": segments_path.stat().st_size},
        },
        "external_validation_groups_consumed": False,
    }
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2)); return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with dataset_use("OPS_SAT", "DEVELOPMENT_SELECTION", "AUDIT", group_ids=selected) as admission:
        for case_index, sid in enumerate(selected, start=1):
            frozen = ops_r21_frozen_remediation_prediction(raw[sid])
            segment_sha = _digest(raw[sid])
            arms = {}
            for arm in ("A2_CONTROLLED_FULL", "A8_ACTIVE_TELEMETRY"):
                print(f"case {case_index}/{len(selected)} group={sid} arm={arm}", flush=True)
                arms[arm] = _run_arm(
                    arm=arm, points=raw[sid], frozen=frozen, source_sha256=segment_sha,
                    url=args.url, model=args.model,
                )
            rows.append({
                "group_id": sid, "channel": raw[sid][0]["channel"], "sample_count": len(raw[sid]),
                "segment_sha256": segment_sha, "frozen_prediction": frozen,
                "arms": arms,
                # Offline join occurs only after both arms have returned.
                "offline_truth": "ANOMALOUS" if truth[sid]["anomaly"] else "NOMINAL",
            })
    summary = {}
    for arm in ("A2_CONTROLLED_FULL", "A8_ACTIVE_TELEMETRY"):
        decided = [row for row in rows if len(row["arms"][arm]["validated_candidate_states"]) == 1
                   and not row["arms"][arm]["revision"]["abstain_recommended"]]
        correct = sum(row["arms"][arm]["validated_candidate_states"][0] == row["offline_truth"] for row in decided)
        nominal = [row for row in rows if row["offline_truth"] == "NOMINAL"]
        false_alarms = sum(row["arms"][arm]["validated_candidate_states"] == ["ANOMALOUS"]
                           and not row["arms"][arm]["revision"]["abstain_recommended"] for row in nominal)
        summary[arm] = {
            "case_count": len(rows), "decided_count": len(decided), "abstain_or_ambiguous_count": len(rows) - len(decided),
            "correct_decided_count": correct,
            "accuracy_with_abstention_as_incorrect": correct / len(rows),
            "healthy_false_alarm_count": false_alarms, "healthy_case_count": len(nominal),
            "active_observation_exposed_count": sum(row["arms"][arm]["active_observation_exposed"] for row in rows),
            "active_observation_attributable_count": sum(row["arms"][arm]["active_observation_attributable"] for row in rows),
            "elapsed_s": sum(row["arms"][arm]["elapsed_s"] for row in rows),
            "prompt_tokens": sum(row["arms"][arm]["usage"]["prompt_tokens"] for row in rows),
            "completion_tokens": sum(row["arms"][arm]["usage"]["completion_tokens"] for row in rows),
            "llm_call_count": sum(row["arms"][arm]["usage"]["call_count"] for row in rows),
        }
    report = {
        "schema_version": "astrograph-ops-sat-llm-active-shadow-report.v1",
        "status": "EXPLORATORY_DEVELOPMENT_RESULT",
        "preflight_receipt": preflight, "dataset_admission": admission,
        "arms": summary, "cases": rows,
        "frozen_detector": {
            "case_count": len(rows),
            "correct_count": sum(
                bool(row["frozen_prediction"]["predicted_anomaly"])
                == (row["offline_truth"] == "ANOMALOUS") for row in rows
            ),
            "healthy_false_alarm_count": sum(
                row["offline_truth"] == "NOMINAL"
                and bool(row["frozen_prediction"]["predicted_anomaly"]) for row in rows
            ),
            "anomaly_detection_count": sum(
                row["offline_truth"] == "ANOMALOUS"
                and bool(row["frozen_prediction"]["predicted_anomaly"]) for row in rows
            ),
        },
        "comparison": {
            "accuracy_delta_a8_minus_a2": summary["A8_ACTIVE_TELEMETRY"]["accuracy_with_abstention_as_incorrect"] - summary["A2_CONTROLLED_FULL"]["accuracy_with_abstention_as_incorrect"],
            "healthy_false_alarm_delta_a8_minus_a2": summary["A8_ACTIVE_TELEMETRY"]["healthy_false_alarm_count"] - summary["A2_CONTROLLED_FULL"]["healthy_false_alarm_count"],
            "active_effect_identifiable": summary["A8_ACTIVE_TELEMETRY"]["active_observation_exposed_count"] > 0,
            "false_alarm_per_runtime_hour": None,
            "false_alarm_rate_reason": "NO_RELIABLE_CONTINUOUS_RUNTIME_DENOMINATOR",
        },
        "claim_boundary": (
            "Development-selection reanalysis of a previously seen OPS-SAT dataset. Binary anomaly states only; "
            "no root-cause diagnosis, independent validation, formal EvidenceFact, production promotion, or flight claim."
        ),
    }
    report["report_sha256"] = _digest(report)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output_dir / "report.json"), "summary": summary,
                      "comparison": report["comparison"], "report_sha256": report["report_sha256"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
