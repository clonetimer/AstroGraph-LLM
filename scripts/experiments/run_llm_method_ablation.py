#!/usr/bin/env python3
"""Run a same-case live-vLLM pilot of the AstroGraph LLM method contribution.

This evaluates the event-level reasoning layer only. It does not retrain data-driven
models and must not be described as a full FD-LLM reproduction or confirmatory study.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from astrograph.data_driven.experience import ExperienceLedger, ExperienceRecord
from astrograph.evaluation import (
    LLMMethodAblationRow, annotate_net_diagnostic_effect, build_confirmatory_inference_runtime_snapshot,
    load_confirmatory_design, paired_arm_effect, summarize_llm_method_ablation,
)
from astrograph.llm import OpenAICompatibleConfig, OpenAICompatibleStructuredBackend
from astrograph.onboarding import sha256_file
from scripts.ai_native.run_llm_hybrid_ci_regression import (
    _decision,
    _invoke_chain,
    _load_rows,
    _sample_pair,
)
from scripts.experiments.fixed_probe_action_policy import FixedProbeActionBackend


ARM_ORDER = [
    "A0_ALGORITHM_ONLY",
    "A1_LATE_LLM",
    "A2_CONTROLLED_FULL",
    "A3_NO_KG_OBSERVATION",
    "A4_NO_RAG_OAG_CONTEXT",
    "A5_NO_EXPERIENCE_CONTEXT",
    "A6_NO_OBSERVATION_FEEDBACK",
    "A7_NO_ACTIVE_PROBE",
]

# Separate exploratory comparison; the historical eight-arm campaign is unchanged.
ACTIVE_TELEMETRY_ARM_ORDER = [
    "A0_ALGORITHM_ONLY",
    "A1_LATE_LLM",
    "A2_CONTROLLED_FULL",
    "A8_ACTIVE_TELEMETRY",
]
FIXED_PROBE_ARM_ID = "A9_FIXED_PROBE_POLICY"

INSPECTION_CAPABILITIES = [
    "inspect_evidence_status",
    "inspect_candidate_fusion",
    "inspect_kg_support",
    "inspect_retrieval_context",
    "inspect_experience_context",
    "inspect_deterministic_core",
    "inspect_verification_plan",
    "inspect_probe_catalog",
    "execute_diagnostic_probe",
]

CASES = [
    {
        "case_id": "ADCS-RW-JAM-TEST-PILOT",
        "dataset": "adcs",
        "dataset_class": "adcs_rw_jamming",
        "true_fault_id": "ADCS_RW_JAM",
    },
    {
        "case_id": "PHM-HEATER-STUCK-TEST-PILOT",
        "dataset": "phm",
        "dataset_class": "thermal_heater_stuck_on",
        "true_fault_id": "THERMAL_HEATER_STUCK_ON",
    },
]

def _load_case_manifest(
    path: Path, *, expected_campaign_id: str | None = None, expected_case_count: int | None = None
) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "astrograph-llm-method-case-manifest.v1":
        raise ValueError("unsupported LLM method case manifest schema")
    if expected_campaign_id is not None and payload.get("campaign_id") != expected_campaign_id:
        raise ValueError("case manifest campaign_id differs from frozen confirmatory design")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("case manifest must contain cases")
    seen = set()
    normalized = []
    for item in cases:
        required = ("case_id", "dataset", "dataset_class", "true_fault_id", "sample_id", "group_id")
        if not all(str(item.get(key, "")).strip() for key in required):
            raise ValueError("case manifest entry is incomplete")
        if item["dataset"] not in {"adcs", "phm"}:
            raise ValueError("case manifest dataset must be adcs or phm")
        if item["case_id"] in seen:
            raise ValueError("duplicate case_id in case manifest")
        seen.add(item["case_id"]); normalized.append(dict(item))
    declared_count = payload.get("case_count")
    if declared_count != len(normalized):
        raise ValueError("case manifest case_count does not match cases")
    if expected_case_count is not None and len(normalized) != expected_case_count:
        raise ValueError("case manifest denominator differs from frozen confirmatory design")
    return normalized



def _inference_snapshot(*, url: str, model: str, inference_profile: Path | None, provider: str) -> dict:
    snapshot = build_confirmatory_inference_runtime_snapshot(
        vllm_url=url, vllm_model=model, profile_path=inference_profile
    )
    if provider == snapshot["provider"]:
        return snapshot
    # Only the exploratory route may adapt the frozen numerical settings to a
    # different OpenAI-compatible provider. Never present it as the R16 runtime.
    adapted = {key: value for key, value in snapshot.items() if key != "inference_config_sha256"}
    adapted.update({
        "provider": provider,
        "profile_adaptation": "EXPLORATORY_PROVIDER_OVERRIDE_NOT_CONFIRMATORY",
        # The observed LM Studio endpoint rejects json_object but accepts
        # json_schema. Keep this transport choice explicit in the run receipt.
        **({"response_format_strategy": "json_schema"} if provider == "lmstudio" else {}),
    })
    adapted["inference_config_sha256"] = hashlib.sha256(
        json.dumps(adapted, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return adapted


def _backend(
    url: str, model: str, inference_profile: Path | None = None, *, provider: str = "vllm",
) -> OpenAICompatibleStructuredBackend:
    snapshot = _inference_snapshot(
        url=url, model=model, inference_profile=inference_profile, provider=provider,
    )
    config = OpenAICompatibleConfig(
        base_url=snapshot["base_url"], provider=snapshot["provider"], model=snapshot["model"], temperature=snapshot["temperature"],
        max_tokens=snapshot["configured_max_tokens"], timeout_s=snapshot["timeout_s"], response_format_strategy=snapshot["response_format_strategy"],
        include_reference_hypothesis_set=snapshot["include_reference_hypothesis_set"], prompt_profile=snapshot["prompt_profile"], repair_attempts=snapshot["repair_attempts"],
        seed=snapshot["seed"], candidate_top_k=snapshot["candidate_top_k"], candidate_top_k_mode=snapshot["candidate_top_k_mode"], candidate_top_k_min=snapshot["candidate_top_k_min"],
        candidate_top_k_max=snapshot["candidate_top_k_max"], budget_preflight=snapshot["budget_preflight"], budget_policy=snapshot["budget_policy"], max_tokens_ceiling=snapshot["max_tokens_ceiling"],
        context_length=snapshot["context_length"], detect_finish_reason_length=snapshot["detect_finish_reason_length"], tokenizer_mode=snapshot["tokenizer_mode"], tokenizer_path=snapshot["tokenizer_path"],
        tokenizer_local_files_only=snapshot["tokenizer_local_files_only"], tokenizer_trust_remote_code=snapshot["tokenizer_trust_remote_code"], prompt_token_safety_margin=snapshot["prompt_token_safety_margin"], prompt_token_reserve=snapshot["prompt_token_reserve"],
    )
    return OpenAICompatibleStructuredBackend(config)


def _arm_overrides(arm_id: str, experience_root: Path, *, active_exploratory: bool = False) -> dict:
    common = {
        "pre_reasoning_experience_ledger_path": str(experience_root),
        "llm_diagnostic_loop_max_steps": 4,
    }
    if active_exploratory and arm_id in {"A2_CONTROLLED_FULL", "A8_ACTIVE_TELEMETRY", FIXED_PROBE_ARM_ID}:
        common["llm_shadow_candidate_revision_enabled"] = True
    if arm_id == "A0_ALGORITHM_ONLY":
        return common
    if arm_id == "A1_LATE_LLM":
        return {**common, "llm_diagnostic_loop_enabled": False}
    if arm_id == "A2_CONTROLLED_FULL":
        return common
    if arm_id == "A3_NO_KG_OBSERVATION":
        return {
            **common,
            "llm_diagnostic_capability_ids": [
                item for item in INSPECTION_CAPABILITIES if item != "inspect_kg_support"
            ],
        }
    if arm_id == "A4_NO_RAG_OAG_CONTEXT":
        return {
            **common,
            "llm_retrieval_context_enabled": False,
            "llm_diagnostic_capability_ids": [
                item for item in INSPECTION_CAPABILITIES
                if item not in {"inspect_retrieval_context", "inspect_experience_context"}
            ],
        }
    if arm_id == "A5_NO_EXPERIENCE_CONTEXT":
        return {
            **common,
            "llm_experience_context_enabled": False,
            "llm_diagnostic_capability_ids": [
                item for item in INSPECTION_CAPABILITIES
                if item != "inspect_experience_context"
            ],
        }
    if arm_id == "A6_NO_OBSERVATION_FEEDBACK":
        return {**common, "llm_diagnostic_observation_feedback_enabled": False}
    if arm_id == "A7_NO_ACTIVE_PROBE":
        return {
            **common,
            "llm_diagnostic_capability_ids": [
                item for item in INSPECTION_CAPABILITIES
                if item not in {"inspect_probe_catalog", "execute_diagnostic_probe"}
            ],
        }
    if arm_id in {"A8_ACTIVE_TELEMETRY", FIXED_PROBE_ARM_ID}:
        return {**common, "llm_active_telemetry_observation_enabled": True}
    raise ValueError(f"unknown arm: {arm_id}")


def _make_experience(root: Path) -> None:
    ledger = ExperienceLedger(root)
    ledger.record(ExperienceRecord(
        experience_id="EXP-LLM-ABLATION-PILOT",
        source_cycle_id="llm-method-ablation-preregistered-pilot",
        dataset_id="cross-dataset-method-pilot",
        proposal_summary="Compare paired reference and mechanism-consistent evidence before acceptance.",
        decision="WITHHELD",
    ))
    ledger.backfill_outcome(
        "EXP-LLM-ABLATION-PILOT",
        outcome="confirmed engineering lesson",
        lesson="Cross-check paired residual evidence, KG support and deterministic core before synthesis.",
    )
    ledger.approve_reuse(
        "EXP-LLM-ABLATION-PILOT",
        reviewer="project-owner-approved-experiment",
        fault_ids=["ADCS_RW_JAM", "THERMAL_HEATER_STUCK_ON"],
    )


def _token_counts(backend: OpenAICompatibleStructuredBackend | None) -> tuple[int, int]:
    prompt = 0
    completion = 0
    for call in backend.call_history if backend is not None else []:
        usage = ((call.get("output_budget") or {}).get("server_usage") or {})
        prompt += int(usage.get("prompt_tokens", 0))
        completion += int(usage.get("completion_tokens", 0))
    return prompt, completion


def _select_nominal_controls(
    rows: list, class_attr: str, *, limit: int = 2, sample_index: int | None = None,
) -> list:
    """Pick disclosed, deterministic healthy controls without reading their labels at inference time."""
    if limit < 1:
        raise ValueError("nominal control limit must be positive")
    controls = sorted(
        (
            row for row in rows
            if getattr(row, "split", None) == "test" and getattr(row, class_attr) == "nominal"
            and (sample_index is None or getattr(row, "sample_index", None) == sample_index)
        ),
        key=lambda row: str(row.sample_id),
    )
    if not controls:
        raise ValueError("exploratory active telemetry comparison requires healthy test controls")
    return controls[:limit]


def _select_pilot_fault_sample(candidates: list, *, case: dict, exploratory: bool):
    """Freeze a stable pilot identity before any model inference."""
    if not candidates:
        raise ValueError(f"pilot fault case has no test sample: {case['case_id']}")
    if not exploratory and len(candidates) != 1:
        raise ValueError(f"frozen case must resolve to exactly one test sample: {case['case_id']} ({len(candidates)})")
    selected = sorted(candidates, key=lambda item: str(item.sample_id))[0]
    if len(candidates) > 1 and sum(item.sample_id == selected.sample_id for item in candidates) != 1:
        raise ValueError(f"pilot sample identity is ambiguous: {case['case_id']}")
    return selected


def _nominal_control_metrics(rows: list[dict], *, arm_order: list[str]) -> dict:
    """Use all selected healthy cases as the denominator, including review/abstention."""
    if not rows:
        raise ValueError("healthy control rows are required")
    by_arm = {arm: [row for row in rows if row["arm_id"] == arm] for arm in arm_order}
    case_sets = [{row["case_id"] for row in items} for items in by_arm.values()]
    if not case_sets[0] or any(cases != case_sets[0] for cases in case_sets):
        raise ValueError("healthy controls must have identical cases in every arm")
    if any(len(items) != len(case_sets[0]) for items in by_arm.values()):
        raise ValueError("healthy controls contain duplicate case/arm rows")
    return {
        arm: {
            "case_count": len(items),
            "formal_false_alarm_count": sum(bool(row["accepted_fault_id"]) for row in items),
            "formal_false_alarm_rate": round(
                sum(bool(row["accepted_fault_id"]) for row in items) / len(items), 6
            ),
            "human_review_count": sum(bool(row["requires_human_review"]) for row in items),
            "mean_latency_s": round(sum(float(row.get("latency_s", 0)) for row in items) / len(items), 6),
            "prompt_tokens": sum(int(row.get("prompt_tokens", 0)) for row in items),
            "completion_tokens": sum(int(row.get("completion_tokens", 0)) for row in items),
            "valid_tool_call_count": sum(int(row.get("valid_tool_call_count", 0)) for row in items),
            "active_telemetry_exposed_case_count": sum(
                int(row.get("new_raw_observation_count", 0)) > 0 for row in items
            ),
            "shadow_revision_available_case_count": sum(
                row.get("shadow_revision_available") is True for row in items
            ),
            "shadow_candidate_set_changed_case_count": sum(
                row.get("shadow_candidate_set_changed") is True for row in items
            ),
            "active_telemetry_attributable_case_count": sum(
                row.get("active_telemetry_attributable") is True for row in items
            ),
            "shadow_abstain_recommendation_count": sum(
                row.get("shadow_abstain_recommended") is True for row in items
            ),
        }
        for arm, items in by_arm.items()
    }


def _verify_exploratory_input_copy(
    *, work: Path, case_id: str, arm_id: str, source_sha256: str, nominal_sha256: str,
) -> None:
    copied = work / case_id / arm_id
    if (
        sha256_file(copied / "run" / "telemetry.csv") != source_sha256
        or sha256_file(copied / "nominal" / "telemetry.csv") != nominal_sha256
    ):
        raise RuntimeError(f"exploratory arm input changed for {case_id}/{arm_id}")


def _preflight_active_experiment(
    *, vllm_url: str, vllm_model: str, inference_profile: Path, provider: str = "vllm",
    sample_index: int | None = None,
) -> dict:
    """Check external inputs and model availability before creating run artifacts."""
    if not inference_profile.is_file():
        raise ValueError(f"inference profile is unavailable: {inference_profile}")
    datasets = {}
    for dataset in ("adcs", "phm"):
        rows, profile_id, dataset_id, class_attr = _load_rows(dataset)
        controls = _select_nominal_controls(
            rows, class_attr, limit=1 if sample_index is not None else 2,
            sample_index=sample_index,
        )
        fault_cases = [case for case in CASES if case["dataset"] == dataset]
        for case in fault_cases:
            matches = [
                item for item in rows
                if getattr(item, "split", None) == "test"
                and getattr(item, class_attr) == case["dataset_class"]
                and (not case.get("sample_id") or getattr(item, "sample_id", None) == case["sample_id"])
                and (sample_index is None or getattr(item, "sample_index", None) == sample_index)
            ]
            selected = _select_pilot_fault_sample(matches, case=case, exploratory=True)
            sample, nominal = _sample_pair(selected, rows, dataset)
            for item in (sample, nominal):
                if not Path(item.telemetry_path).is_file():
                    raise ValueError(f"pilot telemetry is unavailable: {item.telemetry_path}")
        for item in controls:
            if not Path(item.telemetry_path).is_file():
                raise ValueError(f"healthy control telemetry is unavailable: {item.telemetry_path}")
        datasets[dataset] = {
            "dataset_id": dataset_id, "profile_id": profile_id,
            "test_nominal_count": len(controls),
            "pilot_fault_case_count": len(fault_cases),
        }
    endpoint = vllm_url.rstrip("/") + "/models"
    with urllib.request.urlopen(endpoint, timeout=5) as response:
        models = json.load(response).get("data", [])
    if vllm_model not in {item.get("id") for item in models if isinstance(item, dict)}:
        raise ValueError(f"model {vllm_model!r} is not served by {endpoint}")
    return {
        "datasets": datasets, "model_id": vllm_model, "model_endpoint": endpoint,
        "provider": provider, "exploratory_sample_index": sample_index,
    }


def _row_from_result(*, case: dict, arm_id: str, result, elapsed: float, backend):
    decision = _decision(result)
    predicted = decision.get("accepted_fault_id")
    payload = result.output_payload or {}
    trajectory = payload.get("llm_diagnostic_trajectory") or {}
    steps = trajectory.get("steps", [])
    valid_calls = sum(
        item.get("policy_status") in {"EXECUTED_READ_ONLY", "EXECUTED_SHADOW_PROBE", "ACCEPTED"}
        for item in steps
    )
    rejected_calls = sum(str(item.get("policy_status", "")).startswith("REJECTED") for item in steps)
    audit = trajectory.get("substance_audit") or {}
    contribution = trajectory.get("contribution_audit") or {}
    shadow = payload.get("shadow_candidate_revision") or {}
    runtime_meta = (payload.get("llm_runtime_snapshot") or {}).get("runtime_snapshot") or {}
    probe_catalog_meta = runtime_meta.get("active_telemetry_probe_catalog") or {}
    candidate_membership = None
    if backend is not None:
        recorded_sets = [
            set(str(x) for x in (call.get("selected_candidate_fault_ids") or []))
            for call in backend.call_history
            if "selected_candidate_fault_ids" in call
        ]
        if recorded_sets:
            candidate_membership = any(case["true_fault_id"] in values for values in recorded_sets)
    prompt_tokens, completion_tokens = _token_counts(backend)
    return LLMMethodAblationRow(
        case_id=case["case_id"],
        dataset_id=(
            "whole_spacecraft_adcs_paired_dataset_v5_2_2"
            if case["dataset"] == "adcs"
            else "whole_spacecraft_phm_dataset_v6_0"
        ),
        group_id=str(case.get("group_id") or case["case_id"]),
        arm_id=arm_id,
        true_fault_id=case["true_fault_id"],
        predicted_fault_id=predicted,
        abstained=predicted is None,
        latency_s=elapsed,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        valid_tool_call_count=valid_calls,
        rejected_tool_call_count=rejected_calls,
        observation_conditioned_replan_count=int(
            audit.get("observation_conditioned_replan_count", 0)
        ),
        evidence_citation_valid=rejected_calls == 0,
        substance_status=str(audit.get("status", "NOT_EXECUTED")),
        true_fault_in_candidate_set=candidate_membership,
        probe_execution_count=int(contribution.get("probe_execution_count", 0)),
        new_observation_count=int(contribution.get("new_observation_count", 0)),
        new_raw_observation_count=int(contribution.get("new_raw_observation_count", 0)),
        formal_evidence_generated_count=int(contribution.get("formal_evidence_generated_count", 0)),
        resolved_gap_count=int(contribution.get("resolved_gap_count", 0)),
        diagnostic_contribution_status=str(contribution.get("status", "NOT_EXECUTED")),
        shadow_revision_available=shadow.get("status") == "AVAILABLE",
        shadow_candidate_set_changed=bool(shadow.get("candidate_set_changed", False)),
        shadow_abstain_recommended=shadow.get("abstain_recommended"),
        active_telemetry_attributable=bool(shadow.get("active_telemetry_attributable", False)),
        shadow_revision_reason_code=str(shadow.get("reason_code", "NOT_EXECUTED")),
        true_fault_in_shadow_candidate_set=(
            case["true_fault_id"] in shadow.get("revised_candidate_fault_ids", [])
            if shadow.get("status") == "AVAILABLE" else None
        ),
        active_probe_available_count=int(probe_catalog_meta.get("available_probe_count", 0)),
        active_probe_admission_status=str(probe_catalog_meta.get("admission_status", "NOT_CONFIGURED")),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--vllm-model", default="Qwen3.5-9B")
    parser.add_argument("--provider", choices=("vllm", "lmstudio"), default="vllm")
    parser.add_argument("--report-name", default="report.json")
    parser.add_argument("--case-manifest", type=Path, help="Frozen astrograph-llm-method-case-manifest.v1")
    parser.add_argument("--inference-profile", type=Path, help="Frozen R16 OpenAI-compatible inference profile JSON.")
    parser.add_argument(
        "--active-telemetry-exploratory", action="store_true",
        help="Run a separate same-budget A0/A1/A2/A8 research comparison; not the frozen confirmatory campaign.",
    )
    parser.add_argument("--preflight-only", action="store_true", help="Validate active pilot inputs without writing output.")
    parser.add_argument(
        "--fixed-probe-comparator", action="store_true",
        help="Add exploratory A9: fixed host-ordered probe action policy with the same loop limit and final LLM.",
    )
    parser.add_argument(
        "--exploratory-sample-index", type=int,
        help="Select one test Monte Carlo sample index per dataset for an additional exploratory seed check.",
    )
    args = parser.parse_args()
    if args.active_telemetry_exploratory and args.case_manifest:
        parser.error("active telemetry exploration cannot reuse the frozen confirmatory case manifest")
    if args.fixed_probe_comparator and not args.active_telemetry_exploratory:
        parser.error("fixed probe comparator requires active telemetry exploration")
    if args.provider != "vllm" and not args.active_telemetry_exploratory:
        parser.error("non-vLLM provider is allowed only for the separate exploratory campaign")
    arm_order = (
        [*ACTIVE_TELEMETRY_ARM_ORDER, FIXED_PROBE_ARM_ID]
        if args.fixed_probe_comparator else
        ACTIVE_TELEMETRY_ARM_ORDER if args.active_telemetry_exploratory else ARM_ORDER
    )
    if args.preflight_only and not args.active_telemetry_exploratory:
        parser.error("--preflight-only requires --active-telemetry-exploratory")
    if args.exploratory_sample_index is not None and (
        not args.active_telemetry_exploratory or args.exploratory_sample_index < 0
    ):
        parser.error("--exploratory-sample-index requires active exploration and a nonnegative index")
    if Path(args.report_name).name != args.report_name or not args.report_name.endswith(".json"):
        parser.error("report-name must be a plain JSON filename")
    if args.output_dir.exists() and not args.output_dir.is_dir():
        parser.error("output-dir points to a file")
    if not args.preflight_only and args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output-dir must be new or empty; existing experiment artifacts are never overwritten")
    design_for_profile = load_confirmatory_design(ROOT)
    inference_profile = (args.inference_profile.resolve() if args.inference_profile else (ROOT / str(design_for_profile["inference_profile_path"])).resolve())
    preflight = None
    if args.active_telemetry_exploratory:
        preflight = _preflight_active_experiment(
            vllm_url=args.vllm_url, vllm_model=args.vllm_model,
            inference_profile=inference_profile, provider=args.provider,
            sample_index=args.exploratory_sample_index,
        )
        if args.preflight_only:
            print(json.dumps(preflight, ensure_ascii=False, indent=2))
            return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    work = args.output_dir / "work"
    work.mkdir(parents=True)
    experience_root = work / "experience"
    _make_experience(experience_root)

    rows: list[LLMMethodAblationRow] = []
    nominal_rows: list[dict] = []
    exploratory_sources: list[dict] = []
    exploratory_policies: dict[str, dict[str, list[dict]]] = {}
    trajectories: dict[str, dict[str, dict]] = {}
    nominal_trajectories: dict[str, dict[str, dict]] = {}
    nominal_shadow_revisions: dict[str, dict[str, dict]] = {}
    fixed_policy_action_counts: dict[str, int] = {}
    failures: list[dict] = []
    if args.case_manifest:
        design = load_confirmatory_design(ROOT)
        cases = _load_case_manifest(
            args.case_manifest,
            expected_campaign_id=design["campaign_id"],
            expected_case_count=int(design["case_selection"]["expected_case_count"]),
        )
    else:
        cases = (
            [
                {**case, "case_id": f"{case['case_id']}-MC{args.exploratory_sample_index:02d}"}
                for case in CASES
            ]
            if args.exploratory_sample_index is not None else CASES
        )
    for case in cases:
        all_rows, profile_id, dataset_id, class_attr = _load_rows(case["dataset"])
        candidates = [
            item for item in all_rows
            if getattr(item, "split", None) == "test"
            and getattr(item, class_attr) == case["dataset_class"]
            and (not case.get("sample_id") or getattr(item, "sample_id", None) == case["sample_id"])
            and (
                args.exploratory_sample_index is None
                or getattr(item, "sample_index", None) == args.exploratory_sample_index
            )
        ]
        selected = _select_pilot_fault_sample(
            candidates, case=case, exploratory=args.active_telemetry_exploratory,
        )
        sample, nominal = _sample_pair(selected, all_rows, case["dataset"])
        source_sha = nominal_sha = None
        if args.active_telemetry_exploratory:
            source_sha = sha256_file(Path(sample.telemetry_path))
            nominal_sha = sha256_file(Path(nominal.telemetry_path))
            exploratory_sources.append({
                "case_id": case["case_id"],
                "dataset_id": dataset_id,
                "sample_id": sample.sample_id,
                "nominal_sample_id": nominal.sample_id,
                "source_sha256": source_sha,
                "nominal_sha256": nominal_sha,
                "role": "FAULT",
            })
        if case.get("nominal_sample_id") and getattr(nominal, "sample_id", None) != case["nominal_sample_id"]:
            raise RuntimeError(f"frozen nominal identity mismatch: {case['case_id']}")
        for arm_id in arm_order:
            backend = None if arm_id == "A0_ALGORITHM_ONLY" else _backend(
                args.vllm_url, args.vllm_model, inference_profile, provider=args.provider
            )
            if arm_id == FIXED_PROBE_ARM_ID:
                backend = FixedProbeActionBackend(backend)
            started = time.perf_counter()
            try:
                result = _invoke_chain(
                    work / case["case_id"] / arm_id,
                    "rule" if backend is None else "llm",
                    backend,
                    sample,
                    nominal,
                    profile_id=profile_id,
                    dataset_id=dataset_id,
                    episode_tag=f"{case['case_id']}-{arm_id}",
                    input_overrides=_arm_overrides(
                        arm_id, experience_root, active_exploratory=args.active_telemetry_exploratory
                    ),
                )
                if result.status.value != "SUCCEEDED":
                    error = result.error.model_dump(mode="json") if result.error else {}
                    raise RuntimeError(
                        f"workflow status {result.status.value}: "
                        f"{error.get('code', 'NO_ERROR_CODE')}: {error.get('message', '')}"
                    )
                if args.active_telemetry_exploratory:
                    if source_sha is None or nominal_sha is None:
                        raise RuntimeError("exploratory source identity is missing")
                    _verify_exploratory_input_copy(
                        work=work, case_id=case["case_id"], arm_id=arm_id,
                        source_sha256=source_sha, nominal_sha256=nominal_sha,
                    )
                elapsed = round(time.perf_counter() - started, 6)
                rows.append(_row_from_result(
                    case=case, arm_id=arm_id, result=result, elapsed=elapsed, backend=backend
                ))
                if args.active_telemetry_exploratory:
                    exploratory_policies.setdefault(case["case_id"], {})[arm_id] = (
                        (result.output_payload or {}).get("runtime_policy_receipts") or []
                    )
                trajectories.setdefault(case["case_id"], {})[arm_id] = (
                    (result.output_payload or {}).get("llm_diagnostic_trajectory") or {}
                )
                if isinstance(backend, FixedProbeActionBackend):
                    fixed_policy_action_counts[case["case_id"]] = backend.fixed_action_count
                print(
                    f"[{case['case_id']}] {arm_id}: "
                    f"{rows[-1].predicted_fault_id or 'ABSTAIN'} {elapsed:.3f}s",
                    flush=True,
                )
            except Exception as exc:
                failures.append({
                    "case_id": case["case_id"],
                    "arm_id": arm_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                })
                print(f"[{case['case_id']}] {arm_id}: FAILED {type(exc).__name__}", flush=True)

    if args.active_telemetry_exploratory and failures:
        (args.output_dir / "failed_attempts.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        raise RuntimeError(f"ablation has {len(failures)} failed fault-arm executions; healthy controls were not run")

    if args.active_telemetry_exploratory:
        for dataset in ("adcs", "phm"):
            all_rows, profile_id, dataset_id, class_attr = _load_rows(dataset)
            for sample in _select_nominal_controls(
                all_rows, class_attr,
                limit=1 if args.exploratory_sample_index is not None else 2,
                sample_index=args.exploratory_sample_index,
            ):
                sample_digest = hashlib.sha256(str(sample.sample_id).encode("utf-8")).hexdigest()[:12]
                case_id = f"NOMINAL-{dataset}-{sample_digest}"
                source_sha = sha256_file(Path(sample.telemetry_path))
                exploratory_sources.append({
                    "case_id": case_id,
                    "dataset_id": dataset_id,
                    "source_sha256": source_sha,
                    "nominal_sha256": source_sha,
                    "role": "NOMINAL_CONTROL",
                })
                for arm_id in arm_order:
                    backend = None if arm_id == "A0_ALGORITHM_ONLY" else _backend(
                        args.vllm_url, args.vllm_model, inference_profile, provider=args.provider
                    )
                    if arm_id == FIXED_PROBE_ARM_ID:
                        backend = FixedProbeActionBackend(backend)
                    started = time.perf_counter()
                    try:
                        result = _invoke_chain(
                            work / case_id / arm_id,
                            "rule" if backend is None else "llm",
                            backend,
                            sample,
                            sample,
                            profile_id=profile_id,
                            dataset_id=dataset_id,
                            episode_tag=f"{case_id}-{arm_id}",
                            input_overrides=_arm_overrides(
                                arm_id, experience_root, active_exploratory=True
                            ),
                        )
                        _verify_exploratory_input_copy(
                            work=work, case_id=case_id, arm_id=arm_id,
                            source_sha256=source_sha, nominal_sha256=source_sha,
                        )
                        if result.status.value != "SUCCEEDED":
                            error = result.error.model_dump(mode="json") if result.error else {}
                            raise RuntimeError(
                                f"nominal control workflow status {result.status.value}: "
                                f"{error.get('code', 'NO_ERROR_CODE')}: {error.get('message', '')}"
                            )
                        decision = _decision(result)
                        exploratory_policies.setdefault(case_id, {})[arm_id] = (
                            (result.output_payload or {}).get("runtime_policy_receipts") or []
                        )
                        trajectory = (result.output_payload or {}).get("llm_diagnostic_trajectory") or {}
                        nominal_trajectories.setdefault(case_id, {})[arm_id] = trajectory
                        if isinstance(backend, FixedProbeActionBackend):
                            fixed_policy_action_counts[case_id] = backend.fixed_action_count
                        contribution = trajectory.get("contribution_audit") or {}
                        shadow = (result.output_payload or {}).get("shadow_candidate_revision") or {}
                        nominal_shadow_revisions.setdefault(case_id, {})[arm_id] = shadow
                        prompt_tokens, completion_tokens = _token_counts(backend)
                        valid_tool_calls = sum(
                            step.get("policy_status") in {"EXECUTED_READ_ONLY", "EXECUTED_SHADOW_PROBE", "ACCEPTED"}
                            for step in trajectory.get("steps", [])
                        )
                        nominal_rows.append({
                            "case_id": case_id,
                            "dataset_id": dataset_id,
                            "sample_id": sample.sample_id,
                            "arm_id": arm_id,
                            "accepted_fault_id": decision.get("accepted_fault_id"),
                            "requires_human_review": decision.get("requires_human_review"),
                            "latency_s": round(time.perf_counter() - started, 6),
                            "new_raw_observation_count": int(contribution.get("new_raw_observation_count", 0)),
                            "shadow_revision_available": shadow.get("status") == "AVAILABLE",
                            "shadow_candidate_set_changed": shadow.get("candidate_set_changed") is True,
                            "active_telemetry_attributable": shadow.get("active_telemetry_attributable") is True,
                            "shadow_abstain_recommended": shadow.get("abstain_recommended"),
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "valid_tool_call_count": valid_tool_calls,
                        })
                    except Exception as exc:
                        failures.append({
                            "case_id": case_id, "arm_id": arm_id,
                            "error_type": type(exc).__name__, "error": str(exc)[:1000],
                        })
                        print(f"[{case_id}] {arm_id}: FAILED {type(exc).__name__}", flush=True)

    if failures:
        (args.output_dir / "failed_attempts.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        raise RuntimeError(f"ablation has {len(failures)} failed arm executions")
    if args.active_telemetry_exploratory:
        for case_id, receipts_by_arm in exploratory_policies.items():
            if set(receipts_by_arm) != set(arm_order):
                raise RuntimeError(f"runtime policy receipts missing an arm for {case_id}")
            serialized = {
                json.dumps(receipts, sort_keys=True, ensure_ascii=False)
                for receipts in receipts_by_arm.values()
            }
            if len(serialized) != 1:
                raise RuntimeError(f"runtime policy or threshold binding differs between arms for {case_id}")
    rows = annotate_net_diagnostic_effect(rows, baseline_arm_id="A0_ALGORITHM_ONLY")
    report = summarize_llm_method_ablation(rows, expected_arm_ids=arm_order)
    if args.fixed_probe_comparator:
        report["paired_comparisons"]["A8_ACTIVE_TELEMETRY__vs__A9_FIXED_PROBE_POLICY"] = (
            paired_arm_effect(
                rows,
                baseline_arm_id="A8_ACTIVE_TELEMETRY",
                challenger_arm_id=FIXED_PROBE_ARM_ID,
            )
        )
    report.update({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": args.provider,
        "model": args.vllm_model,
        "base_url": args.vllm_url,
        "inference_runtime": _inference_snapshot(
            url=args.vllm_url, model=args.vllm_model,
            inference_profile=inference_profile, provider=args.provider,
        ),
        "experiment_scope": (
            "ACTIVE_TELEMETRY_FIXED_PROBE_COMPARATOR_EXPLORATORY"
            if args.fixed_probe_comparator else
            "ACTIVE_TELEMETRY_TWO_CASE_EXPLORATORY_PILOT"
            if args.active_telemetry_exploratory else
            "FROZEN_CONFIRMATORY_CASE_MANIFEST" if args.case_manifest else
            "TWO_CASE_PREREGISTERED_LIVE_MODEL_PILOT"
        ),
        "case_manifest": str(args.case_manifest) if args.case_manifest else None,
        **({"fixed_probe_comparator": {
            "policy_id": FixedProbeActionBackend.policy_id,
            "fixed_action_counts_by_case": fixed_policy_action_counts,
            "same_max_steps_and_probe_catalog_as_a8": True,
            "llm_action_selection_calls": 0,
            "final_llm_reasoning_preserved": True,
            "claim_boundary": (
                "A9 is a fixed host action baseline, not another LLM planner. The maximum tool budget "
                "matches A8, while realized tool calls and LLM planning costs may differ."
            ),
        }} if args.fixed_probe_comparator else {}),
        **({
            "nominal_controls": {
                "selection": (
                    f"first_test_nominal_sample_per_dataset_with_sample_index_{args.exploratory_sample_index}"
                    if args.exploratory_sample_index is not None else
                    "up_to_first_two_test_nominal_samples_per_dataset_by_sample_id"
                ),
                "rows": nominal_rows,
                "trajectories": nominal_trajectories,
                "shadow_candidate_revisions": nominal_shadow_revisions,
                "arm_metrics": _nominal_control_metrics(nominal_rows, arm_order=arm_order),
                "false_alarms_per_runtime_hour": None,
                "runtime_rate_unavailable_reason": "Pilot records case outcomes, not a validated continuous monitoring duration denominator.",
                "metric_semantics": (
                    "formal_false_alarm_rate counts accepted non-null fault IDs over every selected healthy "
                    "case. Human review is reported separately and is not scored as a formal false alarm."
                ),
            },
            "exploratory_metric_contract": {
                "study_status": "EXPLORATORY_NOT_CONFIRMATORY",
                "task": "same-case event-level fault identification plus healthy formal false alarms",
                "truth_visibility": "offline scoring only; labels and case IDs are not model observations",
                "selection": (
                    f"lexicographically first test fault and nominal sample per dataset at sample_index={args.exploratory_sample_index}"
                    if args.exploratory_sample_index is not None else
                    "lexicographically first test sample per declared fault class plus up to two test nominal samples per dataset"
                ),
                "split": "test, selected by dataset index before model inference",
                "inference_unit": "case_id; grouped confirmatory inference is not claimed",
                "denominator": "all selected cases per arm, including abstention and human review",
                "metric_algorithm": "astrograph-llm-method-ablation-report.v2 plus nominal-control formal false alarms",
                "source_receipts": exploratory_sources,
                "runtime_policy_receipts_by_case_arm": exploratory_policies,
                "promotion_allowed": False,
                "preflight_receipt": preflight,
            },
        } if args.active_telemetry_exploratory else {}),
        "trajectories": trajectories,
        "information_boundary": {
            "A3": "Removes direct KG observation from the LLM capability catalog; upstream BiRRF still contains governed KG rankings.",
            "A4": "Removes the entire pre-generation retrieval package from both deliberation and final LLM hypothesis generation.",
            "A5": "Removes approved experience hits while retaining document and ontology retrieval.",
            "A6": "Hides observation content during action selection; the final synthesis still receives the completed trajectory.",
            "A7": "Removes inspect_probe_catalog and execute_diagnostic_probe while retaining the remaining controlled diagnostic loop.",
            **({"A8": "Adds only admitted runtime telemetry probes to A2 with the same four-step LLM budget; "
                      "the probes remain shadow observations."} if args.active_telemetry_exploratory else {}),
            **({"A9": "Uses the same admitted probes and four-step loop as A8, but a fixed host action "
                      "selector replaces LLM action calls; final hypothesis generation remains LLM-based. "
                      "Maximum budgets match, realized calls and tokens need not."} if args.fixed_probe_comparator else {}),
        },
        "not_executed": [
            "Exact FD-LLM reproduction: requires its numerical time-series encoding/training protocol and is not represented by late AstroGraph prompting.",
            "Confirmatory all-class/group inference: requires a separately frozen cost and multiple-comparison budget after this pilot.",
        ],
    })
    report_path = args.output_dir / args.report_name
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    shutil.rmtree(work)
    print(f"[report] {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
