#!/usr/bin/env python3
"""AI-Native CI regression: whole-spacecraft llm/hybrid evaluation without vLLM.

Runs the whole-spacecraft online diagnosis chain in rule and llm modes on real
ADCS v5.2.2 / PHM v6.0 telemetry, compares DecisionGovernor outcomes, and
emits decision-consistency plus hallucination-rejection (fail-closed)
statistics as baseline evidence for the AI-Native roadmap.

Usage:
    ASTROGRAPH_ADCS_V522_DATA_ROOT=<root> \
        python scripts/ai_native/run_llm_hybrid_ci_regression.py --dataset adcs --output-dir out
    ASTROGRAPH_PHM_V60_DATA_ROOT=<root> \
        python scripts/ai_native/run_llm_hybrid_ci_regression.py --dataset phm --output-dir out
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from astrograph.orchestration.models import (
    ExecutionPolicy,
    ExecutionStatus,
    PermissionScope,
    TaskEnvelope,
    WorkflowContext,
)
from astrograph.orchestration.whole_spacecraft_online_workflow import (
    WholeSpacecraftOnlineDiagnosisWorkflow,
)
from astrograph.dataset_profiles import DatasetProfileRegistry
from astrograph.policy_profiles import RuntimePolicyRegistry
from astrograph.services import ReferenceStructuredReasoningBackend
from astrograph.simulation import SimulationRunCompleted

ADCS_DATASET = Path(os.environ.get(
    "ASTROGRAPH_ADCS_V522_DATA_ROOT",
    str(ROOT / "data" / "external" / "whole_spacecraft_adcs_paired_dataset_v5_2_2"),
))
PHM_DATASET = Path(os.environ.get(
    "ASTROGRAPH_PHM_V60_DATA_ROOT",
    str(ROOT / "data" / "external" / "whole_spacecraft_phm_dataset_v6_0"),
))
DATASET_PROFILE_REGISTRY = ROOT / "configs/dataset_profiles/registry.json"
RUNTIME_POLICY_REGISTRY = ROOT / "configs/runtime_policies/registry.json"

DEVIATION_MODES = ("corrupt_evidence", "drop_all_faults")


def _permissions():
    return [
        PermissionScope.READ_DATA,
        PermissionScope.READ_KNOWLEDGE,
        PermissionScope.RUN_DIAGNOSIS,
        PermissionScope.WRITE_AUDIT,
        PermissionScope.WRITE_ARTIFACTS,
    ]


def _invoke_chain(
    tmp_path: Path, mode: str, backend, sample, nominal, *, profile_id: str, dataset_id: str,
    episode_tag: str | None = None,
    input_overrides: dict | None = None,
):
    run_dir = tmp_path / "run"
    nominal_dir = tmp_path / "nominal"
    run_dir.mkdir(parents=True)
    nominal_dir.mkdir(parents=True)
    shutil.copy2(sample.telemetry_path, run_dir / "telemetry.csv")
    shutil.copy2(nominal.telemetry_path, nominal_dir / "telemetry.csv")
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "schema_version": "astrograph-ai-native-runtime.v1", "scenario": "",
        "events": [], "event_count": 0, "row_count": 301,
        "context_of_use": "synthetic whole-spacecraft AI-Native CI regression",
    }), encoding="utf-8")
    tag = episode_tag or mode
    event = SimulationRunCompleted(
        event_id=f"WS-AI-NATIVE-{tag}", run_id=f"ws-ai-native-{tag}",
        run_dir=str(run_dir), nominal_run_dir=str(nominal_dir),
        metadata={"source": dataset_id, "split": "test", "episode_tag": episode_tag},
    )
    permissions = _permissions()
    envelope = TaskEnvelope(
        task_id=f"WS-AI-NATIVE-{tag}", task_type="simulation.online_diagnosis",
        input_payload={
            "simulation_event": event.model_dump(mode="json"),
            "dataset_profile_id": profile_id,
            "artifact_output_dir": str(tmp_path / "artifacts"),
            "shadow_mode": True,
            "reasoner_mode": "llm" if backend is not None else "rule",
            **(input_overrides or {}),
        },
        execution_policy=ExecutionPolicy(
            deterministic_only=backend is None, allow_llm=backend is not None,
        ),
        permission_scope=permissions,
    )
    context = WorkflowContext(
        workflow_id="simulation.whole_spacecraft.online_diagnosis.v0.5.6",
        run_id=f"ws-ai-native-{mode}", permissions=permissions,
    )
    return WholeSpacecraftOnlineDiagnosisWorkflow(
        reasoning_backend=backend,
        dataset_profile_registry=DatasetProfileRegistry(index_path=DATASET_PROFILE_REGISTRY),
        runtime_policy_registry=RuntimePolicyRegistry(RUNTIME_POLICY_REGISTRY),
    ).invoke(envelope, context)


def _decision(result):
    payload = result.output_payload or {}
    record = payload.get("decision_record") or {}
    return {
        "disposition": record.get("disposition"),
        "accepted_fault_id": record.get("accepted_fault_id"),
        "requires_human_review": record.get("requires_human_review"),
    }


def _reasoning_meta(result):
    payload = result.output_payload or {}
    hypothesis_set = payload.get("hypothesis_set") or {}
    metadata = hypothesis_set.get("validation_metadata") or {}
    return {
        "repair_attempt_count": metadata.get("repair_attempt_count", 0),
        "repair_succeeded": metadata.get("repair_succeeded"),
        "degraded_to_reference": metadata.get("degraded_to_reference", False),
    }


def _load_rows(dataset: str):
    if dataset == "adcs":
        from scripts.integration import run_adcs_v522_regression as adcs
        rows = adcs.load_trajectories(ADCS_DATASET)
        return rows, "adcs.v5_2_2", "whole_spacecraft_adcs_paired_dataset_v5_2_2", "diagnostic_class"
    if dataset == "phm":
        from astrograph.data_driven.phm_v60 import load_records
        rows = load_records(PHM_DATASET)
        return rows, "phm.v6_0", "whole_spacecraft_phm_dataset_v6_0", "condition_id"
    raise ValueError(f"unknown dataset: {dataset}")


def _sample_pair(row, rows, dataset: str):
    if dataset == "adcs":
        nominal = next(
            r for r in rows if r.diagnostic_class == "nominal" and r.trajectory_pair_id == row.trajectory_pair_id
        )
    else:
        nominal = next(
            r for r in rows
            if r.condition_id == "nominal" and r.subsystem == row.subsystem and r.sample_index == row.sample_index
        )
    return row, nominal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("adcs", "phm"), default="adcs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-classes", type=int, default=0, help="0 = all classes")
    args = parser.parse_args()
    rows, profile_id, dataset_id, class_attr = _load_rows(args.dataset)
    test_rows = [r for r in rows if getattr(r, "split", None) == "test" and getattr(r, class_attr) != "nominal"]
    classes = sorted({getattr(r, class_attr) for r in test_rows})
    if args.max_classes:
        classes = classes[: args.max_classes]
    workdir = args.output_dir / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    per_class: list[dict] = []
    consistent_count = 0
    reject_count = 0
    deviation_run_count = 0
    for fault_class in classes:
        row = next(r for r in test_rows if getattr(r, class_attr) == fault_class)
        sample, nominal = _sample_pair(row, rows, args.dataset)
        base = workdir / (fault_class.replace(":", "_") + "-" + uuid.uuid4().hex[:6])
        rule = _invoke_chain(base / "rule", "rule", None, sample, nominal,
                             profile_id=profile_id, dataset_id=dataset_id)
        llm = _invoke_chain(base / "llm", "llm", ReferenceStructuredReasoningBackend(), sample, nominal,
                            profile_id=profile_id, dataset_id=dataset_id)
        rule_ok = rule.status == ExecutionStatus.SUCCEEDED
        llm_ok = llm.status == ExecutionStatus.SUCCEEDED
        rule_decision = _decision(rule) if rule_ok else {"disposition": "FAILED"}
        llm_decision = _decision(llm) if llm_ok else {"disposition": "FAILED"}
        consistent = rule_decision == llm_decision
        consistent_count += int(consistent)
        entry: dict = {
            class_attr: fault_class,
            "sample_id": sample.sample_id,
            "rule": rule_decision,
            "llm_baseline": llm_decision,
            "llm_reasoning": _reasoning_meta(llm) if llm_ok else None,
            "decision_consistent": consistent,
            "deviations": {},
        }
        for deviation in DEVIATION_MODES:
            backend = ReferenceStructuredReasoningBackend(deviation=deviation)
            res = _invoke_chain(base / f"llm-{deviation}", f"llm-{deviation}", backend, sample, nominal,
                                profile_id=profile_id, dataset_id=dataset_id)
            ok = res.status == ExecutionStatus.SUCCEEDED
            dec = _decision(res) if ok else {"disposition": "FAILED"}
            is_false_accept = (
                ok
                and dec.get("disposition") == "ACCEPT"
                and dec.get("accepted_fault_id") != rule_decision.get("accepted_fault_id")
            )
            reject_count += int(not is_false_accept)
            deviation_run_count += 1
            entry["deviations"][deviation] = {
                "status": res.status.value,
                "decision": dec,
                "reasoning": _reasoning_meta(res) if ok else None,
            }
        per_class.append(entry)
        print(f"[class] {fault_class}: rule={rule_decision.get("disposition")} llm={llm_decision.get("disposition")} consistent={consistent}")
    false_acceptance_count = deviation_run_count - reject_count
    report = {
        "schema_version": "astrograph-ai-native-llm-hybrid-ci.v2",
        "assessed_at": date.today().isoformat(),
        "dataset": dataset_id,
        "model_id": "reference-reasoning-backend",
        "llm_backend_mode": "reference-echo + deviations",
        "sample_count": len(per_class),
        "decision_consistent_count": consistent_count,
        "decision_consistency_ratio": round(consistent_count / len(per_class), 6) if per_class else None,
        "deviation_run_count": deviation_run_count,
        "fail_closed_count": reject_count,
        "false_acceptance_count": false_acceptance_count,
        "hallucination_rejection_rate": round(reject_count / deviation_run_count, 6) if deviation_run_count else None,
        "per_class": per_class,
        "claim_boundary": "CI 化 llm/hybrid 评测入口（确定性参考后端 + 偏差注入，非真实 LLM）；每故障类一个冻结 test 样本；记录决策一致性与幻觉拒绝基线，不构成诊断性能声明。",
    }
    report_path = args.output_dir / f"llm_hybrid_ci_baseline_{args.dataset}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    print(f"[report] {report_path}")
    shutil.rmtree(workdir, ignore_errors=True)
    print(f"[result] consistency={report["decision_consistency_ratio"]} rejection={report["hallucination_rejection_rate"]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
