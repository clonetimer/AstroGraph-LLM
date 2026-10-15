#!/usr/bin/env python3
"""R25 DXC09 Sample development-only command-aware residual/graph replay.

The 74 injected sample cases are deterministically split by the already frozen
scenario-byte SHA256. This is an internal development replay, not independent
validation. Competition/Rematch bytes are never consumed by this runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

from astrograph.evaluation.data_exposure import dataset_use, guard_development, registry
from astrograph.evaluation.dxc_hybrid_model_based import (
    DXCHybridResidualConfig,
    extract_numeric_sequence,
    fit_hybrid_residual_model,
    hybrid_model_artifact,
    predict_hybrid_model_based,
)
from astrograph.evaluation.dxc_statistical_predictor import predict as baseline_predict
from astrograph.evaluation.external_dataset_contract import seal
from astrograph.evaluation.nasa_adapt_adapter import parse_scenario, system_catalog

CONFIG = DXCHybridResidualConfig(
    ridge_alpha=10.0,
    startup_calibration_s=30.0,
    residual_z_threshold=8.0,
    persistence_decisions=1,
    top_residual_sensors=4,
    support_topology_hops=3,
    conflict_topology_hops=1,
    top_components=2,
    command_context_s=10.0,
    include_recent_command_in_structural_conflict=False,
    max_diagnosis_cardinality=4,
    max_diagnosis_candidates=64,
    max_search_nodes=5000,
)
BASELINE_CONFIG = {
    "baseline_duration_s": 30,
    "window_samples": 16,
    "decision_stride_s": 5,
    "score_threshold": 85,
    "minimum_reference_samples": 5,
    "current_drop_fraction": 0.05,
}


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _component_set(faults):
    return {str(value).split(":", 1)[0] for value in faults}


def _aggregate(rows):
    tp = fp = fn = ctp = cfp = cfn = exact = overlap = 0
    early = post = none = 0
    delays = []
    mbd_case_with_truth_overlap = mbd_case_exact_candidate = mbd_cases = 0
    for row in rows:
        truth = set(row["true_fault_ids"]); pred = set(row["predicted_fault_ids"])
        tp += len(truth & pred); fp += len(pred - truth); fn += len(truth - pred)
        tc = _component_set(truth); pc = _component_set(pred)
        ctp += len(tc & pc); cfp += len(pc - tc); cfn += len(tc - pc)
        exact += int(truth == pred)
        overlap += int(bool(truth & pred))
        status = row["detection_timing_status"]
        early += int(status == "EARLY_BEFORE_INJECTION")
        post += int(status == "POST_INJECTION_FIRST_ALARM")
        none += int(status == "NO_ALARM")
        if row.get("post_injection_delay_s") is not None:
            delays.append(float(row["post_injection_delay_s"]))
        if "minimal_diagnosis_candidates" in row:
            mbd_cases += 1
            candidates = [set(x) for x in row.get("minimal_diagnosis_candidates", [])]
            mbd_case_with_truth_overlap += int(any(bool(truth & candidate) for candidate in candidates))
            mbd_case_exact_candidate += int(any(truth == candidate for candidate in candidates))
    def prf(a,b,c):
        p = a/(a+b) if a+b else 0.0; r = a/(a+c) if a+c else 0.0
        f = 2*p*r/(p+r) if p+r else 0.0
        return p,r,f
    p,r,f = prf(tp,fp,fn); cp,cr,cf=prf(ctp,cfp,cfn)
    n=len(rows)
    return {
        "case_count": n,
        "fault_element_precision_micro": p,
        "fault_element_recall_micro": r,
        "fault_element_f1_micro": f,
        "component_precision_micro": cp,
        "component_recall_micro": cr,
        "component_f1_micro": cf,
        "exact_fault_set_rate": exact/n if n else 0.0,
        "faulty_case_any_fault_element_overlap_rate": overlap/n if n else 0.0,
        "early_first_alarm_case_count": early,
        "post_injection_first_alarm_case_count": post,
        "no_alarm_case_count": none,
        "post_injection_first_alarm_rate": post/n if n else 0.0,
        "median_post_injection_delay_s": median(delays) if delays else None,
        "minimal_diagnosis_case_count": mbd_cases,
        "minimal_diagnosis_any_truth_overlap_case_rate": (mbd_case_with_truth_overlap/mbd_cases if mbd_cases else None),
        "minimal_diagnosis_exact_truth_candidate_case_rate": (mbd_case_exact_candidate/mbd_cases if mbd_cases else None),
    }


def _timing(first_alarm, injection_s):
    if first_alarm is None:
        return "NO_ALARM", None
    if first_alarm < injection_s:
        return "EARLY_BEFORE_INJECTION", None
    return "POST_INJECTION_FIRST_ALARM", float(first_alarm-injection_s)


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--sample-root", required=True)
    ap.add_argument("--system-xml", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--include-command-conflict", action="store_true", help="Negative-development experiment: expand a unique recent command target into structural conflicts.")
    args=ap.parse_args()
    run_config = CONFIG.model_copy(update={
        "include_recent_command_in_structural_conflict": bool(args.include_command_conflict),
        "max_diagnosis_cardinality": 3 if args.include_command_conflict else CONFIG.max_diagnosis_cardinality,
        "max_diagnosis_candidates": 32 if args.include_command_conflict else CONFIG.max_diagnosis_candidates,
        "max_search_nodes": 5000,
    })
    sample_root=Path(args.sample_root).resolve(); system_xml=Path(args.system_xml).resolve()
    output=Path(args.output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    catalog=system_catalog(system_xml.read_bytes())
    allowed=set(registry()["datasets"]["DXC09_SAMPLE"]["group_ids_by_partition"]["DEVELOPMENT"])

    cases=[]
    for path in sorted(sample_root.glob("*.scn")):
        raw=path.read_bytes(); group_id=_sha_bytes(raw)
        if group_id not in allowed:
            raise ValueError("R25_DXC_SAMPLE_GROUP_NOT_REGISTERED")
        records,truth=parse_scenario(raw.decode("utf-8"))
        if truth["injection_time"] is None:
            continue
        split_bucket=int(group_id[:8],16)%5
        split="TRAIN" if split_bucket in (0,1,2) else "SELECTION" if split_bucket==3 else "INTERNAL_REPLAY"
        cases.append({"path":path,"group_id":group_id,"records":records,"truth":truth,"split":split})
    if len(cases)!=74:
        raise ValueError(f"R25_DXC_INJECTED_CASE_COUNT_DRIFT:{len(cases)}")

    train=[x for x in cases if x["split"]=="TRAIN"]
    selection=[x for x in cases if x["split"]=="SELECTION"]
    replay=[x for x in cases if x["split"]=="INTERNAL_REPLAY"]
    train_ids=[x["group_id"] for x in train]
    with dataset_use("DXC09_SAMPLE","DEVELOPMENT","FIT",group_ids=train_ids):
        guard_development("FIT",group_ids=train_ids)
        sequences=[]
        for case in train:
            injection=int(case["truth"]["injection_time"])
            admitted=[r for r in case["records"] if int(r["time_tick"]) < injection]
            sequences.append(extract_numeric_sequence(admitted,catalog))
        model=fit_hybrid_residual_model(sequences,alpha=run_config.ridge_alpha)

    def run_hybrid(subset):
        ids=[x["group_id"] for x in subset]; rows=[]
        with dataset_use("DXC09_SAMPLE","DEVELOPMENT","OPTIMIZE",group_ids=ids):
            for case in subset:
                result=predict_hybrid_model_based(case["records"],catalog,model,run_config)
                injection_s=float(case["truth"]["injection_time"])/1000.0
                status,delay=_timing(result.first_alarm_time_s,injection_s)
                rows.append({
                    "case_id":case["path"].stem,
                    "group_id":case["group_id"],
                    "true_fault_ids":case["truth"]["true_fault_ids"],
                    "predicted_fault_ids":list(result.predicted_fault_ids),
                    "predicted_component_ids":list(result.final_component_ids),
                    "injection_time_s":injection_s,
                    "first_alarm_time_s":result.first_alarm_time_s,
                    "detection_timing_status":status,
                    "post_injection_delay_s":delay,
                    "minimal_diagnosis_search_complete": None if result.minimal_diagnosis is None else result.minimal_diagnosis.search_complete,
                    "minimal_diagnosis_candidate_count": 0 if result.minimal_diagnosis is None else len(result.minimal_diagnosis.subset_minimal_diagnoses),
                    "minimal_diagnosis_candidates": [] if result.minimal_diagnosis is None else [list(item.fault_ids) for item in result.minimal_diagnosis.subset_minimal_diagnoses],
                })
        return rows

    selection_rows=run_hybrid(selection)
    replay_rows=run_hybrid(replay)

    baseline_rows=[]; replay_ids=[x["group_id"] for x in replay]
    with dataset_use("DXC09_SAMPLE","DEVELOPMENT","OPTIMIZE",group_ids=replay_ids):
        for case in replay:
            result=baseline_predict(case["records"],catalog,BASELINE_CONFIG)
            alarms=[row for row in result["timeline"] if row["predicted_fault_ids"]]
            first=alarms[0]["time_s"] if alarms else None
            injection_s=float(case["truth"]["injection_time"])/1000.0
            status,delay=_timing(first,injection_s)
            baseline_rows.append({
                "case_id":case["path"].stem,
                "group_id":case["group_id"],
                "true_fault_ids":case["truth"]["true_fault_ids"],
                "predicted_fault_ids":result["predicted_fault_ids"],
                "injection_time_s":injection_s,
                "first_alarm_time_s":first,
                "detection_timing_status":status,
                "post_injection_delay_s":delay,
            })

    payload={
        "schema_version":"astrograph-r25-dxc-hybrid-development-report.v1",
        "dataset_id":"DXC09_SAMPLE",
        "partition":"DEVELOPMENT",
        "source_system_xml_sha256":_sha_bytes(system_xml.read_bytes()),
        "injected_case_count":len(cases),
        "split_policy":"int(first_8_hex_chars_of_registered_scn_sha256,16)%5; buckets 0/1/2 TRAIN, 3 SELECTION, 4 INTERNAL_REPLAY",
        "split_counts":{"TRAIN":len(train),"SELECTION":len(selection),"INTERNAL_REPLAY":len(replay)},
        "config":run_config.model_dump(mode="json"),
        "config_status":("NEGATIVE_COMMAND_CONFLICT_EXPANSION_EXPERIMENT" if args.include_command_conflict else "EXPLORATORY_POST_R24_DEVELOPMENT_CONFIGURATION_NOT_PREREGISTERED"),
        "model_sha256":model.model_sha256,
        "model_artifact":hybrid_model_artifact(model),
        "training_pair_source_case_count":len(train),
        "selection_metrics":_aggregate(selection_rows),
        "internal_replay_metrics":_aggregate(replay_rows),
        "r20_frozen_baseline_same_internal_replay_metrics":_aggregate(baseline_rows),
        "internal_replay_case_rows":replay_rows,
        "baseline_internal_replay_case_rows":baseline_rows,
        "seen_competition_or_rematch_regenerated":False,
        "status":"DEVELOPMENT_REMEDIATION_EVIDENCE_ONLY",
        "qualification":"NOT_EXTERNAL_VALIDATION_NOT_FLIGHT_QUALIFIED",
        "claim_boundary":"Internal hash split of the already development-exposed DXC09 sample. Improved timing/precision is not independent validation; graph support remains an engineering heuristic and minimal diagnosis quality depends on residual/conflict validity.",
    }
    result=seal(payload); output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({
        "split_counts":payload["split_counts"],
        "selection":payload["selection_metrics"],
        "internal_replay":payload["internal_replay_metrics"],
        "baseline_internal_replay":payload["r20_frozen_baseline_same_internal_replay_metrics"],
        "receipt_sha256":result["receipt_sha256"],
    },indent=2))
    return 0

if __name__=="__main__": raise SystemExit(main())
