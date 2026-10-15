#!/usr/bin/env python3
"""R26 DXC09-sample development experiment for discrete-state consistency fusion.

Competition/Rematch partitions are never consumed.  The R25 hash split is kept:
TRAIN fits only the continuous residual model from pre-injection telemetry,
SELECTION chooses among four *predeclared* fusion policies, and INTERNAL_REPLAY
is reported only as seen-development audit (it was already exposed in R25).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

from astrograph.evaluation.data_exposure import dataset_use, guard_development, registry
from astrograph.evaluation.dxc_discrete_consistency import (
    DXCDiscreteConsistencyConfig,
    diagnose_discrete_state_consistency,
)
from astrograph.evaluation.dxc_hybrid_fusion_r26 import DXCFusionPolicy, fuse_dxc_r26
from astrograph.evaluation.dxc_hybrid_model_based import (
    DXCHybridResidualConfig,
    extract_numeric_sequence,
    fit_hybrid_residual_model,
    hybrid_model_artifact,
    predict_hybrid_model_based,
)
from astrograph.evaluation.external_dataset_contract import seal
from astrograph.evaluation.nasa_adapt_adapter import parse_scenario, system_catalog

CONTINUOUS_CONFIG = DXCHybridResidualConfig(
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
DISCRETE_CONFIG = DXCDiscreteConsistencyConfig(
    startup_s=30.0,
    command_suppression_s=1.0,
    position_sensor_component_type="PositionSensor",
    open_state_value=False,
)
POLICIES = (
    DXCFusionPolicy.CONTINUOUS_ONLY,
    DXCFusionPolicy.UNION,
    DXCFusionPolicy.DISCRETE_OVERRIDE,
    DXCFusionPolicy.DISCRETE_FIRST_SINGLE_CONTINUOUS,
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _components(faults):
    return {str(value).split(":", 1)[0] for value in faults}


def _timing(first_alarm: float | None, injection_s: float):
    if first_alarm is None:
        return "NO_ALARM", None
    if first_alarm < injection_s:
        return "EARLY_BEFORE_INJECTION", None
    return "POST_INJECTION_FIRST_ALARM", float(first_alarm - injection_s)


def _aggregate(rows):
    tp = fp = fn = ctp = cfp = cfn = exact = overlap = 0
    early = post = none = 0
    delays = []
    pred_count = 0
    discrete_case_count = 0
    for row in rows:
        truth = set(row["true_fault_ids"])
        pred = set(row["predicted_fault_ids"])
        tp += len(truth & pred); fp += len(pred - truth); fn += len(truth - pred)
        tc = _components(truth); pc = _components(pred)
        ctp += len(tc & pc); cfp += len(pc - tc); cfn += len(tc - pc)
        exact += int(truth == pred); overlap += int(bool(truth & pred))
        pred_count += len(pred)
        discrete_case_count += int(bool(row["discrete_fault_ids"]))
        status = row["detection_timing_status"]
        early += int(status == "EARLY_BEFORE_INJECTION")
        post += int(status == "POST_INJECTION_FIRST_ALARM")
        none += int(status == "NO_ALARM")
        if row.get("post_injection_delay_s") is not None:
            delays.append(float(row["post_injection_delay_s"]))

    def prf(a, b, c):
        p = a / (a + b) if a + b else 0.0
        r = a / (a + c) if a + c else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f

    p, r, f = prf(tp, fp, fn)
    cp, cr, cf = prf(ctp, cfp, cfn)
    n = len(rows)
    return {
        "case_count": n,
        "fault_element_precision_micro": p,
        "fault_element_recall_micro": r,
        "fault_element_f1_micro": f,
        "component_precision_micro": cp,
        "component_recall_micro": cr,
        "component_f1_micro": cf,
        "exact_fault_set_rate": exact / n if n else 0.0,
        "faulty_case_any_fault_element_overlap_rate": overlap / n if n else 0.0,
        "mean_predicted_fault_count": pred_count / n if n else 0.0,
        "discrete_branch_active_case_count": discrete_case_count,
        "early_first_alarm_case_count": early,
        "post_injection_first_alarm_case_count": post,
        "no_alarm_case_count": none,
        "post_injection_first_alarm_rate": post / n if n else 0.0,
        "median_post_injection_delay_s": median(delays) if delays else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-root", required=True)
    ap.add_argument("--system-xml", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    root = Path(args.sample_root).resolve()
    xml_path = Path(args.system_xml).resolve()
    output = Path(args.output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    catalog = system_catalog(xml_path.read_bytes())
    allowed = set(registry()["datasets"]["DXC09_SAMPLE"]["group_ids_by_partition"]["DEVELOPMENT"])

    cases = []
    for path in sorted(root.glob("*.scn")):
        raw = path.read_bytes(); gid = _sha(raw)
        if gid not in allowed:
            raise ValueError("R26_DXC_SAMPLE_GROUP_NOT_REGISTERED")
        records, truth = parse_scenario(raw.decode("utf-8"))
        if truth["injection_time"] is None:
            continue
        bucket = int(gid[:8], 16) % 5
        split = "TRAIN" if bucket in (0, 1, 2) else "SELECTION" if bucket == 3 else "INTERNAL_REPLAY"
        cases.append({"path": path, "group_id": gid, "records": records, "truth": truth, "split": split})
    if len(cases) != 74:
        raise ValueError(f"R26_DXC_INJECTED_CASE_COUNT_DRIFT:{len(cases)}")

    train = [x for x in cases if x["split"] == "TRAIN"]
    selection = [x for x in cases if x["split"] == "SELECTION"]
    replay = [x for x in cases if x["split"] == "INTERNAL_REPLAY"]
    train_ids = [x["group_id"] for x in train]
    with dataset_use("DXC09_SAMPLE", "DEVELOPMENT", "FIT", group_ids=train_ids):
        guard_development("FIT", group_ids=train_ids)
        sequences = []
        for case in train:
            injection = int(case["truth"]["injection_time"])
            sequences.append(extract_numeric_sequence(
                [record for record in case["records"] if int(record["time_tick"]) < injection],
                catalog,
            ))
        model = fit_hybrid_residual_model(sequences, alpha=CONTINUOUS_CONFIG.ridge_alpha)

    def base_case(case):
        continuous = predict_hybrid_model_based(case["records"], catalog, model, CONTINUOUS_CONFIG)
        discrete = diagnose_discrete_state_consistency(case["records"], catalog, DISCRETE_CONFIG)
        return continuous, discrete

    def fused_row(case, continuous, discrete, policy):
        fused = fuse_dxc_r26(continuous, discrete, policy)
        injection_s = float(case["truth"]["injection_time"]) / 1000.0
        status, delay = _timing(fused.first_alarm_time_s, injection_s)
        return {
            "case_id": case["path"].stem,
            "group_id": case["group_id"],
            "true_fault_ids": case["truth"]["true_fault_ids"],
            "predicted_fault_ids": list(fused.predicted_fault_ids),
            "continuous_fault_ids": list(fused.continuous_fault_ids),
            "discrete_fault_ids": list(fused.discrete_fault_ids),
            "discrete_events": [item.model_dump(mode="json") for item in discrete.events],
            "injection_time_s": injection_s,
            "first_alarm_time_s": fused.first_alarm_time_s,
            "detection_timing_status": status,
            "post_injection_delay_s": delay,
            "fused_result_sha256": fused.result_sha256,
        }

    policy_results = {}
    selection_ids = [x["group_id"] for x in selection]
    with dataset_use("DXC09_SAMPLE", "DEVELOPMENT", "SELECT", group_ids=selection_ids):
        guard_development("SELECT", group_ids=selection_ids)
        selection_base = [(case, *base_case(case)) for case in selection]
        for policy in POLICIES:
            rows = [fused_row(case, continuous, discrete, policy) for case, continuous, discrete in selection_base]
            policy_results[policy.value] = {"metrics": _aggregate(rows), "case_rows": rows}

    # Strict isolation is the primary DXC objective: exact fault-set first, then
    # micro fault-element F1, component F1, then fewer emitted faults.
    selected_policy = sorted(
        POLICIES,
        key=lambda policy: (
            -policy_results[policy.value]["metrics"]["exact_fault_set_rate"],
            -policy_results[policy.value]["metrics"]["fault_element_f1_micro"],
            -policy_results[policy.value]["metrics"]["component_f1_micro"],
            policy_results[policy.value]["metrics"]["mean_predicted_fault_count"],
            policy.value,
        ),
    )[0]

    replay_ids = [x["group_id"] for x in replay]
    with dataset_use("DXC09_SAMPLE", "DEVELOPMENT", "OPTIMIZE", group_ids=replay_ids):
        replay_base = [(case, *base_case(case)) for case in replay]
        replay_rows = [fused_row(case, continuous, discrete, selected_policy) for case, continuous, discrete in replay_base]
        continuous_rows = [fused_row(case, continuous, discrete, DXCFusionPolicy.CONTINUOUS_ONLY) for case, continuous, discrete in replay_base]
        discrete_rows = [fused_row(case, continuous, discrete, DXCFusionPolicy.DISCRETE_OVERRIDE) for case, continuous, discrete in replay_base]

    payload = {
        "schema_version": "astrograph-r26-dxc-discrete-consistency-development-report.v1",
        "dataset_id": "DXC09_SAMPLE",
        "partition": "DEVELOPMENT",
        "source_system_xml_sha256": _sha(xml_path.read_bytes()),
        "split_policy": "R25 frozen hash split: 0/1/2 TRAIN, 3 SELECTION, 4 INTERNAL_REPLAY",
        "split_counts": {"TRAIN": len(train), "SELECTION": len(selection), "INTERNAL_REPLAY": len(replay)},
        "continuous_config": CONTINUOUS_CONFIG.model_dump(mode="json"),
        "discrete_config": DISCRETE_CONFIG.model_dump(mode="json"),
        "continuous_model_artifact": hybrid_model_artifact(model),
        "predeclared_fusion_policies": [item.value for item in POLICIES],
        "selection_key": [
            "max exact_fault_set_rate",
            "max fault_element_f1_micro",
            "max component_f1_micro",
            "min mean_predicted_fault_count",
            "stable policy name",
        ],
        "selection_policy_results": policy_results,
        "selected_policy": selected_policy.value,
        "internal_replay_metrics": _aggregate(replay_rows),
        "r25_continuous_same_internal_replay_metrics": _aggregate(continuous_rows),
        "discrete_override_same_internal_replay_metrics": _aggregate(discrete_rows),
        "internal_replay_case_rows": replay_rows,
        "seen_competition_or_rematch_regenerated": False,
        "internal_replay_was_already_exposed_in_r25": True,
        "status": "DEVELOPMENT_FUSION_SELECTED_INTERNAL_REPLAY_AUDIT_ONLY",
        "qualification": "NOT_EXTERNAL_VALIDATION_NOT_FLIGHT_QUALIFIED",
        "claim_boundary": (
            "R26 uses only DXC09 Sample DEVELOPMENT. INTERNAL_REPLAY was already observed in R25 and therefore is not independent validation. "
            "The discrete branch admits bool state observations under a separate semantic contract; it does not relax continuous-unit governance."
        ),
    }
    sealed = seal(payload)
    output.write_text(json.dumps(sealed, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "selected_policy": selected_policy.value,
        "selection": policy_results[selected_policy.value]["metrics"],
        "internal_replay": payload["internal_replay_metrics"],
        "r25_continuous": payload["r25_continuous_same_internal_replay_metrics"],
        "receipt_sha256": sealed["receipt_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
