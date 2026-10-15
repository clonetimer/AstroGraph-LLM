#!/usr/bin/env python3
"""R27 DXC expected-response development audit.

Uses only DXC09 Sample DEVELOPMENT.  The R26 continuous model/config and hash
split are reused verbatim.  R27 rules were developed after contact with both
SELECTION and INTERNAL_REPLAY, so neither slice is treated as independent
validation in this report.  Competition/Rematch are forbidden.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

from astrograph.diagnosis.dxc_expected_response import DXCExpectedResponseConfig
from astrograph.evaluation.data_exposure import dataset_use, registry
from astrograph.evaluation.dxc_discrete_consistency import DXCDiscreteConsistencyConfig, diagnose_discrete_state_consistency
from astrograph.evaluation.dxc_hybrid_fusion_r26 import DXCFusionPolicy, fuse_dxc_r26
from astrograph.evaluation.dxc_hybrid_model_based import (
    DXCHybridResidualConfig,
    hybrid_model_from_artifact,
    predict_hybrid_model_based,
)
from astrograph.evaluation.dxc_r27_expected_response import assemble_dxc_r27_prediction
from astrograph.evaluation.external_dataset_contract import seal
from astrograph.evaluation.nasa_adapt_adapter import parse_scenario, system_catalog


SOURCE_ROOT = Path(__file__).resolve().parents[2]


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
    for row in rows:
        truth = set(row["true_fault_ids"])
        pred = set(row["predicted_fault_ids"])
        tp += len(truth & pred); fp += len(pred - truth); fn += len(truth - pred)
        tc = _components(truth); pc = _components(pred)
        ctp += len(tc & pc); cfp += len(pc - tc); cfn += len(tc - pc)
        exact += int(truth == pred); overlap += int(bool(truth & pred)); pred_count += len(pred)
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

    p, r, f = prf(tp, fp, fn); cp, cr, cf = prf(ctp, cfp, cfn); n = len(rows)
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
    ap.add_argument("--r26-development-receipt", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    root = Path(args.sample_root).resolve(); xml_path = Path(args.system_xml).resolve()
    r26_path = Path(args.r26_development_receipt).resolve(); output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    catalog = system_catalog(xml_path.read_bytes())
    r26 = json.loads(r26_path.read_text(encoding="utf-8"))
    model = hybrid_model_from_artifact(r26["continuous_model_artifact"])
    continuous_config = DXCHybridResidualConfig.model_validate(r26["continuous_config"])
    discrete_config = DXCDiscreteConsistencyConfig.model_validate(r26["discrete_config"])
    expected_config = DXCExpectedResponseConfig()
    allowed = set(registry()["datasets"]["DXC09_SAMPLE"]["group_ids_by_partition"]["DEVELOPMENT"])

    cases = []
    for path in sorted(root.glob("*.scn")):
        raw = path.read_bytes(); gid = _sha(raw)
        if gid not in allowed:
            raise ValueError("R27_DXC_SAMPLE_GROUP_NOT_REGISTERED")
        records, truth = parse_scenario(raw.decode("utf-8"))
        if truth["injection_time"] is None:
            continue
        bucket = int(gid[:8], 16) % 5
        split = "TRAIN" if bucket in (0, 1, 2) else "SELECTION" if bucket == 3 else "INTERNAL_REPLAY"
        cases.append({"path": path, "group_id": gid, "records": records, "truth": truth, "split": split})
    if len(cases) != 74:
        raise ValueError(f"R27_DXC_INJECTED_CASE_COUNT_DRIFT:{len(cases)}")

    def evaluate(case):
        continuous = predict_hybrid_model_based(case["records"], catalog, model, continuous_config)
        discrete = diagnose_discrete_state_consistency(case["records"], catalog, discrete_config)
        r26_pred = fuse_dxc_r26(continuous, discrete, DXCFusionPolicy.DISCRETE_FIRST_SINGLE_CONTINUOUS)
        r27_pred = assemble_dxc_r27_prediction(
            records=case["records"], catalog=catalog,
            continuous=continuous, discrete=discrete,
            expected_response_config=expected_config,
        )
        injection_s = float(case["truth"]["injection_time"]) / 1000.0
        r26_status, r26_delay = _timing(r26_pred.first_alarm_time_s, injection_s)
        r27_status, r27_delay = _timing(r27_pred.first_alarm_time_s, injection_s)
        common = {
            "case_id": case["path"].stem,
            "group_id": case["group_id"],
            "true_fault_ids": case["truth"]["true_fault_ids"],
            "injection_time_s": injection_s,
        }
        return (
            {
                **common,
                "predicted_fault_ids": list(r26_pred.predicted_fault_ids),
                "first_alarm_time_s": r26_pred.first_alarm_time_s,
                "detection_timing_status": r26_status,
                "post_injection_delay_s": r26_delay,
            },
            {
                **common,
                "predicted_fault_ids": list(r27_pred.predicted_fault_ids),
                "first_alarm_time_s": r27_pred.first_alarm_time_s,
                "detection_timing_status": r27_status,
                "post_injection_delay_s": r27_delay,
                "expected_response_result_sha256": r27_pred.expected_response.result_sha256,
                "expected_response_candidates": [x.model_dump(mode="json") for x in r27_pred.expected_response.candidates],
                "reclassified_position_sensor_count": r27_pred.expected_response.reclassified_position_sensor_count,
            },
        )

    partitions = {}
    for split, operation in (("SELECTION", "SELECT"), ("INTERNAL_REPLAY", "OPTIMIZE")):
        subset = [case for case in cases if case["split"] == split]
        gids = [case["group_id"] for case in subset]
        with dataset_use("DXC09_SAMPLE", "DEVELOPMENT", operation, group_ids=gids):
            paired = [evaluate(case) for case in subset]
        r26_rows = [item[0] for item in paired]; r27_rows = [item[1] for item in paired]
        partitions[split] = {
            "r26_metrics": _aggregate(r26_rows),
            "r27_metrics": _aggregate(r27_rows),
            "r26_case_rows": r26_rows,
            "r27_case_rows": r27_rows,
        }

    combined_r27_rows = [row for split in ("SELECTION", "INTERNAL_REPLAY") for row in partitions[split]["r27_case_rows"]]
    combined_r26_rows = [row for split in ("SELECTION", "INTERNAL_REPLAY") for row in partitions[split]["r26_case_rows"]]

    payload = {
        "schema_version": "astrograph-r27-dxc-expected-response-development-report.v1",
        "dataset_id": "DXC09_SAMPLE",
        "partition": "DEVELOPMENT",
        "source_system_xml_sha256": _sha(xml_path.read_bytes()),
        "source_r26_receipt_sha256": str(r26["receipt_sha256"]),
        "continuous_model_sha256": model.model_sha256,
        "r26_detection_policy_retained": "DISCRETE_FIRST_SINGLE_CONTINUOUS",
        "r27_expected_response_config": expected_config.model_dump(mode="json"),
        "algorithm_source_sha256": _sha((SOURCE_ROOT / "src/astrograph/diagnosis/dxc_expected_response.py").read_bytes()),
        "integration_adapter_sha256": _sha((SOURCE_ROOT / "src/astrograph/evaluation/dxc_r27_expected_response.py").read_bytes()),
        "algorithm_policy_sha256": _sha((SOURCE_ROOT / "configs/evaluation/r27_dxc_algorithm_policy.v1.json").read_bytes()),
        "split_policy": "R25/R26 frozen hash split: 0/1/2 TRAIN, 3 SELECTION, 4 INTERNAL_REPLAY",
        "partitions": partitions,
        "combined_post_contact_development": {
            "r26_metrics": _aggregate(combined_r26_rows),
            "r27_metrics": _aggregate(combined_r27_rows),
        },
        "seen_competition_or_rematch_regenerated": False,
        "agent_required": False,
        "legacy_algorithm_retained": True,
        "runtime_default_changed": False,
        "status": "POST_CONTACT_DEVELOPMENT_ALGORITHM_AUDIT_ONLY",
        "qualification": "NOT_EXTERNAL_VALIDATION_NOT_PRODUCTION_DEFAULT_NOT_FLIGHT_QUALIFIED",
        "claim_boundary": (
            "R27 rules were developed after seeing DXC09 Sample development outcomes, including the prior INTERNAL_REPLAY. "
            "Metrics are remediation evidence only. Competition/Rematch are not consumed and the legacy diagnostic branch remains available."
        ),
    }
    sealed = seal(payload)
    output.write_text(json.dumps(sealed, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "selection_r26": partitions["SELECTION"]["r26_metrics"],
        "selection_r27": partitions["SELECTION"]["r27_metrics"],
        "replay_r26": partitions["INTERNAL_REPLAY"]["r26_metrics"],
        "replay_r27": partitions["INTERNAL_REPLAY"]["r27_metrics"],
        "receipt_sha256": sealed["receipt_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
