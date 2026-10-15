#!/usr/bin/env python3
"""R28 DXC qualitative forward-model post-contact development audit.

The runner consumes the frozen R27 development predictions plus the same DXC09
Sample DEVELOPMENT scenario bytes.  It does not rerun or inspect Competition or
Rematch and therefore cannot create external-validation evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from astrograph.diagnosis.dxc_forward_response import DXCForwardResponseConfig, rank_dxc_forward_response
from astrograph.evaluation.data_exposure import dataset_use, registry
from astrograph.evaluation.external_dataset_contract import seal, verify_seal
from astrograph.evaluation.nasa_adapt_adapter import parse_scenario, system_catalog

SOURCE_ROOT = Path(__file__).resolve().parents[2]


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _components(faults):
    return {str(value).split(":", 1)[0] for value in faults}


def _aggregate(rows):
    tp = fp = fn = ctp = cfp = cfn = exact = overlap = 0
    pred_count = resolved = 0
    for row in rows:
        truth = set(row["true_fault_ids"])
        pred = set(row["predicted_fault_ids"])
        tp += len(truth & pred); fp += len(pred - truth); fn += len(truth - pred)
        tc = _components(truth); pc = _components(pred)
        ctp += len(tc & pc); cfp += len(pc - tc); cfn += len(tc - pc)
        exact += int(truth == pred); overlap += int(bool(truth & pred)); pred_count += len(pred)
        resolved += int(bool(pred))

    def prf(a, b, c):
        p = a / (a + b) if a + b else 0.0
        r = a / (a + c) if a + c else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f

    p, r, f = prf(tp, fp, fn); cp, cr, cf = prf(ctp, cfp, cfn); n = len(rows)
    resolved_rows = [row for row in rows if row["predicted_fault_ids"]]
    resolved_exact = sum(set(row["true_fault_ids"]) == set(row["predicted_fault_ids"]) for row in resolved_rows)
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
        "diagnostic_resolution_rate": resolved / n if n else 0.0,
        "unresolved_case_count": n - resolved,
        "resolved_exact_fault_set_rate": resolved_exact / len(resolved_rows) if resolved_rows else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-root", required=True)
    ap.add_argument("--system-xml", required=True)
    ap.add_argument("--r27-development-receipt", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    sample_root = Path(args.sample_root).resolve()
    xml_path = Path(args.system_xml).resolve()
    r27_path = Path(args.r27_development_receipt).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    r27 = json.loads(r27_path.read_text(encoding="utf-8"))
    verify_seal(r27)
    catalog = system_catalog(xml_path.read_bytes())
    cfg = DXCForwardResponseConfig()
    allowed = set(registry()["datasets"]["DXC09_SAMPLE"]["group_ids_by_partition"]["DEVELOPMENT"])

    partitions = {}
    for split in ("SELECTION", "INTERNAL_REPLAY"):
        base_rows = r27["partitions"][split]["r27_case_rows"]
        gids = [str(row["group_id"]) for row in base_rows]
        if not set(gids) <= allowed:
            raise ValueError("R28_DXC_SAMPLE_GROUP_NOT_REGISTERED")
        rows = []
        with dataset_use("DXC09_SAMPLE", "DEVELOPMENT", "OPTIMIZE", group_ids=gids):
            for base in base_rows:
                path = sample_root / f"{base['case_id']}.scn"
                raw = path.read_bytes()
                gid = _sha(raw)
                if gid != base["group_id"]:
                    raise ValueError(f"R28_DXC_SAMPLE_IDENTITY_DRIFT:{base['case_id']}")
                records, truth = parse_scenario(raw.decode("utf-8"))
                if sorted(truth["true_fault_ids"]) != sorted(base["true_fault_ids"]):
                    raise ValueError(f"R28_DXC_TRUTH_IDENTITY_DRIFT:{base['case_id']}")
                decision = rank_dxc_forward_response(
                    records=records,
                    catalog=catalog,
                    base_fault_ids=base["predicted_fault_ids"],
                    first_alarm_time_s=base.get("first_alarm_time_s"),
                    config=cfg,
                )
                rows.append({
                    "case_id": base["case_id"],
                    "group_id": gid,
                    "true_fault_ids": base["true_fault_ids"],
                    "r27_predicted_fault_ids": base["predicted_fault_ids"],
                    "predicted_fault_ids": list(decision.selected_fault_ids),
                    "first_alarm_time_s": base.get("first_alarm_time_s"),
                    "forward_response_result_sha256": decision.result_sha256,
                    "vetoed_fault_ids": list(decision.vetoed_fault_ids),
                    "unresolved_reason_codes": list(decision.unresolved_reason_codes),
                    "fits": [item.model_dump(mode="json") for item in decision.fits],
                })
        partitions[split] = {
            "r27_metrics": r27["partitions"][split]["r27_metrics"],
            "r28_metrics": _aggregate(rows),
            "r28_case_rows": rows,
        }

    all_rows = [row for split in ("SELECTION", "INTERNAL_REPLAY") for row in partitions[split]["r28_case_rows"]]
    changed = [row for row in all_rows if set(row["r27_predicted_fault_ids"]) != set(row["predicted_fault_ids"])]
    unresolved = [row for row in all_rows if not row["predicted_fault_ids"]]
    payload = {
        "schema_version": "astrograph-r28-dxc-forward-simulation-development-report.v1",
        "dataset_id": "DXC09_SAMPLE",
        "partition": "POST_CONTACT_DEVELOPMENT",
        "source_system_xml_sha256": _sha(xml_path.read_bytes()),
        "source_r27_receipt_sha256": r27["receipt_sha256"],
        "r28_forward_config": cfg.model_dump(mode="json"),
        "algorithm_source_sha256": _sha((SOURCE_ROOT / "src/astrograph/diagnosis/dxc_forward_response.py").read_bytes()),
        "integration_adapter_sha256": _sha((SOURCE_ROOT / "src/astrograph/evaluation/dxc_r28_forward_simulation.py").read_bytes()),
        "split_policy": "Same already-contacted R25/R26/R27 DXC09 Sample hash split; no independent validation slice remains.",
        "partitions": partitions,
        "combined_post_contact_development": {
            "r27_metrics": r27["combined_post_contact_development"]["r27_metrics"],
            "r28_metrics": _aggregate(all_rows),
        },
        "changed_case_count": len(changed),
        "changed_case_ids": [row["case_id"] for row in changed],
        "unresolved_case_count": len(unresolved),
        "unresolved_case_ids": [row["case_id"] for row in unresolved],
        "modeled_scope": [
            "numeric sensor Stuck observation transition",
            "large additive numeric sensor Offset observation transition",
            "LightBulb25W FailedOff local-thermal/shared-light forward response",
            "sensor-only Offset consistency veto with explicit unresolved disposition",
        ],
        "continuous_physics_simulator": False,
        "agent_required": False,
        "legacy_algorithm_retained": True,
        "runtime_default_changed": False,
        "competition_or_rematch_regenerated": False,
        "new_holdout_supplied": False,
        "stop_rule": "STOP_AFTER_FIXED_R28_MODEL_AUDIT_NO_FURTHER_CASE_SPECIFIC_RULE_OPTIMIZATION",
        "status": "POST_CONTACT_DEVELOPMENT_FORWARD_MODEL_AUDIT_ONLY",
        "qualification": "NOT_EXTERNAL_VALIDATION_NOT_PRODUCTION_DEFAULT_NOT_FLIGHT_QUALIFIED",
        "claim_boundary": (
            "R28 was developed after inspection of DXC09 Sample development outcomes. Metrics are post-contact remediation evidence only. "
            "The qualitative forward model is not HyDE, a continuous electrical simulator, posterior probability, or formal evidence. "
            "Competition/Rematch are not consumed and production runtime defaults are unchanged."
        ),
    }
    sealed = seal(payload)
    output.write_text(json.dumps(sealed, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "selection": partitions["SELECTION"]["r28_metrics"],
        "internal_replay": partitions["INTERNAL_REPLAY"]["r28_metrics"],
        "combined": sealed["combined_post_contact_development"]["r28_metrics"],
        "changed_case_ids": sealed["changed_case_ids"],
        "unresolved_case_ids": sealed["unresolved_case_ids"],
        "receipt_sha256": sealed["receipt_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
