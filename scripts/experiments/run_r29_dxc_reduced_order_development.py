#!/usr/bin/env python3
"""R29 DXC reduced-order physical simulation post-contact development audit.

Consumes the frozen R28 development receipt and the same DXC09 Sample DEVELOPMENT
scenario bytes.  R28-resolved predictions pass through unchanged; only the generic
R28 unresolved physical-response disposition may enter the reduced-order simulator.
Competition/Rematch are never read by this runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from astrograph.diagnosis.dxc_reduced_order_physics import (
    DXCReducedOrderPhysicsConfig,
    rank_dxc_reduced_order_physics,
)
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
    ap.add_argument("--r28-development-receipt", required=True)
    ap.add_argument("--design", default=str(SOURCE_ROOT / "configs/evaluation/dxc_r29_reduced_order.v1.json"))
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    sample_root = Path(args.sample_root).resolve()
    xml_path = Path(args.system_xml).resolve()
    r28_path = Path(args.r28_development_receipt).resolve()
    design_path = Path(args.design).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    r28 = json.loads(r28_path.read_text(encoding="utf-8"))
    verify_seal(r28)
    design = json.loads(design_path.read_text(encoding="utf-8"))
    cfg = DXCReducedOrderPhysicsConfig.model_validate(design["config"])
    catalog = system_catalog(xml_path.read_bytes())
    allowed = set(registry()["datasets"]["DXC09_SAMPLE"]["group_ids_by_partition"]["DEVELOPMENT"])

    partitions = {}
    for split in ("SELECTION", "INTERNAL_REPLAY"):
        base_rows = r28["partitions"][split]["r28_case_rows"]
        gids = [str(row["group_id"]) for row in base_rows]
        if not set(gids) <= allowed:
            raise ValueError("R29_DXC_SAMPLE_GROUP_NOT_REGISTERED")
        rows = []
        with dataset_use("DXC09_SAMPLE", "DEVELOPMENT", "OPTIMIZE", group_ids=gids):
            for base in base_rows:
                path = sample_root / f"{base['case_id']}.scn"
                raw = path.read_bytes()
                gid = _sha(raw)
                if gid != base["group_id"]:
                    raise ValueError(f"R29_DXC_SAMPLE_IDENTITY_DRIFT:{base['case_id']}")
                records, truth = parse_scenario(raw.decode("utf-8"))
                if sorted(truth["true_fault_ids"]) != sorted(base["true_fault_ids"]):
                    raise ValueError(f"R29_DXC_TRUTH_IDENTITY_DRIFT:{base['case_id']}")
                decision = rank_dxc_reduced_order_physics(
                    records=records,
                    catalog=catalog,
                    base_fault_ids=base["predicted_fault_ids"],
                    base_unresolved_reason_codes=base.get("unresolved_reason_codes", []),
                    first_alarm_time_s=base.get("first_alarm_time_s"),
                    config=cfg,
                )
                rows.append({
                    "case_id": base["case_id"],
                    "group_id": gid,
                    "true_fault_ids": base["true_fault_ids"],
                    "r28_predicted_fault_ids": base["predicted_fault_ids"],
                    "predicted_fault_ids": list(decision.selected_fault_ids),
                    "first_alarm_time_s": base.get("first_alarm_time_s"),
                    "application_status": decision.application_status,
                    "reason_codes": list(decision.reason_codes),
                    "reduced_order_result_sha256": decision.result_sha256,
                    "selected_diagnosis_id": decision.selected_diagnosis_id,
                    "candidate_fits": [item.model_dump(mode="json") for item in decision.candidate_fits],
                })
        partitions[split] = {
            "r28_metrics": r28["partitions"][split]["r28_metrics"],
            "r29_metrics": _aggregate(rows),
            "r29_case_rows": rows,
        }

    all_rows = [row for split in ("SELECTION", "INTERNAL_REPLAY") for row in partitions[split]["r29_case_rows"]]
    changed = [row for row in all_rows if set(row["r28_predicted_fault_ids"]) != set(row["predicted_fault_ids"])]
    unresolved = [row for row in all_rows if not row["predicted_fault_ids"]]
    physics_applied = [row for row in all_rows if row["application_status"] != "PASSTHROUGH_R28_RESOLVED"]
    payload = {
        "schema_version": "astrograph-r29-dxc-reduced-order-development-report.v1",
        "dataset_id": "DXC09_SAMPLE",
        "partition": "POST_CONTACT_DEVELOPMENT",
        "source_system_xml_sha256": _sha(xml_path.read_bytes()),
        "source_r28_receipt_sha256": r28["receipt_sha256"],
        "design_sha256": _sha(design_path.read_bytes()),
        "r29_config": cfg.model_dump(mode="json"),
        "simulation_provider_source_sha256": _sha((SOURCE_ROOT / "src/astrograph/simulation/adapt_reduced_order.py").read_bytes()),
        "diagnosis_source_sha256": _sha((SOURCE_ROOT / "src/astrograph/diagnosis/dxc_reduced_order_physics.py").read_bytes()),
        "integration_adapter_sha256": _sha((SOURCE_ROOT / "src/astrograph/evaluation/dxc_r29_reduced_order.py").read_bytes()),
        "split_policy": "Same already-contacted DXC09 Sample DEVELOPMENT groups; no independent validation slice remains.",
        "partitions": partitions,
        "combined_post_contact_development": {
            "r28_metrics": r28["combined_post_contact_development"]["r28_metrics"],
            "r29_metrics": _aggregate(all_rows),
        },
        "changed_case_count": len(changed),
        "changed_case_ids": [row["case_id"] for row in changed],
        "physics_application_case_count": len(physics_applied),
        "physics_application_case_ids": [row["case_id"] for row in physics_applied],
        "unresolved_case_count": len(unresolved),
        "unresolved_case_ids": [row["case_id"] for row in unresolved],
        "modeled_scope": [
            "ADAPT LightBulb*W FailedOff rated-power model parsed from public component type",
            "candidate-specific branch-current numerical trace under observed VAC boundary condition",
            "local temperature cooling consistency when a temperature sensor exists",
            "local relay-position and related-command consistency",
            "R28 resolved predictions pass through unchanged",
        ],
        "numerical_time_series_simulation": True,
        "continuous_network_solver": False,
        "agent_required": False,
        "legacy_r28_retained": True,
        "runtime_default_changed": False,
        "competition_or_rematch_regenerated": False,
        "new_holdout_supplied": False,
        "stop_rule": design["stop_rule"],
        "status": "POST_CONTACT_DEVELOPMENT_REDUCED_ORDER_SIMULATION_AUDIT_ONLY",
        "qualification": "NOT_EXTERNAL_VALIDATION_NOT_PRODUCTION_DEFAULT_NOT_FLIGHT_QUALIFIED",
        "claim_boundary": (
            "R29 was implemented after R28 development outcomes were already observed. The fixed reduced-order resolver is grounded in the public ADAPT graph, "
            "rated light-load power and telemetry boundary conditions rather than case-ID rules, but all DXC09 Sample metrics remain post-contact development evidence. "
            "It is not VirtualADAPT/Simulink, a full network solver, posterior probability, formal EvidenceFact, or external validation."
        ),
    }
    sealed = seal(payload)
    output.write_text(json.dumps(sealed, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "selection": partitions["SELECTION"]["r29_metrics"],
        "internal_replay": partitions["INTERNAL_REPLAY"]["r29_metrics"],
        "combined": sealed["combined_post_contact_development"]["r29_metrics"],
        "changed_case_ids": sealed["changed_case_ids"],
        "physics_application_case_ids": sealed["physics_application_case_ids"],
        "unresolved_case_ids": sealed["unresolved_case_ids"],
        "receipt_sha256": sealed["receipt_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
