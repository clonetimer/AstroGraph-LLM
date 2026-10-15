#!/usr/bin/env python3
"""R31 truth-aligned public-sample WaterPump mechanism-fit audit.

This runner intentionally uses the DXC09 Sample injection timestamp only to
align a physics-fit window for two already-contacted WaterPump examples.  It is
therefore an AUDIT artifact, not an online diagnosis benchmark.  Candidate mode
selection itself remains truth-free: a fixed FailedOff base activates the
corroborator and the physical best-fit mode is frozen before the true mode is
joined for descriptive agreement reporting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from astrograph.diagnosis.dxc_pump_reduced_order import DXCWaterPumpPhysicsConfig, evaluate_dxc_water_pump_physics
from astrograph.evaluation.data_exposure import dataset_use, registry
from astrograph.evaluation.external_dataset_contract import seal
from astrograph.evaluation.nasa_adapt_adapter import parse_scenario, system_catalog
from astrograph.simulation.adapt_pump_reduced_order import extract_water_pump_type_parameters

SOURCE_ROOT = Path(__file__).resolve().parents[2]
CASES = ("Exp_659_pb_t2", "Exp_660_pb_t2")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-root", required=True)
    ap.add_argument("--system-xml", required=True)
    ap.add_argument("--design", default=str(SOURCE_ROOT / "configs/evaluation/dxc_r31_waterpump_reduced_order.v1.json"))
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    sample_root = Path(args.sample_root).resolve()
    xml_path = Path(args.system_xml).resolve()
    design_path = Path(args.design).resolve()
    design = json.loads(design_path.read_text(encoding="utf-8"))
    config = DXCWaterPumpPhysicsConfig.model_validate(design["config"])
    xml_bytes = xml_path.read_bytes()
    catalog = system_catalog(xml_bytes)
    type_parameters = extract_water_pump_type_parameters(xml_bytes)
    allowed = set(registry()["datasets"]["DXC09_SAMPLE"]["group_ids_by_partition"]["DEVELOPMENT"])

    rows = []
    group_ids = []
    inputs = []
    for case_id in CASES:
        path = sample_root / f"{case_id}.scn"
        raw = path.read_bytes()
        gid = _sha(raw)
        if gid not in allowed:
            raise ValueError(f"R31_PUMP_SAMPLE_NOT_REGISTERED_DEVELOPMENT:{case_id}")
        group_ids.append(gid)
        inputs.append((case_id, raw, gid))

    with dataset_use("DXC09_SAMPLE", "DEVELOPMENT", "AUDIT", group_ids=group_ids):
        for case_id, raw, gid in inputs:
            records, truth = parse_scenario(raw.decode("utf-8"))
            if truth["injection_time"] is None or len(truth["true_fault_ids"]) != 1:
                raise ValueError(f"R31_PUMP_AUDIT_REQUIRES_ONE_INJECTION:{case_id}")
            true_fault = truth["true_fault_ids"][0]
            component_id = true_fault.split(":", 1)[0]
            if component_id != "PMP420" or true_fault.rsplit(":", 1)[1] not in {"FailedOff", "FlowBlocked"}:
                raise ValueError(f"R31_PUMP_AUDIT_UNEXPECTED_FAULT:{case_id}:{true_fault}")
            transition = float(truth["injection_time"]) / 1000.0
            # Fixed base activation.  The physical-preferred mode below is not
            # seeded with the true mode.
            decision = evaluate_dxc_water_pump_physics(
                records=records,
                catalog=catalog,
                type_parameters=type_parameters,
                base_fault_ids=("PMP420:FailedOff",),
                transition_time_s=transition,
                config=config,
            )
            preferred = decision.physics_preferred_fault_ids[0] if len(decision.physics_preferred_fault_ids) == 1 else None
            rows.append({
                "case_id": case_id,
                "group_id": gid,
                "truth_aligned_transition_time_s": transition,
                "true_fault_id": true_fault,
                "physics_preferred_fault_id": preferred,
                "mode_agreement": preferred == true_fault,
                "application_status": decision.application_status,
                "candidate_fits": [item.model_dump(mode="json") for item in decision.candidate_fits],
                "decision_sha256": decision.result_sha256,
            })

    payload = {
        "schema_version": "astrograph-r31-waterpump-public-sample-mechanism-audit.v1",
        "dataset_id": "DXC09_SAMPLE",
        "partition": "DEVELOPMENT_AUDIT_ONLY",
        "source_system_xml_sha256": _sha(xml_bytes),
        "design_sha256": _sha(design_path.read_bytes()),
        "case_count": len(rows),
        "mode_agreement_count": sum(row["mode_agreement"] for row in rows),
        "mode_agreement_rate": sum(row["mode_agreement"] for row in rows) / len(rows),
        "rows": rows,
        "window_alignment": "FAULT_INJECTION_TRUTH_TIMESTAMP_USED_FOR_MECHANISM_FIT_AUDIT_ONLY",
        "prediction_semantics": "FIXED_BASE_ACTIVATION_THEN_TRUTH_FREE_PHYSICAL_MODE_PREFERENCE",
        "status": "TRUTH_ALIGNED_PUBLIC_SAMPLE_MECHANISM_FIT_AUDIT_NOT_ONLINE_DIAGNOSIS",
        "stop_rule": design["stop_rule"],
        "claim_boundary": (
            "The public DXC09 Sample injection timestamp is used to align the mechanism-fit window, so this artifact cannot support detection-delay, "
            "online localization/classification, independent-validation, or SOTA claims. It only checks whether the frozen WaterPump reduced-order model "
            "orders the two public pump modes consistently with already-known sample faults."
        ),
    }
    result = seal(payload)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "case_count": result["case_count"],
        "mode_agreement_rate": result["mode_agreement_rate"],
        "receipt_sha256": result["receipt_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
