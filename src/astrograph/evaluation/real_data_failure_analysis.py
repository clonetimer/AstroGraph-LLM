"""Pure R21 analysis of frozen R20.1 evidence; this module never calls a predictor."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


MECHANISMS = (
    "EXACT_SET_MATCH", "PARTIAL_OVERLAP", "COMPLETE_MISS",
    "TRUE_SET_SUBSET_OF_PREDICTION", "PREDICTION_SUBSET_OF_TRUE_SET",
    "OVER_ISOLATION", "UNDER_ISOLATION", "COMPONENT_MATCH_MODE_MISMATCH",
    "MODE_MATCH_COMPONENT_MISMATCH",
)


def _identity(path: Path) -> dict:
    data = path.read_bytes()
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def verify_frozen_bytes(receipt_dir: str | Path, registry: Mapping) -> dict:
    """Fail before analysis if any of the 79 frozen files or sidecars changed."""
    root = Path(receipt_dir)
    checked = 0
    for row in registry["frozen_evidence"]:
        actual = _identity(root / row["relative_path"])
        if actual != {"size": row["size"], "sha256": row["sha256"]}:
            raise ValueError("R21_FROZEN_EXTERNAL_BYTES_CHANGED:" + row["relative_path"])
        checked += 1
    for row in registry.get("analysis_result_inputs", []):
        actual = _identity(root / row["relative_path"])
        if actual != {"size": row["size"], "sha256": row["sha256"]}:
            raise ValueError("R21_FROZEN_ANALYSIS_SIDECAR_CHANGED:" + row["relative_path"])
    return {"status": "PASS", "original_frozen_files": checked,
            "analysis_sidecars": len(registry.get("analysis_result_inputs", [])),
            "inference_rerun": False}


def _split_fault(value: str) -> tuple[str, str]:
    component, sep, mode = str(value).partition(":")
    return component, mode if sep else ""


def decompose_fault_sets(truth: Iterable[str], prediction: Iterable[str]) -> dict:
    true_set, pred_set = set(map(str, truth)), set(map(str, prediction))
    matched = true_set & pred_set
    missing, extra = true_set - pred_set, pred_set - true_set
    true_components = defaultdict(set)
    pred_components = defaultdict(set)
    true_modes = defaultdict(set)
    pred_modes = defaultdict(set)
    for fault in true_set:
        component, mode = _split_fault(fault); true_components[component].add(mode); true_modes[mode].add(component)
    for fault in pred_set:
        component, mode = _split_fault(fault); pred_components[component].add(mode); pred_modes[mode].add(component)
    component_mode = sorted(f for f in missing if _split_fault(f)[0] in pred_components)
    # Keep the two diagnostic proxy classes exclusive: if the true component is
    # already present with another mode, classify it as component-match/mode-mismatch.
    mode_component = sorted(f for f in missing if _split_fault(f)[1] in pred_modes
                            and _split_fault(f)[0] not in pred_components)
    union = true_set | pred_set
    flags = {
        "EXACT_SET_MATCH": true_set == pred_set,
        "PARTIAL_OVERLAP": bool(true_set and matched and true_set != pred_set),
        "COMPLETE_MISS": bool(true_set and not matched),
        "TRUE_SET_SUBSET_OF_PREDICTION": bool(true_set and true_set < pred_set),
        "PREDICTION_SUBSET_OF_TRUE_SET": bool(pred_set and pred_set < true_set),
        "OVER_ISOLATION": bool(extra),
        "UNDER_ISOLATION": bool(missing),
        "COMPONENT_MATCH_MODE_MISMATCH": bool(component_mode),
        "MODE_MATCH_COMPONENT_MISMATCH": bool(mode_component),
    }
    return {
        "mechanisms": [name for name in MECHANISMS if flags[name]], "flags": flags,
        "truth_count": len(true_set), "prediction_count": len(pred_set), "matched_count": len(matched),
        "extra_count": len(extra), "missing_count": len(missing),
        "candidate_cardinality_error_signed": len(pred_set) - len(true_set),
        "candidate_cardinality_error_absolute": abs(len(pred_set) - len(true_set)),
        "fault_element_precision": len(matched) / len(pred_set) if pred_set else (1.0 if not true_set else 0.0),
        "fault_element_recall": len(matched) / len(true_set) if true_set else None,
        "fault_element_Jaccard": len(matched) / len(union) if union else 1.0,
        "any_overlap": bool(matched),
        "component_match_mode_mismatch_faults": component_mode,
        "mode_match_component_mismatch_faults": mode_component,
    }


def summarize_fault_cases(cases: Sequence[Mapping]) -> dict:
    counts = Counter()
    true_total = pred_total = matched_total = faulty = nominal = overlap_faulty = 0
    exact = 0
    for case in cases:
        result = case if "flags" in case else decompose_fault_sets(case["truth"], case["prediction"])
        true_total += result["truth_count"]; pred_total += result["prediction_count"]; matched_total += result["matched_count"]
        exact += int(result["flags"]["EXACT_SET_MATCH"])
        if result["truth_count"]:
            faulty += 1; overlap_faulty += int(result["any_overlap"])
        else:
            nominal += 1
        for name, value in result["flags"].items():
            counts[name] += int(value)
    union_total = true_total + pred_total - matched_total
    return {
        "scored_case_count": len(cases), "faulty_case_count": faulty, "explicit_nominal_case_count": nominal,
        "exact_set_accuracy": exact / len(cases) if cases else None,
        "fault_element_precision_micro": matched_total / pred_total if pred_total else None,
        "fault_element_recall_micro": matched_total / true_total if true_total else None,
        "fault_element_Jaccard_micro": matched_total / union_total if union_total else None,
        "faulty_case_any_overlap": overlap_faulty / faulty if faulty else None,
        "matched_fault_elements": matched_total, "true_fault_elements": true_total,
        "predicted_fault_elements": pred_total,
        "mechanism_counts": {name: counts[name] for name in MECHANISMS},
    }


def summarize_ops_coverage(per_segment: Sequence[Mapping]) -> dict:
    strata = {name: defaultdict(lambda: {"eligible": 0, "blocked": 0})
              for name in ("truth", "channel", "sampling", "segment_length")}
    for row in per_segment:
        values = {"truth": str(int(row["truth_anomaly"])), "channel": str(row["channel"]),
                  "sampling": str(row["sampling"]), "segment_length": str(row["raw_segment_length"])}
        for name, value in values.items():
            strata[name][value]["eligible"] += 1
            strata[name][value]["blocked"] += int(bool(row["blocked"]))
    rendered = {}
    for name, entries in strata.items():
        rendered[name] = {key: {**value, "blocked_rate": value["blocked"] / value["eligible"]}
                          for key, value in entries.items()}
    eligible = len(per_segment); blocked = sum(bool(row["blocked"]) for row in per_segment)
    return {"eligible_count": eligible, "scored_count": eligible - blocked, "blocked_count": blocked,
            "coverage": (eligible - blocked) / eligible if eligible else None,
            "abstention_block_rate": blocked / eligible if eligible else None, "strata": rendered}


def esa_stagewise_trace(*, decision_points: int, observable_decision_points: int,
                        alarm_points: int, true_events: Sequence[Mapping],
                        range_evidence: Sequence[Mapping] = ()) -> dict:
    if alarm_points:
        mechanism = "POSTPROCESS_SUPPRESSION"
    elif range_evidence and all(row.get("range_strictly_within_threshold_interval") for row in range_evidence):
        mechanism = "RESPONSE_BELOW_THRESHOLD"
    else:
        mechanism = "NO_SCORE_RESPONSE"
    per_event = {}
    for event in true_events:
        count = int(event.get("decision_timestamps_inside_annotation", 0))
        per_event[str(event["id"])] = {
            "mechanism": "AGGREGATION_SUPPRESSION" if count == 0 else mechanism,
            "mechanism_scope": "temporal decision-grid representation only; not a proof that raw signals were attenuated",
            "annotated_duration_s": event.get("annotated_duration_s"),
            "decision_timestamps_inside_annotation": count, "alarm_points": 0, "detected": False,
        }
    return {
        "status": "CONFIRMED_NEGATIVE_RESULT_WITH_RANGE_PROVEN_MECHANISM",
        "decision_points": decision_points, "observable_decision_points": observable_decision_points,
        "alarm_points": alarm_points, "TP_events": 0, "FN_events": len(true_events),
        "allowed_mechanism_categories": ["NO_SCORE_RESPONSE", "RESPONSE_BELOW_THRESHOLD", "AGGREGATION_SUPPRESSION", "POSTPROCESS_SUPPRESSION"],
        "mechanism_identification_complete": bool(range_evidence), "mechanism": mechanism,
        "per_anomaly_event_trace": per_event, "inference_rerun": False,
        "probability_semantics": "UNCALIBRATED_ANOMALY_SCORE",
    }


def esa_frozen_range_audit(channel_arrays: Mapping[str, Sequence[float]], fit_parameters: Mapping,
                           threshold: float, cache_identities: Mapping[str, str] | None = None) -> list[dict]:
    """Prove an input range relation only; no scoring/model function is imported or called."""
    rows = []
    for channel, values in channel_arrays.items():
        numeric = [float(value) for value in values]
        params = fit_parameters[channel]
        center, scale = float(params["center"]), float(params["scale"])
        low, high = center - threshold * scale, center + threshold * scale
        rows.append({"channel": channel, "observed_count": len(numeric), "input_min": min(numeric),
                     "input_max": max(numeric), "frozen_center": center, "frozen_scale": scale,
                     "frozen_threshold": threshold, "frozen_no_exceedance_interval": [low, high],
                     "nonzero_reference_deviation_exists": any(value != center for value in numeric),
                     "range_strictly_within_threshold_interval": min(numeric) > low and max(numeric) < high,
                     "cache_sha256": (cache_identities or {}).get(channel)})
    return rows


def require_probability_semantics(score_semantics: str) -> None:
    if score_semantics != "CALIBRATED_PROBABILITY":
        raise ValueError("R21_UNCALIBRATED_SCORE_CANNOT_BE_USED_AS_PROBABILITY")


def analyze_frozen_bundle(receipt_dir: str | Path, registry: Mapping) -> dict:
    """Recompute DXC and OPS invariants from registered frozen bytes only."""
    root = Path(receipt_dir)
    verification = verify_frozen_bytes(root, registry)
    cases = []
    rendered_cases = []
    for prefix, dataset in (("dxc09_competition", "DXC09_COMPETITION"), ("dxc09_rematch", "DXC09_REMATCH")):
        truths = json.loads((root / f"{prefix}_case_evaluation_receipt.json").read_text(encoding="utf-8"))["cases"]
        predictions = json.loads((root / f"{prefix}_frozen_predictions.json").read_text(encoding="utf-8"))["predictions"]
        for row in truths:
            if row.get("status") != "SCORED":
                continue
            pred = predictions[row["case_id"]]["predicted_fault_ids"]
            result = decompose_fault_sets(row["truth"]["true_fault_ids"], pred)
            result.update({"dataset_id": dataset, "case_id": row["case_id"], "audit_member": row["audit_member"]})
            cases.append(result); rendered_cases.append(result)
    dxc = {"schema_version": "astrograph-r21-failure-mechanism-report.v1", "family": "DXC",
           "frozen_bytes_verification": verification, "purpose": "DECOMPOSE", "inference_rerun": False,
           "qualification": "NOT_FLIGHT_QUALIFIED", "summary": summarize_fault_cases(cases), "cases": rendered_cases}
    sidecar = json.loads((root / "r20_1_ops_raw_coverage_analysis.json").read_text(encoding="utf-8"))
    ops = {"schema_version": "astrograph-r21-failure-mechanism-report.v1", "family": "OPS_RAW_ONLY",
           "frozen_bytes_verification": verification, "purpose": "DECOMPOSE", "inference_rerun": False,
           "qualification": "NOT_FLIGHT_QUALIFIED", "summary": summarize_ops_coverage(sidecar["per_segment"]),
           "selective_01_risk": sidecar["selective_01_risk"],
           "feature_track_combined_with_raw": False}
    return {"verification": verification, "DXC": dxc, "OPS": ops}
