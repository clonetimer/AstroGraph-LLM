#!/usr/bin/env python3
"""Build a frozen R36 DEVELOPMENT case artifact from ADCS v5.2.2.

The adapter reuses AstroGraph's existing leakage-controlled ADCS v5.2.2
engineering harness, trains the predeclared sklearn family on the frozen train
split, and aggregates fixed-threshold window predictions to the frozen test
trajectory cohort.  It never promotes the result beyond DEVELOPMENT.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "scripts" / "integration" / "run_adcs_v522_regression.py"


def load_harness():
    spec = importlib.util.spec_from_file_location("astrograph_adcs_v522_harness", HARNESS)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load ADCS v5.2.2 harness")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    h = load_harness()
    trajectories = h.load_trajectories(args.dataset_root)
    split_audit = h.audit_split(trajectories)
    if split_audit["status"] != "PASS":
        raise ValueError("ADCS frozen split audit failed")

    class_names = sorted({row.diagnostic_class for row in trajectories if row.diagnostic_class != "nominal"})
    component_names = sorted(set(h.FAULT_TO_COMPONENT.values()))
    class_to_index = {name: index for index, name in enumerate(class_names)}
    component_to_index = {name: index for index, name in enumerate(component_names)}

    windows = h.build_windows(trajectories, class_to_index, component_to_index)
    from astrograph.evaluation.data_exposure import dataset_use
    consumed_group_ids = [
        row.sample_id for row in trajectories if row.split in {"train", "test"}
    ]
    with dataset_use(
        "SYNTHETIC_PROTOCOL", "DEVELOPMENT", "FIT", group_ids=consumed_group_ids
    ) as development_provenance:
        impute, mean, scale = h.fit_train_scaler(windows["train"].x)
        train = h.scale_windows(windows["train"], impute, mean, scale)
        test = h.scale_windows(windows["test"], impute, mean, scale)
        report, det_score, component_prob, conditioned_prob = h.run_ml(
            train, test, class_names, component_names
        )
    threshold = float(report["detection_threshold"])
    by_sample: dict[str, list[int]] = {}
    for index, sample_id in enumerate(test.trajectory_ids):
        by_sample.setdefault(sample_id, []).append(index)

    test_rows = [row for row in trajectories if row.split == "test"]
    cases: list[dict] = []
    for row in test_rows:
        indices = by_sample[row.sample_id]
        label_times, active, onset = h._read_truth(row.labels_path)
        duration_s = float(label_times[-1] - label_times[0])
        if duration_s <= 0:
            raise ValueError(f"non-positive trajectory duration: {row.sample_id}")
        true_fault = row.diagnostic_class != "nominal"
        true_onset = None if not true_fault or math.isnan(onset) else float(onset)

        ordered = sorted(indices, key=lambda i: float(test.end_times_s[i]))
        alarm_starts = [
            index
            for position, index in enumerate(ordered)
            if det_score[index] >= threshold
            and (position == 0 or det_score[ordered[position - 1]] < threshold)
        ]
        first_alarm_index = alarm_starts[0] if alarm_starts else None

        # Alarm episodes are generated without truth.  Truth onset is joined
        # only here, during evaluation, to select a new episode that begins
        # after the event.  An alarm already active before onset is reported as
        # a pre-onset alarm and cannot masquerade as zero-delay detection.
        if true_fault:
            eligible = [
                i for i in alarm_starts
                if true_onset is not None and test.end_times_s[i] >= true_onset
            ]
        else:
            eligible = alarm_starts

        detected = bool(eligible)
        detection_index = min(eligible, key=lambda i: float(test.end_times_s[i])) if eligible else None
        detection_time = float(test.end_times_s[detection_index]) if detection_index is not None else None
        predicted_component = (
            component_names[int(np.argmax(component_prob[detection_index]))]
            if detection_index is not None else None
        )
        predicted_fault = (
            class_names[int(np.argmax(conditioned_prob[detection_index]))]
            if detection_index is not None else None
        )
        cases.append({
            "case_id": row.sample_id,
            "true_detected": true_fault,
            "predicted_detected": detected,
            "duration_s": duration_s,
            "true_onset_s": true_onset,
            "first_alarm_time_s": (
                float(test.end_times_s[first_alarm_index])
                if first_alarm_index is not None else None
            ),
            "detection_time_s": detection_time,
            "true_component_id": h.FAULT_TO_COMPONENT[row.diagnostic_class] if true_fault else None,
            "predicted_component_id": predicted_component,
            "true_fault_mode_id": row.diagnostic_class if true_fault else None,
            "predicted_fault_mode_id": predicted_fault,
            "true_open_set": False,
            "abstained": bool(true_fault and not detected),
        })

    payload = {
        "schema_version": "astrograph-r36-diagnostic-cases.v1",
        "dataset_id": "ADCS_V5_2_2_USER_SUPPLIED",
        "protocol_id": "R36_ADCS_DEV_SKLEARN_LOGREG_V1",
        "cohort_id": "ADCS_V5_2_2_FROZEN_TEST_SAMPLE_INDEX_4_9",
        "evidence_class": "DEVELOPMENT",
        "model_family": "machine_learning",
        "detection_threshold": threshold,
        "case_unit": "trajectory",
        "prediction_aggregation": (
            "fixed-threshold alarm episodes are generated independently of truth; "
            "fault detection time is the first new episode beginning at/after truth onset, "
            "with pre-onset alarms reported separately"
        ),
        "split_audit": split_audit,
        "development_provenance": {
            **{key: value for key, value in development_provenance.items() if key != "group_ids"},
            "group_count": len(development_provenance.get("group_ids") or []),
        },
        "harness_report": report,
        "cases": cases,
        "claim_boundary": (
            "Synthetic ADCS v5.2.2 development evaluation using the frozen train/test split and a "
            "predeclared sklearn family. It is not flight evidence or qualification."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    args.out.write_bytes(raw)
    print(json.dumps({
        "out": str(args.out),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "case_count": len(cases),
        "fault_case_count": sum(item["true_detected"] for item in cases),
        "normal_case_count": sum(not item["true_detected"] for item in cases),
        "detection_threshold": threshold,
        "model_family": "machine_learning",
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
