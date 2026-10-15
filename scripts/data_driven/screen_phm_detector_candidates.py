#!/usr/bin/env python3
"""Validation-only PHM screen for independently selectable detector candidates."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.data_driven import compare_phm_causal_windows as phm
from astrograph.data_driven.phm_v60 import audit_split, load_records


CANDIDATES = {
    "residual_legacy": ("residual_tcn", "legacy_summary", ("none",), "detection", 0.0, None, 1.0, 0.0, 0.2),
    "residual_change_logistic": ("residual_tcn", "change_summary", ("none",), "detection", 0.0, None, 1.0, 0.0, 0.2),
    "multiview_legacy_logistic": ("causal_multiview_detector_tcn", "legacy_summary", ("none",), "detection", 0.0, None, 1.0, 0.0, 0.2),
    "multiview_change_logistic": ("causal_multiview_detector_tcn", "change_summary", ("none",), "detection", 0.0, None, 1.0, 0.0, 0.2),
    "residual_initial_calibration": ("residual_tcn", "legacy_summary", ("initial_logit_delta",), "detection", 0.0, None, 1.0, 0.0, 0.2),
    "multiview_initial_calibration": (
        "causal_multiview_detector_tcn", "legacy_summary", ("initial_logit_delta",), "detection", 0.0, None, 1.0, 0.0, 0.2
    ),
    "residual_aux_0_1": ("residual_tcn", "legacy_summary", ("none",), "detection_aux", 0.1, None, 1.0, 0.0, 0.2),
    "residual_aux_0_3": ("residual_tcn", "legacy_summary", ("none",), "detection_aux", 0.3, None, 1.0, 0.0, 0.2),
    "multiview_aux_0_1": (
        "causal_multiview_detector_tcn", "legacy_summary", ("none",), "detection_aux", 0.1, None, 1.0, 0.0, 0.2
    ),
    "multiview_aux_fixed_0_6": (
        "causal_multiview_detector_tcn", "legacy_summary", ("none",), "detection_aux", 0.1, 0.6, 1.0, 0.0, 0.2
    ),
    "multiview_aux_nominal_2": (
        "causal_multiview_detector_tcn", "legacy_summary", ("none",), "detection_aux", 0.1, None, 2.0, 0.0, 0.2
    ),
    "multiview_aux_nominal_4": (
        "causal_multiview_detector_tcn", "legacy_summary", ("none",), "detection_aux", 0.1, None, 4.0, 0.0, 0.2
    ),
    "multiview_aux_nominal_4_cvar": (
        "causal_multiview_detector_tcn", "legacy_summary", ("none",), "detection_aux", 0.1, None, 4.0, 0.25, 0.2
    ),
}
DEFAULT_CANDIDATES = tuple(name for name in CANDIDATES if "nominal_" not in name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=phm.DEFAULT_DATASET)
    parser.add_argument(
        "--candidates", nargs="+", choices=sorted(CANDIDATES), default=list(DEFAULT_CANDIDATES)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260826, 20260827, 20260828])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--minimum-epochs", type=int, default=20)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--channel-limit", type=int, default=96)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--label-contract", choices=["event_active", "manifestation_aware"], default="manifestation_aware")
    parser.add_argument("--training-sampling", choices=["window_shuffle", "trajectory_uniform"], default="trajectory_uniform")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    hierarchy = json.loads(phm.HIERARCHY.read_text(encoding="utf-8"))
    source = load_records(args.dataset_root)
    split_audit = audit_split(source)
    if split_audit["status"] != "PASS":
        raise RuntimeError(split_audit)
    records = [row for row in source if row.split != "test"]
    class_names = sorted(hierarchy["fault_to_component"])
    component_names = sorted(set(hierarchy["fault_to_component"].values()))
    selected, median = phm.select_train_channels(
        records, class_names, args.channel_limit, label_contract=args.label_contract
    )
    unscaled = phm.build_windows(
        records, selected, class_names, component_names, hierarchy["fault_to_component"],
        label_contract=args.label_contract,
    )
    windows = phm.scale_windows({name: unscaled[name] for name in ("train", "validation")}, median)

    runs = []
    for candidate in args.candidates:
        (
            model, feature_profile, score_profiles, task_scope, auxiliary_weight,
            fixed_threshold, nominal_weight, cvar_weight, cvar_fraction,
        ) = CANDIDATES[candidate]
        for seed in args.seeds:
            result = phm.train_one(
                model, windows["train"], windows["validation"], windows["validation"],
                class_names, component_names, hierarchy["fault_to_component"], seed, args.epochs,
                selected_channels=selected, device=args.device, batch_size=args.batch_size,
                minimum_epochs=args.minimum_epochs, early_stop_patience=args.early_stop_patience,
                evaluate_test_split=False, label_contract=args.label_contract,
                training_sampling=args.training_sampling, task_scope=task_scope,
                detector_feature_profile=feature_profile,
                detection_score_normalization_profiles=score_profiles,
                diagnosis_auxiliary_weight=auxiliary_weight,
                fixed_detection_threshold=fixed_threshold,
                nominal_trajectory_weight=nominal_weight,
                detection_cvar_weight=cvar_weight,
                detection_cvar_fraction=cvar_fraction,
            )
            runs.append({"candidate": candidate, **result})
    aggregate = {}
    for candidate in args.candidates:
        values = [row for row in runs if row["candidate"] == candidate]
        scores = [float(row["best_validation"]["detection_macro_f1"]) for row in values]
        aggregate[candidate] = {
            "detection_macro_f1_mean": statistics.fmean(scores),
            "detection_macro_f1_values": scores,
            "selected_detector": [row["best_validation"]["detector"] for row in values],
            "selected_threshold": [row["best_validation"]["detection_threshold"] for row in values],
        }
    report = {
        "schema_version": "astrograph-phm-detector-candidate-screen.v1",
        "protocol": {
            "scope": "validation_only; test records excluded before telemetry loading",
            "label_contract": args.label_contract,
            "training_sampling": args.training_sampling,
            "seeds": args.seeds,
            "candidate_contracts": {
                name: {
                    "deep_model": CANDIDATES[name][0],
                    "logistic_features": CANDIDATES[name][1],
                    "score_normalization_profiles": CANDIDATES[name][2],
                    "task_scope": CANDIDATES[name][3],
                    "diagnosis_auxiliary_weight": CANDIDATES[name][4],
                    "fixed_detection_threshold": CANDIDATES[name][5],
                    "nominal_trajectory_weight": CANDIDATES[name][6],
                    "detection_cvar_weight": CANDIDATES[name][7],
                    "detection_cvar_fraction": CANDIDATES[name][8],
                }
                for name in args.candidates
            },
        },
        "split_audit": split_audit,
        "selected_channels": selected,
        "runs": runs,
        "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
