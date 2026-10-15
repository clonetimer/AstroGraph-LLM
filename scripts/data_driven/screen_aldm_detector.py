#!/usr/bin/env python3
"""Validation-only ALDM-inspired detector screen on ADCS and PHM.

The script never opens Test telemetry or labels.  It compares the label-free
detector with the existing low-capacity supervised logistic detector under the
same Train-only preprocessing and Validation-only causal alarm-policy search.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.integration import run_adcs_v522_regression as adcs
from astrograph.data_driven.alarm_policy import (
    AlarmPolicy,
    alarm_event_metrics,
    apply_alarm_policy,
    candidate_alarm_policies,
)
from astrograph.data_driven.evaluation_protocol import binary_aupr, multiclass_metrics
from astrograph.data_driven.lab.aldm_detector import (
    ALDMDetectorConfig,
    IQRScoreCalibration,
    fit_aldm_detector,
    score_aldm_detector,
)
from astrograph.data_driven.mechanism_features import (
    compose_columns,
    load_feature_bank,
    profile_columns,
    require_available,
)
from astrograph.data_driven.phm_v60 import audit_split as audit_phm_split
from astrograph.data_driven.phm_v60 import is_allowed_channel, load_records
from astrograph.data_driven.preprocessing import fit_channel_preprocessor


DEFAULT_ADCS = Path(os.environ.get(
    "ASTROGRAPH_ADCS_V522_DATA_ROOT",
    str(ROOT / "data" / "external" / "whole_spacecraft_adcs_paired_dataset_v5_2_2"),
))
DEFAULT_PHM = Path(os.environ.get(
    "ASTROGRAPH_PHM_V60_DATA_ROOT",
    str(ROOT / "data" / "external" / "whole_spacecraft_phm_dataset_v6_0"),
))
FEATURE_BANK = ROOT / "configs/data_driven/mechanism_feature_banks.json"
EXPERIMENT_CONFIG = ROOT / "configs/data_driven_models/aldm_detector_research.v1.json"


@dataclass
class DetectionWindows:
    x: np.ndarray
    truth: np.ndarray
    group_ids: list[str]
    times_s: np.ndarray
    onset_times_s: np.ndarray


def _subset(values: DetectionWindows, limit: int | None) -> DetectionWindows:
    if limit is None or len(values.x) <= limit:
        return values
    indices = np.linspace(0, len(values.x) - 1, limit, dtype=int)
    return DetectionWindows(
        x=values.x[indices],
        truth=values.truth[indices],
        group_ids=[values.group_ids[index] for index in indices],
        times_s=values.times_s[indices],
        onset_times_s=values.onset_times_s[indices],
    )


def _preprocess(
    train: DetectionWindows,
    validation: DetectionWindows,
) -> tuple[DetectionWindows, DetectionWindows, dict[str, object]]:
    state = fit_channel_preprocessor(train.x, method="robust_clip")
    return (
        DetectionWindows(
            state.transform(train.x), train.truth, train.group_ids,
            train.times_s, train.onset_times_s,
        ),
        DetectionWindows(
            state.transform(validation.x), validation.truth, validation.group_ids,
            validation.times_s, validation.onset_times_s,
        ),
        state.receipt(),
    )


def load_adcs_windows(
    root: Path,
    *,
    feature_profile: str,
) -> tuple[DetectionWindows, DetectionWindows, dict[str, Any], list[str]]:
    trajectories = adcs.load_trajectories(root)
    split_audit = adcs.audit_split(trajectories)
    if split_audit["status"] != "PASS":
        raise RuntimeError(split_audit)
    candidates = [row for row in trajectories if row.split != "test"]
    bank = load_feature_bank(FEATURE_BANK)
    columns = compose_columns(
        adcs.FEATURE_COLUMNS,
        profile_columns(bank, "adcs_v5_2_2", feature_profile),
    )
    available = list(pd.read_csv(candidates[0].telemetry_path, nrows=0).columns)
    require_available(columns, available)
    class_names = sorted({
        row.diagnostic_class for row in candidates if row.diagnostic_class != "nominal"
    })
    windows = adcs.build_windows(
        candidates,
        {name: index for index, name in enumerate(class_names)},
        {name: index for index, name in enumerate(adcs.COMPONENT_NAMES)},
        feature_columns=columns,
    )

    def convert(value: adcs.WindowSet) -> DetectionWindows:
        return DetectionWindows(
            x=value.x,
            truth=value.detection,
            group_ids=value.trajectory_ids,
            times_s=value.end_times_s,
            onset_times_s=value.onset_times_s,
        )

    train, validation, preprocessing = _preprocess(
        convert(windows["train"]), convert(windows["validation"])
    )
    return train, validation, {
        "split_audit": split_audit,
        "preprocessing": preprocessing,
        "test_telemetry_or_labels_opened": 0,
    }, columns


def select_phm_channels(root: Path, *, limit: int) -> list[str]:
    index = pd.read_csv(root / "sample_index.csv")
    first_non_test = index[index["sample_index"].map(adcs.split_for_index) != "test"].iloc[0]
    available = list(pd.read_csv(root / first_non_test["telemetry_file"], nrows=0).columns)
    allowed = [name for name in available if is_allowed_channel(str(name))]
    mechanism = profile_columns(
        load_feature_bank(FEATURE_BANK), "phm_v6_0", "mechanism_required"
    )
    require_available(mechanism, available)
    result = compose_columns(mechanism, [name for name in allowed if name not in mechanism])
    return result[:limit]


def load_phm_windows(
    root: Path,
    *,
    channel_limit: int,
    window_size: int,
    stride: int = 5,
) -> tuple[DetectionWindows, DetectionWindows, dict[str, Any], list[str]]:
    records = load_records(root)
    split_audit = audit_phm_split(records)
    if split_audit["status"] != "PASS":
        raise RuntimeError(split_audit)
    channels = select_phm_channels(root, limit=channel_limit)
    index = pd.read_csv(root / "sample_index.csv")
    rows = index[index["sample_index"].map(adcs.split_for_index) != "test"]
    buffers = {
        name: {"x": [], "truth": [], "groups": [], "times": [], "onsets": []}
        for name in ("train", "validation")
    }
    for number, row in enumerate(rows.itertuples(index=False), start=1):
        split = adcs.split_for_index(int(row.sample_index))
        telemetry = pd.read_csv(root / row.telemetry_file, usecols=["time_s", *channels])
        labels = pd.read_csv(root / row.labels_file, usecols=["time_s", "event_active"])
        times = telemetry["time_s"].to_numpy(dtype=float)
        label_times = labels["time_s"].to_numpy(dtype=float)
        if len(times) != len(label_times) or not np.array_equal(times, label_times):
            raise ValueError(f"PHM telemetry/label time mismatch: {row.sample_id}")
        active = labels["event_active"].astype(str).str.lower().isin(
            {"1", "true"}
        ).to_numpy(dtype=bool)
        onset = float(times[np.flatnonzero(active)[0]]) if active.any() else math.nan
        values = telemetry[channels].to_numpy(dtype=np.float32)
        target = buffers[split]
        for end in range(window_size - 1, len(values), stride):
            target["x"].append(values[end - window_size + 1 : end + 1])
            target["truth"].append(int(active[end]))
            target["groups"].append(str(row.sample_id))
            target["times"].append(float(times[end]))
            target["onsets"].append(onset)
        if number % 100 == 0:
            print(f"[phm-windows] {number}/{len(rows)} trajectories", flush=True)

    def finish(name: str) -> DetectionWindows:
        values = buffers[name]
        return DetectionWindows(
            x=np.asarray(values["x"], dtype=np.float32),
            truth=np.asarray(values["truth"], dtype=np.int64),
            group_ids=values["groups"],
            times_s=np.asarray(values["times"], dtype=np.float64),
            onset_times_s=np.asarray(values["onsets"], dtype=np.float64),
        )

    train, validation, preprocessing = _preprocess(finish("train"), finish("validation"))
    return train, validation, {
        "split_audit": split_audit,
        "preprocessing": preprocessing,
        "test_telemetry_or_labels_opened": 0,
    }, channels


def logistic_scores(
    train: DetectionWindows,
    values: DetectionWindows,
    *,
    seed: int,
) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    train_summary = adcs.summary_features(train.x)
    value_summary = adcs.summary_features(values.x)
    model = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=seed
    )
    model.fit(train_summary, train.truth)
    probabilities = model.predict_proba(value_summary)
    return probabilities[:, list(model.classes_).index(1)]


def evaluate_validation_scores(
    values: DetectionWindows,
    scores: Sequence[float],
    *,
    alarm_policy_mode: str,
) -> dict[str, Any]:
    score_values = np.asarray(scores, dtype=np.float64)
    if len(score_values) != len(values.truth):
        raise ValueError("score count does not match Validation windows")
    best: dict[str, Any] | None = None
    policies = candidate_alarm_policies(
        alarm_policy_mode, (0.30, 0.40, 0.50, 0.60, 0.70)
    )
    for policy in policies:
        predicted = np.asarray(apply_alarm_policy(
            score_values.tolist(), values.group_ids, values.times_s.tolist(), policy
        ))
        point = multiclass_metrics(
            values.truth.tolist(), predicted.tolist(), labels=[0, 1]
        )
        event = alarm_event_metrics(
            values.truth.tolist(), predicted.tolist(), values.group_ids,
            values.times_s.tolist(), values.onset_times_s.tolist(),
        )
        candidate = {
            "alarm_policy": policy.to_dict(),
            "point": point,
            "event": event,
        }
        ordering = (
            point["macro_f1"],
            event["fault_event_recall"],
            -event["false_alarm_window_rate"],
            -policy.on_threshold,
        )
        if best is None or ordering > best["_ordering"]:
            best = {**candidate, "_ordering": ordering}
    assert best is not None
    best.pop("_ordering")
    best["detection_aupr"] = binary_aupr(
        (values.truth == 1).tolist(), score_values.tolist()
    )
    best["selection_scope"] = "VALIDATION_ONLY"
    return best


def _no_regression(candidate: float | None, baseline: float | None, *, lower_is_better: bool) -> bool:
    if baseline is None:
        return candidate is None
    if candidate is None:
        return False
    return candidate <= baseline if lower_is_better else candidate >= baseline


def promotion_decision(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    minimum_gain: float,
) -> dict[str, Any]:
    gain = candidate["point"]["macro_f1"] - baseline["point"]["macro_f1"]
    recall_ok = _no_regression(
        candidate["event"]["fault_event_recall"],
        baseline["event"]["fault_event_recall"],
        lower_is_better=False,
    )
    false_alarm_ok = _no_regression(
        candidate["event"]["false_alarm_episodes_per_hour"],
        baseline["event"]["false_alarm_episodes_per_hour"],
        lower_is_better=True,
    )
    delay_ok = _no_regression(
        candidate["event"]["p95_detection_delay_s"],
        baseline["event"]["p95_detection_delay_s"],
        lower_is_better=True,
    )
    passed = gain >= minimum_gain and recall_ok and false_alarm_ok and delay_ok
    return {
        "status": "PASS_CONTINUE_TO_CONFIRMATION" if passed else "STOP_NO_VALIDATION_GAIN",
        "macro_f1_gain": gain,
        "minimum_required_gain": minimum_gain,
        "event_recall_no_regression": recall_ok,
        "false_alarm_rate_no_regression": false_alarm_ok,
        "p95_delay_no_regression": delay_ok,
        "test_authorized": False,
    }


def run_dataset(
    dataset_id: str,
    train: DetectionWindows,
    validation: DetectionWindows,
    metadata: dict[str, Any],
    channels: list[str],
    *,
    implementation: dict[str, Any],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str,
    alarm_policy_mode: str,
    minimum_gain: float,
) -> dict[str, Any]:
    baseline_started = time.perf_counter()
    baseline_scores = logistic_scores(train, validation, seed=seed)
    baseline = evaluate_validation_scores(
        validation, baseline_scores, alarm_policy_mode=alarm_policy_mode
    )
    baseline["training_and_scoring_seconds"] = time.perf_counter() - baseline_started

    config = ALDMDetectorConfig(
        window_size=train.x.shape[1],
        channel_count=train.x.shape[2],
        latent_size=int(implementation["latent_size"]),
        context_size=int(implementation["context_size"]),
        hidden_size=int(implementation["hidden_size"]),
        flow_layers=int(implementation["flow_layers"]),
        contrastive_weight=float(implementation["contrastive_weight"]),
        contrastive_temperature=float(implementation["contrastive_temperature"]),
        assignment_temperature=float(implementation["assignment_temperature"]),
        mask_fraction=float(implementation["mask_fraction"]),
    )
    started = time.perf_counter()
    model, training = fit_aldm_detector(
        train.x,
        config,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        seed=seed,
    )
    raw_train = score_aldm_detector(model, train.x, device=device)
    raw_validation = score_aldm_detector(model, validation.x, device=device)
    calibration = IQRScoreCalibration.fit(raw_train)
    candidate_scores = calibration.transform(raw_validation)
    candidate = evaluate_validation_scores(
        validation, candidate_scores, alarm_policy_mode=alarm_policy_mode
    )
    candidate["training_and_scoring_seconds"] = time.perf_counter() - started
    candidate["training"] = training
    candidate["calibration"] = calibration.receipt()
    candidate["model_config"] = config.receipt()
    return {
        "dataset": dataset_id,
        "window_counts": {"train": len(train.x), "validation": len(validation.x)},
        "channel_count": len(channels),
        "channels": channels,
        **metadata,
        "baseline": {"family": "class_balanced_logistic_window_summary", **baseline},
        "candidate": {"family": "aldm_inspired_label_free_detector", **candidate},
        "promotion_decision": promotion_decision(
            baseline, candidate, minimum_gain=minimum_gain
        ),
    }


def render(report: dict[str, Any]) -> str:
    rows = []
    for result in report["datasets"]:
        for key in ("baseline", "candidate"):
            item = result[key]
            event = item["event"]
            rows.append(
                f"| {result['dataset']} | {item['family']} | "
                f"{item['point']['macro_f1']:.4f} | {item['detection_aupr']:.4f} | "
                f"{event['fault_event_recall']:.4f} | "
                f"{event['false_alarm_episodes_per_hour']} | "
                f"{event['p95_detection_delay_s']} |"
            )
    decisions = "\n".join(
        f"- {item['dataset']}: `{item['promotion_decision']['status']}`, "
        f"Validation Macro-F1 gain `{item['promotion_decision']['macro_f1_gain']:.6f}`"
        for item in report["datasets"]
    )
    return """# ALDM-inspired detector Validation-only screen

This is an independent ALDM-inspired E0 research implementation, not the
authors' official reproduction. Test telemetry and labels were not opened.

| Dataset | Model | Detection Macro-F1 | AUPR | Event recall | False alarms/hour | p95 delay (s) |
|---|---|---:|---:|---:|---:|---:|
""" + "\n".join(rows) + "\n\n## Stop-gate decisions\n\n" + decisions + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=["adcs", "phm"], default=["adcs", "phm"])
    parser.add_argument("--adcs-root", type=Path, default=DEFAULT_ADCS)
    parser.add_argument("--phm-root", type=Path, default=DEFAULT_PHM)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-train-windows", type=int)
    parser.add_argument("--max-validation-windows", type=int)
    parser.add_argument("--alarm-policy", choices=["point", "hysteresis"], default="hysteresis")
    args = parser.parse_args()

    frozen = json.loads(EXPERIMENT_CONFIG.read_text(encoding="utf-8"))
    training = frozen["training"]
    implementation = frozen["implementation"]
    epochs = args.epochs or int(training["epochs"])
    batch_size = args.batch_size or int(training["batch_size"])
    seed = args.seed or int(training["seeds"][0])
    minimum_gain = float(frozen["promotion_gate"]["minimum_validation_macro_f1_gain"])
    results = []
    if "adcs" in args.datasets:
        train, validation, metadata, channels = load_adcs_windows(
            args.adcs_root,
            feature_profile=frozen["datasets"]["adcs_v5_2_2"]["feature_profile"],
        )
        results.append(run_dataset(
            "adcs_v5_2_2", _subset(train, args.max_train_windows),
            _subset(validation, args.max_validation_windows), metadata, channels,
            implementation=implementation, epochs=epochs, batch_size=batch_size,
            learning_rate=float(training["learning_rate"]), seed=seed, device=args.device,
            alarm_policy_mode=args.alarm_policy, minimum_gain=minimum_gain,
        ))
    if "phm" in args.datasets:
        train, validation, metadata, channels = load_phm_windows(
            args.phm_root,
            channel_limit=int(frozen["datasets"]["phm_v6_0"]["channel_limit"]),
            window_size=int(implementation["window_size"]),
        )
        results.append(run_dataset(
            "phm_v6_0", _subset(train, args.max_train_windows),
            _subset(validation, args.max_validation_windows), metadata, channels,
            implementation=implementation, epochs=epochs, batch_size=batch_size,
            learning_rate=float(training["learning_rate"]), seed=seed, device=args.device,
            alarm_policy_mode=args.alarm_policy, minimum_gain=minimum_gain,
        ))
    report = {
        "schema_version": "astrograph-aldm-inspired-validation-screen.v1",
        "experiment_config": str(EXPERIMENT_CONFIG.relative_to(ROOT)),
        "evaluation_scope": "VALIDATION_ONLY_TEST_NOT_OPENED",
        "device": args.device,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "smoke_limits": {
            "max_train_windows": args.max_train_windows,
            "max_validation_windows": args.max_validation_windows,
        },
        "datasets": results,
        "claim_boundary": (
            "E0 synthetic Validation-only screening; no Test, localization, classification, "
            "KG, BiRRF, RAG, Runtime or production claim."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(render(report), encoding="utf-8")
    print(args.output_dir / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
