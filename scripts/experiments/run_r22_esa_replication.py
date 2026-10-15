from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from astrograph.evaluation.external_dataset_contract import file_identity
from astrograph.evaluation.replication_r22 import (
    build_esa_preprocessed_input_receipt,
    esa_channel_payload_identity,
    evaluate_esa_transport_replication,
    generate_esa_transport_predictions,
    load_replication_plan,
    verify_new_holdout_admission,
)


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_channel_payload(path: Path, channel: str) -> dict:
    with np.load(path, allow_pickle=False) as data:
        required = {"time_ns", "value", "observed"}
        if not required.issubset(set(data.files)):
            raise ValueError("R22_ESA_PREPROCESSED_NPZ_SCHEMA_INVALID:" + channel)
        payload = {
            "channel": channel,
            "time_ns": np.asarray(data["time_ns"], dtype=np.int64),
            "value": np.asarray(data["value"], dtype=np.float64),
            "observed": np.asarray(data["observed"], dtype=np.bool_),
        }
    esa_channel_payload_identity(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run frozen R22 ESA transport replication on admitted preprocessed input.")
    parser.add_argument("--source", type=Path, required=True, help="Original new-holdout source bound by intake receipt")
    parser.add_argument("--intake-receipt", type=Path, required=True)
    parser.add_argument("--group-admission", type=Path, required=True)
    parser.add_argument("--preprocessed-dir", type=Path, required=True)
    parser.add_argument("--preprocessor-source", type=Path, required=True)
    parser.add_argument("--evaluation-range-json", type=Path, required=True,
                        help="JSON [lower_ns, upper_ns] derived without truth labels")
    parser.add_argument("--truth-json", type=Path,
                        help="Optional JSON object {full_range_ns:[...], events:[...]}; read only after prediction receipt is written")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plan = load_replication_plan()
    intake = json.loads(args.intake_receipt.read_text(encoding="utf-8"))
    groups = json.loads(args.group_admission.read_text(encoding="utf-8"))
    verify_new_holdout_admission(args.source, intake, groups)
    spec = plan["families"]["ESA"]
    evaluation_range = json.loads(args.evaluation_range_json.read_text(encoding="utf-8"))
    if not isinstance(evaluation_range, list):
        raise ValueError("R22_ESA_EVALUATION_RANGE_JSON_MUST_BE_LIST")

    payloads: dict[str, dict] = {}
    artifacts: dict[str, dict] = {}
    for channel in spec["channels"]:
        path = args.preprocessed_dir / f"{channel}.npz"
        if not path.is_file() or path.is_symlink():
            raise ValueError("R22_ESA_PREPROCESSED_CHANNEL_FILE_MISSING:" + channel)
        payload = load_channel_payload(path, channel)
        payloads[channel] = payload
        artifacts[channel] = {**file_identity(path), **esa_channel_payload_identity(payload)}

    pre = build_esa_preprocessed_input_receipt(
        intake_receipt=intake,
        group_receipt=groups,
        channel_artifacts=artifacts,
        preprocessor_source_sha256=file_identity(args.preprocessor_source)["sha256"],
        transform=spec["required_preprocessing_transform"],
        evaluation_range_ns=evaluation_range,
        plan=plan,
    )
    write(args.output_dir / "r22_esa_preprocessed_input_receipt.json", pre)
    predictions = generate_esa_transport_predictions(
        channel_payloads=payloads,
        group_receipt=groups,
        preprocessed_receipt=pre,
        plan=plan,
    )
    write(args.output_dir / "r22_esa_predictions_pretruth.json", predictions)

    if args.truth_json is None:
        print(json.dumps({"status": predictions["status"], "truth_joined": False}, ensure_ascii=False))
        return 0

    truth_payload = json.loads(args.truth_json.read_text(encoding="utf-8"))
    if not isinstance(truth_payload, dict) or not isinstance(truth_payload.get("events"), list):
        raise ValueError("R22_ESA_TRUTH_JSON_SCHEMA_INVALID")
    report = evaluate_esa_transport_replication(
        predictions,
        truth_events=truth_payload["events"],
        full_range_ns=truth_payload.get("full_range_ns", evaluation_range),
        group_receipt=groups,
        plan=plan,
    )
    write(args.output_dir / "r22_esa_transport_replication_report.json", report)
    print(json.dumps({
        "status": report["status"],
        "corrected_eventwise_F0.5": report["metrics"]["corrected_eventwise_F0.5"],
        "superiority_contrast_allowed": report["superiority_contrast_allowed"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
