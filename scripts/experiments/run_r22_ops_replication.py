from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from astrograph.evaluation.ops_sat_adapter import RAW_FIELDS, TRUTH_FIELDS, validate_online_fields
from astrograph.evaluation.replication_r22 import evaluate_ops_replication, generate_ops_replication_predictions


def write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def segments_csv(source: Path) -> Path:
    source = source.resolve()
    return source / "segments.csv" if source.is_dir() else source


def load_online_only(path: Path) -> dict[str, list[dict]]:
    online: dict[str, list[dict]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or []) != RAW_FIELDS | TRUTH_FIELDS:
            raise ValueError("R22_OPS_RAW_SCHEMA_DRIFT")
        for row in reader:
            sid = str(row["segment"])
            point = {key: row[key] for key in RAW_FIELDS}
            point["value"] = float(point["value"])
            validate_online_fields(point, track="raw")
            online.setdefault(sid, []).append(point)
    return online


def load_truth_only(path: Path) -> dict[str, int]:
    truth: dict[str, int] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or []) != RAW_FIELDS | TRUTH_FIELDS:
            raise ValueError("R22_OPS_RAW_SCHEMA_DRIFT")
        for row in reader:
            sid = str(row["segment"]); label = int(row["anomaly"])
            if label not in (0, 1):
                raise ValueError("R22_OPS_LABEL_SEMANTICS_DRIFT")
            if sid in truth and truth[sid] != label:
                raise ValueError("R22_OPS_SEGMENT_TRUTH_DRIFT")
            truth[sid] = label
    return truth


def main() -> int:
    parser = argparse.ArgumentParser(description="Run frozen R20 baseline and R21 OPS remediation on one admitted new R22 holdout.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--intake-receipt", type=Path, required=True)
    parser.add_argument("--group-admission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    intake = json.loads(args.intake_receipt.read_text(encoding="utf-8"))
    groups = json.loads(args.group_admission.read_text(encoding="utf-8"))

    # Prediction pass deliberately does not read anomaly/train/label values.
    online = load_online_only(segments_csv(args.source))
    predictions = generate_ops_replication_predictions(args.source, raw_segments=online, intake_receipt=intake, group_receipt=groups)
    write(args.output_dir / "r22_ops_predictions_pretruth.json", predictions)

    # Truth is joined only after the immutable prediction receipt is persisted.
    truth = load_truth_only(segments_csv(args.source))
    report = evaluate_ops_replication(predictions, truth, groups)
    write(args.output_dir / "r22_ops_replication_report.json", report)
    print(json.dumps({"status": report["status"], "primary_delta": report["primary_delta"], "coverage_delta": report["coverage_delta"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
