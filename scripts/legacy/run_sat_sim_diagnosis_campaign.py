#!/usr/bin/env python3
"""Run the AstroGraph paired Sat-Sim diagnosis campaign without truth leakage.

Inference phase reads only case_index.csv, input_manifest.json and paired telemetry.
Evaluation phase is deliberately separate and opens evaluation_truth.json only after all
prediction files have been written.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from astrograph.diagnosis.sat_sim_paired import (  # noqa: E402
    PairedDiagnosisStatus,
    SatSimPairedDiagnosisEngine,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_case_index(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_inference(dataset_root: Path, output_root: Path) -> list[dict[str, Any]]:
    case_rows = _read_case_index(dataset_root / "case_index.csv")
    engine = SatSimPairedDiagnosisEngine()
    predictions: list[dict[str, Any]] = []
    prediction_root = output_root / "predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)

    for row in case_rows:
        case_id = row["case_id"]
        case_dir = dataset_root / row["input_dir"]
        manifest_path = case_dir / "input_manifest.json"
        nominal_path = case_dir / "nominal.csv"
        telemetry_path = case_dir / "telemetry.csv"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        if manifest.get("truth_available_to_diagnosis") is not False:
            raise RuntimeError(f"{case_id}: truth isolation flag is not false")
        for path, expected_key in (
            (nominal_path, "nominal_sha256"),
            (telemetry_path, "telemetry_sha256"),
        ):
            expected = str(manifest.get(expected_key, ""))
            actual = _sha256(path)
            if expected != actual:
                raise RuntimeError(f"{case_id}: checksum mismatch for {path.name}")

        result = engine.diagnose_paths(
            case_id=case_id,
            backend=manifest["backend"],
            nominal_path=nominal_path,
            effect_path=telemetry_path,
        )
        payload = result.model_dump(mode="json")
        payload["input_manifest_sha256"] = _sha256(manifest_path)
        payload["nominal_sha256"] = _sha256(nominal_path)
        payload["telemetry_sha256"] = _sha256(telemetry_path)
        payload["difficulty"] = row["difficulty"]
        payload["kind"] = row["kind"]
        payload["truth_file_opened_during_inference"] = False
        prediction_path = prediction_root / f"{case_id}.json"
        _write_json(prediction_path, payload)
        predictions.append(payload)

    _write_json(output_root / "inference_manifest.json", {
        "schema_version": "astrograph-sat-sim-inference-manifest.v1",
        "engine_id": engine.engine_id,
        "case_count": len(predictions),
        "prediction_count": len(list(prediction_root.glob("CASE-*.json"))),
        "truth_opened": False,
        "labels_used": any(p.get("labels_used") for p in predictions),
        "truth_used": any(p.get("truth_used") for p in predictions),
        "prohibited_column_use_count": sum(len(p.get("prohibited_columns_used", [])) for p in predictions),
    })
    return predictions


def _evaluate(truth_path: Path, output_root: Path, predictions: list[dict[str, Any]]) -> dict[str, Any]:
    # This is the first point at which hidden truth is opened.
    truth_payload = json.loads(truth_path.read_text(encoding="utf-8"))
    truth_by_case = {row["case_id"]: row for row in truth_payload["cases"]}
    pred_by_case = {row["case_id"]: row for row in predictions}
    if set(truth_by_case) != set(pred_by_case):
        raise RuntimeError("prediction/truth case set mismatch")

    detailed: list[dict[str, Any]] = []
    for case_id in sorted(truth_by_case):
        truth = truth_by_case[case_id]
        pred = pred_by_case[case_id]
        conclusion = pred.get("conclusion_effect_id")
        status = pred.get("status")
        exact = conclusion == truth["effect_id"]
        diagnosed_exact = status == PairedDiagnosisStatus.DIAGNOSED.value and exact
        top_candidates = [item["effect_id"] for item in pred.get("candidates", [])]
        top5 = truth["effect_id"] in top_candidates
        detailed.append({
            "case_id": case_id,
            "difficulty": pred["difficulty"],
            "backend": pred["backend"],
            "kind": truth["kind"],
            "truth_effect_id": truth["effect_id"],
            "predicted_effect_id": conclusion or "",
            "status": status,
            "top_score": pred.get("top_score", 0.0),
            "score_margin": pred.get("score_margin", 0.0),
            "exact_match": exact,
            "diagnosed_exact": diagnosed_exact,
            "truth_in_top5": top5,
        })

    result_path = output_root / "diagnosis_results.csv"
    with result_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detailed[0].keys()))
        writer.writeheader()
        writer.writerows(detailed)

    def group_metrics(key: str) -> dict[str, Any]:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in detailed:
            buckets[str(row[key])].append(row)
        return {
            name: {
                "case_count": len(rows),
                "exact_count": sum(bool(r["exact_match"]) for r in rows),
                "exact_accuracy": sum(bool(r["exact_match"]) for r in rows) / len(rows),
                "diagnosed_exact_count": sum(bool(r["diagnosed_exact"]) for r in rows),
                "diagnosed_exact_rate": sum(bool(r["diagnosed_exact"]) for r in rows) / len(rows),
                "top5_count": sum(bool(r["truth_in_top5"]) for r in rows),
                "top5_rate": sum(bool(r["truth_in_top5"]) for r in rows) / len(rows),
                "status_counts": dict(Counter(r["status"] for r in rows)),
            }
            for name, rows in sorted(buckets.items())
        }

    summary = {
        "schema_version": "astrograph-sat-sim-diagnosis-summary.v1",
        "case_count": len(detailed),
        "top_candidate_exact_count": sum(bool(r["exact_match"]) for r in detailed),
        "top_candidate_accuracy": sum(bool(r["exact_match"]) for r in detailed) / len(detailed),
        "confirmed_exact_count": sum(bool(r["diagnosed_exact"]) for r in detailed),
        "confirmed_exact_rate": sum(bool(r["diagnosed_exact"]) for r in detailed) / len(detailed),
        # Compatibility aliases with explicit semantics above.
        "exact_count": sum(bool(r["exact_match"]) for r in detailed),
        "exact_accuracy": sum(bool(r["exact_match"]) for r in detailed) / len(detailed),
        "diagnosed_exact_count": sum(bool(r["diagnosed_exact"]) for r in detailed),
        "diagnosed_exact_rate": sum(bool(r["diagnosed_exact"]) for r in detailed) / len(detailed),
        "top5_count": sum(bool(r["truth_in_top5"]) for r in detailed),
        "top5_rate": sum(bool(r["truth_in_top5"]) for r in detailed) / len(detailed),
        "status_counts": dict(Counter(r["status"] for r in detailed)),
        "by_difficulty": group_metrics("difficulty"),
        "by_backend": group_metrics("backend"),
        "by_kind": group_metrics("kind"),
        "false_predictions": [row for row in detailed if not row["exact_match"]],
        "claim_boundary": (
            "Evaluation against hidden labels after inference; laboratory campaign metrics, "
            "not independent generalization or calibrated diagnostic probability."
        ),
    }
    _write_json(output_root / "diagnosis_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument("--truth-path", type=Path, default=None, help="External hidden truth used only after inference files are written.")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists: {output_root}; pass --overwrite")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    predictions = _run_inference(dataset_root, output_root)
    if args.inference_only:
        payload = {
            "status": "PASS_INFERENCE_ONLY",
            "case_count": len(predictions),
            "truth_opened": False,
        }
        _write_json(output_root / "inference_only_summary.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    truth_path = args.truth_path.resolve() if args.truth_path is not None else dataset_root / "evaluation_truth.json"
    if not truth_path.exists():
        raise SystemExit("truth path is required for evaluation; use --inference-only to run without truth")
    summary = _evaluate(truth_path, output_root, predictions)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["case_count"] == 31 else 2


if __name__ == "__main__":
    raise SystemExit(main())
