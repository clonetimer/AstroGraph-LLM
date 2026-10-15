#!/usr/bin/env python3
"""Run single-trace catalog diagnosis, then evaluate against externally loaded truth."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import random
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astrograph.diagnosis.sat_sim_catalog import (  # noqa: E402
    NORMAL_CLASS_ID,
    SatSimCatalogDiagnosisEngine,
)


def _truth(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return (
        {str(item["case_id"]): str(item["effect_id"]) for item in payload["cases"]},
        {str(item["case_id"]): str(item["kind"]) for item in payload["cases"]},
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shuffle-seed", type=int, default=20260728)
    args = parser.parse_args()

    engine = SatSimCatalogDiagnosisEngine.from_path(args.model)
    index = pd.read_csv(args.dataset / "case_index.csv")
    rows = list(index.itertuples(index=False))
    random.Random(args.shuffle_seed).shuffle(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    prediction_dir = args.output / "predictions"
    prediction_dir.mkdir(exist_ok=True)

    # Inference phase deliberately does not load the truth manifest.
    inference_records: list[dict[str, object]] = []
    for row in rows:
        case_dir = args.dataset / str(row.input_dir)
        for trace_kind, filename in (("nominal", "nominal.csv"), ("effect", "telemetry.csv")):
            opaque_id = f"TRACE-{len(inference_records) + 1:03d}"
            result = engine.diagnose_path(
                case_id=opaque_id,
                backend=str(row.backend),
                telemetry_path=case_dir / filename,
            )
            payload = result.model_dump(mode="json")
            (prediction_dir / f"{opaque_id}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            inference_records.append({
                "opaque_id": opaque_id,
                "source_case_id": str(row.case_id),
                "difficulty": str(row.difficulty),
                "backend": str(row.backend),
                "trace_kind": trace_kind,
                "prediction": payload,
            })
    (args.output / "inference_manifest.json").write_text(
        json.dumps({
            "schema_version": "astrograph-sat-sim-catalog-inference.v1",
            "truth_loaded_during_inference": False,
            "paired_nominal_supplied_to_effect_inference": False,
            "record_count": len(inference_records),
            "records": inference_records,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Evaluation phase begins only after every prediction has been persisted.
    truth, kinds = _truth(args.truth)
    result_rows: list[dict[str, object]] = []
    group_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in inference_records:
        case_id = str(record["source_case_id"])
        expected = NORMAL_CLASS_ID if record["trace_kind"] == "nominal" else truth[case_id]
        prediction = record["prediction"]
        actual = prediction["conclusion_effect_id"] or NORMAL_CLASS_ID
        correct = actual == expected
        group_counts[str(record["difficulty"])]["total"] += 1
        group_counts[str(record["difficulty"])]["correct"] += int(correct)
        result_rows.append({
            "opaque_id": record["opaque_id"],
            "case_id": case_id,
            "difficulty": record["difficulty"],
            "backend": record["backend"],
            "kind": "normal" if record["trace_kind"] == "nominal" else kinds[case_id],
            "trace_kind": record["trace_kind"],
            "expected": expected,
            "predicted": actual,
            "status": prediction["status"],
            "correct": correct,
            "model_vote_share": prediction["model_vote_share"],
            "resolution_basis": prediction["resolution_basis"],
        })
    frame = pd.DataFrame(result_rows)
    frame.to_csv(args.output / "catalog_results.csv", index=False)
    normal = frame[frame.trace_kind == "nominal"]
    effects = frame[frame.trace_kind == "effect"]
    summary = {
        "schema_version": "astrograph-sat-sim-catalog-evaluation.v1",
        "trace_count": len(frame),
        "effect_case_count": len(effects),
        "normal_case_count": len(normal),
        "overall_correct": int(frame.correct.sum()),
        "effect_correct": int(effects.correct.sum()),
        "normal_correct": int(normal.correct.sum()),
        "effect_accuracy": float(effects.correct.mean()),
        "normal_specificity": float(normal.correct.mean()),
        "difficulty": {
            key: {"correct": value["correct"], "total": value["total"]}
            for key, value in sorted(group_counts.items())
        },
        "truth_loaded_during_inference": False,
        "paired_nominal_supplied_to_effect_inference": False,
        "score_semantics": "ENSEMBLE_VOTE_SHARE_NOT_CALIBRATED_PROBABILITY",
    }
    (args.output / "catalog_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["overall_correct"] == summary["trace_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
