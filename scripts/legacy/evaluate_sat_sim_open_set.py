#!/usr/bin/env python3
"""Evaluate known-catalog retention and independent real-Basilisk unknown rejection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astrograph.diagnosis.sat_sim_catalog import NORMAL_CLASS_ID  # noqa: E402
from astrograph.diagnosis.sat_sim_open_set import SatSimOpenSetDiagnosisEngine, sanitize_telemetry  # noqa: E402


def _truth(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["case_id"]): str(item["effect_id"]) for item in payload["cases"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--known-dataset", type=Path, required=True)
    parser.add_argument("--known-truth", type=Path, required=True)
    parser.add_argument("--unknown-root", type=Path, required=True)
    parser.add_argument("--unknown-names", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    engine = SatSimOpenSetDiagnosisEngine.from_path(args.model)
    args.output.mkdir(parents=True, exist_ok=True)
    truth = _truth(args.known_truth)
    index = pd.read_csv(args.known_dataset / "case_index.csv")
    rows: list[dict[str, object]] = []
    for row in index.itertuples(index=False):
        case_dir = args.known_dataset / str(row.input_dir)
        for trace_kind, filename in (("nominal", "nominal.csv"), ("effect", "telemetry.csv")):
            result = engine.diagnose_path(
                case_id=f"KNOWN-{row.case_id}-{trace_kind}",
                backend=str(row.backend),
                telemetry_path=case_dir / filename,
            )
            expected = NORMAL_CLASS_ID if trace_kind == "nominal" else truth[str(row.case_id)]
            actual = result.conclusion_effect_id or (NORMAL_CLASS_ID if result.status == "NORMAL" else result.status)
            rows.append({
                "group": "known",
                "case_id": str(row.case_id),
                "trace_kind": trace_kind,
                "expected": expected,
                "status": result.status,
                "predicted": actual,
                "correct": actual == expected,
                "membership_confidence": result.catalog_membership_confidence,
                "conformity_p_value": result.catalog_conformity_p_value,
                "model_vote_share": result.model_vote_share,
                "profile_distance": result.profile_distance,
            })
    unknown_names = [item.strip() for item in args.unknown_names.split(",") if item.strip()]
    sanitized = args.output / "unknown_sanitized"
    sanitized.mkdir(exist_ok=True)
    for name in unknown_names:
        raw = pd.read_csv(args.unknown_root / name / "telemetry.csv", low_memory=False)
        clean = raw[[column for column in raw.columns if not column.lower().startswith(("label.", "truth.")) and "effect.active" not in column.lower()]]
        clean = sanitize_telemetry(clean)
        clean.to_csv(sanitized / f"{name}.csv", index=False)
        result = engine.diagnose_frame(case_id=f"UNKNOWN-{name}", backend="unified", telemetry=clean)
        rows.append({
            "group": "unknown",
            "case_id": name,
            "trace_kind": "effect",
            "expected": "UNKNOWN_FAULT",
            "status": result.status,
            "predicted": result.conclusion_effect_id or result.status,
            "correct": result.status == "UNKNOWN_FAULT",
            "membership_confidence": result.catalog_membership_confidence,
            "conformity_p_value": result.catalog_conformity_p_value,
            "model_vote_share": result.model_vote_share,
            "profile_distance": result.profile_distance,
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "open_set_results.csv", index=False)
    known = frame[frame.group == "known"]
    unknown = frame[frame.group == "unknown"]
    summary = {
        "schema_version": "astrograph-open-set-evaluation.v1",
        "known_trace_count": int(len(known)),
        "known_correct": int(known.correct.sum()),
        "known_accuracy": float(known.correct.mean()),
        "unknown_trace_count": int(len(unknown)),
        "unknown_rejected": int(unknown.correct.sum()),
        "unknown_recall": float(unknown.correct.mean()),
        "false_known_accept_count": int((~unknown.correct).sum()),
        "truth_loaded_during_inference": False,
        "score_semantics": "CATALOG_MEMBERSHIP_CONFIDENCE_CALIBRATED_ON_BASILISK_LAB_DATA",
    }
    (args.output / "open_set_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["known_correct"] == summary["known_trace_count"] and summary["unknown_rejected"] == summary["unknown_trace_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
