#!/usr/bin/env python3
"""Retrospective 5-fold grouped comparison for PHM v6.0 trace-summary cascades.

This experiment keeps the historical post-activation trace-summary information
boundary.  It is deliberately separate from the unknown-onset causal-window PHM
benchmark and must not replace docs/CURRENT_DATA_DRIVEN_METRICS.md.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from astrograph.data_driven.phm_v60 import (  # noqa: E402
    PHMDataset,
    build_dataset,
    load_records,
    train_three_stage,
    train_three_stage_adaptive,
)

HIERARCHY = ROOT / "configs" / "datasets" / "phm_v6_0" / "diagnosis_hierarchy.json"


def _fold_records(records: list[Any], fold: int) -> list[Any]:
    output = []
    for record in records:
        remainder = record.sample_index % 5
        split = "test" if remainder == fold else (
            "validation" if remainder == (fold + 1) % 5 else "train"
        )
        output.append(dataclasses.replace(record, split=split))
    return output


def _metrics(result: dict[str, Any]) -> dict[str, float]:
    return {
        "detection": float(result["detection"]["macro_f1"]),
        "localization": float(result["localization"]["macro_f1"]),
        "classification": float(result["classification_predicted_component"]["macro_f1"]),
        "end_to_end": float(result["end_to_end_18_class"]["macro_f1"]),
    }


def _aggregate(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for metric in ("detection", "localization", "classification", "end_to_end"):
        values = [float(row[key][metric]) for row in rows]
        result[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "minimum": float(np.min(values)),
            "values": values,
        }
    return result


def _markdown(report: dict[str, Any]) -> str:
    baseline = report["aggregate"]["legacy_lr"]
    adaptive = report["aggregate"]["adaptive"]
    lines = [
        "# PHM v6.0 adaptive classical cascade — retrospective Group CV",
        "",
        "> Scope: post-activation trace-summary screening. This is not the unknown-onset causal-window formal PHM metric.",
        "",
        "| Metric | legacy LR mean | adaptive mean | delta | adaptive minimum |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in ("detection", "localization", "classification", "end_to_end"):
        lines.append(
            f"| {metric} | {baseline[metric]['mean']:.5f} | {adaptive[metric]['mean']:.5f} | "
            f"{adaptive[metric]['mean'] - baseline[metric]['mean']:+.5f} | {adaptive[metric]['minimum']:.5f} |"
        )
    lines.extend([
        "",
        "Selection is performed independently inside each fold: Train fits candidates, Validation selects candidate/threshold/route, Test is evaluated after freeze.",
        "The same trace-summary information boundary is used for both legacy and adaptive branches.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    hierarchy = json.loads(HIERARCHY.read_text(encoding="utf-8"))
    mapping = hierarchy["fault_to_component"]
    records = load_records(args.dataset_root)
    dataset = build_dataset(args.dataset_root, mapping, progress=True)

    rows: list[dict[str, Any]] = []
    for fold in range(5):
        folded = _fold_records(records, fold)
        fold_dataset = PHMDataset(
            features=dataset.features,
            records=folded,
            components=dataset.components,
        )
        legacy = train_three_stage(fold_dataset, mapping)
        adaptive = train_three_stage_adaptive(fold_dataset, mapping)
        legacy.pop("models", None)
        adaptive.pop("models", None)
        row = {
            "fold": fold,
            "test_groups": sorted({r.sample_index for r in folded if r.split == "test"}),
            "validation_groups": sorted({r.sample_index for r in folded if r.split == "validation"}),
            "legacy_lr": _metrics(legacy),
            "adaptive": _metrics(adaptive),
            "adaptive_selection": adaptive["selection"],
        }
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "adaptive_selection"}, ensure_ascii=False), flush=True)

    report = {
        "schema_version": "astrograph-phm-v60-adaptive-group-cv.v1",
        "scope": "POST_ACTIVATION_TRACE_SUMMARY_RETROSPECTIVE_GROUP_CV",
        "fold_protocol": "sample_index % 5; test=fold; validation=(fold+1)%5; remaining groups=train",
        "folds": rows,
        "aggregate": {
            "legacy_lr": _aggregate(rows, "legacy_lr"),
            "adaptive": _aggregate(rows, "adaptive"),
        },
        "claim_boundary": (
            "Retrospective synthetic trace-summary evaluation with known activation boundary. "
            "Not an unknown-onset online, production, flight, or safety claim."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    print(args.output_dir / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
