#!/usr/bin/env python3
"""Merge bounded ADCS A0–A7 slices into one group-respecting report."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from astrograph.assurance import ArtifactRoot, canonical_sha256
from astrograph.experiments import (
    AblationVariant,
    GroupedPrediction,
    paired_group_accuracy_delta,
    summarize_grouped_classification,
)

BOOTSTRAP_SEED = 20260829


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _truth_counts(records: list[GroupedPrediction]) -> dict[str, dict[str, int]]:
    by_truth: dict[str, list[GroupedPrediction]] = defaultdict(list)
    for row in records:
        by_truth[row.truth].append(row)
    return {
        truth: {
            "count": len(rows),
            "correct": sum(row.prediction == truth for row in rows),
            "abstained": sum(row.prediction is None for row in rows),
        }
        for truth, rows in sorted(by_truth.items())
    }


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    args = parser.parse_args()

    reports = [json.loads((path / "report.json").read_text(encoding="utf-8")) for path in args.chunk]
    plan_shas = {report["plan_sha256"] for report in reports}
    if len(plan_shas) != 1:
        raise ValueError(f"chunk plan SHA mismatch: {sorted(plan_shas)}")
    plan_sha = next(iter(plan_shas))
    plan = reports[0]["plan"]
    fault_universe = tuple(plan["fault_universe"])

    slices = sorted(
        (
            int(report["evaluation"]["slice_start_index"]),
            int(report["evaluation"]["slice_stop_index_exclusive"]),
        )
        for report in reports
    )
    expected_start = 0
    for start, stop in slices:
        if start != expected_start or stop <= start:
            raise ValueError(f"chunk slices are not contiguous/nonoverlapping at {start}:{stop}")
        expected_start = stop
    expected_total = int(reports[0]["evaluation"]["full_fault_trajectory_count"])
    if expected_start != expected_total:
        raise ValueError(f"chunk slices cover {expected_start} rows, expected {expected_total}")

    rows: list[dict[str, str]] = []
    for path in args.chunk:
        rows.extend(_read_csv(path / "predictions.csv"))
    observation_ids = [row["observation_id"] for row in rows]
    if len(observation_ids) != len(set(observation_ids)):
        raise ValueError("merged chunks contain duplicate evaluation observation IDs")
    if len(rows) != expected_total:
        raise ValueError(f"merged prediction count is {len(rows)}, expected {expected_total}")

    by_telemetry: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_telemetry[row["telemetry_sha256"]].append(row)
    cross_truth_duplicates = {
        sha: sorted({row["truth"] for row in members})
        for sha, members in by_telemetry.items()
        if len({row["truth"] for row in members}) > 1
    }
    if cross_truth_duplicates:
        raise ValueError(f"exact telemetry duplicates cross truth labels: {cross_truth_duplicates}")
    duplicate_groups = {sha: members for sha, members in by_telemetry.items() if len(members) > 1}
    unique_rows = [members[0] for _, members in sorted(by_telemetry.items())]

    records_by_variant: dict[str, list[GroupedPrediction]] = {}
    unique_records_by_variant: dict[str, list[GroupedPrediction]] = {}
    metrics: dict[str, Any] = {}
    unique_metrics: dict[str, Any] = {}
    for variant in AblationVariant:
        key = variant.value
        records_by_variant[key] = [
            GroupedPrediction(
                observation_id=row["observation_id"],
                group_id=row["trajectory_group_id"],
                truth=row["truth"],
                prediction=row[key] or None,
            )
            for row in rows
        ]
        unique_records_by_variant[key] = [
            GroupedPrediction(
                observation_id=row["observation_id"],
                group_id=row["trajectory_group_id"],
                truth=row["truth"],
                prediction=row[key] or None,
            )
            for row in unique_rows
        ]
        summary = summarize_grouped_classification(
            records_by_variant[key],
            label_universe=fault_universe,
            bootstrap_iterations=args.bootstrap_iterations,
            bootstrap_seed=BOOTSTRAP_SEED,
        ).model_dump(mode="json")
        unique_summary = summarize_grouped_classification(
            unique_records_by_variant[key],
            label_universe=fault_universe,
            bootstrap_iterations=args.bootstrap_iterations,
            bootstrap_seed=BOOTSTRAP_SEED,
        ).model_dump(mode="json")
        summary.update(
            {
                "top3_recall": sum(_bool(row[f"{key}__top3_hit"]) for row in rows) / len(rows),
                "episodes_with_dependency_adjustment": sum(
                    _bool(row[f"{key}__dependency_adjusted"]) for row in rows
                ),
                "class_counts": _truth_counts(records_by_variant[key]),
            }
        )
        unique_summary.update(
            {
                "top3_recall": (
                    sum(_bool(row[f"{key}__top3_hit"]) for row in unique_rows) / len(unique_rows)
                ),
                "episodes_with_dependency_adjustment": sum(
                    _bool(row[f"{key}__dependency_adjusted"]) for row in unique_rows
                ),
                "class_counts": _truth_counts(unique_records_by_variant[key]),
            }
        )
        metrics[key] = summary
        unique_metrics[key] = unique_summary

    paired: dict[str, Any] = {}
    unique_paired: dict[str, Any] = {}
    for reference in (AblationVariant.A0_DATA_ONLY, AblationVariant.A6_ALL_UNADJUSTED):
        key = f"A7_minus_{reference.value}"
        paired[key] = paired_group_accuracy_delta(
            records_by_variant[reference.value],
            records_by_variant[AblationVariant.A7_ALL_LINEAGE_ADJUSTED.value],
            reference_id=reference.value,
            candidate_id=AblationVariant.A7_ALL_LINEAGE_ADJUSTED.value,
            bootstrap_iterations=args.bootstrap_iterations,
            bootstrap_seed=BOOTSTRAP_SEED,
        ).model_dump(mode="json")
        unique_paired[key] = paired_group_accuracy_delta(
            unique_records_by_variant[reference.value],
            unique_records_by_variant[AblationVariant.A7_ALL_LINEAGE_ADJUSTED.value],
            reference_id=reference.value,
            candidate_id=AblationVariant.A7_ALL_LINEAGE_ADJUSTED.value,
            bootstrap_iterations=args.bootstrap_iterations,
            bootstrap_seed=BOOTSTRAP_SEED,
        ).model_dump(mode="json")

    report = {
        "schema_version": "astrograph-adcs-v522-merged-frozen-ablation-report.v1",
        "status": "COMPLETED",
        "plan": plan,
        "plan_sha256": plan_sha,
        "chunk_count": len(args.chunk),
        "chunk_slices": slices,
        "evaluation": {
            "fault_trajectory_count": len(rows),
            "independent_group_count": len({row["trajectory_group_id"] for row in rows}),
            "fault_class_count": len(fault_universe),
            "unique_telemetry_count": len(unique_rows),
            "exact_duplicate_telemetry_row_count": len(rows) - len(unique_rows),
            "exact_duplicate_telemetry_hash_count": len(duplicate_groups),
            "cross_truth_duplicate_hash_count": len(cross_truth_duplicates),
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "frozen_176_row_metrics": metrics,
        "unique_telemetry_164_row_sensitivity": unique_metrics,
        "paired_group_effects": paired,
        "unique_telemetry_paired_group_effects": unique_paired,
        "duplicate_telemetry_audit": {
            "duplicate_hash_count": len(duplicate_groups),
            "duplicate_row_count": sum(len(items) for items in duplicate_groups.values()),
            "redundant_copy_count": len(rows) - len(unique_rows),
            "cross_truth_duplicates": cross_truth_duplicates,
        },
        "limitations": reports[0]["limitations"],
        "claim_boundary": reports[0]["claim_boundary"],
    }
    report["report_sha256"] = canonical_sha256(report)

    artifacts = ArtifactRoot(args.output_dir, source_root=ROOT)
    artifacts.write_json("report.json", report)
    fields = list(rows[0]) if rows else []
    artifacts.write_generated("predictions.csv", lambda path: _write_csv(path, rows, fields))

    lines = [
        "# ADCS v5.2.2 frozen A0–A7 trajectory-level ablation",
        "",
        f"- Plan SHA-256: `{plan_sha}`",
        f"- Fault trajectories: {len(rows)}",
        f"- Independent trajectory groups: {report['evaluation']['independent_group_count']}",
        f"- Unique telemetry SHA-256 values: {len(unique_rows)}",
        f"- Redundant exact telemetry copies: {len(rows) - len(unique_rows)}",
        "",
        "| Variant | Accuracy | Macro F1 | Group-macro accuracy | 95% cluster bootstrap accuracy CI | Abstain | Top-3 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in AblationVariant:
        item = metrics[variant.value]
        ci = item["cluster_bootstrap_accuracy_ci95"]
        lines.append(
            f"| {variant.value} | {item['accuracy']:.4f} | {item['macro_f1']:.4f} | "
            f"{item['group_macro_accuracy']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | "
            f"{item['abstention_rate']:.4f} | {item['top3_recall']:.4f} |"
        )
    lines += [
        "",
        "## Unique-telemetry sensitivity",
        "",
        "| Variant | Accuracy | Macro F1 | Group-macro accuracy |",
        "|---|---:|---:|---:|",
    ]
    for variant in AblationVariant:
        item = unique_metrics[variant.value]
        lines.append(
            f"| {variant.value} | {item['accuracy']:.4f} | {item['macro_f1']:.4f} | {item['group_macro_accuracy']:.4f} |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "This is a held-out synthetic E0 research ablation. Only six independent groups are available; "
        "probabilities are uncalibrated; configured rule/KG evidence covers four of fourteen ADCS fault IDs; "
        "and no G0–G4 or flight qualification status is changed by this report.",
        "",
    ]
    artifacts.write_text("summary.md", "\n".join(lines))

    print(json.dumps({
        "status": "COMPLETED",
        "rows": len(rows),
        "groups": report["evaluation"]["independent_group_count"],
        "unique_telemetry": len(unique_rows),
        "accuracy": {key: round(value["accuracy"], 6) for key, value in metrics.items()},
        "report_sha256": report["report_sha256"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
