#!/usr/bin/env python3
"""Run the PHM v6.0 five-class mapped-subset A0–A7 replay.

The data branch is a replay of the frozen LOSO top-1 predictions from
``audit/phm_predictions.csv``.  It is not a probability replay and has no
feature-level lineage.  Rule/KG evidence is the currently registered label-free
heater package, which covers one of the five mapped classes.  Consequently this
experiment is a coverage-boundary diagnostic, not a full 17-class PHM result.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from astrograph.assurance import ArtifactRoot, canonical_sha256
from astrograph.evidence.executable import stable_id
from astrograph.evidence.configured_runtime import ConfiguredRuntimeEvidenceBuilder
from astrograph.experiments import (
    AblationVariant,
    FrozenAblationPlan,
    FrozenAblationRunner,
    GroupedPrediction,
    paired_group_accuracy_delta,
    summarize_grouped_classification,
)
from astrograph.fusion import BiRRFConfig, FaultRanking, RankedFault, RankingSource, ScoreSemantics
from astrograph.fusion.ranking import build_backward_evidence_ranking, build_forward_graph_ranking
from astrograph.simulation.models import SimulationRunCompleted

EVALUATION_ID = "paired_residual/global_17_fault_identification/full"
DEFAULT_EVIDENCE = ROOT / "configs" / "datasets" / "phm_v6_0" / "runtime_evidence.json"
DEFAULT_FAULT_MAP = ROOT / "configs" / "datasets" / "phm_v6_0" / "fault_class_map.json"
BOOTSTRAP_SEED = 20260829


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_index(dataset_root: Path) -> dict[str, dict[str, str]]:
    with (dataset_root / "sample_index.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["sample_id"]: row for row in rows}


def _data_ranking(
    *, episode_id: str, predicted_dataset_class: str, class_map: dict[str, str], audit_sha256: str
) -> FaultRanking:
    mapped = class_map.get(predicted_dataset_class)
    candidates = [] if mapped is None else [
        RankedFault(
            fault_id=mapped,
            rank=1,
            score=1.0,
            source=RankingSource.DATA_DRIVEN,
            lineage_complete=False,
            contributes_to_fusion=True,
            metadata={"replay_semantics": "frozen_external_top1_only"},
        )
    ]
    return FaultRanking(
        ranking_id=stable_id(
            "RANK",
            {
                "episode_id": episode_id,
                "source": "phm-v6-paired-residual-global-top1-replay",
                "mapped_prediction": mapped,
                "audit_sha256": audit_sha256,
            },
        ),
        episode_id=episode_id,
        source=RankingSource.DATA_DRIVEN,
        score_semantics=ScoreSemantics.RANKING_SCORE,
        candidates=candidates,
        producer_id="phm-v6-paired-residual-global-17-loso-audit-replay",
        producer_version="v6.0-audit",
        metadata={
            "evaluation_id": EVALUATION_ID,
            "top1_only": True,
            "calibrated": False,
            "feature_lineage_available": False,
        },
        claim_boundary=(
            "Replay of frozen LOSO top-1 audit output; not a probability ranking and not a live model inference."
        ),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    artifacts = ArtifactRoot(args.output_dir, source_root=ROOT)
    artifacts.ensure()
    class_map = json.loads(DEFAULT_FAULT_MAP.read_text(encoding="utf-8"))["mappings"]
    fault_universe = tuple(sorted(set(class_map.values())))
    builder = ConfiguredRuntimeEvidenceBuilder(DEFAULT_EVIDENCE)
    evidence_targets = sorted({item.target_fault_id for item in builder.package.evidence_definitions})

    predictions_path = dataset_root / "audit" / "phm_predictions.csv"
    audit_sha = sha256_file(predictions_path)
    predictions = pd.read_csv(predictions_path)
    global_rows = predictions[predictions["evaluation_id"] == EVALUATION_ID].copy()
    if len(global_rows) != 510 or global_rows["truth"].nunique() != 17:
        raise ValueError("frozen PHM global-17 audit shape differs from expected 510 rows / 17 classes")
    full_global_accuracy = float((global_rows["truth"] == global_rows["predicted"]).mean())

    index = _load_index(dataset_root)
    mapped = global_rows[global_rows["truth"].isin(class_map)].copy()
    if len(mapped) != 150:
        raise ValueError(f"expected 150 mapped-truth PHM rows, got {len(mapped)}")
    missing_samples = sorted(set(mapped["sample_id"]) - set(index))
    if missing_samples:
        raise ValueError(f"audit references missing sample IDs: {missing_samples[:3]}")

    nominal_by_pair = {
        row["trajectory_pair_id"]: row
        for row in index.values()
        if row["diagnostic_class"] == "nominal"
    }
    used_files: set[str] = set()
    for sample_id in mapped["sample_id"]:
        row = index[str(sample_id)]
        used_files.add(row["telemetry_file"])
        nominal = nominal_by_pair[row["trajectory_pair_id"]]
        used_files.add(nominal["telemetry_file"])
    telemetry_hashes = {
        rel: sha256_file(dataset_root / rel) for rel in sorted(used_files)
    }
    dataset_identity = {
        "sample_index_sha256": sha256_file(dataset_root / "sample_index.csv"),
        "audit_predictions_sha256": audit_sha,
        "telemetry_sha256_by_relative_path": telemetry_hashes,
        "evaluation_id": EVALUATION_ID,
        "mapped_truth_sample_ids": sorted(map(str, mapped["sample_id"])),
    }
    split_identity = canonical_sha256(
        {
            "policy": "frozen leave-one-Monte-Carlo-group-out audit replay",
            "group_key": "mc_group",
            "rows": sorted(
                (
                    str(row.sample_id), int(row.mc_group), str(row.truth), str(row.predicted)
                )
                for row in mapped.itertuples(index=False)
            ),
        }
    )
    config = BiRRFConfig()
    plan = FrozenAblationPlan(
        plan_id="phm-v6.0-mapped-five-class-top1-replay-a0-a7-b005",
        dataset_id="whole-spacecraft-phm-v6.0:mapped-five-class-loso-replay",
        dataset_sha256=canonical_sha256(dataset_identity),
        split_id="leave-one-mc-group-out-audit-replay",
        split_sha256=split_identity,
        calibration_id="TOP1_ONLY_UNCALIBRATED_EXTERNAL_REPLAY",
        calibration_sha256=canonical_sha256(
            {"audit_sha256": audit_sha, "semantics": "top1-only", "calibrated": False}
        ),
        fault_universe=fault_universe,
        policy_id="birrf-default-engineering-policy",
        policy_sha256=canonical_sha256(config.model_dump(mode="json")),
        code_version="0.7.1.post24+B005",
        data_branch_id=f"{EVALUATION_ID}:{audit_sha}",
        forward_branch_id=(
            f"configured-runtime-forward:{builder.package.package_id}@{builder.package.version}:"
            f"{builder.package_sha256}"
        ),
        backward_branch_id=(
            f"configured-runtime-backward:{builder.package.package_id}@{builder.package.version}:"
            f"{builder.package_sha256}"
        ),
        base_config=config,
        metadata={
            "full_dataset_class_count": 17,
            "strict_ontology_mapping_count": len(class_map),
            "runtime_evidence_target_count": len(evidence_targets),
            "data_branch_semantics": "frozen LOSO top1 replay; no scores/probabilities",
            "independent_group_key": "mc_group",
        },
    )
    runner = FrozenAblationRunner(plan)

    nominal_frames = {
        pair: pd.read_csv(dataset_root / row["telemetry_file"], low_memory=False)
        for pair, row in nominal_by_pair.items()
        if pair in {index[str(sample_id)]["trajectory_pair_id"] for sample_id in mapped["sample_id"]}
    }
    records: dict[str, list[GroupedPrediction]] = {variant.value: [] for variant in AblationVariant}
    output_rows: list[dict[str, Any]] = []
    dependency_counts = defaultdict(int)

    for number, audit_row in enumerate(mapped.itertuples(index=False), start=1):
        row = index[str(audit_row.sample_id)]
        if row["diagnostic_class"] != str(audit_row.truth):
            raise ValueError("PHM audit truth differs from sample-index diagnostic class")
        pair = row["trajectory_pair_id"]
        telemetry_path = dataset_root / row["telemetry_file"]
        telemetry_sha = telemetry_hashes[row["telemetry_file"]]
        nominal = nominal_by_pair[pair]
        event = SimulationRunCompleted(
            event_id=f"evt-{telemetry_sha[:20]}",
            run_id=f"run-{telemetry_sha[:20]}",
            run_dir=str(telemetry_path.parent),
            nominal_run_dir=str((dataset_root / nominal["telemetry_file"]).parent),
            runtime_id="phm-v6.0-mapped-replay",
        )
        source = pd.read_csv(telemetry_path, low_memory=False)
        bundle, gate = builder.build_from_frames(event, source=source, nominal=nominal_frames[pair])
        if not gate.passed:
            raise ValueError("PHM configured runtime gate failed on a mapped replay sample")
        data = _data_ranking(
            episode_id=bundle.episode.episode_id,
            predicted_dataset_class=str(audit_row.predicted),
            class_map=class_map,
            audit_sha256=audit_sha,
        )
        forward = build_forward_graph_ranking(bundle)
        backward = build_backward_evidence_ranking(bundle, candidate_fault_ids=fault_universe)
        run = runner.run(
            data_factory=lambda ranking=data: ranking,
            forward_factory=lambda ranking=forward: ranking,
            backward_factory=lambda ranking=backward: ranking,
        )
        truth = class_map[str(audit_row.truth)]
        observation_id = "EVAL-" + canonical_sha256({"sample_id": str(audit_row.sample_id)})[:24].upper()
        row_output: dict[str, Any] = {
            "observation_id": observation_id,
            "mc_group": str(audit_row.mc_group),
            "truth": truth,
            "external_predicted_dataset_class": str(audit_row.predicted),
            "mapped_external_prediction": class_map.get(str(audit_row.predicted), ""),
        }
        for variant in AblationVariant:
            result = run.results[variant.value]
            records[variant.value].append(
                GroupedPrediction(
                    observation_id=observation_id,
                    group_id=str(audit_row.mc_group),
                    truth=truth,
                    prediction=result.top_fault_id,
                )
            )
            row_output[variant.value] = result.top_fault_id or ""
            adjusted = bool(result.metadata.get("evidence_lineage_dependency_adjusted_fault_ids"))
            row_output[f"{variant.value}__dependency_adjusted"] = adjusted
            dependency_counts[variant.value] += int(adjusted)
        output_rows.append(row_output)
        if number % 30 == 0:
            print(f"[phm-mapped-a0-a7] {number}/{len(mapped)}", flush=True)

    metrics = {
        variant.value: {
            **summarize_grouped_classification(
                records[variant.value],
                label_universe=fault_universe,
                bootstrap_iterations=args.bootstrap_iterations,
                bootstrap_seed=BOOTSTRAP_SEED,
            ).model_dump(mode="json"),
            "episodes_with_dependency_adjustment": dependency_counts[variant.value],
        }
        for variant in AblationVariant
    }
    paired = {
        f"A7_minus_{reference.value}": paired_group_accuracy_delta(
            records[reference.value],
            records[AblationVariant.A7_ALL_LINEAGE_ADJUSTED.value],
            reference_id=reference.value,
            candidate_id=AblationVariant.A7_ALL_LINEAGE_ADJUSTED.value,
            bootstrap_iterations=args.bootstrap_iterations,
            bootstrap_seed=BOOTSTRAP_SEED,
        ).model_dump(mode="json")
        for reference in (AblationVariant.A0_DATA_ONLY, AblationVariant.A6_ALL_UNADJUSTED)
    }

    report = {
        "schema_version": "astrograph-phm-v60-mapped-ablation-report.v1",
        "status": "COMPLETED_MAPPED_SUBSET_ONLY",
        "plan": plan.model_dump(mode="json"),
        "plan_sha256": plan.plan_sha256,
        "full_global_17_class_audit": {
            "evaluation_id": EVALUATION_ID,
            "row_count": len(global_rows),
            "class_count": int(global_rows["truth"].nunique()),
            "mc_group_count": int(global_rows["mc_group"].nunique()),
            "top1_accuracy": full_global_accuracy,
        },
        "coverage": {
            "dataset_fault_class_count": 17,
            "strict_ontology_mapping_count": len(class_map),
            "strict_ontology_mapping_fraction": len(class_map) / 17.0,
            "runtime_evidence_target_count": len(evidence_targets),
            "runtime_evidence_fraction_of_full": len(evidence_targets) / 17.0,
            "runtime_evidence_fraction_of_mapped": len(evidence_targets) / len(class_map),
            "runtime_evidence_targets": evidence_targets,
            "full_17_class_a0_a7_status": "BLOCKED_INCOMPLETE_ONTOLOGY_AND_RUNTIME_EVIDENCE_COVERAGE",
        },
        "mapped_subset": {
            "row_count": len(mapped),
            "independent_mc_group_count": int(mapped["mc_group"].nunique()),
            "class_count": len(fault_universe),
            "variant_metrics": metrics,
            "paired_group_effects": paired,
        },
        "limitations": [
            "Only five of seventeen PHM dataset fault classes have strict AstroGraph ontology mappings.",
            "Configured runtime rule/KG evidence covers only THERMAL_HEATER_STUCK_ON (one of five mapped classes).",
            "The data branch replays frozen top-1 LOSO predictions and has neither full rankings nor probabilities.",
            "The replay data branch lacks feature-level evidence lineage, so A7 cannot estimate data-vs-KG dependence.",
            "The mapped-subset result must not be reported as a full 17-class PHM A0–A7 result.",
        ],
        "claim_boundary": (
            "Synthetic mapped-subset coverage experiment only. Full 17-class AstroGraph fusion remains blocked; "
            "no qualification status changes."
        ),
    }
    report["report_sha256"] = canonical_sha256(report)
    artifacts.write_json("report.json", report)
    fields = list(output_rows[0])
    artifacts.write_generated("predictions.csv", lambda path: _write_csv(path, output_rows, fields))
    lines = [
        "# PHM v6.0 mapped-subset A0–A7 replay",
        "",
        f"- Full global-17 audit accuracy: {full_global_accuracy:.4f} (510 rows, 10 MC groups)",
        f"- Strict ontology coverage: {len(class_map)}/17",
        f"- Runtime evidence coverage: {len(evidence_targets)}/17 full classes; {len(evidence_targets)}/{len(class_map)} mapped classes",
        "- Full 17-class A0–A7: **BLOCKED**",
        "",
        "| Variant | Mapped-subset accuracy | Macro F1 | Group-macro accuracy | Abstain |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in AblationVariant:
        item = metrics[variant.value]
        lines.append(
            f"| {variant.value} | {item['accuracy']:.4f} | {item['macro_f1']:.4f} | "
            f"{item['group_macro_accuracy']:.4f} | {item['abstention_rate']:.4f} |"
        )
    lines += [
        "",
        "This five-class replay is a boundary audit, not a substitute for completing the missing twelve ontology mappings and sixteen missing runtime-evidence classes.",
        "",
    ]
    artifacts.write_text("summary.md", "\n".join(lines))
    print(json.dumps({
        "status": report["status"],
        "full_17_accuracy": round(full_global_accuracy, 6),
        "mapped_rows": len(mapped),
        "variant_accuracy": {key: round(value["accuracy"], 6) for key, value in metrics.items()},
        "report_sha256": report["report_sha256"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
