#!/usr/bin/env python3
"""Run a truth-isolated, trajectory-level A0–A7 ADCS v5.2.2 ablation.

Inference consumes telemetry only.  ``diagnostic_class`` from ``sample_index.csv``
is attached only after every branch has produced its ranking, for offline scoring.
The experiment evaluates the frozen test split (sample_index 4/9) at one terminal
20-sample window per fault trajectory.  KG evidence remains the configured
label-free paired full-run E0 evidence package.

This is a research ablation, not a qualification gate.  In particular, the model
probabilities are uncalibrated and the six independent test trajectory groups
limit statistical precision.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from astrograph.assurance import ArtifactRoot, canonical_sha256
from astrograph.data_driven import ModelInferenceRequest, ObservationScope
from astrograph.evidence.configured_runtime import ConfiguredRuntimeEvidenceBuilder
from astrograph.experiments import (
    AblationVariant,
    FrozenAblationPlan,
    FrozenAblationRunner,
    GroupedPrediction,
    paired_group_accuracy_delta,
    summarize_grouped_classification,
)
from astrograph.fusion import BiRRFConfig
from astrograph.fusion.ranking import build_backward_evidence_ranking, build_forward_graph_ranking
from astrograph.integrations.adcs_three_task_torch import (
    build_registry_from_adcs_three_task_provider,
    load_adcs_three_task_torch_package,
)
from astrograph.simulation.models import SimulationRunCompleted

TEST_INDICES = frozenset({4, 9})
DEFAULT_EVIDENCE = ROOT / "configs" / "datasets" / "adcs_v5_2_2" / "runtime_evidence.json"
DEFAULT_FAULT_MAP = ROOT / "configs" / "datasets" / "adcs_v5_2_2" / "fault_class_map.json"
BOOTSTRAP_SEED = 20260829


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(dataset_root: Path) -> list[dict[str, str]]:
    with (dataset_root / "sample_index.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def evaluation_fingerprint(
    dataset_root: Path,
    test_fault_rows: list[dict[str, str]],
    nominal_by_group: dict[str, dict[str, str]],
    *,
    include_labels: bool = False,
) -> tuple[str, dict[str, str], dict[str, str]]:
    relative_files = {
        row["telemetry_file"] for row in test_fault_rows
    } | {
        nominal_by_group[row["trajectory_pair_id"]]["telemetry_file"] for row in test_fault_rows
    }
    hashes = {
        relative: sha256_file(dataset_root / relative)
        for relative in sorted(relative_files)
    }
    label_hashes = (
        {
            row["labels_file"]: sha256_file(dataset_root / row["labels_file"])
            for row in sorted(test_fault_rows, key=lambda item: item["labels_file"])
        }
        if include_labels else {}
    )
    identity = {
        "schema_version": "astrograph-adcs-v522-evaluation-dataset-identity.v1",
        "sample_index_sha256": sha256_file(dataset_root / "sample_index.csv"),
        "telemetry_sha256_by_relative_path": hashes,
        "label_sha256_by_relative_path": label_hashes,
    }
    return canonical_sha256(identity), hashes, label_hashes


def split_identity(rows: list[dict[str, str]]) -> str:
    return canonical_sha256(
        {
            "policy": "sample_index modulo 5; frozen test={4,9}",
            "rows": [
                {
                    "sample_id": row["sample_id"],
                    "trajectory_pair_id": row["trajectory_pair_id"],
                    "sample_index": int(row["sample_index"]),
                }
                for row in sorted(rows, key=lambda item: item["sample_id"])
            ],
        }
    )


def _top_k_ids(result, k: int = 3) -> set[str]:
    if not result.candidates:
        return set()
    return {item.fault_id for item in result.candidates if item.final_rank <= k}


def _class_counts(records: list[GroupedPrediction]) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    by_truth: dict[str, list[GroupedPrediction]] = defaultdict(list)
    for item in records:
        by_truth[item.truth].append(item)
    for truth, rows in sorted(by_truth.items()):
        output[truth] = {
            "count": len(rows),
            "correct": sum(item.prediction == truth for item in rows),
            "abstained": sum(item.prediction is None for item in rows),
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-package", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--window-policy",
        choices=("terminal", "oracle_active_plus_delay", "max_detection_stride"),
        default="terminal",
    )
    parser.add_argument("--oracle-delay-samples", type=int, default=30)
    parser.add_argument(
        "--scan-stride", type=int, default=5,
        help="label-free endpoint stride for max_detection_stride",
    )
    parser.add_argument("--start-index", type=int, default=0, help="zero-based slice start for bounded execution")
    parser.add_argument("--limit", type=int, default=0, help="slice length; 0 evaluates from start through the end")
    args = parser.parse_args()

    if args.torch_threads <= 0:
        raise ValueError("--torch-threads must be positive")
    import torch
    torch.set_num_threads(args.torch_threads)

    dataset_root = args.dataset_root.resolve()
    model_package = args.model_package.resolve()
    artifacts = ArtifactRoot(args.output_dir, source_root=ROOT)
    artifacts.ensure()

    if sha256_file(model_package) != args.model_sha256:
        raise ValueError("model package SHA-256 does not match --model-sha256")
    provider = load_adcs_three_task_torch_package(
        model_package,
        expected_sha256=args.model_sha256,
        device=args.device,
    )
    registry = build_registry_from_adcs_three_task_provider(provider)
    evidence_builder = ConfiguredRuntimeEvidenceBuilder(DEFAULT_EVIDENCE)
    fault_map = json.loads(DEFAULT_FAULT_MAP.read_text(encoding="utf-8"))["mappings"]
    fault_universe = tuple(sorted(provider.runtime_fault_ids))
    if set(fault_map.values()) != set(fault_universe):
        raise ValueError("ADCS class map and model runtime fault universe differ")

    all_rows = load_rows(dataset_root)
    test_rows = [row for row in all_rows if int(row["sample_index"]) in TEST_INDICES]
    full_test_fault_rows = [row for row in test_rows if row["diagnostic_class"] != "nominal"]
    nominal_by_group = {
        row["trajectory_pair_id"]: row
        for row in test_rows
        if row["diagnostic_class"] == "nominal"
    }
    missing_nominal = sorted(
        {row["trajectory_pair_id"] for row in full_test_fault_rows} - set(nominal_by_group)
    )
    if missing_nominal:
        raise ValueError(f"test trajectories lack paired nominal runs: {missing_nominal}")
    if args.start_index < 0 or args.start_index > len(full_test_fault_rows):
        raise ValueError("--start-index is outside the frozen fault trajectory range")
    stop_index = (
        len(full_test_fault_rows)
        if args.limit <= 0
        else min(len(full_test_fault_rows), args.start_index + args.limit)
    )
    test_fault_rows = full_test_fault_rows[args.start_index:stop_index]

    if args.oracle_delay_samples < 0:
        raise ValueError("--oracle-delay-samples must be nonnegative")
    if args.scan_stride <= 0:
        raise ValueError("--scan-stride must be positive")
    oracle_window = args.window_policy == "oracle_active_plus_delay"
    scanned_window = args.window_policy == "max_detection_stride"
    dataset_sha, telemetry_hashes, label_hashes = evaluation_fingerprint(
        dataset_root, full_test_fault_rows, nominal_by_group, include_labels=oracle_window
    )
    split_sha = split_identity(full_test_fault_rows)
    base_config = BiRRFConfig()
    plan = FrozenAblationPlan(
        plan_id=f"adcs-v5.2.2-{args.window_policy}-a0-a7-b005",
        dataset_id="whole-spacecraft-adcs-paired-dataset-v5.2.2:test-fault-trajectories",
        dataset_sha256=dataset_sha,
        split_id="frozen-test-sample-index-4-9",
        split_sha256=split_sha,
        calibration_id="UNCALIBRATED_P2_MODEL_OUTPUT",
        calibration_sha256=canonical_sha256(
            {
                "calibrated": False,
                "model_package_sha256": args.model_sha256,
                "reason": "P2 model does not declare CALIBRATED_PROBABILITY",
            }
        ),
        fault_universe=fault_universe,
        policy_id="birrf-default-engineering-policy",
        policy_sha256=canonical_sha256(base_config.model_dump(mode="json")),
        code_version="0.7.1.post24+B005",
        data_branch_id=f"{provider.manifest.manifest_id}:{args.model_sha256}",
        forward_branch_id=(
            f"configured-runtime-forward:{evidence_builder.package.package_id}@"
            f"{evidence_builder.package.version}:{evidence_builder.package_sha256}"
        ),
        backward_branch_id=(
            f"configured-runtime-backward:{evidence_builder.package.package_id}@"
            f"{evidence_builder.package.version}:{evidence_builder.package_sha256}"
        ),
        base_config=base_config,
        metadata={
            "evaluation_unit": (
                "one terminal model window per held-out fault trajectory"
                if args.window_policy == "terminal" else
                (
                    "one oracle-selected active-fault model window per held-out fault trajectory"
                    if oracle_window else
                    "one telemetry-only maximum-detection window selected from a fixed-stride scan per held-out fault trajectory"
                )
            ),
            "model_window_length": int(provider.manifest.input_contract["window_length"]),
            "kg_evidence_window": "paired full run",
            "window_policy": args.window_policy,
            "oracle_delay_samples": args.oracle_delay_samples if oracle_window else None,
            "scan_stride": args.scan_stride if scanned_window else None,
            "truth_isolation": (
                "ORACLE CONDITIONAL TASK: fault-active labels choose the evaluation window; label values are not model features"
                if oracle_window else
                "diagnostic_class and fault-active labels are not used in inference or window selection; inference reads telemetry only"
            ),
            "independent_group_key": "trajectory_pair_id",
            "expected_test_group_count": 6,
            "expected_fault_trajectory_count": 176,
        },
    )
    runner = FrozenAblationRunner(plan)

    records_by_variant: dict[str, list[GroupedPrediction]] = {
        variant.value: [] for variant in AblationVariant
    }
    episode_rows: list[dict[str, Any]] = []
    telemetry_sha_by_observation: dict[str, str] = {}
    gates_failed = 0
    dependency_adjustment_counts: Counter[str] = Counter()
    top3_hits: Counter[str] = Counter()

    channels = list(provider.selected_channels)
    window_length = int(provider.manifest.input_contract["window_length"])
    nominal_frames = {
        group_id: pd.read_csv(dataset_root / row["telemetry_file"], low_memory=False)
        for group_id, row in nominal_by_group.items()
    }

    for number, row in enumerate(test_fault_rows, start=1):
        nominal = nominal_by_group[row["trajectory_pair_id"]]
        telemetry_path = dataset_root / row["telemetry_file"]
        nominal_path = dataset_root / nominal["telemetry_file"]
        telemetry_sha = telemetry_hashes[row["telemetry_file"]]
        event = SimulationRunCompleted(
            event_id=f"evt-{telemetry_sha[:20]}",
            run_id=f"run-{telemetry_sha[:20]}",
            run_dir=str(telemetry_path.parent),
            nominal_run_dir=str(nominal_path.parent),
            runtime_id="adcs-v5.2.2-offline-ablation",
        )
        source_frame = pd.read_csv(telemetry_path, low_memory=False)
        bundle, gate = evidence_builder.build_from_frames(
            event, source=source_frame, nominal=nominal_frames[row["trajectory_pair_id"]]
        )
        if not gate.passed:
            gates_failed += 1
            raise ValueError(f"runtime evidence gate failed for held-out sample {number}")

        missing_model_columns = sorted({"time_s", *channels} - set(source_frame.columns))
        if missing_model_columns:
            raise ValueError(f"held-out telemetry lacks model columns: {missing_model_columns}")
        frame = source_frame[["time_s", *channels]]
        if len(frame) < window_length:
            raise ValueError("held-out telemetry is shorter than the frozen model window")
        times = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(float)
        numeric_values = (
            frame[channels]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(np.float32)
        )

        def request_for_endpoint(endpoint_index: int) -> ModelInferenceRequest:
            window_start = endpoint_index - window_length + 1
            values = numeric_values[window_start : endpoint_index + 1]
            time_values = times[window_start : endpoint_index + 1]
            if not np.isfinite(values).all() or not np.isfinite(time_values).all():
                raise ValueError("held-out model window contains non-finite values")
            payload = {
                "channel_order": channels,
                "telemetry_window": values.tolist(),
            }
            return ModelInferenceRequest(
                episode_id=bundle.episode.episode_id,
                model_id=provider.manifest.model_id,
                model_version=provider.manifest.model_version,
                feature_schema_id=provider.manifest.feature_schema_ids[0],
                input_data_sha256=canonical_sha256(payload),
                registered_fault_ids=list(fault_universe),
                payload=payload,
                observation_scope=ObservationScope(
                    scope_id=f"{args.window_policy}-{telemetry_sha[:20]}-{endpoint_index}",
                    source_id="whole-spacecraft-adcs-v5.2.2-telemetry",
                    window_start_s=float(times[window_start]),
                    window_end_s=float(times[endpoint_index]),
                    source_data_sha256=telemetry_sha,
                    trajectory_group_id=row["trajectory_pair_id"],
                ),
            )

        selection_candidate_count = 1
        selected_detection_probability: float | None = None
        if oracle_window:
            labels = pd.read_csv(dataset_root / row["labels_file"], usecols=["label.fault_active"])
            raw_active = labels["label.fault_active"]
            numeric_active = pd.to_numeric(raw_active, errors="coerce")
            active = (
                numeric_active.fillna(0).to_numpy(float) > 0.5
                if numeric_active.notna().any()
                else raw_active.astype(str).str.lower().isin({"1", "true", "yes"}).to_numpy()
            )
            indices = np.flatnonzero(active)
            if not len(indices):
                raise ValueError("oracle active-window policy requires at least one active fault label")
            endpoint = min(
                len(frame) - 1,
                max(window_length - 1, int(indices[0]) + args.oracle_delay_samples),
            )
            prediction = registry.predict(request_for_endpoint(endpoint))
        elif scanned_window:
            candidate_endpoints = list(range(window_length - 1, len(frame), args.scan_stride))
            if candidate_endpoints[-1] != len(frame) - 1:
                candidate_endpoints.append(len(frame) - 1)
            requests: list[ModelInferenceRequest] = []
            eligible_endpoints: list[int] = []
            for candidate_endpoint in candidate_endpoints:
                try:
                    requests.append(request_for_endpoint(candidate_endpoint))
                except ValueError as exc:
                    if "non-finite" not in str(exc):
                        raise
                    continue
                eligible_endpoints.append(candidate_endpoint)
            if not requests:
                raise ValueError("no finite telemetry window is available for label-free scan")
            predictions = registry.predict_many(requests)
            detection_scores = [
                float(item.metadata["raw_output_metadata"]["detection_probability"])
                for item in predictions
            ]
            # Ties select the earliest endpoint, which is deterministic and favors
            # earlier diagnosis without consulting labels.
            selected_index = max(
                range(len(predictions)),
                key=lambda index: (detection_scores[index], -eligible_endpoints[index]),
            )
            endpoint = eligible_endpoints[selected_index]
            prediction = predictions[selected_index]
            selection_candidate_count = len(predictions)
            selected_detection_probability = detection_scores[selected_index]
        else:
            endpoint = len(frame) - 1
            prediction = registry.predict(request_for_endpoint(endpoint))
        if selected_detection_probability is None:
            selected_detection_probability = float(
                prediction.metadata["raw_output_metadata"]["detection_probability"]
            )
        data_ranking = prediction.to_fault_ranking(list(fault_universe))
        forward_ranking = build_forward_graph_ranking(bundle)
        backward_ranking = build_backward_evidence_ranking(
            bundle,
            candidate_fault_ids=fault_universe,
        )
        run = runner.run(
            data_factory=lambda ranking=data_ranking: ranking,
            forward_factory=lambda ranking=forward_ranking: ranking,
            backward_factory=lambda ranking=backward_ranking: ranking,
        )

        # Truth enters only here, after all branch outputs and A0–A7 results exist.
        truth = fault_map[row["diagnostic_class"]]
        evaluation_observation_id = "EVAL-" + canonical_sha256({"sample_id": row["sample_id"]})[:24].upper()
        telemetry_sha_by_observation[evaluation_observation_id] = telemetry_sha
        variant_top3: dict[str, bool] = {}
        variant_dependency_adjusted: dict[str, bool] = {}
        for variant in AblationVariant:
            result = run.results[variant.value]
            records_by_variant[variant.value].append(
                GroupedPrediction(
                    observation_id=evaluation_observation_id,
                    group_id=row["trajectory_pair_id"],
                    truth=truth,
                    prediction=result.top_fault_id,
                )
            )
            variant_top3[variant.value] = truth in _top_k_ids(result, 3)
            if variant_top3[variant.value]:
                top3_hits[variant.value] += 1
            variant_dependency_adjusted[variant.value] = bool(
                result.metadata.get("evidence_lineage_dependency_adjusted_fault_ids")
            )
            if variant_dependency_adjusted[variant.value]:
                dependency_adjustment_counts[variant.value] += 1

        episode_rows.append(
            {
                "observation_id": evaluation_observation_id,
                "inference_episode_id": bundle.episode.episode_id,
                "telemetry_sha256": telemetry_sha,
                "model_window_endpoint_index": endpoint,
                "window_selection_candidate_count": selection_candidate_count,
                "selected_detection_probability": selected_detection_probability,
                "trajectory_group_id": row["trajectory_pair_id"],
                "truth": truth,
                "data_top": data_ranking.candidates[0].fault_id if data_ranking.candidates else None,
                "forward_top": (
                    next((item.fault_id for item in forward_ranking.candidates if item.rank == 1), None)
                ),
                "backward_top": (
                    next((item.fault_id for item in backward_ranking.candidates if item.rank == 1), None)
                ),
                **{variant.value: run.results[variant.value].top_fault_id for variant in AblationVariant},
                **{f"{variant.value}__top3_hit": variant_top3[variant.value] for variant in AblationVariant},
                **{
                    f"{variant.value}__dependency_adjusted": variant_dependency_adjusted[variant.value]
                    for variant in AblationVariant
                },
            }
        )
        if number % 20 == 0 or number == len(test_fault_rows):
            print(f"[adcs-a0-a7] {number}/{len(test_fault_rows)}", flush=True)

    telemetry_counts = Counter(telemetry_sha_by_observation.values())
    duplicate_telemetry_hashes = {key: value for key, value in telemetry_counts.items() if value > 1}
    unique_observation_ids: set[str] = set()
    seen_telemetry: set[str] = set()
    for row in episode_rows:
        telemetry_sha = str(row["telemetry_sha256"])
        if telemetry_sha not in seen_telemetry:
            seen_telemetry.add(telemetry_sha)
            unique_observation_ids.add(str(row["observation_id"]))
    unique_records_by_variant = {
        variant.value: [
            item for item in records_by_variant[variant.value]
            if item.observation_id in unique_observation_ids
        ]
        for variant in AblationVariant
    }

    summaries = {
        variant.value: summarize_grouped_classification(
            records_by_variant[variant.value],
            label_universe=fault_universe,
            bootstrap_iterations=args.bootstrap_iterations,
            bootstrap_seed=BOOTSTRAP_SEED,
        ).model_dump(mode="json")
        for variant in AblationVariant
    }
    unique_summaries = {
        variant.value: summarize_grouped_classification(
            unique_records_by_variant[variant.value],
            label_universe=fault_universe,
            bootstrap_iterations=args.bootstrap_iterations,
            bootstrap_seed=BOOTSTRAP_SEED,
        ).model_dump(mode="json")
        for variant in AblationVariant
    }
    paired_deltas: dict[str, Any] = {}
    for reference in (AblationVariant.A0_DATA_ONLY, AblationVariant.A6_ALL_UNADJUSTED):
        delta = paired_group_accuracy_delta(
            records_by_variant[reference.value],
            records_by_variant[AblationVariant.A7_ALL_LINEAGE_ADJUSTED.value],
            reference_id=reference.value,
            candidate_id=AblationVariant.A7_ALL_LINEAGE_ADJUSTED.value,
            bootstrap_iterations=args.bootstrap_iterations,
            bootstrap_seed=BOOTSTRAP_SEED,
        )
        paired_deltas[f"A7_minus_{reference.value}"] = delta.model_dump(mode="json")

    report = {
        "schema_version": "astrograph-adcs-v522-frozen-ablation-report.v1",
        "status": (
            "FAILED" if gates_failed else
            ("COMPLETED" if len(test_fault_rows) == len(full_test_fault_rows) else "PARTIAL_SLICE")
        ),
        "plan": plan.model_dump(mode="json"),
        "plan_sha256": plan.plan_sha256,
        "model_package": {
            "sha256": args.model_sha256,
            "manifest_id": provider.manifest.manifest_id,
            "calibrated_probabilities": False,
        },
        "evaluation": {
            "fault_trajectory_count": len(test_fault_rows),
            "full_fault_trajectory_count": len(full_test_fault_rows),
            "slice_start_index": args.start_index,
            "slice_stop_index_exclusive": stop_index,
            "independent_group_count": len({row["trajectory_pair_id"] for row in test_fault_rows}),
            "fault_class_count": len(fault_universe),
            "runtime_data_gate_failures": gates_failed,
            "exact_duplicate_telemetry_row_count": sum(telemetry_counts.values()) - len(telemetry_counts),
            "exact_duplicate_telemetry_hash_count": len(duplicate_telemetry_hashes),
            "unique_telemetry_count": len(telemetry_counts),
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "variant_metrics": {
            variant.value: {
                **summaries[variant.value],
                "top3_recall": (
                    top3_hits[variant.value] / len(test_fault_rows) if test_fault_rows else 0.0
                ),
                "class_counts": _class_counts(records_by_variant[variant.value]),
                "episodes_with_dependency_adjustment": dependency_adjustment_counts[variant.value],
            }
            for variant in AblationVariant
        },
        "unique_telemetry_sensitivity": {
            "selection": "first evaluation row per exact telemetry SHA-256; duplicate hashes never cross diagnostic classes",
            "observation_count": len(unique_observation_ids),
            "variant_metrics": unique_summaries,
        },
        "paired_group_effects": paired_deltas,
        "limitations": [
            "Only six independent frozen test trajectory groups are available.",
            "The frozen 176-row test set contains exact duplicate telemetry rows; unique-telemetry sensitivity is reported separately.",
            "The P2 model probabilities are uncalibrated; no probability-based risk claim is made.",
            (
                "The oracle-active experiment uses ground-truth fault activity only to choose the evaluation window and is not end-to-end."
                if oracle_window else
                (
                    "The label-free maximum-detection selector reuses the frozen detector to choose among fixed-stride telemetry windows; it is not independently calibrated."
                    if scanned_window else
                    "The terminal-window experiment is label-free in window selection but may evaluate after transient faults have cleared."
                )
            ),
            "The data branch uses one selected 20-sample window while configured KG evidence aggregates the paired full run.",
            "Configured rule/KG evidence covers only four of the fourteen ADCS fault IDs.",
            "This experiment evaluates fault identification only; nominal detection is not scored here.",
            "Synthetic paired evidence is E0 and does not establish external or flight validity.",
        ],
        "claim_boundary": (
            (
                "Held-out synthetic trajectory-level research ablation. The terminal policy is label-free in window selection; "
                "the oracle policy uses truth only to select an active diagnostic window and therefore measures conditional identification, not end-to-end diagnosis. "
                "Results do not change AstroGraph G0-G4 or flight qualification status."
            )
        ),
    }
    artifacts.write_json("report.json", report)

    fieldnames = [
        "observation_id",
        "inference_episode_id",
        "telemetry_sha256",
        "model_window_endpoint_index",
        "window_selection_candidate_count",
        "selected_detection_probability",
        "trajectory_group_id",
        "truth",
        "data_top",
        "forward_top",
        "backward_top",
        *[variant.value for variant in AblationVariant],
        *[f"{variant.value}__top3_hit" for variant in AblationVariant],
        *[f"{variant.value}__dependency_adjusted" for variant in AblationVariant],
    ]
    artifacts.write_generated(
        "predictions.csv",
        lambda path: _write_csv(path, episode_rows, fieldnames),
    )
    print(json.dumps({
        "status": report["status"],
        "fault_trajectories": len(test_fault_rows),
        "groups": report["evaluation"]["independent_group_count"],
        "variant_accuracy": {
            key: round(value["accuracy"], 6)
            for key, value in report["variant_metrics"].items()
        },
        "output": str(args.output_dir.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
