#!/usr/bin/env python3
"""Generate frozen label-free ADCS paired-residual temporal traces."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from astrograph.assurance import canonical_sha256
from astrograph.experiments.model_independent_temporal import (
    FrozenModelIndependentTrace,
    FrozenTraceBundle,
    ModelIndependentScaleContract,
    TraceVariant,
    normalized_residual_scores,
    reference_prefix_statistic,
)

PREFIX_POINTS = 31
PREFIX_QUANTILE = 0.95


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_hashed_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop("manifest_sha256", None)
    actual = canonical_sha256(payload)
    payload["manifest_sha256"] = expected
    if expected != actual:
        raise ValueError(f"manifest hash mismatch for {path}")
    return payload


def load_contract(path: Path) -> ModelIndependentScaleContract:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop("contract_sha256", None)
    payload.pop("fit_components", None)
    fields = set(ModelIndependentScaleContract.model_fields)
    contract = ModelIndependentScaleContract.model_validate({key: value for key, value in payload.items() if key in fields})
    if expected is not None and expected != contract.contract_sha256:
        raise ValueError("scale contract hash mismatch")
    return contract


def read_matrix(runtime_root: Path, relative_path: str, expected_sha256: str, channels: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    path = (runtime_root / relative_path).resolve()
    if runtime_root not in path.parents:
        raise ValueError("runtime telemetry path escapes root")
    if sha256_file(path) != expected_sha256:
        raise ValueError("runtime telemetry SHA-256 mismatch")
    frame = pd.read_csv(path, usecols=["time_s", *channels])
    return frame["time_s"].to_numpy(dtype=np.float64), frame.loc[:, channels].to_numpy(dtype=np.float64)


def shift_reference(reference: np.ndarray, points: int) -> np.ndarray:
    if points <= 0:
        return reference.copy()
    prefix = np.repeat(reference[:1], points, axis=0)
    return np.concatenate([prefix, reference[:-points]], axis=0)


def deterministic_noise(shape: tuple[int, int], scale: np.ndarray, fraction: float, token: str) -> np.ndarray:
    seed = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, fraction, size=shape) * np.broadcast_to(scale, shape)


def variant_matrices(
    variant: TraceVariant,
    current: np.ndarray,
    reference: np.ndarray,
    alternate_reference: np.ndarray,
    contract: ModelIndependentScaleContract,
    token: str,
) -> tuple[np.ndarray, np.ndarray]:
    if variant == TraceVariant.EXACT_PAIRED:
        return current, reference
    if variant == TraceVariant.REFERENCE_SHIFT_1S:
        return current, shift_reference(reference, 1)
    if variant == TraceVariant.REFERENCE_SHIFT_2S:
        return current, shift_reference(reference, 2)
    if variant == TraceVariant.CROSS_MC_REFERENCE:
        return current, alternate_reference
    if variant in {TraceVariant.CURRENT_NOISE_1PCT_SCALE, TraceVariant.CURRENT_NOISE_5PCT_SCALE}:
        fraction = 0.01 if variant == TraceVariant.CURRENT_NOISE_1PCT_SCALE else 0.05
        noisy = current.copy()
        finite = np.isfinite(noisy)
        perturbation = deterministic_noise(noisy.shape, np.asarray(contract.channel_scale), fraction, f"{token}:{variant.value}")
        noisy[finite] += perturbation[finite]
        return noisy, reference
    raise ValueError(f"unsupported trace variant {variant}")


def write_bundle(path: Path, bundle: FrozenTraceBundle) -> None:
    payload = {
        "artifact_schema_version": "astrograph-model-independent-trace-bundle-artifact.v1",
        "bundle": bundle.model_dump(mode="json", exclude_computed_fields=True),
        "trace_sha256s": {trace.opaque_trajectory_id: trace.trace_sha256 for trace in bundle.traces},
        "bundle_sha256": bundle.bundle_sha256,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, compresslevel=9, mtime=0) as handle:
            handle.write(encoded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--scale-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    runtime_root = args.runtime_root.resolve()
    manifest = load_hashed_manifest(args.runtime_manifest.resolve())
    contract = load_contract(args.scale_contract.resolve())
    if manifest.get("ground_truth_fields_present") is not False or manifest.get("source_paths_exposed") is not False:
        raise ValueError("runtime manifest violates label-isolation contract")
    if set(manifest["development_groups"]) != {3, 8} or set(manifest["excluded_groups"]) != {4, 9}:
        raise ValueError("runtime development/post-hoc group contract differs")
    if tuple(manifest["channel_order"]) != contract.channel_order:
        raise ValueError("runtime channel order differs from scale contract")
    if manifest["dataset_archive_sha256"] != contract.dataset_archive_sha256:
        raise ValueError("runtime/scale dataset identity differs")

    variants = tuple(TraceVariant)
    traces_by_variant: dict[TraceVariant, list[FrozenModelIndependentTrace]] = {variant: [] for variant in variants}
    for entry in manifest["entries"]:
        current_time, current = read_matrix(runtime_root, entry["current_relative_path"], entry["current_sha256"], contract.channel_order)
        reference_time, reference = read_matrix(runtime_root, entry["reference_relative_path"], entry["reference_sha256"], contract.channel_order)
        alternate_time, alternate = read_matrix(runtime_root, entry["alternate_reference_relative_path"], entry["alternate_reference_sha256"], contract.channel_order)
        if not np.array_equal(current_time, reference_time) or not np.array_equal(current_time, alternate_time):
            raise ValueError("current/reference timestamp grids differ")
        for variant in variants:
            variant_current, variant_reference = variant_matrices(
                variant, current, reference, alternate, contract, entry["opaque_trajectory_id"]
            )
            score, missing = normalized_residual_scores(variant_current, variant_reference, contract)
            prefix = reference_prefix_statistic(score, prefix_points=PREFIX_POINTS, quantile=PREFIX_QUANTILE)
            traces_by_variant[variant].append(
                FrozenModelIndependentTrace(
                    opaque_trajectory_id=entry["opaque_trajectory_id"],
                    independent_group_id=entry["independent_group_id"],
                    development_group=int(entry["development_group"]),
                    variant=variant,
                    current_telemetry_sha256=entry["current_sha256"],
                    reference_telemetry_sha256=(
                        entry["alternate_reference_sha256"]
                        if variant == TraceVariant.CROSS_MC_REFERENCE
                        else entry["reference_sha256"]
                    ),
                    alternate_reference_sha256=entry["alternate_reference_sha256"],
                    scale_contract_sha256=contract.contract_sha256,
                    time_s=tuple(float(value) for value in current_time),
                    trigger_score=tuple(float(value) for value in score),
                    source_missing_fraction=tuple(float(value) for value in missing),
                    reference_prefix_points=PREFIX_POINTS,
                    reference_prefix_quantile=PREFIX_QUANTILE,
                    reference_prefix_statistic=prefix,
                )
            )

    bundles: dict[str, dict] = {}
    for variant in variants:
        traces = traces_by_variant[variant]
        bundle = FrozenTraceBundle(
            bundle_id=f"ADCS-MI-{variant.value}-R001",
            variant=variant,
            dataset_archive_sha256=contract.dataset_archive_sha256,
            runtime_manifest_sha256=manifest["manifest_sha256"],
            scale_contract_sha256=contract.contract_sha256,
            traces=tuple(sorted(traces, key=lambda item: item.opaque_trajectory_id)),
        )
        output = args.output_dir / f"{variant.value.lower()}.trace_bundle.json.gz"
        write_bundle(output, bundle)
        bundles[variant.value] = {
            "path": output.name,
            "sha256": sha256_file(output),
            "bundle_sha256": bundle.bundle_sha256,
            "trace_count": len(bundle.traces),
            "reference_prefix_min": min(trace.reference_prefix_statistic for trace in bundle.traces),
            "reference_prefix_median": float(np.median([trace.reference_prefix_statistic for trace in bundle.traces])),
            "reference_prefix_max": max(trace.reference_prefix_statistic for trace in bundle.traces),
        }

    index = {
        "schema_version": "astrograph-model-independent-trace-index.v1",
        "dataset_archive_sha256": contract.dataset_archive_sha256,
        "runtime_manifest_sha256": manifest["manifest_sha256"],
        "scale_contract_sha256": contract.contract_sha256,
        "development_groups": [3, 8],
        "excluded_groups": [4, 9],
        "trace_bundles": bundles,
        "truth_access_during_generation": False,
        "stress_outcomes_read_during_generation": False,
    }
    index["index_sha256"] = canonical_sha256(index)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "trace_index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(index, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
