#!/usr/bin/env python3
"""Generate frozen label-free bounded mismatch-tolerant ADCS temporal traces."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from astrograph.assurance import canonical_sha256
from astrograph.experiments.mismatch_tolerant_temporal import (
    FrozenMismatchTolerantTrace,
    FrozenMismatchTraceBundle,
    MismatchTolerantAlignmentContract,
    REQUIRED_DEVELOPMENT_VARIANTS,
    generate_mismatch_tolerant_scores,
    shift_reference_with_validity,
)
from astrograph.experiments.model_independent_temporal import (
    ModelIndependentScaleContract,
    TraceVariant,
)


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


def load_scale_contract(path: Path) -> ModelIndependentScaleContract:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop("contract_sha256", None)
    payload.pop("fit_components", None)
    fields = set(ModelIndependentScaleContract.model_fields)
    contract = ModelIndependentScaleContract.model_validate(
        {key: value for key, value in payload.items() if key in fields}
    )
    if expected is not None and expected != contract.contract_sha256:
        raise ValueError("scale contract hash mismatch")
    return contract


def load_alignment_contract(path: Path) -> MismatchTolerantAlignmentContract:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop("contract_sha256", None)
    payload.pop("candidate_lags", None)
    fields = set(MismatchTolerantAlignmentContract.model_fields)
    contract = MismatchTolerantAlignmentContract.model_validate(
        {key: value for key, value in payload.items() if key in fields}
    )
    if expected is not None and expected != contract.contract_sha256:
        raise ValueError("alignment contract hash mismatch")
    return contract


def read_matrix(
    runtime_root: Path,
    relative_path: str,
    expected_sha256: str,
    channels: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    path = (runtime_root / relative_path).resolve()
    if runtime_root not in path.parents:
        raise ValueError("runtime telemetry path escapes root")
    if sha256_file(path) != expected_sha256:
        raise ValueError("runtime telemetry SHA-256 mismatch")
    frame = pd.read_csv(path, usecols=["time_s", *channels])
    return (
        frame["time_s"].to_numpy(dtype=np.float64),
        frame.loc[:, channels].to_numpy(dtype=np.float64),
    )


def create_shifted_stress_reference(reference: np.ndarray, points: int) -> np.ndarray:
    shifted, _ = shift_reference_with_validity(reference, points)
    return shifted


def deterministic_noise(
    shape: tuple[int, int],
    scale: np.ndarray,
    fraction: float,
    token: str,
    variant: TraceVariant,
) -> np.ndarray:
    seed = int(hashlib.sha256(f"{token}:{variant.value}".encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, fraction, size=shape) * np.broadcast_to(scale, shape)


def variant_matrices(
    variant: TraceVariant,
    current: np.ndarray,
    reference: np.ndarray,
    alternate_reference: np.ndarray,
    scale_contract: ModelIndependentScaleContract,
    token: str,
) -> tuple[np.ndarray, np.ndarray]:
    if variant == TraceVariant.EXACT_PAIRED:
        return current, reference
    if variant == TraceVariant.REFERENCE_SHIFT_1S:
        return current, create_shifted_stress_reference(reference, 1)
    if variant == TraceVariant.REFERENCE_SHIFT_2S:
        return current, create_shifted_stress_reference(reference, 2)
    if variant == TraceVariant.CROSS_MC_REFERENCE:
        return current, alternate_reference
    if variant in {
        TraceVariant.CURRENT_NOISE_1PCT_SCALE,
        TraceVariant.CURRENT_NOISE_5PCT_SCALE,
    }:
        fraction = 0.01 if variant == TraceVariant.CURRENT_NOISE_1PCT_SCALE else 0.05
        noisy = current.copy()
        finite = np.isfinite(noisy)
        perturbation = deterministic_noise(
            noisy.shape,
            np.asarray(scale_contract.channel_scale, dtype=np.float64),
            fraction,
            token,
            variant,
        )
        noisy[finite] += perturbation[finite]
        return noisy, reference
    raise ValueError(f"unsupported trace variant {variant}")


def write_bundle(path: Path, bundle: FrozenMismatchTraceBundle) -> None:
    payload = {
        "artifact_schema_version": "astrograph-mismatch-tolerant-trace-bundle-artifact.v1",
        "bundle": bundle.model_dump(mode="json", exclude_computed_fields=True),
        "trace_sha256s": {
            trace.opaque_trajectory_id: trace.trace_sha256 for trace in bundle.traces
        },
        "bundle_sha256": bundle.bundle_sha256,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_handle,
            compresslevel=9,
            mtime=0,
        ) as handle:
            handle.write(encoded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--scale-contract", type=Path, required=True)
    parser.add_argument("--alignment-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    runtime_root = args.runtime_root.resolve()
    manifest = load_hashed_manifest(args.runtime_manifest.resolve())
    scale_contract = load_scale_contract(args.scale_contract.resolve())
    alignment_contract = load_alignment_contract(args.alignment_contract.resolve())

    if manifest.get("ground_truth_fields_present") is not False:
        raise ValueError("runtime manifest contains or exposes ground truth")
    if manifest.get("source_paths_exposed") is not False:
        raise ValueError("runtime manifest exposes source paths")
    if set(manifest["development_groups"]) != {3, 8}:
        raise ValueError("runtime development groups must be exactly 3/8")
    if set(manifest["excluded_groups"]) != {4, 9}:
        raise ValueError("runtime excluded groups must be exactly 4/9")
    if tuple(manifest["channel_order"]) != scale_contract.channel_order:
        raise ValueError("runtime channel order differs from scale contract")
    if manifest["dataset_archive_sha256"] != scale_contract.dataset_archive_sha256:
        raise ValueError("runtime/scale dataset identity differs")
    if alignment_contract.dataset_archive_sha256 != scale_contract.dataset_archive_sha256:
        raise ValueError("alignment/scale dataset identity differs")
    if alignment_contract.scale_contract_sha256 != scale_contract.contract_sha256:
        raise ValueError("alignment/scale contract identity differs")

    traces_by_variant: dict[TraceVariant, list[FrozenMismatchTolerantTrace]] = {
        variant: [] for variant in REQUIRED_DEVELOPMENT_VARIANTS
    }
    for entry in manifest["entries"]:
        current_time, current = read_matrix(
            runtime_root,
            entry["current_relative_path"],
            entry["current_sha256"],
            scale_contract.channel_order,
        )
        reference_time, reference = read_matrix(
            runtime_root,
            entry["reference_relative_path"],
            entry["reference_sha256"],
            scale_contract.channel_order,
        )
        alternate_time, alternate = read_matrix(
            runtime_root,
            entry["alternate_reference_relative_path"],
            entry["alternate_reference_sha256"],
            scale_contract.channel_order,
        )
        if not np.array_equal(current_time, reference_time) or not np.array_equal(
            current_time, alternate_time
        ):
            raise ValueError("current/reference timestamp grids differ")
        if not np.allclose(
            np.diff(current_time),
            alignment_contract.sample_period_s,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("telemetry sample period differs from alignment contract")

        for variant in REQUIRED_DEVELOPMENT_VARIANTS:
            variant_current, variant_reference = variant_matrices(
                variant,
                current,
                reference,
                alternate,
                scale_contract,
                entry["opaque_trajectory_id"],
            )
            score, missing, valid, baseline, alignment = generate_mismatch_tolerant_scores(
                variant_current,
                variant_reference,
                scale_contract,
                alignment_contract,
            )
            traces_by_variant[variant].append(
                FrozenMismatchTolerantTrace(
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
                    scale_contract_sha256=scale_contract.contract_sha256,
                    alignment_contract_sha256=alignment_contract.contract_sha256,
                    time_s=tuple(float(value) for value in current_time),
                    trigger_score=tuple(float(value) for value in score),
                    source_missing_fraction=tuple(float(value) for value in missing),
                    alignment_valid=tuple(bool(value) for value in valid),
                    prefix_baseline_score=baseline,
                    alignment=alignment,
                )
            )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_index: dict[str, dict] = {}
    for variant in REQUIRED_DEVELOPMENT_VARIANTS:
        bundle = FrozenMismatchTraceBundle(
            bundle_id=f"ADCS-MT-{variant.value}-R001",
            variant=variant,
            dataset_archive_sha256=scale_contract.dataset_archive_sha256,
            runtime_manifest_sha256=manifest["manifest_sha256"],
            scale_contract_sha256=scale_contract.contract_sha256,
            alignment_contract_sha256=alignment_contract.contract_sha256,
            traces=tuple(
                sorted(
                    traces_by_variant[variant],
                    key=lambda item: item.opaque_trajectory_id,
                )
            ),
        )
        output = output_dir / f"{variant.value.lower()}.mismatch_trace_bundle.json.gz"
        write_bundle(output, bundle)
        fits = np.asarray([trace.alignment.fit_score for trace in bundle.traces])
        lags: dict[str, int] = {}
        for trace in bundle.traces:
            key = str(trace.alignment.selected_lag_points)
            lags[key] = lags.get(key, 0) + 1
        bundle_index[variant.value] = {
            "path": output.name,
            "sha256": sha256_file(output),
            "bundle_sha256": bundle.bundle_sha256,
            "trace_count": len(bundle.traces),
            "alignment_fit_quantiles": {
                "q00": float(np.quantile(fits, 0.0)),
                "q50": float(np.quantile(fits, 0.5)),
                "q90": float(np.quantile(fits, 0.9)),
                "q99": float(np.quantile(fits, 0.99)),
                "q100": float(np.quantile(fits, 1.0)),
            },
            "selected_lag_distribution": dict(sorted(lags.items(), key=lambda item: int(item[0]))),
        }

    index = {
        "schema_version": "astrograph-mismatch-tolerant-trace-index.v1",
        "dataset_archive_sha256": scale_contract.dataset_archive_sha256,
        "runtime_manifest_sha256": manifest["manifest_sha256"],
        "scale_contract_sha256": scale_contract.contract_sha256,
        "alignment_contract_sha256": alignment_contract.contract_sha256,
        "development_groups": [3, 8],
        "excluded_groups": [4, 9],
        "development_variants": [
            variant.value for variant in REQUIRED_DEVELOPMENT_VARIANTS
        ],
        "trace_bundles": bundle_index,
        "truth_access_during_generation": False,
        "outcome_data_read_during_generation": False,
        "full_trajectory_alignment_used": False,
        "alignment_prefix_points": alignment_contract.prefix_points,
        "maximum_absolute_lag_points": alignment_contract.maximum_absolute_lag_points,
        "padded_edge_action": alignment_contract.padded_edge_action,
    }
    index["index_sha256"] = canonical_sha256(index)
    (output_dir / "mismatch_trace_index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(index, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
