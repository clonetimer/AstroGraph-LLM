#!/usr/bin/env python3
"""Fit a label-free robust ADCS paired-residual scale contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from astrograph.assurance import canonical_sha256
from astrograph.experiments.model_independent_temporal import ModelIndependentScaleContract


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected = raw.pop("manifest_sha256", None)
    actual = canonical_sha256(raw)
    raw["manifest_sha256"] = expected
    if expected != actual:
        raise ValueError("scale-fit manifest content hash mismatch")
    if set(raw["scale_fit_groups"]) & {4, 9}:
        raise ValueError("groups 4/9 cannot fit the scale contract")
    if raw.get("ground_truth_fields_present") is not False:
        raise ValueError("scale-fit manifest must be label-free")
    return raw


def discrete_channel(channel: str) -> bool:
    tokens = (
        ".valid",
        ".saturated",
        ".cell_healthy",
        "axis_valid",
        "invalid_count",
        "source_mode",
        "stuck",
        "command_accepted",
        "solution_fresh",
    )
    return any(token in channel for token in tokens)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quantile", type=float, default=0.99)
    args = parser.parse_args()

    runtime_root = args.runtime_root.resolve()
    manifest = load_manifest(args.manifest.resolve())
    channel_order = tuple(manifest["channel_order"])
    families: dict[str, list[np.ndarray]] = {}
    for entry in manifest["entries"]:
        path = (runtime_root / entry["relative_path"]).resolve()
        if runtime_root not in path.parents:
            raise ValueError("scale-fit path escapes runtime root")
        if sha256_file(path) != entry["sha256"]:
            raise ValueError("scale-fit telemetry SHA-256 mismatch")
        frame = pd.read_csv(path, usecols=["time_s", *channel_order])
        matrix = frame.loc[:, channel_order].to_numpy(dtype=np.float64)
        families.setdefault(entry["reference_family_id"], []).append(matrix)

    channel_values: list[list[np.ndarray]] = [[] for _ in channel_order]
    for family_id, matrices in sorted(families.items()):
        if len(matrices) != len(manifest["scale_fit_groups"]):
            raise ValueError(f"reference family {family_id} has incomplete scale-fit groups")
        stack = np.stack(matrices)
        median_at_time = np.nanmedian(stack, axis=0)
        cross_group = np.abs(stack - median_at_time)
        temporal = np.abs(np.diff(stack, axis=1))
        for index in range(len(channel_order)):
            channel_values[index].append(cross_group[:, :, index].reshape(-1))
            channel_values[index].append(temporal[:, :, index].reshape(-1))

    scales: list[float] = []
    components: list[dict] = []
    for channel, arrays in zip(channel_order, channel_values):
        values = np.concatenate(arrays)
        values = values[np.isfinite(values)]
        quantile_value = float(np.quantile(values, args.quantile)) if len(values) else 0.0
        minimum = 1.0 if discrete_channel(channel) else 1e-6
        scale = max(quantile_value, minimum)
        scales.append(scale)
        components.append({
            "channel": channel,
            "fit_quantile": quantile_value,
            "minimum_scale": minimum,
            "selected_scale": scale,
        })

    contract = ModelIndependentScaleContract(
        contract_id="ADCS-MI-SCALE-R001",
        dataset_archive_sha256=manifest["dataset_archive_sha256"],
        scale_fit_manifest_sha256=manifest["manifest_sha256"],
        scale_fit_groups=tuple(manifest["scale_fit_groups"]),
        excluded_groups=tuple(manifest["excluded_groups"]),
        channel_order=channel_order,
        channel_scale=tuple(scales),
        scale_quantile=args.quantile,
    )
    output = {
        **contract.model_dump(mode="json"),
        "fit_components": components,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "contract_sha256": contract.contract_sha256,
        "channel_count": len(channel_order),
        "scale_min": min(scales),
        "scale_median": float(np.median(scales)),
        "scale_max": max(scales),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
