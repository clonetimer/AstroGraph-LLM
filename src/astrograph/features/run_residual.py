from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load_manifests(audit_root: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for directory in audit_root.iterdir():
        if directory.is_dir() and len(directory.name) == 32:
            manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8-sig"))
            result[str(manifest["run_id"])] = manifest
    return result


def build_fault_run_residual_features(
    package_root: str | Path,
    split_manifest_path: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    root = Path(package_root)
    manifests = _load_manifests(root / "03_run_audit")
    split = json.loads(Path(split_manifest_path).read_text(encoding="utf-8-sig"))
    split_lookup = {
        str(run_id): str(split_name)
        for split_name, run_ids in split["run_ids"].items()
        for run_id in run_ids
    }

    raw = pd.read_csv(root / "02_dataset" / "aggregate" / "telemetry_raw.csv", low_memory=False)
    feature_cols = [c for c in raw.columns if c not in {"record_id", "run_id", "time_s"}]
    for column in feature_cols:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    run_frames = {
        str(run_id): frame.sort_values("time_s").reset_index(drop=True)
        for run_id, frame in raw.groupby("run_id")
        if manifests[str(run_id)]["case_type"] in {"nominal", "fault"}
    }

    rows: list[dict] = []
    for run_id, manifest in manifests.items():
        if manifest["case_type"] not in {"nominal", "fault"}:
            continue
        current = run_frames[run_id]
        pair_id = run_id if manifest["case_type"] == "nominal" else str(manifest["paired_nominal_run_id"])
        baseline = run_frames[pair_id]
        merged = current[["time_s", *feature_cols]].merge(
            baseline[["time_s", *feature_cols]],
            on="time_s",
            suffixes=("_cur", "_base"),
            how="inner",
        )
        record = {
            "run_id": run_id,
            "case_id": str(manifest["case_id"]),
            "split": split_lookup[run_id],
            "paired_nominal_run_id": pair_id,
        }
        for column in feature_cols:
            values = merged[f"{column}_cur"].to_numpy(dtype=float) - merged[f"{column}_base"].to_numpy(dtype=float)
            record[f"{column}__mean"] = float(np.nanmean(values))
            record[f"{column}__std"] = float(np.nanstd(values))
            record[f"{column}__max_abs"] = float(np.nanmax(np.abs(values)))
            record[f"{column}__mean_abs"] = float(np.nanmean(np.abs(values)))
            record[f"{column}__last"] = float(values[-1])
            record[f"{column}__delta"] = float(values[-1] - values[0])
        rows.append(record)

    features = pd.DataFrame(rows)
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output, index=False, encoding="utf-8-sig")
    return features
