#!/usr/bin/env python3
"""Train open-set membership calibration around the frozen catalog model."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astrograph.diagnosis.sat_sim_catalog import NORMAL_CLASS_ID, load_catalog_artifact  # noqa: E402
from astrograph.diagnosis.sat_sim_open_set import (  # noqa: E402
    save_open_set_artifact,
    sanitize_telemetry,
    train_open_set_artifact,
)


def _truth(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["case_id"]): str(item["effect_id"]) for item in payload["cases"]}


def _labeled_examples(dataset: Path, truth_path: Path) -> dict[str, list[tuple[str, pd.DataFrame]]]:
    truth = _truth(truth_path)
    index = pd.read_csv(dataset / "case_index.csv")
    result: dict[str, list[tuple[str, pd.DataFrame]]] = defaultdict(list)
    for row in index.itertuples(index=False):
        case_dir = dataset / str(row.input_dir)
        result[str(row.backend)].append((NORMAL_CLASS_ID, pd.read_csv(case_dir / "nominal.csv", low_memory=False)))
        result[str(row.backend)].append((truth[str(row.case_id)], pd.read_csv(case_dir / "telemetry.csv", low_memory=False)))
    return dict(result)


def _known_frames(dataset: Path) -> dict[str, list[pd.DataFrame]]:
    index = pd.read_csv(dataset / "case_index.csv")
    result: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for row in index.itertuples(index=False):
        case_dir = dataset / str(row.input_dir)
        result[str(row.backend)].append(pd.read_csv(case_dir / "nominal.csv", low_memory=False))
        result[str(row.backend)].append(pd.read_csv(case_dir / "telemetry.csv", low_memory=False))
    return dict(result)


def _unknown_frames(root: Path, names: list[str]) -> dict[str, list[pd.DataFrame]]:
    result: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for name in names:
        frame = pd.read_csv(root / name / "telemetry.csv", low_memory=False)
        clean = frame[[column for column in frame.columns if not column.lower().startswith(("label.", "truth.")) and "effect.active" not in column.lower()]]
        result["unified"].append(sanitize_telemetry(clean))
    return dict(result)


def _merge(*sources: dict[str, list[tuple[str, pd.DataFrame]]]) -> dict[str, list[tuple[str, pd.DataFrame]]]:
    result: dict[str, list[tuple[str, pd.DataFrame]]] = defaultdict(list)
    for source in sources:
        for backend, values in source.items():
            result[backend].extend(values)
    return dict(result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--development-truth", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--holdout-truth", type=Path, required=True)
    parser.add_argument("--known-calibration", type=Path, required=True)
    parser.add_argument("--unknown-root", type=Path, required=True)
    parser.add_argument("--unknown-names", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-version", default="0.6.5-open-set-model")
    args = parser.parse_args()

    base = load_catalog_artifact(args.base_model)
    training = _merge(
        _labeled_examples(args.development, args.development_truth),
        _labeled_examples(args.holdout, args.holdout_truth),
    )
    known = _known_frames(args.known_calibration)
    unknown_names = [item.strip() for item in args.unknown_names.split(",") if item.strip()]
    unknown = _unknown_frames(args.unknown_root, unknown_names)
    artifact = train_open_set_artifact(
        base_artifact=base,
        training_frames=training,
        known_calibration_frames=known,
        unknown_calibration_frames=unknown,
        artifact_version=args.artifact_version,
    )
    save_open_set_artifact(artifact, args.output)
    sha = hashlib.sha256(args.output.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "astrograph-sat-sim-open-set-model-manifest.v1",
        "artifact_version": args.artifact_version,
        "model_file": args.output.name,
        "model_sha256": sha,
        "base_model_sha256": hashlib.sha256(args.base_model.read_bytes()).hexdigest(),
        "training_sources": [str(args.development), str(args.holdout)],
        "known_calibration_source": str(args.known_calibration),
        "unknown_calibration_source": str(args.unknown_root),
        "unknown_calibration_effects": unknown_names,
        "backend_calibration": {
            backend: {
                "knownness_threshold": float(payload["knownness_threshold"]),
                "threshold_strategy": str(payload["threshold_strategy"]),
                "calibration_known_probability_min": float(payload["calibration_known_probability_min"]),
                "calibration_unknown_probability_max": float(payload["calibration_unknown_probability_max"]),
                "calibration_count_known": int(payload["calibration_count_known"]),
                "calibration_count_unknown": int(payload["calibration_count_unknown"]),
            }
            for backend, payload in artifact["backend_calibration"].items()
        },
        "score_semantics": artifact["score_semantics"],
        "diagnostic_confidence_available": False,
        "claim_boundary": artifact["claim_boundary"],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "model_sha256": sha}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
