#!/usr/bin/env python3
"""Train the single-trace Sat-Sim catalog diagnosis model.

Training labels are loaded from an external truth manifest.  Model input files remain
label-isolated and contain only Basilisk telemetry plus backend metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astrograph.diagnosis.sat_sim_catalog import (  # noqa: E402
    NORMAL_CLASS_ID,
    save_catalog_artifact,
    train_catalog_artifact,
)


def _truth(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["case_id"]): str(item["effect_id"]) for item in payload["cases"]}


def _examples(dataset: Path, truth_path: Path) -> tuple[dict[str, list[tuple[str, pd.DataFrame]]], dict[str, int]]:
    truth = _truth(truth_path)
    index = pd.read_csv(dataset / "case_index.csv")
    examples: dict[str, list[tuple[str, pd.DataFrame]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    for row in index.itertuples(index=False):
        case_dir = dataset / str(row.input_dir)
        if str(row.case_id) not in truth:
            raise RuntimeError(f"missing external truth for {row.case_id}")
        manifest = json.loads((case_dir / "input_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("truth_available_to_diagnosis") is not False or manifest.get("labels_removed") is not True:
            raise RuntimeError(f"input isolation contract failed: {row.case_id}")
        nominal = pd.read_csv(case_dir / "nominal.csv", low_memory=False)
        telemetry = pd.read_csv(case_dir / "telemetry.csv", low_memory=False)
        backend = str(row.backend)
        effect_id = truth[str(row.case_id)]
        examples[backend].append((NORMAL_CLASS_ID, nominal))
        examples[backend].append((effect_id, telemetry))
        counts[NORMAL_CLASS_ID] += 1
        counts[effect_id] += 1
    return dict(examples), dict(counts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--development-truth", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--holdout-truth", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-version", default="0.6.5-catalog-model")
    args = parser.parse_args()

    development, development_counts = _examples(args.development, args.development_truth)
    holdout, holdout_counts = _examples(args.holdout, args.holdout_truth)
    combined: dict[str, list[tuple[str, pd.DataFrame]]] = defaultdict(list)
    for source in (development, holdout):
        for backend, rows in source.items():
            combined[backend].extend(rows)
    artifact = train_catalog_artifact(training_frames=dict(combined), model_version=args.model_version)
    save_catalog_artifact(artifact, args.model)
    model_sha = hashlib.sha256(args.model.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "astrograph-sat-sim-catalog-model-manifest.v1",
        "model_version": args.model_version,
        "model_file": args.model.name,
        "model_sha256": model_sha,
        "training_sources": [str(args.development), str(args.holdout)],
        "truth_sources_external": [str(args.development_truth), str(args.holdout_truth)],
        "development_class_counts": development_counts,
        "holdout_class_counts": holdout_counts,
        "backend_class_ids": {
            backend: sorted(set(label for label, _ in rows))
            for backend, rows in combined.items()
        },
        "inference_requires_paired_nominal": False,
        "inference_uses_truth": False,
        "score_semantics": artifact["score_semantics"],
        "claim_boundary": artifact["claim_boundary"],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "model": str(args.model), "sha256": model_sha}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
