#!/usr/bin/env python3
"""Verify the frozen dev13 PHM v6.0 retrospective screening bundle on a dataset root."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from astrograph.data_driven.phm_v60 import (  # noqa: E402
    audit_split,
    build_dataset,
    load_records,
    load_three_stage_promoted_bundle,
)

DEFAULT_BUNDLE = ROOT / "models" / "phm_v6_0" / "promoted_extra_trees_v1"
HIERARCHY = ROOT / "configs" / "datasets" / "phm_v6_0" / "diagnosis_hierarchy.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = load_records(args.dataset_root)
    split_audit = audit_split(records)
    if split_audit["status"] != "PASS":
        raise RuntimeError(split_audit)
    mapping = json.loads(HIERARCHY.read_text(encoding="utf-8"))["fault_to_component"]
    dataset = build_dataset(args.dataset_root, mapping, progress=True)
    result = load_three_stage_promoted_bundle(
        dataset, args.bundle_dir, dataset_root=args.dataset_root
    )
    result.pop("models", None)
    manifest = json.loads((args.bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    expected = manifest["fixed_split_metrics"]
    actual = {
        "detection_macro_f1": result["detection"]["macro_f1"],
        "localization_macro_f1": result["localization"]["macro_f1"],
        "classification_macro_f1": result["classification_predicted_component"]["macro_f1"],
        "end_to_end_18_class_macro_f1": result["end_to_end_18_class"]["macro_f1"],
    }
    metrics_match = all(abs(float(actual[key]) - float(expected[key])) <= 1e-12 for key in actual)
    report = {
        "schema_version": "astrograph-phm-v60-promoted-bundle-verification.v1",
        "status": "PASS" if metrics_match else "FAIL",
        "dataset_root": str(args.dataset_root.resolve()),
        "bundle_dir": str(args.bundle_dir.resolve()),
        "split_audit": split_audit,
        "expected_metrics": expected,
        "actual_metrics": actual,
        "metrics_match": metrics_match,
        "strategy": result["strategy"],
        "bundle": result["model_bundle"],
        "claim_boundary": result["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0 if metrics_match else 2


if __name__ == "__main__":
    raise SystemExit(main())
