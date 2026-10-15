#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from astrograph.evaluation.calibration_campaign import (
    CalibrationContext,
    evaluate_calibration_transfer_campaign,
    evaluate_fixed_threshold_transfer,
)
from astrograph.evaluation.selective_risk import FrozenPrediction, TruthRecord


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline independent-group calibration-transfer audit; no calibration fitting.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--design", type=Path, default=Path("configs/evaluation/calibration_transfer_r18.v1.json"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold-receipt", type=Path)
    args = parser.parse_args()

    bundle = _load(args.input)
    design = _load(args.design)
    predictions = [FrozenPrediction(**row) for row in bundle["predictions"]]
    truths = [TruthRecord(**row) for row in bundle["truths"]]
    contexts = [CalibrationContext(**row) for row in bundle["contexts"]]
    report = evaluate_calibration_transfer_campaign(
        predictions,
        truths,
        contexts,
        required_dimensions=design["required_dimensions"],
        baseline_values=bundle["baseline_values"],
        required_partitions=design["required_partitions"],
        ece_bins=int(design["ece_bins"]),
        bootstrap_resamples=int(design["bootstrap_resamples"]),
        bootstrap_seed=int(design["bootstrap_seed"]),
    )
    output = {"calibration_transfer": report}
    if args.threshold_receipt:
        output["selective_threshold_transfer"] = evaluate_fixed_threshold_transfer(
            predictions,
            truths,
            contexts,
            threshold_receipt=_load(args.threshold_receipt),
            required_dimensions=design["required_dimensions"],
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
