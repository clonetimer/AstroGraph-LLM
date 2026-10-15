#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from astrograph.evaluation.r31_pump_blind_campaign import (
    evaluate_frozen_predictions,
    generate_blind_cases,
    predict_cases_truth_free,
)

p = argparse.ArgumentParser()
p.add_argument("--output-dir", required=True)
p.add_argument("--seed", type=int, default=3101)
p.add_argument("--repetitions-per-mode", type=int, default=40)
a = p.parse_args()
out = Path(a.output_dir)
out.mkdir(parents=True, exist_ok=True)

cases = generate_blind_cases(seed=a.seed, repetitions_per_mode=a.repetitions_per_mode)
predictions = predict_cases_truth_free(cases)
# Freeze truth-free predictions before serializing hidden mode truth.
pred_doc = {
    "schema_version": "astrograph-r31-pump-blind-predictions.v1",
    "seed": a.seed,
    "repetitions_per_mode": a.repetitions_per_mode,
    "predictions": [item.model_dump(mode="json") for item in predictions],
}
(out / "predictions_before_truth_join.json").write_text(
    json.dumps(pred_doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

report = evaluate_frozen_predictions(cases=cases, predictions=predictions, seed=a.seed)
truth_doc = {
    "schema_version": "astrograph-r31-pump-blind-truth.v1",
    "seed": a.seed,
    "cases": [
        {"case_id": item.case_id, "hidden_fault_id": item.hidden_fault_id, "case_sha256": item.case_sha256}
        for item in cases
    ],
}
(out / "truth_after_prediction_freeze.json").write_text(
    json.dumps(truth_doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
(out / "report.json").write_text(
    json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, sort_keys=True))
