#!/usr/bin/env python3
"""Offline human scoring for an already completed R11-F blind campaign.

This script never imports/configures an LLM backend and never reruns inference.
Human assessments must not be stored inside model input manifests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from astrograph.evaluation.material_llm_blind_campaign import score_blind_material_reviews


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--ratings", required=True, type=Path,
                        help="JSON object with schema_version and reviews array, entered after inference.")
    args = parser.parse_args()
    root = args.campaign_root.resolve(strict=True)
    outcome_path = root / "outcomes.json"
    manifest_path = root / "manifest.json"
    outcomes = json.loads(outcome_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if outcomes.get("manifest_sha256") != digest:
        raise ValueError("BLIND_CAMPAIGN_MANIFEST_CHANGED")
    if (manifest.get("schema_version") != "astrograph-material-llm-blind-campaign.v1"
            or outcomes.get("schema_version") != "astrograph-material-llm-blind-outcomes.v1"):
        raise ValueError("INVALID_BLIND_CAMPAIGN_SCHEMAS")
    score_path = root / "human_scores.json"
    if score_path.exists():
        raise FileExistsError("BLIND_HUMAN_SCORES_ALREADY_PRESENT")
    ratings_path = args.ratings.resolve(strict=True)
    if ratings_path.is_relative_to(root):
        raise ValueError("BLIND_RATINGS_MUST_BE_SEPARATE_FROM_LLM_RUN_ARTIFACTS")
    payload = json.loads(ratings_path.read_text(encoding="utf-8"))
    if (set(payload) != {"schema_version", "reviews"} or
            payload["schema_version"] != "astrograph-material-llm-human-ratings.v1" or
            not isinstance(payload["reviews"], list)):
        raise ValueError("BLIND_HUMAN_RATINGS_SCHEMA_INVALID")
    scores = score_blind_material_reviews(outcomes, payload["reviews"])
    scores["campaign_manifest_sha256"] = digest
    scores["outcomes_sha256"] = hashlib.sha256(outcome_path.read_bytes()).hexdigest()
    temporary = score_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(scores, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(score_path)
    print(json.dumps({"status": "HUMAN_SCORES_RECORDED",
                      "rated_cells": scores["rated_cells"],
                      "diagnostic_accuracy_gain": None,
                      "scores": str(score_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
