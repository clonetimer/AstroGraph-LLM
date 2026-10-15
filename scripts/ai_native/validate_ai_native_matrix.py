#!/usr/bin/env python3
"""Validate the AI-Native maturity matrix and feedback baseline evidence.

Phase 0 deliverable: every score in the matrix must be bound to an existing
repository file, and both JSON documents must satisfy their frozen schema.

Usage:
    python scripts/ai_native/validate_ai_native_matrix.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "governance/evidence/ai_native/ai_native_maturity_matrix.json"
BASELINE = ROOT / "governance/evidence/ai_native/feedback_loop_baseline.json"
VALID_SCORES = {"S0", "S1", "S2", "S3"}


def _check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate_matrix(errors: list[str]) -> None:
    payload = json.loads(MATRIX.read_text(encoding="utf-8"))
    _check(payload.get("schema_version") == "astrograph-ai-native-maturity-matrix.v1",
           "matrix schema_version mismatch", errors)
    _check(payload.get("overall_level") in {"L1", "L2", "L1-L2", "L3", "L4", "L5"},
           "matrix overall_level invalid", errors)
    features = payload.get("features")
    _check(isinstance(features, list) and len(features) >= 6, "matrix features missing", errors)
    for feature in features or []:
        fid = feature.get("feature_id")
        _check(feature.get("score") in VALID_SCORES,
               f"{fid}: score not in {sorted(VALID_SCORES)}", errors)
        for item in feature.get("evidence", []):
            path = ROOT / item["path"]
            _check(path.is_file(), f"{fid}: evidence file missing: {item["path"]}", errors)
    l0 = payload.get("l0_l5_position", {})
    for dimension, level in l0.items():
        _check(str(level).startswith("L"), f"l0_l5_position.{dimension} invalid: {level}", errors)


def validate_baseline(errors: list[str]) -> None:
    payload = json.loads(BASELINE.read_text(encoding="utf-8"))
    _check(payload.get("schema_version") == "astrograph-ai-native-feedback-baseline.v1",
           "baseline schema_version mismatch", errors)
    _check(payload.get("feedback_loop_status") in {"ABSENT", "PARTIAL", "OPERATIONAL"},
           "baseline feedback_loop_status invalid", errors)
    for item in payload.get("evidence", []):
        path = ROOT / item["path"]
        _check(path.is_file(), f"baseline evidence file missing: {item["path"]}", errors)
    baseline_metrics = payload.get("baseline_metrics", {})
    _check("feedback_cycle_count" in baseline_metrics, "baseline feedback_cycle_count missing", errors)


def main() -> int:
    errors: list[str] = []
    if not MATRIX.is_file():
        print(f"[matrix] missing: {MATRIX}", file=sys.stderr)
        return 1
    if not BASELINE.is_file():
        print(f"[baseline] missing: {BASELINE}", file=sys.stderr)
        return 1
    validate_matrix(errors)
    validate_baseline(errors)
    if errors:
        for message in errors:
            print(f"[FAIL] {message}", file=sys.stderr)
        return 1
    print("[PASS] ai-native maturity matrix and feedback baseline are self-consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
