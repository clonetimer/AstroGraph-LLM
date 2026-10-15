#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from astrograph.evaluation.diagnostic_regression import BenchmarkSnapshot, compare_snapshots


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two provenance-compatible R36 benchmark snapshots")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    baseline = BenchmarkSnapshot.model_validate_json(args.baseline.read_text(encoding="utf-8"))
    candidate = BenchmarkSnapshot.model_validate_json(args.candidate.read_text(encoding="utf-8"))
    comparison = compare_snapshots(baseline, candidate)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(comparison.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return 0 if comparison.comparable else 2


if __name__ == "__main__":
    raise SystemExit(main())
