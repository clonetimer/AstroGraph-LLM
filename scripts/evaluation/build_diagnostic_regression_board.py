#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from astrograph.evaluation.diagnostic_regression import BenchmarkSnapshot, build_regression_board

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "evaluation" / "r36_diagnostic_benchmarks.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the R36 diagnostic regression board")
    parser.add_argument("--version", required=True)
    parser.add_argument("--snapshot", type=Path, action="append", default=[])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    snapshots = [
        BenchmarkSnapshot.model_validate_json(path.read_text(encoding="utf-8"))
        for path in args.snapshot
    ]
    board = build_regression_board(
        astrograph_version=args.version,
        benchmark_ids=config["benchmark_ids"],
        snapshots=snapshots,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(board.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
