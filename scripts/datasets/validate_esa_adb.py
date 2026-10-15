#!/usr/bin/env python3
"""Run bounded, read-only structural validation against a local ESA-ADB archive.

Parameters:
  --root DATASET_ROOT
  --output validation_report.json
  [--profile schema_profile.json]
  [--timestamp-column NAME]
  [--label-columns comma,separated,names]
  [--max-entities N] [--max-rows N] [--trust-local-pickle]

Exit code is 0 for PASS/WARNING and 2 for FAIL. No raw rows are persisted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from astrograph.data_sources import (  # noqa: E402
    DataSourceType,
    DatasetSourceConfig,
    build_data_source,
    validate_data_source,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=None)
    parser.add_argument("--timestamp-column", default="")
    parser.add_argument("--label-columns", default="")
    parser.add_argument(
        "--trust-local-pickle",
        action="store_true",
        help=(
            "Acknowledge that raw ESA-ADB ZIP archives contain trusted pandas "
            "pickle objects that may execute code while loading"
        ),
    )
    parser.add_argument("--max-entities", type=int, default=100)
    parser.add_argument("--max-rows", type=int, default=10_000)
    args = parser.parse_args()

    labels = [item.strip() for item in args.label_columns.split(",") if item.strip()]
    source = build_data_source(
        DatasetSourceConfig(
            source_type=DataSourceType.ESA_ADB,
            root=str(args.root),
            read_only=True,
            lazy=True,
            recursive=True,
            timestamp_column=args.timestamp_column or None,
            label_columns=labels,
            profile_path=None if args.profile is None else str(args.profile),
        metadata={"trusted_pickle": bool(args.trust_local_pickle)},
        )
    )
    report = validate_data_source(
        source,
        max_entities=args.max_entities,
        max_rows_per_entity=args.max_rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 2 if report.status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
