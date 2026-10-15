#!/usr/bin/env python3
"""Generate redacted structural evidence for a local ESA-ADB archive.

Parameters:
  --root DATASET_ROOT
  --output EVIDENCE_DIR
  [--profile SCHEMA_PROFILE.json]
  [--timestamp-column NAME]
  [--label-columns comma,separated,names]
  [--max-entities N] [--max-rows N] [--trust-local-pickle]
  [--hash-mode none|metadata|full] [--trust-local-pickle]
  [--include-statistics] [--include-label-values]

The script is read-only. It does not copy raw rows, does not use the network,
and refuses to place its evidence directory inside the dataset root.
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
    ProbeOptions,
    build_data_source,
    probe_data_source,
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
    parser.add_argument("--hash-mode", choices=("none", "metadata", "full"), default="metadata")
    parser.add_argument("--include-statistics", action="store_true")
    parser.add_argument("--include-label-values", action="store_true")
    args = parser.parse_args()

    labels = [item.strip() for item in args.label_columns.split(",") if item.strip()]
    config = DatasetSourceConfig(
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
    artifacts = probe_data_source(
        build_data_source(config),
        args.output,
        options=ProbeOptions(
            max_entities=args.max_entities,
            max_rows_per_entity=args.max_rows,
            hash_mode=args.hash_mode,
            include_statistics=args.include_statistics,
            include_label_values=args.include_label_values,
            root_token="${ASTROGRAPH_ESA_ADB_ROOT}",
        ),
    )
    print(json.dumps(artifacts.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
