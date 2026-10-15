#!/usr/bin/env python3
"""Verify every archived/copied byte in a materialized confirmatory snapshot tree."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from astrograph.evaluation.llm_method_campaign import verify_confirmatory_snapshot_export_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-root", type=Path, required=True)
    args = parser.parse_args()
    result = verify_confirmatory_snapshot_export_manifest(args.snapshot_root.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "STABLE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
