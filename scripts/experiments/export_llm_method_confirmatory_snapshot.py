#!/usr/bin/env python3
"""Write and immediately verify the path-independent export manifest for a materialized confirmatory snapshot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from astrograph.evaluation.llm_method_campaign import (
    verify_confirmatory_snapshot_export_manifest,
    write_confirmatory_snapshot_export_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.snapshot_root.resolve()
    manifest = write_confirmatory_snapshot_export_manifest(root)
    verification = verify_confirmatory_snapshot_export_manifest(root)
    payload = {
        "schema_version": "astrograph-llm-method-confirmatory-snapshot-export-command.v1",
        "snapshot_root": str(root),
        "manifest": manifest,
        "verification": verification,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if verification.get("status") == "STABLE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
