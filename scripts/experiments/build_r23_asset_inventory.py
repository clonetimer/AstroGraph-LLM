from __future__ import annotations

import argparse
import json
from pathlib import Path

from astrograph.evaluation.r23_assets import build_zip_asset_inventory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", action="append", nargs=3, metavar=("ASSET_ID", "ROLE", "ZIP"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inventories = [
        build_zip_asset_inventory(Path(path), asset_id=asset_id, role=role).model_dump(mode="json")
        for asset_id, role, path in args.asset
    ]
    payload = {
        "schema_version": "astrograph-r23-source-asset-inventory-bundle.v1",
        "assets": inventories,
        "selection_policy": "SCHEMA_AND_CONTENT_IDENTITY_ONLY_NO_EMBEDDED_EVALUATION_REUSE",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
