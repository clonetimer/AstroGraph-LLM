#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from astrograph.assurance.archive_intake import IntakeStatus, scan_zip_archive


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only AstroGraph dataset ZIP intake scan.")
    parser.add_argument("archive")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--skip-crc", action="store_true")
    args = parser.parse_args()
    result = scan_zip_archive(args.archive, args.dataset_id, perform_crc_test=not args.skip_crc)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 1 if result.status is IntakeStatus.REJECT else 0


if __name__ == "__main__":
    raise SystemExit(main())
