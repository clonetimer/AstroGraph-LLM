"""No-diagnosis preview of numbered user reports; does not require a model.

Example:
    python scripts/review_reported_observations.py --file examples/r5_south_solar_wing_observations.txt --output observation_preview.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from astrograph.retrieval.reported_observations import review_numbered_observations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', required=True, type=Path, help='UTF-8 text with numbered observations')
    parser.add_argument('--output', type=Path, help='Optional JSON output path')
    args = parser.parse_args()
    path = args.file.expanduser().resolve(strict=True)
    if not path.is_file() or path.stat().st_size > 64_000:
        parser.error('input must be a regular UTF-8 file of at most 64 KiB')
    review = review_numbered_observations(path.read_text(encoding='utf-8-sig'))
    result = json.dumps(review, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        target = args.output.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(result, encoding='utf-8')
        print(str(target))
    else:
        print(result, end='')


if __name__ == '__main__':
    main()
