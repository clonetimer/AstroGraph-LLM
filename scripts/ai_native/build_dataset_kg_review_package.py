#!/usr/bin/env python3
"""Build the governed 31-class dataset-to-KG human-review package."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from astrograph.knowledge.coverage_review import build_dataset_kg_review_package


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adcs-generator", type=Path, required=True)
    parser.add_argument("--phm-generator", type=Path, required=True)
    parser.add_argument("--adcs-dataset-root", type=Path, required=True)
    parser.add_argument("--phm-dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_dataset_kg_review_package(
        project_root=ROOT,
        adcs_generator_path=args.adcs_generator,
        phm_generator_path=args.phm_generator,
        adcs_dataset_root=args.adcs_dataset_root,
        phm_dataset_root=args.phm_dataset_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
