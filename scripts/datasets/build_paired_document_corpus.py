#!/usr/bin/env python3
"""Build AstroGraph's train/development numeric--document paired corpus."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from astrograph.retrieval.document_corpus_builder import (  # noqa: E402
    build_paired_document_corpus,
)


TIMER_ROOT = REPO_ROOT.parents[1]
DEFAULT_ADCS = (
    TIMER_ROOT
    / "sat/data/generated/whole_spacecraft_adcs_paired_dataset_v5_2_2"
)
DEFAULT_PHM = (
    TIMER_ROOT
    / "sat/data/generated/whole_spacecraft_phm_dataset_v6_0"
)
DEFAULT_ADCS_GENERATOR = (
    TIMER_ROOT / "sat/scripts/generate_whole_spacecraft_adcs_eps_dataset_v522.py"
)
DEFAULT_PHM_GENERATOR = (
    TIMER_ROOT / "sat/scripts/generate_whole_spacecraft_phm_dataset_v6.py"
)
DEFAULT_POLICY = REPO_ROOT / "configs/research/document_corpus_access_policy_r001.json"
DEFAULT_OUTPUT = (
    REPO_ROOT / ".astrograph-state/research/document_corpus/development_r002"
)
DEFAULT_RECEIPT = (
    REPO_ROOT
    / "governance/evidence/research/document_corpus/development_r002/build_receipt.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adcs-root", type=Path, default=DEFAULT_ADCS)
    parser.add_argument("--phm-root", type=Path, default=DEFAULT_PHM)
    parser.add_argument("--adcs-generator", type=Path, default=DEFAULT_ADCS_GENERATOR)
    parser.add_argument("--phm-generator", type=Path, default=DEFAULT_PHM_GENERATOR)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--limit-per-dataset", type=int)
    parser.add_argument("--corpus-revision", default="r002")
    args = parser.parse_args()
    if args.limit_per_dataset is not None and args.limit_per_dataset < 1:
        parser.error("--limit-per-dataset must be positive")

    _, receipt = build_paired_document_corpus(
        datasets=[
            ("adcs-v522", args.adcs_root.resolve()),
            ("phm-v60", args.phm_root.resolve()),
        ],
        output_root=args.output.resolve(),
        policy_path=args.policy.resolve(),
        generator_paths={
            "adcs-v522": args.adcs_generator.resolve(),
            "phm-v60": args.phm_generator.resolve(),
        },
        include_development=not args.train_only,
        limit_per_dataset=args.limit_per_dataset,
        corpus_revision=args.corpus_revision,
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
