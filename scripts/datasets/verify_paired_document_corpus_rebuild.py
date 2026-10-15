#!/usr/bin/env python3
"""Rebuild a paired document corpus in a fresh temporary directory and compare hashes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from astrograph.retrieval.document_corpus_builder import (  # noqa: E402
    verify_paired_document_corpus_rebuild,
)


TIMER_ROOT = REPO_ROOT.parents[1]
CORPUS = REPO_ROOT / ".astrograph-state/research/document_corpus/development_r002"
OUTPUT = (
    REPO_ROOT
    / "governance/evidence/research/document_corpus/development_r002/rebuild_verification.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=CORPUS / "manifest.json")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--temporary-parent", type=Path,
        default=REPO_ROOT / ".astrograph-state/research/document_corpus",
    )
    parser.add_argument("--corpus-revision", default="r002")
    args = parser.parse_args()
    receipt = verify_paired_document_corpus_rebuild(
        existing_manifest_path=args.manifest,
        datasets=[
            (
                "adcs-v522",
                TIMER_ROOT / "sat/data/generated/whole_spacecraft_adcs_paired_dataset_v5_2_2",
            ),
            (
                "phm-v60",
                TIMER_ROOT / "sat/data/generated/whole_spacecraft_phm_dataset_v6_0",
            ),
        ],
        policy_path=REPO_ROOT / "configs/research/document_corpus_access_policy_r001.json",
        generator_paths={
            "adcs-v522": TIMER_ROOT / "sat/scripts/generate_whole_spacecraft_adcs_eps_dataset_v522.py",
            "phm-v60": TIMER_ROOT / "sat/scripts/generate_whole_spacecraft_phm_dataset_v6.py",
        },
        temporary_parent=args.temporary_parent,
        corpus_revision=args.corpus_revision,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
