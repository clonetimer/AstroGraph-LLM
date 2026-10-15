#!/usr/bin/env python3
"""Audit the generated paired document corpus without rebuilding it."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from astrograph.retrieval.document_corpus_builder import (  # noqa: E402
    audit_paired_document_corpus,
)


TIMER_ROOT = REPO_ROOT.parents[1]
DEFAULT_CORPUS = REPO_ROOT / ".astrograph-state/research/document_corpus/development_r002"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "governance/evidence/research/document_corpus/development_r002/audit_receipt.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    corpus_root = args.corpus_root.resolve()
    audit = audit_paired_document_corpus(
        manifest_path=corpus_root / "manifest.json",
        corpus_root=corpus_root,
        datasets={
            "adcs-v522": TIMER_ROOT / "sat/data/generated/whole_spacecraft_adcs_paired_dataset_v5_2_2",
            "phm-v60": TIMER_ROOT / "sat/data/generated/whole_spacecraft_phm_dataset_v6_0",
        },
        generator_paths={
            "adcs-v522": TIMER_ROOT / "sat/scripts/generate_whole_spacecraft_adcs_eps_dataset_v522.py",
            "phm-v60": TIMER_ROOT / "sat/scripts/generate_whole_spacecraft_phm_dataset_v6.py",
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False))
    return 0 if audit["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
