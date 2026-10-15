#!/usr/bin/env python3
"""Run the frozen P0.5 identifier/layout label-sidechannel attack."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from astrograph.retrieval.label_sidechannel import (  # noqa: E402
    run_document_label_sidechannel_attack,
)


TIMER_ROOT = REPO_ROOT.parents[1]
DEFAULT_CORPUS = REPO_ROOT / ".astrograph-state/research/document_corpus/development_r002"
DEFAULT_PREREG = REPO_ROOT / "configs/research/document_label_sidechannel_attack_r001.json"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "governance/evidence/research/document_corpus/label_sidechannel_r001/report_after_mitigation.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--adcs-index", type=Path,
        default=TIMER_ROOT / "sat/data/generated/whole_spacecraft_adcs_paired_dataset_v5_2_2/sample_index.csv",
    )
    parser.add_argument(
        "--phm-index", type=Path,
        default=TIMER_ROOT / "sat/data/generated/whole_spacecraft_phm_dataset_v6_0/sample_index.csv",
    )
    args = parser.parse_args()
    report = run_document_label_sidechannel_attack(
        preregistration_path=args.preregistration,
        manifest_path=args.corpus / "manifest.json",
        corpus_root=args.corpus,
        dataset_index_paths={"adcs-v522": args.adcs_index, "phm-v60": args.phm_index},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "status": report["status"],
        "report_sha256": report["report_sha256"],
    }, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
