#!/usr/bin/env python3
"""Audit P0.5 taxonomy and document-type coverage without opening post-hoc data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from astrograph.retrieval.document_coverage import audit_document_coverage  # noqa: E402


TIMER_ROOT = REPO_ROOT.parents[1]
DEFAULT_MANIFEST = (
    REPO_ROOT / ".astrograph-state/research/document_corpus/development_r002/manifest.json"
)
DEFAULT_FAULT_FACTS = (
    REPO_ROOT
    / ".astrograph-state/research/document_corpus/fault_pack_r001/canonical_fault_document_facts.json"
)
DEFAULT_NOMINAL_FACTS = (
    REPO_ROOT
    / ".astrograph-state/research/document_corpus/fault_pack_r001/canonical_nominal_document_facts.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "governance/evidence/research/document_corpus/coverage_r001/audit_receipt.json"
)
DEFAULT_PHM_SEVERITY = REPO_ROOT / "configs/research/phm_v6_severity_contract_r001.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--fault-facts", type=Path, default=DEFAULT_FAULT_FACTS)
    parser.add_argument("--nominal-facts", type=Path, default=DEFAULT_NOMINAL_FACTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phm-severity-contract", type=Path, default=DEFAULT_PHM_SEVERITY)
    parser.add_argument(
        "--adcs-index",
        type=Path,
        default=(
            TIMER_ROOT
            / "sat/data/generated/whole_spacecraft_adcs_paired_dataset_v5_2_2/sample_index.csv"
        ),
    )
    parser.add_argument(
        "--phm-index",
        type=Path,
        default=(
            TIMER_ROOT
            / "sat/data/generated/whole_spacecraft_phm_dataset_v6_0/sample_index.csv"
        ),
    )
    args = parser.parse_args()
    receipt = audit_document_coverage(
        manifest_path=args.manifest,
        dataset_index_paths={
            "adcs-v522": args.adcs_index,
            "phm-v60": args.phm_index,
        },
        static_dataset_ids={
            "adcs-v522": "adcs.v5_2_2",
            "phm-v60": "phm.v6_0",
        },
        fault_facts_path=args.fault_facts,
        nominal_facts_path=args.nominal_facts,
        severity_contract_paths={"phm-v60": args.phm_severity_contract},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "status": receipt["status"],
        "audit_sha256": receipt["audit_sha256"],
    }, ensure_ascii=False))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
