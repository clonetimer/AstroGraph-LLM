#!/usr/bin/env python3
"""Build the governed fault-confusion document pack through RAG and KG staging."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from astrograph.retrieval.confusion_document_builder import (
    build_fault_confusion_document_pack,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/research/fault_confusion_groups_r001.json"
DEFAULT_FACTS = (
    PROJECT_ROOT
    / ".astrograph-state/research/document_corpus/fault_pack_r001/canonical_fault_document_facts.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / ".astrograph-state/research/document_corpus/confusion_pack_r001"
)
DEFAULT_RECEIPT = (
    PROJECT_ROOT
    / "governance/evidence/research/document_corpus/confusion_pack_r001/build_receipt.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--facts", type=Path, default=DEFAULT_FACTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    manifest = build_fault_confusion_document_pack(
        config_path=args.config,
        fault_facts_path=args.facts,
        output_root=args.output,
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        key: value
        for key, value in manifest.items()
        if key not in {"documents"}
    }
    receipt["manifest_path"] = str((args.output.resolve() / "manifest.json"))
    receipt["manifest_file_sha256"] = __import__("hashlib").sha256(
        (args.output.resolve() / "manifest.json").read_bytes()
    ).hexdigest()
    args.receipt.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
