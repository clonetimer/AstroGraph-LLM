#!/usr/bin/env python3
"""Run a governed document pack through local-vLLM candidate graph extraction."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from astrograph.agents.knowledge_curation import VllmKnowledgeExtractionBackend
from astrograph.knowledge.document_graph_pack import build_document_pack_candidate_graphs


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = PROJECT_ROOT / ".astrograph-state/research/document_corpus/confusion_pack_r001"
DEFAULT_OUTPUT = PROJECT_ROOT / ".astrograph-state/research/document_corpus/confusion_candidate_graph_r001"
DEFAULT_RECEIPT = PROJECT_ROOT / "governance/evidence/research/document_corpus/confusion_candidate_graph_r001/build_receipt.json"
DEFAULT_REFREEZE = PROJECT_ROOT / "governance/evidence/ai_native/kg_coverage_review_20260831/supplemental_review_refreeze_20260901.json"


def _refreeze_sha(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = str(payload.get("refreeze_sha256", ""))
    if len(value) != 64:
        raise ValueError("review refreeze receipt has no valid refreeze_sha256")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--review-refreeze", type=Path, default=DEFAULT_REFREEZE)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--maximum-documents", type=int)
    parser.add_argument("--fault-id", action="append", dest="fault_ids")
    args = parser.parse_args()
    backend = VllmKnowledgeExtractionBackend(
        base_url=args.base_url, model_id=args.model, timeout_s=180.0, max_tokens=4096
    )
    backend.discover_model()
    result = build_document_pack_candidate_graphs(
        manifest_path=args.corpus / "manifest.json",
        corpus_root=args.corpus,
        output_root=args.output,
        backend=backend,
        excluded_review_refreeze_sha256=_refreeze_sha(args.review_refreeze),
        max_workers=args.workers,
        maximum_documents=args.maximum_documents,
        selected_fault_ids=args.fault_ids,
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    compact = {key: value for key, value in result.items() if key != "results"}
    compact["local_receipt_path"] = str((args.output / "receipt.json").resolve())
    compact["local_receipt_file_sha256"] = hashlib.sha256(
        (args.output / "receipt.json").read_bytes()
    ).hexdigest()
    args.receipt.write_text(
        json.dumps(compact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
