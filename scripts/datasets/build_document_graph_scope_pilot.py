#!/usr/bin/env python3
"""Execute the frozen paired-corpus document graph scope pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from astrograph.agents.knowledge_curation import VllmKnowledgeExtractionBackend
from astrograph.knowledge.document_graph_scope_pilot import build_document_graph_scope_pilot


ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / ".astrograph-state/research/document_corpus/development_r002"
OUTPUT = ROOT / ".astrograph-state/research/document_corpus/document_graph_scope_pilot_r001"
CONFIG = ROOT / "configs/research/document_graph_scope_pilot_r001.json"
REFREEZE = ROOT / "governance/evidence/ai_native/kg_coverage_review_20260831/supplemental_review_refreeze_20260901.json"
RECEIPT = ROOT / "governance/evidence/research/document_corpus/document_graph_scope_pilot_r001/build_receipt.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    parser.add_argument("--review-refreeze", type=Path, default=REFREEZE)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    refreeze = json.loads(args.review_refreeze.read_text(encoding="utf-8"))["refreeze_sha256"]
    backend = VllmKnowledgeExtractionBackend(
        base_url=args.base_url, model_id=args.model, timeout_s=180, max_tokens=4096
    )
    backend.discover_model()
    result = build_document_graph_scope_pilot(
        config_path=args.config,
        manifest_path=args.corpus / "manifest.json",
        corpus_root=args.corpus,
        output_root=args.output,
        backend=backend,
        excluded_review_refreeze_sha256=refreeze,
        max_workers=args.workers,
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    compact = {key: value for key, value in result.items() if key != "results"}
    compact["backend_model_id"] = backend.model_id
    compact["backend_model_version"] = backend.model_version
    compact["local_receipt_path"] = str((args.output / "receipt.json").resolve())
    compact["local_receipt_file_sha256"] = hashlib.sha256(
        (args.output / "receipt.json").read_bytes()
    ).hexdigest()
    args.receipt.write_text(json.dumps(compact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
