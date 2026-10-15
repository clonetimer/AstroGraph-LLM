from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astrograph.integrations.platform_knowledge import UploadedDocumentKnowledgeIndex


DEFAULT_CASES = [
    {"query": "反作用轮故障诊断", "anchor": "反作用轮"},
    {"query": "互惠排序 RAG", "anchor": "互惠"},
    {"query": "IxRT 星载部署", "anchor": "IxRT"},
    {"query": "LoRA 微调", "anchor": "LoRA"},
    {"query": "异常信息摘要", "anchor": "异常信息"},
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate local uploaded-document retrieval on an explicitly supplied document.")
    parser.add_argument("--document", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    document = Path(args.document).expanduser().resolve()
    if not document.is_file():
        raise SystemExit("document not found")
    index = UploadedDocumentKnowledgeIndex(top_k=args.top_k)
    chunks, receipts = index.build([document])
    results = []
    reciprocal_ranks = []
    for case in DEFAULT_CASES:
        hits = index.retrieve(case["query"], chunks)
        rank = None
        for item in hits:
            if case["anchor"].casefold() in str(item.get("quote") or "").casefold():
                rank = int(item["rank"])
                break
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        results.append({**case, "anchor_rank": rank, "hit": rank is not None, "top_hits": [
            {"rank": item["rank"], "score": item["score"], "knowledge_id": item["knowledge_id"], "section_path": item["section_path"]}
            for item in hits
        ]})
    payload = {
        "schema_version": "astrograph-platform-knowledge-retrieval-benchmark.v1",
        "document": str(document),
        "document_ingestion": receipts,
        "case_count": len(results),
        "recall_at_k": sum(1 for item in results if item["hit"]) / len(results),
        "mrr_at_k": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "cases": results,
        "claim_boundary": "Curated lexical-retrieval sanity benchmark on one supplied design document; not diagnosis accuracy or semantic truth validation.",
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
