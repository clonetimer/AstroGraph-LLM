#!/usr/bin/env python3
"""Build the frozen document-to-graph extraction ablation report."""
from __future__ import annotations

import json
from pathlib import Path

from astrograph.knowledge.document_graph_ablation import build_document_graph_ablation_report


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    report = build_document_graph_ablation_report(
        config_path=ROOT / "configs/research/document_graph_ablation_r001.json",
        project_root=ROOT,
        output_path=ROOT
        / "governance/evidence/research/document_corpus/document_graph_ablation_r001/report.json",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
