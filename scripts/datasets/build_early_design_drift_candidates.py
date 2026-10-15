#!/usr/bin/env python3
"""Build traceable machine design-drift candidates for the early AstroGraph DOCX."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from astrograph.retrieval.design_drift import build_design_drift_candidates  # noqa: E402


SOURCE = REPO_ROOT.parent / "智能故障检测软件_设计报告.docx"
OUTPUT = (
    REPO_ROOT
    / "governance/evidence/research/document_corpus/early_design_report_r001/design_drift_candidates.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    artifact = build_design_drift_candidates(
        source_path=args.source, repo_root=REPO_ROOT
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "status": artifact["status"],
        "candidate_count": artifact["candidate_count"],
        "artifact_sha256": artifact["artifact_sha256"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
