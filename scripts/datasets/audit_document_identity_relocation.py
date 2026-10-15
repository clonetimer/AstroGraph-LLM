#!/usr/bin/env python3
"""Audit candidate-KG identity after relocating the real early design DOCX."""
from __future__ import annotations

import json
from pathlib import Path

from astrograph.knowledge.document_identity import audit_document_identity_relocation
from astrograph.retrieval.document_governance import GovernedDocumentDescriptor


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent / "智能故障检测软件_设计报告.docx"
INVENTORY = ROOT / "governance/evidence/research/document_corpus/early_design_report_r001/machine_inventory.json"
OUTPUT = ROOT / "governance/evidence/research/document_corpus/document_identity_relocation_r001/audit.json"


def main() -> int:
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    descriptor = GovernedDocumentDescriptor.model_validate(inventory["descriptor"])
    audit = audit_document_identity_relocation(
        source_path=SOURCE,
        descriptor=descriptor,
        independence_group="early-design-report-origin",
        license_id="project-private-design-report",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
