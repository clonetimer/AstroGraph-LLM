#!/usr/bin/env python3
"""Recompute every generated Runtime anomaly-screening document independently."""
from __future__ import annotations

import json
from pathlib import Path

from astrograph.retrieval.anomaly_screening_audit import audit_anomaly_screening_documents


ROOT = Path(__file__).resolve().parents[2]
TIMER = ROOT.parents[1]
OUTPUT = ROOT / "governance/evidence/research/document_corpus/anomaly_screening_recomputation_r001/audit.json"


def main() -> int:
    audit = audit_anomaly_screening_documents(
        config_path=ROOT / "configs/research/anomaly_screening_recomputation_r001.json",
        manifest_path=ROOT / ".astrograph-state/research/document_corpus/development_r002/manifest.json",
        corpus_root=ROOT / ".astrograph-state/research/document_corpus/development_r002",
        dataset_roots={
            "adcs-v522": TIMER / "sat/data/generated/whole_spacecraft_adcs_paired_dataset_v5_2_2",
            "phm-v60": TIMER / "sat/data/generated/whole_spacecraft_phm_dataset_v6_0",
        },
        production_builder_path=ROOT / "src/astrograph/retrieval/document_corpus_builder.py",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in audit.items() if key not in {"violations", "mismatch_examples"}},
                     ensure_ascii=False, indent=2))
    return 0 if audit["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
