#!/usr/bin/env python3
"""Build and fail-closed freeze the current VLM transcription review handoff."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from astrograph.retrieval.vlm_transcription_review import (
    build_vlm_transcription_review_handoff, freeze_vlm_transcription_review,
)


ROOT = Path(__file__).resolve().parents[2]
TRANSCRIPTION = ROOT / ".astrograph-state/research/document_corpus/vlm_scan_transcription_r001"
SOURCE = ROOT / ".astrograph-state/research/document_corpus/fault_pack_r001/fault_design_fmea_test_handbook.pdf"
HANDOFF = ROOT / ".astrograph-state/research/document_corpus/vlm_scan_transcription_review_r001"
FREEZE = ROOT / ".astrograph-state/research/document_corpus/vlm_scan_transcription_review_freeze_r001"
TRACKED = ROOT / "governance/evidence/research/document_corpus/vlm_scan_transcription_review_r001/build_receipt.json"
TRACKED_FREEZE = ROOT / "governance/evidence/research/document_corpus/vlm_scan_transcription_review_r001/decision_freeze.json"


def main() -> int:
    handoff = build_vlm_transcription_review_handoff(
        transcription_root=TRANSCRIPTION, source_pdf_path=SOURCE, output_root=HANDOFF)
    compact = {**handoff, "local_review_sheet_path": str((HANDOFF / "human_review.csv").resolve()),
               "local_review_sheet_file_sha256": hashlib.sha256((HANDOFF / "human_review.csv").read_bytes()).hexdigest()}
    TRACKED.parent.mkdir(parents=True, exist_ok=True)
    TRACKED.write_text(json.dumps(compact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    frozen = freeze_vlm_transcription_review(
        transcription_root=TRANSCRIPTION, source_pdf_path=SOURCE,
        review_sheet_path=HANDOFF / "human_review.csv", output_root=FREEZE)
    TRACKED_FREEZE.write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"handoff": compact, "freeze": frozen}, ensure_ascii=False, indent=2))
    return 0 if frozen["status"] != "BLOCKED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
