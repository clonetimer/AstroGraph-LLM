#!/usr/bin/env python3
"""Build the governed Qwen3.5 scan transcription candidate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from astrograph.retrieval.vlm_transcription import (
    VllmVisionTranscriptionBackend, build_vlm_scan_transcription_candidate,
)


ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / ".astrograph-state/research/document_corpus/vlm_scan_transcription_r001"
TRACKED = ROOT / "governance/evidence/research/document_corpus/vlm_scan_transcription_r001/build_receipt.json"


def main() -> int:
    receipt = build_vlm_scan_transcription_candidate(
        config_path=ROOT / "configs/research/vlm_scan_transcription_r001.json",
        source_pdf_path=ROOT / ".astrograph-state/research/document_corpus/fault_pack_r001/fault_design_fmea_test_handbook.pdf",
        scan_image_path=ROOT / ".astrograph-state/research/document_corpus/pdf_stress_r001/rotated_noisy_scan.png",
        output_root=LOCAL,
        backend=VllmVisionTranscriptionBackend(base_url="http://127.0.0.1:8000", model_id="Qwen3.5-9B"),
    )
    compact = {**receipt, "local_receipt_path": str((LOCAL / "receipt.json").resolve()),
               "local_receipt_file_sha256": hashlib.sha256((LOCAL / "receipt.json").read_bytes()).hexdigest()}
    TRACKED.parent.mkdir(parents=True, exist_ok=True)
    TRACKED.write_text(json.dumps(compact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
