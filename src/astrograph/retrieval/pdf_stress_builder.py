"""Build deterministic scan/rotation pressure artifacts for PDF ingestion."""
from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .document_corpus_builder import sha256_file
from .structured_document import SectionFirstDocumentIngestor


def _tool_version(command: str, argument: str) -> str:
    result = subprocess.run(
        [command, argument], capture_output=True, text=True, timeout=10, check=False
    )
    text = (result.stdout + result.stderr).strip().splitlines()
    return text[0][:300] if text else "UNKNOWN"


def _normalize_pdf_metadata(path: Path) -> None:
    data = path.read_bytes()
    replacements = [
        (rb"D:\d{14}[+-]\d{2}'\d{2}'", b"D:20000101000000+00'00'"),
        (rb"D:\d{14}Z", b"D:20000101000000Z"),
        (rb"D:\d{14}", b"D:20000101000000"),
    ]
    for pattern, replacement in replacements:
        data = re.sub(pattern, replacement, data)
    path.write_bytes(data)


def build_pdf_stress_pack(
    *,
    source_pdf: Path,
    output_root: Path,
    seed: int = 20260905,
) -> dict[str, object]:
    pdftoppm = shutil.which("pdftoppm")
    magick = shutil.which("magick")
    if pdftoppm is None or magick is None:
        raise RuntimeError("pdftoppm and ImageMagick are required for PDF stress generation")
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"refusing to overwrite nonempty stress directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="astrograph-pdf-stress-") as temp:
        temp_root = Path(temp)
        raster_prefix = temp_root / "source_page"
        subprocess.run(
            [
                pdftoppm,
                "-f", "1", "-singlefile", "-png", "-r", "120",
                str(source_pdf), str(raster_prefix),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        noisy_png = output_root / "rotated_noisy_scan.png"
        subprocess.run(
            [
                magick,
                "-seed", str(seed), str(raster_prefix.with_suffix(".png")),
                "-background", "white", "-rotate", "1.2",
                "-attenuate", "0.08", "+noise", "Gaussian",
                "-colorspace", "Gray", "-strip", str(noisy_png),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        scanned_pdf = output_root / "rotated_noisy_scan.pdf"
        subprocess.run(
            [magick, str(noisy_png), "-strip", str(scanned_pdf)],
            check=True,
            capture_output=True,
            timeout=30,
        )
        _normalize_pdf_metadata(scanned_pdf)
    parser_outcome = "UNEXPECTED_TEXT_LAYER_PRESENT"
    parser_message = ""
    try:
        SectionFirstDocumentIngestor().parse_pdf_layout(scanned_pdf)
    except ValueError as exc:
        parser_message = str(exc)
        if "OCR output must be supplied" in parser_message:
            parser_outcome = "EXPECTED_FAIL_CLOSED_OCR_REQUIRED"
    if parser_outcome != "EXPECTED_FAIL_CLOSED_OCR_REQUIRED":
        raise ValueError(f"scan pressure document did not fail closed: {parser_outcome}")
    sidecar = output_root / "ocr_review_requirement.txt"
    sidecar.write_text(
        "OCR_REQUIRED\n"
        "No OCR engine is bundled in this environment. Supply OCR output as a separate, "
        "hash-bound representation and complete human region/text review before retrieval or KG extraction.\n",
        encoding="utf-8",
    )
    return {
        "schema_version": "astrograph-pdf-stress-pack-receipt.v1",
        "status": "PRESSURE_SAMPLE_READY_OCR_GOLD_PENDING",
        "source_pdf_sha256": sha256_file(source_pdf),
        "seed": seed,
        "transform": {
            "page": 1,
            "raster_dpi": 120,
            "rotation_deg": 1.2,
            "gaussian_noise_attenuation": 0.08,
            "colorspace": "Gray",
        },
        "tool_versions": {
            "pdftoppm": _tool_version(pdftoppm, "-v"),
            "magick": _tool_version(magick, "-version"),
        },
        "artifacts": {
            noisy_png.name: sha256_file(noisy_png),
            scanned_pdf.name: sha256_file(scanned_pdf),
            sidecar.name: sha256_file(sidecar),
        },
        "parser_outcome": parser_outcome,
        "parser_message": parser_message,
        "claim_boundary": (
            "Deterministic layout-pressure artifact only. No OCR quality, text recovery, "
            "retrieval performance, or document realism claim is made until human gold exists."
        ),
    }


__all__ = ["build_pdf_stress_pack"]
