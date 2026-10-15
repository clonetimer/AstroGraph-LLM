import shutil

import pytest

from astrograph.retrieval.document_pack_builder import _write_pdf
from astrograph.retrieval.pdf_stress_builder import build_pdf_stress_pack


@pytest.mark.skipif(
    shutil.which("pdftoppm") is None or shutil.which("magick") is None,
    reason="PDF pressure tools unavailable",
)
def test_scanned_pdf_pressure_sample_is_deterministic_and_fails_closed(tmp_path):
    source = tmp_path / "source.pdf"
    _write_pdf(source, "# Design\n\nObservable current changed after command.\n")
    first = build_pdf_stress_pack(source_pdf=source, output_root=tmp_path / "first")
    second = build_pdf_stress_pack(source_pdf=source, output_root=tmp_path / "second")

    assert first["parser_outcome"] == "EXPECTED_FAIL_CLOSED_OCR_REQUIRED"
    assert first["artifacts"] == second["artifacts"]
    assert (tmp_path / "first/rotated_noisy_scan.pdf").read_bytes().startswith(b"%PDF-")
    assert "OCR_REQUIRED" in (tmp_path / "first/ocr_review_requirement.txt").read_text(encoding="utf-8")
