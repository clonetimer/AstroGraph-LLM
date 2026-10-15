import hashlib
import json

from astrograph.retrieval.vlm_transcription import build_vlm_scan_transcription_candidate


class _Backend:
    model_id = "fixture-vlm"
    def transcribe(self, *, image_data_url, prompt, max_tokens):
        assert image_data_url.startswith("data:image/png;base64,")
        return {"lines": ["Fixture title", "Table row"], "uncertain_spans": []}, {"id": "raw"}


def test_vlm_transcription_remains_unreviewed_and_non_admitted(tmp_path, monkeypatch) -> None:
    pdf, image = tmp_path / "source.pdf", tmp_path / "scan.png"
    pdf.write_bytes(b"pdf"); image.write_bytes(b"png")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "schema_version":"astrograph-vlm-scan-transcription-config.v1", "experiment_id":"fixture",
        "source_pdf_sha256":hashlib.sha256(b"pdf").hexdigest(),
        "scan_image_sha256":hashlib.sha256(b"png").hexdigest(), "source_page":1,
        "model_id":"fixture-vlm", "temperature":0, "max_tokens":100,
        "prompt":"transcribe", "admission":"MACHINE_TRANSCRIPTION_HUMAN_REVIEW_REQUIRED",
        "claim_boundary":"fixture only"}))
    class _Element:
        source_locator = "pdf#page:1:block:1"; text_representation = "Fixture title Table row"
    monkeypatch.setattr("astrograph.retrieval.vlm_transcription.SectionFirstDocumentIngestor.parse_pdf_layout",
                        lambda self, path: ([_Element()], [], object()))
    receipt = build_vlm_scan_transcription_candidate(
        config_path=config, source_pdf_path=pdf, scan_image_path=image,
        output_root=tmp_path / "out", backend=_Backend())
    assert receipt["representation_reviewed"] is False
    assert receipt["rag_admission_allowed"] is False
    assert receipt["kg_extraction_allowed"] is False
    assert receipt["normalized_character_similarity"] > 0.9
