"""Governed VLM transcription candidates for scanned document representations."""
from __future__ import annotations

import base64
import hashlib
import json
import re
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .structured_document import SectionFirstDocumentIngestor


class VlmPageTranscription(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    lines: list[str] = Field(min_length=1)
    uncertain_spans: list[str] = Field(default_factory=list)


class VisionTranscriptionBackend(Protocol):
    model_id: str
    def transcribe(self, *, image_data_url: str, prompt: str, max_tokens: int) -> tuple[dict[str, Any], dict[str, Any]]: ...


class VllmVisionTranscriptionBackend:
    def __init__(self, *, base_url: str, model_id: str, timeout_s: int = 180) -> None:
        self.base_url = base_url.rstrip("/"); self.model_id = model_id; self.timeout_s = timeout_s

    def transcribe(self, *, image_data_url: str, prompt: str, max_tokens: int) -> tuple[dict[str, Any], dict[str, Any]]:
        schema = VlmPageTranscription.model_json_schema()
        payload = {
            "model": self.model_id, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "VlmPageTranscription", "schema": schema, "strict": True,
            }},
        }
        request = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            raw = json.loads(response.read().decode("utf-8"))
        content = raw["choices"][0]["message"]["content"]
        return json.loads(content), raw


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[\w.+:/-]+", value.casefold(), flags=re.UNICODE))


def build_vlm_scan_transcription_candidate(
    *, config_path: str | Path, source_pdf_path: str | Path, scan_image_path: str | Path,
    output_root: str | Path, backend: VisionTranscriptionBackend,
) -> dict[str, Any]:
    config_path, source_pdf_path = Path(config_path).resolve(), Path(source_pdf_path).resolve()
    scan_image_path, output_root = Path(scan_image_path).resolve(), Path(output_root).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "astrograph-vlm-scan-transcription-config.v1":
        raise ValueError("unsupported VLM scan transcription config")
    if _sha(source_pdf_path) != config["source_pdf_sha256"] or _sha(scan_image_path) != config["scan_image_sha256"]:
        raise ValueError("VLM transcription input hash mismatch")
    if backend.model_id != config["model_id"]:
        raise ValueError("VLM transcription model differs from frozen config")
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("refusing to overwrite nonempty VLM transcription output")
    output_root.mkdir(parents=True, exist_ok=True)
    image_url = "data:image/png;base64," + base64.b64encode(scan_image_path.read_bytes()).decode("ascii")
    candidate_payload, raw = backend.transcribe(
        image_data_url=image_url, prompt=config["prompt"], max_tokens=int(config["max_tokens"])
    )
    candidate = VlmPageTranscription.model_validate(candidate_payload)
    elements, _, _ = SectionFirstDocumentIngestor().parse_pdf_layout(source_pdf_path)
    page_prefix = f"pdf#page:{int(config['source_page'])}:"
    reference = "\n".join(
        item.text_representation for item in elements if item.source_locator.startswith(page_prefix)
    )
    if not reference.strip():
        raise ValueError("native PDF machine reference page is empty")
    transcription = "\n".join(candidate.lines)
    normalized_reference, normalized_candidate = _normalize(reference), _normalize(transcription)
    reference_tokens = set(normalized_reference.split())
    candidate_tokens = set(normalized_candidate.split())
    candidate_body = {
        "schema_version": "astrograph-vlm-page-transcription-candidate.v1",
        "experiment_id": config["experiment_id"],
        "model_id": backend.model_id,
        "source_pdf_sha256": config["source_pdf_sha256"],
        "scan_image_sha256": config["scan_image_sha256"],
        "source_page": config["source_page"],
        "prompt_sha256": hashlib.sha256(config["prompt"].encode("utf-8")).hexdigest(),
        "transcription": candidate.model_dump(mode="json"),
        "representation_reviewed": False,
        "rag_admission_allowed": False,
        "kg_extraction_allowed": False,
    }
    candidate_doc = {**candidate_body, "candidate_sha256": _canonical_sha(candidate_body)}
    candidate_path, raw_path = output_root / "transcription_candidate.json", output_root / "raw_response.json"
    candidate_path.write_text(json.dumps(candidate_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    body = {
        "schema_version": "astrograph-vlm-scan-transcription-receipt.v1",
        "status": config["admission"],
        "configuration_sha256": _sha(config_path),
        "model_id": backend.model_id,
        "source_pdf_sha256": config["source_pdf_sha256"],
        "scan_image_sha256": config["scan_image_sha256"],
        "source_page": config["source_page"],
        "line_count": len(candidate.lines),
        "uncertain_span_count": len(candidate.uncertain_spans),
        "native_reference_text_sha256": hashlib.sha256(reference.encode("utf-8")).hexdigest(),
        "normalized_character_similarity": SequenceMatcher(
            None, normalized_reference, normalized_candidate, autojunk=False
        ).ratio(),
        "native_reference_token_coverage": (
            len(reference_tokens & candidate_tokens) / len(reference_tokens) if reference_tokens else 0.0
        ),
        "candidate_file_sha256": _sha(candidate_path),
        "raw_response_file_sha256": _sha(raw_path),
        "representation_reviewed": False,
        "rag_admission_allowed": False,
        "kg_extraction_allowed": False,
        "publication_allowed": False,
        "claim_boundary": config["claim_boundary"],
    }
    receipt = {**body, "receipt_sha256": _canonical_sha(body)}
    (output_root / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


__all__ = ["VlmPageTranscription", "VllmVisionTranscriptionBackend", "build_vlm_scan_transcription_candidate"]
