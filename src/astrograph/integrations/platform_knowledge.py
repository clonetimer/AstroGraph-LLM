"""Session-scoped knowledge enrichment for the upload-to-report platform.

This module is deliberately *post diagnosis*.  Uploaded documents, PaperSeek
results and Anything2Ontology outputs may explain or route reviewed knowledge,
but they never enter candidate generation, BiRRF or Decision Governor inputs.
The frozen diagnosis snapshot is the authority for the selected candidate.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from astrograph.llm import OpenAICompatibleConfig, OpenAICompatibleStructuredBackend
from astrograph.retrieval.reported_observations import review_numbered_observations
from astrograph.retrieval.observation_balanced import retrieve_observation_balanced
from astrograph.integrations.process_control import run_managed_subprocess
from astrograph.oag.paperseek_http import PaperSeekConfig, PaperSeekHTTPClient
from astrograph.oag.paperseek_cli import PaperSeekCLIConfig, PaperSeekCLIClient
from astrograph.retrieval.structured_document import SectionFirstDocumentIngestor
from astrograph.tools.material_review import (
    summarize_material_causal_review, execute_material_tool_plan,
)


PLATFORM_KNOWLEDGE_SCHEMA_VERSION = "astrograph-platform-knowledge-enrichment.v1"
_DOCUMENT_SUFFIXES = {".txt", ".md", ".docx", ".pdf"}
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9_+.-]*", re.I)
_CJK_SEQUENCE_RE = re.compile(r"[\u4e00-\u9fff]+")
_MAX_QUERY_CHARS = 3000
_MAX_LOCAL_HITS = 12
_MAX_CHUNK_CHARS_FOR_MODEL = 6000
_MAX_COMMAND_OUTPUT_CHARS = 12000
_MAX_ONTOLOGY_CONTEXT_CHARS = 24000
_MAX_ONTOLOGY_FILE_CHARS = 8000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tokens(text: str) -> list[str]:
    """Tokenize mixed English/Chinese technical text without an external segmenter.

    Latin identifiers are preserved as whole tokens.  CJK text contributes the
    contiguous phrase (for short phrases), characters and adjacent bigrams.
    The extra bigrams reduce the false overlap caused by single-character-only
    matching while keeping the local upload path dependency-free.
    """
    tokens = [item.casefold() for item in _WORD_RE.findall(text)]
    for sequence in _CJK_SEQUENCE_RE.findall(text):
        if len(sequence) <= 24:
            tokens.append(sequence)
        tokens.extend(sequence)
        tokens.extend(sequence[index:index + 2] for index in range(max(0, len(sequence) - 1)))
    return tokens


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    maximum, minimum = max(values), min(values)
    if maximum <= minimum:
        return [1.0 if maximum > 0.0 else 0.0 for _ in values]
    return [(value - minimum) / (maximum - minimum) for value in values]


def _bm25(query_terms: list[str], documents: list[list[str]]) -> list[float]:
    if not query_terms or not documents:
        return [0.0 for _ in documents]
    average_length = sum(len(document) for document in documents) / len(documents)
    document_frequency = Counter(term for document in documents for term in set(document))
    scores: list[float] = []
    for document in documents:
        frequencies = Counter(document)
        score = 0.0
        for term in sorted(set(query_terms)):
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            df = document_frequency[term]
            inverse = math.log(1.0 + (len(documents) - df + 0.5) / (df + 0.5))
            denominator = frequency + 1.5 * (
                1.0 - 0.75 + 0.75 * len(document) / max(1.0, average_length)
            )
            score += inverse * frequency * 2.5 / max(denominator, 1e-12)
        scores.append(score)
    return scores


class PlatformKnowledgeClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["OBSERVATION", "LOCAL_KNOWLEDGE", "LITERATURE", "ONTOLOGY_CANDIDATE", "HYPOTHESIS", "MATERIAL_LOG_CUE", "MATERIAL_REFERENCE", "GRAPH_ASSOCIATION", "CONFIRMED_CASE"]
    text: str = Field(min_length=1, max_length=4000)
    citation_ids: list[str] = Field(default_factory=list)

    @field_validator("citation_ids")
    @classmethod
    def unique_citations(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("claim citation IDs must be unique")
        return value


class PlatformKnowledgeExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    candidate_fault_id: str | None = None
    abstained: bool
    summary: str = Field(min_length=1, max_length=6000)
    claims: list[PlatformKnowledgeClaim] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    recommended_review_actions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def abstention_matches_candidate(self):
        if self.abstained != (self.candidate_fault_id is None):
            raise ValueError("abstention must exactly correspond to candidate_fault_id=None")
        return self


class PlatformKnowledgeConfig(BaseModel):
    """Administrator-owned service configuration; users do not supply endpoints."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    local_top_k: int = Field(default=5, ge=1, le=_MAX_LOCAL_HITS)
    paperseek_url: str | None = None
    paperseek_source_root: str | None = None
    paperseek_data_source: str = "openalex"
    paperseek_top_k: int = Field(default=5, ge=1, le=20)
    paperseek_cli_timeout_s: float = Field(default=180.0, gt=0, le=3600)
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_provider: str = "local-openai-compatible"
    llm_timeout_s: float = Field(default=120.0, gt=0, le=3600)
    llm_response_format_strategy: Literal["auto", "json_schema", "json_object", "plain_json"] = "auto"
    llm_max_tokens: int = Field(default=1200, ge=64, le=8192)
    a2o_enabled: bool = False
    source_tool_url: str | None = None
    source_tool_adapter_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    a2o_source_root: str | None = None
    a2o_python_executable: str | None = None
    a2o_timeout_s: float = Field(default=900.0, gt=0, le=7200)
    require_configured_services: bool = False
    material_tool_calls_enabled: bool = False  # Explicit opt-in: an additional bounded LLM selection call.

    @classmethod
    def from_environment(cls) -> "PlatformKnowledgeConfig":
        def yes(name: str, default: str = "0") -> bool:
            return os.environ.get(name, default).strip().casefold() in {"1", "true", "yes", "on"}

        def number(name: str, default: str) -> float:
            return float(os.environ.get(name, default))

        return cls(
            local_top_k=int(os.environ.get("ASTROGRAPH_PLATFORM_LOCAL_TOP_K", "5")),
            paperseek_url=os.environ.get("ASTROGRAPH_PLATFORM_PAPERSEEK_URL") or None,
            paperseek_source_root=os.environ.get("ASTROGRAPH_PLATFORM_PAPERSEEK_SOURCE_ROOT") or None,
            paperseek_data_source=os.environ.get("ASTROGRAPH_PLATFORM_PAPERSEEK_SOURCE", "openalex"),
            paperseek_top_k=int(os.environ.get("ASTROGRAPH_PLATFORM_PAPERSEEK_TOP_K", "5")),
            paperseek_cli_timeout_s=number("ASTROGRAPH_PLATFORM_PAPERSEEK_CLI_TIMEOUT_S", "180"),
            llm_base_url=os.environ.get("ASTROGRAPH_PLATFORM_LLM_BASE_URL") or None,
            llm_model=os.environ.get("ASTROGRAPH_PLATFORM_LLM_MODEL") or None,
            llm_provider=os.environ.get("ASTROGRAPH_PLATFORM_LLM_PROVIDER", "local-openai-compatible"),
            llm_response_format_strategy=os.environ.get("ASTROGRAPH_PLATFORM_LLM_RESPONSE_FORMAT_STRATEGY", "auto"),
            llm_max_tokens=int(os.environ.get("ASTROGRAPH_PLATFORM_LLM_MAX_TOKENS", "1200")),
            llm_timeout_s=number("ASTROGRAPH_PLATFORM_LLM_TIMEOUT_S", "120"),
            a2o_enabled=yes("ASTROGRAPH_PLATFORM_A2O_ENABLED"),
            source_tool_url=os.environ.get("ASTROGRAPH_PLATFORM_SOURCE_TOOL_URL") or None,
            source_tool_adapter_sha256=os.environ.get("ASTROGRAPH_PLATFORM_SOURCE_TOOL_ADAPTER_SHA256") or None,
            a2o_source_root=os.environ.get("ASTROGRAPH_PLATFORM_A2O_SOURCE_ROOT") or None,
            a2o_python_executable=os.environ.get("ASTROGRAPH_PLATFORM_A2O_PYTHON") or None,
            a2o_timeout_s=number("ASTROGRAPH_PLATFORM_A2O_TIMEOUT_S", "900"),
            require_configured_services=yes("ASTROGRAPH_PLATFORM_REQUIRE_KNOWLEDGE_SERVICES"),
            material_tool_calls_enabled=yes("ASTROGRAPH_PLATFORM_MATERIAL_TOOL_CALLS_ENABLED"),
        )


@dataclass(frozen=True)
class _LocalChunk:
    chunk_id: str
    source_id: str
    filename: str
    source_sha256: str
    text: str
    section_path: tuple[str, ...]
    source_locators: tuple[str, ...]
    review_status: str
    content_safety_status: str
    modalities: tuple[str, ...] = ()
    table_header_context: str = ""
    derived_table_window: bool = False


class UploadedDocumentKnowledgeIndex:
    """Native, session-only parser/index for user-supplied explanatory material."""

    def __init__(self, *, top_k: int = 5) -> None:
        if top_k < 1 or top_k > _MAX_LOCAL_HITS:
            raise ValueError("invalid local knowledge top_k")
        self.top_k = top_k
        self.ingestor = SectionFirstDocumentIngestor(
            maximum_section_units=18, maximum_chunk_chars=1800,
            preserve_modality_boundaries=True, split_long_text=True,
        )

    def _ingest_one(self, path: Path) -> tuple[list[_LocalChunk], dict[str, Any]]:
        suffix = path.suffix.lower()
        if suffix not in _DOCUMENT_SUFFIXES:
            raise ValueError(f"unsupported knowledge document suffix: {suffix}")
        source_sha = _sha256_path(path)
        source_id = "upload:" + source_sha[:20]
        if suffix in {".txt", ".md"}:
            raw = path.read_bytes()
            text = raw.decode("utf-8-sig")
            elements, warnings = self.ingestor.parse_markdown(
                text, asset_root=path.parent if suffix == ".md" else None
            )
            # An author-supplied alt attribute is NOT an independently verified
            # diagram interpretation. Keep the image but prevent its use as
            # factual RAG evidence until a reviewer attests the description.
            image_count = sum(e.kind.value == "IMAGE" for e in elements)
            if image_count:
                from astrograph.retrieval.structured_document import DocumentElementKind
                elements = [e.model_copy(update={"representation_reviewed": False})
                            if e.kind == DocumentElementKind.IMAGE else e for e in elements]
                warnings.extend(["IMAGE_ALT_REQUIRES_INDEPENDENT_REVIEW"])
            ingestion = self.ingestor.ingest_elements(
                source_id=source_id,
                source_uri=f"upload://{path.name}",
                source_sha256=source_sha,
                knowledge_version="session-upload-v1",
                elements=elements,
                warnings=warnings,
            )
            parser = "ASTROGRAPH_MARKDOWN_NATIVE"
        elif suffix == ".docx":
            ingestion, audit = self.ingestor.ingest_docx(
                path, source_id=source_id, knowledge_version="session-upload-v1"
            )
            warnings = list(audit.warnings)
            parser = "ASTROGRAPH_DOCX_NATIVE"
        else:
            try:
                ingestion, audit = self.ingestor.ingest_pdf(
                    path, source_id=source_id, knowledge_version="session-upload-v1"
                )
                warnings = list(audit.warnings)
                parser = "ASTROGRAPH_PDF_LAYOUT"
            except ValueError as exc:
                # Only scanned/no-text fallback: parser failures on otherwise
                # structured PDFs must not be silently hidden by machine OCR.
                if ('no extractable text' not in str(exc)
                        or os.environ.get('ASTROGRAPH_MATERIAL_OCR_ENGINE','').lower()
                        not in {'tesseract','docling'}):
                    raise
                from astrograph.retrieval.advanced_multimodal import (
                    ocr_scanned_pdf, docling_pdf_candidates,
                )
                mode = os.environ['ASTROGRAPH_MATERIAL_OCR_ENGINE'].lower()
                candidate = (ocr_scanned_pdf(path, languages=os.environ.get(
                    'ASTROGRAPH_MATERIAL_OCR_LANGUAGES', 'chi_sim+eng'))
                    if mode == 'tesseract' else docling_pdf_candidates(path))
                if not candidate['elements']:
                    raise ValueError('OCR produced no reviewable text') from exc
                ingestion = self.ingestor.ingest_elements(
                    source_id=source_id, source_uri=f'upload://{path.name}',
                    source_sha256=source_sha, knowledge_version='ocr-candidate-v1',
                    elements=candidate['elements'],
                    warnings=['OCR_OR_VISION_OUTPUT_REQUIRES_INDEPENDENT_REVIEW',
                              *(candidate.get('warnings') or [])],
                )
                warnings = list(ingestion.warnings)
                parser = 'UNREVIEWED_'+candidate['parser']
        chunks = [
            _LocalChunk(
                chunk_id=chunk.chunk_id,
                source_id=source_id,
                filename=path.name,
                source_sha256=source_sha,
                text=chunk.text,
                section_path=tuple(chunk.section_path),
                source_locators=tuple(chunk.source_locators),
                review_status=chunk.review_status,
                content_safety_status=chunk.content_safety_status,
                modalities=tuple(chunk.modalities),
            )
            for chunk in ingestion.chunks
        ]
        return chunks, {
            "filename": path.name,
            "source_id": source_id,
            "source_sha256": source_sha,
            "parser": parser,
            "status": ("UNREVIEWED_OCR_CANDIDATES" if parser.startswith("UNREVIEWED_")
                       else "PARSED"),
            "review_required": parser.startswith("UNREVIEWED_"),
            "chunk_count": len(chunks),
            "warnings": sorted(set(warnings)),
            "diagnostic_vote_allowed": False,
            "active_kg_write_allowed": False,
        }

    def build(self, paths: list[Path]) -> tuple[list[_LocalChunk], list[dict[str, Any]]]:
        chunks: list[_LocalChunk] = []
        receipts: list[dict[str, Any]] = []
        for path in sorted(paths, key=lambda item: item.name.casefold()):
            try:
                parsed, receipt = self._ingest_one(path)
                chunks.extend(parsed)
                receipts.append(receipt)
            except Exception as exc:
                receipts.append({
                    "filename": path.name,
                    "source_sha256": _sha256_path(path),
                    "status": "PARSE_FAILED",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "diagnostic_vote_allowed": False,
                    "active_kg_write_allowed": False,
                })
        return chunks, receipts

    def retrieve(self, query: str, chunks: list[_LocalChunk]) -> list[dict[str, Any]]:
        query = query.strip()[:_MAX_QUERY_CHARS]
        if not query or not chunks:
            return []
        terms = _tokens(query)
        documents = [_tokens(' '.join(chunk.section_path) + ' ' + chunk.text) for chunk in chunks]
        lexical = _normalize(_bm25(terms, documents))
        query_set = set(terms)
        scored: list[tuple[float, int, _LocalChunk, float]] = []
        for index, (chunk, bm25) in enumerate(zip(chunks, lexical)):
            document_set = set(documents[index])
            coverage = len(query_set & document_set) / max(1, len(query_set))
            score = 0.75 * bm25 + 0.25 * coverage
            if score > 0.0:
                scored.append((score, index, chunk, coverage))
        scored.sort(key=lambda item: (-item[0], item[2].chunk_id))
        hits: list[dict[str, Any]] = []
        for rank, (score, _, chunk, coverage) in enumerate(scored[: self.top_k], start=1):
            hits.append({
                "knowledge_id": chunk.chunk_id,
                "rank": rank,
                "score": round(score, 9),
                "term_coverage": round(coverage, 9),
                "filename": chunk.filename,
                "source_id": chunk.source_id,
                "source_sha256": chunk.source_sha256,
                "section_path": list(chunk.section_path),
                "source_locators": list(chunk.source_locators),
                "quote": chunk.text[:_MAX_CHUNK_CHARS_FOR_MODEL],
                "representation_review_status": chunk.review_status,
                "representation_content_status": chunk.content_safety_status,
                "knowledge_validation_status": "UNREVIEWED_SESSION_UPLOAD",
                "role": "UPLOADED_KNOWLEDGE_NOT_OBSERVATION",
                "diagnostic_vote_allowed": False,
            })
        return hits


class Anything2OntologyCLIAdapter:
    """Execute the supplied Anything2Ontology CLI in a job-isolated workspace.

    Installed console scripts are preferred.  When ``source_root`` points at the
    supplied Anything2Ontology repository, the same four CLI modules are executed
    through the current Python interpreter with a scoped ``PYTHONPATH``.  No
    installation or Active-KG write is performed by this adapter.
    """

    stages = (
        ("anything2md", "anything2markdown.cli", ["run"]),
        ("md2chunks", "markdown2chunks.cli", ["run"]),
        ("chunks2skus", "chunks2skus.cli", ["run"]),
        ("skus2ontology", "skus2ontology.cli", ["run", "--skip-chatbot"]),
    )

    def __init__(self, *, timeout_s: float = 900.0, source_root: str | Path | None = None,
                 python_executable: str | Path | None = None) -> None:
        self.timeout_s = timeout_s
        self.source_root = Path(source_root).expanduser().resolve() if source_root else None
        self.python_executable = str(Path(python_executable).expanduser().absolute()) if python_executable else sys.executable
        if not Path(self.python_executable).is_file() or not os.access(self.python_executable, os.X_OK):
            raise ValueError("A2O Python interpreter must be an existing executable")

    def _source_module_available(self, module: str) -> bool:
        if self.source_root is None or not self.source_root.is_dir():
            return False
        package = module.split(".", 1)[0]
        return (self.source_root / "src" / package / "cli.py").is_file()

    def _resolve_stage(self, command: str, module: str, arguments: list[str]) -> tuple[list[str], str] | None:
        # An administrator-specified source_root pins the third-party revision.
        # Falling back to a different global executable would make provenance and
        # reproducibility ambiguous, so only use PATH when no source root is set.
        if self.source_root is not None:
            if self._source_module_available(module):
                return [self.python_executable, "-m", module, *arguments], "SOURCE_TREE_MODULE"
            return None
        installed = shutil.which(command)
        if installed:
            return [installed, *arguments], "INSTALLED_CLI"
        if os.name == "nt":
            # PATHEXT ignores extensionless POSIX-style console fixtures.  Admit
            # only an exact PATH file whose shebang binds it to the configured
            # Python interpreter; arbitrary interpreters remain rejected.
            expected = ("#!" + str(Path(self.python_executable).resolve())).casefold().encode()
            portable_python_shebangs = {
                b"#!/usr/bin/env python",
                b"#!/usr/bin/env python3",
                b"#!/usr/bin/python",
                b"#!/usr/bin/python3",
            }
            for raw_root in os.environ.get("PATH", "").split(os.pathsep):
                if not raw_root:
                    continue
                candidate = Path(raw_root) / command
                try:
                    first_line = candidate.read_bytes().splitlines()[:1]
                except OSError:
                    continue
                shebang = first_line[0].strip().lower() if first_line else b""
                if candidate.is_file() and (shebang == expected or shebang in portable_python_shebangs):
                    return [self.python_executable, str(candidate), *arguments], "PYTHON_SHEBANG_CLI"
        return None

    def available(self) -> bool:
        return all(self._resolve_stage(command, module, arguments) is not None
                   for command, module, arguments in self.stages)

    def discovery(self) -> dict[str, Any]:
        stages = {}
        for command, module, arguments in self.stages:
            resolved = self._resolve_stage(command, module, arguments)
            stages[command] = {
                "available": resolved is not None,
                "mode": resolved[1] if resolved else "UNAVAILABLE",
                "module": module,
            }
        return {
            "source_root": str(self.source_root) if self.source_root else None,
            "stages": stages,
            "ready": all(item["available"] for item in stages.values()),
        }

    def probe(self) -> dict[str, Any]:
        """Check whether each real CLI/module can import and render help.

        This does not execute extraction or call an LLM; it catches missing Python
        dependencies that static source-tree discovery alone cannot detect.
        """
        stages: list[dict[str, Any]] = []
        env = os.environ.copy()
        if self.source_root is not None:
            source_path = str(self.source_root / "src")
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = source_path + (os.pathsep + existing if existing else "")
        for command, module, arguments in self.stages:
            resolved = self._resolve_stage(command, module, arguments)
            if resolved is None:
                stages.append({
                    "command": command, "status": "UNAVAILABLE", "mode": "UNAVAILABLE", "exit_code": None,
                })
                continue
            invocation, mode = resolved
            help_invocation = invocation[:]
            # Drop the stage action and request top-level help instead.
            if mode == "INSTALLED_CLI":
                help_invocation = [help_invocation[0], "--help"]
            else:
                help_invocation = help_invocation[:3] + ["--help"]
            try:
                completed = run_managed_subprocess(
                    help_invocation, capture_output=True, text=True, timeout=min(self.timeout_s, 30.0),
                    check=False, env=env,
                )
                status = "READY" if completed.returncode == 0 else "IMPORT_OR_CONFIG_ERROR"
                stages.append({
                    "command": command,
                    "status": status,
                    "mode": mode,
                    "exit_code": completed.returncode,
                    "stdout_tail": (completed.stdout or "")[-2000:],
                    "stderr_tail": (completed.stderr or "")[-4000:],
                })
            except Exception as exc:
                stages.append({
                    "command": command, "status": "PROBE_FAILED", "mode": mode, "exit_code": None,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                })
        return {
            "status": "READY" if stages and all(item["status"] == "READY" for item in stages) else "NOT_READY",
            "stages": stages,
            "claim_boundary": "Probe checks CLI import/help only; it does not prove ontology extraction quality.",
        }

    def run(self, paths: list[Path], output_root: Path) -> dict[str, Any]:
        if not paths:
            return {"status": "NO_DOCUMENTS", "active_kg_modified": False, "review_material": []}
        resolved_stages = []
        missing = []
        for command, module, arguments in self.stages:
            resolved = self._resolve_stage(command, module, arguments)
            if resolved is None:
                missing.append(command)
            else:
                resolved_stages.append((command, arguments, resolved[0], resolved[1]))
        if missing:
            return {
                "status": "NOT_CONFIGURED",
                "missing_commands": missing,
                "source_root": str(self.source_root) if self.source_root else None,
                "active_kg_modified": False,
                "review_material": [],
            }
        workspace = output_root / "anything2ontology"
        if workspace.exists():
            shutil.rmtree(workspace)
        (workspace / "input").mkdir(parents=True)
        for source in paths:
            shutil.copy2(source, workspace / "input" / source.name)
        env = os.environ.copy()
        if self.source_root is not None:
            source_path = str(self.source_root / "src")
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = source_path + (os.pathsep + existing if existing else "")
        env.update({
            "INPUT_DIR": str(workspace / "input"),
            "OUTPUT_DIR": str(workspace / "output"),
            "LOG_DIR": str(workspace / "logs"),
            "SKUS_OUTPUT_DIR": str(workspace / "output" / "skus"),
            "ONTOLOGY_DIR": str(workspace / "ontology"),
            "LANGUAGE": env.get("ASTROGRAPH_PLATFORM_A2O_LANGUAGE", "zh"),
        })
        receipts: list[dict[str, Any]] = []
        for command, arguments, invocation, mode in resolved_stages:
            started = time.monotonic()
            completed = run_managed_subprocess(
                invocation,
                cwd=workspace,
                env=env,
                text=True,
                capture_output=True,
                timeout=self.timeout_s,
                check=False,
            )
            receipt = {
                "command": command,
                "arguments": arguments,
                "execution_mode": mode,
                "exit_code": completed.returncode,
                "duration_s": round(time.monotonic() - started, 6),
                "stdout_tail": (completed.stdout or "")[-_MAX_COMMAND_OUTPUT_CHARS:],
                "stderr_tail": (completed.stderr or "")[-_MAX_COMMAND_OUTPUT_CHARS:],
            }
            receipts.append(receipt)
            if completed.returncode != 0:
                return {
                    "status": "FAILED",
                    "commands": receipts,
                    "active_kg_modified": False,
                    "workspace": str(workspace),
                    "review_material": [],
                }
        ontology = workspace / "ontology"
        files: list[dict[str, Any]] = []
        review_material: list[dict[str, Any]] = []
        remaining_context = _MAX_ONTOLOGY_CONTEXT_CHARS
        if ontology.is_dir():
            for path in sorted(p for p in ontology.rglob("*") if p.is_file()):
                relative = path.relative_to(workspace).as_posix()
                digest = _sha256_path(path)
                files.append({
                    "path": relative,
                    "sha256": digest,
                    "size_bytes": path.stat().st_size,
                })
                if remaining_context <= 0 or path.suffix.lower() not in {".md", ".txt", ".json", ".yaml", ".yml"}:
                    continue
                try:
                    text = path.read_text(encoding="utf-8-sig")
                except (OSError, UnicodeError):
                    continue
                excerpt = text[: min(_MAX_ONTOLOGY_FILE_CHARS, remaining_context)].strip()
                if not excerpt:
                    continue
                remaining_context -= len(excerpt)
                review_material.append({
                    "ontology_id": "ONTOLOGY-" + digest[:20],
                    "path": relative,
                    "sha256": digest,
                    "quote": excerpt,
                    "knowledge_validation_status": "REVIEW_REQUIRED",
                    "role": "ONTOLOGY_CANDIDATE_NOT_OBSERVATION",
                    "diagnostic_vote_allowed": False,
                    "active_kg_write_allowed": False,
                })
        return {
            "status": "REVIEW_REQUIRED" if files else "NO_OUTPUT",
            "commands": receipts,
            "workspace": str(workspace),
            "ontology_files": files,
            "review_material": review_material,
            "active_kg_modified": False,
            "claim_boundary": "Anything2Ontology output is session-scoped review material and never an automatic Active KG patch.",
        }


def _diagnosis_search_text(snapshot: dict[str, Any]) -> str:
    parts = [
        str(snapshot.get("candidate_fault_id") or ""),
        *[str(item) for item in snapshot.get("candidate_fault_ids", [])],
        *[str(item) for item in snapshot.get("location_candidates", [])],
        *[str(item.get("field") or "") for item in snapshot.get("evidence", []) if isinstance(item, dict)],
        *[str(item.get("interpretation") or "") for item in snapshot.get("evidence", []) if isinstance(item, dict)],
    ]
    return " ".join(item for item in parts if item).strip()[:_MAX_QUERY_CHARS]


_MEASUREMENT_RE = re.compile(
    r"(?<![\w.])([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*"
    r"(°C|degC|°|deg|rad/s|rad|rpm|N[·.\-]?m|Nm|mA|A|V|W|Hz|Pa|bps|Wh|mAh|K|ms|s|%)"
    r"(?![A-Za-z])",
    re.IGNORECASE,
)


def _normalize_measurement_unit(unit: str) -> str:
    value = str(unit or "").strip().replace(" ", "").casefold()
    aliases = {
        "a": "A", "ma": "mA", "v": "V", "w": "W", "hz": "Hz", "pa": "Pa",
        "bps": "bps", "wh": "Wh", "mah": "mAh", "k": "K", "s": "s", "ms": "ms",
        "%": "%", "°": "deg", "deg": "deg", "°c": "degC", "degc": "degC",
        "rad": "rad", "rad/s": "rad/s", "rpm": "rpm",
        "nm": "N·m", "n·m": "N·m", "n.m": "N·m", "n-m": "N·m",
    }
    return aliases.get(value, str(unit or "").strip())


def _infer_measurement_unit(field: str) -> str | None:
    low = re.sub(r"_\d+$", "", str(field or "").casefold())
    for suffix, unit in (
        ("_motor_current_a", "A"), ("_current_a", "A"), ("_voltage_v", "V"),
        ("_power_w", "W"), ("_temp_c", "degC"), ("_temperature_c", "degC"),
        ("_deg", "deg"), ("_rad_s", "rad/s"), ("_rad", "rad"),
        ("_nm", "N·m"), ("_rpm", "rpm"), ("_hz", "Hz"), ("_pa", "Pa"),
        ("_bps", "bps"), ("_wh", "Wh"), ("_s", "s"),
    ):
        if low.endswith(suffix):
            return unit
    return None


def _text_measurements(text: str) -> list[tuple[float, str]]:
    found: list[tuple[float, str]] = []
    for match in _MEASUREMENT_RE.finditer(str(text or "")):
        try:
            number = float(match.group(1))
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            found.append((number, _normalize_measurement_unit(match.group(2))))
    return found


def _evidence_measurements(item: dict[str, Any]) -> list[tuple[float, str]]:
    pairs: list[tuple[float, str]] = []
    for key in ("interpretation", "statistic", "summary"):
        pairs.extend(_text_measurements(str(item.get(key) or "")))
    explicit_unit = str(
        item.get("unit") or item.get("observed_unit") or item.get("output_unit") or ""
    ).strip()
    if not explicit_unit:
        explicit_unit = _infer_measurement_unit(str(item.get("field") or "")) or ""
    for key in ("observed_value", "value", "metric_value"):
        value = item.get(key)
        if value is None or isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and explicit_unit:
            pairs.append((number, _normalize_measurement_unit(explicit_unit)))
    return pairs


def _validate_observation_numeric_grounding(
    text: str,
    citation_ids: set[str],
    diagnosis_snapshot: dict[str, Any],
    *,
    context: str = "observation",
) -> None:
    """Reject measured quantities that are not present in the cited frozen evidence.

    Citation-ID membership alone is insufficient grounding: a model could otherwise cite
    a real evidence ID while replacing ``0.2 A`` with ``900 A``.  This validator is
    deliberately narrow and deterministic.  It checks only explicit number+unit
    measurements, leaving qualitative prose and document-sourced quantities to their
    own citation contracts.
    """
    measurements = _text_measurements(text)
    if not measurements:
        return
    evidence_by_id = {
        str(item.get("evidence_id")): item
        for item in diagnosis_snapshot.get("evidence", [])
        if isinstance(item, dict) and item.get("evidence_id")
    }
    supported: list[tuple[float, str]] = []
    for citation_id in citation_ids:
        item = evidence_by_id.get(str(citation_id))
        if item is not None:
            supported.extend(_evidence_measurements(item))
    for value, unit in measurements:
        if any(
            unit == supported_unit and math.isclose(value, supported_value, rel_tol=1e-9, abs_tol=1e-12)
            for supported_value, supported_unit in supported
        ):
            continue
        raise ValueError(
            f"{context} contains an uncited or numerically inconsistent measured value: {value:g} {unit}"
        )


def _validate_explanation(
    explanation: PlatformKnowledgeExplanation,
    diagnosis_snapshot: dict[str, Any],
    local_hits: list[dict[str, Any]],
    literature_documents: list[dict[str, Any]],
    ontology_review_material: list[dict[str, Any]] | None = None,
    material_review_context: dict[str, Any] | None = None,
) -> None:
    frozen = diagnosis_snapshot.get("candidate_fault_id")
    if explanation.candidate_fault_id != frozen:
        raise ValueError("LLM may not override the frozen diagnosis candidate or abstention")
    allowed_evidence = {
        str(item.get("evidence_id"))
        for item in diagnosis_snapshot.get("evidence", [])
        if isinstance(item, dict) and item.get("evidence_id")
    }
    allowed_local = {str(item["knowledge_id"]) for item in local_hits}
    allowed_literature = {str(item["document_id"]) for item in literature_documents}
    # Anything2Ontology products are review-only knowledge candidates, never observations.
    allowed_ontology = {
        str(item["ontology_id"])
        for item in (ontology_review_material or [])
        if isinstance(item, dict) and item.get("ontology_id")
    }
    context_candidates=(material_review_context or {}).get('candidates') or []
    allowed_material_log={str(e['evidence_id']) for c in context_candidates
              for e in ((c.get('log_evidence_matrix') or {}).get('events') or [])}
    allowed_material_reference={str(h['knowledge_id']) for c in context_candidates
              for h in ((c.get('reference_retrieval') or {}).get('hits') or [])}
    allowed_graph={str(p['graph_path_id']) for c in context_candidates
              for p in (c.get('graph_evidence_paths') or []) if p.get('graph_path_id')}
    allowed_case={str(h['case_id']) for c in context_candidates
              for h in ((c.get('historical_confirmed_cases') or {}).get('hits') or [])}
    material_types={'MATERIAL_LOG_CUE':allowed_material_log,'MATERIAL_REFERENCE':allowed_material_reference,
                    'GRAPH_ASSOCIATION':allowed_graph,'CONFIRMED_CASE':allowed_case}
    for claim in explanation.claims:
        citations = set(claim.citation_ids)
        if claim.kind == "OBSERVATION" and (not citations or not citations <= allowed_evidence):
            raise ValueError("observation claim must cite only frozen evidence IDs")
        if claim.kind == "OBSERVATION":
            _validate_observation_numeric_grounding(
                claim.text, citations, diagnosis_snapshot, context="observation claim"
            )
        if claim.kind == "LOCAL_KNOWLEDGE" and (not citations or not citations <= allowed_local):
            raise ValueError("local knowledge claim must cite retrieved upload chunk IDs")
        if claim.kind == "LITERATURE" and (not citations or not citations <= allowed_literature):
            raise ValueError("literature claim must cite retrieved PaperSeek document IDs")
        if claim.kind == "ONTOLOGY_CANDIDATE" and (not citations or not citations <= allowed_ontology):
            raise ValueError("ontology candidate claim must cite review-only Anything2Ontology IDs")
        if claim.kind in material_types and (not citations or not citations <= material_types[claim.kind]):
            raise ValueError(f'{claim.kind} cites an unknown or unapproved material context ID')
        if claim.kind == "HYPOTHESIS" and not citations <= (
            allowed_evidence | allowed_local | allowed_literature | allowed_ontology |
            allowed_material_log | allowed_material_reference | allowed_graph | allowed_case
        ):
            raise ValueError("hypothesis references unknown citations")
        if claim.kind == "HYPOTHESIS":
            evidence_citations = citations & allowed_evidence
            if evidence_citations:
                _validate_observation_numeric_grounding(
                    claim.text, evidence_citations, diagnosis_snapshot, context="hypothesis evidence measurement"
                )
    _validate_observation_numeric_grounding(
        explanation.summary, allowed_evidence, diagnosis_snapshot, context="explanation summary"
    )


def _local_retrieval_quality(
    diagnosis_snapshot: dict[str, Any],
    hits: list[dict[str, Any]],
    ingestion: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate = str(diagnosis_snapshot.get("candidate_fault_id") or "").casefold()
    candidate_anchored = [
        item for item in hits if candidate and candidate in str(item.get("quote") or "").casefold()
    ]
    parsed_sources = [item for item in ingestion if item.get("status") == "PARSED"]
    unique_sources = {str(item.get("source_sha256") or "") for item in hits}
    if candidate_anchored:
        grounding = "DIRECT_CANDIDATE_ANCHOR"
    elif hits:
        grounding = "LEXICAL_CONTEXT_ONLY"
    else:
        grounding = "NO_RELEVANT_UPLOAD_CONTEXT"
    return {
        "parsed_source_count": len(parsed_sources),
        "retrieved_hit_count": len(hits),
        "unique_retrieved_source_count": len(unique_sources),
        "candidate_anchor_hit_count": len(candidate_anchored),
        "top_score": hits[0]["score"] if hits else None,
        "grounding_status": grounding,
        "claim_boundary": "Retrieval quality describes lexical grounding only; it is not diagnosis accuracy.",
    }


def _deterministic_grounded_explanation(
    diagnosis_snapshot: dict[str, Any],
    local_hits: list[dict[str, Any]],
    literature_documents: list[dict[str, Any]],
    ontology_review_material: list[dict[str, Any]],
) -> PlatformKnowledgeExplanation:
    candidate = diagnosis_snapshot.get("candidate_fault_id")
    claims: list[PlatformKnowledgeClaim] = []
    for item in diagnosis_snapshot.get("evidence", [])[:4]:
        if not isinstance(item, dict) or not item.get("evidence_id"):
            continue
        text = str(item.get("interpretation") or item.get("field") or "Frozen diagnosis evidence").strip()
        if text:
            claims.append(PlatformKnowledgeClaim(
                kind="OBSERVATION", text=text[:1200], citation_ids=[str(item["evidence_id"])],
            ))
    for item in local_hits[:3]:
        quote = str(item.get("quote") or "").strip().replace("\n", " ")
        if quote:
            claims.append(PlatformKnowledgeClaim(
                kind="LOCAL_KNOWLEDGE",
                text=("上传资料片段：" + quote[:900]),
                citation_ids=[str(item["knowledge_id"])],
            ))
    for item in literature_documents[:2]:
        text = str(item.get("text") or "").strip().replace("\n", " ")
        title = str(item.get("title") or "").strip()
        if text and item.get("document_id"):
            claims.append(PlatformKnowledgeClaim(
                kind="LITERATURE",
                text=(f"文献《{title}》摘要片段：" if title else "文献摘要片段：") + text[:900],
                citation_ids=[str(item["document_id"])],
            ))
    for item in ontology_review_material[:2]:
        quote = str(item.get("quote") or "").strip().replace("\n", " ")
        if quote and item.get("ontology_id"):
            claims.append(PlatformKnowledgeClaim(
                kind="ONTOLOGY_CANDIDATE",
                text="待审核本体候选片段：" + quote[:900],
                citation_ids=[str(item["ontology_id"])],
            ))
    if candidate:
        summary = (
            f"冻结诊断候选为 {candidate}。当前说明由程序按已冻结证据和已检索知识逐项摘录生成，"
            "未调用大模型，也不会改变 BiRRF 或 Decision Governor 的结果。"
        )
    else:
        summary = (
            "底层诊断未给出可冻结故障候选。当前说明仅整理可追溯证据与知识，不把文档或文献提升为故障结论。"
        )
    explanation = PlatformKnowledgeExplanation(
        candidate_fault_id=str(candidate) if candidate else None,
        abstained=candidate is None,
        summary=summary,
        claims=claims,
        limitations=[
            "DETERMINISTIC_REFERENCE_SYNTHESIS：不是大语言模型推理。",
            "上传资料、实时文献和本体候选均不是当前航天器观测。",
            "引用存在性已约束，但未执行全文语义真实性审查。",
        ],
        recommended_review_actions=[
            "核对冻结诊断证据、检索片段与知识来源。",
            "需要自然语言综合推理时配置受控 Qwen/DeepSeek 兼容服务。",
        ],
    )
    _validate_explanation(explanation, diagnosis_snapshot, local_hits, literature_documents, ontology_review_material)
    return explanation


class PlatformKnowledgeEnrichmentService:
    def __init__(
        self,
        config: PlatformKnowledgeConfig | None = None,
        *,
        paperseek_client: Any | None = None,
        llm_backend: Any | None = None,
        a2o_adapter: Anything2OntologyCLIAdapter | None = None,
    ) -> None:
        self.config = config or PlatformKnowledgeConfig.from_environment()
        self._paperseek_configuration_error: str | None = None
        if paperseek_client is not None:
            self.paperseek_client = paperseek_client
        elif self.config.paperseek_url:
            self.paperseek_client = PaperSeekHTTPClient(PaperSeekConfig(
                base_url=self.config.paperseek_url,
                data_source=self.config.paperseek_data_source,
                top_k=self.config.paperseek_top_k,
            ))
        elif self.config.paperseek_source_root:
            try:
                cli_client = PaperSeekCLIClient(PaperSeekCLIConfig(
                    source_root=self.config.paperseek_source_root,
                    data_source=self.config.paperseek_data_source,
                    top_k=self.config.paperseek_top_k,
                    timeout_s=self.config.paperseek_cli_timeout_s,
                ))
                if cli_client.available():
                    self.paperseek_client = cli_client
                else:
                    self.paperseek_client = None
                    self._paperseek_configuration_error = "PaperSeek source tree does not expose a runnable CLI"
            except Exception as exc:
                self.paperseek_client = None
                self._paperseek_configuration_error = f"{type(exc).__name__}: {exc}"
        else:
            self.paperseek_client = None
        if llm_backend is not None:
            self.llm_backend = llm_backend
        elif self.config.llm_base_url and self.config.llm_model:
            key = os.environ.get("ASTROGRAPH_LLM_API_KEY", "local-no-secret")
            self.llm_backend = OpenAICompatibleStructuredBackend(OpenAICompatibleConfig(
                provider=self.config.llm_provider,
                base_url=self.config.llm_base_url,
                api_key=SecretStr(key),
                model=self.config.llm_model,
                temperature=0.0,
                timeout_s=self.config.llm_timeout_s,
                max_tokens=self.config.llm_max_tokens,
                response_format_strategy=self.config.llm_response_format_strategy,
                tokenizer_mode="disabled",
            ))
        else:
            self.llm_backend = None
        self.a2o_adapter = a2o_adapter or Anything2OntologyCLIAdapter(
            timeout_s=self.config.a2o_timeout_s, source_root=self.config.a2o_source_root,
            python_executable=self.config.a2o_python_executable
        )

    @classmethod
    def from_environment(cls) -> "PlatformKnowledgeEnrichmentService":
        return cls(PlatformKnowledgeConfig.from_environment())

    def capabilities(self) -> dict[str, Any]:
        paperseek_backend = None
        if self.paperseek_client is not None:
            backend_mode = getattr(self.paperseek_client, "backend_mode", None)
            paperseek_backend = backend_mode() if callable(backend_mode) else type(self.paperseek_client).__name__
        return {
            "schema_version": "astrograph-platform-knowledge-capabilities.v2",
            "native_document_parsers": ["TXT", "MARKDOWN", "DOCX", "PDF_LAYOUT"],
            "local_session_retrieval": True,
            "retrieval_quality_receipt": True,
            "deterministic_grounded_synthesis": True,
            "paperseek": (
                "CONFIGURED" if self.paperseek_client is not None
                else "CONFIGURATION_ERROR" if self._paperseek_configuration_error
                else "NOT_CONFIGURED"
            ),
            "paperseek_backend": paperseek_backend,
            "paperseek_configuration_error": self._paperseek_configuration_error,
            "llm": "CONFIGURED" if self.llm_backend is not None else "NOT_CONFIGURED",
            "anything2ontology": (
                "CONFIGURED" if self.config.a2o_enabled and self.a2o_adapter.available()
                else "ENABLED_COMMANDS_MISSING" if self.config.a2o_enabled
                else "DISABLED"
            ),
            "anything2ontology_discovery": self.a2o_adapter.discovery() if hasattr(self.a2o_adapter, "discovery") else None,
            "documents_can_change_diagnosis": False,
            "active_kg_auto_write": False,
        }

    def enrich(
        self,
        *,
        task: str,
        diagnosis_snapshot: dict[str, Any],
        document_paths: list[Path],
        output_root: Path,
        material_review_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        output_root.mkdir(parents=True, exist_ok=True)
        timings: dict[str, float] = {}
        reported_observations = review_numbered_observations(task)

        local_started = time.monotonic()
        local_index = UploadedDocumentKnowledgeIndex(top_k=self.config.local_top_k)
        chunks, ingestion = local_index.build(document_paths)
        controlled_query = _diagnosis_search_text(diagnosis_snapshot)
        # The task can narrow explanation retrieval, never candidate generation.
        retrieval_query = " ".join([controlled_query, task.strip()[:1000]]).strip()
        local_hits, retrieval_strategy = retrieve_observation_balanced(
            local_index, chunks, controlled_query=controlled_query, task=task,
            observation_ledger=reported_observations,
        )
        retrieval_quality = _local_retrieval_quality(diagnosis_snapshot, local_hits, ingestion)
        timings["local_parse_and_retrieval"] = round(time.monotonic() - local_started, 6)

        paper_started = time.monotonic()
        paperseek: dict[str, Any]
        if self.paperseek_client is None and self._paperseek_configuration_error:
            paperseek = {
                "status": "FAILED",
                "documents": [],
                "error": {"type": "ConfigurationError", "message": self._paperseek_configuration_error},
            }
        elif self.paperseek_client is None:
            paperseek = {"status": "NOT_CONFIGURED", "documents": []}
        elif not diagnosis_snapshot.get("candidate_fault_id"):
            paperseek = {"status": "NOT_TRIGGERED_NO_DIAGNOSIS_CANDIDATE", "documents": []}
        else:
            try:
                # User prose is intentionally excluded from network/CLI search.
                paperseek = self.paperseek_client.search(
                    "Spacecraft fault diagnosis mechanisms evidence " + controlled_query
                )
            except Exception as exc:
                paperseek = {
                    "status": "FAILED",
                    "documents": [],
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
        timings["paperseek"] = round(time.monotonic() - paper_started, 6)

        a2o_started = time.monotonic()
        if self.config.a2o_enabled:
            try:
                a2o = self.a2o_adapter.run(document_paths, output_root)
            except Exception as exc:
                a2o = {
                    "status": "FAILED",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "active_kg_modified": False,
                    "review_material": [],
                }
        else:
            a2o = {"status": "DISABLED", "active_kg_modified": False, "review_material": []}
        timings["anything2ontology"] = round(time.monotonic() - a2o_started, 6)

        a2o.setdefault("review_material", [])
        documents = list(paperseek.get("documents") or [])
        ontology_review_material = list(a2o.get("review_material") or [])

        generation_started = time.monotonic()
        generation: dict[str, Any]
        if self.llm_backend is None:
            generation = {
                "status": "NOT_CONFIGURED",
                "explanation": None,
                "semantic_hallucination_assessment": "NOT_EVALUATED",
            }
        else:
            try:
                causal_review_for_llm = (
                    summarize_material_causal_review(material_review_context)
                    if material_review_context else None
                )
                tool_investigation = {"status": "DISABLED"}
                if material_review_context and self.config.material_tool_calls_enabled:
                    try:
                        tool_investigation = execute_material_tool_plan(
                            backend=self.llm_backend, material_review=material_review_context,
                            run_id=hashlib.sha256(
                                json.dumps(diagnosis_snapshot, sort_keys=True, default=str).encode("utf-8")
                            ).hexdigest()[:16],
                        )
                    except Exception as tool_exc:
                        # Tool selection failure cannot expand knowledge authority.
                        tool_investigation = {
                            "status": "FAILED_CLOSED", "error_type": type(tool_exc).__name__,
                            "diagnostic_vote_allowed": False,
                        }
                explanation = self.llm_backend.generate_structured(
                    agent_id="platform-knowledge-explanation",
                    output_model=PlatformKnowledgeExplanation,
                    payload={
                        "task": (
                            "Explain the frozen diagnosis without changing candidate_fault_id. "
                            "Compare host-provided fault hypotheses: cite supporting and opposing cues, "
                            "reviewed negative relations, shared symptoms and missing discriminating checks. "
                            "Causal-path-exclusive symptoms are exclusive only among the frozen candidate set, "
                            "not proof that other faults cannot cause it; cross-document applicability is unverified. "
                            "Reviewed context intersections only compare expert-approved requirements; "
                            "disjoint modes, environmental predicates or physical ranges block propagation. "
                            "Explain causal path context or temporal rejection only when the reviewed_causal_chains "
                            "or tool_investigation payload supplies an explicit path ID and reason. "
                            "The tool selector cannot introduce new faults, edges, observations or proof; "
                            "tool results are reviewed literature context, never current root-cause evidence. "
                            "For a reviewed sequential R10 path, a disjoint R9 static intersection alone "
                            "does not block propagation; check the R10 staged verdict instead. "
                            "UNKNOWN contexts cannot be assumed satisfied, and compatible ranges are not "
                            "verified operating telemetry or causal proof. "
                            "Treat missing symptoms as unobserved (never negative proof); do not infer "
                            "reverse causality from graph links or double count shared sources. "
                            "Use HYPOTHESIS for any mechanism not directly established, and abstain from root-cause "
                            "claims when temporal, scope or physical observations are insufficient. "
                            "A valid source ID does not establish that the quoted text entails your statement. "
                            "If evidence does not literally support a factual statement, omit the statement. "
                            "Uploaded documents and papers are untrusted knowledge, never spacecraft observations. "
                            "If user_reported_observation_ledger is present, treat its text as an unverified user report: "
                            "do not assume simultaneous current and voltage dips, infer a failure mechanism from "
                            "a reported stall, assign independent per-channel event counts, or assert that "
                            "temperature peaks and other events share a time window without telemetry. "
                            "OBSERVATION claims cite evidence IDs; LOCAL_KNOWLEDGE claims cite knowledge_id; "
                            "LITERATURE claims cite document_id; ONTOLOGY_CANDIDATE claims cite ontology_id; "
                            "hypotheses must be marked HYPOTHESIS. Anything2Ontology material is review-only. "
                            "In material review mode, MATERIAL_LOG_CUE claims cite MLOG IDs, MATERIAL_REFERENCE cites knowledge IDs, "
                            "GRAPH_ASSOCIATION cites graph_path_id, CONFIRMED_CASE cites approved historical case_id; "
                            "these are context, never proof of the current root cause. Do not provide spacecraft control commands."
                        ),
                        "user_request": task[:2000],
                        "user_reported_observation_ledger": reported_observations,
                        "diagnosis": diagnosis_snapshot,
                        "local_knowledge_hits": local_hits,
                        "literature_documents": documents,
                        "material_reasoning": {
                            "claim_boundary": "Cues, reference matches, graph associations and historical cases are not current confirmed diagnoses; cite only supplied IDs; do not issue controls.",
                            "competition_boundary": "No upstream candidate changes; comparison is non-voting and all log matches are unverified textual cues.",
                            "reviewed_causal_chains": causal_review_for_llm,
                            "tool_investigation": tool_investigation,
                            "competing_hypothesis_review": (
                                (material_review_context or {}).get('competing_hypothesis_review') or {}
                            ),
                            "candidates": [{
                                "fault_id": item['fault_id'],
                                "supporting_and_opposing_log_cues": [
                                    {"evidence_id":x['evidence_id'],"assessment":x['assessment'],
                                     "excerpt":x['excerpt'][:260]} for x in
                                    (item.get('log_evidence_matrix') or {}).get('events',[])[:7]
                                ],
                                "reviewed_counterclaims": [
                                    {"edge_id": x['edge_id'], "polarity": x['polarity'],
                                     "claim_modality": x['claim_modality'], "quote": x['quote'][:420],
                                     "status": x['status']}
                                    for x in ((item.get('reference_retrieval') or {})
                                              .get('reviewed_counterevidence') or {}).get('items', [])[:5]
                                ],
                                "bidirectional_review": {
                                    "missing_expected_symptom_ids":
                                       (item.get('bidirectional_graph_review') or {}).get('missing_symptom_cue_ids', [])[:12],
                                    "expected_symptoms": [
                                        {"symptom_id": x['symptom_id'],
                                         "supporting_log_cue_ids": x['supporting_log_cue_ids'],
                                         "contradicting_log_cue_ids": x['contradicting_log_cue_ids']}
                                        for x in (item.get('bidirectional_graph_review') or {}).get('expected_symptoms', [])[:12]
                                    ],
                                },
                                "contrastive_reference_review": item.get('contrastive_reference_review') or {},
                                "candidate_references": [
                                    {"knowledge_id":x['knowledge_id'],"quote":x['quote'][:450],
                                     "knowledge_validation_status":x['knowledge_validation_status']}
                                    for x in (item.get('reference_retrieval') or {}).get('hits',[])[:4]
                                ],
                                "read_only_graph_paths": [
                                    {"graph_path_id":x.get('graph_path_id'),"relation_type":x['relation_type'],
                                     "evidence_definition_id":x['evidence_definition_id']}
                                    for x in item.get('graph_evidence_paths',[])[:8]
                                ],
                                "independently_confirmed_historical_cases": [
                                    {"case_id":x['case_id'],"verified_fault_id":x['verified_fault_id'],
                                     "relation_status":x['relation_status'],"truth_source_ref":x['truth_source_ref']}
                                    for x in (item.get('historical_confirmed_cases') or {}).get('hits',[])[:5]
                                ],
                            } for item in (material_review_context or {}).get('candidates',[])[:5]],
                        } if material_review_context else None,
                        "anything2ontology": {
                            "status": a2o.get("status"),
                            "active_kg_modified": False,
                            "review_material": ontology_review_material,
                        },
                    },
                )
                explanation = PlatformKnowledgeExplanation.model_validate(explanation.model_dump())
                _validate_explanation(
                    explanation, diagnosis_snapshot, local_hits, documents, ontology_review_material, material_review_context
                )
                generation = {
                    "status": "SUCCEEDED",
                    "explanation": explanation.model_dump(mode="json"),
                    "citation_reference_validation": "PASS",
                    "material_tool_investigation": tool_investigation,
                    "semantic_hallucination_assessment": "NOT_EVALUATED",
                    "backend_receipt": getattr(self.llm_backend, "last_call_metadata", {}),
                }
            except Exception as exc:
                generation = {
                    "status": "FAILED",
                    "explanation": None,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "semantic_hallucination_assessment": "NOT_EVALUATED",
                }
        timings["llm_generation"] = round(time.monotonic() - generation_started, 6)

        deterministic = _deterministic_grounded_explanation(
            diagnosis_snapshot, local_hits, documents, ontology_review_material
        )
        deterministic_payload = {
            "status": "SUCCEEDED",
            "mode": "DETERMINISTIC_REFERENCE_SYNTHESIS",
            "explanation": deterministic.model_dump(mode="json"),
            "citation_reference_validation": "PASS",
            "semantic_hallucination_assessment": "NOT_EVALUATED",
        }
        if generation.get("status") == "SUCCEEDED":
            final_explanation = {
                "status": "SUCCEEDED",
                "source": "LLM_GROUNDED",
                "explanation": generation["explanation"],
                "citation_reference_validation": generation.get("citation_reference_validation"),
                "semantic_hallucination_assessment": generation.get("semantic_hallucination_assessment"),
            }
        else:
            final_explanation = {
                "status": "SUCCEEDED",
                "source": "DETERMINISTIC_REFERENCE_SYNTHESIS",
                "explanation": deterministic_payload["explanation"],
                "citation_reference_validation": "PASS",
                "semantic_hallucination_assessment": "NOT_EVALUATED",
                "llm_status": generation.get("status"),
            }

        service_failures = [
            name
            for name, value in (("paperseek", paperseek), ("anything2ontology", a2o), ("llm", generation))
            if value.get("status") == "FAILED"
        ]
        if service_failures and self.config.require_configured_services:
            raise RuntimeError("required knowledge service failed: " + ",".join(service_failures))

        status = "SUCCEEDED"
        if service_failures:
            status = "PARTIAL"
        elif not document_paths and self.paperseek_client is None and self.llm_backend is None:
            status = "NOT_TRIGGERED"
        timings["total"] = round(sum(timings.values()), 6)
        result = {
            "schema_version": PLATFORM_KNOWLEDGE_SCHEMA_VERSION,
            "status": status,
            "completed_at": _now(),
            "diagnosis_frozen": True,
            "user_reported_observation_ledger": reported_observations,
            "diagnostic_candidate_can_be_overridden": False,
            "active_kg_modified": False,
            "retrieval_query_sha256": hashlib.sha256(retrieval_query.encode()).hexdigest(),
            "document_ingestion": ingestion,
            "local_retrieval": {
                "status": "SUCCEEDED" if chunks else "NO_PARSED_DOCUMENTS",
                "query_source": "FROZEN_DIAGNOSIS_PLUS_USER_EXPLANATION_REQUEST",
                "hits": local_hits,
                "quality_receipt": retrieval_quality,
                "strategy_receipt": retrieval_strategy,
            },
            "paperseek": paperseek,
            "anything2ontology": a2o,
            "generation": generation,
            "deterministic_synthesis": deterministic_payload,
            "final_explanation": final_explanation,
            "timings_s": timings,
            "claim_boundary": (
                "Knowledge enrichment is post-diagnosis. Uploaded text, live literature, ontology candidates "
                "and generated explanations cannot vote in BiRRF or modify the frozen diagnosis or Active KG."
            ),
        }
        (output_root / "knowledge_enrichment.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return result


__all__ = [
    "Anything2OntologyCLIAdapter",
    "PlatformKnowledgeClaim",
    "PlatformKnowledgeConfig",
    "PlatformKnowledgeEnrichmentService",
    "PlatformKnowledgeExplanation",
    "UploadedDocumentKnowledgeIndex",
]
