"""Section-first, modality-preserving ingestion for governed RAG/OAG.

Parsers such as Docling may emit :class:`DocumentElement` records.  This layer
binds their provenance and chunks by document structure.  It never treats an
image caption/OCR result as verified source truth.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from collections import Counter
from enum import Enum
from pathlib import Path
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import GovernedTextChunk, RetrievalLane
from .document_governance import GovernedDocumentDescriptor


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class DocumentElementKind(str, Enum):
    TEXT = "TEXT"
    TABLE = "TABLE"
    FORMULA = "FORMULA"
    IMAGE = "IMAGE"


class DocumentElement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    element_id: str
    kind: DocumentElementKind
    section_path: list[str] = Field(min_length=1)
    text_representation: str = Field(min_length=1)
    source_locator: str = Field(min_length=1)
    asset_uri: str | None = None
    asset_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    representation_method: str = Field(min_length=1)
    representation_reviewed: bool = False
    style_id: str | None = None
    style_name: str | None = None

    @model_validator(mode="after")
    def image_is_traceable(self) -> "DocumentElement":
        if self.kind == DocumentElementKind.IMAGE and not self.asset_uri:
            raise ValueError("image elements require asset_uri")
        return self


class StructuredDocumentIngestion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-structured-document-ingestion.v1"
    source_id: str
    source_uri: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    elements: list[DocumentElement] = Field(min_length=1)
    chunks: list[GovernedTextChunk] = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)
    document_descriptor: GovernedDocumentDescriptor | None = None
    ingestion_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def identity_matches(self) -> "StructuredDocumentIngestion":
        expected = _sha(self.model_dump(mode="json", exclude={"ingestion_sha256"}))
        if expected != self.ingestion_sha256:
            raise ValueError("ingestion_sha256 mismatch")
        return self


class DocxParseAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-docx-parse-audit.v1"
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    nonempty_paragraph_count: int = Field(ge=0)
    heading_count: int = Field(ge=0)
    headings: list[dict[str, object]] = Field(default_factory=list)
    table_count: int = Field(ge=0)
    table_row_count: int = Field(ge=0)
    table_cell_count: int = Field(ge=0)
    image_count: int = Field(ge=0)
    media_sha256: dict[str, str] = Field(default_factory=dict)
    native_formula_count: int = Field(ge=0)
    style_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class PdfParseAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-pdf-parse-audit.v1"
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parser_id: str = Field(min_length=1)
    page_count: int = Field(ge=1)
    text_page_count: int = Field(ge=0)
    text_block_count: int = Field(ge=0)
    table_like_block_count: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)


class SectionFirstDocumentIngestor:
    """Build retrieval chunks without crossing section boundaries.

    `maximum_section_units` bounds semantic elements, never characters.  A long
    paragraph/table/formula/image remains atomic and is flagged for review.
    """

    _heading = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
    _image = re.compile(r"!\[([^]]*)\]\(([^)\s]+)(?:\s+[\"'].*?[\"'])?\)")

    _w = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    _a = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
    _r = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    _pr = "{http://schemas.openxmlformats.org/package/2006/relationships}"
    _m = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"

    def __init__(self, *, maximum_section_units: int = 24,
                 maximum_chunk_chars: int | None = None,
                 preserve_modality_boundaries: bool = False,
                 split_long_text: bool = False) -> None:
        if maximum_section_units < 1:
            raise ValueError("maximum_section_units must be positive")
        if maximum_chunk_chars is not None and not 256 <= maximum_chunk_chars <= 100_000:
            raise ValueError("maximum_chunk_chars must be 256..100000")
        if split_long_text and maximum_chunk_chars is None:
            raise ValueError("split_long_text requires a character budget")
        self.maximum_section_units = maximum_section_units
        self.maximum_chunk_chars = maximum_chunk_chars
        self.preserve_modality_boundaries = preserve_modality_boundaries
        self.split_long_text = split_long_text

    @staticmethod
    def _split_text_element(element: DocumentElement, budget: int) -> list[DocumentElement]:
        """Exact-position subspans, never generated summaries or rewritten excerpts."""
        text = element.text_representation
        if len(text) <= budget or element.kind != DocumentElementKind.TEXT:
            return [element]
        parts: list[DocumentElement] = []
        start = 0
        while start < len(text):
            limit = min(len(text), start + budget)
            end = limit
            if limit < len(text):
                # Prefer sentence or whitespace boundaries in the latter half;
                # fall back to a hard, lossless cut for unbroken telemetry text.
                for j in range(limit, start + budget // 2, -1):
                    if text[j - 1] in "\n。！？.!?；; " :
                        end = j
                        break
            piece = text[start:end]
            if piece:
                parts.append(element.model_copy(update={
                    "element_id": f"{element.element_id}:part:{len(parts)+1:04d}",
                    "text_representation": piece,
                    "source_locator": f"{element.source_locator}#chars:{start}-{end}",
                }))
            start = end
        if "".join(part.text_representation for part in parts) != text:
            raise ValueError("text splitting lost source bytes")
        return parts

    def parse_markdown(self, text: str, *, asset_root: str | Path | None = None) -> tuple[list[DocumentElement], list[str]]:
        lines = text.splitlines()
        headings: list[str] = []
        elements: list[DocumentElement] = []
        warnings: list[str] = []
        block: list[str] = []
        ordinal = 0

        def emit(raw: str, kind: DocumentElementKind, locator: str, **extra: object) -> None:
            nonlocal ordinal
            content = raw.strip()
            if not content:
                return
            ordinal += 1
            elements.append(DocumentElement(
                element_id=f"element:{ordinal:05d}", kind=kind,
                section_path=list(headings) or ["Document"], text_representation=content,
                source_locator=locator, representation_method="markdown-native",
                # The representation is the source text itself rather than an
                # OCR/caption/layout reconstruction.  Images are handled by the
                # separate branch below and remain review-gated.
                representation_reviewed=True, **extra,
            ))

        def flush(end_line: int) -> None:
            nonlocal block
            if not block:
                return
            raw = "\n".join(block)
            stripped = raw.strip()
            start = end_line - len(block) + 1
            kind = DocumentElementKind.TEXT
            if stripped.startswith("|") and "\n|" in stripped:
                kind = DocumentElementKind.TABLE
            elif (stripped.startswith("$$") and stripped.endswith("$$")) or (
                stripped.startswith("\\[") and stripped.endswith("\\]")
            ):
                kind = DocumentElementKind.FORMULA
            emit(raw, kind, f"lines:{start}-{end_line}")
            block = []

        for index, line in enumerate(lines, start=1):
            heading = self._heading.match(line)
            if heading:
                flush(index - 1)
                level, title = len(heading.group(1)), heading.group(2).strip()
                headings = headings[: level - 1] + [title]
                continue
            image = self._image.fullmatch(line.strip())
            if image:
                flush(index - 1)
                alt, uri = image.groups()
                path = Path(asset_root, uri) if asset_root and not re.match(r"^[a-z]+://", uri) else None
                asset_hash = hashlib.sha256(path.read_bytes()).hexdigest() if path and path.is_file() else None
                reviewed = bool(alt.strip()) and asset_hash is not None
                if not reviewed:
                    warnings.append(f"IMAGE_REPRESENTATION_REVIEW_REQUIRED:{index}")
                ordinal += 1
                elements.append(DocumentElement(
                    element_id=f"element:{ordinal:05d}", kind=DocumentElementKind.IMAGE,
                    section_path=list(headings) or ["Document"],
                    text_representation=alt.strip() or "[image without reviewed description]",
                    source_locator=f"line:{index}", asset_uri=uri, asset_sha256=asset_hash,
                    representation_method="markdown-alt", representation_reviewed=reviewed,
                ))
                continue
            if not line.strip():
                flush(index - 1)
            else:
                block.append(line)
        flush(len(lines))
        if not elements:
            raise ValueError("document contains no retrievable elements")
        return elements, warnings

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @classmethod
    def _docx_styles(cls, archive: ZipFile) -> dict[str, tuple[str | None, int | None]]:
        if "word/styles.xml" not in archive.namelist():
            return {}
        root = ElementTree.fromstring(archive.read("word/styles.xml"))
        result: dict[str, tuple[str | None, int | None]] = {}
        for style in root.findall(cls._w + "style"):
            style_id = style.get(cls._w + "styleId")
            if not style_id:
                continue
            name_node = style.find(cls._w + "name")
            outline_node = style.find(".//" + cls._w + "outlineLvl")
            name = name_node.get(cls._w + "val") if name_node is not None else None
            outline = None
            if outline_node is not None:
                try:
                    outline = int(outline_node.get(cls._w + "val", ""))
                except ValueError:
                    outline = None
            result[style_id] = (name, outline)
        return result

    @classmethod
    def _docx_relationships(cls, archive: ZipFile) -> tuple[dict[str, str], list[str]]:
        name = "word/_rels/document.xml.rels"
        if name not in archive.namelist():
            return {}, []
        root = ElementTree.fromstring(archive.read(name))
        relationships: dict[str, str] = {}
        warnings: list[str] = []
        for relation in root.findall(cls._pr + "Relationship"):
            relation_id = relation.get("Id")
            target = relation.get("Target")
            if relation.get("TargetMode") == "External":
                warnings.append(f"DOCX_EXTERNAL_RELATIONSHIP_BLOCKED:{relation_id or 'UNKNOWN'}")
                continue
            if relation_id and target:
                relationships[relation_id] = target
        return relationships, warnings

    def parse_docx(
        self,
        path: str | Path,
        *,
        maximum_archive_entries: int = 10_000,
        maximum_uncompressed_bytes: int = 256 * 1024 * 1024,
    ) -> tuple[list[DocumentElement], list[str], DocxParseAudit]:
        """Parse DOCX natively without flattening it through PDF or Markdown."""

        source_path = Path(path).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source_sha = self._file_sha256(source_path)
        try:
            archive = ZipFile(source_path)
        except BadZipFile as exc:
            raise ValueError("DOCX is not a valid OOXML archive") from exc
        with archive:
            infos = archive.infolist()
            if len(infos) > maximum_archive_entries:
                raise ValueError("DOCX archive entry limit exceeded")
            if sum(item.file_size for item in infos) > maximum_uncompressed_bytes:
                raise ValueError("DOCX uncompressed size limit exceeded")
            names = set(archive.namelist())
            if "word/document.xml" not in names:
                raise ValueError("DOCX is missing word/document.xml")
            styles = self._docx_styles(archive)
            relationships, warnings = self._docx_relationships(archive)
            root = ElementTree.fromstring(archive.read("word/document.xml"))
            body = root.find(self._w + "body")
            if body is None:
                raise ValueError("DOCX document body is missing")

            heading_stack: list[tuple[int, str]] = []
            heading_records: list[dict[str, object]] = []
            elements: list[DocumentElement] = []
            style_counts: Counter[str] = Counter()
            paragraph_count = 0
            table_count = 0
            table_rows = 0
            table_cells = 0
            image_count = 0
            formula_count = 0
            ordinal = 0
            media_hashes: dict[str, str] = {}

            for block_index, child in enumerate(body, start=1):
                if child.tag == self._w + "p":
                    text = "".join(node.text or "" for node in child.iter(self._w + "t")).strip()
                    style_node = child.find(".//" + self._w + "pStyle")
                    style_id = style_node.get(self._w + "val") if style_node is not None else None
                    style_name, outline = styles.get(style_id or "", (None, None))
                    if text:
                        paragraph_count += 1
                        style_counts[style_id or "(none)"] += 1
                    if text and outline is not None:
                        level = max(1, min(outline + 1, 9))
                        while heading_stack and heading_stack[-1][0] >= level:
                            heading_stack.pop()
                        heading_stack.append((level, text))
                        heading_records.append(
                            {
                                "level": level,
                                "text": text,
                                "style_id": style_id,
                                "style_name": style_name,
                                "source_locator": f"word/document.xml#paragraph:{block_index}",
                            }
                        )
                        continue

                    blips = list(child.iter(self._a + "blip"))
                    for image_index, blip in enumerate(blips, start=1):
                        relation_id = blip.get(self._r + "embed")
                        target = relationships.get(relation_id or "")
                        if not target:
                            warnings.append(
                                f"DOCX_IMAGE_RELATIONSHIP_MISSING:{block_index}:{image_index}"
                            )
                            continue
                        archive_name = "word/" + target.lstrip("/")
                        if archive_name not in names:
                            warnings.append(f"DOCX_IMAGE_ASSET_MISSING:{archive_name}")
                            continue
                        data = archive.read(archive_name)
                        asset_hash = hashlib.sha256(data).hexdigest()
                        media_hashes[archive_name] = asset_hash
                        image_count += 1
                        ordinal += 1
                        elements.append(
                            DocumentElement(
                                element_id=f"element:{ordinal:05d}",
                                kind=DocumentElementKind.IMAGE,
                                section_path=[item[1] for item in heading_stack] or ["Document"],
                                text_representation=(
                                    text or f"[DOCX image awaiting reviewed description: {Path(target).name}]"
                                ),
                                source_locator=(
                                    f"word/document.xml#paragraph:{block_index}:image:{image_index}"
                                ),
                                asset_uri=f"docx://{archive_name}",
                                asset_sha256=asset_hash,
                                representation_method="docx-native-drawing",
                                representation_reviewed=False,
                                style_id=style_id,
                                style_name=style_name,
                            )
                        )
                        warnings.append(
                            f"IMAGE_REPRESENTATION_REVIEW_REQUIRED:{block_index}:{image_index}"
                        )

                    formulas = list(child.iter(self._m + "oMath"))
                    for formula_index, formula in enumerate(formulas, start=1):
                        formula_text = "".join(
                            node.text or "" for node in formula.iter(self._m + "t")
                        ).strip()
                        if not formula_text:
                            formula_text = "".join(
                                node.text or "" for node in formula.iter(self._w + "t")
                            ).strip()
                        formula_count += 1
                        ordinal += 1
                        elements.append(
                            DocumentElement(
                                element_id=f"element:{ordinal:05d}",
                                kind=DocumentElementKind.FORMULA,
                                section_path=[item[1] for item in heading_stack] or ["Document"],
                                text_representation=formula_text or "[OMML formula]",
                                source_locator=(
                                    f"word/document.xml#paragraph:{block_index}:formula:{formula_index}"
                                ),
                                representation_method="docx-native-omml",
                                representation_reviewed=False,
                                style_id=style_id,
                                style_name=style_name,
                            )
                        )
                        warnings.append(
                            f"FORMULA_REPRESENTATION_REVIEW_REQUIRED:{block_index}:{formula_index}"
                        )

                    if text:
                        ordinal += 1
                        elements.append(
                            DocumentElement(
                                element_id=f"element:{ordinal:05d}",
                                kind=DocumentElementKind.TEXT,
                                section_path=[item[1] for item in heading_stack] or ["Document"],
                                text_representation=text,
                                source_locator=f"word/document.xml#paragraph:{block_index}",
                                representation_method="docx-native-paragraph",
                                representation_reviewed=True,
                                style_id=style_id,
                                style_name=style_name,
                            )
                        )
                elif child.tag == self._w + "tbl":
                    table_count += 1
                    rows: list[list[str]] = []
                    for row in child.findall(self._w + "tr"):
                        values: list[str] = []
                        for cell in row.findall(self._w + "tc"):
                            value = " ".join(
                                (node.text or "").strip()
                                for node in cell.iter(self._w + "t")
                                if (node.text or "").strip()
                            )
                            values.append(value)
                        rows.append(values)
                    table_rows += len(rows)
                    table_cells += sum(len(row) for row in rows)
                    width = max((len(row) for row in rows), default=0)
                    if width:
                        normalized = [row + [""] * (width - len(row)) for row in rows]
                        table_text = "\n".join(
                            "| " + " | ".join(value.replace("|", "\\|") for value in row) + " |"
                            for row in normalized
                        )
                        ordinal += 1
                        elements.append(
                            DocumentElement(
                                element_id=f"element:{ordinal:05d}",
                                kind=DocumentElementKind.TABLE,
                                section_path=[item[1] for item in heading_stack] or ["Document"],
                                text_representation=table_text,
                                source_locator=f"word/document.xml#table:{table_count}",
                                representation_method="docx-native-table",
                                representation_reviewed=True,
                            )
                        )

            if not elements:
                raise ValueError("DOCX contains no retrievable elements")
            warnings = sorted(set(warnings))
            audit = DocxParseAudit(
                source_sha256=source_sha,
                nonempty_paragraph_count=paragraph_count,
                heading_count=len(heading_records),
                headings=heading_records,
                table_count=table_count,
                table_row_count=table_rows,
                table_cell_count=table_cells,
                image_count=image_count,
                media_sha256=dict(sorted(media_hashes.items())),
                native_formula_count=formula_count,
                style_counts=dict(sorted(style_counts.items())),
                warnings=warnings,
            )
            return elements, warnings, audit

    def parse_pdf_layout(
        self,
        path: str | Path,
        *,
        timeout_s: float = 30.0,
        maximum_bytes: int = 256 * 1024 * 1024,
    ) -> tuple[list[DocumentElement], list[str], PdfParseAudit]:
        """Extract page-bounded PDF text with Poppler; never invent OCR content."""

        source_path = Path(path).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        if source_path.stat().st_size > maximum_bytes:
            raise ValueError("PDF size limit exceeded")
        pdftotext = shutil.which("pdftotext")
        pdfinfo = shutil.which("pdfinfo")
        if pdftotext is None or pdfinfo is None:
            raise RuntimeError("Poppler pdftotext and pdfinfo are required for PDF parsing")
        try:
            info = subprocess.run(
                [pdfinfo, str(source_path)],
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            extracted = subprocess.run(
                [pdftotext, "-layout", "-enc", "UTF-8", str(source_path), "-"],
                check=True,
                capture_output=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("PDF parser timed out") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or b"") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            raise ValueError(f"Poppler rejected PDF: {str(detail)[:500]}") from exc
        match = re.search(r"(?m)^Pages:\s+(\d+)\s*$", info.stdout)
        if match is None:
            raise ValueError("pdfinfo did not report a page count")
        page_count = int(match.group(1))
        raw = extracted.stdout.decode("utf-8", errors="replace")
        raw_pages = raw.split("\f")
        if raw_pages and not raw_pages[-1].strip():
            raw_pages.pop()
        warnings: list[str] = []
        elements: list[DocumentElement] = []
        table_like = 0
        ordinal = 0
        text_pages = 0
        for page_number in range(1, page_count + 1):
            page = raw_pages[page_number - 1] if page_number <= len(raw_pages) else ""
            if not page.strip():
                warnings.append(f"PDF_PAGE_TEXT_EMPTY_OCR_REQUIRED:{page_number}")
                continue
            text_pages += 1
            blocks = [item.strip() for item in re.split(r"\n\s*\n", page) if item.strip()]
            for block_index, block in enumerate(blocks, 1):
                lines = block.splitlines()
                looks_tabular = len(lines) >= 2 and sum(bool(re.search(r"\S\s{2,}\S", line)) for line in lines) >= 2
                kind = DocumentElementKind.TABLE if looks_tabular else DocumentElementKind.TEXT
                if looks_tabular:
                    table_like += 1
                    warnings.append(
                        f"PDF_TABLE_STRUCTURE_REVIEW_REQUIRED:{page_number}:{block_index}"
                    )
                ordinal += 1
                elements.append(
                    DocumentElement(
                        element_id=f"element:{ordinal:05d}",
                        kind=kind,
                        section_path=[f"Page {page_number}"],
                        text_representation=block,
                        source_locator=f"pdf#page:{page_number}:block:{block_index}",
                        representation_method="poppler-pdftotext-layout",
                        representation_reviewed=not looks_tabular,
                    )
                )
        # Raster assets are listed, NOT auto-captioned or treated as ground
        # truth.  PDF vector plots and scanned formulae may still need Docling
        # or independently reviewed multimodal/OCR processing.
        image_tool = shutil.which("pdfimages")
        if image_tool is not None:
            try:
                listing = subprocess.run(
                    [image_tool, "-list", str(source_path)], capture_output=True,
                    text=True, timeout=min(timeout_s, 10.0), check=True,
                ).stdout
                for row in listing.splitlines():
                    found = re.match(r"^\s*(\d+)\s+(\d+)\s+(image|stencil|mask|smask)\s+", row)
                    if not found:
                        continue
                    page_no, image_no = int(found.group(1)), int(found.group(2))
                    ordinal += 1
                    locator = f"pdf#page:{page_no}:image:{image_no}"
                    elements.append(DocumentElement(
                        element_id=f"element:{ordinal:05d}",
                        kind=DocumentElementKind.IMAGE,
                        section_path=[f"Page {page_no}"],
                        text_representation="[PDF raster image: unreviewed image content]",
                        source_locator=locator, asset_uri=f"{source_path.name}#{locator}",
                        representation_method="poppler-pdfimages-metadata-only",
                        representation_reviewed=False,
                    ))
                    warnings.append(f"PDF_IMAGE_REQUIRES_VISUAL_REVIEW:{page_no}:{image_no}")
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError):
                warnings.append("PDF_IMAGE_ENUMERATION_UNAVAILABLE")
        if not any(item.kind in {DocumentElementKind.TEXT, DocumentElementKind.TABLE}
                   for item in elements):
            raise ValueError(
                "PDF contains no extractable text; OCR output must be supplied as a separate reviewed representation"
            )
        warnings = sorted(set(warnings))
        audit = PdfParseAudit(
            source_sha256=self._file_sha256(source_path),
            parser_id="poppler-pdftotext-layout",
            page_count=page_count,
            text_page_count=text_pages,
            text_block_count=len(elements),
            table_like_block_count=table_like,
            warnings=warnings,
        )
        return elements, warnings, audit

    def ingest_elements(
        self, *, source_id: str, source_uri: str, source_sha256: str,
        knowledge_version: str, elements: list[DocumentElement], warnings: list[str] | None = None,
        document_descriptor: GovernedDocumentDescriptor | None = None,
    ) -> StructuredDocumentIngestion:
        if (
            document_descriptor is not None
            and document_descriptor.source_sha256 != source_sha256
        ):
            raise ValueError("document descriptor source_sha256 mismatch")
        # Keep the original per-kind semantics. Images, formulae and tables are
        # atomic and never broken across chunks.  Only TEXT may be losslessly
        # split, preserving character offsets into the exact extracted source.
        safe_elements: list[DocumentElement] = []
        extra_warnings = list(warnings or [])
        for element in elements:
            if self.maximum_chunk_chars and self.split_long_text:
                safe_elements.extend(self._split_text_element(element, self.maximum_chunk_chars))
            else:
                safe_elements.append(element)
            if (self.maximum_chunk_chars and element.kind != DocumentElementKind.TEXT
                    and len(element.text_representation) > self.maximum_chunk_chars):
                extra_warnings.append(
                    f"OVERSIZED_ATOMIC_{element.kind.value}_REVIEW_REQUIRED:{element.source_locator}"
                )
        elements = safe_elements
        groups: list[list[DocumentElement]] = []
        current: list[DocumentElement] = []
        current_chars = 0
        for element in elements:
            new_size = len(element.text_representation) + (2 if current else 0)
            boundary = bool(current) and (
                element.section_path != current[0].section_path
                or len(current) >= self.maximum_section_units
                or (self.maximum_chunk_chars is not None
                    and current_chars + new_size > self.maximum_chunk_chars)
                or (self.preserve_modality_boundaries
                    and (element.kind != DocumentElementKind.TEXT
                         or current[0].kind != DocumentElementKind.TEXT))
            )
            if boundary:
                groups.append(current)
                current = []
                current_chars = 0
                new_size = len(element.text_representation)
            current.append(element)
            current_chars += new_size
        if current:
            groups.append(current)
        chunks: list[GovernedTextChunk] = []
        for index, group in enumerate(groups, start=1):
            text = "\n\n".join(item.text_representation for item in group)
            section_path = group[0].section_path
            section_id = "section:" + _sha(section_path)[:16]
            structured = {
                "text": text, "section_id": section_id, "section_path": section_path,
                "modalities": sorted({item.kind.value for item in group}),
                "source_locators": [item.source_locator for item in group],
            }
            # Every derived representation must pass its own review gate.  In
            # particular, an unreviewed OMML rendering or PDF table layout is
            # no safer than an unreviewed image caption.
            safe = all(item.representation_reviewed for item in group)
            chunks.append(GovernedTextChunk(
                chunk_id=f"{source_id}:section:{index:04d}", source_id=source_id, source_uri=source_uri,
                source_sha256=source_sha256, chunk_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                structure_sha256=_sha(structured), knowledge_version=knowledge_version, text=text,
                section_id=section_id, section_path=section_path,
                modalities=structured["modalities"], source_locators=structured["source_locators"],
                review_status="HUMAN_REVIEWED" if safe else "MACHINE_VALIDATED",
                content_safety_status="PASS" if safe else "REVIEW_REQUIRED",
                lane=RetrievalLane.LOCAL_DOCUMENT,
                document_descriptor=document_descriptor,
            ))
        payload = {
            "schema_version": "astrograph-structured-document-ingestion.v1", "source_id": source_id,
            "source_uri": source_uri, "source_sha256": source_sha256, "elements": elements,
            "chunks": chunks, "warnings": sorted(set(extra_warnings)),
            "document_descriptor": document_descriptor,
        }
        serial = {
            key: [x.model_dump(mode="json") for x in value]
            if key in {"elements", "chunks"}
            else value.model_dump(mode="json")
            if isinstance(value, BaseModel)
            else value
            for key, value in payload.items()
        }
        return StructuredDocumentIngestion(**payload, ingestion_sha256=_sha(serial))

    def ingest_markdown(
        self, text: str, *, source_id: str, source_uri: str, knowledge_version: str,
        asset_root: str | Path | None = None,
        document_descriptor: GovernedDocumentDescriptor | None = None,
        source_sha256: str | None = None,
    ) -> StructuredDocumentIngestion:
        source_sha = source_sha256 or hashlib.sha256(text.encode("utf-8")).hexdigest()
        elements, warnings = self.parse_markdown(text, asset_root=asset_root)
        return self.ingest_elements(
            source_id=source_id, source_uri=source_uri, source_sha256=source_sha,
            knowledge_version=knowledge_version, elements=elements, warnings=warnings,
            document_descriptor=document_descriptor,
        )

    def ingest_docx(
        self,
        path: str | Path,
        *,
        source_id: str,
        knowledge_version: str,
        document_descriptor: GovernedDocumentDescriptor | None = None,
    ) -> tuple[StructuredDocumentIngestion, DocxParseAudit]:
        source_path = Path(path).resolve()
        elements, warnings, audit = self.parse_docx(source_path)
        ingestion = self.ingest_elements(
            source_id=source_id,
            source_uri=str(source_path),
            source_sha256=audit.source_sha256,
            knowledge_version=knowledge_version,
            elements=elements,
            warnings=warnings,
            document_descriptor=document_descriptor,
        )
        return ingestion, audit

    def ingest_pdf(
        self,
        path: str | Path,
        *,
        source_id: str,
        knowledge_version: str,
        document_descriptor: GovernedDocumentDescriptor | None = None,
    ) -> tuple[StructuredDocumentIngestion, PdfParseAudit]:
        source_path = Path(path).resolve()
        elements, warnings, audit = self.parse_pdf_layout(source_path)
        ingestion = self.ingest_elements(
            source_id=source_id,
            source_uri=str(source_path),
            source_sha256=audit.source_sha256,
            knowledge_version=knowledge_version,
            elements=elements,
            warnings=warnings,
            document_descriptor=document_descriptor,
        )
        return ingestion, audit


__all__ = [
    "DocumentElement",
    "DocumentElementKind",
    "DocxParseAudit",
    "PdfParseAudit",
    "SectionFirstDocumentIngestor",
    "StructuredDocumentIngestion",
]
