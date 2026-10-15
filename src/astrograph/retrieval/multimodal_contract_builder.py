"""Deterministic DOCX contract fixture with native table, image and OMML formula."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from astrograph.knowledge.construction import process_governed_document

from .document_governance import (
    DocumentAuthority,
    DocumentReviewState,
    DocumentType,
    DocumentVisibilityClass,
    GovernedDocumentDescriptor,
)


_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)
_FORMULA = "z_t=(x_t-mu_ref)/max(sigma_ref,epsilon)"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_stable_docx(
    path: Path, *, image_caption: str | None = "Illustrative evidence flow."
) -> None:
    caption_xml = (
        f"<w:r><w:t>{image_caption}</w:t></w:r>" if image_caption is not None else ""
    )
    document = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
 <w:body>
  <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Multimodal parser contract</w:t></w:r></w:p>
  <w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr><w:r><w:t>Equation</w:t></w:r></w:p>
  <w:p><m:oMath><m:r><m:t>{_FORMULA}</m:t></m:r></m:oMath></w:p>
  <w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr><w:r><w:t>Evidence table</w:t></w:r></w:p>
  <w:tbl>
   <w:tr><w:tc><w:p><w:r><w:t>Channel</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Unit</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Status</w:t></w:r></w:p></w:tc></w:tr>
   <w:tr><w:tc><w:p><w:r><w:t>rw.current</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>A</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>candidate</w:t></w:r></w:p></w:tc></w:tr>
  </w:tbl>
  <w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr><w:r><w:t>Diagram</w:t></w:r></w:p>
  <w:p>{caption_xml}<w:r><w:drawing><a:blip r:embed="rIdImage"/></w:drawing></w:r></w:p>
  <w:sectPr/>
 </w:body>
</w:document>'''
    styles = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
 <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:pPr><w:outlineLvl w:val="0"/></w:pPr></w:style>
 <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:pPr><w:outlineLvl w:val="1"/></w:pPr></w:style>
</w:styles>'''
    content_types = '''<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="xml" ContentType="application/xml"/>
 <Default Extension="png" ContentType="image/png"/>
 <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
 <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>'''
    root_rels = '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/package/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>'''
    document_rels = '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
 <Relationship Id="rIdImage" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/evidence.png"/>
</Relationships>'''

    def write(archive: ZipFile, name: str, data: bytes) -> None:
        info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = ZIP_DEFLATED
        info.external_attr = 0o600 << 16
        archive.writestr(info, data)

    with ZipFile(path, "w") as archive:
        for name, text in (
            ("[Content_Types].xml", content_types),
            ("_rels/.rels", root_rels),
            ("word/document.xml", document),
            ("word/styles.xml", styles),
            ("word/_rels/document.xml.rels", document_rels),
        ):
            write(archive, name, text.encode("utf-8"))
        write(archive, "word/media/evidence.png", _PNG)


def build_multimodal_docx_contract(*, output_root: str | Path) -> dict[str, object]:
    output_root = Path(output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("refusing to overwrite nonempty multimodal contract directory")
    output_root.mkdir(parents=True, exist_ok=True)
    docx = output_root / "multimodal_parser_contract.docx"
    _write_stable_docx(docx)
    source_sha = _sha(docx)
    descriptor = GovernedDocumentDescriptor(
        document_id="astrograph:multimodal-parser-contract:r001",
        document_type=DocumentType.DESIGN_PROPOSAL,
        visibility_class=DocumentVisibilityClass.DESIGN_PUBLIC,
        source_uri="corpus://contracts/multimodal_parser_contract.docx",
        source_sha256=source_sha,
        review_state=DocumentReviewState.REVIEW_REQUIRED,
        authority=DocumentAuthority.NON_NORMATIVE,
        synthetic_from_simulation=False,
        runtime_vote_allowed=False,
    )
    processed = process_governed_document(
        descriptor=descriptor,
        source_path=docx,
        independence_group="astrograph-parser-contract-r001",
        license_id="project-generated",
    )
    elements = processed.retrieval_ingestion.elements
    formula = [item for item in elements if item.kind.value == "FORMULA"]
    tables = [item for item in elements if item.kind.value == "TABLE"]
    images = [item for item in elements if item.kind.value == "IMAGE"]
    gold_candidate = {
        "schema_version": "astrograph-multimodal-parse-gold-candidate.v1",
        "review_state": "MACHINE_GOLD_CANDIDATE_HUMAN_REVIEW_REQUIRED",
        "expected_formula_text": _FORMULA,
        "expected_table_text": "| Channel | Unit | Status |\n| rw.current | A | candidate |",
        "expected_image_caption": "Illustrative evidence flow.",
        "expected_image_sha256": hashlib.sha256(_PNG).hexdigest(),
        "parsed_formula_match": len(formula) == 1 and formula[0].text_representation == _FORMULA,
        "parsed_table_match": len(tables) == 1 and tables[0].text_representation == "| Channel | Unit | Status |\n| rw.current | A | candidate |",
        "parsed_image_match": len(images) == 1 and images[0].asset_sha256 == hashlib.sha256(_PNG).hexdigest(),
        "human_review_complete": False,
    }
    gold_path = output_root / "machine_gold_candidate.json"
    gold_path.write_text(json.dumps(gold_candidate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "schema_version": "astrograph-multimodal-docx-contract-receipt.v1",
        "status": "MACHINE_VALIDATED_HUMAN_GOLD_PENDING",
        "source_sha256": source_sha,
        "machine_gold_candidate_sha256": _sha(gold_path),
        "element_count": len(elements),
        "modality_counts": {
            kind: sum(item.kind.value == kind for item in elements)
            for kind in ("TEXT", "TABLE", "FORMULA", "IMAGE")
        },
        "crosswalk_count": len(processed.crosswalk),
        "crosswalk_complete": len(processed.crosswalk) == len(elements),
        "formula_text_preserved": gold_candidate["parsed_formula_match"],
        "table_cells_preserved_as_rows": gold_candidate["parsed_table_match"],
        "image_hash_preserved": gold_candidate["parsed_image_match"],
        "warnings": processed.retrieval_ingestion.warnings,
        "publication_allowed": False,
        "runtime_vote_allowed": False,
        "human_gold_complete": False,
        "claim_boundary": "Deterministic parser fixture and machine gold candidate only; human multimodal gold remains pending.",
    }


__all__ = ["build_multimodal_docx_contract"]
