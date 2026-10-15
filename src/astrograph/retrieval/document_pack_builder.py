"""Build governed multi-format design/FMEA/test document packs."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from astrograph.knowledge.construction import stage_governed_document_source

from .document_corpus_builder import sha256_file
from .document_governance import (
    ClaimEpistemicStatus,
    DocumentAuthority,
    DocumentReviewState,
    DocumentType,
    DocumentVisibilityClass,
    GovernedDocumentDescriptor,
)
from .structured_document import SectionFirstDocumentIngestor


def _json_list(value: str) -> list[object]:
    parsed = json.loads(value or "[]")
    if not isinstance(parsed, list):
        raise ValueError("coverage matrix list field is not a list")
    return parsed


def _coverage_rows(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    output: list[dict[str, object]] = []
    for row in rows:
        dataset_id = row.get("dataset_id") or row.get("\ufeffdataset_id")
        if not dataset_id:
            raise ValueError("coverage row is missing dataset_id")
        output.append(
            {
                "dataset_id": dataset_id,
                "fault_class_id": row["dataset_fault_class_id"],
                "physical_condition_ids": _json_list(row["physical_condition_ids"]),
                "generator_condition_summaries": _json_list(row["generator_condition_summaries"]),
                "runtime_fault_id": row["runtime_fault_id"] or None,
                "proposed_runtime_fault_id": row["proposed_runtime_fault_id"] or None,
                "component_id": row["component_id"] or None,
                "mechanism_id": row["mechanism_id"] or None,
                "expected_effect_pattern_ids": _json_list(row["expected_effect_pattern_ids"]),
                "evidence_definition_ids": _json_list(row["evidence_definition_ids"]),
                "coverage_status": row["coverage_status"],
                "review_status": row["review_status"],
            }
        )
    return output


def _markdown(rows: list[dict[str, object]], source_hashes: dict[str, str]) -> str:
    lines = [
        "# AstroGraph synthetic fault design, FMEA, and test handbook",
        "",
        "> Status: REVIEW_REQUIRED. This handbook is generated from simulator and existing coverage metadata.",
        "> Associations are candidate engineering knowledge, not independently confirmed causality.",
        "",
        "## Source identity",
        "",
    ]
    lines.extend(f"- {name}: `{digest}`" for name, digest in sorted(source_hashes.items()))
    for index, row in enumerate(rows, 1):
        fault = str(row["fault_class_id"])
        summaries = row["generator_condition_summaries"]
        descriptions = sorted(
            {
                str(item.get("description", "")).strip()
                for item in summaries
                if isinstance(item, dict) and item.get("description")
            }
        )
        parameters = sorted(
            {
                str(parameter)
                for item in summaries
                if isinstance(item, dict)
                for parameter in item.get("parameter_names", [])
            }
        )
        lines.extend(
            [
                "",
                f"## {index}. {fault}",
                "",
                "### Design-source candidate",
                "",
                f"- Dataset scope: `{row['dataset_id']}`",
                f"- Component candidate: `{row['component_id'] or 'UNRESOLVED'}`",
                f"- Runtime fault candidate: `{row['runtime_fault_id'] or row['proposed_runtime_fault_id'] or 'UNRESOLVED'}`",
                f"- Generator description: {'; '.join(descriptions) or 'not supplied'}",
                f"- Declared parameters: {', '.join(parameters) or 'none supplied'}",
                "- Epistemic status: `DESIGN_PROPOSAL`",
                "",
                "### FMEA candidate",
                "",
                "| Item | Candidate value | Status |",
                "|---|---|---|",
                f"| Failure class | `{fault}` | dataset vocabulary |",
                f"| Failure mechanism | `{row['mechanism_id'] or 'UNRESOLVED'}` | TBox association only |",
                f"| Effect patterns | `{', '.join(str(x) for x in row['expected_effect_pattern_ids']) or 'UNRESOLVED'}` | TBox association only |",
                f"| Existing evidence definitions | `{', '.join(str(x) for x in row['evidence_definition_ids']) or 'NONE'}` | {row['coverage_status']} |",
                "| Actual causal propagation | not asserted | independent evidence required |",
                "",
                "### Test-procedure candidate",
                "",
                "1. Select a train/development episode from a provenance group different from any historical case used.",
                "2. Admit only telemetry observations at or before the declared cutoff time.",
                "3. Execute registered evidence definitions; do not invent a threshold from this document.",
                "4. Record supporting, contradicting, missing, and inapplicable evidence separately.",
                "5. Compare the candidate against data-only under the frozen split and report abstention behavior.",
                "",
                "Formula attachment reference: `attachments/standardized_residual.tex`.",
                "Evidence-flow figure reference: `attachments/evidence_flow.svg`.",
            ]
        )
    return "\n".join(lines) + "\n"


def _paragraph(text: str, style: str | None = None) -> str:
    style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f'<w:p>{style_xml}<w:r><w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>'


def _table(rows: list[list[str]]) -> str:
    body = []
    for row in rows:
        cells = "".join(
            f"<w:tc><w:p><w:r><w:t>{escape(value)}</w:t></w:r></w:p></w:tc>"
            for value in row
        )
        body.append(f"<w:tr>{cells}</w:tr>")
    return "<w:tbl>" + "".join(body) + "</w:tbl>"


def _write_docx(
    path: Path,
    rows: list[dict[str, object]],
    *,
    title: str = "AstroGraph synthetic fault design, FMEA, and test handbook",
) -> None:
    blocks = [
        _paragraph(title, "Heading1"),
        _paragraph("REVIEW_REQUIRED; candidate engineering knowledge only."),
    ]
    for index, row in enumerate(rows, 1):
        fault = str(row["fault_class_id"])
        summaries = row["generator_condition_summaries"]
        descriptions = "; ".join(
            str(item.get("description", ""))
            for item in summaries
            if isinstance(item, dict) and item.get("description")
        ) or "not supplied"
        blocks.extend(
            [
                _paragraph(f"{index}. {fault}", "Heading2"),
                _paragraph("Design-source candidate", "Heading3"),
                _paragraph(f"Dataset scope: {row['dataset_id']}"),
                _paragraph(f"Generator description: {descriptions}"),
                _paragraph("Epistemic status: DESIGN_PROPOSAL"),
                _paragraph("FMEA candidate", "Heading3"),
                _table(
                    [
                        ["Item", "Candidate value", "Status"],
                        ["Failure class", fault, "dataset vocabulary"],
                        ["Failure mechanism", str(row["mechanism_id"] or "UNRESOLVED"), "TBox association only"],
                        ["Effect patterns", ", ".join(str(x) for x in row["expected_effect_pattern_ids"]) or "UNRESOLVED", "TBox association only"],
                        ["Actual causal propagation", "not asserted", "independent evidence required"],
                    ]
                ),
                _paragraph("Test-procedure candidate", "Heading3"),
                _paragraph("Use only causally visible telemetry and execute registered evidence definitions."),
                _paragraph("Do not invent thresholds from this document; preserve contradiction and abstention."),
            ]
        )
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>" + "".join(blocks) + "<w:sectPr/></w:body></w:document>"
    )
    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:pPr><w:outlineLvl w:val="0"/></w:pPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:pPr><w:outlineLvl w:val="1"/></w:pPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/><w:pPr><w:outlineLvl w:val="2"/></w:pPr></w:style>'
        "</w:styles>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
        "</Types>"
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        "</Relationships>"
    )
    document_rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        "</Relationships>"
    )
    def write_stable(archive: ZipFile, name: str, content: str) -> None:
        info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = ZIP_DEFLATED
        info.external_attr = 0o600 << 16
        archive.writestr(info, content.encode("utf-8"))

    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        write_stable(archive, "[Content_Types].xml", content_types)
        write_stable(archive, "_rels/.rels", root_rels)
        write_stable(archive, "word/document.xml", document)
        write_stable(archive, "word/styles.xml", styles)
        write_stable(archive, "word/_rels/document.xml.rels", document_rels)


def _write_pdf(path: Path, markdown: str) -> None:
    """Write a deterministic, dependency-free ASCII review rendering."""

    raw_lines = [re.sub(r"[`#|>*]", "", line).strip() for line in markdown.splitlines()]
    lines = [line.encode("ascii", "replace").decode("ascii")[:105] for line in raw_lines if line.strip()]
    pages = [lines[index:index + 48] for index in range(0, len(lines), 48)] or [["Empty"]]
    objects: list[bytes] = []
    page_ids = [4 + 2 * index for index in range(len(pages))]
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{item} 0 R" for item in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for page_number, page_lines in enumerate(pages):
        page_id = page_ids[page_number]
        content_id = page_id + 1
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>".encode()
        )
        commands = ["BT /F1 8 Tf 40 805 Td 10 TL"]
        for line in page_lines:
            safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            commands.append(f"({safe}) Tj T*")
        commands.append("ET")
        stream = "\n".join(commands).encode("ascii")
        objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
    output = bytearray(b"%PDF-1.4\n%AstroGraph\n")
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    path.write_bytes(output)


def build_fault_document_pack(
    *,
    coverage_matrix_path: Path,
    output_root: Path,
    generator_paths: dict[str, Path],
) -> dict[str, object]:
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"refusing to overwrite nonempty document pack: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    rows = _coverage_rows(coverage_matrix_path)
    source_hashes = {
        "coverage_matrix": sha256_file(coverage_matrix_path),
        **{name: sha256_file(path) for name, path in generator_paths.items()},
    }
    markdown = _markdown(rows, source_hashes)
    markdown_path = output_root / "fault_design_fmea_test_handbook.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    docx_path = output_root / "fault_design_fmea_test_handbook.docx"
    _write_docx(docx_path, rows)
    pdf_path = output_root / "fault_design_fmea_test_handbook.pdf"
    _write_pdf(pdf_path, markdown)
    facts_path = output_root / "canonical_fault_document_facts.json"
    facts_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = output_root / "fault_test_matrix.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["dataset_id", "fault_class_id", "coverage_status", "review_status"],
        )
        writer.writeheader()
        writer.writerows(
            {key: row[key] for key in writer.fieldnames} for row in rows
        )
    attachment_root = output_root / "attachments"
    attachment_root.mkdir()
    formula_path = attachment_root / "standardized_residual.tex"
    formula_path.write_text(
        "% Candidate analysis formula; calibration source must be supplied separately.\n"
        "z_t = \\frac{x_t - \\mu_{ref}}{\\max(\\sigma_{ref}, \\epsilon)}\n",
        encoding="utf-8",
    )
    svg_path = attachment_root / "evidence_flow.svg"
    svg_path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="180" viewBox="0 0 900 180">'
        '<rect width="900" height="180" fill="white"/>'
        '<g fill="#e8f0fe" stroke="#315b8a" stroke-width="2">'
        '<rect x="30" y="55" width="180" height="70" rx="8"/><rect x="260" y="55" width="180" height="70" rx="8"/>'
        '<rect x="490" y="55" width="180" height="70" rx="8"/><rect x="720" y="55" width="150" height="70" rx="8"/></g>'
        '<g font-family="sans-serif" font-size="16" text-anchor="middle" fill="#162a40">'
        '<text x="120" y="85">Causal telemetry</text><text x="120" y="106">to cutoff</text>'
        '<text x="350" y="85">Registered evidence</text><text x="350" y="106">definitions</text>'
        '<text x="580" y="85">Candidate KG</text><text x="580" y="106">association</text>'
        '<text x="795" y="85">Belief +</text><text x="795" y="106">abstention</text></g>'
        '<g stroke="#315b8a" stroke-width="3"><path d="M210 90H260"/><path d="M440 90H490"/><path d="M670 90H720"/></g>'
        '<text x="450" y="155" font-family="sans-serif" font-size="13" text-anchor="middle" fill="#7a2430">Illustrative design flow; not evidence of implementation or causality</text>'
        "</svg>\n",
        encoding="utf-8",
    )
    nominal_rows = [
        {
            "dataset_id": dataset_id,
            "fault_class_id": "nominal_operation",
            "generator_condition_summaries": [
                {
                    "description": (
                        "Nominal operation reference. Absence of a registered anomaly is not "
                        "proof of health; evidence completeness and abstention remain required."
                    ),
                    "parameter_names": [],
                }
            ],
            "runtime_fault_id": None,
            "proposed_runtime_fault_id": None,
            "component_id": "whole_spacecraft",
            "mechanism_id": None,
            "expected_effect_pattern_ids": [],
            "evidence_definition_ids": [],
            "coverage_status": "NOMINAL_REFERENCE_ONLY",
            "review_status": "HUMAN_REVIEW_REQUIRED",
        }
        for dataset_id in sorted({str(row["dataset_id"]) for row in rows})
    ]
    nominal_markdown = _markdown(nominal_rows, source_hashes).replace(
        "synthetic fault design, FMEA, and test handbook",
        "synthetic nominal operation reference",
        1,
    )
    nominal_md_path = output_root / "nominal_operation_reference.md"
    nominal_md_path.write_text(nominal_markdown, encoding="utf-8")
    nominal_docx_path = output_root / "nominal_operation_reference.docx"
    _write_docx(
        nominal_docx_path,
        nominal_rows,
        title="AstroGraph synthetic nominal operation reference",
    )
    nominal_pdf_path = output_root / "nominal_operation_reference.pdf"
    _write_pdf(nominal_pdf_path, nominal_markdown)
    nominal_facts_path = output_root / "canonical_nominal_document_facts.json"
    nominal_facts_path.write_text(
        json.dumps(nominal_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    docx_sha = sha256_file(docx_path)
    descriptor = GovernedDocumentDescriptor(
        document_id="astrograph:synthetic-fault-handbook:r001",
        document_type=DocumentType.DESIGN_PROPOSAL,
        visibility_class=DocumentVisibilityClass.DESIGN_PUBLIC,
        source_uri="corpus://design/fault_design_fmea_test_handbook.docx",
        source_sha256=docx_sha,
        review_state=DocumentReviewState.REVIEW_REQUIRED,
        authority=DocumentAuthority.NON_NORMATIVE,
        synthetic_from_simulation=True,
        runtime_vote_allowed=False,
    )
    staged = stage_governed_document_source(
        descriptor=descriptor,
        source_path=docx_path,
        independence_group="simulator-generator-and-existing-coverage",
        license_id="project-generated",
    )
    parser = SectionFirstDocumentIngestor()
    _, _, docx_audit = parser.parse_docx(docx_path)
    _, _, pdf_audit = parser.parse_pdf_layout(pdf_path)
    artifacts = {
        str(path.relative_to(output_root)): sha256_file(path)
        for path in (
            markdown_path, docx_path, pdf_path, facts_path, csv_path, formula_path, svg_path,
            nominal_md_path, nominal_docx_path, nominal_pdf_path, nominal_facts_path,
        )
    }
    return {
        "schema_version": "astrograph-fault-document-pack-receipt.v1",
        "status": "DRAFT_REVIEW_REQUIRED",
        "fault_class_count": len(rows),
        "nominal_reference_count": len(nominal_rows),
        "dataset_counts": {
            dataset: sum(row["dataset_id"] == dataset for row in rows)
            for dataset in sorted({str(row["dataset_id"]) for row in rows})
        },
        "source_hashes": source_hashes,
        "artifacts": artifacts,
        "document_descriptor": descriptor.model_dump(mode="json"),
        "kg_staging": {
            "staging_sha256": staged.staging_sha256,
            "source_tree_sha256": staged.source_tree.tree_sha256,
            "section_count": len(staged.source_tree.sections),
            "extraction_task_count": len(staged.extraction_tasks),
            "graph_scope": staged.graph_scope.value,
            "default_epistemic_status": ClaimEpistemicStatus.DESIGN_PROPOSAL.value,
            "publication_allowed": False,
            "runtime_vote_allowed": False,
        },
        "format_coverage": ["MARKDOWN", "DOCX", "PDF", "CSV", "JSON", "SVG", "LATEX"],
        "parse_audits": {
            "docx": {
                "source_sha256": docx_audit.source_sha256,
                "nonempty_paragraph_count": docx_audit.nonempty_paragraph_count,
                "heading_count": docx_audit.heading_count,
                "table_count": docx_audit.table_count,
                "table_row_count": docx_audit.table_row_count,
                "table_cell_count": docx_audit.table_cell_count,
                "image_count": docx_audit.image_count,
                "native_formula_count": docx_audit.native_formula_count,
                "warnings": docx_audit.warnings,
            },
            "pdf": pdf_audit.model_dump(mode="json"),
        },
        "claim_boundary": (
            "Synthetic multi-format development material generated from one simulator/coverage "
            "provenance cluster. It supplies document-processing workload and candidate associations, "
            "not independent causal confirmation, human review, runtime evidence, or Active KG publication."
        ),
    }


__all__ = ["build_fault_document_pack"]
