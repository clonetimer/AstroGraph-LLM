import hashlib

from astrograph.retrieval import PreReasoningRetrievalRequest, SectionFirstDocumentIngestor
from astrograph.retrieval.models import DiagnosticRetrievalQuery
from astrograph.retrieval.service import PreReasoningRetrievalService
from astrograph.retrieval.structured_document import DocumentElementKind


def test_markdown_ingestion_is_section_first_and_preserves_modalities(tmp_path):
    image = tmp_path / "curve.png"
    image.write_bytes(b"fake-png")
    markdown = """# ADCS
Residual diagnosis paragraph.

| signal | limit |
|---|---|
| r | 0.2 |

$$r(t)=y(t)-\\hat y(t)$$

![reviewed residual curve](curve.png)

# EPS
Battery capacity diagnosis paragraph.
"""
    result = SectionFirstDocumentIngestor(maximum_section_units=2).ingest_markdown(
        markdown, source_id="doc:manual", source_uri="manual.md", knowledge_version="v1",
        asset_root=tmp_path,
    )
    assert {tuple(chunk.section_path) for chunk in result.chunks} == {("ADCS",), ("EPS",)}
    assert all(len({tuple(e.section_path) for e in result.elements if e.source_locator in c.source_locators}) == 1 for c in result.chunks)
    assert {kind for chunk in result.chunks for kind in chunk.modalities} == {"TEXT", "TABLE", "FORMULA", "IMAGE"}
    image_element = next(item for item in result.elements if item.kind.value == "IMAGE")
    assert image_element.asset_sha256 == hashlib.sha256(b"fake-png").hexdigest()
    assert all(chunk.structure_sha256 for chunk in result.chunks)


def test_unreviewed_image_is_filtered_from_generation_context():
    markdown = "# ADCS\n\n![](missing.png)"
    result = SectionFirstDocumentIngestor().ingest_markdown(
        markdown, source_id="doc:unsafe", source_uri="unsafe.md", knowledge_version="v1"
    )
    request = PreReasoningRetrievalRequest(
        query=DiagnosticRetrievalQuery(query_id="q", episode_id="e", text="image"),
        chunks=result.chunks,
    )
    package = PreReasoningRetrievalService().retrieve(request)
    assert package.local_hits == []
    assert package.governance_filtered_chunk_count == 1
    assert result.warnings == ["IMAGE_REPRESENTATION_REVIEW_REQUIRED:3"]


def test_any_unreviewed_representation_is_filtered_from_generation_context():
    ingestor = SectionFirstDocumentIngestor()
    elements = [
        ingestor.parse_markdown("# Safe\n\nsource text")[0][0],
    ]
    unsafe = elements[0].model_copy(
        update={
            "element_id": "element:unsafe-table",
            "kind": DocumentElementKind.TABLE,
            "text_representation": "A    B\n1    2",
            "source_locator": "pdf#page:1:block:1",
            "representation_method": "poppler-pdftotext-layout",
            "representation_reviewed": False,
        }
    )
    result = ingestor.ingest_elements(
        source_id="doc:unsafe-table",
        source_uri="unsafe.pdf",
        source_sha256="a" * 64,
        knowledge_version="v1",
        elements=[unsafe],
        warnings=["PDF_TABLE_STRUCTURE_REVIEW_REQUIRED:1:1"],
    )
    assert result.chunks[0].review_status == "MACHINE_VALIDATED"
    assert result.chunks[0].content_safety_status == "REVIEW_REQUIRED"
    package = PreReasoningRetrievalService().retrieve(
        PreReasoningRetrievalRequest(
            query=DiagnosticRetrievalQuery(query_id="q", episode_id="e", text="table"),
            chunks=result.chunks,
        )
    )
    assert package.local_hits == []
    assert package.governance_filtered_chunk_count == 1


def test_single_long_semantic_element_is_not_split_by_character_count():
    paragraph = "residual " * 5000
    result = SectionFirstDocumentIngestor(maximum_section_units=1).ingest_markdown(
        "# Chapter\n" + paragraph,
        source_id="doc:long", source_uri="long.md", knowledge_version="v1",
    )
    assert len(result.chunks) == 1
    assert result.chunks[0].text == paragraph.strip()
