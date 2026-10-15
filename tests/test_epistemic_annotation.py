from astrograph.retrieval import DocumentElement, DocumentElementKind
from astrograph.retrieval.epistemic_annotation import suggest_epistemic_span


def _element(text, kind=DocumentElementKind.TEXT):
    return DocumentElement(
        element_id="e:1",
        kind=kind,
        section_path=["Design"],
        text_representation=text,
        source_locator="paragraph:1",
        asset_uri="docx://image.png" if kind == DocumentElementKind.IMAGE else None,
        representation_method="fixture",
    )


def test_design_language_never_becomes_machine_source_fact():
    candidate = suggest_epistemic_span(_element("后续将采用大模型完成诊断。"))
    assert candidate.suggested_status.value == "DESIGN_PROPOSAL"
    assert candidate.runtime_evidence_eligible is False
    assert candidate.human_decision is None


def test_resource_number_is_estimate_and_image_is_illustrative():
    estimate = suggest_epistemic_span(_element("模型运行需要 8 GB 显存。"))
    image = suggest_epistemic_span(_element("诊断界面样例", DocumentElementKind.IMAGE))
    assert estimate.suggested_status.value == "ENGINEERING_ESTIMATE"
    assert image.suggested_status.value == "ILLUSTRATIVE"


def test_unmarked_statement_defaults_to_proposal_not_fact():
    candidate = suggest_epistemic_span(_element("系统由异常检测模块和检索模块组成。"))
    assert candidate.suggested_status.value == "DESIGN_PROPOSAL"
    assert candidate.trigger_codes == ["DOCUMENT_LEVEL_PROPOSAL_DEFAULT"]
