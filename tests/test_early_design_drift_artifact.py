import hashlib
import json
from pathlib import Path

from astrograph.retrieval import SectionFirstDocumentIngestor


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "智能故障检测软件_设计报告.docx"
ARTIFACT = (
    ROOT
    / "governance/evidence/research/document_corpus/early_design_report_r001/design_drift_candidates.json"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_design_drift_candidates_remain_source_and_code_traced() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert payload["status"] == "MACHINE_CANDIDATES_HUMAN_REVIEW_REQUIRED"
    assert payload["source_document_sha256"] == _sha(SOURCE)
    assert payload["candidate_count"] == len(payload["candidates"]) == 6
    assert len({item["drift_id"] for item in payload["candidates"]}) == 6
    assert payload["publication_allowed"] is False
    assert payload["active_kg_modified"] is False
    assert payload["runtime_modified"] is False

    elements, _, _ = SectionFirstDocumentIngestor().parse_docx(SOURCE)
    by_locator = {item.source_locator: item for item in elements}
    # This immutable r001 evidence describes the source supplied in post37.
    # Later source changes are checked by their own regression and governance
    # contract; do not rewrite historical evidence to match the working tree.
    source_manifest = json.loads((
        ROOT / "governance/baselines/post37-platform-source-manifest.json"
    ).read_text(encoding="utf-8"))["files"]
    for candidate in payload["candidates"]:
        element = by_locator[candidate["source_locator"]]
        assert candidate["source_span_sha256"] == hashlib.sha256(
            element.text_representation.encode("utf-8")
        ).hexdigest()
        assert all(
            term in element.text_representation
            for term in candidate["required_anchor_terms"]
        )
        assert candidate["review_state"] == "MACHINE_CANDIDATE_HUMAN_REVIEW_REQUIRED"
        assert candidate["active_kg_modified"] is False
        assert candidate["runtime_modified"] is False
        for evidence in candidate["current_code_evidence"]:
            assert evidence["sha256"] == source_manifest[evidence["path"]]
