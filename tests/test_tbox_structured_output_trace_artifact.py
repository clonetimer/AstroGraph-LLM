import hashlib
import json
from pathlib import Path

from astrograph.diagnosis.structured_output import StructuredDiagnosisOutput


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT
    / "governance/evidence/research/document_corpus/tbox_structured_output_trace_r001"
)


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_tbox_path_survives_into_hash_bound_structured_output() -> None:
    receipt = json.loads((EVIDENCE / "audit.json").read_text(encoding="utf-8"))
    body = {key: value for key, value in receipt.items() if key != "audit_sha256"}
    assert receipt["audit_sha256"] == _canonical_sha(body)
    assert receipt["status"] == "PASS"
    assert receipt["matching_tbox_path_count"] > 0
    assert receipt["diagnostic_effect_claimed"] is False
    assert receipt["runtime_vote_allowed"] is False
    output_path = EVIDENCE / "structured_output.json"
    assert hashlib.sha256(output_path.read_bytes()).hexdigest() == receipt[
        "structured_output_file_sha256"
    ]
    output = StructuredDiagnosisOutput.model_validate_json(output_path.read_text(encoding="utf-8"))
    assert output.output_sha256 == receipt["structured_output_sha256"]
    summary = output.knowledge_and_retrieval_summary
    assert summary["ontology_path_references"]
    assert summary["ontology_asset_audits"]
    assert summary["path_use_attribution"] == "CONTEXT_AVAILABLE_NOT_CAUSAL_ATTRIBUTION"
    assert summary["diagnostic_effect_claimed"] is False
