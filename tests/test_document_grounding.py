from astrograph.agents.knowledge_curation import (
    KnowledgeExtractionProposal,
    KnowledgeSourceProfile,
)
from astrograph.knowledge.document_grounding import audit_document_proposal_grounding


def _profile():
    return KnowledgeSourceProfile(
        source_id="source:test", source_path=__file__, source_sha256="1" * 64,
        independence_group="g", license_id="test", media_type="text/markdown",
        extraction_method="test", extracted_text_sha256="2" * 64,
        fragments=[{
            "fragment_id": "fragment:test:table", "source_id": "source:test",
            "location": "table", "text_sha256": "3" * 64,
            "text": "| Channel | Present | First | Last | Minimum | Maximum | Mean / distinct |\n|---|---:|---:|---:|---:|---:|---:|\n| x_deg | 10 | 1 | 2 | 0 | 3 | 1.5 |",
        }],
    )


def _proposal(*, distinct=False):
    properties = {"channel": "x_deg", "count": 10, "first": 1, "last": 2,
                  "minimum": 0, "maximum": 3, "mean": 1.5}
    if distinct:
        properties["distinct"] = 10
    return KnowledgeExtractionProposal(
        source_profile_sha256="4" * 64,
        nodes=[{
            "node_id": "candidate:x", "node_type": "StateEstimate",
            "canonical_name": "x_deg", "properties": properties,
            "source_fragment_ids": ["fragment:test:table"], "confidence": 0.9,
        }], proposer_id="test", proposer_version="1",
    )


def test_grounding_accepts_exact_table_values() -> None:
    audit = audit_document_proposal_grounding(source_profile=_profile(), proposal=_proposal())
    assert audit["status"] == "PASS"
    assert audit["checked_explicit_field_count"] == 6
    assert audit["grounded_explicit_field_count"] == 6
    assert audit["source_trace_status"] == "COMPLETE"
    assert audit["explicit_field_grounding_status"] == "GROUNDED"


def test_grounding_accepts_llm_field_aliases_without_skipping_values() -> None:
    proposal = _proposal().model_copy(
        update={
            "nodes": [
                _proposal().nodes[0].model_copy(
                    update={
                        "properties": {
                            "channel": "x_deg",
                            "sample_count": 10,
                            "min_value": 0,
                            "max_value": 3,
                            "mean_value": 1.5,
                        }
                    }
                )
            ]
        }
    )
    audit = audit_document_proposal_grounding(
        source_profile=_profile(), proposal=proposal
    )
    assert audit["status"] == "PASS"
    assert audit["checked_explicit_field_count"] == 4
    assert audit["grounded_explicit_field_count"] == 4
    assert audit["explicit_field_grounding_status"] == "GROUNDED"


def test_grounding_rejects_invented_distinct_count() -> None:
    audit = audit_document_proposal_grounding(
        source_profile=_profile(), proposal=_proposal(distinct=True)
    )
    assert audit["status"] == "BLOCKED"
    assert [item["code"] for item in audit["findings"]] == ["UNSUPPORTED_DISTINCT_COUNT"]


def test_grounding_rejects_unknown_source_fragment() -> None:
    proposal = _proposal().model_copy(
        update={
            "nodes": [
                _proposal().nodes[0].model_copy(
                    update={"source_fragment_ids": ["fragment:not-present"]}
                )
            ]
        }
    )
    audit = audit_document_proposal_grounding(
        source_profile=_profile(), proposal=proposal
    )
    assert audit["status"] == "BLOCKED"
    assert audit["source_trace_status"] == "INCOMPLETE"
    assert "SOURCE_FRAGMENT_NOT_FOUND" in {
        item["code"] for item in audit["findings"]
    }
