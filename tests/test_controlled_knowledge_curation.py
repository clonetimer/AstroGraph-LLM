from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from astrograph.agents.knowledge_curation import (
    ControlledKnowledgeCurationWorkflow,
    KnowledgeCandidateCriticAgent,
    KnowledgeCurationAgent,
    KnowledgeExtractionProposal,
    KnowledgeSourceProfilerAgent,
    _profile_set_sha256,
)


def _sources(tmp_path: Path) -> list[dict[str, Any]]:
    source_a = tmp_path / "manual-a.md"
    source_b = tmp_path / "manual-b.md"
    source_a.write_text("Rotor friction can produce torque-authority loss.", encoding="utf-8")
    source_b.write_text("Independent test evidence links rotor friction to reduced torque.", encoding="utf-8")
    return [
        {
            "source_id": "manual-a",
            "source_path": source_a,
            "independence_group": "independent-manual-a",
            "license_id": "test-fixture",
            "extraction_method": "plain-text",
            "fragments": [{"location": "section 2", "text": source_a.read_text(encoding="utf-8")}],
        },
        {
            "source_id": "manual-b",
            "source_path": source_b,
            "independence_group": "independent-manual-b",
            "license_id": "test-fixture",
            "extraction_method": "plain-text",
            "fragments": [{"location": "section 5", "text": source_b.read_text(encoding="utf-8")}],
        },
    ]


def _profiles(sources: list[dict[str, Any]]):
    profiler = KnowledgeSourceProfilerAgent()
    return [profiler.profile(**source) for source in sources]


def _draft(profiles, **updates: Any) -> dict[str, Any]:
    fragments = [profile.fragments[0].fragment_id for profile in profiles]
    draft: dict[str, Any] = {
        "source_profile_sha256": _profile_set_sha256(profiles),
        "nodes": [
            {
                "node_id": "candidate:mechanism:rotor-friction",
                "node_type": "FailureMechanism",
                "canonical_name": "Rotor friction increase",
                "description": "Friction-related loss in a rotating actuator.",
                "source_fragment_ids": fragments,
                "confidence": 0.91,
            },
            {
                "node_id": "candidate:effect:torque-authority-loss",
                "node_type": "EffectPattern",
                "canonical_name": "Torque authority loss",
                "source_fragment_ids": fragments,
                "confidence": 0.9,
            },
        ],
        "relations": [
            {
                "edge_id": "candidate-edge:friction-propagates-torque-loss",
                "source_id": "candidate:mechanism:rotor-friction",
                "relation_type": "PROPAGATES_TO",
                "target_id": "candidate:effect:torque-authority-loss",
                "source_fragment_ids": fragments,
                "confidence": 0.88,
                "rationale": "Two independent sources describe the same mechanism-to-effect direction.",
            }
        ],
        "unresolved_claims": [],
        "proposer_id": "draft",
        "proposer_version": "1",
    }
    draft.update(updates)
    return draft


def _run(tmp_path: Path, draft_transform=None):
    sources = _sources(tmp_path)
    profiles = _profiles(sources)
    draft = _draft(profiles)
    if draft_transform is not None:
        draft_transform(draft, profiles)
    before = sorted(path.name for path in tmp_path.iterdir())
    result = ControlledKnowledgeCurationWorkflow().run(
        sources=sources,
        approved_existing_nodes={},
        approved_extraction_methods=["plain-text"],
        draft_proposal=draft,
    )
    after = sorted(path.name for path in tmp_path.iterdir())
    return result, before, after


def test_controlled_curation_is_read_only_and_only_ready_for_human_review(tmp_path: Path) -> None:
    result, before, after = _run(tmp_path)

    assert result.validation.status == "READY_FOR_HUMAN_REVIEW"
    assert result.validation.publication_allowed is False
    assert result.validation.requires_human_approval is True
    assert result.validation.active_graph_modified is False
    assert result.files_written == []
    assert before == after == ["manual-a.md", "manual-b.md"]
    assert [step.agent_id for step in result.trace] == [
        "knowledge-source-profiler-agent",
        "knowledge-curation-agent",
        "knowledge-candidate-critic-agent",
    ]


@pytest.mark.parametrize(
    ("expected_code", "transform"),
    [
        (
            "INSUFFICIENT_INDEPENDENT_CAUSAL_SOURCES",
            lambda draft, profiles: draft["relations"][0].update(
                source_fragment_ids=[profiles[0].fragments[0].fragment_id]
            ),
        ),
        (
            "V3_RELATION_SHAPE_NOT_ALLOWED",
            lambda draft, profiles: draft["nodes"][0].update(node_type="EffectPattern"),
        ),
        (
            "UNKNOWN_SOURCE_FRAGMENT",
            lambda draft, profiles: draft["nodes"][0].update(
                source_fragment_ids=["fragment:invented:0000000000000000"]
            ),
        ),
        (
            "LOW_CONFIDENCE",
            lambda draft, profiles: draft["relations"][0].update(confidence=0.1),
        ),
        (
            "AMBIGUOUS_CANDIDATE",
            lambda draft, profiles: draft["nodes"][0].update(ambiguous=True),
        ),
        (
            "UNAPPROVED_OR_UNKNOWN_TARGET_NODE",
            lambda draft, profiles: draft["relations"][0].update(target_id="effect:invented"),
        ),
        (
            "EXECUTABLE_CONTENT_FORBIDDEN",
            lambda draft, profiles: draft.update(executable_content_present=True),
        ),
        (
            "SOURCE_PROFILE_HASH_MISMATCH",
            lambda draft, profiles: draft.update(source_profile_sha256="0" * 64),
        ),
        (
            "DUPLICATE_CANDIDATE_NODE_ID",
            lambda draft, profiles: draft["nodes"].append(dict(draft["nodes"][0])),
        ),
    ],
)
def test_critic_blocks_each_candidate_safety_boundary(
    tmp_path: Path,
    expected_code: str,
    transform,
) -> None:
    result, _, _ = _run(tmp_path, transform)
    assert result.validation.status == "BLOCKED"
    assert expected_code in {item.code for item in result.validation.findings}


def test_unapproved_extraction_method_is_blocked(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    profiles = _profiles(sources)
    result = ControlledKnowledgeCurationWorkflow().run(
        sources=sources,
        approved_existing_nodes={},
        approved_extraction_methods=["pdftotext-layout"],
        draft_proposal=_draft(profiles),
    )
    assert result.validation.status == "BLOCKED"
    assert "UNAPPROVED_EXTRACTION_METHOD" in {
        item.code for item in result.validation.findings
    }


def test_two_files_in_same_independence_group_do_not_satisfy_causal_gate(
    tmp_path: Path,
) -> None:
    sources = _sources(tmp_path)
    sources[1]["independence_group"] = sources[0]["independence_group"]
    profiles = _profiles(sources)
    result = ControlledKnowledgeCurationWorkflow().run(
        sources=sources,
        approved_existing_nodes={},
        approved_extraction_methods=["plain-text"],
        draft_proposal=_draft(profiles),
    )
    assert result.validation.status == "BLOCKED"
    assert "INSUFFICIENT_INDEPENDENT_CAUSAL_SOURCES" in {
        item.code for item in result.validation.findings
    }


def test_source_file_hash_drift_is_blocked(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    profiles = _profiles(sources)
    proposal = KnowledgeExtractionProposal.model_validate(_draft(profiles))
    Path(profiles[0].source_path).write_text("changed after profiling", encoding="utf-8")

    receipt = KnowledgeCandidateCriticAgent().review(
        source_profiles=profiles,
        proposal=proposal,
        approved_existing_nodes={},
        approved_extraction_methods=["plain-text"],
    )
    assert receipt.status == "BLOCKED"
    assert "SOURCE_FILE_HASH_DRIFT" in {item.code for item in receipt.findings}


def test_unknown_schema_enum_is_rejected_before_review(tmp_path: Path) -> None:
    profiles = _profiles(_sources(tmp_path))
    draft = _draft(profiles)
    draft["relations"][0]["relation_type"] = "LLM_INVENTED_CAUSE"
    with pytest.raises(ValidationError):
        KnowledgeExtractionProposal.model_validate(draft)


def test_empty_proposal_cannot_claim_review_readiness(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    profiles = _profiles(sources)
    draft = _draft(profiles, nodes=[], relations=[])
    result = ControlledKnowledgeCurationWorkflow().run(
        sources=sources,
        approved_existing_nodes={},
        approved_extraction_methods=["plain-text"],
        draft_proposal=draft,
    )
    assert result.validation.status == "BLOCKED"
    assert "EMPTY_KNOWLEDGE_PROPOSAL" in {
        item.code for item in result.validation.findings
    }


class _RecordingBackend:
    model_id = "test-structured-knowledge-model"
    model_version = "1"

    def __init__(self, proposal: KnowledgeExtractionProposal) -> None:
        self.proposal = proposal
        self.payload: dict[str, Any] | None = None

    def generate_structured(self, *, agent_id: str, output_model: type, payload: dict[str, Any]):
        assert agent_id == "knowledge-curation-agent"
        assert output_model is KnowledgeExtractionProposal
        self.payload = payload
        return self.proposal


def test_llm_backend_only_receives_governed_fragments_and_cannot_publish(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    profiles = _profiles(sources)
    backend = _RecordingBackend(KnowledgeExtractionProposal.model_validate(_draft(profiles)))
    result = ControlledKnowledgeCurationWorkflow(
        curator=KnowledgeCurationAgent(backend=backend)
    ).run(
        sources=sources,
        approved_existing_nodes={},
        approved_extraction_methods=["plain-text"],
    )

    assert backend.payload is not None
    assert backend.payload["source_profile_sha256"] == _profile_set_sha256(profiles)
    assert all(item["fragments"] for item in backend.payload["source_profiles"])
    assert "allowed_relation_shapes" in backend.payload
    assert result.proposal.proposer_mode == "LLM"
    assert result.validation.status == "READY_FOR_HUMAN_REVIEW"
    assert result.validation.publication_allowed is False
    assert result.files_written == []
