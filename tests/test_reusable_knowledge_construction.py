from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from astrograph.agents import VllmKnowledgeExtractionBackend
from astrograph.agents.knowledge_curation import (
    KnowledgeExtractionProposal,
    KnowledgeNodeCandidate,
)
from astrograph.knowledge import (
    CanonicalizationAction,
    KnowledgeConstructionBundle,
    KnowledgeExtractionPassKind,
    KnowledgeNodeTypeV3,
    RELATION_ALLOWLIST,
    build_generic_fault_ontology_bundle,
    build_knowledge_construction_bundle,
    canonicalize_knowledge_proposal,
    profile_structured_source,
)


ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY = ROOT / "configs/knowledge/generic_fault_ontology.json"
ARCHETYPES = ROOT / "configs/semantics/core_component_archetypes.json"
SEMANTICS = ROOT / "configs/semantics/core_semantic_schema.json"
REFREEZE = (
    ROOT
    / "governance/evidence/ai_native/kg_coverage_review_20260831"
    / "supplemental_review_refreeze_20260901.json"
)


def _refreeze_sha() -> str:
    return json.loads(REFREEZE.read_text(encoding="utf-8"))["refreeze_sha256"]


@pytest.mark.parametrize(
    ("suffix", "content"),
    [
        (".md", "# Root\nintro\n## Child\nsame\n## Child 2\nsame\n"),
        (".json", '{"groups":[{"id":"a","value":1},{"id":"b","value":1}]}'),
        (".csv", "id,value\na,1\nb,1\n"),
    ],
)
def test_structured_adapters_are_stable_and_keep_duplicate_text_locations(
    tmp_path: Path, suffix: str, content: str
) -> None:
    source = tmp_path / f"source{suffix}"
    source.write_text(content, encoding="utf-8")
    first_profile, first_tree = profile_structured_source(
        source_id="source:test:stable",
        source_path=source,
        independence_group="test-group",
        license_id="test-fixture",
    )
    second_profile, second_tree = profile_structured_source(
        source_id="source:test:stable",
        source_path=source,
        independence_group="test-group",
        license_id="test-fixture",
    )
    assert first_profile == second_profile
    assert first_tree == second_tree
    assert len({item.fragment_id for item in first_profile.fragments}) == len(
        first_profile.fragments
    )
    assert all(
        item.parent_section_id is None or item.ancestor_section_ids
        for item in first_tree.sections
    )


def test_current_tbox_rebuild_is_deterministic_traced_and_review_bounded() -> None:
    kwargs = {
        "ontology_path": ONTOLOGY,
        "component_catalog_path": ARCHETYPES,
        "semantic_schema_path": SEMANTICS,
        "excluded_review_refreeze_sha256": _refreeze_sha(),
    }
    first = build_generic_fault_ontology_bundle(**kwargs)
    second = build_generic_fault_ontology_bundle(**kwargs)

    assert first.model_dump_json() == second.model_dump_json()
    assert first.status == "CANDIDATE_READY_FOR_HUMAN_REVIEW"
    assert first.candidate_validation.status == "READY_FOR_HUMAN_REVIEW"
    assert first.quality.status == "PASS"
    assert (first.quality.node_count, first.quality.edge_count) == (52, 90)
    assert first.quality.connected_component_count == 1
    assert first.quality.orphan_node_count == 0
    assert first.quality.traced_node_fraction == 1.0
    assert first.quality.traced_edge_fraction == 1.0
    assert first.excluded_review_refreeze_sha256 == _refreeze_sha()
    assert first.publication_allowed is False
    assert first.active_graph_modified is False
    assert first.candidate_graph is not None
    assert {task.pass_kind for task in first.extraction_tasks} == set(
        KnowledgeExtractionPassKind
    )
    assert all(task.section_context_summary for task in first.extraction_tasks)
    assert all(
        len(task.ancestor_section_ids) == len(task.ancestor_context_summaries)
        for task in first.extraction_tasks
    )
    assert all(not item.auto_applied for item in first.expansion_suggestions)
    assert first.expansion_suggestions
    node_types = {
        item.node_id: item.node_type for item in first.candidate_graph.nodes
    }
    existing = {
        (item.source_id, item.relation_type, item.target_id)
        for item in first.candidate_graph.edges
    }
    assert all(
        all(
            (
                node_types[item.source_id],
                relation_type,
                node_types[item.target_id],
            )
            in RELATION_ALLOWLIST
            and (item.source_id, relation_type, item.target_id) not in existing
            for relation_type in item.candidate_relation_types
        )
        for item in first.expansion_suggestions
    )


def test_bundle_hash_tampering_is_rejected() -> None:
    bundle = build_generic_fault_ontology_bundle(
        ONTOLOGY,
        ARCHETYPES,
        SEMANTICS,
        excluded_review_refreeze_sha256=_refreeze_sha(),
    )
    payload = bundle.model_dump(mode="json")
    payload["build_id"] = "KGBUILD-TAMPERED"
    with pytest.raises(ValidationError, match="bundle_sha256 mismatch"):
        KnowledgeConstructionBundle.model_validate(payload)
    content_payload = bundle.model_dump(mode="json")
    content_payload["content_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="content_sha256 mismatch"):
        KnowledgeConstructionBundle.model_validate(content_payload)


def test_new_extraction_cannot_weaken_causal_gate_or_compile_active() -> None:
    bundle = build_generic_fault_ontology_bundle(
        ONTOLOGY,
        ARCHETYPES,
        SEMANTICS,
        excluded_review_refreeze_sha256=_refreeze_sha(),
    )
    common = {
        "source_profiles": bundle.source_profiles,
        "source_trees": bundle.source_trees,
        "proposal": bundle.proposal,
        "approved_existing_nodes": {},
        "approved_extraction_methods": [
            item.extraction_method for item in bundle.source_profiles
        ],
        "excluded_review_refreeze_sha256": _refreeze_sha(),
        "graph_id": "graph:test:new-extraction",
    }
    with pytest.raises(ValueError, match="cannot weaken"):
        build_knowledge_construction_bundle(
            **common, minimum_independent_causal_sources=1
        )
    from astrograph.knowledge import LifecycleStatus

    with pytest.raises(ValueError, match="cannot compile directly to ACTIVE"):
        build_knowledge_construction_bundle(
            **common, lifecycle_status=LifecycleStatus.ACTIVE
        )


def test_canonicalization_merges_exact_but_never_fuzzy_candidates() -> None:
    proposal = KnowledgeExtractionProposal(
        source_profile_sha256="1" * 64,
        nodes=[
            KnowledgeNodeCandidate(
                node_id="candidate:effect:one",
                node_type=KnowledgeNodeTypeV3.EFFECT_PATTERN,
                canonical_name="Torque loss",
                description="Delivered torque decreases.",
                source_fragment_ids=["fragment:test:a"],
                confidence=0.9,
            ),
            KnowledgeNodeCandidate(
                node_id="candidate:effect:two",
                node_type=KnowledgeNodeTypeV3.EFFECT_PATTERN,
                canonical_name="torque-loss",
                description="Delivered torque decreases.",
                source_fragment_ids=["fragment:test:b"],
                confidence=0.9,
            ),
            KnowledgeNodeCandidate(
                node_id="candidate:effect:near",
                node_type=KnowledgeNodeTypeV3.EFFECT_PATTERN,
                canonical_name="Torque losses",
                description="Delivered torque decreases.",
                source_fragment_ids=["fragment:test:c"],
                confidence=0.9,
            ),
        ],
        proposer_id="test",
        proposer_version="1",
    )
    report = canonicalize_knowledge_proposal(proposal, fuzzy_review_threshold=0.75)
    assert report.status == "PASS"
    assert len(report.concepts) == 2
    assert any(
        item.action == CanonicalizationAction.MERGE_EXACT for item in report.decisions
    )
    assert any(
        item.action == CanonicalizationAction.KEEP_DISTINCT_REVIEW_SUGGESTED
        for item in report.decisions
    )


def test_type_conflict_fails_closed() -> None:
    proposal = KnowledgeExtractionProposal(
        source_profile_sha256="2" * 64,
        nodes=[
            KnowledgeNodeCandidate(
                node_id="candidate:one",
                node_type=KnowledgeNodeTypeV3.EFFECT_PATTERN,
                canonical_name="Shared name",
                source_fragment_ids=["fragment:test:a"],
                confidence=1.0,
            ),
            KnowledgeNodeCandidate(
                node_id="candidate:two",
                node_type=KnowledgeNodeTypeV3.FAILURE_MECHANISM,
                canonical_name="Shared-name",
                source_fragment_ids=["fragment:test:b"],
                confidence=1.0,
            ),
        ],
        proposer_id="test",
        proposer_version="1",
    )
    report = canonicalize_knowledge_proposal(proposal)
    assert report.status == "BLOCKED"
    assert "CANONICAL_NAME_TYPE_CONFLICT" in report.blocker_codes


def test_vllm_backend_uses_model_discovery_and_strict_json_schema(monkeypatch) -> None:
    backend = VllmKnowledgeExtractionBackend(base_url="http://127.0.0.1:8000")
    proposal = KnowledgeExtractionProposal(
        source_profile_sha256="3" * 64,
        proposer_id="mock-vllm",
        proposer_version="1",
    )
    calls: list[tuple[str, dict | None]] = []

    def fake_request(path: str, payload: dict | None = None):
        calls.append((path, payload))
        if path == "/v1/models":
            return {"data": [{"id": "local-test-model"}]}
        return {
            "choices": [{"message": {"content": proposal.model_dump_json()}}]
        }

    monkeypatch.setattr(backend, "_json_request", fake_request)
    result = backend.generate_structured(
        agent_id="knowledge-curation-agent",
        output_model=KnowledgeExtractionProposal,
        payload={"governed": True},
    )
    assert result == proposal
    assert calls[0][0] == "/v1/models"
    assert calls[1][0] == "/v1/chat/completions"
    request = calls[1][1]
    assert request is not None
    assert request["model"] == "local-test-model"
    assert request["temperature"] == 0.0
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["strict"] is True
