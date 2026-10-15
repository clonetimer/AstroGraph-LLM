from pathlib import Path

import pytest

from astrograph.agents.knowledge_curation import KnowledgeExtractionProposal
from astrograph.knowledge import process_governed_document
from astrograph.knowledge.construction import (
    DocumentKnowledgeGraphScope,
    knowledge_source_profile_set_sha256,
    stage_quarantined_document_source,
)
from astrograph.knowledge.document_graph import (
    build_document_candidate_graph,
    document_graph_extraction_policy,
    propose_staged_document_knowledge,
)
from astrograph.knowledge.schema_v3 import GraphLayer, NODE_LAYERS
from astrograph.retrieval.document_governance import (
    DocumentAuthority,
    DocumentReviewState,
    DocumentType,
    DocumentVisibilityClass,
    GovernedDocumentDescriptor,
)


FREEZE_SHA = "5" * 64


class _Backend:
    model_id = "scope-aware-fixture"
    model_version = "1"

    def __init__(self, *, invalid_scope: bool = False) -> None:
        self.invalid_scope = invalid_scope
        self.payload = None

    def generate_structured(self, *, agent_id, output_model, payload):
        self.payload = payload
        fragment_id = payload["source_profiles"][0]["fragments"][0]["fragment_id"]
        node_type = "DiagnosticHypothesis" if self.invalid_scope else "FailureMechanism"
        return KnowledgeExtractionProposal(
            source_profile_sha256=payload["source_profile_sha256"],
            nodes=[
                {
                    "node_id": "candidate:test:fault-a",
                    "node_type": node_type,
                    "canonical_name": "Fault A",
                    "description": "A source-grounded extraction candidate.",
                    "source_fragment_ids": [fragment_id],
                    "confidence": 0.9,
                }
            ],
            proposer_id="fixture",
            proposer_version="1",
        )


def _processed(tmp_path: Path):
    source = tmp_path / "candidate.md"
    source.write_text(
        "# Fault A\n\nFault A is a review-required mechanism candidate.\n",
        encoding="utf-8",
    )
    import hashlib

    descriptor = GovernedDocumentDescriptor(
        document_id="astrograph:test:tbox-document:r001",
        document_type=DocumentType.DESIGN_PROPOSAL,
        visibility_class=DocumentVisibilityClass.DESIGN_PUBLIC,
        source_uri="corpus://test/candidate.md",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        review_state=DocumentReviewState.REVIEW_REQUIRED,
        authority=DocumentAuthority.NON_NORMATIVE,
        provenance_group_id="fixture-single-source",
        runtime_vote_allowed=False,
    )
    return process_governed_document(
        descriptor=descriptor,
        source_path=source,
        independence_group="fixture-single-source",
        license_id="test-fixture",
    )


def test_tbox_policy_is_narrower_than_the_global_schema() -> None:
    policy = document_graph_extraction_policy(
        DocumentKnowledgeGraphScope.TBOX_CANDIDATE
    )
    assert policy["allowed_node_types"]
    assert all(NODE_LAYERS[item] is GraphLayer.TBOX for item in policy["allowed_node_types"])
    assert "DiagnosticHypothesis" not in {
        item.value for item in policy["allowed_node_types"]
    }
    assert any("confusion-set alternative" in rule for rule in policy["additional_rules"])


def test_builds_review_only_tbox_candidate_graph(tmp_path: Path) -> None:
    backend = _Backend()
    bundle = build_document_candidate_graph(
        processed=_processed(tmp_path),
        backend=backend,
        excluded_review_refreeze_sha256=FREEZE_SHA,
        graph_id="graph:test:document-candidate",
    )
    assert backend.payload is not None
    assert backend.payload["governance_context"]["graph_scope"] == "TBOX_CANDIDATE"
    assert backend.payload["governance_context"]["publication_allowed"] is False
    assert any(
        "representation withheld pending human review" in rule
        for rule in backend.payload["rules"]
    )
    assert bundle.status == "CANDIDATE_READY_FOR_HUMAN_REVIEW"
    assert bundle.candidate_graph is not None
    assert bundle.candidate_graph.nodes[0].lifecycle_status.value == "DRAFT"
    assert bundle.candidate_graph.nodes[0].properties["publication_allowed"] is False
    assert bundle.candidate_graph.nodes[0].properties["runtime_vote_allowed"] is False
    assert bundle.active_graph_modified is False
    assert bundle.publication_allowed is False


def test_non_tbox_llm_output_is_rejected_before_compilation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="out-of-scope candidate nodes"):
        build_document_candidate_graph(
            processed=_processed(tmp_path),
            backend=_Backend(invalid_scope=True),
            excluded_review_refreeze_sha256=FREEZE_SHA,
            graph_id="graph:test:invalid-document-candidate",
        )


def test_episode_and_quarantine_policies_exclude_runtime_decisions() -> None:
    episode = document_graph_extraction_policy(
        DocumentKnowledgeGraphScope.EPISODE_EVIDENCE_CANDIDATE
    )
    quarantine = document_graph_extraction_policy(
        DocumentKnowledgeGraphScope.QUARANTINE_EVALUATOR_CANDIDATE
    )
    assert "DiagnosticHypothesis" not in {item.value for item in episode["allowed_node_types"]}
    assert "DecisionRecord" not in {item.value for item in episode["allowed_node_types"]}
    assert "FailureMechanism" not in {item.value for item in quarantine["allowed_node_types"]}
    assert "DiagnosticHypothesis" in {item.value for item in quarantine["allowed_node_types"]}
    assert any("evaluator-only truth" in rule for rule in quarantine["additional_rules"])


def test_truth_json_has_a_graph_only_quarantine_staging_lane(tmp_path: Path) -> None:
    import hashlib
    import json

    source = tmp_path / "truth.json"
    source.write_text(json.dumps({"episode_id": "episode:test", "label": "fault_a"}), encoding="utf-8")
    descriptor = GovernedDocumentDescriptor(
        document_id="doc:truth:test",
        document_type=DocumentType.CAMPAIGN_REPORT,
        visibility_class=DocumentVisibilityClass.TRUTH_QUARANTINED,
        source_uri="quarantine://test/truth.json",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        review_state=DocumentReviewState.MACHINE_VALIDATED,
        authority=DocumentAuthority.NON_NORMATIVE,
        provenance_group_id="fixture-group",
        episode_id="episode:test",
        contains_episode_truth=True,
        synthetic_from_simulation=True,
        runtime_vote_allowed=False,
    )
    staging = stage_quarantined_document_source(
        descriptor=descriptor,
        source_path=source,
        independence_group="fixture-group",
        license_id="test-fixture",
    )
    assert staging.graph_scope is DocumentKnowledgeGraphScope.QUARANTINE_EVALUATOR_CANDIDATE
    assert staging.source_profile.source_role == "EXPERIMENT_EVIDENCE"
    assert staging.publication_allowed is False
    assert staging.runtime_vote_allowed is False
    assert staging.requires_human_review is True


def test_resolved_closure_uses_historical_graph_scope_not_runtime_truth(tmp_path: Path) -> None:
    import hashlib

    source = tmp_path / "closure.md"
    source.write_text("# Closure\n\nHistorical root cause candidate: fault_a.\n", encoding="utf-8")
    descriptor = GovernedDocumentDescriptor(
        document_id="doc:closure:test",
        document_type=DocumentType.CLOSURE_REPORT,
        visibility_class=DocumentVisibilityClass.RESOLVED_CASE_TRAIN_ONLY,
        source_uri="corpus://historical/test/closure.md",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        review_state=DocumentReviewState.REVIEW_REQUIRED,
        authority=DocumentAuthority.NON_NORMATIVE,
        provenance_group_id="historical-group",
        episode_id="episode:historical",
        contains_root_cause=True,
        contains_episode_truth=True,
        synthetic_from_simulation=True,
        runtime_vote_allowed=False,
    )
    processed = process_governed_document(
        descriptor=descriptor,
        source_path=source,
        independence_group="historical-group",
        license_id="test-fixture",
    )
    assert processed.knowledge_staging.graph_scope is DocumentKnowledgeGraphScope.HISTORICAL_CASE_CANDIDATE
    assert processed.knowledge_staging.runtime_vote_allowed is False
    assert processed.publication_allowed is False


def test_section_batched_extraction_sends_one_profiled_fragment_per_call(tmp_path: Path) -> None:
    import hashlib

    source = tmp_path / "multi.md"
    source.write_text("# First\n\nAlpha.\n\n# Second\n\nBeta.\n", encoding="utf-8")
    descriptor = GovernedDocumentDescriptor(
        document_id="astrograph:test:multi:r001",
        document_type=DocumentType.DESIGN_PROPOSAL,
        visibility_class=DocumentVisibilityClass.DESIGN_PUBLIC,
        source_uri="corpus://test/multi.md",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        review_state=DocumentReviewState.REVIEW_REQUIRED,
        authority=DocumentAuthority.NON_NORMATIVE,
        provenance_group_id="fixture",
    )
    processed = process_governed_document(
        descriptor=descriptor, source_path=source,
        independence_group="fixture", license_id="test-fixture",
    )

    class RecordingBackend(_Backend):
        def __init__(self):
            super().__init__()
            self.payloads = []
            self.output_schemas = []

        def generate_structured(self, **kwargs):
            self.payloads.append(kwargs["payload"])
            self.output_schemas.append(kwargs["output_model"].model_json_schema())
            return super().generate_structured(**kwargs)

    backend = RecordingBackend()
    proposal = propose_staged_document_knowledge(
        staging=processed.knowledge_staging,
        backend=backend,
        section_batched=True,
    )
    assert len(backend.payloads) == len({
        task.fragment_id for task in processed.knowledge_staging.extraction_tasks
    })
    assert len(backend.payloads) >= 2
    assert all(
        sum(len(profile["fragments"]) for profile in payload["source_profiles"]) == 1
        for payload in backend.payloads
    )
    assert all(payload["governance_context"]["fragment_task_objective"] for payload in backend.payloads)
    assert all(schema["properties"]["nodes"]["maxItems"] == 6 for schema in backend.output_schemas)
    assert all(schema["properties"]["relations"]["maxItems"] == 6 for schema in backend.output_schemas)
    assert proposal.source_profile_sha256 == backend.payloads[0]["source_profile_sha256"]


def test_episode_identity_is_deterministic_when_llm_returns_no_content_candidates(tmp_path: Path) -> None:
    import hashlib

    source = tmp_path / "observation.md"
    source.write_text("# Observation\n\nNo calibrated evidence claim.\n", encoding="utf-8")
    descriptor = GovernedDocumentDescriptor(
        document_id="doc:observation:empty",
        document_type=DocumentType.TEST_LOG,
        visibility_class=DocumentVisibilityClass.OBSERVATION_CAUSAL,
        source_uri="corpus://runtime/empty/observation.md",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        review_state=DocumentReviewState.MACHINE_VALIDATED,
        authority=DocumentAuthority.NON_NORMATIVE,
        provenance_group_id="group:dev",
        episode_id="episode:empty",
        valid_time_end_s=120,
        observable_cutoff_time_s=120,
        runtime_vote_allowed=False,
    )
    processed = process_governed_document(
        descriptor=descriptor, source_path=source,
        independence_group="group:dev", license_id="test-fixture",
    )

    class EmptyBackend:
        model_id = "empty-fixture"
        model_version = "1"

        def generate_structured(self, *, agent_id, output_model, payload):
            assert payload["approved_existing_nodes"] == {
                "episode:empty": "ObservationEpisode",
                "evidence-source:doc:observation:empty": "EvidenceSource",
            }
            return KnowledgeExtractionProposal(
                source_profile_sha256=payload["source_profile_sha256"],
                proposer_id="fixture", proposer_version="1",
            )

    bundle = build_document_candidate_graph(
        processed=processed,
        backend=EmptyBackend(),
        excluded_review_refreeze_sha256=FREEZE_SHA,
        graph_id="graph:test:empty-episode",
    )
    assert bundle.status == "CANDIDATE_READY_FOR_HUMAN_REVIEW"
    assert bundle.candidate_graph is not None
    assert {node.node_id for node in bundle.candidate_graph.nodes} == {
        "episode:empty",
        "evidence-source:doc:observation:empty",
    }
    assert all(node.properties["descriptor_scaffold"] is True for node in bundle.candidate_graph.nodes)


def test_explicit_generated_tbox_fields_survive_an_empty_llm_proposal_without_causal_edge(
    tmp_path: Path,
) -> None:
    import hashlib

    source = tmp_path / "generated-fmea.md"
    source.write_text(
        "# Candidate\n\n"
        "- Target fault: `adcs_rw_jamming`\n"
        "- Component candidate: `reaction_wheel`\n"
        "- Mechanism association: `mechanism:output-stagnation`\n"
        "- Registered evidence definition: `ED-RW-JAM-1`\n",
        encoding="utf-8",
    )
    descriptor = GovernedDocumentDescriptor(
        document_id="astrograph:test:generated-fmea:r001",
        document_type=DocumentType.DESIGN_PROPOSAL,
        visibility_class=DocumentVisibilityClass.DESIGN_PUBLIC,
        source_uri="corpus://test/generated-fmea.md",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        review_state=DocumentReviewState.REVIEW_REQUIRED,
        authority=DocumentAuthority.NON_NORMATIVE,
        provenance_group_id="synthetic-generator-family",
        synthetic_from_simulation=True,
        runtime_vote_allowed=False,
    )
    processed = process_governed_document(
        descriptor=descriptor,
        source_path=source,
        independence_group="synthetic-generator-family",
        license_id="test-fixture",
    )
    empty = KnowledgeExtractionProposal(
        source_profile_sha256=knowledge_source_profile_set_sha256([
            processed.knowledge_staging.source_profile
        ]),
        proposer_id="empty-fixture",
        proposer_version="1",
    )
    bundle = build_document_candidate_graph(
        processed=processed,
        proposal=empty,
        excluded_review_refreeze_sha256=FREEZE_SHA,
        graph_id="graph:test:generated-fmea",
    )
    assert bundle.status == "CANDIDATE_READY_FOR_HUMAN_REVIEW"
    assert bundle.candidate_graph is not None
    assert {node.node_type.value for node in bundle.candidate_graph.nodes} == {
        "ComponentArchetype", "FailureMechanism"
    }
    mechanism = next(
        node for node in bundle.candidate_graph.nodes
        if node.node_type.value == "FailureMechanism"
    )
    assert mechanism.node_id == "mechanism:output-stagnation"
    assert mechanism.properties["registered_evidence_definition_ids"] == ["ED-RW-JAM-1"]
    assert mechanism.properties["diagnostic_evidence_allowed"] is False
    assert bundle.candidate_graph.edges == []
    assert bundle.proposal.unresolved_claims[0].startswith("UNCONFIRMED_TBOX_ASSOCIATION|")
