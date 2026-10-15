import hashlib
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from pydantic import ValidationError

from astrograph.agents.knowledge_curation import (
    KnowledgeExtractionProposal,
    KnowledgeNodeCandidate,
)
from astrograph.retrieval import (
    ClaimEpistemicStatus,
    ClaimTemporalScope,
    DiagnosticRetrievalQuery,
    DocumentAccessContext,
    DocumentAccessMode,
    DocumentAuthority,
    DocumentEvidenceClaim,
    DocumentReviewState,
    DocumentType,
    DocumentVisibilityClass,
    GovernedDocumentDescriptor,
    GovernedTextChunk,
    PreReasoningRetrievalConfig,
    PreReasoningRetrievalRequest,
    PreReasoningRetrievalService,
    SectionFirstDocumentIngestor,
    bind_document_evidence_claim,
    evaluate_document_access,
)
from astrograph.knowledge.construction import (
    DocumentKnowledgeGraphScope,
    GovernedDocumentProcessingBundle,
    KnowledgeNodeTypeV3,
    govern_document_knowledge_proposal,
    knowledge_source_profile_set_sha256,
    process_governed_document,
    process_governed_docx,
    stage_governed_document_source,
)
from astrograph.knowledge.document_graph import build_document_candidate_graph


SHA = "a" * 64


def _descriptor(**overrides):
    payload = {
        "document_id": "doc:design",
        "document_type": DocumentType.DESIGN_MANUAL,
        "visibility_class": DocumentVisibilityClass.DESIGN_PUBLIC,
        "source_uri": "manual.docx",
        "source_sha256": SHA,
        "review_state": DocumentReviewState.HUMAN_REVIEWED,
        "authority": DocumentAuthority.ENGINEERING_REFERENCE,
    }
    payload.update(overrides)
    return GovernedDocumentDescriptor(**payload)


def _chunk(descriptor):
    text = "reaction wheel current anomaly"
    return GovernedTextChunk(
        chunk_id="chunk:1",
        source_id="doc:1",
        source_uri="manual.docx",
        source_sha256=SHA,
        chunk_sha256=hashlib.sha256(text.encode()).hexdigest(),
        knowledge_version="v1",
        text=text,
        document_descriptor=descriptor,
    )


def _retrieve(descriptor, context):
    request = PreReasoningRetrievalRequest(
        query=DiagnosticRetrievalQuery(
            query_id="query:1",
            episode_id="episode:current",
            text="reaction wheel anomaly",
            document_access_context=context,
        ),
        chunks=[_chunk(descriptor)],
        config=PreReasoningRetrievalConfig(
            require_local_and_ontology=False,
            minimum_local_score=0.0,
        ),
    )
    return PreReasoningRetrievalService().retrieve(request)


def test_proposal_and_illustrative_claims_cannot_vote():
    with pytest.raises(ValidationError, match="cannot be runtime evidence"):
        DocumentEvidenceClaim(
            claim_id="claim:1",
            subject="IxRT",
            predicate="IS_SELECTED_FOR",
            object="deployment",
            source_document_id="doc:proposal",
            source_document_sha256=SHA,
            source_locator="paragraph:1",
            epistemic_status=ClaimEpistemicStatus.DESIGN_PROPOSAL,
            source_review_state=DocumentReviewState.HUMAN_REVIEWED,
            source_authority=DocumentAuthority.APPROVED_SOURCE,
            temporal_scope=ClaimTemporalScope.TIME_INDEPENDENT,
            source_span_sha256=SHA,
            runtime_evidence_eligible=True,
        )


def test_claim_binding_preserves_file_generator_and_causal_time() -> None:
    descriptor = _descriptor(
        document_id="doc:causal",
        document_type=DocumentType.TEST_LOG,
        visibility_class=DocumentVisibilityClass.OBSERVATION_CAUSAL,
        review_state=DocumentReviewState.HUMAN_REVIEWED,
        authority=DocumentAuthority.APPROVED_SOURCE,
        episode_id="episode:1",
        valid_time_start_s=0.0,
        valid_time_end_s=60.0,
        observable_cutoff_time_s=60.0,
        generator_sha256="b" * 64,
        synthetic_from_simulation=True,
        runtime_vote_allowed=True,
    )
    claim = bind_document_evidence_claim(
        claim_id="claim:causal",
        subject="reaction-wheel",
        predicate="HAS_OBSERVATION",
        object="current-rise",
        source_span="Current increased before the cutoff.",
        source_locator="lines:10-12",
        descriptor=descriptor,
        epistemic_status=ClaimEpistemicStatus.SOURCE_FACT,
        publication_eligible=True,
        runtime_evidence_eligible=True,
    )
    assert claim.source_document_sha256 == SHA
    assert claim.source_generator_sha256 == "b" * 64
    assert claim.temporal_scope is ClaimTemporalScope.TIME_BOUNDED
    assert claim.valid_time_end_s == claim.observable_cutoff_time_s == 60.0


def test_unreviewed_proposal_claim_cannot_be_marked_publishable() -> None:
    descriptor = _descriptor(
        document_id="doc:unreviewed-proposal",
        document_type=DocumentType.DESIGN_PROPOSAL,
        review_state=DocumentReviewState.REVIEW_REQUIRED,
        authority=DocumentAuthority.NON_NORMATIVE,
    )
    with pytest.raises(ValidationError, match="publishable claims require human review"):
        bind_document_evidence_claim(
            claim_id="claim:proposal",
            subject="pipeline",
            predicate="WILL_USE",
            object="future-model",
            source_span="The design proposes a future model.",
            source_locator="paragraph:4",
            descriptor=descriptor,
            epistemic_status=ClaimEpistemicStatus.DESIGN_PROPOSAL,
            publication_eligible=True,
        )


def test_machine_validated_design_and_historical_documents_require_human_review():
    design = _descriptor(
        document_id="design-machine-only",
        review_state=DocumentReviewState.MACHINE_VALIDATED,
    )
    historical = _descriptor(
        document_id="closure-machine-only",
        document_type=DocumentType.CLOSURE_REPORT,
        visibility_class=DocumentVisibilityClass.RESOLVED_CASE_TRAIN_ONLY,
        review_state=DocumentReviewState.MACHINE_VALIDATED,
        provenance_group_id="group:past",
        contains_root_cause=True,
    )
    context = DocumentAccessContext(
        current_provenance_group_id="group:current",
        allowed_historical_provenance_group_ids=["group:past"],
    )

    assert evaluate_document_access(design, context).reason_code == "DESIGN_HUMAN_REVIEW_REQUIRED"
    assert (
        evaluate_document_access(historical, context).reason_code
        == "HISTORICAL_HUMAN_REVIEW_REQUIRED"
    )


def test_truth_is_unconditionally_blocked_from_runtime_but_visible_to_evaluator():
    descriptor = _descriptor(
        visibility_class=DocumentVisibilityClass.TRUTH_QUARANTINED,
        contains_episode_truth=True,
    )
    runtime = evaluate_document_access(
        descriptor,
        DocumentAccessContext(mode=DocumentAccessMode.RUNTIME_DIAGNOSIS),
    )
    evaluator = evaluate_document_access(
        descriptor,
        DocumentAccessContext(mode=DocumentAccessMode.OFFLINE_EVALUATION),
    )
    assert (runtime.allowed, runtime.reason_code) == (False, "TRUTH_QUARANTINED")
    assert (evaluator.allowed, evaluator.reason_code) == (True, "EVALUATOR_TRUTH_ACCESS")


def test_current_group_closure_is_blocked_before_retrieval_scoring():
    descriptor = _descriptor(
        document_type=DocumentType.CLOSURE_REPORT,
        visibility_class=DocumentVisibilityClass.RESOLVED_CASE_TRAIN_ONLY,
        provenance_group_id="group:current",
        contains_root_cause=True,
    )
    package = _retrieve(
        descriptor,
        DocumentAccessContext(
            episode_id="episode:current",
            current_provenance_group_id="group:current",
            allowed_historical_provenance_group_ids=["group:current"],
        ),
    )
    assert package.local_hits == []
    assert package.access_filtered_chunk_count == 1
    assert package.document_access_reason_counts == {"CURRENT_GROUP_CLOSURE_BLOCKED": 1}


def test_allowlisted_other_group_closure_is_retrievable():
    descriptor = _descriptor(
        document_type=DocumentType.CLOSURE_REPORT,
        visibility_class=DocumentVisibilityClass.RESOLVED_CASE_TRAIN_ONLY,
        provenance_group_id="group:past",
        contains_root_cause=True,
    )
    package = _retrieve(
        descriptor,
        DocumentAccessContext(
            episode_id="episode:current",
            current_provenance_group_id="group:current",
            allowed_historical_provenance_group_ids=["group:past"],
        ),
    )
    assert [item.chunk_id for item in package.local_hits] == ["chunk:1"]
    assert package.document_access_reason_counts == {}


def test_future_observation_is_blocked_and_past_observation_is_allowed():
    descriptor = _descriptor(
        document_type=DocumentType.TEST_LOG,
        visibility_class=DocumentVisibilityClass.OBSERVATION_CAUSAL,
        episode_id="episode:current",
        valid_time_start_s=0.0,
        valid_time_end_s=20.0,
        observable_cutoff_time_s=20.0,
    )
    future = _retrieve(
        descriptor,
        DocumentAccessContext(
            episode_id="episode:current",
            observation_time_s=10.0,
        ),
    )
    past = _retrieve(
        descriptor,
        DocumentAccessContext(
            episode_id="episode:current",
            observation_time_s=20.0,
        ),
    )
    assert future.document_access_reason_counts == {"FUTURE_OBSERVATION_BLOCKED": 1}
    assert [item.chunk_id for item in past.local_hits] == ["chunk:1"]


def _write_minimal_docx(path):
    document = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
 <w:body>
  <w:p><w:pPr><w:pStyle w:val="h1"/></w:pPr><w:r><w:t>Design</w:t></w:r></w:p>
  <w:p><w:pPr><w:pStyle w:val="h2"/></w:pPr><w:r><w:t>First</w:t></w:r></w:p>
  <w:p><w:r><w:t>Reaction wheel design proposal.</w:t></w:r></w:p>
  <w:p><w:pPr><w:pStyle w:val="h2"/></w:pPr><w:r><w:t>Second</w:t></w:r></w:p>
  <w:tbl>
   <w:tr><w:tc><w:p><w:r><w:t>Signal</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Limit</w:t></w:r></w:p></w:tc></w:tr>
   <w:tr><w:tc><w:p><w:r><w:t>current</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>2 A</w:t></w:r></w:p></w:tc></w:tr>
  </w:tbl>
  <w:p><m:oMath><m:r><m:t>z=(x-mu)/sigma</m:t></m:r></m:oMath></w:p>
  <w:p><w:r><w:drawing><a:blip r:embed="rId1"/></w:drawing></w:r></w:p>
  <w:p><w:r><w:t>Figure 1 Diagnostic flow.</w:t></w:r></w:p>
  <w:sectPr/>
 </w:body>
</w:document>"""
    styles = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
 <w:style w:type="paragraph" w:styleId="h1"><w:name w:val="heading 1"/><w:pPr><w:outlineLvl w:val="0"/></w:pPr></w:style>
 <w:style w:type="paragraph" w:styleId="h2"><w:name w:val="heading 2"/><w:pPr><w:outlineLvl w:val="1"/></w:pPr></w:style>
</w:styles>"""
    relationships = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image1.png"/>
</Relationships>"""
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", document)
        archive.writestr("word/styles.xml", styles)
        archive.writestr("word/_rels/document.xml.rels", relationships)
        archive.writestr("word/media/image1.png", b"fake-png")


def test_docx_native_parser_preserves_headings_tables_and_images(tmp_path):
    path = tmp_path / "design.docx"
    _write_minimal_docx(path)
    ingestor = SectionFirstDocumentIngestor(maximum_section_units=1)
    elements, warnings, audit = ingestor.parse_docx(path)
    assert audit.nonempty_paragraph_count == 5
    assert audit.heading_count == 3
    assert audit.table_count == 1
    assert audit.table_row_count == 2
    assert audit.table_cell_count == 4
    assert audit.image_count == 1
    assert audit.native_formula_count == 1
    assert audit.headings[0]["text"] == "Design"
    assert {item.kind.value for item in elements} == {"TEXT", "TABLE", "IMAGE", "FORMULA"}
    assert any(item.section_path == ["Design", "First"] for item in elements)
    assert any(item.section_path == ["Design", "Second"] for item in elements)
    assert not any(item.section_path == ["Design", "First", "Second"] for item in elements)
    assert any(item.asset_uri == "docx://word/media/image1.png" for item in elements)
    assert any(item.text_representation.startswith("| Signal | Limit |") for item in elements)
    assert any(item.text_representation == "z=(x-mu)/sigma" for item in elements)
    assert warnings == [
        "FORMULA_REPRESENTATION_REVIEW_REQUIRED:6:1",
        "IMAGE_REPRESENTATION_REVIEW_REQUIRED:7:1",
    ]


def test_docx_can_stage_for_candidate_kg_without_publishing(tmp_path):
    path = tmp_path / "design.docx"
    _write_minimal_docx(path)
    source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    descriptor = _descriptor(
        document_id="design:proposal:fixture",
        document_type=DocumentType.DESIGN_PROPOSAL,
        source_uri=str(path),
        source_sha256=source_sha,
        review_state=DocumentReviewState.REVIEW_REQUIRED,
        authority=DocumentAuthority.NON_NORMATIVE,
    )

    staged = stage_governed_document_source(
        descriptor=descriptor,
        source_path=path,
        independence_group="fixture-design-origin",
        license_id="test-fixture",
    )

    assert staged.graph_scope is DocumentKnowledgeGraphScope.TBOX_CANDIDATE
    assert staged.default_epistemic_status is ClaimEpistemicStatus.DESIGN_PROPOSAL
    assert staged.publication_allowed is False
    assert staged.runtime_vote_allowed is False
    assert staged.requires_human_review is True
    assert staged.source_profile.extraction_method == "astrograph-docx-native-section-v1"
    assert staged.extraction_tasks
    assert any("[TABLE;" in section.text for section in staged.source_tree.sections)
    assert any("[IMAGE;" in section.text for section in staged.source_tree.sections)
    source_text = "\n".join(section.text for section in staged.source_tree.sections)
    assert "z=(x-mu)/sigma" not in source_text
    assert "FORMULA representation withheld pending human review" in source_text
    assert "IMAGE representation withheld pending human review" in source_text


def test_docx_is_traceably_processed_for_retrieval_and_candidate_kg(tmp_path):
    path = tmp_path / "design.docx"
    _write_minimal_docx(path)
    source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    descriptor = _descriptor(
        document_id="design:dual-use:fixture",
        document_type=DocumentType.DESIGN_PROPOSAL,
        source_uri=str(path),
        source_sha256=source_sha,
        review_state=DocumentReviewState.REVIEW_REQUIRED,
        authority=DocumentAuthority.NON_NORMATIVE,
    )

    processed = process_governed_docx(
        descriptor=descriptor,
        source_path=path,
        independence_group="fixture-design-origin",
        license_id="test-fixture",
    )

    assert processed.retrieval_ready is True
    assert processed.candidate_kg_staged is True
    assert processed.candidate_graph_built is False
    assert processed.publication_allowed is False
    assert processed.runtime_vote_allowed is False
    assert len(processed.crosswalk) == len(processed.retrieval_ingestion.elements)

    fragment_id = processed.knowledge_staging.source_profile.fragments[-1].fragment_id
    proposal = KnowledgeExtractionProposal(
        source_profile_sha256=knowledge_source_profile_set_sha256(
            [processed.knowledge_staging.source_profile]
        ),
        nodes=[KnowledgeNodeCandidate(
            node_id="candidate:component:wheel",
            node_type=KnowledgeNodeTypeV3.COMPONENT_ARCHETYPE,
            canonical_name="reaction wheel",
            source_fragment_ids=[fragment_id],
            confidence=0.9,
        )],
        proposer_id="fixture",
        proposer_version="1",
    )
    governed = govern_document_knowledge_proposal(
        staging=processed.knowledge_staging, proposal=proposal
    )
    assert governed.nodes[0].properties["source_document_sha256"] == source_sha
    assert governed.nodes[0].properties["epistemic_status"] == "DESIGN_PROPOSAL"
    assert governed.nodes[0].properties["publication_allowed"] is False

    spoofed = proposal.model_copy(update={
        "nodes": [proposal.nodes[0].model_copy(update={
            "properties": {"runtime_vote_allowed": True}
        })]
    })
    with pytest.raises(ValueError, match="override document governance"):
        govern_document_knowledge_proposal(
            staging=processed.knowledge_staging, proposal=spoofed
        )
    assert {item.element_id for item in processed.crosswalk} == {
        item.element_id for item in processed.retrieval_ingestion.elements
    }
    assert processed.retrieval_ingestion.source_sha256 == (
        processed.knowledge_staging.source_profile.source_sha256
    )

    payload = processed.model_dump(mode="json")
    payload["crosswalk"][0]["retrieval_chunk_id"] = "missing:chunk"
    with pytest.raises(ValidationError, match="unknown retrieval chunk"):
        GovernedDocumentProcessingBundle.model_validate(payload)
    content_payload = processed.model_dump(mode="json")
    content_payload["content_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="content_sha256 mismatch"):
        GovernedDocumentProcessingBundle.model_validate(content_payload)


def test_markdown_is_processed_once_for_retrieval_and_candidate_kg(tmp_path):
    path = tmp_path / "manual.md"
    path.write_text(
        "# Reaction wheel manual\n\n## Observation\n\nCurrent rise may require review.\n",
        encoding="utf-8",
    )
    source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    descriptor = _descriptor(
        document_id="design:markdown:fixture",
        source_uri=str(path),
        source_sha256=source_sha,
    )
    processed = process_governed_document(
        descriptor=descriptor,
        source_path=path,
        independence_group="fixture-markdown-origin",
        license_id="test-fixture",
    )
    assert processed.knowledge_staging.graph_scope is DocumentKnowledgeGraphScope.TBOX_CANDIDATE
    assert processed.knowledge_staging.source_profile.extraction_method == (
        "astrograph-markdown-section-v1"
    )
    assert len(processed.crosswalk) == len(processed.retrieval_ingestion.elements)


def test_candidate_kg_content_identity_survives_source_relocation(tmp_path):
    content = "# Wheel observation\n\n## Evidence\n\nCurrent increased at 120 s.\n"
    paths = [tmp_path / "host-a" / "manual.md", tmp_path / "host-b" / "manual.md"]
    for path in paths:
        path.parent.mkdir()
        path.write_text(content, encoding="utf-8")
    source_sha = hashlib.sha256(paths[0].read_bytes()).hexdigest()
    processed = []
    for path in paths:
        descriptor = _descriptor(
            document_id="design:relocatable:fixture",
            source_uri=str(path),
            source_sha256=source_sha,
        )
        processed.append(process_governed_document(
            descriptor=descriptor,
            source_path=path,
            independence_group="fixture-relocation-origin",
            license_id="test-fixture",
        ))

    left, right = [item.knowledge_staging for item in processed]
    assert left.source_profile.source_path != right.source_profile.source_path
    assert left.source_tree.tree_sha256 == right.source_tree.tree_sha256
    assert knowledge_source_profile_set_sha256([left.source_profile]) == (
        knowledge_source_profile_set_sha256([right.source_profile])
    )
    assert left.staging_sha256 == right.staging_sha256
    constructions = []
    for item in processed:
        staging = item.knowledge_staging
        proposal = KnowledgeExtractionProposal(
            source_profile_sha256=knowledge_source_profile_set_sha256(
                [staging.source_profile]
            ),
            nodes=[{
                "node_id": "candidate:relocatable:mechanism",
                "node_type": "FailureMechanism",
                "canonical_name": "Relocatable mechanism",
                "source_fragment_ids": [staging.source_profile.fragments[0].fragment_id],
                "confidence": 0.9,
            }],
            proposer_id="fixture",
            proposer_version="1",
        )
        constructions.append(build_document_candidate_graph(
            processed=item,
            proposal=proposal,
            excluded_review_refreeze_sha256="7" * 64,
            graph_id="graph:relocatable:fixture",
        ))
    first, second = constructions
    assert first.build_id == second.build_id
    assert first.content_sha256 == second.content_sha256
    assert first.bundle_sha256 != second.bundle_sha256


def test_early_design_proposal_descriptor_remains_non_voting(tmp_path):
    path = tmp_path / "design.docx"
    _write_minimal_docx(path)
    source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    descriptor = GovernedDocumentDescriptor(
        document_id="early-design",
        document_type=DocumentType.DESIGN_PROPOSAL,
        visibility_class=DocumentVisibilityClass.DESIGN_PUBLIC,
        source_uri=str(path),
        source_sha256=source_sha,
        review_state=DocumentReviewState.REVIEW_REQUIRED,
        authority=DocumentAuthority.NON_NORMATIVE,
        runtime_vote_allowed=False,
    )
    ingestion, audit = SectionFirstDocumentIngestor(maximum_section_units=1).ingest_docx(
        path,
        source_id="early-design",
        knowledge_version="r001",
        document_descriptor=descriptor,
    )
    assert ingestion.document_descriptor == descriptor
    assert audit.source_sha256 == source_sha
    paragraph = next(chunk for chunk in ingestion.chunks if "proposal" in chunk.text)
    package = PreReasoningRetrievalService().retrieve(
        PreReasoningRetrievalRequest(
            query=DiagnosticRetrievalQuery(
                query_id="q",
                episode_id="e",
                text="proposal",
                document_access_context=DocumentAccessContext(),
            ),
            chunks=[paragraph],
            config=PreReasoningRetrievalConfig(
                require_local_and_ontology=False,
                minimum_local_score=0.0,
            ),
        )
    )
    assert package.local_hits == []
    assert package.document_access_reason_counts == {"DOCUMENT_NOT_REVIEWED": 1}
