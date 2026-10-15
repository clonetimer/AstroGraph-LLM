import hashlib

import pytest
from pydantic import ValidationError

from astrograph.retrieval import (
    CorpusArtifact,
    CorpusSplitRole,
    DocumentAuthority,
    DocumentCorpusIndex,
    DocumentCorpusManifest,
    DocumentReviewState,
    DocumentType,
    DocumentVisibilityClass,
    GovernedDocumentDescriptor,
    PairedDocumentBundle,
    audit_runtime_index,
)


SHA = "b" * 64


def _artifact(document_id="log:opaque", visibility=DocumentVisibilityClass.OBSERVATION_CAUSAL):
    truth = visibility == DocumentVisibilityClass.TRUTH_QUARANTINED
    descriptor = GovernedDocumentDescriptor(
        document_id=document_id,
        document_type=DocumentType.TEST_LOG,
        visibility_class=visibility,
        source_uri=f"corpus/{document_id}.md",
        source_sha256=SHA,
        review_state=DocumentReviewState.MACHINE_VALIDATED,
        authority=DocumentAuthority.NON_NORMATIVE,
        episode_id="episode:opaque" if visibility == DocumentVisibilityClass.OBSERVATION_CAUSAL else None,
        observable_cutoff_time_s=60.0 if visibility == DocumentVisibilityClass.OBSERVATION_CAUSAL else None,
        valid_time_end_s=60.0 if visibility == DocumentVisibilityClass.OBSERVATION_CAUSAL else None,
        contains_episode_truth=truth,
    )
    return CorpusArtifact(
        relative_path=f"artifacts/{hashlib.sha256(document_id.encode()).hexdigest()}.md",
        sha256=SHA,
        media_type="text/markdown",
        descriptor=descriptor,
    )


def _bundle(artifacts):
    return PairedDocumentBundle(
        bundle_id="bundle:opaque",
        dataset_id="adcs-v522",
        opaque_episode_id="episode:opaque",
        source_episode_ref_sha256=SHA,
        telemetry_sha256=SHA,
        provenance_group_id="group:0",
        split_role=CorpusSplitRole.TRAIN,
        artifacts=artifacts,
    )


def test_runtime_and_evaluator_indexes_are_physically_disjoint():
    with pytest.raises(ValidationError, match="must be disjoint"):
        DocumentCorpusIndex(
            runtime_document_ids=["doc:1"],
            evaluator_truth_document_ids=["doc:1"],
        )


def test_causal_document_must_bind_to_opaque_episode():
    artifact = _artifact()
    payload = artifact.descriptor.model_dump()
    payload["episode_id"] = "episode:other"
    mismatched = artifact.model_copy(
        update={"descriptor": GovernedDocumentDescriptor(**payload)}
    )
    with pytest.raises(ValidationError, match="opaque bundle episode"):
        _bundle([mismatched])


def test_runtime_index_audit_blocks_truth_and_path_proxies():
    truth = _artifact("truth:1", DocumentVisibilityClass.TRUTH_QUARANTINED)
    proxy = _artifact("condition_id:wheel_fault")
    audit = audit_runtime_index([truth, proxy], forbidden_tokens=("condition_id",))
    assert audit.passed is False
    assert any("TRUTH_IN_RUNTIME_INDEX" in item for item in audit.violations)
    assert any("FORBIDDEN_TOKEN:condition_id" in item for item in audit.violations)


def test_manifest_rejects_truth_in_runtime_index():
    observable = _artifact()
    truth = _artifact("truth:1", DocumentVisibilityClass.TRUTH_QUARANTINED)
    bundle = _bundle([observable, truth])
    with pytest.raises(ValidationError, match="quarantined truth"):
        DocumentCorpusManifest(
            corpus_id="pilot-r001",
            release_state="DRAFT",
            policy_sha256=SHA,
            bundles=[bundle],
            index=DocumentCorpusIndex(runtime_document_ids=["truth:1"]),
        )


def test_manifest_accepts_observation_runtime_and_truth_evaluator_split():
    observable = _artifact()
    truth = _artifact("truth:1", DocumentVisibilityClass.TRUTH_QUARANTINED)
    manifest = DocumentCorpusManifest(
        corpus_id="pilot-r001",
        release_state="DRAFT",
        policy_sha256=SHA,
        bundles=[_bundle([observable, truth])],
        index=DocumentCorpusIndex(
            runtime_document_ids=["log:opaque"],
            evaluator_truth_document_ids=["truth:1"],
        ),
    )
    assert manifest.index.runtime_document_ids == ["log:opaque"]


def test_reviewed_cross_group_historical_truth_may_be_indexed_but_not_same_group_access():
    descriptor = GovernedDocumentDescriptor(
        document_id="closure:historical",
        document_type=DocumentType.CLOSURE_REPORT,
        visibility_class=DocumentVisibilityClass.RESOLVED_CASE_TRAIN_ONLY,
        source_uri="corpus://historical/opaque.md",
        source_sha256=SHA,
        review_state=DocumentReviewState.HUMAN_REVIEWED,
        authority=DocumentAuthority.ENGINEERING_REFERENCE,
        provenance_group_id="group:0",
        contains_root_cause=True,
        contains_episode_truth=True,
    )
    artifact = CorpusArtifact(
        relative_path="historical/opaque.md",
        sha256=SHA,
        media_type="text/markdown",
        descriptor=descriptor,
    )
    assert audit_runtime_index([artifact]).passed is True
