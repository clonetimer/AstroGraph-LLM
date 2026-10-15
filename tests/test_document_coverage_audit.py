import csv
import hashlib
import json
from pathlib import Path

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
    audit_document_coverage,
)


SHA = "a" * 64


def _artifact(document_id: str, document_type: DocumentType) -> CorpusArtifact:
    historical = document_type == DocumentType.CLOSURE_REPORT
    visibility = (
        DocumentVisibilityClass.RESOLVED_CASE_TRAIN_ONLY
        if historical
        else DocumentVisibilityClass.OBSERVATION_CAUSAL
    )
    return CorpusArtifact(
        relative_path=f"docs/{document_id}.md",
        sha256=SHA,
        media_type="text/markdown",
        descriptor=GovernedDocumentDescriptor(
            document_id=document_id,
            document_type=document_type,
            visibility_class=visibility,
            source_uri=f"corpus://docs/{document_id}.md",
            source_sha256=SHA,
            review_state=DocumentReviewState.REVIEW_REQUIRED,
            authority=DocumentAuthority.NON_NORMATIVE,
            provenance_group_id="fixture:mc-group:0",
            episode_id="episode:fault" if not historical else None,
            valid_time_end_s=60.0 if not historical else None,
            observable_cutoff_time_s=60.0 if not historical else None,
            contains_root_cause=historical,
            contains_episode_truth=historical,
            runtime_vote_allowed=False,
        ),
    )


def _bundle(sample_id: str, fault_class: str) -> PairedDocumentBundle:
    episode = "episode:fault" if fault_class != "nominal" else "episode:nominal"
    artifacts = [
        _artifact(f"{fault_class}:log", DocumentType.TEST_LOG),
        _artifact(f"{fault_class}:anomaly", DocumentType.ANOMALY_REPORT),
    ]
    artifacts = [
        item.model_copy(
            update={
                "descriptor": item.descriptor.model_copy(update={"episode_id": episode})
            }
        )
        for item in artifacts
    ]
    if fault_class != "nominal":
        artifacts.append(_artifact(f"{fault_class}:closure", DocumentType.CLOSURE_REPORT))
    return PairedDocumentBundle(
        bundle_id=f"bundle:{fault_class}",
        dataset_id="fixture-v1",
        opaque_episode_id=episode,
        source_episode_ref_sha256=hashlib.sha256(sample_id.encode()).hexdigest(),
        telemetry_sha256=SHA,
        provenance_group_id="fixture:mc-group:0",
        split_role=CorpusSplitRole.TRAIN,
        artifacts=artifacts,
    )


def _write_inputs(tmp_path: Path, *, include_nominal: bool = True) -> dict[str, Path]:
    index = tmp_path / "sample_index.csv"
    with index.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "sample_id", "sample_index", "diagnostic_class", "subsystem",
                "mission_id", "fault_injection_phase",
                "calibration",
            ],
        )
        writer.writeheader()
        writer.writerows([
            {
                "sample_id": "fault:0", "sample_index": "0",
                "diagnostic_class": "wheel_fault", "subsystem": "adcs",
                "mission_id": "pointing", "fault_injection_phase": "settled",
                "calibration": "{'severity_band': 'hard_failure'}",
            },
            {
                "sample_id": "nominal:0", "sample_index": "0",
                "diagnostic_class": "nominal", "subsystem": "adcs",
                "mission_id": "pointing", "fault_injection_phase": "nominal",
                "calibration": "{'severity_band': 'nominal'}",
            },
        ])
    manifest = DocumentCorpusManifest(
        corpus_id="fixture",
        release_state="DRAFT",
        policy_sha256=SHA,
        bundles=[_bundle("fault:0", "wheel_fault"), _bundle("nominal:0", "nominal")],
        index=DocumentCorpusIndex(),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    fault_facts = tmp_path / "fault_facts.json"
    fault_facts.write_text(
        json.dumps([{"dataset_id": "fixture.v1", "fault_class_id": "wheel_fault"}]),
        encoding="utf-8",
    )
    nominal_facts = tmp_path / "nominal_facts.json"
    nominal_facts.write_text(
        json.dumps(
            [{"dataset_id": "fixture.v1", "fault_class_id": "nominal_operation"}]
            if include_nominal else []
        ),
        encoding="utf-8",
    )
    return {
        "index": index,
        "manifest": manifest_path,
        "fault_facts": fault_facts,
        "nominal_facts": nominal_facts,
    }


def test_document_coverage_audit_requires_all_class_document_roles(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    report = audit_document_coverage(
        manifest_path=paths["manifest"],
        dataset_index_paths={"fixture-v1": paths["index"]},
        static_dataset_ids={"fixture-v1": "fixture.v1"},
        fault_facts_path=paths["fault_facts"],
        nominal_facts_path=paths["nominal_facts"],
    )
    assert report["status"] == "PASS"
    assert report["datasets"]["fixture-v1"]["class_coverage"] == 1.0
    assert report["datasets"]["fixture-v1"]["document_types"]["coverage"] == 1.0
    assert report["datasets"]["fixture-v1"]["eligible_episode_bundle_coverage"] == 1.0
    assert report["datasets"]["fixture-v1"]["paired_episode_causal_document_coverage"] == 1.0
    assert report["datasets"]["fixture-v1"]["posthoc_bundle_count"] == 0
    assert report["posthoc_telemetry_or_truth_opened"] is False
    assert report["source_sample_identifiers_emitted"] is False


def test_document_coverage_audit_fails_when_nominal_reference_is_missing(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path, include_nominal=False)
    report = audit_document_coverage(
        manifest_path=paths["manifest"],
        dataset_index_paths={"fixture-v1": paths["index"]},
        static_dataset_ids={"fixture-v1": "fixture.v1"},
        fault_facts_path=paths["fault_facts"],
        nominal_facts_path=paths["nominal_facts"],
    )
    assert report["status"] == "FAIL"
    assert report["datasets"]["fixture-v1"]["classes"]["nominal"]["passed"] is False
    assert "fixture-v1:CLASS_DOCUMENT_COVERAGE:nominal" in report["violations"]
