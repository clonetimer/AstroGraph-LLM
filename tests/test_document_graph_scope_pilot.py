import hashlib
import json
from pathlib import Path

from astrograph.agents.knowledge_curation import KnowledgeExtractionProposal
from astrograph.knowledge.document_graph_scope_pilot import build_document_graph_scope_pilot
from astrograph.retrieval.document_corpus import (
    CorpusArtifact,
    CorpusSplitRole,
    DocumentCorpusIndex,
    DocumentCorpusManifest,
    PairedDocumentBundle,
)
from astrograph.retrieval.document_governance import (
    DocumentAuthority,
    DocumentReviewState,
    DocumentType,
    DocumentVisibilityClass,
    GovernedDocumentDescriptor,
)


class _Backend:
    model_id = "scope-pilot-fixture"
    model_version = "1"

    def generate_structured(self, *, agent_id, output_model, payload):
        allowed = set(payload["allowed_node_types"])
        if "FailureMechanism" in allowed and "DiagnosticHypothesis" in allowed:
            node_type = "FailureMechanism"  # historical
        elif "DiagnosticHypothesis" in allowed:
            node_type = "DiagnosticHypothesis"  # quarantine
        else:
            node_type = "EvidenceSource"  # episode
        fragment = payload["source_profiles"][0]["fragments"][0]["fragment_id"]
        return KnowledgeExtractionProposal(
            source_profile_sha256=payload["source_profile_sha256"],
            nodes=[{
                "node_id": f"candidate:fixture:{node_type}",
                "node_type": node_type,
                "canonical_name": f"Fixture {node_type}",
                "source_fragment_ids": [fragment],
                "confidence": 0.9,
            }],
            proposer_id="fixture", proposer_version="1",
        )


def _artifact(root: Path, relative: str, descriptor: GovernedDocumentDescriptor) -> CorpusArtifact:
    path = root / relative
    return CorpusArtifact(
        relative_path=relative,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        media_type="application/json" if path.suffix == ".json" else "text/markdown",
        descriptor=descriptor,
    )


def test_scope_pilot_keeps_runtime_historical_and_truth_graphs_separate(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    paths = {
        "obs": "runtime/dev/observation_120s.md",
        "anomaly": "runtime/dev/anomaly_120s.md",
        "truth": "quarantine/dev/truth.json",
        "closure": "historical/train/closure.md",
    }
    for key, relative in paths.items():
        path = corpus / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if key == "truth":
            path.write_text(json.dumps({"episode": "episode:dev", "label": "fault_a"}), encoding="utf-8")
        else:
            path.write_text(f"# {key}\n\nFixture {key} content.\n", encoding="utf-8")

    def descriptor(key: str) -> GovernedDocumentDescriptor:
        path = corpus / paths[key]
        common = dict(
            document_id=f"doc:{key}", source_uri=f"corpus://{paths[key]}",
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            authority=DocumentAuthority.NON_NORMATIVE, provenance_group_id="fixture:group",
            episode_id="episode:dev" if key != "closure" else "episode:train",
            synthetic_from_simulation=True, runtime_vote_allowed=False,
        )
        if key in {"obs", "anomaly"}:
            return GovernedDocumentDescriptor(
                **common,
                document_type=DocumentType.TEST_LOG if key == "obs" else DocumentType.ANOMALY_REPORT,
                visibility_class=DocumentVisibilityClass.OBSERVATION_CAUSAL,
                review_state=DocumentReviewState.MACHINE_VALIDATED,
                valid_time_end_s=120, observable_cutoff_time_s=120,
            )
        if key == "truth":
            return GovernedDocumentDescriptor(
                **common, document_type=DocumentType.CAMPAIGN_REPORT,
                visibility_class=DocumentVisibilityClass.TRUTH_QUARANTINED,
                review_state=DocumentReviewState.MACHINE_VALIDATED,
                contains_episode_truth=True,
            )
        return GovernedDocumentDescriptor(
            **common, document_type=DocumentType.CLOSURE_REPORT,
            visibility_class=DocumentVisibilityClass.RESOLVED_CASE_TRAIN_ONLY,
            review_state=DocumentReviewState.REVIEW_REQUIRED,
            contains_root_cause=True, contains_episode_truth=True,
        )

    artifacts = {key: _artifact(corpus, relative, descriptor(key)) for key, relative in paths.items()}
    dev = PairedDocumentBundle(
        bundle_id="bundle:dev", dataset_id="fixture", opaque_episode_id="episode:dev",
        source_episode_ref_sha256="1" * 64, telemetry_sha256="2" * 64,
        provenance_group_id="fixture:group", split_role=CorpusSplitRole.DEVELOPMENT,
        artifacts=[artifacts["obs"], artifacts["anomaly"], artifacts["truth"]],
    )
    train = PairedDocumentBundle(
        bundle_id="bundle:train", dataset_id="fixture", opaque_episode_id="episode:train",
        source_episode_ref_sha256="3" * 64, telemetry_sha256="4" * 64,
        provenance_group_id="fixture:group", split_role=CorpusSplitRole.TRAIN,
        artifacts=[artifacts["closure"]],
    )
    manifest = DocumentCorpusManifest(
        corpus_id="fixture-corpus", release_state="FROZEN", policy_sha256="9" * 64,
        bundles=[dev, train],
        index=DocumentCorpusIndex(
            runtime_document_ids=["doc:obs", "doc:anomaly"],
            evaluator_truth_document_ids=["doc:truth"],
        ),
    )
    manifest_path = corpus / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "schema_version": "astrograph-document-graph-scope-pilot.v1",
        "pilot_id": "fixture", "source_corpus_id": "fixture-corpus",
        "datasets": {"fixture": {
            "episode_split_role": "DEVELOPMENT", "episode_provenance_group_id": "fixture:group",
            "historical_split_role": "TRAIN", "historical_provenance_group_id": "fixture:group",
        }},
        "selection": {
            "causal_cutoff_seconds": 120,
            "episode_document_types": ["TEST_LOG", "ANOMALY_REPORT"],
            "historical_document_type": "CLOSURE_REPORT",
            "quarantine_document_type": "CAMPAIGN_REPORT",
        },
        "claim_boundary": "fixture only",
    }), encoding="utf-8")
    receipt = build_document_graph_scope_pilot(
        config_path=config, manifest_path=manifest_path, corpus_root=corpus,
        output_root=tmp_path / "output", backend=_Backend(),
        excluded_review_refreeze_sha256="8" * 64, max_workers=2,
    )
    assert receipt["selected_document_count"] == 4
    assert receipt["lane_counts"] == {"episode": 2, "historical": 1, "quarantine": 1}
    assert receipt["candidate_ready_count"] == 4
    assert receipt["llm_raw_candidate_counts"]["nodes"] == 4
    assert receipt["deterministic_scaffold_counts"]["nodes"] == 8
    assert receipt["deterministic_content_extraction_counts"] == {
        "nodes": 0,
        "relations": 0,
    }
    assert receipt["runtime_evaluator_overlap"] == []
    assert receipt["portable_content_identity_counts"] == {
        "document_processing": 3,
        "construction": 4,
        "missing": 0,
    }
    assert receipt["selection_used_document_contents"] is False
    by_lane = {row["lane"]: row for row in receipt["results"]}
    assert by_lane["quarantine"]["graph_scope"] == "QUARANTINE_EVALUATOR_CANDIDATE"
    assert by_lane["historical"]["graph_scope"] == "HISTORICAL_CASE_CANDIDATE"
    assert all(not row["publication_allowed"] and not row["runtime_vote_allowed"] for row in receipt["results"])
