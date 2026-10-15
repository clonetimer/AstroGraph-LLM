import hashlib
import csv
import json

from astrograph.agents.knowledge_curation import KnowledgeExtractionProposal
from astrograph.knowledge.document_graph_gold import build_document_graph_gold_handoff
from astrograph.knowledge.document_graph_gold_evaluation import evaluate_document_graph_gold
from astrograph.knowledge.document_graph_scope_pilot import build_document_graph_scope_pilot
from astrograph.retrieval.document_corpus import (
    CorpusArtifact, CorpusSplitRole, DocumentCorpusIndex, DocumentCorpusManifest, PairedDocumentBundle,
)
from astrograph.retrieval.document_governance import (
    DocumentAuthority, DocumentReviewState, DocumentType, DocumentVisibilityClass, GovernedDocumentDescriptor,
)


class _Backend:
    model_id = "fixture"; model_version = "1"
    def generate_structured(self, *, agent_id, output_model, payload):
        fragment = payload["source_profiles"][0]["fragments"][0]["fragment_id"]
        return KnowledgeExtractionProposal(
            source_profile_sha256=payload["source_profile_sha256"],
            nodes=[{"node_id": "candidate:fixture:state", "node_type": "StateEstimate",
                    "canonical_name": "fixture state", "source_fragment_ids": [fragment],
                    "confidence": 0.9}], proposer_id="fixture", proposer_version="1")


def test_gold_handoff_keeps_human_fields_blank_and_covers_empty_documents(tmp_path):
    corpus = tmp_path / "corpus"; corpus.mkdir()
    artifacts = []
    for key, doc_type in (("obs", DocumentType.TEST_LOG), ("anomaly", DocumentType.ANOMALY_REPORT)):
        path = corpus / f"{key}.md"; path.write_text(f"# {key}\n\ncontent\n")
        descriptor = GovernedDocumentDescriptor(
            document_id=f"doc:{key}", document_type=doc_type,
            visibility_class=DocumentVisibilityClass.OBSERVATION_CAUSAL,
            source_uri=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            review_state=DocumentReviewState.MACHINE_VALIDATED,
            authority=DocumentAuthority.NON_NORMATIVE, provenance_group_id="g",
            episode_id="ep", valid_time_end_s=120, observable_cutoff_time_s=120,
            synthetic_from_simulation=True, runtime_vote_allowed=False)
        artifacts.append(CorpusArtifact(relative_path=path.name, media_type="text/markdown",
                                        sha256=descriptor.source_sha256,
                                        descriptor=descriptor))
    # Add isolated truth and a historical closure required by the frozen selector.
    for key, visibility, doc_type, episode in (
        ("truth", DocumentVisibilityClass.TRUTH_QUARANTINED, DocumentType.CAMPAIGN_REPORT, "ep"),
        ("closure", DocumentVisibilityClass.RESOLVED_CASE_TRAIN_ONLY, DocumentType.CLOSURE_REPORT, "train-ep"),
    ):
        path = corpus / f"{key}.md"; path.write_text(f"# {key}\n\ncontent\n")
        descriptor = GovernedDocumentDescriptor(
            document_id=f"doc:{key}", document_type=doc_type, visibility_class=visibility,
            source_uri=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            review_state=DocumentReviewState.REVIEW_REQUIRED,
            authority=DocumentAuthority.NON_NORMATIVE, provenance_group_id="g", episode_id=episode,
            contains_episode_truth=True, contains_root_cause=key == "closure",
            synthetic_from_simulation=True, runtime_vote_allowed=False)
        artifacts.append(CorpusArtifact(relative_path=path.name, media_type="text/markdown",
                                        sha256=descriptor.source_sha256,
                                        descriptor=descriptor))
    manifest = DocumentCorpusManifest(
        corpus_id="gold-fixture", release_state="FROZEN", policy_sha256="9" * 64,
        bundles=[
            PairedDocumentBundle(bundle_id="dev", dataset_id="d", opaque_episode_id="ep",
                source_episode_ref_sha256="1"*64, telemetry_sha256="2"*64,
                provenance_group_id="g", split_role=CorpusSplitRole.DEVELOPMENT, artifacts=artifacts[:3]),
            PairedDocumentBundle(bundle_id="train", dataset_id="d", opaque_episode_id="train-ep",
                source_episode_ref_sha256="3"*64, telemetry_sha256="4"*64,
                provenance_group_id="g", split_role=CorpusSplitRole.TRAIN, artifacts=[artifacts[3]]),
        ], index=DocumentCorpusIndex(runtime_document_ids=["doc:obs", "doc:anomaly"],
                                    evaluator_truth_document_ids=["doc:truth"]))
    manifest_path = corpus / "manifest.json"; manifest_path.write_text(manifest.model_dump_json())
    config = tmp_path / "config.json"; config.write_text(json.dumps({
        "schema_version":"astrograph-document-graph-scope-pilot.v1", "pilot_id":"g",
        "source_corpus_id":"gold-fixture", "datasets":{"d":{"episode_split_role":"DEVELOPMENT",
        "episode_provenance_group_id":"g","historical_split_role":"TRAIN","historical_provenance_group_id":"g"}},
        "selection":{"causal_cutoff_seconds":120,"episode_document_types":["TEST_LOG","ANOMALY_REPORT"],
        "historical_document_type":"CLOSURE_REPORT","quarantine_document_type":"CAMPAIGN_REPORT"},
        "claim_boundary":"fixture"}))
    pilot = tmp_path / "pilot"
    build_document_graph_scope_pilot(config_path=config, manifest_path=manifest_path, corpus_root=corpus,
        output_root=pilot, backend=_Backend(), excluded_review_refreeze_sha256="8"*64)
    handoff = build_document_graph_gold_handoff(pilot_root=pilot, output_root=tmp_path / "gold")
    assert handoff["document_count"] == 4
    assert handoff["candidate_review_item_count"] > 0
    assert handoff["fragment_gold_item_count"] >= 4
    assert handoff["human_decision_count"] == 0
    assert handoff["precision_evaluation_ready"] is False
    assert handoff["recall_evaluation_ready"] is False
    blocked = evaluate_document_graph_gold(
        pilot_root=pilot, handoff_root=tmp_path / "gold",
        output_path=tmp_path / "blocked-evaluation.json",
    )
    assert blocked["status"] == "BLOCKED"
    assert blocked["strict_exact_metrics"]["precision"] is None

    candidate_path = tmp_path / "gold/candidate_adjudication.csv"
    with candidate_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream); candidate_fields = list(reader.fieldnames or []); candidates = list(reader)
    gold_by_document = {}
    for row in candidates:
        row.update({"decision": "CORRECT", "reviewer": "human",
                    "review_comment": "verified against fragment"})
        candidate = json.loads(row["candidate_json"])
        target = gold_by_document.setdefault(row["document_id"], {"nodes": [], "relations": []})
        if row["item_kind"] == "NODE":
            target["nodes"].append({"node_type": candidate["node_type"],
                                    "canonical_name": candidate["canonical_name"]})
        else:
            target["relations"].append({"relation_type": candidate["relation_type"],
                "source_id": candidate["source_id"], "target_id": candidate["target_id"]})
    with candidate_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=candidate_fields); writer.writeheader(); writer.writerows(candidates)
    fragment_path = tmp_path / "gold/fragment_gold_annotations.csv"
    with fragment_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream); fragment_fields = list(reader.fieldnames or []); fragments = list(reader)
    assigned = set()
    for row in fragments:
        values = gold_by_document.get(row["document_id"], {"nodes": [], "relations": []})
        first = row["document_id"] not in assigned
        row.update({"gold_nodes_json": json.dumps(values["nodes"] if first else []),
                    "gold_relations_json": json.dumps(values["relations"] if first else []),
                    "annotation_complete": "true", "reviewer": "human",
                    "review_comment": "complete fragment review"})
        assigned.add(row["document_id"])
    with fragment_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fragment_fields); writer.writeheader(); writer.writerows(fragments)
    evaluated = evaluate_document_graph_gold(
        pilot_root=pilot, handoff_root=tmp_path / "gold",
        output_path=tmp_path / "evaluation.json",
    )
    assert evaluated["status"] == "PASS"
    assert evaluated["strict_exact_metrics"]["precision"] == 1.0
    assert evaluated["strict_exact_metrics"]["recall"] == 1.0
