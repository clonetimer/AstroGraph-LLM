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
    run_document_label_sidechannel_attack,
)


SHA = "b" * 64


def _opaque(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def test_registered_identifier_and_layout_attack_uses_development_only(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    index_rows = []
    bundles = []
    runtime_ids = []
    for sequence, (group, fault_class) in enumerate(
        [(0, "fault_a"), (1, "fault_a"), (0, "fault_b"), (1, "fault_b"),
         (3, "fault_a"), (3, "fault_b")]
    ):
        sample_id = f"sample:{sequence}"
        token = _opaque(sample_id)
        episode_id = f"episode:{token}"
        artifacts = []
        for kind, document_type in (
            ("observation", DocumentType.TEST_LOG),
            ("anomaly", DocumentType.ANOMALY_REPORT),
        ):
            document_id = f"doc:{kind}:{token}"
            relative = f"runtime/{token}/{kind}_60s.md"
            path = corpus / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Runtime record\n\n" + (" " * 200), encoding="utf-8")
            artifacts.append(CorpusArtifact(
                relative_path=relative,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                media_type="text/markdown",
                descriptor=GovernedDocumentDescriptor(
                    document_id=document_id,
                    document_type=document_type,
                    visibility_class=DocumentVisibilityClass.OBSERVATION_CAUSAL,
                    source_uri=f"corpus://{relative}",
                    source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    review_state=DocumentReviewState.MACHINE_VALIDATED,
                    authority=DocumentAuthority.NON_NORMATIVE,
                    provenance_group_id=f"fixture:mc-group:{group}",
                    episode_id=episode_id,
                    valid_time_end_s=60.0,
                    observable_cutoff_time_s=60.0,
                    runtime_vote_allowed=False,
                ),
            ))
            runtime_ids.append(document_id)
        bundles.append(PairedDocumentBundle(
            bundle_id=f"bundle:{token}",
            dataset_id="fixture",
            opaque_episode_id=episode_id,
            source_episode_ref_sha256=hashlib.sha256(sample_id.encode()).hexdigest(),
            telemetry_sha256=SHA,
            provenance_group_id=f"fixture:mc-group:{group}",
            split_role=(CorpusSplitRole.TRAIN if group in {0, 1} else CorpusSplitRole.DEVELOPMENT),
            artifacts=artifacts,
        ))
        index_rows.append({
            "sample_id": sample_id,
            "sample_index": str(group),
            "diagnostic_class": fault_class,
        })
    manifest_path = corpus / "manifest.json"
    manifest_path.write_text(
        DocumentCorpusManifest(
            corpus_id="fixture",
            release_state="DRAFT",
            policy_sha256=SHA,
            bundles=bundles,
            index=DocumentCorpusIndex(runtime_document_ids=runtime_ids),
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    index_path = tmp_path / "sample_index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)
    prereg_path = tmp_path / "prereg.json"
    prereg_path.write_text(json.dumps({
        "frozen_before_execution": True,
        "random_seed": 7,
        "forbidden_groups": [4, 9],
        "experiment_id": "fixture",
        "claim_boundary": "fixture only",
    }), encoding="utf-8")

    report = run_document_label_sidechannel_attack(
        preregistration_path=prereg_path,
        manifest_path=manifest_path,
        corpus_root=corpus,
        dataset_index_paths={"fixture": index_path},
    )

    assert report["status"] == "PASS"
    assert report["datasets"]["fixture"]["train_count"] == 4
    assert report["datasets"]["fixture"]["evaluation_count"] == 2
    assert report["posthoc_telemetry_labels_or_truth_opened"] is False
    assert report["document_body_tokens_used"] is False
