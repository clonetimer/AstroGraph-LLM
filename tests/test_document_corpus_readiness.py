import hashlib
import json
from pathlib import Path

import pytest

from astrograph.knowledge.tbox_claim_release import build_approved_tbox_claim_release
from astrograph.retrieval.document_corpus_readiness import assess_document_corpus_readiness


ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _fixture(tmp_path: Path, *, include_metadata: bool = True) -> Path:
    descriptor = {
        "document_type": "TEST_LOG", "visibility_class": "OBSERVATION_CAUSAL",
        "review_state": "MACHINE_VALIDATED",
    }
    if include_metadata:
        descriptor.update({
            "license_id": "LicenseRef-Test", "usage_boundary": "RESEARCH_ONLY",
            "template_version": "fixture-v1",
        })
    manifest = {
        "corpus_id": "fixture", "release_state": "DRAFT_MACHINE_GENERATED_REVIEW_REQUIRED",
        "bundles": [{"artifacts": [{
            "descriptor": descriptor
        }]}],
    }
    payloads = {
        "coverage": {"status": "PASS", "violations": []},
        "corpus_build": {},
        "corpus_audit": {"status": "PASS", "violations": [], "posthoc_telemetry_or_truth_opened": False},
        "rebuild": {"status": "PASS", "manifest_semantically_identical": True,
                    "artifact_hash_mismatch_count": 0},
        "sidechannel": {"status": "PASS", "violations": []},
        "multimodal": {"human_gold_complete": False},
        "chunking": {"status": "COMPLETED_MACHINE_SOURCE_SPAN_ONLY",
                     "promotion_decision": "NONE_MACHINE_BENCHMARK_NOT_HUMAN_GOLD"},
        "early_design_gold": {"status": "UNREVIEWED_TEMPLATE"},
        "scan_review": {"representation_reviewed": False},
        "corpus_manifest": manifest,
    }
    manifest_sha = _write(tmp_path / "corpus_manifest.json", manifest)
    payloads["corpus_build"] = {
        "manifest_sha256": manifest_sha, "posthoc_telemetry_or_truth_opened": False,
    }
    payloads["corpus_audit"]["manifest_sha256"] = manifest_sha
    evidence = {}
    for name, payload in payloads.items():
        path = tmp_path / f"{name}.json"
        digest = manifest_sha if name == "corpus_manifest" else _write(path, payload)
        evidence[name] = {"path": path.name, "sha256": digest}
    spec = tmp_path / "spec.json"
    _write(spec, {
        "schema_version": "astrograph-p05-corpus-readiness-evidence.v1",
        "assessment_id": "fixture-r001", "evidence": evidence,
    })
    return spec


def test_readiness_reports_machine_passes_and_human_or_missing_evidence(tmp_path: Path) -> None:
    spec = _fixture(tmp_path)
    output = tmp_path / "assessment.json"
    receipt = assess_document_corpus_readiness(
        project_root=tmp_path, evidence_spec_path=spec, output_path=output
    )
    statuses = {gate["gate_id"]: gate["status"] for gate in receipt["gates"]}
    assert receipt["status"] == "BLOCKED"
    assert receipt["ready_for_p1"] is False
    assert statuses["P05-01"] == "PASS"
    assert statuses["P05-02"] == "PASS"
    assert statuses["P05-04"] == "PASS"
    assert statuses["P05-05"] == "PASS"
    assert statuses["P05-06"] == "BLOCKED_HUMAN_GOLD_REQUIRED"
    assert statuses["P05-07"] == "BLOCKED_HUMAN_RETRIEVAL_GOLD_REQUIRED"
    assert statuses["P05-09"] == "BLOCKED_RELEASE_FREEZE_REQUIRED"
    assert receipt["active_graph_modified"] is False

    with pytest.raises(ValueError, match="refusing to overwrite"):
        assess_document_corpus_readiness(
            project_root=tmp_path, evidence_spec_path=spec, output_path=output
        )


def test_readiness_rejects_changed_or_incomplete_evidence(tmp_path: Path) -> None:
    spec = _fixture(tmp_path)
    (tmp_path / "coverage.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch: coverage"):
        assess_document_corpus_readiness(
            project_root=tmp_path, evidence_spec_path=spec,
            output_path=tmp_path / "assessment.json",
        )
    assert not (tmp_path / "assessment.json").exists()

    spec = _fixture(tmp_path / "second")
    data = json.loads(spec.read_text(encoding="utf-8"))
    data["evidence"].pop("scan_review")
    spec.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="exact required evidence set"):
        assess_document_corpus_readiness(
            project_root=tmp_path / "second", evidence_spec_path=spec,
            output_path=tmp_path / "second/assessment.json",
        )


def test_readiness_distinguishes_missing_metadata_from_release_freeze(tmp_path: Path) -> None:
    spec = _fixture(tmp_path, include_metadata=False)
    receipt = assess_document_corpus_readiness(
        project_root=tmp_path, evidence_spec_path=spec,
        output_path=tmp_path / "assessment.json",
    )
    gate = next(item for item in receipt["gates"] if item["gate_id"] == "P05-09")
    assert gate["status"] == "BLOCKED_METADATA_REQUIRED"


def test_readiness_v2_tracks_controlled_rewrite_human_review(tmp_path: Path) -> None:
    spec = _fixture(tmp_path)
    data = json.loads(spec.read_text(encoding="utf-8"))
    rewrite_body = {
        "status": "HUMAN_REVIEW_REQUIRED", "document_count": 31, "block_count": 161,
    }
    rewrite = {**rewrite_body, "receipt_sha256": _canonical_sha(rewrite_body)}
    rewrite_path = tmp_path / "controlled_rewrite.json"
    rewrite_sha = _write(rewrite_path, rewrite)
    review_path = tmp_path / "controlled_rewrite_review.json"
    review_sha = _write(review_path, {
        "status": "BLOCKED", "rewrite_pack_receipt_file_sha256": rewrite_sha,
        "unsupported_claim_rate": None,
    })
    data["schema_version"] = "astrograph-p05-corpus-readiness-evidence.v2"
    data["evidence"].update({
        "controlled_rewrite": {"path": rewrite_path.name, "sha256": rewrite_sha},
        "controlled_rewrite_review": {"path": review_path.name, "sha256": review_sha},
    })
    spec.write_text(json.dumps(data), encoding="utf-8")
    receipt = assess_document_corpus_readiness(
        project_root=tmp_path, evidence_spec_path=spec,
        output_path=tmp_path / "assessment.json",
    )
    gate = next(item for item in receipt["gates"] if item["gate_id"] == "P05-08")
    assert gate["status"] == "BLOCKED_HUMAN_REWRITE_REVIEW_REQUIRED"
    assert gate["evidence"]["candidate_rewrite_document_count"] == 31


def test_readiness_v3_accepts_only_scoped_source_bound_tbox_claims(tmp_path: Path) -> None:
    spec = _fixture(tmp_path)
    data = json.loads(spec.read_text(encoding="utf-8"))
    rewrite_body = {"status": "HUMAN_REVIEW_REQUIRED", "document_count": 31, "block_count": 161}
    rewrite = {**rewrite_body, "receipt_sha256": _canonical_sha(rewrite_body)}
    rewrite_path = tmp_path / "controlled_rewrite.json"
    rewrite_sha = _write(rewrite_path, rewrite)
    review_path = tmp_path / "controlled_rewrite_review.json"
    review_sha = _write(review_path, {
        "status": "BLOCKED", "rewrite_pack_receipt_file_sha256": rewrite_sha,
        "unsupported_claim_rate": None,
    })
    claims_path = tmp_path / "approved_claims.json"
    build_approved_tbox_claim_release(
        graph_path=ROOT / "governance/evidence/ai_native/reusable_kg_construction_20260901/candidate_graph.json",
        construction_bundle_path=ROOT / "governance/evidence/ai_native/reusable_kg_construction_20260901/construction_bundle.json",
        approval_path=ROOT / "governance/evidence/ai_native/tbox_vocabulary_approval_20260902/approval.json",
        output_path=claims_path,
    )
    claims_sha = hashlib.sha256(claims_path.read_bytes()).hexdigest()
    data["schema_version"] = "astrograph-p05-corpus-readiness-evidence.v3"
    data["evidence"].update({
        "controlled_rewrite": {"path": rewrite_path.name, "sha256": rewrite_sha},
        "controlled_rewrite_review": {"path": review_path.name, "sha256": review_sha},
        "approved_claims": {"path": claims_path.name, "sha256": claims_sha},
    })
    spec.write_text(json.dumps(data), encoding="utf-8")

    receipt = assess_document_corpus_readiness(
        project_root=tmp_path, evidence_spec_path=spec,
        output_path=tmp_path / "assessment.json",
    )
    gate = next(item for item in receipt["gates"] if item["gate_id"] == "P05-03")
    assert gate["status"] == "PASS"
    assert gate["evidence"]["approved_publishable_claim_count"] == 22
    assert gate["evidence"]["approved_claim_issue_count"] == 0


def test_readiness_distinguishes_complete_gold_that_misses_promotion_thresholds(
    tmp_path: Path,
) -> None:
    spec = _fixture(tmp_path)
    chunking_path = tmp_path / "chunking.json"
    chunking = {
        "status": "COMPLETE",
        "human_gold_complete": True,
        "promotion_decision": "NO_PROMOTION_HUMAN_GOLD_THRESHOLDS_NOT_MET",
    }
    chunking_sha = _write(chunking_path, chunking)
    data = json.loads(spec.read_text(encoding="utf-8"))
    data["evidence"]["chunking"] = {
        "path": chunking_path.name,
        "sha256": chunking_sha,
    }
    spec.write_text(json.dumps(data), encoding="utf-8")

    receipt = assess_document_corpus_readiness(
        project_root=tmp_path,
        evidence_spec_path=spec,
        output_path=tmp_path / "assessment.json",
    )
    gate = next(item for item in receipt["gates"] if item["gate_id"] == "P05-07")
    assert gate["status"] == "BLOCKED_RETRIEVAL_PROMOTION_THRESHOLDS_NOT_MET"
    assert gate["evidence"]["human_gold_complete"] is True


@pytest.mark.parametrize(
    "promotion_decision",
    ["PROMOTE_SECTION_AWARE_HUMAN_GOLD", "PROMOTE_SECTION_CONTEXT_AWARE_HUMAN_GOLD"],
)
def test_readiness_accepts_both_registered_section_aware_protocol_names(
    tmp_path: Path, promotion_decision: str,
) -> None:
    spec = _fixture(tmp_path)
    chunking_path = tmp_path / "chunking.json"
    chunking_sha = _write(chunking_path, {
        "status": "COMPLETE",
        "human_gold_complete": True,
        "promotion_decision": promotion_decision,
    })
    data = json.loads(spec.read_text(encoding="utf-8"))
    data["evidence"]["chunking"] = {
        "path": chunking_path.name,
        "sha256": chunking_sha,
    }
    spec.write_text(json.dumps(data), encoding="utf-8")

    receipt = assess_document_corpus_readiness(
        project_root=tmp_path,
        evidence_spec_path=spec,
        output_path=tmp_path / "assessment.json",
    )
    gate = next(item for item in receipt["gates"] if item["gate_id"] == "P05-07")
    assert gate["status"] == "PASS"
