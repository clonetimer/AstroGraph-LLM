"""Fail-closed assessment of the ten P0.5 document-corpus acceptance gates."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .document_governance import DocumentEvidenceClaim


REQUIRED_EVIDENCE = {
    "coverage", "corpus_build", "corpus_audit", "rebuild", "sidechannel",
    "multimodal", "chunking", "early_design_gold", "scan_review", "corpus_manifest",
}
CONTROLLED_REWRITE_EVIDENCE = {"controlled_rewrite", "controlled_rewrite_review"}
APPROVED_CLAIM_EVIDENCE = {"approved_claims"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _gate(gate_id: str, title: str, status: str, evidence: dict[str, Any]) -> dict[str, Any]:
    return {"gate_id": gate_id, "title": title, "status": status, "evidence": evidence}


def assess_document_corpus_readiness(
    *, project_root: str | Path, evidence_spec_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    spec_path = Path(evidence_spec_path).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise ValueError("refusing to overwrite P0.5 readiness assessment")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    schema_version = spec.get("schema_version")
    if schema_version not in {
        "astrograph-p05-corpus-readiness-evidence.v1",
        "astrograph-p05-corpus-readiness-evidence.v2",
        "astrograph-p05-corpus-readiness-evidence.v3",
    }:
        raise ValueError("unsupported P0.5 evidence specification")
    entries = spec.get("evidence", {})
    # Copy the module-level contracts: using ``|=`` on the shared set would
    # silently make a later v1 assessment require v2/v3 evidence.
    expected_evidence = set(REQUIRED_EVIDENCE)
    if schema_version.endswith((".v2", ".v3")):
        expected_evidence |= CONTROLLED_REWRITE_EVIDENCE
    if schema_version.endswith(".v3"):
        expected_evidence |= APPROVED_CLAIM_EVIDENCE
    if set(entries) != expected_evidence:
        raise ValueError("P0.5 evidence specification must contain the exact required evidence set")

    loaded: dict[str, Any] = {}
    evidence_files: dict[str, dict[str, str]] = {}
    for name in sorted(expected_evidence):
        entry = entries[name]
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"P0.5 evidence path is missing or outside project root: {name}")
        actual = _sha(path)
        if actual != entry["sha256"]:
            raise ValueError(f"P0.5 evidence hash mismatch: {name}")
        loaded[name] = json.loads(path.read_text(encoding="utf-8"))
        evidence_files[name] = {"path": entry["path"], "sha256": actual}

    manifest = loaded["corpus_manifest"]
    build = loaded["corpus_build"]
    audit = loaded["corpus_audit"]
    if _sha(root / entries["corpus_manifest"]["path"]) != build.get("manifest_sha256"):
        raise ValueError("corpus build receipt and manifest identity differ")
    if audit.get("manifest_sha256") != build.get("manifest_sha256"):
        raise ValueError("corpus audit and build receipt identities differ")

    bundles = manifest.get("bundles", [])
    document_types: Counter[str] = Counter()
    bundles_without_causal_log = 0
    publishable_claim_count = 0
    descriptors_missing_license = 0
    llm_rewrite_document_count = 0
    for bundle in bundles:
        has_causal_log = False
        for artifact in bundle.get("artifacts", []):
            descriptor = artifact.get("descriptor", {})
            document_type = str(descriptor.get("document_type", ""))
            document_types[document_type] += 1
            if document_type == "TEST_LOG" and descriptor.get("visibility_class") == "OBSERVATION_CAUSAL":
                has_causal_log = True
            if descriptor.get("review_state") in {"HUMAN_APPROVED", "PUBLISHED"}:
                publishable_claim_count += len(artifact.get("claims", []))
            if not descriptor.get("license_id") or not descriptor.get("usage_boundary") or not descriptor.get("template_version"):
                descriptors_missing_license += 1
            if descriptor.get("generation_mode") == "LLM_CONTROLLED_REWRITE":
                llm_rewrite_document_count += 1
        if not has_causal_log:
            bundles_without_causal_log += 1

    coverage = loaded["coverage"]
    sidechannel = loaded["sidechannel"]
    multimodal = loaded["multimodal"]
    chunking = loaded["chunking"]
    early_gold = loaded["early_design_gold"]
    scan_review = loaded["scan_review"]
    rebuild = loaded["rebuild"]
    rewrite = loaded.get("controlled_rewrite")
    rewrite_review = loaded.get("controlled_rewrite_review")
    approved_claims = loaded.get("approved_claims")
    if rewrite is not None:
        rewrite_body = {key: value for key, value in rewrite.items() if key != "receipt_sha256"}
        if _canonical_sha(rewrite_body) != rewrite.get("receipt_sha256"):
            raise ValueError("controlled rewrite semantic receipt hash mismatch")
    if rewrite_review is not None and rewrite is not None:
        if rewrite_review.get("rewrite_pack_receipt_file_sha256") != entries["controlled_rewrite"]["sha256"]:
            raise ValueError("controlled rewrite review is not bound to the supplied rewrite pack")
    approved_claim_count = 0
    approved_claim_issues: list[str] = []
    approved_claim_status = None
    if approved_claims is not None:
        claim_body = {
            key: value for key, value in approved_claims.items() if key != "release_sha256"
        }
        if _canonical_sha(claim_body) != approved_claims.get("release_sha256"):
            raise ValueError("approved document claim release semantic hash mismatch")
        approved_claim_status = approved_claims.get("status")
        raw_claims = approved_claims.get("claims", [])
        raw_bindings = approved_claims.get("source_bindings", [])
        if approved_claims.get("association_semantics") != "TBOX_ASSOCIATION_ONLY":
            approved_claim_issues.append("INVALID_ASSOCIATION_SEMANTICS")
        if len(raw_claims) != 22:
            approved_claim_issues.append("APPROVED_TBOX_CLAIM_COUNT_MUST_BE_22")
        for flag in (
            "active_graph_publication_allowed",
            "episode_evidence_approved",
            "actual_causal_propagation_approved",
            "runtime_vote_allowed",
            "active_graph_modified",
        ):
            if approved_claims.get(flag) is not False:
                approved_claim_issues.append(f"SCOPE_FLAG_MUST_BE_FALSE:{flag}")
        if (
            approved_claims.get("claim_count") != len(raw_claims)
            or approved_claims.get("source_binding_count") != len(raw_bindings)
        ):
            approved_claim_issues.append("CLAIM_OR_BINDING_COUNT_MISMATCH")
        bindings = {
            str(item.get("claim_id")): item
            for item in raw_bindings
            if isinstance(item, dict) and item.get("claim_id")
        }
        if len(bindings) != len(raw_bindings):
            approved_claim_issues.append("DUPLICATE_OR_INVALID_SOURCE_BINDING")
        for raw_claim in raw_claims:
            try:
                claim = DocumentEvidenceClaim.model_validate(raw_claim)
            except Exception:
                approved_claim_issues.append("INVALID_DOCUMENT_EVIDENCE_CLAIM")
                continue
            binding = bindings.get(claim.claim_id)
            if binding is None:
                approved_claim_issues.append(f"MISSING_SOURCE_BINDING:{claim.claim_id}")
                continue
            if (
                binding.get("source_id") != claim.source_document_id
                or binding.get("source_file_sha256") != claim.source_document_sha256
                or binding.get("source_fragment_id") != claim.source_fragment_id
                or binding.get("source_locator") != claim.source_locator
                or binding.get("source_span_sha256") != claim.source_span_sha256
            ):
                approved_claim_issues.append(f"SOURCE_BINDING_MISMATCH:{claim.claim_id}")
            if not claim.publication_eligible:
                approved_claim_issues.append(f"CLAIM_NOT_PUBLICATION_ELIGIBLE:{claim.claim_id}")
            if (
                claim.epistemic_status.value != "ONTOLOGY_ASSOCIATION"
                or claim.predicate != "MANIFESTS_AS"
                or claim.source_review_state.value != "HUMAN_REVIEWED"
                or claim.source_authority.value != "APPROVED_SOURCE"
                or claim.source_visibility_class is None
                or claim.source_visibility_class.value != "DESIGN_PUBLIC"
                or claim.temporal_scope.value != "TIME_INDEPENDENT"
            ):
                approved_claim_issues.append(f"CLAIM_SCOPE_INVALID:{claim.claim_id}")
            if claim.runtime_evidence_eligible or claim.source_runtime_vote_allowed:
                approved_claim_issues.append(f"CLAIM_RUNTIME_SCOPE_EXCEEDED:{claim.claim_id}")
            approved_claim_count += int(claim.publication_eligible)
    gates = [
        _gate("P05-01", "taxonomy, subsystem, scenario, phase and document-type coverage", "PASS"
              if coverage.get("status") == "PASS" and not coverage.get("violations") else "BLOCKED",
              {"coverage_status": coverage.get("status"), "violation_count": len(coverage.get("violations", []))}),
        _gate("P05-02", "every manifest episode has a causal log bundle", "PASS"
              if bundles and bundles_without_causal_log == 0 else "BLOCKED",
              {"bundle_count": len(bundles), "bundles_without_causal_log": bundles_without_causal_log}),
        _gate(
            "P05-03",
            "publishable claims have span, time, hashes and epistemic status",
            (
                "PASS"
                if approved_claims is not None
                and approved_claim_status == "APPROVED_TBOX_CLAIM_SET"
                and approved_claims.get("claim_set_publication_eligible") is True
                and approved_claim_count > 0
                and not approved_claim_issues
                else "BLOCKED_APPROVED_CLAIM_SET_INVALID"
                if approved_claims is not None
                else "BLOCKED_NO_PUBLISHABLE_CLAIM_SET"
                if publishable_claim_count == 0
                else "REQUIRES_CLAIM_FIELD_AUDIT"
            ),
            {
                "manifest_publishable_claim_count": publishable_claim_count,
                "approved_claim_release_status": approved_claim_status,
                "approved_publishable_claim_count": approved_claim_count,
                "approved_claim_issue_count": len(approved_claim_issues),
            },
        ),
        _gate("P05-04", "test and confirmation root-cause leakage is zero", "PASS"
              if audit.get("status") == "PASS" and not audit.get("violations") and build.get("posthoc_telemetry_or_truth_opened") is False else "BLOCKED",
              {"corpus_audit_status": audit.get("status"), "posthoc_truth_opened": build.get("posthoc_telemetry_or_truth_opened")}),
        _gate("P05-05", "identifier, path and layout side-channel attack stays near prior", "PASS"
              if sidechannel.get("status") == "PASS" and not sidechannel.get("violations") else "BLOCKED",
              {"attack_status": sidechannel.get("status"), "violation_count": len(sidechannel.get("violations", []))}),
        _gate("P05-06", "human gold exists for PDF, DOCX, table, image and formula parsing", "PASS"
              if multimodal.get("human_gold_complete") is True and early_gold.get("status") == "COMPLETE" and scan_review.get("representation_reviewed") is True else "BLOCKED_HUMAN_GOLD_REQUIRED",
              {"multimodal_human_gold_complete": multimodal.get("human_gold_complete"), "early_design_gold_status": early_gold.get("status"), "scan_representation_reviewed": scan_review.get("representation_reviewed")}),
        _gate(
            "P05-07",
            "section-aware retrieval passes preregistered human gold",
            (
                "PASS"
                if chunking.get("promotion_decision") in {
                    "PROMOTE_SECTION_AWARE_HUMAN_GOLD",
                    "PROMOTE_SECTION_CONTEXT_AWARE_HUMAN_GOLD",
                }
                else "BLOCKED_RETRIEVAL_PROMOTION_THRESHOLDS_NOT_MET"
                if chunking.get("status") == "COMPLETE"
                and chunking.get("human_gold_complete") is True
                else "BLOCKED_HUMAN_RETRIEVAL_GOLD_REQUIRED"
            ),
            {
                "human_gold_complete": chunking.get("human_gold_complete"),
                "machine_status": chunking.get("status"),
                "promotion_decision": chunking.get("promotion_decision"),
            },
        ),
        _gate(
            "P05-08", "canonical-to-file consistency and zero unsupported LLM rewrite claims",
            (
                "PASS"
                if rewrite_review is not None
                and rewrite_review.get("status") == "READY_FOR_CORPUS_RELEASE_CONSIDERATION"
                and rewrite_review.get("unsupported_claim_rate") == 0.0
                else (
                    "BLOCKED_HUMAN_REWRITE_REVIEW_REQUIRED"
                    if rewrite is not None and rewrite.get("document_count", 0) > 0
                    else "BLOCKED_LLM_REWRITE_EVIDENCE_REQUIRED"
                )
            ),
            {
                "manifest_llm_rewrite_document_count": llm_rewrite_document_count,
                "candidate_rewrite_document_count": rewrite.get("document_count", 0) if rewrite else 0,
                "candidate_rewrite_block_count": rewrite.get("block_count", 0) if rewrite else 0,
                "rewrite_review_status": rewrite_review.get("status") if rewrite_review else None,
                "unsupported_claim_rate": rewrite_review.get("unsupported_claim_rate") if rewrite_review else None,
            },
        ),
        _gate("P05-09", "inventory, license, usage boundary, version and hashes are frozen",
              "BLOCKED_METADATA_REQUIRED" if descriptors_missing_license else (
                  "PASS" if manifest.get("release_state") == "FROZEN" else "BLOCKED_RELEASE_FREEZE_REQUIRED"
              ),
              {"release_state": manifest.get("release_state"), "descriptors_missing_required_metadata": descriptors_missing_license}),
        _gate("P05-10", "canonical corpus rebuild is deterministic", "PASS"
              if rebuild.get("status") == "PASS" and rebuild.get("manifest_semantically_identical") is True and rebuild.get("artifact_hash_mismatch_count") == 0 else "BLOCKED",
              {"rebuild_status": rebuild.get("status"), "artifact_hash_mismatch_count": rebuild.get("artifact_hash_mismatch_count")}),
    ]
    counts = dict(sorted(Counter(gate["status"] for gate in gates).items()))
    ready = all(gate["status"] == "PASS" for gate in gates)
    body = {
        "schema_version": "astrograph-p05-corpus-readiness-assessment.v1",
        "assessment_id": spec["assessment_id"],
        "status": "READY_FOR_P1" if ready else "BLOCKED",
        "evidence_spec_sha256": _sha(spec_path),
        "evidence_files": evidence_files,
        "corpus_id": manifest.get("corpus_id"),
        "bundle_count": len(bundles),
        "document_type_counts": dict(sorted(document_types.items())),
        "gates": gates,
        "gate_status_counts": counts,
        "ready_for_p1": ready,
        "full_test_run": False,
        "active_graph_modified": False,
        "claim_boundary": (
            "This is a fail-closed evidence assessment of the ten P0.5 gates. A BLOCKED result "
            "must not be overridden by machine-only artifacts or interpreted as corpus release approval."
        ),
    }
    receipt = {**body, "assessment_sha256": _canonical_sha(body)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


__all__ = ["assess_document_corpus_readiness"]
