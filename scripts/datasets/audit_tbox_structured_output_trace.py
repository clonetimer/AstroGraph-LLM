#!/usr/bin/env python3
"""Audit numeric evidence -> reviewed TBox OAG -> structured-output trace."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from astrograph.diagnosis.structured_output import build_structured_diagnosis_output
from astrograph.evidence import EpisodeEvidenceBundle
from astrograph.retrieval.evidence_context import EvidenceAwarePreGenerationContextBuilder
from astrograph.retrieval.service import PreReasoningRetrievalService


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = (
    ROOT
    / "governance/evidence/research/document_corpus/tbox_structured_output_trace_r001"
)


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def main() -> int:
    evidence_path = (
        ROOT / "tests/fixtures/evidence/heater_stuck_on/diagnosis/episode_evidence_bundle.json"
    )
    runtime_path = ROOT / "configs/datasets/phm_v6_0/runtime_evidence.json"
    catalog_path = ROOT / "configs/fault_modes/lab_fault_catalog.json"
    projection_root = (
        ROOT
        / "governance/evidence/research/document_corpus/reviewed_tbox_retrieval_r001"
    )
    bundle = EpisodeEvidenceBundle.model_validate_json(
        evidence_path.read_text(encoding="utf-8")
    )
    request = EvidenceAwarePreGenerationContextBuilder(
        runtime_evidence_path=runtime_path,
        fault_catalog_path=catalog_path,
        reviewed_tbox_graph_path=projection_root / "reviewed_tbox_retrieval_graph.json",
        reviewed_tbox_projection_receipt_path=projection_root / "build_receipt.json",
    ).build_request(
        bundle,
        reviewed_tbox_seed_concept_ids=["mechanism:thermal-exchange-degradation"],
    )
    if request is None:
        raise ValueError("fixture did not produce an evidence-aware retrieval request")
    retrieval = PreReasoningRetrievalService().retrieve(request)
    matching_paths = [
        path.model_dump(mode="json")
        for path in retrieval.ontology_paths
        if path.seed_concept_id == "mechanism:thermal-exchange-degradation"
    ]
    if not matching_paths:
        raise ValueError("reviewed TBox path did not reach the retrieval package")

    structured = build_structured_diagnosis_output({
        "simulation_event": {"run_id": "TRACE-CONTRACT-PHM-HEATER"},
        "data_gate_report": {"passed": True},
        "episode_evidence_bundle": bundle.model_dump(mode="json"),
        "pre_reasoning_retrieval": retrieval.model_dump(mode="json"),
        "hypothesis_set": {"hypotheses": [], "validation_metadata": {}},
        "decision_record": {
            "episode_id": bundle.episode.episode_id,
            "disposition": "WITHHELD",
            "formal_release_allowed": False,
            "requires_human_review": True,
            "reasons": ["TRACE_CONTRACT_ONLY"],
            "unresolved_gaps": ["NO_DIAGNOSTIC_PERFORMANCE_CLAIM"],
        },
        "diagnosis_policy_decision": {
            "status": "WITHHELD",
            "blocking_reasons": ["TRACE_CONTRACT_ONLY"],
            "warnings": [],
        },
        "verification_plan": {"verification_plan_id": "TRACE-CONTRACT", "tasks": []},
    })
    trace = structured.knowledge_and_retrieval_summary
    if not trace["ontology_path_references"]:
        raise ValueError("structured output dropped ontology path references")
    if trace["diagnostic_effect_claimed"] is not False:
        raise ValueError("trace contract unexpectedly claimed a diagnostic effect")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    structured_path = OUTPUT / "structured_output.json"
    structured_path.write_text(structured.model_dump_json(indent=2) + "\n", encoding="utf-8")
    body = {
        "schema_version": "astrograph-tbox-structured-output-trace-audit.v1",
        "status": "PASS",
        "inputs": {
            "episode_evidence_bundle_sha256": _file_sha(evidence_path),
            "runtime_evidence_sha256": _file_sha(runtime_path),
            "fault_catalog_sha256": _file_sha(catalog_path),
            "tbox_projection_receipt_sha256": _file_sha(
                projection_root / "build_receipt.json"
            ),
            "tbox_projection_graph_sha256": _file_sha(
                projection_root / "reviewed_tbox_retrieval_graph.json"
            ),
        },
        "implementation_sha256": {
            relative: _file_sha(ROOT / relative)
            for relative in (
                "src/astrograph/retrieval/evidence_context.py",
                "src/astrograph/retrieval/v3_tbox_adapter.py",
                "src/astrograph/retrieval/service.py",
                "src/astrograph/diagnosis/structured_output.py",
            )
        },
        "episode_id": bundle.episode.episode_id,
        "retrieval_package_id": retrieval.package_id,
        "retrieval_status": retrieval.status.value,
        "reviewed_tbox_seed_concept_ids": ["mechanism:thermal-exchange-degradation"],
        "matching_tbox_path_count": len(matching_paths),
        "matching_tbox_paths": matching_paths,
        "structured_output_id": structured.output_id,
        "structured_output_sha256": structured.output_sha256,
        "structured_output_file_sha256": _file_sha(structured_path),
        "structured_trace_path_count": len(trace["ontology_path_references"]),
        "retrieval_only": True,
        "diagnostic_effect_claimed": False,
        "formal_release_allowed": False,
        "runtime_vote_allowed": False,
        "active_graph_modified": False,
        "claim_boundary": (
            "Executable trace-contract evidence that numerical episode evidence can seed a governed "
            "pre-generation OAG package and that the final structured output retains source-bound "
            "TBox paths. It is not a diagnosis-accuracy, causal-effect or runtime-voting claim."
        ),
    }
    receipt = {**body, "audit_sha256": _canonical_sha(body)}
    receipt_path = OUTPUT / "audit.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
