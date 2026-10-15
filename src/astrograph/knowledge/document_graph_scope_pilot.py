"""Run a frozen, truth-isolated pilot across episode, historical and evaluator graphs."""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from astrograph.agents.knowledge_curation import KnowledgeExtractionBackend
from astrograph.retrieval.document_corpus import DocumentCorpusManifest, PairedDocumentBundle
from astrograph.retrieval.document_governance import (
    DocumentType,
    DocumentVisibilityClass,
)

from .construction import (
    DocumentKnowledgeStagingBundle,
    process_governed_document,
    stage_quarantined_document_source,
    write_construction_bundle,
)
from .document_graph import (
    build_staged_document_candidate_graph,
    propose_staged_document_knowledge,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _select_documents(
    manifest: DocumentCorpusManifest, config: dict[str, Any]
) -> list[tuple[str, PairedDocumentBundle, Any]]:
    selected: list[tuple[str, PairedDocumentBundle, Any]] = []
    cutoff = float(config["selection"]["causal_cutoff_seconds"])
    for dataset_id, rule in sorted(config["datasets"].items()):
        episode_bundles = sorted(
            (
                bundle
                for bundle in manifest.bundles
                if bundle.dataset_id == dataset_id
                and bundle.split_role.value == rule["episode_split_role"]
                and bundle.provenance_group_id == rule["episode_provenance_group_id"]
            ),
            key=lambda bundle: bundle.opaque_episode_id,
        )
        if not episode_bundles:
            raise ValueError(f"no episode pilot bundle for {dataset_id}")
        episode = episode_bundles[0]
        for document_type in config["selection"]["episode_document_types"]:
            matches = [
                artifact
                for artifact in episode.artifacts
                if artifact.descriptor.document_type.value == document_type
                and artifact.descriptor.visibility_class
                is DocumentVisibilityClass.OBSERVATION_CAUSAL
                and artifact.descriptor.observable_cutoff_time_s is not None
                and abs(artifact.descriptor.observable_cutoff_time_s - cutoff) < 1e-6
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"expected one {document_type}@{cutoff:g}s for {dataset_id}; got {len(matches)}"
                )
            selected.append(("episode", episode, matches[0]))
        truths = [
            artifact
            for artifact in episode.artifacts
            if artifact.descriptor.document_type.value
            == config["selection"]["quarantine_document_type"]
            and artifact.descriptor.visibility_class
            is DocumentVisibilityClass.TRUTH_QUARANTINED
        ]
        if len(truths) != 1:
            raise ValueError(f"expected one quarantined truth document for {dataset_id}")
        selected.append(("quarantine", episode, truths[0]))

        historical_bundles = sorted(
            (
                bundle
                for bundle in manifest.bundles
                if bundle.dataset_id == dataset_id
                and bundle.split_role.value == rule["historical_split_role"]
                and bundle.provenance_group_id == rule["historical_provenance_group_id"]
                and any(
                    artifact.descriptor.document_type.value
                    == config["selection"]["historical_document_type"]
                    for artifact in bundle.artifacts
                )
            ),
            key=lambda bundle: bundle.opaque_episode_id,
        )
        if not historical_bundles:
            raise ValueError(f"no historical pilot bundle for {dataset_id}")
        historical = historical_bundles[0]
        closures = [
            artifact
            for artifact in historical.artifacts
            if artifact.descriptor.document_type.value
            == config["selection"]["historical_document_type"]
            and artifact.descriptor.visibility_class
            is DocumentVisibilityClass.RESOLVED_CASE_TRAIN_ONLY
        ]
        if len(closures) != 1:
            raise ValueError(f"expected one historical closure for {dataset_id}")
        selected.append(("historical", historical, closures[0]))
    return selected


def build_document_graph_scope_pilot(
    *,
    config_path: str | Path,
    manifest_path: str | Path,
    corpus_root: str | Path,
    output_root: str | Path,
    backend: KnowledgeExtractionBackend,
    excluded_review_refreeze_sha256: str,
    max_workers: int = 1,
) -> dict[str, Any]:
    """Execute the preregistered four-lane pilot without opening truth for selection."""

    config_path = Path(config_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    corpus_root = Path(corpus_root).resolve()
    output_root = Path(output_root).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "astrograph-document-graph-scope-pilot.v1":
        raise ValueError("unsupported document graph scope pilot schema")
    manifest = DocumentCorpusManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if manifest.corpus_id != config["source_corpus_id"]:
        raise ValueError("scope pilot config and corpus identity differ")
    selected = _select_documents(manifest, config)
    runtime_ids = set(manifest.index.runtime_document_ids)
    evaluator_ids = set(manifest.index.evaluator_truth_document_ids)
    for lane, _, artifact in selected:
        path = (corpus_root / artifact.relative_path).resolve()
        if not path.is_relative_to(corpus_root) or not path.is_file() or _sha(path) != artifact.sha256:
            raise ValueError(f"pilot source file is missing, outside root or changed: {artifact.relative_path}")
        document_id = artifact.descriptor.document_id
        if lane == "episode" and document_id not in runtime_ids:
            raise ValueError("episode pilot document is absent from Runtime index")
        if lane == "quarantine" and (
            document_id not in evaluator_ids or document_id in runtime_ids
        ):
            raise ValueError("quarantine pilot document is not evaluator-only")
        if lane == "historical" and document_id in evaluator_ids:
            raise ValueError("historical closure must not share evaluator truth index membership")
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("refusing to overwrite nonempty document graph scope pilot directory")
    output_root.mkdir(parents=True, exist_ok=True)

    def run(item: tuple[str, PairedDocumentBundle, Any]) -> dict[str, Any]:
        lane, bundle, artifact = item
        descriptor = artifact.descriptor
        source = (corpus_root / artifact.relative_path).resolve()
        processed = None
        if lane == "quarantine":
            staging = stage_quarantined_document_source(
                descriptor=descriptor,
                source_path=source,
                independence_group=bundle.provenance_group_id,
                license_id="project-generated",
            )
        else:
            processed = process_governed_document(
                descriptor=descriptor,
                source_path=source,
                independence_group=bundle.provenance_group_id,
                license_id="project-generated",
                knowledge_version="document-graph-scope-pilot-r001",
            )
            staging = processed.knowledge_staging
        destination = (
            output_root
            / lane
            / bundle.dataset_id
            / descriptor.document_id.replace(":", "_")
        )
        destination.mkdir(parents=True, exist_ok=True)
        staging_path = destination / "knowledge_staging.json"
        staging_path.write_text(staging.model_dump_json(indent=2) + "\n", encoding="utf-8")
        raw_path = destination / "raw_extraction_proposal.json"
        proposal = None
        try:
            proposal = propose_staged_document_knowledge(
                staging=staging, backend=backend, section_batched=True
            )
            raw_path.write_text(proposal.model_dump_json(indent=2) + "\n", encoding="utf-8")
            construction = build_staged_document_candidate_graph(
                staging=staging,
                proposal=proposal,
                excluded_review_refreeze_sha256=excluded_review_refreeze_sha256,
                graph_id=f"graph:{lane}:{bundle.dataset_id}:{descriptor.document_id}:r001",
            )
            paths = write_construction_bundle(construction, output_dir=destination)
            return {
                "lane": lane,
                "dataset_id": bundle.dataset_id,
                "split_role": bundle.split_role.value,
                "provenance_group_id": bundle.provenance_group_id,
                "opaque_episode_id": bundle.opaque_episode_id,
                "document_id": descriptor.document_id,
                "document_type": descriptor.document_type.value,
                "visibility_class": descriptor.visibility_class.value,
                "source_relative_path": artifact.relative_path,
                "source_sha256": artifact.sha256,
                "graph_scope": staging.graph_scope.value,
                "status": construction.status,
                "candidate_graph_built": construction.candidate_graph is not None,
                "document_processing_content_sha256": (
                    processed.content_sha256 if processed is not None else None
                ),
                "construction_content_sha256": construction.content_sha256,
                "proposal_node_count": len(construction.proposal.nodes),
                "proposal_relation_count": len(construction.proposal.relations),
                "llm_raw_node_count": len(proposal.nodes),
                "llm_raw_relation_count": len(proposal.relations),
                "llm_raw_unresolved_claim_count": len(proposal.unresolved_claims),
                "deterministic_scaffold_node_count": sum(
                    bool(node.properties.get("descriptor_scaffold"))
                    for node in construction.proposal.nodes
                ),
                "deterministic_scaffold_relation_count": sum(
                    bool(relation.properties.get("descriptor_scaffold"))
                    for relation in construction.proposal.relations
                ),
                "deterministic_content_node_count": sum(
                    bool(node.properties.get("deterministic_document_extraction"))
                    for node in construction.proposal.nodes
                ),
                "deterministic_content_relation_count": sum(
                    bool(relation.properties.get("deterministic_document_extraction"))
                    for relation in construction.proposal.relations
                ),
                "unresolved_claim_count": len(construction.proposal.unresolved_claims),
                "blocker_codes": sorted({
                    finding.code for finding in construction.candidate_validation.findings
                    if finding.severity == "BLOCKER"
                }),
                "raw_extraction_proposal_sha256": _sha(raw_path),
                "artifact_sha256": {
                    **{name: _sha(path) for name, path in sorted(paths.items())},
                    "knowledge_staging": _sha(staging_path),
                },
                "publication_allowed": False,
                "runtime_vote_allowed": False,
                "active_graph_modified": False,
            }
        except Exception as exc:
            failure = {
                "lane": lane,
                "dataset_id": bundle.dataset_id,
                "document_id": descriptor.document_id,
                "document_type": descriptor.document_type.value,
                "visibility_class": descriptor.visibility_class.value,
                "source_relative_path": artifact.relative_path,
                "source_sha256": artifact.sha256,
                "graph_scope": staging.graph_scope.value,
                "status": "BLOCKED_BY_EXCEPTION",
                "exception_type": type(exc).__name__,
                "detail": str(exc)[:2000],
                "candidate_graph_built": False,
                "document_processing_content_sha256": (
                    processed.content_sha256 if processed is not None else None
                ),
                "construction_content_sha256": None,
                "raw_extraction_proposal_sha256": _sha(raw_path) if raw_path.is_file() else None,
                "knowledge_staging_sha256": _sha(staging_path),
                "llm_raw_node_count": len(proposal.nodes) if proposal is not None else None,
                "llm_raw_relation_count": len(proposal.relations) if proposal is not None else None,
                "llm_raw_unresolved_claim_count": (
                    len(proposal.unresolved_claims) if proposal is not None else None
                ),
                "publication_allowed": False,
                "runtime_vote_allowed": False,
                "active_graph_modified": False,
            }
            failure_path = destination / "failure.json"
            failure_path.write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return {**failure, "failure_sha256": _sha(failure_path)}

    results: list[dict[str, Any]] = []
    if max_workers == 1:
        results = [run(item) for item in selected]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run, item) for item in selected]
            for future in as_completed(futures):
                results.append(future.result())
    results.sort(key=lambda row: (row["lane"], row["dataset_id"], row["document_id"]))
    body = {
        "schema_version": "astrograph-document-graph-scope-pilot-receipt.v1",
        "pilot_id": config["pilot_id"],
        "configuration_sha256": _sha(config_path),
        "source_manifest_sha256": _sha(manifest_path),
        "pipeline_source_sha256": {
            "knowledge_curation": _sha(
                Path(__file__).parents[1] / "agents" / "knowledge_curation.py"
            ),
            "construction": _sha(Path(__file__).with_name("construction.py")),
            "document_graph": _sha(Path(__file__).with_name("document_graph.py")),
            "document_grounding": _sha(
                Path(__file__).with_name("document_grounding.py")
            ),
            "scope_pilot": _sha(Path(__file__)),
        },
        "selection_used_document_contents": False,
        "selected_document_count": len(results),
        "lane_counts": {
            lane: sum(row["lane"] == lane for row in results)
            for lane in ("episode", "historical", "quarantine")
        },
        "candidate_ready_count": sum(
            row["status"] == "CANDIDATE_READY_FOR_HUMAN_REVIEW" for row in results
        ),
        "blocked_count": sum(
            row["status"] != "CANDIDATE_READY_FOR_HUMAN_REVIEW" for row in results
        ),
        "llm_raw_candidate_counts": {
            "nodes": sum(row.get("llm_raw_node_count") or 0 for row in results),
            "relations": sum(row.get("llm_raw_relation_count") or 0 for row in results),
            "unresolved_claims": sum(
                row.get("llm_raw_unresolved_claim_count") or 0 for row in results
            ),
        },
        "deterministic_scaffold_counts": {
            "nodes": sum(
                row.get("deterministic_scaffold_node_count") or 0 for row in results
            ),
            "relations": sum(
                row.get("deterministic_scaffold_relation_count") or 0 for row in results
            ),
        },
        "deterministic_content_extraction_counts": {
            "nodes": sum(
                row.get("deterministic_content_node_count") or 0 for row in results
            ),
            "relations": sum(
                row.get("deterministic_content_relation_count") or 0 for row in results
            ),
        },
        "portable_content_identity_counts": {
            "document_processing": sum(
                bool(row.get("document_processing_content_sha256")) for row in results
            ),
            "construction": sum(
                bool(row.get("construction_content_sha256")) for row in results
            ),
            "missing": sum(
                not bool(row.get("construction_content_sha256"))
                or (
                    row["lane"] != "quarantine"
                    and not bool(row.get("document_processing_content_sha256"))
                )
                for row in results
            ),
        },
        "runtime_document_ids": sorted(
            row["document_id"] for row in results if row["lane"] == "episode"
        ),
        "evaluator_only_document_ids": sorted(
            row["document_id"] for row in results if row["lane"] == "quarantine"
        ),
        "runtime_evaluator_overlap": [],
        "results": results,
        "publication_allowed": False,
        "runtime_vote_allowed": False,
        "active_graph_modified": False,
        "claim_boundary": config["claim_boundary"],
    }
    receipt = {**body, "receipt_sha256": _canonical_sha(body)}
    (output_root / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


__all__ = ["build_document_graph_scope_pilot"]
