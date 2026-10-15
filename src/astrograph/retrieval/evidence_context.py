"""Evidence-aware context assembly between BiRRF and structured generation.

The assembler deliberately derives its query only from runtime evidence and
BiRRF candidates.  It turns reviewed, hash-pinned AstroGraph configuration into
read-only retrieval material; it does not publish knowledge or create evidence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from astrograph.evidence import (
    ConfiguredRuntimeEvidencePackage,
    EpisodeEvidenceBundle,
    EvidenceFactStatus,
)
from astrograph.fusion import CandidateFusionPackage

from .models import (
    DiagnosticRetrievalQuery,
    GovernedTextChunk,
    OntologyConcept,
    OntologyRelation,
    PreReasoningRetrievalRequest,
)
from .experience_context import build_experience_retrieval_assets
from .v3_tbox_adapter import load_reviewed_v3_tbox_retrieval_assets


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_suffix(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20].upper()


class EvidenceAwarePreGenerationContextBuilder:
    """Build a governed retrieval request from existing diagnosis artifacts.

    ``runtime_evidence_path`` and ``fault_catalog_path`` are expected to be
    verified Dataset Profile assets in online use.  Their content hashes are
    copied into every generated chunk/concept/relation so citations remain
    reproducible outside the live process.
    """

    builder_id = "evidence-aware-pre-generation-context"
    builder_version = "1.0.0"

    def __init__(
        self,
        *,
        runtime_evidence_path: str | Path,
        fault_catalog_path: str | Path,
        experience_ledger_path: str | Path | None = None,
        memory_store_path: str | Path | None = None,
        reviewed_tbox_graph_path: str | Path | None = None,
        reviewed_tbox_projection_receipt_path: str | Path | None = None,
    ) -> None:
        self.runtime_evidence_path = Path(runtime_evidence_path).resolve()
        self.fault_catalog_path = Path(fault_catalog_path).resolve()
        self.experience_ledger_path = experience_ledger_path
        self.memory_store_path = memory_store_path
        if (reviewed_tbox_graph_path is None) != (
            reviewed_tbox_projection_receipt_path is None
        ):
            raise ValueError(
                "reviewed TBox graph and projection receipt must be configured together"
            )
        self.reviewed_tbox_concepts: list[OntologyConcept] = []
        self.reviewed_tbox_relations: list[OntologyRelation] = []
        self.reviewed_tbox_audit: dict[str, object] | None = None
        if reviewed_tbox_graph_path is not None:
            (
                self.reviewed_tbox_concepts,
                self.reviewed_tbox_relations,
                self.reviewed_tbox_audit,
            ) = load_reviewed_v3_tbox_retrieval_assets(
                graph_path=reviewed_tbox_graph_path,
                projection_receipt_path=reviewed_tbox_projection_receipt_path,
            )
        self.runtime_evidence = ConfiguredRuntimeEvidencePackage.model_validate_json(
            self.runtime_evidence_path.read_text(encoding="utf-8")
        )
        catalog = json.loads(self.fault_catalog_path.read_text(encoding="utf-8"))
        rows = catalog.get("fault_modes")
        if not isinstance(rows, list):
            raise ValueError("fault catalog must contain a fault_modes list")
        self.catalog_id = str(catalog.get("catalog_id") or "astrograph-fault-catalog")
        self.catalog_version = str(catalog.get("catalog_version") or "unknown")
        self.fault_rows = {
            str(item["fault_id"]): item
            for item in rows
            if isinstance(item, dict) and item.get("fault_id")
        }
        self.runtime_sha256 = _sha256_file(self.runtime_evidence_path)
        self.catalog_sha256 = _sha256_file(self.fault_catalog_path)

    @staticmethod
    def _fault_concept_id(fault_id: str) -> str:
        return f"fault:{fault_id}"

    @staticmethod
    def _feature_concept_id(feature_id: str) -> str:
        return f"feature:{feature_id}"

    def _knowledge_assets(
        self,
    ) -> tuple[list[GovernedTextChunk], list[OntologyConcept], list[OntologyRelation]]:
        definitions_by_fault: dict[str, list[Any]] = {}
        for definition in self.runtime_evidence.evidence_definitions:
            definitions_by_fault.setdefault(definition.target_fault_id, []).append(definition)
        feature_by_id = {
            item.feature_definition_id: item
            for item in self.runtime_evidence.feature_definitions
        }
        fault_ids = sorted(definitions_by_fault)
        missing_catalog = sorted(set(fault_ids) - set(self.fault_rows))
        if missing_catalog:
            raise ValueError(
                "runtime evidence faults are absent from the reviewed fault catalog: "
                f"{missing_catalog}"
            )

        concepts: list[OntologyConcept] = []
        relations: list[OntologyRelation] = []
        chunks: list[GovernedTextChunk] = []
        for fault_id in fault_ids:
            row = self.fault_rows[fault_id]
            fault_concept_id = self._fault_concept_id(fault_id)
            concepts.append(
                OntologyConcept(
                    concept_id=fault_concept_id,
                    labels=[fault_id, str(row.get("display_name") or fault_id)],
                    aliases=[fault_id.replace("_", " ").lower()],
                    source_id=f"{self.catalog_id}:{fault_id}",
                    source_sha256=self.catalog_sha256,
                    knowledge_version=self.catalog_version,
                )
            )
            telemetry = [
                f"{item.get('channel')} ({item.get('role')}: {item.get('expected_effect')})"
                for item in row.get("telemetry_channels", [])
                if isinstance(item, dict) and item.get("channel")
            ]
            catalog_text = "\n".join(
                [
                    f"Fault ID: {fault_id}",
                    f"Name: {row.get('display_name', fault_id)}",
                    f"Subsystem: {row.get('subsystem', 'UNKNOWN')}",
                    f"Description: {row.get('description', '')}",
                    "Reviewed telemetry expectations: " + "; ".join(telemetry),
                    "Boundary: catalog semantics guide retrieval and explanation only; "
                    "they are not episode evidence or causal proof.",
                ]
            )
            chunks.append(
                GovernedTextChunk(
                    chunk_id=f"catalog:{fault_id}",
                    source_id=f"{self.catalog_id}:{fault_id}",
                    source_uri=f"repo://{self.fault_catalog_path.name}#{fault_id}",
                    source_sha256=self.catalog_sha256,
                    chunk_sha256=_sha256_text(catalog_text),
                    knowledge_version=self.catalog_version,
                    text=catalog_text,
                    concept_ids=[fault_concept_id],
                )
            )

            evidence_lines: list[str] = []
            evidence_concepts = [fault_concept_id]
            for definition in sorted(
                definitions_by_fault[fault_id], key=lambda item: item.evidence_definition_id
            ):
                feature = feature_by_id[definition.feature_definition_id]
                feature_concept_id = self._feature_concept_id(feature.feature_definition_id)
                if feature_concept_id not in {item.concept_id for item in concepts}:
                    concepts.append(
                        OntologyConcept(
                            concept_id=feature_concept_id,
                            labels=[feature.name, feature.source_field_template],
                            aliases=[feature.feature_definition_id],
                            source_id=self.runtime_evidence.package_id,
                            source_sha256=self.runtime_sha256,
                            knowledge_version=self.runtime_evidence.version,
                        )
                    )
                if feature_concept_id not in evidence_concepts:
                    evidence_concepts.append(feature_concept_id)
                relations.append(
                    OntologyRelation(
                        relation_id=f"relation:{definition.evidence_definition_id}",
                        source_concept_id=fault_concept_id,
                        relation_type="INDICATED_BY",
                        target_concept_id=feature_concept_id,
                        source_id=self.runtime_evidence.package_id,
                        source_sha256=self.runtime_sha256,
                        knowledge_version=self.runtime_evidence.version,
                    )
                )
                evidence_lines.append(
                    f"{definition.evidence_definition_id}: {definition.name}; "
                    f"feature={feature.name}; comparison={definition.comparison.value} "
                    f"{definition.threshold:g} {definition.threshold_unit}; "
                    f"level={definition.evidence_level}; boundary={definition.claim_boundary}"
                )
            evidence_text = "\n".join(
                [
                    f"Executable evidence package: {self.runtime_evidence.package_id} "
                    f"version {self.runtime_evidence.version}",
                    f"Target fault: {fault_id}",
                    *evidence_lines,
                    f"Package boundary: {self.runtime_evidence.claim_boundary}",
                ]
            )
            chunks.append(
                GovernedTextChunk(
                    chunk_id=f"runtime-evidence:{fault_id}",
                    source_id=self.runtime_evidence.package_id,
                    source_uri=f"repo://{self.runtime_evidence_path.name}#{fault_id}",
                    source_sha256=self.runtime_sha256,
                    chunk_sha256=_sha256_text(evidence_text),
                    knowledge_version=self.runtime_evidence.version,
                    text=evidence_text,
                    concept_ids=evidence_concepts,
                )
            )
        concept_ids = {item.concept_id for item in concepts}
        relation_ids = {item.relation_id for item in relations}
        if concept_ids & {item.concept_id for item in self.reviewed_tbox_concepts}:
            raise ValueError("reviewed TBox concept IDs collide with runtime evidence assets")
        if relation_ids & {item.relation_id for item in self.reviewed_tbox_relations}:
            raise ValueError("reviewed TBox relation IDs collide with runtime evidence assets")
        concepts.extend(self.reviewed_tbox_concepts)
        relations.extend(self.reviewed_tbox_relations)
        return chunks, concepts, relations

    def build_request(
        self,
        bundle: EpisodeEvidenceBundle,
        *,
        candidate_fusion: CandidateFusionPackage | None = None,
        caller_access_tags: list[str] | None = None,
        top_k: int = 8,
        reviewed_tbox_seed_concept_ids: list[str] | None = None,
    ) -> PreReasoningRetrievalRequest | None:
        definition_by_id = {
            item.evidence_definition_id: item for item in bundle.evidence_definitions
        }
        satisfied = [
            item
            for item in bundle.evidence_facts
            if item.status is EvidenceFactStatus.SATISFIED
            and definition_by_id.get(item.evidence_definition_id) is not None
            and definition_by_id[item.evidence_definition_id].critical
        ]
        fault_ids = {item.target_fault_id for item in satisfied}
        if candidate_fusion is not None:
            if candidate_fusion.episode_id != bundle.episode.episode_id:
                raise ValueError("candidate fusion and evidence bundle episode IDs differ")
            if candidate_fusion.birrf_result.top_fault_id:
                fault_ids.add(candidate_fusion.birrf_result.top_fault_id)
            else:
                fault_ids.update(
                    item.fault_id
                    for item in candidate_fusion.birrf_result.candidates
                    if item.final_rank == 1
                )
        available_faults = {
            item.target_fault_id for item in self.runtime_evidence.evidence_definitions
        }
        fault_ids &= available_faults
        if not fault_ids:
            return None

        fault_ids_sorted = sorted(fault_ids)
        tbox_seeds = sorted(set(reviewed_tbox_seed_concept_ids or []))
        if len(tbox_seeds) != len(reviewed_tbox_seed_concept_ids or []):
            raise ValueError("reviewed TBox seed concept IDs must be unique")
        available_tbox = {item.concept_id for item in self.reviewed_tbox_concepts}
        unknown_tbox = sorted(set(tbox_seeds) - available_tbox)
        if unknown_tbox:
            raise ValueError(
                f"reviewed TBox seed concept IDs are absent from the governed projection: {unknown_tbox}"
            )
        evidence_fact_ids = sorted(item.evidence_fact_id for item in satisfied)
        evidence_summary = [
            f"{item.target_fault_id} via {item.evidence_definition_id}="
            f"{item.observed_value} {item.unit}"
            for item in sorted(satisfied, key=lambda value: value.evidence_fact_id)
        ]
        query_text = "; ".join(
            [
                f"AstroGraph episode {bundle.episode.episode_id}",
                "diagnostic candidates " + ", ".join(fault_ids_sorted),
                "satisfied executable evidence "
                + ("; ".join(evidence_summary) if evidence_summary else "none"),
                "retrieve reviewed mechanisms, telemetry expectations, thresholds, and claim boundaries",
            ]
        )
        chunks, concepts, relations = self._knowledge_assets()
        experience_chunks, experience_receipt, knowledge_candidates = (
            build_experience_retrieval_assets(
                eligible_fault_ids=fault_ids_sorted,
                experience_ledger_path=self.experience_ledger_path,
                memory_store_path=self.memory_store_path,
            )
        )
        chunks.extend(experience_chunks)
        concept_ids = [self._fault_concept_id(item) for item in fault_ids_sorted]
        query = DiagnosticRetrievalQuery(
            query_id="QUERY-EVIDENCE-" + _stable_suffix(
                {
                    "episode_id": bundle.episode.episode_id,
                    "fault_ids": fault_ids_sorted,
                    "evidence_fact_ids": evidence_fact_ids,
                    "runtime_sha256": self.runtime_sha256,
                    "catalog_sha256": self.catalog_sha256,
                    "reviewed_tbox_seed_concept_ids": tbox_seeds,
                    "reviewed_tbox_projection_sha256": (
                        self.reviewed_tbox_audit.get("projection_sha256")
                        if self.reviewed_tbox_audit is not None else None
                    ),
                }
            ),
            episode_id=bundle.episode.episode_id,
            text=query_text,
            required_concept_ids=concept_ids,
            seed_concept_ids=[*concept_ids, *tbox_seeds],
            caller_access_tags=caller_access_tags or ["PUBLIC"],
            top_k=top_k,
            construction_mode="EVIDENCE_AWARE",
            grounding_fault_ids=fault_ids_sorted,
            grounding_evidence_fact_ids=evidence_fact_ids,
        )
        return PreReasoningRetrievalRequest(
            query=query,
            chunks=chunks,
            concepts=concepts,
            relations=relations,
            ontology_asset_audits=(
                [self.reviewed_tbox_audit]
                if self.reviewed_tbox_audit is not None else []
            ),
            experience_reuse_receipt=experience_receipt.model_dump(mode="json"),
            knowledge_update_candidates=[
                item.model_dump(mode="json") for item in knowledge_candidates
            ],
        )


__all__ = ["EvidenceAwarePreGenerationContextBuilder"]
