"""Scope-aware conversion of a governed document into a review-only candidate graph."""
from __future__ import annotations

import re
from typing import Any

from astrograph.agents.knowledge_curation import (
    KnowledgeCurationAgent,
    KnowledgeExtractionBackend,
    KnowledgeExtractionFragmentProposal,
    KnowledgeExtractionProposal,
    KnowledgeNodeCandidate,
    KnowledgeRelationCandidate,
)

from .construction import (
    DocumentKnowledgeGraphScope,
    DocumentKnowledgeStagingBundle,
    GovernedDocumentProcessingBundle,
    KnowledgeConstructionBundle,
    build_knowledge_construction_bundle,
    govern_document_knowledge_proposal,
    knowledge_source_profile_set_sha256,
)
from .schema_v3 import (
    GraphLayer,
    KnowledgeNodeTypeV3,
    KnowledgeRelationTypeV3,
    NODE_LAYERS,
    RELATION_ALLOWLIST,
)
from .document_grounding import require_document_proposal_grounding


def document_graph_extraction_policy(
    scope: DocumentKnowledgeGraphScope,
) -> dict[str, Any]:
    """Return explicit generation constraints for one document graph scope."""

    if scope is DocumentKnowledgeGraphScope.TBOX_CANDIDATE:
        allowed_nodes = [
            node_type
            for node_type, layer in NODE_LAYERS.items()
            if layer is GraphLayer.TBOX
        ]
        allowed_set = set(allowed_nodes)
        shapes = sorted(
            f"{source.value}->{relation.value}->{target.value}"
            for source, relation, target in RELATION_ALLOWLIST
            if source in allowed_set and target in allowed_set
        )
        rules = [
            "This source is a TBox design candidate: do not emit episode, evidence, diagnostic-hypothesis, decision or review nodes.",
            "Treat dataset fault identifiers as aliases or source labels for an explicitly described FailureMechanism; do not model them as per-episode DiagnosticHypothesis nodes.",
            "A confusion-set alternative is a comparison candidate, not a CONTRADICTED_BY edge and not proof of any relation.",
            "Do not emit PROPAGATES_TO, MANIFESTS_AS or SUSCEPTIBLE_TO from this single provenance family; retain the statement as unresolved until independent sources exist.",
            "Confidence measures extraction certainty, not factual truth; use ambiguous only when the source-to-schema mapping itself is ambiguous.",
            "Return at most 6 nodes, 6 relations and 6 unresolved claims; prefer an unresolved claim over a speculative candidate.",
        ]
    elif scope is DocumentKnowledgeGraphScope.EPISODE_EVIDENCE_CANDIDATE:
        allowed_nodes = [
            node_type for node_type, layer in NODE_LAYERS.items()
            if layer in {GraphLayer.ABOX, GraphLayer.EVIDENCE}
        ]
        rules = [
            "This source is a causally visible observation document: extract only episode-bound assets and observed/derived evidence available by its cutoff.",
            "Do not emit DiagnosticHypothesis, DecisionRecord, ReviewRecord, root-cause, fault-label or closure-result nodes.",
            "Absence of a reported anomaly is not a health assertion and must not be converted into counterevidence.",
            "Confidence measures extraction certainty, not diagnostic probability or factual authority.",
            "Return at most 6 nodes, 6 relations and 6 unresolved claims; summarize repeated channels instead of emitting one node per row.",
        ]
    elif scope is DocumentKnowledgeGraphScope.HISTORICAL_CASE_CANDIDATE:
        allowed_nodes = [
            node_type for node_type, layer in NODE_LAYERS.items()
            if layer in {GraphLayer.TBOX, GraphLayer.ABOX, GraphLayer.EVIDENCE, GraphLayer.DECISION}
        ]
        rules = [
            "This is a historical resolution candidate, never current-episode truth or runtime evidence.",
            "Preserve the historical episode and provenance group on every extracted claim.",
            "Do not turn a historical association into a universal causal rule; ambiguous mappings remain unresolved.",
            "Return at most 6 nodes, 6 relations and 6 unresolved claims.",
        ]
    elif scope is DocumentKnowledgeGraphScope.QUARANTINE_EVALUATOR_CANDIDATE:
        allowed_nodes = [
            node_type for node_type, layer in NODE_LAYERS.items()
            if layer in {GraphLayer.ABOX, GraphLayer.EVIDENCE, GraphLayer.DECISION}
        ]
        rules = [
            "This source contains evaluator-only truth: never emit a Runtime-visible fact, retrieval instruction or publication request.",
            "Represent labels only as evaluator episode/hypothesis/decision candidates with exact source fragments.",
            "Do not create global TBox mechanisms or causal relations from evaluator labels.",
            "Every evaluator EvidenceFact or DiagnosticHypothesis must be accompanied by the descriptor episode as an ObservationEpisode and linked with an allowlisted episode relation; otherwise leave it unresolved.",
            "Return at most 6 nodes, 6 relations and 6 unresolved claims.",
        ]
    else:  # pragma: no cover - enum exhaustiveness guard
        raise ValueError(f"unsupported document graph scope: {scope}")
    if scope is not DocumentKnowledgeGraphScope.TBOX_CANDIDATE:
        shapes = sorted(
            f"{source.value}->{relation.value}->{target.value}"
            for source, relation, target in RELATION_ALLOWLIST
            if source in set(allowed_nodes) and target in set(allowed_nodes)
        )
    return {
        "allowed_node_types": allowed_nodes,
        "allowed_relation_shapes": shapes,
        "additional_rules": rules,
    }


def propose_document_knowledge(
    *,
    processed: GovernedDocumentProcessingBundle,
    backend: KnowledgeExtractionBackend,
    approved_existing_nodes: dict[str, KnowledgeNodeTypeV3 | str] | None = None,
) -> KnowledgeExtractionProposal:
    """Run scope-aware extraction while leaving governance/compilation untouched."""

    return propose_staged_document_knowledge(
        staging=processed.knowledge_staging,
        backend=backend,
        approved_existing_nodes=approved_existing_nodes,
    )


def propose_staged_document_knowledge(
    *,
    staging: DocumentKnowledgeStagingBundle,
    backend: KnowledgeExtractionBackend,
    approved_existing_nodes: dict[str, KnowledgeNodeTypeV3 | str] | None = None,
    section_batched: bool = False,
) -> KnowledgeExtractionProposal:
    """Run scope-aware extraction for dual-use or graph-only staging."""

    policy = document_graph_extraction_policy(staging.graph_scope)
    existing = {
        **_descriptor_scaffold_existing_nodes(staging),
        **(approved_existing_nodes or {}),
    }
    governance_context = {
            "document_id": staging.document_descriptor.document_id,
            "document_type": staging.document_descriptor.document_type.value,
            "visibility_class": staging.document_descriptor.visibility_class.value,
            "graph_scope": staging.graph_scope.value,
            "epistemic_status": staging.default_epistemic_status.value,
            "episode_id": staging.document_descriptor.episode_id,
            "provenance_group_id": staging.document_descriptor.provenance_group_id,
            "valid_time_start_s": staging.document_descriptor.valid_time_start_s,
            "valid_time_end_s": staging.document_descriptor.valid_time_end_s,
            "observable_cutoff_time_s": staging.document_descriptor.observable_cutoff_time_s,
            "publication_allowed": False,
            "runtime_vote_allowed": False,
    }

    def fragment_objective(fragment_ids: list[str] | None) -> str:
        selected_text = "\n".join(
            fragment.text
            for fragment in staging.source_profile.fragments
            if fragment_ids is None or fragment.fragment_id in set(fragment_ids)
        )
        document_type = staging.document_descriptor.document_type.value
        episode_id = staging.document_descriptor.episode_id
        evidence_source_id = _evidence_source_id(staging)
        if document_type == "ANOMALY_REPORT" and "Heuristic shift" in selected_text:
            return (
                "Extract at most three table rows as Residual candidates. Preserve channel and numeric "
                "summary values, set calibration_status=UNCALIBRATED_RANKING_HEURISTIC and "
                "diagnostic_evidence_allowed=false. Link each Residual to "
                f"{episode_id} with IN_EPISODE and to {evidence_source_id} with DERIVED_FROM."
            )
        if document_type == "TEST_LOG" and "| Channel | Present |" in selected_text:
            return (
                "Extract at most three representative channel summaries as StateEstimate candidates; "
                "do not infer faults or health. The last source column is one Mean / distinct value: "
                "for numeric rows map it only to mean and never invent a separate distinct count. Link each to "
                f"{episode_id} with IN_EPISODE and to {evidence_source_id} with DERIVED_FROM."
            )
        if document_type in {"CLOSURE_REPORT", "CAMPAIGN_REPORT"} and (
            "diagnostic_class" in selected_text
        ):
            return (
                "Extract the explicit diagnostic_class as one DiagnosticHypothesis candidate and link "
                f"{episode_id} to it with HAS_HYPOTHESIS. Preserve historical/evaluator scope; do not "
                "create a global failure mechanism or causal edge."
            )
        return (
            "Extract only schema-valid content explicitly stated in this fragment. An empty proposal is "
            "correct when the fragment contains only boundaries, headings or non-claim metadata."
        )

    def run(fragment_ids: list[str] | None) -> KnowledgeExtractionProposal:
        objective = fragment_objective(fragment_ids)
        return KnowledgeCurationAgent(backend=backend).propose(
            source_profiles=[staging.source_profile],
            approved_existing_nodes=existing,
            allowed_node_types=policy["allowed_node_types"],
            allowed_relation_shapes=policy["allowed_relation_shapes"],
            additional_rules=[
                *policy["additional_rules"],
                "A '[... representation withheld pending human review ...]' marker is governance metadata, "
                "not document content: never emit a node, relation, attribute or claim from that marker.",
                "CURRENT FRAGMENT EXTRACTION TASK (mandatory within schema and provenance limits): "
                + objective,
            ],
            governance_context={
                **governance_context,
                "prompt_fragment_ids": fragment_ids or [],
                "fragment_task_objective": objective,
            },
            prompt_fragment_ids=fragment_ids,
            response_model=(
                KnowledgeExtractionFragmentProposal
                if fragment_ids is not None
                else KnowledgeExtractionProposal
            ),
        )

    if not section_batched:
        return run(None)
    fragment_ids = list(
        dict.fromkeys(task.fragment_id for task in staging.extraction_tasks)
    )
    proposals = [run([fragment_id]) for fragment_id in fragment_ids]

    def unique(items: list[Any]) -> list[Any]:
        seen: set[str] = set()
        output: list[Any] = []
        for item in items:
            key = item.model_dump_json(exclude_none=False)
            if key not in seen:
                seen.add(key)
                output.append(item)
        return output

    return KnowledgeExtractionProposal(
        source_profile_sha256=knowledge_source_profile_set_sha256(
            [staging.source_profile]
        ),
        nodes=unique([node for proposal in proposals for node in proposal.nodes]),
        relations=unique(
            [relation for proposal in proposals for relation in proposal.relations]
        ),
        unresolved_claims=list(
            dict.fromkeys(
                claim for proposal in proposals for claim in proposal.unresolved_claims
            )
        ),
        proposer_mode="LLM",
        proposer_id=backend.model_id,
        proposer_version=backend.model_version,
        executable_content_present=any(
            proposal.executable_content_present for proposal in proposals
        ),
    )


def _descriptor_scaffold_existing_nodes(
    staging: DocumentKnowledgeStagingBundle,
) -> dict[str, KnowledgeNodeTypeV3]:
    episode_id = staging.document_descriptor.episode_id
    if (
        staging.graph_scope is DocumentKnowledgeGraphScope.TBOX_CANDIDATE
        or episode_id is None
    ):
        return {}
    return {
        episode_id: KnowledgeNodeTypeV3.OBSERVATION_EPISODE,
        _evidence_source_id(staging): KnowledgeNodeTypeV3.EVIDENCE_SOURCE,
    }


def _evidence_source_id(staging: DocumentKnowledgeStagingBundle) -> str:
    return f"evidence-source:{staging.document_descriptor.document_id}"


def _explicit_field(
    staging: DocumentKnowledgeStagingBundle, field: str
) -> tuple[str, str] | None:
    markdown = re.compile(rf"(?m)^-\s*{re.escape(field)}:\s*(.+?)\s*$")
    quoted_json = re.compile(rf'"{re.escape(field)}"\s*:\s*"([^"]+)"')
    for fragment in staging.source_profile.fragments:
        match = markdown.search(fragment.text) or quoted_json.search(fragment.text)
        if match:
            return match.group(1).strip(), fragment.fragment_id
    return None


def _clean_explicit_value(value: str) -> str:
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] == "`":
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _explicit_fields(
    staging: DocumentKnowledgeStagingBundle, field: str
) -> list[tuple[str, str]]:
    markdown = re.compile(rf"(?m)^-\s*{re.escape(field)}:\s*(.+?)\s*$")
    output: list[tuple[str, str]] = []
    for fragment in staging.source_profile.fragments:
        for match in markdown.finditer(fragment.text):
            value = _clean_explicit_value(match.group(1))
            if value:
                output.append((value, fragment.fragment_id))
    return output


def _explicit_tbox_scaffold(
    staging: DocumentKnowledgeStagingBundle,
) -> tuple[list[KnowledgeNodeCandidate], list[str]]:
    """Recover generator-declared TBox fields without upgrading association to causality."""

    if staging.graph_scope is not DocumentKnowledgeGraphScope.TBOX_CANDIDATE:
        return [], []
    component = _explicit_field(staging, "Component candidate")
    mechanism = _explicit_field(staging, "Mechanism association")
    if component is None or mechanism is None:
        return [], []
    component_value, component_fragment = (
        _clean_explicit_value(component[0]), component[1]
    )
    mechanism_value, mechanism_fragment = (
        _clean_explicit_value(mechanism[0]), mechanism[1]
    )
    if component_value == "UNRESOLVED" or mechanism_value == "UNRESOLVED":
        return [], []
    if not mechanism_value.startswith("mechanism:") or any(
        char.isspace() for char in mechanism_value
    ):
        return [], []
    target_fault = _explicit_field(staging, "Target fault")
    evidence_ids = [
        value
        for value, _ in _explicit_fields(staging, "Registered evidence definition")
        if value != "NONE"
    ]
    common = {
        "deterministic_document_extraction": True,
        "association_status": "TBOX_ASSOCIATION_ONLY_UNCONFIRMED",
        "diagnostic_evidence_allowed": False,
        "registered_evidence_definition_ids": list(dict.fromkeys(evidence_ids)),
    }
    component_id = "candidate:document-component:" + canonical_id_digest(component_value)
    mechanism_id = "candidate:document-mechanism:" + canonical_id_digest(mechanism_value)
    nodes = [
        KnowledgeNodeCandidate(
            node_id=component_id,
            node_type=KnowledgeNodeTypeV3.COMPONENT_ARCHETYPE,
            canonical_name=component_value,
            description=(
                "Component candidate copied from an explicit generated-document field; "
                "canonical archetype alignment remains human-reviewed."
            ),
            properties={
                **common,
                "declared_mechanism_association_id": mechanism_value,
                "source_fault_label": (
                    _clean_explicit_value(target_fault[0]) if target_fault else None
                ),
            },
            source_fragment_ids=[component_fragment],
            confidence=1.0,
        ),
        KnowledgeNodeCandidate(
            node_id=mechanism_id,
            node_type=KnowledgeNodeTypeV3.FAILURE_MECHANISM,
            canonical_name=mechanism_value,
            canonical_id_hint=mechanism_value,
            description=(
                "Failure-mechanism vocabulary candidate copied from an explicit generated-document "
                "field; its association with the component is not asserted as a causal edge."
            ),
            properties={
                **common,
                "declared_component_candidate": component_value,
                "source_fault_label": (
                    _clean_explicit_value(target_fault[0]) if target_fault else None
                ),
            },
            source_fragment_ids=[mechanism_fragment],
            confidence=1.0,
        ),
    ]
    unresolved = [
        "UNCONFIRMED_TBOX_ASSOCIATION|component="
        + component_value
        + "|mechanism="
        + mechanism_value
        + "|source_fragment_ids="
        + ";".join(sorted({component_fragment, mechanism_fragment}))
    ]
    return nodes, unresolved


def _anomaly_residual_candidates(
    staging: DocumentKnowledgeStagingBundle,
    *,
    episode_id: str,
    evidence_source_id: str,
) -> tuple[list[KnowledgeNodeCandidate], list[KnowledgeRelationCandidate]]:
    if staging.document_descriptor.document_type.value != "ANOMALY_REPORT":
        return [], []
    nodes: list[KnowledgeNodeCandidate] = []
    relations: list[KnowledgeRelationCandidate] = []
    for fragment in staging.source_profile.fragments:
        if "Heuristic shift" not in fragment.text or "| Rank | Channel |" not in fragment.text:
            continue
        for line in fragment.text.splitlines():
            if not line.lstrip().startswith("|"):
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) < 6 or not cells[0].isdigit():
                continue
            rank, channel, baseline, later, scale, shift = cells[:6]
            residual_id = f"residual:{canonical_id_digest(episode_id + ':' + channel + ':' + rank)}"
            candidate_id = f"candidate:{residual_id}"
            try:
                numeric = {
                    "baseline_mean": float(baseline),
                    "later_mean": float(later),
                    "baseline_scale": float(scale),
                    "heuristic_shift": float(shift),
                }
            except ValueError:
                continue
            nodes.append(
                KnowledgeNodeCandidate(
                    node_id=candidate_id,
                    node_type=KnowledgeNodeTypeV3.RESIDUAL,
                    canonical_name=f"Uncalibrated normalized mean shift: {channel}",
                    description=(
                        "Causal-window channel summary copied from the anomaly screening "
                        "table; it is a ranking residual, not a diagnosis or threshold."
                    ),
                    canonical_id_hint=residual_id,
                    properties={
                        "channel": channel,
                        "rank": int(rank),
                        **numeric,
                        "calibration_status": "UNCALIBRATED_RANKING_HEURISTIC",
                        "diagnostic_evidence_allowed": False,
                        "deterministic_document_extraction": True,
                    },
                    source_fragment_ids=[fragment.fragment_id],
                    confidence=1.0,
                )
            )
            for relation_type, target, suffix in (
                (KnowledgeRelationTypeV3.IN_EPISODE, episode_id, "episode"),
                (KnowledgeRelationTypeV3.DERIVED_FROM, evidence_source_id, "source"),
            ):
                relations.append(
                    KnowledgeRelationCandidate(
                        edge_id=f"candidate-edge:{canonical_id_digest(residual_id + ':' + suffix)}",
                        source_id=candidate_id,
                        relation_type=relation_type,
                        target_id=target,
                        source_fragment_ids=[fragment.fragment_id],
                        confidence=1.0,
                        rationale="Deterministic linkage from the governed table row and document descriptor.",
                        properties={"deterministic_document_extraction": True},
                    )
                )
            if len(nodes) >= 12:
                return nodes, relations
    return nodes, relations


def _merge_descriptor_scaffold(
    staging: DocumentKnowledgeStagingBundle,
    proposal: KnowledgeExtractionProposal,
) -> KnowledgeExtractionProposal:
    """Add identity nodes derived from governed metadata, never from LLM inference."""

    episode_id = staging.document_descriptor.episode_id
    if staging.graph_scope is DocumentKnowledgeGraphScope.TBOX_CANDIDATE:
        nodes, unresolved = _explicit_tbox_scaffold(staging)
        existing_hints = {node.canonical_id_hint for node in proposal.nodes}
        existing_ids = {node.node_id for node in proposal.nodes}
        added = [
            node for node in nodes
            if node.node_id not in existing_ids
            and (node.canonical_id_hint is None or node.canonical_id_hint not in existing_hints)
        ]
        return proposal.model_copy(update={
            "nodes": [*added, *proposal.nodes],
            "unresolved_claims": list(dict.fromkeys([
                *unresolved, *proposal.unresolved_claims,
            ])),
        })
    if episode_id is None:
        return proposal
    fragment = staging.source_profile.fragments[0]
    evidence_source_id = _evidence_source_id(staging)
    scaffolds = [
        KnowledgeNodeCandidate(
            node_id="candidate:descriptor-episode:" + canonical_id_digest(episode_id),
            node_type=KnowledgeNodeTypeV3.OBSERVATION_EPISODE,
            canonical_name=episode_id,
            description=(
                "Observation episode identity supplied by the governed document descriptor; "
                "content and diagnostic claims remain separately reviewable."
            ),
            canonical_id_hint=episode_id,
            properties={
                "descriptor_scaffold": True,
                "document_visibility_class": staging.document_descriptor.visibility_class.value,
                "provenance_group_id": staging.document_descriptor.provenance_group_id,
                "observable_cutoff_time_s": staging.document_descriptor.observable_cutoff_time_s,
            },
            source_fragment_ids=[fragment.fragment_id],
            confidence=1.0,
            ambiguous=False,
        ),
        KnowledgeNodeCandidate(
            node_id="candidate:descriptor-source:" + canonical_id_digest(evidence_source_id),
            node_type=KnowledgeNodeTypeV3.EVIDENCE_SOURCE,
            canonical_name=staging.document_descriptor.document_id,
            description="Governed document evidence source; authority remains encoded in protected metadata.",
            canonical_id_hint=evidence_source_id,
            properties={
                "descriptor_scaffold": True,
                "document_type": staging.document_descriptor.document_type.value,
                "source_uri": staging.document_descriptor.source_uri,
            },
            source_fragment_ids=[fragment.fragment_id],
            confidence=1.0,
        ),
    ]
    relations: list[KnowledgeRelationCandidate] = []
    explicit_diagnosis = _explicit_field(staging, "diagnostic_class")
    if explicit_diagnosis is not None and staging.graph_scope in {
        DocumentKnowledgeGraphScope.HISTORICAL_CASE_CANDIDATE,
        DocumentKnowledgeGraphScope.QUARANTINE_EVALUATOR_CANDIDATE,
    }:
        diagnostic_class, diagnostic_fragment = explicit_diagnosis
        hypothesis_id = (
            f"hypothesis:{canonical_id_digest(episode_id + ':' + diagnostic_class)}"
        )
        hypothesis_candidate_id = f"candidate:{hypothesis_id}"
        scaffolds.append(
            KnowledgeNodeCandidate(
                node_id=hypothesis_candidate_id,
                node_type=KnowledgeNodeTypeV3.DIAGNOSTIC_HYPOTHESIS,
                canonical_name=diagnostic_class,
                description=(
                    "Explicit diagnostic_class copied from a historical resolution."
                    if staging.graph_scope
                    is DocumentKnowledgeGraphScope.HISTORICAL_CASE_CANDIDATE
                    else "Evaluator-only diagnostic_class copied from quarantined truth."
                ),
                canonical_id_hint=hypothesis_id,
                properties={
                    "deterministic_document_extraction": True,
                    "diagnostic_class": diagnostic_class,
                    "evaluator_only": staging.graph_scope
                    is DocumentKnowledgeGraphScope.QUARANTINE_EVALUATOR_CANDIDATE,
                    "historical_only": staging.graph_scope
                    is DocumentKnowledgeGraphScope.HISTORICAL_CASE_CANDIDATE,
                },
                source_fragment_ids=[diagnostic_fragment],
                confidence=1.0,
            )
        )
        relations.append(
            KnowledgeRelationCandidate(
                edge_id=f"candidate-edge:{canonical_id_digest(episode_id + ':hypothesis:' + diagnostic_class)}",
                source_id=episode_id,
                relation_type=KnowledgeRelationTypeV3.HAS_HYPOTHESIS,
                target_id=hypothesis_candidate_id,
                source_fragment_ids=[diagnostic_fragment],
                confidence=1.0,
                rationale="The governed document explicitly binds diagnostic_class to its episode descriptor.",
                properties={"deterministic_document_extraction": True},
            )
        )
    residual_nodes, residual_relations = _anomaly_residual_candidates(
        staging,
        episode_id=episode_id,
        evidence_source_id=evidence_source_id,
    )
    scaffolds.extend(residual_nodes)
    relations.extend(residual_relations)
    existing_hints = {node.canonical_id_hint for node in proposal.nodes}
    added_nodes = [node for node in scaffolds if node.canonical_id_hint not in existing_hints]
    existing_edges = {edge.edge_id for edge in proposal.relations}
    added_relations = [edge for edge in relations if edge.edge_id not in existing_edges]
    return proposal.model_copy(
        update={
            "nodes": [*added_nodes, *proposal.nodes],
            "relations": [*added_relations, *proposal.relations],
        }
    )


def canonical_id_digest(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def build_document_candidate_graph(
    *,
    processed: GovernedDocumentProcessingBundle,
    excluded_review_refreeze_sha256: str,
    graph_id: str,
    backend: KnowledgeExtractionBackend | None = None,
    proposal: KnowledgeExtractionProposal | None = None,
    approved_existing_nodes: dict[str, KnowledgeNodeTypeV3 | str] | None = None,
) -> KnowledgeConstructionBundle:
    """Extract and compile a document-bound DRAFT graph without publication rights."""

    if (backend is None) == (proposal is None):
        raise ValueError("provide exactly one of backend or proposal")
    return build_staged_document_candidate_graph(
        staging=processed.knowledge_staging,
        excluded_review_refreeze_sha256=excluded_review_refreeze_sha256,
        graph_id=graph_id,
        backend=backend,
        proposal=proposal,
        approved_existing_nodes=approved_existing_nodes,
    )


def build_staged_document_candidate_graph(
    *,
    staging: DocumentKnowledgeStagingBundle,
    excluded_review_refreeze_sha256: str,
    graph_id: str,
    backend: KnowledgeExtractionBackend | None = None,
    proposal: KnowledgeExtractionProposal | None = None,
    approved_existing_nodes: dict[str, KnowledgeNodeTypeV3 | str] | None = None,
) -> KnowledgeConstructionBundle:
    """Compile a dual-use or evaluator-only staging bundle to a DRAFT graph."""

    if (backend is None) == (proposal is None):
        raise ValueError("provide exactly one of backend or proposal")
    existing = {
        **_descriptor_scaffold_existing_nodes(staging),
        **(approved_existing_nodes or {}),
    }
    if proposal is None:
        assert backend is not None
        proposal = propose_staged_document_knowledge(
            staging=staging,
            backend=backend,
            approved_existing_nodes=existing,
        )
    require_document_proposal_grounding(
        source_profile=staging.source_profile,
        proposal=proposal,
    )
    proposal = _merge_descriptor_scaffold(staging, proposal)
    governed = govern_document_knowledge_proposal(staging=staging, proposal=proposal)
    return build_knowledge_construction_bundle(
        source_profiles=[staging.source_profile],
        source_trees=[staging.source_tree],
        proposal=governed,
        approved_existing_nodes=existing,
        approved_extraction_methods=[staging.source_profile.extraction_method],
        excluded_review_refreeze_sha256=excluded_review_refreeze_sha256,
        graph_id=graph_id,
        build_mode="NEW_EXTRACTION",
        minimum_independent_causal_sources=2,
    )


__all__ = [
    "build_document_candidate_graph",
    "document_graph_extraction_policy",
    "propose_document_knowledge",
    "propose_staged_document_knowledge",
    "build_staged_document_candidate_graph",
]
