"""Content-addressed approval for vocabulary/TBox semantics only.

This deliberately cannot publish an Active KG or approve episode/causal/runtime
claims.  It records the project owner's 2026-09-02 adjudication without
conflating it with the still-frozen dataset-mapping review.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .schema_v3 import KnowledgeGraphV3, KnowledgeNodeTypeV3, KnowledgeRelationTypeV3


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


class TBoxVocabularyApproval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-tbox-vocabulary-approval.v1"] = (
        "astrograph-tbox-vocabulary-approval.v1"
    )
    graph_id: str
    graph_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_source: str = Field(min_length=1)
    decision_actor: str = Field(min_length=1)
    decision_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    approved_failure_mechanism_ids: list[str] = Field(min_length=12, max_length=12)
    approved_effect_pattern_ids: list[str] = Field(min_length=10, max_length=10)
    approved_tbox_association_edge_ids: list[str] = Field(min_length=1)
    association_semantics: Literal["TBOX_ASSOCIATION_ONLY"] = "TBOX_ASSOCIATION_ONLY"
    episode_evidence_approved: Literal[False] = False
    actual_causal_propagation_approved: Literal[False] = False
    runtime_diagnosis_or_voting_allowed: Literal[False] = False
    active_graph_modified: Literal[False] = False
    dataset_mapping_review_unfrozen: Literal[False] = False
    claim_boundary: str = Field(min_length=1)
    approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_identity(self) -> "TBoxVocabularyApproval":
        for values in (
            self.approved_failure_mechanism_ids,
            self.approved_effect_pattern_ids,
            self.approved_tbox_association_edge_ids,
        ):
            if values != sorted(set(values)):
                raise ValueError("approved identifiers must be unique and sorted")
        expected = canonical_sha256(self.model_dump(mode="json", exclude={"approval_sha256"}))
        if self.approval_sha256 != expected:
            raise ValueError("approval_sha256 mismatch")
        return self


def approve_cross_domain_tbox_vocabulary(
    graph: KnowledgeGraphV3,
    *,
    decision_source: str,
    decision_actor: str = "project-owner",
    decision_date: str = "2026-09-02",
) -> TBoxVocabularyApproval:
    """Approve exactly 12 mechanisms, 10 effects and their MANIFESTS_AS TBox edges."""
    mechanisms = sorted(
        node.node_id for node in graph.nodes
        if node.node_type == KnowledgeNodeTypeV3.FAILURE_MECHANISM
    )
    effects = sorted(
        node.node_id for node in graph.nodes
        if node.node_type == KnowledgeNodeTypeV3.EFFECT_PATTERN
    )
    if len(mechanisms) != 12 or len(effects) != 10:
        raise ValueError(
            f"approval scope requires exactly 12 mechanisms and 10 effects; got {len(mechanisms)} and {len(effects)}"
        )
    mechanism_set, effect_set = set(mechanisms), set(effects)
    associations = sorted(
        edge.edge_id for edge in graph.edges
        if edge.relation_type == KnowledgeRelationTypeV3.MANIFESTS_AS
        and edge.source_id in mechanism_set
        and edge.target_id in effect_set
    )
    if not associations:
        raise ValueError("approval scope contains no mechanism-to-effect TBox associations")
    payload = {
        "schema_version": "astrograph-tbox-vocabulary-approval.v1",
        "graph_id": graph.graph_id,
        "graph_sha256": canonical_sha256(graph.model_dump(mode="json")),
        "decision_source": decision_source,
        "decision_actor": decision_actor,
        "decision_date": decision_date,
        "approved_failure_mechanism_ids": mechanisms,
        "approved_effect_pattern_ids": effects,
        "approved_tbox_association_edge_ids": associations,
        "association_semantics": "TBOX_ASSOCIATION_ONLY",
        "episode_evidence_approved": False,
        "actual_causal_propagation_approved": False,
        "runtime_diagnosis_or_voting_allowed": False,
        "active_graph_modified": False,
        "dataset_mapping_review_unfrozen": False,
        "claim_boundary": (
            "The ontology asserts vocabulary-level association only. It does not establish an observed "
            "episode, real causal propagation, or permission for runtime diagnosis/voting."
        ),
    }
    return TBoxVocabularyApproval(**payload, approval_sha256=canonical_sha256(payload))


def write_tbox_vocabulary_approval(approval: TBoxVocabularyApproval, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(approval.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return target


__all__ = ["TBoxVocabularyApproval", "approve_cross_domain_tbox_vocabulary", "write_tbox_vocabulary_approval"]
