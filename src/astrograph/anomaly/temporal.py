from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from astrograph.anomaly.episodes import AnomalyEpisode


class TemporalRelationType(str, Enum):
    OVERLAPS = "OVERLAPS"
    STARTS_WITHIN = "STARTS_WITHIN"
    ANCHOR_PRECEDES_WITHIN = "ANCHOR_PRECEDES_WITHIN"
    RELATED_PRECEDES_WITHIN = "RELATED_PRECEDES_WITHIN"
    CO_OCCURS = "CO_OCCURS"


class EpisodeTemporalRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    anchor_episode_id: str
    anchor_anomaly_id: str
    related_episode_id: str
    related_anomaly_id: str
    relation: TemporalRelationType
    start_delta_s: float
    gap_s: float
    overlap_s: float = Field(ge=0.0)


class EpisodeTemporalConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    constraint_id: str = Field(min_length=1)
    fault_id: str = Field(min_length=1)
    anchor_anomaly_id: str = Field(min_length=1)
    related_anomaly_ids: list[str] = Field(min_length=1)
    allowed_relations: list[TemporalRelationType] = Field(min_length=1)
    max_lag_s: float = Field(default=5.0, ge=0.0)
    min_overlap_s: float = Field(default=0.0, ge=0.0)
    minimum_matches: int = Field(default=1, ge=1)
    action: str = "block_diagnosis"
    penalty: float = Field(default=0.0, ge=0.0)
    rationale: str = ""
    evidence_level: str = "E0"

    @field_validator("action")
    @classmethod
    def validate_action(cls, value: str) -> str:
        if value not in {"block_diagnosis", "penalize"}:
            raise ValueError("action must be block_diagnosis or penalize")
        return value


class EpisodeTemporalConstraintCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-episode-temporal-constraint-catalog.v1"
    catalog_version: str
    constraints: list[EpisodeTemporalConstraint] = Field(default_factory=list)
    claim_boundary: str = "E0 temporal engineering constraints; not a calibrated causal probability."

    @classmethod
    def from_json(cls, path: str | Path) -> "EpisodeTemporalConstraintCatalog":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))

    def constraints_for(self, fault_id: str) -> list[EpisodeTemporalConstraint]:
        return [item for item in self.constraints if item.fault_id == fault_id]


class TemporalConstraintEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    constraint_id: str
    fault_id: str
    satisfied: bool
    action: str
    penalty: float = 0.0
    matched_relation_count: int = 0
    required_match_count: int = 1
    matched_relations: list[EpisodeTemporalRelation] = Field(default_factory=list)
    rationale: str = ""
    evidence_level: str = "E0"


class EpisodeTemporalAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-episode-temporal-analysis.v1"
    catalog_version: str
    relation_count: int
    evaluations_by_fault: dict[str, list[TemporalConstraintEvaluation]] = Field(default_factory=dict)
    relations: list[EpisodeTemporalRelation] = Field(default_factory=list)
    claim_boundary: str


def _interval_metrics(anchor: AnomalyEpisode, related: AnomalyEpisode) -> tuple[float, float, float]:
    start_delta = related.start_s - anchor.start_s
    overlap = max(0.0, min(anchor.end_s, related.end_s) - max(anchor.start_s, related.start_s))
    if overlap > 0.0 or (anchor.start_s == related.start_s and anchor.end_s == related.end_s):
        gap = 0.0
    elif anchor.end_s < related.start_s:
        gap = related.start_s - anchor.end_s
    else:
        gap = anchor.start_s - related.end_s
    return start_delta, gap, overlap


def _matches(
    relation_type: TemporalRelationType,
    anchor: AnomalyEpisode,
    related: AnomalyEpisode,
    *,
    max_lag_s: float,
    min_overlap_s: float,
) -> EpisodeTemporalRelation | None:
    start_delta, gap, overlap = _interval_metrics(anchor, related)
    matched = False
    if relation_type == TemporalRelationType.OVERLAPS:
        matched = overlap >= min_overlap_s and (overlap > 0.0 or min_overlap_s == 0.0)
    elif relation_type == TemporalRelationType.STARTS_WITHIN:
        matched = abs(start_delta) <= max_lag_s
    elif relation_type == TemporalRelationType.ANCHOR_PRECEDES_WITHIN:
        matched = anchor.end_s <= related.start_s and (related.start_s - anchor.end_s) <= max_lag_s
    elif relation_type == TemporalRelationType.RELATED_PRECEDES_WITHIN:
        matched = related.end_s <= anchor.start_s and (anchor.start_s - related.end_s) <= max_lag_s
    elif relation_type == TemporalRelationType.CO_OCCURS:
        matched = overlap >= min_overlap_s or gap <= max_lag_s
    if not matched:
        return None
    return EpisodeTemporalRelation(
        run_id=anchor.run_id,
        anchor_episode_id=anchor.episode_id,
        anchor_anomaly_id=anchor.anomaly_id,
        related_episode_id=related.episode_id,
        related_anomaly_id=related.anomaly_id,
        relation=relation_type,
        start_delta_s=start_delta,
        gap_s=gap,
        overlap_s=overlap,
    )


def analyze_episode_temporal_constraints(
    episodes: Iterable[AnomalyEpisode],
    catalog: EpisodeTemporalConstraintCatalog,
) -> EpisodeTemporalAnalysis:
    items = list(episodes)
    run_ids = {item.run_id for item in items}
    if len(run_ids) > 1:
        raise ValueError("temporal analysis requires episodes from one run")
    by_anomaly: dict[str, list[AnomalyEpisode]] = {}
    for item in items:
        by_anomaly.setdefault(item.anomaly_id, []).append(item)

    all_relations: dict[tuple[str, str, str], EpisodeTemporalRelation] = {}
    evaluations_by_fault: dict[str, list[TemporalConstraintEvaluation]] = {}
    for constraint in catalog.constraints:
        # A constraint is applicable only when its anchor mechanism episode is present.
        # This avoids reporting unrelated faults as temporal violations.
        if not by_anomaly.get(constraint.anchor_anomaly_id):
            continue
        matched: list[EpisodeTemporalRelation] = []
        for anchor in by_anomaly.get(constraint.anchor_anomaly_id, []):
            for related_id in constraint.related_anomaly_ids:
                for related in by_anomaly.get(related_id, []):
                    if anchor.episode_id == related.episode_id:
                        continue
                    for relation_type in constraint.allowed_relations:
                        relation = _matches(
                            relation_type,
                            anchor,
                            related,
                            max_lag_s=constraint.max_lag_s,
                            min_overlap_s=constraint.min_overlap_s,
                        )
                        if relation is not None:
                            matched.append(relation)
                            all_relations[(relation.anchor_episode_id, relation.related_episode_id, relation.relation.value)] = relation
                            break
        unique = sorted(
            {item.model_dump_json(): item for item in matched}.values(),
            key=lambda item: (item.anchor_episode_id, item.related_episode_id, item.relation.value),
        )
        satisfied = len(unique) >= constraint.minimum_matches
        evaluation = TemporalConstraintEvaluation(
            constraint_id=constraint.constraint_id,
            fault_id=constraint.fault_id,
            satisfied=satisfied,
            action=constraint.action,
            penalty=0.0 if satisfied else constraint.penalty,
            matched_relation_count=len(unique),
            required_match_count=constraint.minimum_matches,
            matched_relations=unique,
            rationale=constraint.rationale,
            evidence_level=constraint.evidence_level,
        )
        evaluations_by_fault.setdefault(constraint.fault_id, []).append(evaluation)

    return EpisodeTemporalAnalysis(
        catalog_version=catalog.catalog_version,
        relation_count=len(all_relations),
        evaluations_by_fault=evaluations_by_fault,
        relations=sorted(
            all_relations.values(),
            key=lambda item: (item.anchor_episode_id, item.related_episode_id, item.relation.value),
        ),
        claim_boundary=catalog.claim_boundary,
    )
