from __future__ import annotations

from enum import Enum
import hashlib
import json
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _normalize_score_ties(items, score_field, rank_field):
    """Anchor each tolerance group at its largest score; preserve raw values.

    Approximate equality is not transitive. Pairwise minimum-rank propagation
    can merge incompatible endpoints through an intermediate near-tie.
    """
    groups = []
    for item in sorted(items, key=lambda item: (-getattr(item, score_field), item.fault_id)):
        if not groups or not math.isclose(getattr(groups[-1][0], score_field),
                                          getattr(item, score_field), rel_tol=1e-12, abs_tol=1e-12):
            groups.append([])
        groups[-1].append(item)
    ranks = {item.fault_id: min(getattr(peer, rank_field) for peer in group)
             for group in groups for item in group}
    return [item.model_copy(update={rank_field: ranks[item.fault_id]}) for item in items]


class RankingSource(str, Enum):
    DATA_DRIVEN = "DATA_DRIVEN"
    KG_FORWARD = "KG_FORWARD"
    KG_BACKWARD = "KG_BACKWARD"


class ScoreSemantics(str, Enum):
    """Meaning of the numeric ``score`` carried by one source ranking."""

    RANKING_SCORE = "RANKING_SCORE"
    UNCALIBRATED_CLASS_PROBABILITY = "UNCALIBRATED_CLASS_PROBABILITY"
    CALIBRATED_CLASS_PROBABILITY = "CALIBRATED_CLASS_PROBABILITY"
    NORMALIZED_RANKING_MASS = "NORMALIZED_RANKING_MASS"
    EVIDENCE_STRENGTH = "EVIDENCE_STRENGTH"
    EVIDENCE_COVERAGE = "EVIDENCE_COVERAGE"


class KGPartialCoverageAction(str, Enum):
    """Deterministic action when explicit KG capability coverage is incomplete.

    ``SUPPRESS_BRANCH`` remains the safe engineering default.
    ``SCALE_BY_COVERAGE`` is an experimental, label-free candidate policy whose
    empirical selection must use development evidence only.
    """

    SUPPRESS_BRANCH = "SUPPRESS_BRANCH"
    SCALE_BY_COVERAGE = "SCALE_BY_COVERAGE"


class KGFaultCoverageDeclaration(BaseModel):
    """Content-addressable declaration of executable KG/rule capability.

    This is intentionally distinct from the candidates emitted for one episode.
    A top-k ranking may be short while the underlying evidence package supports a
    broad fault universe; conversely, a ranker may enumerate zero-score candidates
    that have no executable evidence.  Only this declaration is admissible for the
    fault-universe coverage guard.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-kg-fault-coverage-declaration.v1"] = (
        "astrograph-kg-fault-coverage-declaration.v1"
    )
    coverage_basis: Literal[
        "EXECUTABLE_POSITIVE_EVIDENCE_DEFINITIONS",
        "REVIEWED_CALIBRATED_DOCUMENT_CLAIM_MAPPINGS",
    ] = (
        "EXECUTABLE_POSITIVE_EVIDENCE_DEFINITIONS"
    )
    source_artifact_id: str = Field(min_length=1)
    source_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    supported_fault_ids: tuple[str, ...] = ()
    evidence_definition_ids_by_fault: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    claim_boundary: str = (
        "Declares executable positive-vote evidence-schema capability only. It does not assert that "
        "evidence is present or satisfied in a particular episode, and it is not a "
        "performance, causal, or qualification claim."
    )

    @field_validator("supported_fault_ids")
    @classmethod
    def canonical_supported_fault_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item for item in normalized):
            raise ValueError("supported fault IDs must be nonempty")
        if len(normalized) != len(set(normalized)):
            raise ValueError("supported fault IDs must be unique after normalization")
        return tuple(sorted(normalized))

    @field_validator("evidence_definition_ids_by_fault")
    @classmethod
    def canonical_evidence_map(
        cls, value: dict[str, tuple[str, ...]]
    ) -> dict[str, tuple[str, ...]]:
        canonical: dict[str, tuple[str, ...]] = {}
        for raw_fault_id, raw_definition_ids in value.items():
            fault_id = raw_fault_id.strip()
            if not fault_id:
                raise ValueError("evidence coverage map contains an empty fault ID")
            definition_ids = tuple(item.strip() for item in raw_definition_ids)
            if not definition_ids or any(not item for item in definition_ids):
                raise ValueError(
                    f"supported fault {fault_id} requires nonempty evidence definition IDs"
                )
            if len(definition_ids) != len(set(definition_ids)):
                raise ValueError(
                    f"evidence coverage map contains duplicate definition IDs for {fault_id}"
                )
            if fault_id in canonical:
                raise ValueError(
                    f"evidence coverage map contains duplicate normalized fault ID {fault_id}"
                )
            canonical[fault_id] = tuple(sorted(definition_ids))
        return {fault_id: canonical[fault_id] for fault_id in sorted(canonical)}

    @model_validator(mode="after")
    def coverage_map_matches_supported_faults(self) -> "KGFaultCoverageDeclaration":
        if set(self.evidence_definition_ids_by_fault) != set(self.supported_fault_ids):
            raise ValueError(
                "coverage map keys must exactly equal supported_fault_ids"
            )
        return self

    @property
    def declaration_sha256(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class RankedFault(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    fault_id: str = Field(min_length=1)
    rank: int = Field(ge=1)
    score: float
    source: RankingSource
    evidence_fact_ids: list[str] = Field(default_factory=list)
    source_evidence_ids: list[str] = Field(default_factory=list)
    correlation_group_ids: list[str] = Field(default_factory=list)
    lineage_complete: bool = False
    contributes_to_fusion: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("evidence_fact_ids", "source_evidence_ids", "correlation_group_ids")
    @classmethod
    def unique_lineage_ids(cls, value: list[str]) -> list[str]:
        return sorted(set(item for item in value if item))


class FaultRanking(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal[
        "astrograph-fault-ranking.v1",
        "astrograph-fault-ranking.v2",
    ] = "astrograph-fault-ranking.v2"
    ranking_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    source: RankingSource
    score_semantics: ScoreSemantics = ScoreSemantics.RANKING_SCORE
    calibration_id: str | None = None
    candidates: list[RankedFault] = Field(default_factory=list)
    producer_id: str = Field(min_length=1)
    producer_version: str = Field(min_length=1)
    claim_boundary: str = "Engineering candidate ranking; scores are not calibrated flight probabilities."
    kg_fault_coverage: KGFaultCoverageDeclaration | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_score_semantics(self) -> "FaultRanking":
        if any(item.source != self.source for item in self.candidates):
            raise ValueError("candidate source does not match ranking source")
        probability_semantics = {
            ScoreSemantics.UNCALIBRATED_CLASS_PROBABILITY,
            ScoreSemantics.CALIBRATED_CLASS_PROBABILITY,
        }
        data_only_semantics = {*probability_semantics, ScoreSemantics.NORMALIZED_RANKING_MASS}
        if self.score_semantics in data_only_semantics and self.source != RankingSource.DATA_DRIVEN:
            raise ValueError("data-model score semantics are valid only for DATA_DRIVEN rankings")
        if self.score_semantics in data_only_semantics:
            invalid = [item.fault_id for item in self.candidates if not 0.0 <= item.score <= 1.0]
            if invalid:
                raise ValueError(
                    "probability/normalized-mass source scores must be within [0,1]: "
                    f"{sorted(invalid)}"
                )
        if self.score_semantics == ScoreSemantics.CALIBRATED_CLASS_PROBABILITY and not self.calibration_id:
            raise ValueError("calibrated class-probability ranking requires calibration_id")
        if self.source == RankingSource.DATA_DRIVEN and self.kg_fault_coverage is not None:
            raise ValueError("DATA_DRIVEN ranking cannot carry a KG fault-coverage declaration")
        if (
            self.kg_fault_coverage is not None
            and self.schema_version != "astrograph-fault-ranking.v2"
        ):
            raise ValueError(
                "KG fault-coverage declarations require astrograph-fault-ranking.v2"
            )
        if self.kg_fault_coverage is not None:
            supported = set(self.kg_fault_coverage.supported_fault_ids)
            undeclared_active = {
                item.fault_id
                for item in self.candidates
                if item.contributes_to_fusion and item.fault_id not in supported
            }
            if undeclared_active:
                raise ValueError(
                    "KG ranking contains contributing candidates outside its executable "
                    f"coverage declaration: {sorted(undeclared_active)}"
                )
        return self

    @field_validator("candidates")
    @classmethod
    def unique_candidates(cls, value: list[RankedFault]) -> list[RankedFault]:
        ids = [item.fault_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("fault IDs must be unique within one ranking")
        value = _normalize_score_ties(value, 'score', 'rank')
        scores_by_rank: dict[int, float] = {}
        for item in value:
            previous = scores_by_rank.setdefault(item.rank, item.score)
            if not math.isclose(previous, item.score, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("candidates may share a rank only when their source scores are tied")
        for left_index, left in enumerate(value):
            for right in value[left_index + 1 :]:
                if math.isclose(left.score, right.score, rel_tol=1e-12, abs_tol=1e-12):
                    continue
                if left.score > right.score and left.rank >= right.rank:
                    raise ValueError("a higher source score must have a better source rank")
                if right.score > left.score and right.rank >= left.rank:
                    raise ValueError("a higher source score must have a better source rank")
        return value


class BiRRFConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal[
        "astrograph-birrf-config.v1",
        "astrograph-birrf-config.v2",
    ] = "astrograph-birrf-config.v2"
    data_weight: float = Field(default=1.0, ge=0.0)
    forward_weight: float = Field(default=1.0, ge=0.0)
    backward_weight: float = Field(default=1.0, ge=0.0)
    data_smoothing: float = Field(default=60.0, gt=0.0)
    forward_smoothing: float = Field(default=60.0, gt=0.0)
    backward_smoothing: float = Field(default=60.0, gt=0.0)
    normal_class_id: str = "NORMAL"
    data_conflict_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    backward_conflict_coverage: float = Field(default=0.5, ge=0.0, le=1.0)
    collapse_identical_kg_rankings: bool = True
    collapse_highly_correlated_kg_rankings: bool = True
    kg_redundancy_correlation_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    kg_redundancy_min_overlap: float = Field(default=0.90, ge=0.0, le=1.0)
    collapse_shared_correlation_groups: bool = True
    warn_on_incomplete_evidence_lineage: bool = True
    require_complete_evidence_lineage: bool = False
    enforce_kg_fault_universe_coverage: bool = False
    kg_fault_coverage_semantics_version: Literal[
        "astrograph-executable-kg-coverage.v1"
    ] = "astrograph-executable-kg-coverage.v1"
    expected_fault_universe: tuple[str, ...] = ()
    minimum_kg_fault_universe_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    kg_partial_coverage_action: KGPartialCoverageAction = KGPartialCoverageAction.SUPPRESS_BRANCH
    claim_boundary: str = (
        "Default BiRRF weights and smoothing are engineering defaults copied from rank-fusion practice. "
        "Identical or highly correlated, substantially overlapping KG rank orders are collapsed to one "
        "effective vote to prevent duplicate evidence counting. Candidate-level branches that declare a shared "
        "correlation group are also capped to one effective vote; missing lineage remains explicit and may be "
        "configured as a blocking conflict. When an expected fault universe is explicitly frozen, "
        "the coverage guard consumes a content-addressed executable-evidence declaration rather than "
        "an episode candidate list. Incomplete capability may suppress a KG branch or, only under an "
        "explicit experimental policy, attenuate its weight by structural coverage. This prevents sparse "
        "knowledge from asymmetrically overriding unsupported data-branch faults without discarding the "
        "possibility of development-validated partial-coverage fusion. These controls are not calibrated "
        "for flight diagnosis."
    )

    @field_validator("expected_fault_universe")
    @classmethod
    def valid_expected_fault_universe(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item for item in normalized):
            raise ValueError("expected fault universe contains an empty fault ID")
        if len(normalized) != len(set(normalized)):
            raise ValueError("expected fault universe must contain unique IDs after normalization")
        return tuple(sorted(normalized))

    @model_validator(mode="after")
    def require_one_branch(self) -> "BiRRFConfig":
        if self.data_weight + self.forward_weight + self.backward_weight <= 0.0:
            raise ValueError("at least one BiRRF branch weight must be positive")
        if self.enforce_kg_fault_universe_coverage and not self.expected_fault_universe:
            raise ValueError(
                "KG fault-universe coverage enforcement requires expected_fault_universe"
            )
        if (
            self.enforce_kg_fault_universe_coverage
            and self.schema_version != "astrograph-birrf-config.v2"
        ):
            raise ValueError(
                "executable KG coverage enforcement requires astrograph-birrf-config.v2; "
                "v1 candidate-list coverage semantics must not be silently reinterpreted"
            )
        return self


class BiRRFFault(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    fault_id: str = Field(min_length=1)
    final_rank: int = Field(ge=1)
    fused_score: float = Field(ge=0.0)
    source_ranks: dict[str, int] = Field(default_factory=dict)
    source_scores: dict[str, float] = Field(default_factory=dict)
    raw_reciprocal_contributions: dict[str, float] = Field(default_factory=dict)
    reciprocal_contributions: dict[str, float] = Field(default_factory=dict)
    supporting_evidence_fact_ids: list[str] = Field(default_factory=list)
    source_evidence_ids_by_source: dict[str, list[str]] = Field(default_factory=dict)
    correlation_group_ids_by_source: dict[str, list[str]] = Field(default_factory=dict)
    lineage_complete_by_source: dict[str, bool] = Field(default_factory=dict)
    shared_correlation_groups: dict[str, list[str]] = Field(default_factory=dict)
    dependency_adjusted: bool = False


class BiRRFResult(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: str = "astrograph-birrf-result.v1"
    fusion_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    config: BiRRFConfig
    candidates: list[BiRRFFault]
    source_ranking_ids: dict[str, str]
    top_fault_id: str | None = None
    normal_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    normal_rank: int | None = Field(default=None, ge=1)
    normal_score_semantics: str | None = None
    fused_score_semantics: Literal["RECIPROCAL_RANK_SCORE_NOT_PROBABILITY"] = (
        "RECIPROCAL_RANK_SCORE_NOT_PROBABILITY"
    )
    # Legacy aggregate diagnostics retained for backward-compatible reports and prompts.
    # Formal policy gates must consume ``blocking_conflicts`` rather than treating every
    # diagnostic flag as an evidence contradiction.
    conflict_flags: list[str] = Field(default_factory=list)
    blocking_conflicts: list[str] = Field(default_factory=list)
    quality_warnings: list[str] = Field(default_factory=list)
    open_set_actions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    claim_boundary: str = (
        "BiRRF orders candidates for investigation and compact prompting only. Its fused_score is a "
        "weighted reciprocal-rank score, never a class probability. It cannot bypass evidence "
        "validation, Safety Critic, or Decision Governor."
    )

    @field_validator("candidates")
    @classmethod
    def unique_fused_candidates(cls, value: list[BiRRFFault]) -> list[BiRRFFault]:
        ids = [item.fault_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("fused fault IDs must be unique")
        value = _normalize_score_ties(value, 'fused_score', 'final_rank')
        scores_by_rank: dict[int, float] = {}
        for item in value:
            previous = scores_by_rank.setdefault(item.final_rank, item.fused_score)
            if not math.isclose(previous, item.fused_score, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("candidates may share a final rank only when their fused scores are tied")
        for left_index, left in enumerate(value):
            for right in value[left_index + 1 :]:
                if math.isclose(left.fused_score, right.fused_score, rel_tol=1e-12, abs_tol=1e-12):
                    continue
                if left.fused_score > right.fused_score and left.final_rank >= right.final_rank:
                    raise ValueError("a higher fused score must have a better final rank")
                if right.fused_score > left.fused_score and right.final_rank >= left.final_rank:
                    raise ValueError("a higher fused score must have a better final rank")
        return value


class CandidateFusionPackage(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: str = "astrograph-candidate-fusion-package.v1"
    episode_id: str = Field(min_length=1)
    data_ranking: FaultRanking
    forward_ranking: FaultRanking
    backward_ranking: FaultRanking
    birrf_result: BiRRFResult
    registered_fault_ids: list[str]
    generated_by: str = "astrograph-birrf-service"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_episode_and_faults(self) -> "CandidateFusionPackage":
        episode_ids = {
            self.data_ranking.episode_id,
            self.forward_ranking.episode_id,
            self.backward_ranking.episode_id,
            self.birrf_result.episode_id,
            self.episode_id,
        }
        if len(episode_ids) != 1:
            raise ValueError("all candidate-fusion documents must refer to one episode")
        registered = set(self.registered_fault_ids)
        for ranking in (self.data_ranking, self.forward_ranking, self.backward_ranking):
            unknown = {item.fault_id for item in ranking.candidates} - registered
            if unknown:
                raise ValueError(f"ranking contains unregistered faults: {sorted(unknown)}")
        return self
