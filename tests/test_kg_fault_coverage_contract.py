from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from astrograph.evidence import EpisodeEvidenceBundle, EvidenceRole
from astrograph.fusion import (
    BiRRFConfig,
    FaultRanking,
    KGFaultCoverageDeclaration,
    KGPartialCoverageAction,
    RankedFault,
    RankingSource,
    ScoreSemantics,
    build_backward_evidence_ranking,
    build_backward_evidence_ranking_from_associations,
    build_forward_graph_ranking,
    build_forward_graph_ranking_from_associations,
    fuse_rankings,
)
from astrograph.knowledge import GraphEvidenceAssociation

ROOT = Path(__file__).resolve().parents[1]


def _declaration(*fault_ids: str, source_hash: str = "a" * 64) -> KGFaultCoverageDeclaration:
    return KGFaultCoverageDeclaration(
        source_artifact_id="test-executable-evidence@1",
        source_artifact_sha256=source_hash,
        supported_fault_ids=tuple(fault_ids),
        evidence_definition_ids_by_fault={
            fault_id: (f"ED-{fault_id}",) for fault_id in fault_ids
        },
    )


def _ranking(
    source: RankingSource,
    rows: list[tuple[str, float, bool]],
    *,
    declaration: KGFaultCoverageDeclaration | None = None,
) -> FaultRanking:
    semantics = {
        RankingSource.DATA_DRIVEN: ScoreSemantics.UNCALIBRATED_CLASS_PROBABILITY,
        RankingSource.KG_FORWARD: ScoreSemantics.EVIDENCE_STRENGTH,
        RankingSource.KG_BACKWARD: ScoreSemantics.EVIDENCE_COVERAGE,
    }[source]
    ordered = sorted(rows, key=lambda item: (-item[1], item[0]))
    return FaultRanking(
        ranking_id=f"ranking-{source.value.lower()}-{len(rows)}",
        episode_id="episode-coverage-contract",
        source=source,
        score_semantics=semantics,
        candidates=[
            RankedFault(
                fault_id=fault_id,
                rank=index,
                score=score,
                source=source,
                contributes_to_fusion=contributes,
                lineage_complete=True,
            )
            for index, (fault_id, score, contributes) in enumerate(ordered, start=1)
        ],
        producer_id="coverage-contract-test",
        producer_version="1",
        kg_fault_coverage=declaration,
    )


def _data() -> FaultRanking:
    return _ranking(
        RankingSource.DATA_DRIVEN,
        [("F-A", 0.8, True), ("F-B", 0.15, True), ("F-C", 0.05, True)],
    )


def _guard_config(**updates: object) -> BiRRFConfig:
    base = {
        "enforce_kg_fault_universe_coverage": True,
        "expected_fault_universe": ("F-A", "F-B", "F-C"),
        "minimum_kg_fault_universe_coverage": 1.0,
        "collapse_identical_kg_rankings": False,
        "collapse_highly_correlated_kg_rankings": False,
    }
    base.update(updates)
    return BiRRFConfig(**base)


def test_rank_builders_attach_content_addressed_executable_coverage() -> None:
    bundle = EpisodeEvidenceBundle.model_validate_json(
        (
            ROOT
            / "tests/fixtures/evidence/reaction_wheel_friction_increase/diagnosis/episode_evidence_bundle.json"
        ).read_text(encoding="utf-8")
    )
    expected_supported = tuple(
        sorted(
            {
                item.target_fault_id
                for item in bundle.evidence_definitions
                if item.role in {EvidenceRole.REQUIRED, EvidenceRole.SUPPORTING}
            }
        )
    )

    forward = build_forward_graph_ranking(bundle)
    backward = build_backward_evidence_ranking(
        bundle,
        candidate_fault_ids=[*expected_supported, "FAULT_WITHOUT_EXECUTABLE_EVIDENCE"],
    )

    assert forward.schema_version == "astrograph-fault-ranking.v2"
    assert backward.schema_version == "astrograph-fault-ranking.v2"
    assert forward.producer_version == "0.7.1.post24+b006.wip.kgcoverage.v1"
    assert backward.producer_version == "0.7.1.post24+b006.wip.kgcoverage.v1"
    assert forward.kg_fault_coverage is not None
    assert backward.kg_fault_coverage is not None
    assert forward.kg_fault_coverage.supported_fault_ids == expected_supported
    assert backward.kg_fault_coverage.supported_fault_ids == expected_supported
    assert (
        forward.kg_fault_coverage.declaration_sha256
        == backward.kg_fault_coverage.declaration_sha256
    )
    assert "FAULT_WITHOUT_EXECUTABLE_EVIDENCE" in {
        item.fault_id for item in backward.candidates
    }
    assert "FAULT_WITHOUT_EXECUTABLE_EVIDENCE" not in set(
        backward.kg_fault_coverage.supported_fault_ids
    )
    unsupported = next(
        item
        for item in backward.candidates
        if item.fault_id == "FAULT_WITHOUT_EXECUTABLE_EVIDENCE"
    )
    assert unsupported.contributes_to_fusion is False



def test_graph_query_projection_cannot_overclaim_full_package_coverage() -> None:
    bundle = EpisodeEvidenceBundle.model_validate_json(
        (
            ROOT
            / "tests/fixtures/evidence/reaction_wheel_friction_increase/diagnosis/episode_evidence_bundle.json"
        ).read_text(encoding="utf-8")
    )
    selected = next(
        item
        for item in bundle.evidence_definitions
        if item.role == EvidenceRole.REQUIRED
    )
    association = GraphEvidenceAssociation(
        fault_id=selected.target_fault_id,
        evidence_definition_id=selected.evidence_definition_id,
        relation_type="REQUIRES_EVIDENCE",
        source_query_plan_id="GQ-COVERAGE-PROJECTION",
    )

    full = build_forward_graph_ranking(bundle)
    projected_forward = build_forward_graph_ranking_from_associations(bundle, [association])
    projected_backward = build_backward_evidence_ranking_from_associations(
        bundle,
        [association],
    )

    assert full.kg_fault_coverage is not None
    assert projected_forward.kg_fault_coverage is not None
    assert projected_backward.kg_fault_coverage is not None
    assert projected_forward.kg_fault_coverage.supported_fault_ids == (
        selected.target_fault_id,
    )
    assert (
        projected_forward.kg_fault_coverage.declaration_sha256
        == projected_backward.kg_fault_coverage.declaration_sha256
    )
    assert (
        projected_forward.kg_fault_coverage.declaration_sha256
        != full.kg_fault_coverage.declaration_sha256
    )
    # This fixture has no configured-runtime package hash, so the fallback source
    # identity is the canonical projected definition set and must also change.
    assert (
        projected_forward.kg_fault_coverage.source_artifact_sha256
        != full.kg_fault_coverage.source_artifact_sha256
    )

def test_top_k_candidate_truncation_does_not_reduce_capability_coverage() -> None:
    declaration = _declaration("F-A", "F-B", "F-C")
    forward = _ranking(
        RankingSource.KG_FORWARD,
        [("F-A", 1.0, True)],
        declaration=declaration,
    )
    backward = _ranking(
        RankingSource.KG_BACKWARD,
        [("F-B", 1.0, True)],
        declaration=declaration,
    )

    result = fuse_rankings(_data(), forward, backward, config=_guard_config())

    assert result.metadata["kg_ranked_candidate_fault_ids"] == {
        "KG_FORWARD": ["F-A"],
        "KG_BACKWARD": ["F-B"],
    }
    assert result.metadata["kg_declared_executable_fault_ids"] == {
        "KG_FORWARD": ["F-A", "F-B", "F-C"],
        "KG_BACKWARD": ["F-A", "F-B", "F-C"],
    }
    assert result.metadata["kg_executable_fault_universe_coverage"] == {
        "KG_FORWARD": 1.0,
        "KG_BACKWARD": 1.0,
    }
    assert result.metadata["suppressed_kg_sources"] == []


def test_zero_score_enumeration_does_not_inflate_executable_coverage() -> None:
    declaration = _declaration("F-B")
    rows = [("F-B", 1.0, True), ("F-A", 0.0, False), ("F-C", 0.0, False)]
    forward = _ranking(RankingSource.KG_FORWARD, rows, declaration=declaration)
    backward = _ranking(RankingSource.KG_BACKWARD, rows, declaration=declaration)

    result = fuse_rankings(_data(), forward, backward, config=_guard_config())

    assert result.metadata["kg_ranked_candidate_fault_ids"] == {
        "KG_FORWARD": ["F-A", "F-B", "F-C"],
        "KG_BACKWARD": ["F-A", "F-B", "F-C"],
    }
    assert result.metadata["kg_executable_fault_universe_coverage"] == {
        "KG_FORWARD": pytest.approx(1 / 3),
        "KG_BACKWARD": pytest.approx(1 / 3),
    }
    assert result.metadata["suppressed_kg_sources"] == ["KG_BACKWARD", "KG_FORWARD"]
    assert result.top_fault_id == "F-A"


def test_missing_declaration_degrades_to_data_only_without_crashing() -> None:
    forward = _ranking(RankingSource.KG_FORWARD, [("F-B", 1.0, True)])
    backward = _ranking(RankingSource.KG_BACKWARD, [("F-B", 1.0, True)])

    result = fuse_rankings(_data(), forward, backward, config=_guard_config())

    assert result.top_fault_id == "F-A"
    assert result.metadata["kg_executable_fault_universe_coverage"] == {
        "KG_FORWARD": None,
        "KG_BACKWARD": None,
    }
    assert result.metadata["kg_suppression_reasons"] == {
        "KG_FORWARD": "MISSING_EXECUTABLE_COVERAGE_DECLARATION",
        "KG_BACKWARD": "MISSING_EXECUTABLE_COVERAGE_DECLARATION",
    }


def test_contributing_candidate_outside_declaration_is_rejected() -> None:
    with pytest.raises(ValidationError, match="outside its executable coverage declaration"):
        _ranking(
            RankingSource.KG_FORWARD,
            [("F-A", 1.0, True), ("F-B", 0.5, True)],
            declaration=_declaration("F-A"),
        )


def test_model_copy_bypass_of_declaration_is_suppressed_at_fusion_boundary() -> None:
    valid = _ranking(
        RankingSource.KG_FORWARD,
        [("F-A", 1.0, True)],
        declaration=_declaration("F-A"),
    )
    bypassed = valid.model_copy(
        update={
            "candidates": [
                RankedFault(
                    fault_id="F-B",
                    rank=1,
                    score=1.0,
                    source=RankingSource.KG_FORWARD,
                    contributes_to_fusion=True,
                )
            ]
        }
    )
    backward = _ranking(
        RankingSource.KG_BACKWARD,
        [("F-A", 1.0, True)],
        declaration=_declaration("F-A"),
    )

    result = fuse_rankings(_data(), bypassed, backward, config=_guard_config())

    assert result.metadata["kg_suppression_reasons"]["KG_FORWARD"] == (
        "ACTIVE_CANDIDATE_OUTSIDE_EXECUTABLE_COVERAGE_DECLARATION"
    )
    assert result.metadata["effective_branch_weights"]["KG_FORWARD"] == 0.0


def test_active_candidate_outside_expected_universe_is_suppressed() -> None:
    forward = _ranking(
        RankingSource.KG_FORWARD,
        [("F-X", 1.0, True)],
        declaration=_declaration("F-A", "F-B", "F-C", "F-X"),
    )
    backward = _ranking(
        RankingSource.KG_BACKWARD,
        [("F-A", 1.0, True)],
        declaration=_declaration("F-A", "F-B", "F-C"),
    )

    result = fuse_rankings(_data(), forward, backward, config=_guard_config())

    assert result.metadata["kg_suppression_reasons"]["KG_FORWARD"] == (
        "ACTIVE_CANDIDATE_OUTSIDE_EXPECTED_FAULT_UNIVERSE"
    )
    assert result.metadata["kg_active_fault_ids_outside_expected_universe"]["KG_FORWARD"] == [
        "F-X"
    ]


def test_scale_by_coverage_is_explicit_deterministic_candidate_policy() -> None:
    declaration = _declaration("F-A", "F-B")
    forward = _ranking(
        RankingSource.KG_FORWARD,
        [("F-B", 1.0, True)],
        declaration=declaration,
    )
    backward = _ranking(
        RankingSource.KG_BACKWARD,
        [("F-A", 1.0, True)],
        declaration=declaration,
    )
    scaled = fuse_rankings(
        _data(),
        forward,
        backward,
        config=_guard_config(
            forward_weight=1.5,
            backward_weight=0.0,
            kg_partial_coverage_action=KGPartialCoverageAction.SCALE_BY_COVERAGE,
        ),
    )
    suppressed = fuse_rankings(
        _data(),
        forward,
        backward,
        config=_guard_config(forward_weight=1.5, backward_weight=0.0),
    )

    assert scaled.metadata["kg_partial_coverage_action"] == "SCALE_BY_COVERAGE"
    assert scaled.metadata["kg_attenuation_factors"]["KG_FORWARD"] == pytest.approx(2 / 3)
    assert scaled.metadata["effective_branch_weights"]["KG_FORWARD"] == pytest.approx(1.0)
    assert scaled.metadata["attenuated_kg_sources"] == ["KG_FORWARD"]
    assert suppressed.metadata["effective_branch_weights"]["KG_FORWARD"] == 0.0
    assert suppressed.metadata["suppressed_kg_sources"] == ["KG_FORWARD"]


def test_suppressed_sparse_kg_matches_data_only_fused_scores() -> None:
    declaration = _declaration("F-B")
    forward = _ranking(
        RankingSource.KG_FORWARD,
        [("F-B", 1.0, True)],
        declaration=declaration,
    )
    backward = _ranking(
        RankingSource.KG_BACKWARD,
        [("F-B", 1.0, True)],
        declaration=declaration,
    )
    guarded = fuse_rankings(_data(), forward, backward, config=_guard_config())
    data_only = fuse_rankings(
        _data(),
        forward,
        backward,
        config=BiRRFConfig(
            forward_weight=0.0,
            backward_weight=0.0,
            collapse_identical_kg_rankings=False,
            collapse_highly_correlated_kg_rankings=False,
        ),
    )

    guarded_rows = [
        (item.fault_id, item.final_rank, item.fused_score, item.reciprocal_contributions)
        for item in guarded.candidates
    ]
    data_only_rows = [
        (item.fault_id, item.final_rank, item.fused_score, item.reciprocal_contributions)
        for item in data_only.candidates
    ]
    assert guarded_rows == data_only_rows


def test_declaration_digest_is_canonical_and_content_sensitive() -> None:
    first = KGFaultCoverageDeclaration(
        source_artifact_id="pkg@1",
        source_artifact_sha256="a" * 64,
        supported_fault_ids=("F-B", "F-A"),
        evidence_definition_ids_by_fault={
            "F-B": ("ED-B-2", "ED-B-1"),
            "F-A": ("ED-A",),
        },
    )
    reordered = KGFaultCoverageDeclaration(
        source_artifact_id="pkg@1",
        source_artifact_sha256="a" * 64,
        supported_fault_ids=("F-A", "F-B"),
        evidence_definition_ids_by_fault={
            "F-A": ("ED-A",),
            "F-B": ("ED-B-1", "ED-B-2"),
        },
    )
    changed = reordered.model_copy(update={"source_artifact_sha256": "b" * 64})

    assert first.declaration_sha256 == reordered.declaration_sha256
    assert changed.declaration_sha256 != reordered.declaration_sha256


def test_v1_birrf_guard_semantics_are_not_silently_reinterpreted() -> None:
    with pytest.raises(ValidationError, match="v1 candidate-list coverage semantics"):
        BiRRFConfig(
            schema_version="astrograph-birrf-config.v1",
            enforce_kg_fault_universe_coverage=True,
            expected_fault_universe=("F-A",),
        )


def test_v1_fault_ranking_cannot_carry_v2_coverage_declaration() -> None:
    with pytest.raises(ValidationError, match="fault-ranking.v2"):
        FaultRanking(
            schema_version="astrograph-fault-ranking.v1",
            ranking_id="legacy-ranking",
            episode_id="episode-coverage-contract",
            source=RankingSource.KG_FORWARD,
            score_semantics=ScoreSemantics.EVIDENCE_STRENGTH,
            candidates=[
                RankedFault(
                    fault_id="F-A",
                    rank=1,
                    score=1.0,
                    source=RankingSource.KG_FORWARD,
                )
            ],
            producer_id="legacy",
            producer_version="1",
            kg_fault_coverage=_declaration("F-A"),
        )



def test_coverage_identifiers_are_canonicalized_before_hashing() -> None:
    declaration = KGFaultCoverageDeclaration(
        source_artifact_id="pkg@1",
        source_artifact_sha256="a" * 64,
        supported_fault_ids=(" F-B ", "F-A"),
        evidence_definition_ids_by_fault={
            " F-B ": (" ED-B-2 ", "ED-B-1"),
            "F-A": ("ED-A",),
        },
    )
    config = BiRRFConfig(expected_fault_universe=(" F-B ", "F-A"))

    assert declaration.supported_fault_ids == ("F-A", "F-B")
    assert declaration.evidence_definition_ids_by_fault == {
        "F-A": ("ED-A",),
        "F-B": ("ED-B-1", "ED-B-2"),
    }
    assert config.expected_fault_universe == ("F-A", "F-B")


def test_normalized_identifier_collisions_are_rejected() -> None:
    with pytest.raises(ValidationError, match="unique after normalization"):
        KGFaultCoverageDeclaration(
            source_artifact_id="pkg@1",
            source_artifact_sha256="a" * 64,
            supported_fault_ids=("F-A", " F-A "),
            evidence_definition_ids_by_fault={"F-A": ("ED-A",)},
        )

    with pytest.raises(ValidationError, match="duplicate normalized fault ID"):
        KGFaultCoverageDeclaration(
            source_artifact_id="pkg@1",
            source_artifact_sha256="a" * 64,
            supported_fault_ids=("F-A",),
            evidence_definition_ids_by_fault={
                "F-A": ("ED-A",),
                " F-A ": ("ED-A-2",),
            },
        )
