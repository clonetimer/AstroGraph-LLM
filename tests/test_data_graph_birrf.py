from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from astrograph.data_driven import ClassProbability, DataDrivenPrediction
from astrograph.evidence import EpisodeEvidenceBundle, EvidenceFactStatus, EvidenceRole
from astrograph.evidence.executable import stable_id
from astrograph.fusion import (
    BiRRFConfig,
    CandidateFusionPackage,
    FaultRanking,
    KGFaultCoverageDeclaration,
    RankedFault,
    RankingSource,
    build_backward_evidence_ranking,
    build_candidate_fusion_package,
    build_forward_graph_ranking,
    fuse_rankings,
    select_fused_candidates,
)
from astrograph.llm import OpenAICompatibleConfig, OpenAICompatibleStructuredBackend
from astrograph.lineage import observation_correlation_group_id


ROOT = Path(__file__).resolve().parents[1]
BUNDLES = ROOT / "tests/fixtures/evidence"


def _bundle(case_id: str) -> EpisodeEvidenceBundle:
    return EpisodeEvidenceBundle.model_validate_json(
        (BUNDLES / case_id / "diagnosis/episode_evidence_bundle.json").read_text(encoding="utf-8")
    )


def _prediction(bundle: EpisodeEvidenceBundle, ordered: list[tuple[str, float]]) -> DataDrivenPrediction:
    classes = [
        ClassProbability(class_id=class_id, probability=probability, rank=index)
        for index, (class_id, probability) in enumerate(ordered, start=1)
    ]
    return DataDrivenPrediction(
        prediction_id=stable_id("PRED", {"episode": bundle.episode.episode_id, "classes": ordered}),
        episode_id=bundle.episode.episode_id,
        model_id="fixture-hgb",
        model_version="0.5.14-test",
        classes=classes,
        normal_class_id="NORMAL",
        feature_schema_id="FEATURE-SCHEMA-TEST",
        input_data_sha256="0" * 64,
    )


def _kg_coverage(*fault_ids: str) -> KGFaultCoverageDeclaration:
    return KGFaultCoverageDeclaration(
        source_artifact_id="test-evidence-package@1",
        source_artifact_sha256="a" * 64,
        supported_fault_ids=tuple(fault_ids),
        evidence_definition_ids_by_fault={
            fault_id: (f"ED-{fault_id}",) for fault_id in fault_ids
        },
    )


def test_forward_and_backward_rankings_follow_executable_evidence() -> None:
    bundle = _bundle("reaction_wheel_friction_increase")
    forward = build_forward_graph_ranking(bundle)
    backward = build_backward_evidence_ranking(bundle)
    assert forward.source == RankingSource.KG_FORWARD
    assert backward.source == RankingSource.KG_BACKWARD
    assert forward.candidates[0].fault_id == "ADCS_RW_FRICTION_INCREASE"
    assert backward.candidates[0].fault_id == "ADCS_RW_FRICTION_INCREASE"
    assert backward.candidates[0].score == pytest.approx(1.0)
    assert forward.candidates[0].evidence_fact_ids


def test_graph_ranking_lineage_uses_branch_neutral_observation_group() -> None:
    bundle = _bundle("reaction_wheel_friction_increase")
    forward = build_forward_graph_ranking(bundle)
    top = forward.candidates[0]
    fact_id = top.evidence_fact_ids[0]
    fact = next(item for item in bundle.evidence_facts if item.evidence_fact_id == fact_id)
    feature = next(item for item in bundle.feature_results if item.feature_result_id == fact.feature_result_id)
    definition = next(
        item for item in bundle.feature_definitions if item.feature_definition_id == feature.feature_definition_id
    )
    semantic_id = definition.required_semantics[0] if definition.required_semantics else feature.resolved_field
    expected = observation_correlation_group_id(
        episode_id=feature.episode_id,
        window_start_s=feature.window_start_s,
        window_end_s=feature.window_end_s,
        source_field=feature.resolved_field,
        semantic_signal_id=semantic_id,
    )
    assert expected in top.correlation_group_ids
    assert f"FEATURE_RESULT:{feature.feature_result_id}" in top.source_evidence_ids
    assert f"INPUT_DATA_SHA256:{feature.input_data_sha256}" in top.source_evidence_ids


def test_backward_completeness_counts_only_critical_positive_definitions() -> None:
    bundle = _bundle("heater_stuck_on")
    base_definition = bundle.evidence_definitions[0]
    base_fact = bundle.evidence_facts[0]
    required = base_definition.model_copy(
        update={
            "evidence_definition_id": "ED-REQUIRED-MISSING",
            "role": EvidenceRole.REQUIRED,
            "critical": True,
        }
    )
    exclusion = base_definition.model_copy(
        update={
            "evidence_definition_id": "ED-EXCLUSION-OBSERVED",
            "role": EvidenceRole.EXCLUSION,
            "critical": True,
        }
    )
    required_fact = base_fact.model_copy(
        update={
            "evidence_definition_id": required.evidence_definition_id,
            "role": EvidenceRole.REQUIRED,
            "status": EvidenceFactStatus.MISSING_DATA,
        }
    )
    exclusion_fact = base_fact.model_copy(
        update={
            "evidence_definition_id": exclusion.evidence_definition_id,
            "role": EvidenceRole.EXCLUSION,
            "status": EvidenceFactStatus.NOT_SATISFIED,
        }
    )
    focused = bundle.model_copy(
        update={
            "evidence_definitions": [required, exclusion],
            "evidence_facts": [required_fact, exclusion_fact],
        }
    )

    backward = build_backward_evidence_ranking(focused)
    metadata = backward.candidates[0].metadata

    assert metadata["critical_missing_count"] == 1
    assert metadata["critical_positive_definition_count"] == 1
    assert metadata["completeness"] == 0.0


def test_birrf_uses_rank_contributions_not_raw_score_scale() -> None:
    episode = "EP-TEST"
    data = FaultRanking(
        ranking_id="R-DATA",
        episode_id=episode,
        source=RankingSource.DATA_DRIVEN,
        producer_id="model",
        producer_version="1",
        candidates=[
            RankedFault(fault_id="F-A", rank=1, score=0.51, source=RankingSource.DATA_DRIVEN),
            RankedFault(fault_id="F-B", rank=2, score=0.49, source=RankingSource.DATA_DRIVEN),
        ],
    )
    forward = FaultRanking(
        ranking_id="R-FWD",
        episode_id=episode,
        source=RankingSource.KG_FORWARD,
        producer_id="kg",
        producer_version="1",
        candidates=[
            RankedFault(fault_id="F-B", rank=1, score=1000.0, source=RankingSource.KG_FORWARD),
            RankedFault(fault_id="F-A", rank=2, score=0.01, source=RankingSource.KG_FORWARD),
        ],
    )
    backward = FaultRanking(
        ranking_id="R-BACK",
        episode_id=episode,
        source=RankingSource.KG_BACKWARD,
        producer_id="kg",
        producer_version="1",
        candidates=[
            RankedFault(fault_id="F-B", rank=1, score=0.8, source=RankingSource.KG_BACKWARD),
            RankedFault(fault_id="F-A", rank=2, score=0.7, source=RankingSource.KG_BACKWARD),
        ],
    )
    result = fuse_rankings(
        data,
        forward,
        backward,
        config=BiRRFConfig(
            data_smoothing=10,
            forward_smoothing=10,
            backward_smoothing=10,
            collapse_identical_kg_rankings=False,
            collapse_highly_correlated_kg_rankings=False,
        ),
    )
    assert result.candidates[0].fault_id == "F-B"
    expected = 1 / 12 + 1 / 11 + 1 / 11
    assert result.candidates[0].fused_score == pytest.approx(expected)
    assert result.candidates[0].source_scores["KG_FORWARD"] == 1000.0
    assert "KG_BACKWARD" in result.candidates[0].reciprocal_contributions
    assert result.metadata["collapsed_kg_source"] is None


def test_tied_source_scores_remain_tied_and_do_not_create_a_lexical_top_fault() -> None:
    def tied(source: RankingSource) -> FaultRanking:
        return FaultRanking(
            ranking_id=f"R-{source.value}-TIED",
            episode_id="EP-TIED",
            source=source,
            producer_id="test",
            producer_version="1",
            candidates=[
                RankedFault(fault_id=fault_id, rank=1, score=0.0, source=source)
                for fault_id in ("F-Z", "F-A")
            ],
        )

    data = tied(RankingSource.DATA_DRIVEN)
    forward = tied(RankingSource.KG_FORWARD)
    backward = tied(RankingSource.KG_BACKWARD)
    result = fuse_rankings(data, forward, backward)

    assert result.top_fault_id is None
    assert result.metadata["fused_top_fault_ids"] == ["F-A", "F-Z"]
    assert all(item.final_rank == 1 for item in result.candidates)


def test_prediction_rejects_unregistered_fault_before_fusion() -> None:
    bundle = _bundle("heater_stuck_on")
    prediction = _prediction(bundle, [("UNKNOWN_FAULT", 0.8), ("NORMAL", 0.2)])
    with pytest.raises(ValueError, match="unregistered classes"):
        prediction.to_fault_ranking(
            sorted({item.target_fault_id for item in bundle.evidence_definitions})
        )


def test_fusion_selection_keeps_branch_disagreement_for_investigation() -> None:
    bundle = _bundle("heater_stuck_on")
    registered = sorted({item.target_fault_id for item in bundle.evidence_definitions})
    other = next(item for item in registered if item != "THERMAL_HEATER_STUCK_ON")
    probabilities = [(other, 0.7), ("THERMAL_HEATER_STUCK_ON", 0.2), ("NORMAL", 0.1)]
    probabilities.extend(
        (fault_id, 0.0)
        for fault_id in registered
        if fault_id not in {other, "THERMAL_HEATER_STUCK_ON"}
    )
    prediction = _prediction(bundle, probabilities)
    package = build_candidate_fusion_package(
        bundle=bundle,
        data_ranking=prediction.to_fault_ranking(registered),
        normal_probability=prediction.normal_probability,
        normal_rank=prediction.normal_rank,
    )
    reference = json.loads(
        (BUNDLES / "heater_stuck_on" / "diagnosis/hypothesis_set.json").read_text(encoding="utf-8")
    )["hypotheses"]
    result, snapshot = select_fused_candidates(
        package,
        reference,
        mode="adaptive_forced_count",
        configured_top_k=4,
        min_top_k=2,
        max_top_k=6,
    )
    assert other in result.selected_fault_ids
    assert "THERMAL_HEATER_STUCK_ON" in result.selected_fault_ids
    assert snapshot["selection_source"] == "BIRRF_FUSION"
    assert result.ranked_candidates
    assert all(
        item.source_score_semantics == "RECIPROCAL_RANK_SCORE_NOT_PROBABILITY"
        for item in result.ranked_candidates
    )
    assert snapshot["ranked_candidates"][0]["source_score_semantics"] == "RECIPROCAL_RANK_SCORE_NOT_PROBABILITY"
    assert package.birrf_result.conflict_flags


def test_openai_compact_selector_uses_candidate_fusion_when_present() -> None:
    bundle = _bundle("reaction_wheel_friction_increase")
    registered = sorted({item.target_fault_id for item in bundle.evidence_definitions})
    ordered = [("ADCS_RW_FRICTION_INCREASE", 0.8), ("NORMAL", 0.1)]
    remaining = [fault_id for fault_id in registered if fault_id != "ADCS_RW_FRICTION_INCREASE"]
    ordered.extend((fault_id, 0.1 / len(remaining)) for fault_id in remaining)
    prediction = _prediction(bundle, ordered)
    package = build_candidate_fusion_package(
        bundle=bundle,
        data_ranking=prediction.to_fault_ranking(registered),
        normal_probability=prediction.normal_probability,
        normal_rank=prediction.normal_rank,
    )
    reference = json.loads(
        (BUNDLES / "reaction_wheel_friction_increase" / "diagnosis/hypothesis_set.json").read_text(encoding="utf-8")
    )
    backend = OpenAICompatibleStructuredBackend(
        OpenAICompatibleConfig(
            model="local",
            prompt_profile="evidence_protocol_v059_compact",
            candidate_top_k_mode="adaptive_forced_count",
            candidate_top_k_min=2,
            candidate_top_k_max=6,
        )
    )
    selected = backend._select_compact_fault_ids(
        {
            "registered_fault_ids": registered,
            "reference_hypothesis_set": reference,
            "candidate_fusion": package.model_dump(mode="json"),
        }
    )
    assert "ADCS_RW_FRICTION_INCREASE" in selected
    assert backend._last_candidate_selection["selection_source"] == "BIRRF_FUSION"
    assert backend._last_candidate_selection["fusion_id"] == package.birrf_result.fusion_id


def test_candidate_fusion_package_round_trip_is_strict() -> None:
    bundle = _bundle("nominal")
    registered = sorted({item.target_fault_id for item in bundle.evidence_definitions})
    probability = 0.2 / len(registered)
    prediction = _prediction(
        bundle,
        [("NORMAL", 0.8), *[(fault_id, probability) for fault_id in registered]],
    )
    package = build_candidate_fusion_package(
        bundle=bundle,
        data_ranking=prediction.to_fault_ranking(registered),
        normal_probability=prediction.normal_probability,
        normal_rank=prediction.normal_rank,
    )
    restored = CandidateFusionPackage.model_validate_json(package.model_dump_json())
    assert restored.birrf_result.normal_rank == 1
    assert restored.metadata["formal_decision_remains_deterministic"] is True


def test_online_workflow_persists_optional_candidate_fusion(tmp_path: Path, monkeypatch) -> None:
    from astrograph.diagnosis import KGEvidenceReference
    from astrograph.evidence import EvidenceRole
    from astrograph.knowledge import GraphEvidenceAssociation
    from astrograph.orchestration.models import (
        ExecutionPolicy,
        PermissionScope,
        TaskEnvelope,
        WorkflowContext,
    )
    from astrograph.orchestration.whole_spacecraft_online_workflow import (
        LiveKGEvidenceQueryResult,
        WholeSpacecraftOnlineDiagnosisWorkflow,
    )
    from astrograph.simulation import SimulationRunCompleted, WholeSpacecraftRuntimeEvidenceBuilder

    scene_root = ROOT / "tests/fixtures/spacecraft_scenarios"
    protocol_root = ROOT / "protocols/phm_project_a"
    event = SimulationRunCompleted(
        event_id="SIM-EVENT-V0514-TEST",
        run_id="heater_stuck_on_run_0001",
        run_dir=str(scene_root / "heater_stuck_on"),
        nominal_run_dir=str(scene_root / "nominal"),
    )
    preview, _ = WholeSpacecraftRuntimeEvidenceBuilder(protocol_root).build(event)
    registered = sorted({item.target_fault_id for item in preview.evidence_definitions})
    remaining = [item for item in registered if item != "THERMAL_HEATER_STUCK_ON"]
    ordered = [("THERMAL_HEATER_STUCK_ON", 0.8), ("NORMAL", 0.1)]
    ordered.extend((fault_id, 0.1 / len(remaining)) for fault_id in remaining)
    prediction = _prediction(preview, ordered)

    def fake_kg_evidence(*, config, bundle, compiler=None):
        from astrograph.policy_profiles import GovernedGraphQueryCompiler

        assert isinstance(compiler, GovernedGraphQueryCompiler)
        role_relation = {
            EvidenceRole.REQUIRED: "REQUIRES_EVIDENCE",
            EvidenceRole.SUPPORTING: "SUPPORTED_BY",
            EvidenceRole.COUNTER: "CONTRADICTED_BY",
            EvidenceRole.EXCLUSION: "EXCLUDES",
        }
        associations = [
            GraphEvidenceAssociation(
                fault_id=item.target_fault_id,
                evidence_definition_id=item.evidence_definition_id,
                relation_type=role_relation[item.role],
                knowledge_version=config.knowledge_version,
                source_query_plan_id="GQ-FORWARD",
            )
            for item in bundle.evidence_definitions
        ]
        return LiveKGEvidenceQueryResult(
            kg_evidence=KGEvidenceReference(
                release_id=config.release_id,
                knowledge_version=config.knowledge_version,
                nodes_sha256=config.nodes_sha256,
                edges_sha256=config.edges_sha256,
                query_ids=["GQ-FORWARD", "GQ-BACKWARD"],
                query_sha256_values=["a" * 64, "b" * 64],
                read_user=config.read_user,
                row_count=len(associations) * 2,
            ),
            forward_associations=associations,
            backward_associations=[
                item.model_copy(update={"source_query_plan_id": "GQ-BACKWARD"}) for item in associations
            ],
        )

    monkeypatch.setattr(WholeSpacecraftOnlineDiagnosisWorkflow, "_query_live_kg_evidence", staticmethod(fake_kg_evidence))
    permissions = [
        PermissionScope.READ_DATA,
        PermissionScope.READ_KNOWLEDGE,
        PermissionScope.RUN_DIAGNOSIS,
        PermissionScope.WRITE_AUDIT,
        PermissionScope.WRITE_ARTIFACTS,
    ]
    from astrograph.policy_profiles import RuntimePolicyRegistry

    workflow = WholeSpacecraftOnlineDiagnosisWorkflow(
        runtime_policy_registry=RuntimePolicyRegistry(
            ROOT / "configs/runtime_policies/registry.json"
        )
    )
    spatiotemporal_catalog = tmp_path / "heater_spatiotemporal_templates.json"
    spatiotemporal_catalog.write_text(json.dumps({
        "schema_version": "astrograph-spatiotemporal-fault-template-catalog.v1",
        "catalog_version": "test-heater-spatiotemporal-v1",
        "maximum_temporal_lag_s": 5.0,
        "bindings": [
            {
                "anomaly_id": "ANOM_HEATER_UNCOMMANDED_ON",
                "component_instance_id": "component:thermal:heater:payload",
                "component_archetype_id": "converter.power",
                "subsystem_id": "subsystem:thermal",
                "effect_pattern_ids": ["effect:command-response-residual-growth"],
            },
            {
                "anomaly_id": "ANOM_HEATER_CURRENT_PRESENT",
                "component_instance_id": "component:thermal:heater:payload",
                "component_archetype_id": "converter.power",
                "subsystem_id": "subsystem:thermal",
                "effect_pattern_ids": ["effect:uncommanded-energy-input"],
            },
        ],
        "topology_edges": [],
        "templates": [{
            "template_id": "template:test:heater-stuck-on",
            "fault_id": "THERMAL_HEATER_STUCK_ON",
            "node_requirements": [
                {"role": "command_mismatch", "anomaly_ids": ["ANOM_HEATER_UNCOMMANDED_ON"]},
                {"role": "current_response", "anomaly_ids": ["ANOM_HEATER_CURRENT_PRESENT"]},
            ],
            "edge_requirements": [{
                "edge_id": "edge:test:heater-response",
                "source_role": "command_mismatch",
                "target_role": "current_response",
                "allowed_relations": ["OVERLAPS"],
                "maximum_lag_s": 5.0,
            }],
            "minimum_score": 1.0,
            "review_status": "HUMAN_REVIEWED",
        }],
        "claim_boundary": "Test-only E0 temporal association; not causal proof.",
    }), encoding="utf-8")
    spatiotemporal_episodes = [
        {
            "episode_id": "episode-heater-command-0001",
            "run_id": "heater_stuck_on_run_0001",
            "case_id": "heater_stuck_on",
            "anomaly_id": "ANOM_HEATER_UNCOMMANDED_ON",
            "name": "Heater uncommanded on",
            "start_s": 100.0,
            "end_s": 120.0,
            "peak_confidence": 0.9,
        },
        {
            "episode_id": "episode-heater-current-0001",
            "run_id": "heater_stuck_on_run_0001",
            "case_id": "heater_stuck_on",
            "anomaly_id": "ANOM_HEATER_CURRENT_PRESENT",
            "name": "Heater current present",
            "start_s": 101.0,
            "end_s": 120.0,
            "peak_confidence": 0.9,
        },
    ]
    envelope = TaskEnvelope(
        task_id="V0514-WORKFLOW-TEST",
        task_type="simulation.online_diagnosis",
        input_payload={
            "simulation_event": event.model_dump(mode="json"),
            "protocol_root": str(protocol_root),
            "runtime_evidence_builder_id": "legacy.whole_spacecraft.v4",
            "artifact_output_dir": str(tmp_path / "artifacts"),
            "reasoner_mode": "rule",
            "graph_query_policy_profile_id": "graph-query.read-only.v1",
            "fusion_policy_profile_id": "birrf.engineering-default.v1",
            "data_driven_prediction": prediction.model_dump(mode="json"),
            "spatiotemporal_episodes": spatiotemporal_episodes,
            "spatiotemporal_template_catalog_path": str(spatiotemporal_catalog),
            "spatiotemporal_rerank_weight": 0.2,
            "pre_reasoning_retrieval_request": {
                "query": {
                    "query_id": "query:heater-stuck-on",
                    "episode_id": preview.episode.episode_id,
                    "text": "heater stuck on uncommanded current",
                    "required_concept_ids": ["concept:uncommanded-current"],
                },
                "chunks": [{
                    "chunk_id": "chunk:heater-manual",
                    "source_id": "source:heater-manual",
                    "source_uri": "local://manual/heater",
                    "source_sha256": "e" * 64,
                    "chunk_sha256": hashlib.sha256(
                        b"Heater stuck on manifests as uncommanded current."
                    ).hexdigest(),
                    "knowledge_version": "test-retrieval-v1",
                    "text": "Heater stuck on manifests as uncommanded current.",
                    "concept_ids": ["concept:heater-stuck-on", "concept:uncommanded-current"],
                }],
                "concepts": [
                    {
                        "concept_id": "concept:heater-stuck-on",
                        "labels": ["heater stuck on"],
                        "aliases": ["加热器卡开"],
                        "source_id": "source:ontology",
                        "source_sha256": "f" * 64,
                        "knowledge_version": "test-ontology-v1",
                    },
                    {
                        "concept_id": "concept:uncommanded-current",
                        "labels": ["uncommanded current"],
                        "aliases": ["非指令电流"],
                        "source_id": "source:ontology",
                        "source_sha256": "f" * 64,
                        "knowledge_version": "test-ontology-v1",
                    },
                ],
                "relations": [{
                    "relation_id": "relation:heater-manifests-current",
                    "source_concept_id": "concept:heater-stuck-on",
                    "relation_type": "MANIFESTS_AS",
                    "target_concept_id": "concept:uncommanded-current",
                    "source_id": "source:ontology",
                    "source_sha256": "f" * 64,
                    "knowledge_version": "test-ontology-v1",
                }],
            },
            "kg_evidence_binding": {
                "uri": "bolt://127.0.0.1:7687",
                "read_user": "astrograph_reader",
                "read_password_env": "ASTROGRAPH_NEO4J_READ_PASSWORD",
                "release_id": "KG-GOLD-ONLINE-V058",
                "knowledge_version": "astrograph-kg-gold-online-v058",
                "nodes_sha256": "c" * 64,
                "edges_sha256": "d" * 64,
            },
        },
        execution_policy=ExecutionPolicy(deterministic_only=True, allow_llm=False),
        permission_scope=permissions,
    )
    result = workflow.invoke(
        envelope,
        WorkflowContext(
            workflow_id=workflow.workflow_id,
            run_id="V0514-WORKFLOW-TEST",
            permissions=permissions,
        ),
    )
    assert result.status.value == "SUCCEEDED", result.error
    assert {
        item["profile_id"] for item in result.output_payload["runtime_policy_receipts"]
    } == {"graph-query.read-only.v1", "birrf.engineering-default.v1"}
    assert result.output_payload["candidate_fusion"]["birrf_result"]["top_fault_id"] == "THERMAL_HEATER_STUCK_ON"
    assert result.output_payload["candidate_fusion"]["metadata"]["graph_query_backed"] is True
    assert result.output_payload["candidate_fusion"]["metadata"]["kg_evidence_release_id"] == "KG-GOLD-ONLINE-V058"
    assert result.output_payload["candidate_fusion"]["metadata"]["spatiotemporal_rerank_enabled"] is True
    assert result.output_payload["candidate_fusion"]["metadata"]["spatiotemporal_branch_count"] == 0
    assert result.output_payload["component_spatiotemporal_fault_graph"]["nodes"]
    assert result.output_payload["pre_reasoning_retrieval"]["status"] == "SUFFICIENT"
    assert result.output_payload["pre_reasoning_retrieval"]["active_graph_modified"] is False
    assert result.output_payload["paperseek_fallback_decision"]["status"] == "NOT_TRIGGERED"
    assert result.output_payload["spatiotemporal_template_matches"][0]["eligible_for_rerank"] is True
    assert (tmp_path / "artifacts/diagnosis/component_spatiotemporal_fault_graph.json").is_file()
    assert (tmp_path / "artifacts/diagnosis/spatiotemporal_template_matches.json").is_file()
    assert (
        result.output_payload["candidate_fusion"]["forward_ranking"]["producer_id"]
        == "neo4j-read-only-forward-evidence-ranking"
    )
    assert (
        result.output_payload["candidate_fusion"]["backward_ranking"]["producer_id"]
        == "neo4j-read-only-backward-evidence-ranking"
    )
    policy = result.output_payload["diagnosis_policy_decision"]
    assert policy["status"] == "CONDITIONAL_DIAGNOSIS"
    assert "MISSING_KG_EVIDENCE" not in policy["blocking_reasons"]
    assert "UPSTREAM_FORMAL_RELEASE_NOT_ALLOWED" in policy["blocking_reasons"]
    assert result.output_payload["diagnosis_evidence_bundle"]["kg_evidence"]["release_id"] == "KG-GOLD-ONLINE-V058"
    fusion_path = tmp_path / "artifacts/diagnosis/candidate_fusion.json"
    assert fusion_path.is_file()
    assert (tmp_path / "artifacts/diagnosis/diagnosis_evidence_bundle.json").is_file()
    assert (tmp_path / "artifacts/diagnosis/diagnosis_policy_decision.json").is_file()
    assert (tmp_path / "artifacts/diagnosis/pre_reasoning_retrieval.json").is_file()
    assert (tmp_path / "artifacts/diagnosis/paperseek_fallback_decision.json").is_file()
    manifest = json.loads((tmp_path / "artifacts/manifest.json").read_text(encoding="utf-8"))
    assert "candidate_fusion" in manifest["files"]
    assert "diagnosis_evidence_bundle" in manifest["files"]
    assert "diagnosis_policy_decision" in manifest["files"]
    assert "pre_reasoning_retrieval" in manifest["files"]
    assert "paperseek_fallback_decision" in manifest["files"]
    assert manifest["schema_version"] == "astrograph-online-artifact-manifest.v12"
    assert "component_spatiotemporal_fault_graph" in manifest["files"]
    assert "spatiotemporal_template_matches" in manifest["files"]


def test_kg_fault_universe_guard_suppresses_sparse_knowledge_branches() -> None:
    episode = "EP-COVERAGE-GUARD"
    data = FaultRanking(
        ranking_id="R-DATA-COVERAGE",
        episode_id=episode,
        source=RankingSource.DATA_DRIVEN,
        producer_id="model",
        producer_version="1",
        candidates=[
            RankedFault(fault_id="F-A", rank=1, score=0.9, source=RankingSource.DATA_DRIVEN),
            RankedFault(fault_id="F-B", rank=2, score=0.1, source=RankingSource.DATA_DRIVEN),
            RankedFault(fault_id="F-C", rank=3, score=0.0, source=RankingSource.DATA_DRIVEN),
        ],
    )
    forward = FaultRanking(
        ranking_id="R-FWD-COVERAGE",
        episode_id=episode,
        source=RankingSource.KG_FORWARD,
        producer_id="kg",
        producer_version="1",
        candidates=[
            RankedFault(fault_id="F-B", rank=1, score=1.0, source=RankingSource.KG_FORWARD),
        ],
        kg_fault_coverage=_kg_coverage("F-B"),
    )
    backward = FaultRanking(
        ranking_id="R-BACK-COVERAGE",
        episode_id=episode,
        source=RankingSource.KG_BACKWARD,
        producer_id="kg",
        producer_version="1",
        candidates=[
            RankedFault(fault_id="F-B", rank=1, score=1.0, source=RankingSource.KG_BACKWARD),
        ],
        kg_fault_coverage=_kg_coverage("F-B"),
    )

    result = fuse_rankings(
        data,
        forward,
        backward,
        config=BiRRFConfig(
            enforce_kg_fault_universe_coverage=True,
            expected_fault_universe=("F-A", "F-B", "F-C"),
            minimum_kg_fault_universe_coverage=1.0,
        ),
    )

    assert result.top_fault_id == "F-A"
    assert result.metadata["effective_branch_weights"] == {
        "DATA_DRIVEN": 1.0,
        "KG_FORWARD": 0.0,
        "KG_BACKWARD": 0.0,
    }
    assert result.metadata["kg_fault_universe_coverage"] == {
        "KG_FORWARD": pytest.approx(1 / 3),
        "KG_BACKWARD": pytest.approx(1 / 3),
    }
    assert result.metadata["suppressed_kg_sources"] == ["KG_BACKWARD", "KG_FORWARD"]
    assert any(
        warning.startswith("KG_BRANCH_SUPPRESSED_INCOMPLETE_FAULT_COVERAGE=KG_FORWARD")
        for warning in result.quality_warnings
    )


def test_kg_fault_universe_guard_retains_complete_knowledge_branch() -> None:
    episode = "EP-COVERAGE-COMPLETE"
    fault_ids = ("F-A", "F-B")

    def ranking(source: RankingSource, order: tuple[str, ...]) -> FaultRanking:
        return FaultRanking(
            ranking_id=f"R-{source.value}-COMPLETE",
            episode_id=episode,
            source=source,
            producer_id="test",
            producer_version="1",
            candidates=[
                RankedFault(
                    fault_id=fault_id,
                    rank=index,
                    score=float(len(order) - index + 1),
                    source=source,
                )
                for index, fault_id in enumerate(order, start=1)
            ],
            kg_fault_coverage=(
                None
                if source == RankingSource.DATA_DRIVEN
                else _kg_coverage(*fault_ids)
            ),
        )

    result = fuse_rankings(
        ranking(RankingSource.DATA_DRIVEN, ("F-A", "F-B")),
        ranking(RankingSource.KG_FORWARD, ("F-B", "F-A")),
        ranking(RankingSource.KG_BACKWARD, ("F-B", "F-A")),
        config=BiRRFConfig(
            enforce_kg_fault_universe_coverage=True,
            expected_fault_universe=fault_ids,
            minimum_kg_fault_universe_coverage=1.0,
            collapse_identical_kg_rankings=False,
        ),
    )

    assert result.metadata["kg_fault_universe_coverage"] == {
        "KG_FORWARD": 1.0,
        "KG_BACKWARD": 1.0,
    }
    assert result.metadata["suppressed_kg_sources"] == []
    assert result.metadata["effective_branch_weights"]["KG_FORWARD"] == 1.0
    assert result.metadata["effective_branch_weights"]["KG_BACKWARD"] == 1.0

def test_open_set_review_disables_compact_candidate_filtering():
    bundle = _bundle("reaction_wheel_friction_increase")
    registered = sorted({item.target_fault_id for item in bundle.evidence_definitions})
    ordered = [("ADCS_RW_FRICTION_INCREASE", 0.8), ("NORMAL", 0.1)]
    remaining = [fault_id for fault_id in registered if fault_id != "ADCS_RW_FRICTION_INCREASE"]
    ordered.extend((fault_id, 0.1 / len(remaining)) for fault_id in remaining)
    prediction = _prediction(bundle, ordered)
    package = build_candidate_fusion_package(
        bundle=bundle,
        data_ranking=prediction.to_fault_ranking(registered),
        normal_probability=prediction.normal_probability,
        normal_rank=prediction.normal_rank,
    )
    package = package.model_copy(update={
        "birrf_result": package.birrf_result.model_copy(update={
            "open_set_actions": ["DATA_BRANCH_SUPPRESSED_BY_OPEN_SET_POLICY"],
        })
    })
    reference = json.loads(
        (BUNDLES / "reaction_wheel_friction_increase" / "diagnosis/hypothesis_set.json").read_text(encoding="utf-8")
    )["hypotheses"]
    result, snapshot = select_fused_candidates(
        package, reference, mode="fixed", configured_top_k=4, min_top_k=2, max_top_k=6
    )
    assert result.compact_reasoning_allowed is False
    assert result.candidate_set_status == "OPEN_SET_REVIEW_REQUIRED"
    assert result.escape_action == "DISABLE_COMPACT_REASONING"
    assert "OPEN_SET_REQUIRES_UNTRUNCATED_CONTEXT" in result.escape_reason_codes
    assert snapshot["compact_reasoning_allowed"] is False

