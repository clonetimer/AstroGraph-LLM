from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from astrograph.data_driven import (
    BenchmarkPredictionRecord,
    BenchmarkStatus,
    CampaignPlanStatus,
    CampaignScenario,
    CampaignSplitCounts,
    CampaignVariant,
    ClassProbability,
    CredibilityStatus,
    DataDrivenPrediction,
    IndependentRunCampaignSpec,
    IndependentRunRecord,
    RunProvenanceType,
    SplitName,
    audit_independent_runs,
    build_independent_run_campaign_plan,
    build_inventory_from_campaign_plan,
    evaluate_model_benchmark,
)
from astrograph.evidence import EpisodeEvidenceBundle, EvidenceRole
from astrograph.fusion import (
    build_backward_evidence_ranking,
    build_backward_evidence_ranking_from_associations,
    build_forward_graph_ranking,
    build_forward_graph_ranking_from_associations,
)
from astrograph.investigation import GraphQueryPlan
from astrograph.knowledge import (
    CsvEvidenceGraphReferenceExecutor,
    GraphEvidenceAssociation,
    GraphQueryKind,
    ReadOnlyCypherCompiler,
    ReadOnlyGraphQueryExecutor,
    associations_from_query_result,
    build_backward_fault_query_plan,
    build_forward_evidence_query_plan,
)


ROOT = Path(__file__).resolve().parents[1]
BUNDLES = ROOT / "tests/fixtures/evidence"


def _bundle(case_id: str) -> EpisodeEvidenceBundle:
    return EpisodeEvidenceBundle.model_validate_json(
        (BUNDLES / case_id / "diagnosis/episode_evidence_bundle.json").read_text(encoding="utf-8")
    )


def _campaign_spec() -> IndependentRunCampaignSpec:
    return IndependentRunCampaignSpec(
        campaign_id="CAMP-V0516",
        scenarios=[CampaignScenario(scenario_id="fault_a", class_ids=["F-A"], object_id="OBJ-A")],
        seeds=[11, 22, 33],
        variants=[
            CampaignVariant(
                variant_id="OPS-A",
                cli_args=["--noise-profile", "OPS-A"],
                initial_state_id="INIT-A",
                operating_condition_id="OC-A",
                fault_severity_id="SEV-A",
                satellite_instance_id="SAT-A",
            )
        ],
        split_counts=CampaignSplitCounts(train=1, validation=1, test=1),
    )


def _prediction(
    *,
    sample: str,
    run_id: str,
    truth: str,
    probabilities: list[tuple[str, float]],
    rejected: bool = False,
) -> BenchmarkPredictionRecord:
    classes = [
        ClassProbability(class_id=class_id, probability=prob, rank=index)
        for index, (class_id, prob) in enumerate(probabilities, start=1)
    ]
    return BenchmarkPredictionRecord(
        sample_id=sample,
        run_id=run_id,
        truth_class_id=truth,
        open_set_rejected=rejected,
        latency_ms=5.0,
        prediction=DataDrivenPrediction(
            prediction_id=f"P-{sample}",
            episode_id=f"EP-{sample}",
            model_id="MODEL-A",
            model_version="1",
            classes=classes,
            feature_schema_id="FEATURE-V1",
            input_data_sha256="a" * 64,
        ),
    )


def test_campaign_plan_assigns_whole_runs_to_disjoint_splits() -> None:
    plan = build_independent_run_campaign_plan(_campaign_spec())
    assert plan.status == CampaignPlanStatus.PASS
    assert len(plan.runs) == 6  # three nominal + three fault runs
    assert len({item.run_id for item in plan.runs}) == 6
    fault_runs = [item for item in plan.runs if item.scenario_id == "fault_a"]
    assert {item.split for item in fault_runs} == {
        SplitName.TRAIN,
        SplitName.VALIDATION,
        SplitName.TEST,
    }
    nominal_ids = {item.run_id for item in plan.runs if item.scenario_id == "nominal"}
    assert {item.paired_nominal_run_id for item in fault_runs} == nominal_ids
    assert all("--output" in item.command_argv_template for item in plan.runs)


def test_campaign_plan_blocks_insufficient_factor_combinations() -> None:
    spec = _campaign_spec().model_copy(update={"seeds": [11, 22]})
    plan = build_independent_run_campaign_plan(spec)
    assert plan.status == CampaignPlanStatus.BLOCKED
    assert "INSUFFICIENT_FACTOR_COMBINATIONS:2<3" in plan.blocking_reasons


def test_campaign_variant_cannot_override_reserved_runtime_arguments() -> None:
    with pytest.raises(ValueError, match="reserved runtime argument"):
        CampaignVariant(variant_id="BAD", cli_args=["--output", "/tmp/escape"])


def test_inventory_builder_hashes_native_runs_and_strict_audit_passes(tmp_path: Path) -> None:
    plan = build_independent_run_campaign_plan(_campaign_spec())
    for index, run in enumerate(plan.runs):
        scenario_dir = tmp_path / run.expected_scenario_relative_dir
        scenario_dir.mkdir(parents=True)
        (scenario_dir / "telemetry.csv").write_text(
            f"time_s,value\n0,{index}\n1,{index + 1}\n", encoding="utf-8"
        )
        (scenario_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "scenario": run.scenario_id,
                    "seed": run.seed,
                    "duration_s": run.duration_s,
                    "events": [],
                    "runtime": "test-runtime",
                }
            ),
            encoding="utf-8",
        )
    records, missing = build_inventory_from_campaign_plan(plan, output_root=tmp_path)
    assert not missing
    assert all(item.content_sha256 and item.configuration_sha256 for item in records)
    assert all(not Path(str(item.metadata["scenario_dir"])).is_absolute() for item in records)
    assert all(item.metadata["scenario_dir_base"] == "CAMPAIGN_OUTPUT_ROOT" for item in records)
    audit = audit_independent_runs(records, strict_provenance=True)
    assert audit.status == CredibilityStatus.PASS
    assert audit.eligible_independent_run_count == len(records)


def test_strict_audit_rejects_derivatives_duplicates_and_group_leakage() -> None:
    records = [
        IndependentRunRecord(
            run_id="A-TRAIN",
            class_ids=["F-A"],
            split=SplitName.TRAIN,
            provenance_type=RunProvenanceType.SIMULATION_NATIVE,
            independence_group_id="G-SHARED",
            content_sha256="1" * 64,
        ),
        IndependentRunRecord(
            run_id="A-VAL",
            class_ids=["F-A"],
            split=SplitName.VALIDATION,
            provenance_type=RunProvenanceType.AUGMENTED_DERIVATIVE,
            parent_run_id="A-TRAIN",
            independence_group_id="G-SHARED",
            content_sha256="1" * 64,
        ),
        IndependentRunRecord(
            run_id="A-TEST",
            class_ids=["F-A"],
            split=SplitName.TEST,
            provenance_type=RunProvenanceType.SIMULATION_NATIVE,
            independence_group_id="G-TEST",
            content_sha256="2" * 64,
        ),
    ]
    audit = audit_independent_runs(records, strict_provenance=True)
    assert audit.status == CredibilityStatus.BLOCKED
    assert "EXACT_TELEMETRY_CONTENT_REUSED_ACROSS_RUN_IDS" in audit.blocking_reasons
    assert "INDEPENDENCE_GROUP_CROSSES_DATASET_SPLITS" in audit.blocking_reasons
    assert any(reason.startswith("DERIVATIVE_RUN_CANNOT_COUNT_AS_INDEPENDENT") for reason in audit.blocking_reasons)


def test_benchmark_refuses_metrics_until_run_audit_passes() -> None:
    inventory = [
        IndependentRunRecord(run_id="R1", class_ids=["F-A"], split=SplitName.TEST)
    ]
    audit = audit_independent_runs(inventory)
    report = evaluate_model_benchmark(
        [_prediction(sample="1", run_id="R1", truth="F-A", probabilities=[("F-A", 0.9), ("NORMAL", 0.1)])],
        inventory,
        audit,
    )
    assert report.status == BenchmarkStatus.BLOCKED
    assert report.metrics is None
    assert "INDEPENDENT_RUN_AUDIT_NOT_PASS" in report.blocking_reasons


def test_benchmark_reports_known_and_unknown_metrics_only_after_pass() -> None:
    inventory = []
    for class_id in ("F-A", "NORMAL"):
        for split in (SplitName.TRAIN, SplitName.VALIDATION, SplitName.TEST):
            inventory.append(
                IndependentRunRecord(
                    run_id=f"{class_id}-{split.value}",
                    class_ids=[class_id],
                    split=split,
                    provenance_type=RunProvenanceType.SIMULATION_NATIVE,
                    satellite_instance_id="SAT",
                    operating_condition_id="OC",
                    configuration_sha256=("1" if class_id == "F-A" else "2") * 64,
                    content_sha256=(
                        f"{len(inventory) + 1:064x}"[-64:]
                    ),
                    independence_group_id=f"G-{class_id}-{split.value}",
                )
            )
    audit = audit_independent_runs(inventory, strict_provenance=True)
    assert audit.status == CredibilityStatus.PASS
    records = [
        _prediction(
            sample="KNOWN",
            run_id="F-A-TEST",
            truth="F-A",
            probabilities=[("F-A", 0.8), ("NORMAL", 0.2)],
        ),
        _prediction(
            sample="NORMAL",
            run_id="NORMAL-TEST",
            truth="NORMAL",
            probabilities=[("NORMAL", 0.9), ("F-A", 0.1)],
        ),
        _prediction(
            sample="UNKNOWN",
            run_id="NORMAL-TEST",
            truth="UNKNOWN-FAULT",
            probabilities=[("F-A", 0.5), ("NORMAL", 0.5)],
            rejected=True,
        ),
    ]
    report = evaluate_model_benchmark(records, inventory, audit, known_class_ids={"F-A", "NORMAL"})
    assert report.status == BenchmarkStatus.PASS
    assert report.metrics is not None
    assert report.metrics.accuracy == pytest.approx(1.0)
    assert report.metrics.unknown_false_accept_rate == pytest.approx(0.0)


def test_read_only_compiler_parameterizes_untrusted_entity_ids() -> None:
    malicious = "ED-1') MATCH (n) DETACH DELETE n //"
    plan = GraphQueryPlan(
        query_plan_id="GQ-MALICIOUS-ID",
        purpose="test",
        start_entity_ids=[malicious],
        allowed_relation_types=["REQUIRES_EVIDENCE"],
        target_entity_types=["FaultMode"],
        max_depth=1,
        max_results=10,
    )
    compiled = ReadOnlyCypherCompiler().compile(plan)
    assert compiled.query_kind == GraphQueryKind.FORWARD_EVIDENCE_TO_FAULT
    assert malicious not in compiled.cypher
    assert compiled.parameters["start_entity_ids"] == [malicious]
    assert "DELETE" not in compiled.cypher.upper()


def test_read_only_compiler_rejects_unknown_relation_and_excessive_depth() -> None:
    compiler = ReadOnlyCypherCompiler()
    with pytest.raises(ValueError, match="relation types are not allowed"):
        compiler.compile(
            GraphQueryPlan(
                query_plan_id="GQ-BAD-REL",
                purpose="test",
                start_entity_ids=["F-A"],
                allowed_relation_types=["OWNS_DATABASE"],
                target_entity_types=["EvidenceDefinition"],
                max_depth=1,
            )
        )
    with pytest.raises(ValueError, match="max_depth exceeds policy"):
        compiler.compile(
            GraphQueryPlan(
                query_plan_id="GQ-BAD-DEPTH",
                purpose="test",
                start_entity_ids=["F-A"],
                allowed_relation_types=["REQUIRES_EVIDENCE"],
                target_entity_types=["EvidenceDefinition"],
                max_depth=6,
            )
        )


class _FakeSession:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.query = ""
        self.parameters: dict[str, Any] = {}

    def __enter__(self) -> "_FakeSession":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    def run(self, query: str, **parameters: Any) -> list[dict[str, Any]]:
        self.query = query
        self.parameters = parameters
        return self.rows


class _FakeDriver:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.fake_session = _FakeSession(rows)
        self.access_mode = None

    def session(self, **kwargs: Any) -> _FakeSession:
        self.access_mode = kwargs.get("default_access_mode")
        return self.fake_session


def test_read_only_executor_validates_rows_and_associations() -> None:
    plan = build_backward_fault_query_plan(["F-A"])
    compiled = ReadOnlyCypherCompiler().compile(plan)
    driver = _FakeDriver(
        [
            {
                "start_entity_id": "F-A",
                "target_entity_id": "ED-A",
                "start_entity_types": ["FaultMode"],
                "target_entity_types": ["EvidenceDefinition"],
                "relation_type": "REQUIRES_EVIDENCE",
                "path_entity_ids": ["F-A", "ED-A"],
                "path_relation_types": ["REQUIRES_EVIDENCE"],
                "relationship_properties": {"knowledge_version": "KG-V1"},
                "target_properties": {"id": "ED-A"},
            }
        ]
    )
    result = ReadOnlyGraphQueryExecutor(driver).execute(compiled, knowledge_version="KG-V1")
    assert driver.access_mode == "READ"
    assert result.row_count == 1
    associations = associations_from_query_result(result)
    assert associations[0].fault_id == "F-A"
    assert associations[0].evidence_definition_id == "ED-A"


def test_graph_query_associations_reproduce_catalog_rankings() -> None:
    bundle = _bundle("reaction_wheel_friction_increase")
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
            source_query_plan_id="GQ-COMPLETE",
        )
        for item in bundle.evidence_definitions
    ]
    expected_forward = build_forward_graph_ranking(bundle)
    expected_backward = build_backward_evidence_ranking(bundle)
    actual_forward = build_forward_graph_ranking_from_associations(bundle, associations)
    actual_backward = build_backward_evidence_ranking_from_associations(bundle, associations)
    assert [item.fault_id for item in actual_forward.candidates] == [
        item.fault_id for item in expected_forward.candidates
    ]
    assert [item.score for item in actual_backward.candidates] == pytest.approx(
        [item.score for item in expected_backward.candidates]
    )
    assert actual_forward.metadata["graph_query_backed"] is True
    assert actual_forward.producer_version == (
        "0.7.1.post24+b006.wip.kgcoverage.graph.v1"
    )
    assert actual_backward.producer_version == (
        "0.7.1.post24+b006.wip.kgcoverage.graph.v1"
    )
    assert actual_forward.kg_fault_coverage is not None
    assert actual_backward.kg_fault_coverage is not None


def test_graph_association_mismatch_is_fail_closed() -> None:
    bundle = _bundle("heater_stuck_on")
    definition = bundle.evidence_definitions[0]
    bad = GraphEvidenceAssociation(
        fault_id="SOME_OTHER_FAULT",
        evidence_definition_id=definition.evidence_definition_id,
        relation_type="REQUIRES_EVIDENCE",
        source_query_plan_id="GQ-BAD",
    )
    with pytest.raises(ValueError, match="fault/evidence mismatch"):
        build_forward_graph_ranking_from_associations(bundle, [bad])


def test_forward_and_backward_plan_helpers_are_explicit_and_read_only() -> None:
    forward = build_forward_evidence_query_plan(["ED-A"])
    backward = build_backward_fault_query_plan(["F-A"])
    compiler = ReadOnlyCypherCompiler()
    assert compiler.compile(forward).query_kind == GraphQueryKind.FORWARD_EVIDENCE_TO_FAULT
    assert compiler.compile(backward).query_kind == GraphQueryKind.BACKWARD_FAULT_TO_EVIDENCE


def test_csv_graph_reference_executes_same_direct_query_contract() -> None:
    executor = CsvEvidenceGraphReferenceExecutor(
        ROOT / "benchmarks/knowledge_graph/online_extended/nodes.csv",
        ROOT / "benchmarks/knowledge_graph/online_extended/edges.csv",
    )
    bundle = _bundle("heater_stuck_on")
    evidence_ids = [item.evidence_definition_id for item in bundle.evidence_definitions]
    forward = executor.execute(build_forward_evidence_query_plan(evidence_ids))
    backward = executor.execute(
        build_backward_fault_query_plan(sorted({item.target_fault_id for item in bundle.evidence_definitions}))
    )
    assert forward.row_count > 0
    assert backward.row_count >= forward.row_count
    associations = associations_from_query_result(backward)
    assert any(item.fault_id == "THERMAL_HEATER_STUCK_ON" for item in associations)
    assert "not a Neo4j" in backward.claim_boundary


def test_evidence_graph_admin_loader_validates_release_allowlists(tmp_path: Path) -> None:
    from astrograph.knowledge.evidence_graph_admin import load_evidence_graph_csv

    nodes, edges = load_evidence_graph_csv(
        ROOT / "benchmarks/knowledge_graph/online_extended/nodes.csv",
        ROOT / "benchmarks/knowledge_graph/online_extended/edges.csv",
    )
    assert len(nodes) > 100
    assert len(edges) > 400

    bad_nodes = tmp_path / "nodes.csv"
    bad_edges = tmp_path / "edges.csv"
    bad_nodes.write_text(
        "node_id,node_type,name,properties_json\nN1,MaliciousLabel,bad,{}\n",
        encoding="utf-8",
    )
    bad_edges.write_text(
        "edge_id,source_id,relation,target_id,properties_json\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="node label is not allowed"):
        load_evidence_graph_csv(bad_nodes, bad_edges)


def test_benchmark_rejects_invalid_top_k_configuration() -> None:
    inventory = [IndependentRunRecord(run_id="R1", class_ids=["F-A"], split=SplitName.TEST)]
    audit = audit_independent_runs(inventory)
    with pytest.raises(ValueError, match="top_k_values"):
        evaluate_model_benchmark([], inventory, audit, top_k_values=(0,))
