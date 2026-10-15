from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path

from astrograph.audit import JsonlAuditStore
from astrograph.agents.knowledge_curation import (
    KnowledgeExtractionProposal,
    VllmKnowledgeExtractionBackend,
)
from astrograph.config import DiagnosisPolicy
from astrograph.data import (
    audit_round6_dataset,
    load_case_anomalies,
    validate_dataset,
    write_report,
    write_trusted_audit_outputs,
)
from astrograph.data_sources import (
    DataSourceType,
    DatasetSourceConfig,
    ProbeOptions,
    build_data_source,
    probe_data_source,
    validate_data_source,
)
from astrograph.data_driven import (
    BenchmarkPredictionRecord,
    DataDrivenModelManifest,
    IndependentRunAudit,
    IndependentRunCampaignPlan,
    IndependentRunCampaignSpec,
    IndependentRunRecord,
    SplitName,
    audit_independent_runs,
    build_independent_run_campaign_plan,
    build_inventory_from_campaign_plan,
    evaluate_model_benchmark,
    execute_independent_run_campaign,
)
from astrograph.diagnosis import (
    DatasetDiagnosisRequest,
    DatasetDiagnosisRunner,
    JsonDiagnosisRunStateStore,
    RW4_AUTHORITY_ROBUST_DETECTOR_ID,
    RW4_AUTHORITY_ROBUST_LEGACY_DETECTOR_IDS,
    RW4_SINGLE_FAULT_DETECTOR_ID,
    RW4_SINGLE_FAULT_LEGACY_DETECTOR_IDS,
    detect_rw4_authority_robust,
    detect_rw4_single_fault,
    SatSimPairedDiagnosisEngine,
    SatSimCatalogDiagnosisEngine,
    SatSimOpenSetDiagnosisEngine,
    diagnose_paired_multifault,
    diagnose_single_trace_multifault,
    engine_capabilities,
)
from astrograph.diagnosis.dataset_runner import DETECTOR_ID as ROBUST_BASELINE_DETECTOR_ID
from astrograph.knowledge import (
    GraphQueryKind,
    ReadOnlyCypherCompiler,
    csv_repository_from_release,
)
from astrograph.knowledge.repository import CsvKnowledgeRepository
from astrograph.knowledge.document_ingestion import (
    ingest_governed_document_to_directory,
    load_document_descriptor,
)
from astrograph.knowledge.document_candidate_entrypoint import (
    build_document_candidate_package,
)
from astrograph.knowledge.document_candidate_correction import (
    apply_document_candidate_correction,
    build_document_candidate_correction_suggestions,
    build_document_candidate_correction_workbook,
    compile_document_candidate_correction_workbook,
    prepare_document_candidate_correction,
)
from astrograph.knowledge.document_pack_entrypoint import (
    build_document_pack_candidate_packages,
)
from astrograph.knowledge.document_graph_review import (
    build_document_graph_review_handoff,
    build_single_document_graph_review_handoff,
)
from astrograph.knowledge.document_graph_review_decisions import (
    freeze_document_graph_review_decisions,
)
from astrograph.knowledge.document_review_suggestions import (
    build_document_component_archetype_suggestions,
    build_document_review_workbook,
    build_document_review_suggestions,
    compile_document_review_workbook,
)
from astrograph.retrieval.document_corpus_readiness import (
    assess_document_corpus_readiness,
)
from astrograph.retrieval.early_design_gold_review import (
    build_early_design_gold_review_workbook,
    freeze_early_design_gold_review,
)
from astrograph.retrieval.multimodal_gold_review import (
    build_multimodal_gold_review_workbook,
    freeze_multimodal_gold_review,
)
from astrograph.retrieval.chunking_gold_review import (
    build_chunking_gold_review_workbook,
    freeze_chunking_gold_review,
)
from astrograph.retrieval.contextual_chunking_confirmation import (
    build_contextual_chunking_gold_workbook,
    freeze_contextual_chunking_gold,
    run_contextual_chunking_confirmation,
)
from astrograph.retrieval.document_controlled_rewrite import (
    build_controlled_rewrite_pack,
    freeze_controlled_rewrite_review,
)
from astrograph.retrieval.document_rewrite_handoff import (
    materialize_reviewed_rewrite_dual_use_pack,
)
from astrograph.retrieval.document_rewrite_review_suggestions import (
    build_rewrite_review_suggestions,
    compile_rewrite_review_workbook,
)
from astrograph.knowledge.document_graph_patch import (
    build_document_graph_patch_proposal,
    compile_document_graph_patch_candidate,
    replay_document_graph_patch_topology,
)
from astrograph.knowledge.tbox_claim_release import (
    build_approved_tbox_claim_release,
)
from astrograph.schemas import WorkflowInput
from astrograph.runtime import (
    DIAGNOSIS_PIPELINE_ID,
    UNIFIED_DIAGNOSIS_PIPELINE_ID,
    RuntimeRequest,
    RuntimeStatus,
    build_application_runtime,
)
from astrograph.service import build_service_from_environment
from astrograph.simulation import SyntheticDatasetSpec, generate_synthetic_dataset
from astrograph.investigation import GraphQueryPlan
from astrograph.plugin_factory import (
    GeneratedPluginValidator,
    ObjectPluginBuildSpec,
    ObjectPluginFactoryService,
)
from astrograph.governance import (
    AppendOnlyJsonlLedger,
    GateEvidence,
    ImprovementContract,
    ProblemRecord,
    ProblemStack,
    decide_release,
    load_json_yaml,
    sha256_file,
    validate_contract_candidate,
    verify_evidence_artifacts,
    verify_gate_query_receipt,
)


def _diagnose(args: argparse.Namespace) -> int:
    if getattr(args, "telemetry", None) is not None:
        return _diagnose_unified(args)
    policy = DiagnosisPolicy.from_json(args.policy)
    repo = CsvKnowledgeRepository(args.edges) if args.edges else csv_repository_from_release(Path("."))
    anomalies = load_case_anomalies(args.fault_detection, args.case)
    request = WorkflowInput(
        case_id=args.case,
        request_id=args.request_id or uuid.uuid4().hex,
        anomalies=anomalies,
        metadata={"data_profile": "quick", "source": str(args.fault_detection)},
    )
    runtime = build_application_runtime(
        repository=repo,
        policy=policy,
        audit_store=JsonlAuditStore(args.audit),
        project_root=Path("."),
    )
    result = runtime.execute(
        RuntimeRequest(
            request_id=request.request_id,
            pipeline_id=DIAGNOSIS_PIPELINE_ID,
            payload=request.model_dump(mode="json"),
            metadata={"entrypoint": "astrograph.cli diagnose"},
        )
    )
    if result.status != RuntimeStatus.SUCCEEDED:
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return 2
    state = result.output
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{args.case}_diagnosis.json").write_text(
        json.dumps(state["diagnosis"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / f"{args.case}_report.md").write_text(state["report_markdown"], encoding="utf-8")
    (output / f"{args.case}_audit.json").write_text(
        json.dumps(state["audit"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "ok": True,
        "case": args.case,
        "status": state["diagnosis"]["status"],
        "runtime_pipeline": DIAGNOSIS_PIPELINE_ID,
    }, ensure_ascii=False))
    return 0


def _diagnose_unified(args: argparse.Namespace) -> int:
    runtime = build_service_from_environment(Path(".")).runtime
    request = RuntimeRequest(
        request_id=getattr(args, "request_id", None) or uuid.uuid4().hex,
        pipeline_id=UNIFIED_DIAGNOSIS_PIPELINE_ID,
        payload={
            "mode": "single",
            "case_id": args.case,
            "telemetry_path": str(args.telemetry),
            "nominal_path": str(args.nominal) if args.nominal is not None else None,
            "backend": args.backend,
            "catalog_path": str(args.catalog) if args.catalog is not None else None,
            "catalog_model_path": str(args.catalog_model) if args.catalog_model is not None else None,
            "open_set_model_path": str(args.open_set_model) if args.open_set_model is not None else None,
        },
        metadata={"entrypoint": "astrograph.cli diagnose --telemetry"},
    )
    execution = runtime.execute(request)
    if execution.status != RuntimeStatus.SUCCEEDED:
        print(json.dumps(execution.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return 2
    payload = execution.output
    output_path = args.output_json or (args.output / f"{args.case}_unified_diagnosis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    diagnosis_status = str(payload.get("status", "DATA_ERROR"))
    print(json.dumps({
        "ok": diagnosis_status != "DATA_ERROR",
        "case": payload.get("case_id", args.case),
        "status": diagnosis_status,
        "primary_effect_id": payload.get("primary_effect_id"),
        "runtime_pipeline": UNIFIED_DIAGNOSIS_PIPELINE_ID,
        "output": str(output_path),
    }, ensure_ascii=False))
    return 0 if diagnosis_status != "DATA_ERROR" else 2


def _diagnose_unified_batch(args: argparse.Namespace) -> int:
    runtime = build_service_from_environment(Path(".")).runtime
    execution = runtime.execute(RuntimeRequest(
        request_id=uuid.uuid4().hex,
        pipeline_id=UNIFIED_DIAGNOSIS_PIPELINE_ID,
        payload={"mode": "batch", "manifest_path": str(args.manifest)},
        metadata={"entrypoint": "astrograph.cli diagnose-batch"},
    ))
    if execution.status != RuntimeStatus.SUCCEEDED:
        print(json.dumps(execution.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return 2
    payload = execution.output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    results = payload.get("results", [])
    has_data_error = any(str(item.get("status", "")) == "DATA_ERROR" for item in results)
    print(json.dumps({
        "ok": not has_data_error,
        "case_count": len(results),
        "status_counts": payload.get("status_counts", {}),
        "runtime_pipeline": UNIFIED_DIAGNOSIS_PIPELINE_ID,
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0 if not has_data_error else 2


def _diagnosis_capabilities(args: argparse.Namespace) -> int:
    payload = {
        "schema_version": "astrograph-unified-capabilities.v1",
        "engines": [item.model_dump(mode="json") for item in engine_capabilities()],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _runtime_describe(args: argparse.Namespace) -> int:
    service = build_service_from_environment(args.project_root)
    payload = service.runtime.describe()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _runtime_execute(args: argparse.Namespace) -> int:
    service = build_service_from_environment(args.project_root)
    request = RuntimeRequest.model_validate_json(args.request.read_text(encoding="utf-8"))
    result = service.runtime.execute(request)
    payload = result.model_dump(mode="json")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if result.status == RuntimeStatus.SUCCEEDED else 2


def _validate(args: argparse.Namespace) -> int:
    report = validate_dataset(args.data_dir)
    write_report(report, args.output)
    print(json.dumps({"ok": report.status.value == "PASS", "status": report.status.value, "output": str(args.output)}, ensure_ascii=False))
    return 0 if report.status.value == "PASS" else 2



def _generate_object_plugin(args: argparse.Namespace) -> int:
    spec = ObjectPluginBuildSpec.model_validate_json(args.spec.read_text(encoding="utf-8"))
    result = ObjectPluginFactoryService().generate(spec, args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _validate_object_plugin(args: argparse.Namespace) -> int:
    result = GeneratedPluginValidator().validate(args.package_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0



def _validate_data_model_manifest(args: argparse.Namespace) -> int:
    manifest = DataDrivenModelManifest.model_validate_json(args.manifest.read_text(encoding="utf-8"))
    result = {
        "ok": True,
        "manifest_id": manifest.manifest_id,
        "feeds_birrf_directly": manifest.feeds_birrf_directly,
        "runtime": manifest.runtime.value,
        "task_type": manifest.task_type.value,
        "output_semantics": manifest.output_semantics.value,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _audit_independent_model_runs(args: argparse.Namespace) -> int:
    raw = json.loads(args.inventory.read_text(encoding="utf-8"))
    records_payload = raw.get("records", raw) if isinstance(raw, dict) else raw
    records = [IndependentRunRecord.model_validate(item) for item in records_payload]
    required_splits = tuple(SplitName(item) for item in args.required_splits.split(",") if item)
    audit = audit_independent_runs(
        records,
        minimum_runs_per_class=args.minimum_runs_per_class,
        required_splits=required_splits,
        strict_provenance=args.strict_provenance,
        reject_exact_content_duplicates=not args.allow_exact_content_duplicates,
        reject_group_leakage=not args.allow_group_leakage,
    )
    payload = audit.model_dump(mode="json")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if audit.status.value != "BLOCKED" else 2


def _plan_independent_run_campaign(args: argparse.Namespace) -> int:
    spec = IndependentRunCampaignSpec.model_validate_json(args.spec.read_text(encoding="utf-8"))
    plan = build_independent_run_campaign_plan(spec)
    payload = plan.model_dump(mode="json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if plan.status.value == "PASS" else 2


def _execute_independent_run_campaign(args: argparse.Namespace) -> int:
    plan = IndependentRunCampaignPlan.model_validate_json(args.plan.read_text(encoding="utf-8"))
    report = execute_independent_run_campaign(
        plan,
        runtime_root=args.runtime_root,
        output_root=args.output_root,
        stop_on_failure=not args.continue_on_failure,
    )
    payload = report.model_dump(mode="json")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if report.status == "PASS" else 2


def _build_campaign_inventory(args: argparse.Namespace) -> int:
    plan = IndependentRunCampaignPlan.model_validate_json(args.plan.read_text(encoding="utf-8"))
    records, missing = build_inventory_from_campaign_plan(plan, output_root=args.output_root)
    payload = {
        "schema_version": "astrograph-independent-run-inventory.v1",
        "campaign_id": plan.campaign_id,
        "campaign_plan_sha256": plan.plan_sha256,
        "records": [item.model_dump(mode="json") for item in records],
        "missing_run_ids": missing,
        "complete": not missing,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not missing else 2


def _compile_read_only_graph_query(args: argparse.Namespace) -> int:
    plan = GraphQueryPlan.model_validate_json(args.plan.read_text(encoding="utf-8"))
    compiled = ReadOnlyCypherCompiler().compile(
        plan,
        query_kind=GraphQueryKind(args.query_kind),
        knowledge_version=args.knowledge_version or None,
    )
    payload = compiled.model_dump(mode="json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _evaluate_model_benchmark(args: argparse.Namespace) -> int:
    predictions_raw = json.loads(args.predictions.read_text(encoding="utf-8"))
    predictions_payload = (
        predictions_raw.get("records", predictions_raw)
        if isinstance(predictions_raw, dict)
        else predictions_raw
    )
    inventory_raw = json.loads(args.inventory.read_text(encoding="utf-8"))
    inventory_payload = (
        inventory_raw.get("records", inventory_raw)
        if isinstance(inventory_raw, dict)
        else inventory_raw
    )
    records = [BenchmarkPredictionRecord.model_validate(item) for item in predictions_payload]
    inventory = [IndependentRunRecord.model_validate(item) for item in inventory_payload]
    audit = IndependentRunAudit.model_validate_json(args.audit.read_text(encoding="utf-8"))
    known_class_ids = (
        {item for item in args.known_class_ids.split(",") if item}
        if args.known_class_ids
        else None
    )
    top_k_values = tuple(int(item) for item in args.top_k.split(",") if item)
    report = evaluate_model_benchmark(
        records,
        inventory,
        audit,
        evaluated_split=SplitName(args.evaluated_split),
        known_class_ids=known_class_ids,
        top_k_values=top_k_values,
    )
    payload = report.model_dump(mode="json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if report.status.value == "PASS" else 2


def _source_config_from_args(args: argparse.Namespace) -> DatasetSourceConfig:
    source_type = DataSourceType(args.source_type)
    root = str(getattr(args, "root", "") or "")
    if source_type == DataSourceType.ESA_ADB and not root:
        root = os.environ.get("ASTROGRAPH_ESA_ADB_ROOT", "")
    if source_type == DataSourceType.SIMULATION and not root:
        root = os.environ.get("ASTROGRAPH_SIMULATION_DATA_ROOT", "")
    if source_type == DataSourceType.SAT_AGENT_PLATFORM and not root:
        root = os.environ.get("ASTROGRAPH_SAT_AGENT_ROOT", "")
    if not root:
        raise ValueError("dataset root is required (or configure the matching ASTROGRAPH_*_ROOT environment variable)")
    labels_raw = str(getattr(args, "label_columns", "") or "")
    labels = [item for item in (part.strip() for part in labels_raw.split(",")) if item]
    profile = getattr(args, "profile", None)
    return DatasetSourceConfig(
        source_type=source_type,
        root=root,
        read_only=True,
        lazy=True,
        recursive=True,
        timestamp_column=(getattr(args, "timestamp_column", None) or None),
        label_columns=labels,
        profile_path=None if profile is None else str(profile),
        metadata={
            "trusted_pickle": bool(getattr(args, "trust_local_pickle", False)),
        },
    )


def _dataset_probe(args: argparse.Namespace) -> int:
    source = build_data_source(_source_config_from_args(args))
    options = ProbeOptions(
        max_entities=args.max_entities,
        max_rows_per_entity=args.max_rows,
        hash_mode=args.hash_mode,
        include_statistics=args.include_statistics,
        include_label_values=args.include_label_values,
        root_token=(
            "${ASTROGRAPH_ESA_ADB_ROOT}"
            if args.source_type == DataSourceType.ESA_ADB.value
            else "${ASTROGRAPH_SIMULATION_DATA_ROOT}"
            if args.source_type == DataSourceType.SIMULATION.value
            else "${ASTROGRAPH_SAT_AGENT_ROOT}"
            if args.source_type == DataSourceType.SAT_AGENT_PLATFORM.value
            else "${ASTROGRAPH_DATA_ROOT}"
        ),
    )
    artifacts = probe_data_source(source, args.output, options=options)
    payload = artifacts.model_dump(mode="json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _dataset_validate_source(args: argparse.Namespace) -> int:
    source = build_data_source(_source_config_from_args(args))
    report = validate_data_source(
        source,
        max_entities=args.max_entities,
        max_rows_per_entity=args.max_rows,
    )
    payload = report.model_dump(mode="json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if report.status in {"PASS", "WARNING"} else 2


def _simulate_generate_dataset(args: argparse.Namespace) -> int:
    spec = SyntheticDatasetSpec.model_validate_json(args.spec.read_text(encoding="utf-8"))
    profile = None
    if args.schema_profile:
        profile = json.loads(args.schema_profile.read_text(encoding="utf-8"))
    manifest = generate_synthetic_dataset(spec, args.output_root, schema_profile=profile)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _dataset_detector_selection(args: argparse.Namespace) -> tuple[object | None, str]:
    name = str(getattr(args, "detector", "robust-baseline") or "robust-baseline")
    if name == "auto":
        run_id = str(getattr(args, "run_id", ""))
        state = JsonDiagnosisRunStateStore(args.state_root).get(run_id)
        if state is None:
            raise KeyError(f"diagnosis run state not found: {run_id}")
        saved = str(
            (state.stage_outputs.get("RUN_CONFIG", {}) or {}).get("detector_id", "")
            or (state.stage_outputs.get("DETECTED", {}) or {}).get("detector_id", "")
            or ROBUST_BASELINE_DETECTOR_ID
        )
        if saved == RW4_AUTHORITY_ROBUST_DETECTOR_ID:
            return detect_rw4_authority_robust, saved
        if saved == RW4_SINGLE_FAULT_DETECTOR_ID or saved in RW4_SINGLE_FAULT_LEGACY_DETECTOR_IDS:
            # Preserve the persisted detector identity and implementation for historical replay.
            return detect_rw4_single_fault, saved
        if saved == ROBUST_BASELINE_DETECTOR_ID:
            name = "robust-baseline"
        else:
            raise ValueError(f"unsupported saved detector_id: {saved}")
    if name == "rw4-single-fault":
        return detect_rw4_authority_robust, RW4_AUTHORITY_ROBUST_DETECTOR_ID
    if name == "rw4-single-fault-v2":
        return detect_rw4_single_fault, RW4_SINGLE_FAULT_DETECTOR_ID
    if name == "robust-baseline":
        return None, ROBUST_BASELINE_DETECTOR_ID
    raise ValueError(f"unsupported dataset detector: {name}")


def _build_dataset_diagnosis_runner(args: argparse.Namespace) -> DatasetDiagnosisRunner:
    source = build_data_source(_source_config_from_args(args))
    repository = (
        CsvKnowledgeRepository(args.edges)
        if args.edges
        else csv_repository_from_release(Path("."))
    )
    policy = (
        DiagnosisPolicy.from_json(args.policy)
        if getattr(args, "policy", None)
        else DiagnosisPolicy()
    )
    audit_store = JsonlAuditStore(args.audit) if getattr(args, "audit", None) else None
    detector, detector_id = _dataset_detector_selection(args)
    return DatasetDiagnosisRunner(
        source,
        repository,
        state_store=JsonDiagnosisRunStateStore(args.state_root),
        policy=policy,
        audit_store=audit_store,
        detector=detector,
        detector_id=detector_id,
    )


def _diagnosis_dataset_run(args: argparse.Namespace) -> int:
    runner = _build_dataset_diagnosis_runner(args)
    channels = [item for item in (part.strip() for part in args.channels.split(",")) if item]
    signal_mapping: dict = {}
    if args.signal_mapping:
        signal_mapping = json.loads(args.signal_mapping.read_text(encoding="utf-8"))
    request = DatasetDiagnosisRequest(
        request_id=args.request_id or uuid.uuid4().hex,
        entity_id=args.entity,
        start=args.start or None,
        end=args.end or None,
        channels=channels,
        baseline_fraction=args.baseline_fraction,
        z_threshold=args.z_threshold,
        minimum_baseline_points=args.minimum_baseline_points,
        signal_mapping=signal_mapping,
        metadata={"cli": "astrograph diagnosis run"},
    )
    result = runner.run(request, output_dir=args.output, force=args.force)
    payload = result.model_dump(mode="json")
    print(
        json.dumps(
            {
                "ok": True,
                "run_id": result.run_id,
                "diagnosis_id": result.diagnosis_id,
                "conclusion": result.conclusion.model_dump(mode="json"),
                "replayed": result.replayed,
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.conclusion.level.value != "system_error" else 2


def _diagnosis_dataset_resume(args: argparse.Namespace) -> int:
    runner = _build_dataset_diagnosis_runner(args)
    result = runner.resume(args.run_id, output_dir=args.output)
    print(
        json.dumps(
            {
                "ok": True,
                "run_id": result.run_id,
                "diagnosis_id": result.diagnosis_id,
                "conclusion": result.conclusion.model_dump(mode="json"),
                "replayed": result.replayed,
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.conclusion.level.value != "system_error" else 2




def _diagnosis_sat_sim_paired(args: argparse.Namespace) -> int:
    result = SatSimPairedDiagnosisEngine().diagnose_paths(
        case_id=args.case_id,
        backend=args.backend,
        nominal_path=args.nominal,
        effect_path=args.telemetry,
    )
    payload = result.model_dump(mode="json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "ok": result.status.value not in {"DATA_ERROR"},
        "case_id": result.case_id,
        "status": result.status.value,
        "conclusion_effect_id": result.conclusion_effect_id,
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0 if result.status.value not in {"DATA_ERROR"} else 2



def _diagnosis_sat_sim_multifault(args: argparse.Namespace) -> int:
    result = diagnose_paired_multifault(
        case_id=args.case_id,
        nominal_csv=args.nominal,
        effect_csv=args.telemetry,
        catalog_path=args.catalog,
    )
    payload = result.model_dump(mode="json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "ok": result.status.value not in {"DATA_ERROR"},
        "case_id": result.case_id,
        "status": result.status.value,
        "detected_effect_ids": result.detected_effect_ids,
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0 if result.status.value not in {"DATA_ERROR"} else 2


def _diagnosis_sat_sim_multifault_single(args: argparse.Namespace) -> int:
    result = diagnose_single_trace_multifault(
        case_id=args.case_id,
        telemetry_csv=args.telemetry,
        catalog_path=args.catalog,
    )
    payload = result.model_dump(mode="json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "ok": result.status.value not in {"DATA_ERROR"},
        "case_id": result.case_id,
        "status": result.status.value,
        "detected_effect_ids": result.detected_effect_ids,
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0 if result.status.value not in {"DATA_ERROR"} else 2


def _diagnosis_sat_sim_catalog(args: argparse.Namespace) -> int:
    result = SatSimCatalogDiagnosisEngine.from_path(args.model).diagnose_path(
        case_id=args.case_id,
        backend=args.backend,
        telemetry_path=args.telemetry,
    )
    payload = result.model_dump(mode="json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "ok": result.status != "DATA_ERROR",
        "case_id": result.case_id,
        "status": result.status,
        "conclusion_effect_id": result.conclusion_effect_id,
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0 if result.status != "DATA_ERROR" else 2


def _diagnosis_sat_sim_open_set(args: argparse.Namespace) -> int:
    result = SatSimOpenSetDiagnosisEngine.from_path(args.model).diagnose_path(
        case_id=args.case_id,
        backend=args.backend,
        telemetry_path=args.telemetry,
    )
    payload = result.model_dump(mode="json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "ok": result.status not in {"DATA_ERROR", "OUT_OF_SCOPE"},
        "case_id": result.case_id,
        "status": result.status,
        "conclusion_effect_id": result.conclusion_effect_id,
        "catalog_membership_confidence": result.catalog_membership_confidence,
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0 if result.status not in {"DATA_ERROR", "OUT_OF_SCOPE"} else 2

def _diagnosis_dataset_inspect(args: argparse.Namespace) -> int:
    store = JsonDiagnosisRunStateStore(args.state_root)
    state = store.get(args.run_id)
    if state is None:
        print(json.dumps({"ok": False, "error": "RUN_NOT_FOUND", "run_id": args.run_id}, ensure_ascii=False))
        return 2
    print(json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0

def _audit_trusted_round6_data(args: argparse.Namespace) -> int:
    audit, rows, leakage = audit_round6_dataset(
        args.dataset_root,
        args.sat_agent_root,
        dataset_archive_sha256=args.dataset_archive_sha256,
        sat_agent_archive_sha256=args.sat_agent_archive_sha256,
    )
    output = args.output_dir
    write_trusted_audit_outputs(
        audit,
        rows,
        leakage,
        audit_path=output / "trusted_dataset_audit.json",
        manifest_jsonl_path=output / "trusted_case_manifest.jsonl",
        matrix_csv_path=output / "trusted_case_matrix.csv",
        leakage_path=output / "public_trace_leakage.json",
    )
    print(json.dumps(audit.to_dict(), ensure_ascii=False, indent=2))
    return 0 if audit.status == "PASS" else 2


def _governance_validate(args: argparse.Namespace) -> int:
    contract = load_json_yaml(args.contract, ImprovementContract)
    report = validate_contract_candidate(
        contract,
        contract_path=args.contract,
        candidate_root=args.candidate_root,
        expected_contract_sha256=args.expected_contract_sha256,
    )
    payload = report.model_dump(mode="json")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if report.status.value == "PASS" else 2


def _governance_decide(args: argparse.Namespace) -> int:
    if sha256_file(args.contract) != args.expected_contract_sha256:
        print(json.dumps({"ok": False, "error": "CONTRACT_HASH_MISMATCH"}, ensure_ascii=False))
        return 2
    contract = load_json_yaml(args.contract, ImprovementContract)
    evidence = GateEvidence.model_validate_json(args.gate_evidence.read_text(encoding="utf-8"))
    artifact_errors = verify_evidence_artifacts(evidence, args.evidence_root)
    receipt_errors = verify_gate_query_receipt(
        evidence,
        receipt_path=args.query_receipt,
        contract=contract,
        expected_contract_sha256=args.expected_contract_sha256,
    )
    decision = decide_release(
        contract,
        evidence,
        expected_contract_sha256=args.expected_contract_sha256,
        expected_evaluator_sha256=args.expected_evaluator_sha256,
        expected_evaluator_module_sha256=args.expected_evaluator_module_sha256,
        evidence_artifact_errors=artifact_errors,
        query_receipt_errors=receipt_errors,
    )
    payload = decision.model_dump(mode="json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if decision.status.value == "ACCEPT" else 2


def _governance_audit_ledger(args: argparse.Namespace) -> int:
    result = AppendOnlyJsonlLedger(args.ledger).audit()
    payload = result.model_dump(mode="json")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if result.status == "PASS" else 2


def _problem_stack(args: argparse.Namespace) -> int:
    stack = ProblemStack(args.events, args.current)
    if args.problem_action == "open":
        record = ProblemRecord.model_validate_json(args.record.read_text(encoding="utf-8"))
        view = stack.open(record)
    elif args.problem_action == "view":
        view = stack.view()
    else:
        patch = {}
        if args.patch is not None:
            patch = json.loads(args.patch.read_text(encoding="utf-8"))
            if not isinstance(patch, dict):
                raise ValueError("problem transition patch must be a JSON object")
        view = stack.transition(args.problem_id, args.problem_action, patch)
    print(json.dumps(view.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


def _process_governed_document(args: argparse.Namespace) -> int:
    descriptor = load_document_descriptor(args.descriptor)
    receipt = ingest_governed_document_to_directory(
        descriptor=descriptor,
        source_path=args.source,
        output_dir=args.output,
        independence_group=args.independence_group,
        license_id=args.license_id,
        knowledge_version=args.knowledge_version,
    )
    print(receipt.model_dump_json(indent=2))
    return 0


def _build_document_candidate(args: argparse.Namespace) -> int:
    proposal = None
    backend = None
    if args.proposal is not None:
        proposal = KnowledgeExtractionProposal.model_validate_json(
            args.proposal.read_text(encoding="utf-8-sig")
        )
    else:
        backend = VllmKnowledgeExtractionBackend(
            base_url=args.base_url,
            model_id=args.model,
            timeout_s=args.timeout_s,
            max_tokens=args.max_tokens,
        )
        backend.discover_model()
    receipt = build_document_candidate_package(
        ingestion_package_dir=args.input_package,
        output_dir=args.output,
        review_refreeze_path=args.review_refreeze,
        expected_review_refreeze_file_sha256=args.expected_review_refreeze_sha256,
        graph_id=args.graph_id,
        backend=backend,
        proposal=proposal,
        section_batched=not args.whole_document,
    )
    print(receipt.model_dump_json(indent=2))
    return 2 if receipt.status == "BLOCKED_BY_EXCEPTION" else 0


def _build_document_pack_candidates(args: argparse.Namespace) -> int:
    backend_factory = None
    proposal_root = args.frozen_proposal_root
    if proposal_root is None:
        probe = VllmKnowledgeExtractionBackend(
            base_url=args.base_url,
            model_id=args.model,
            timeout_s=args.timeout_s,
            max_tokens=args.max_tokens,
        )
        model_id = probe.discover_model()

        def backend_factory() -> VllmKnowledgeExtractionBackend:
            return VllmKnowledgeExtractionBackend(
                base_url=args.base_url,
                model_id=model_id,
                timeout_s=args.timeout_s,
                max_tokens=args.max_tokens,
            )

    receipt = build_document_pack_candidate_packages(
        manifest_path=args.manifest,
        corpus_root=args.corpus_root,
        output_root=args.output,
        review_refreeze_path=args.review_refreeze,
        expected_review_refreeze_file_sha256=args.expected_review_refreeze_sha256,
        backend_factory=backend_factory,
        frozen_proposal_root=proposal_root,
        selected_fault_ids=args.fault_id,
        maximum_documents=args.maximum_documents,
        section_batched=not args.whole_document,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 2 if receipt["status_counts"].get("BLOCKED_BY_EXCEPTION", 0) else 0


def _prepare_document_graph_review(args: argparse.Namespace) -> int:
    receipt = build_single_document_graph_review_handoff(
        candidate_package_root=args.candidate_package,
        output_dir=args.output,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


def _prepare_document_pack_graph_review(args: argparse.Namespace) -> int:
    target = args.output.resolve()
    if target.exists():
        raise ValueError("refusing to overwrite document pack graph review")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        receipt = build_document_graph_review_handoff(
            primary_root=args.candidate_pack,
            supplemental_roots=[],
            output_dir=temporary,
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


def _freeze_document_pack_graph_review(args: argparse.Namespace) -> int:
    target = args.output.resolve()
    if target.exists():
        raise ValueError("refusing to overwrite document pack graph review freeze")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        receipt = freeze_document_graph_review_decisions(
            review_sheet_path=args.review_sheet,
            graph_roots=[args.candidate_pack],
            output_path=temporary,
            require_complete=not args.allow_incomplete,
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["status"] != "BLOCKED" else 2


def _suggest_document_pack_review(args: argparse.Namespace) -> int:
    target = args.output.resolve()
    if target.exists():
        raise ValueError("refusing to overwrite document pack review suggestions")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        receipt = build_document_review_suggestions(
            candidate_pack_root=args.candidate_pack,
            review_root=args.review,
            base_graph_path=args.base_graph,
            output_dir=temporary,
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


def _build_document_pack_review_workbook(args: argparse.Namespace) -> int:
    receipt = build_document_review_workbook(
        candidate_pack_root=args.candidate_pack,
        review_root=args.review,
        suggestion_root=args.suggestions,
        component_suggestion_root=args.component_suggestions,
        output_dir=args.output,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


def _suggest_document_component_archetypes(args: argparse.Namespace) -> int:
    receipt = build_document_component_archetype_suggestions(
        review_root=args.review,
        suggestion_root=args.suggestions,
        channel_map_catalog_path=args.channel_map_catalog,
        base_graph_path=args.base_graph,
        output_dir=args.output,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


def _compile_document_pack_review_workbook(args: argparse.Namespace) -> int:
    receipt = compile_document_review_workbook(
        candidate_pack_root=args.candidate_pack,
        workbook_root=args.workbook_root,
        workbook_path=args.workbook,
        output_dir=args.output,
        require_complete=not args.allow_incomplete,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["status"] != "BLOCKED" else 2


def _audit_document_corpus_readiness(args: argparse.Namespace) -> int:
    target = args.output.resolve()
    if target.exists():
        raise ValueError("refusing to overwrite P0.5 readiness assessment")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        receipt = assess_document_corpus_readiness(
            project_root=args.project_root,
            evidence_spec_path=args.evidence_spec,
            output_path=temporary,
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["ready_for_p1"] else 2


def _prepare_early_design_gold_review(args: argparse.Namespace) -> int:
    receipt = build_early_design_gold_review_workbook(
        inventory_path=args.inventory,
        epistemic_candidates_path=args.epistemic_candidates,
        drift_candidates_path=args.drift_candidates,
        review_template_path=args.review_template,
        project_root=args.project_root,
        output_dir=args.output,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


def _freeze_early_design_gold_review(args: argparse.Namespace) -> int:
    receipt = freeze_early_design_gold_review(
        workbook_root=args.workbook,
        output_path=args.output,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["status"] == "COMPLETE" else 2


def _prepare_multimodal_gold_review(args: argparse.Namespace) -> int:
    receipt = build_multimodal_gold_review_workbook(
        source_path=args.source,
        build_receipt_path=args.build_receipt,
        machine_candidate_path=args.machine_candidate,
        review_template_path=args.review_template,
        output_dir=args.output,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


def _freeze_multimodal_gold_review(args: argparse.Namespace) -> int:
    receipt = freeze_multimodal_gold_review(
        source_path=args.source,
        build_receipt_path=args.build_receipt,
        machine_candidate_path=args.machine_candidate,
        review_template_path=args.review_template,
        workbook_root=args.workbook,
        output_path=args.output,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["status"] == "COMPLETE" else 2


def _prepare_chunking_gold_review(args: argparse.Namespace) -> int:
    receipt = build_chunking_gold_review_workbook(
        preregistration_path=args.preregistration,
        machine_results_path=args.machine_results,
        handbook_path=args.handbook,
        canonical_facts_path=args.canonical_facts,
        protocol_path=args.protocol,
        output_dir=args.output,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


def _freeze_chunking_gold_review(args: argparse.Namespace) -> int:
    receipt = freeze_chunking_gold_review(
        preregistration_path=args.preregistration,
        machine_results_path=args.machine_results,
        handbook_path=args.handbook,
        canonical_facts_path=args.canonical_facts,
        protocol_path=args.protocol,
        workbook_root=args.workbook,
        output_path=args.output,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["status"] == "COMPLETE" else 2


def _compile_approved_tbox_claims(args: argparse.Namespace) -> int:
    receipt = build_approved_tbox_claim_release(
        graph_path=args.graph,
        construction_bundle_path=args.construction_bundle,
        approval_path=args.approval,
        output_path=args.output,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


def _run_contextual_chunking_confirmation(args: argparse.Namespace) -> int:
    result = run_contextual_chunking_confirmation(
        preregistration_path=args.preregistration,
        ingestion_path=args.ingestion,
    )
    target = args.output.resolve()
    if target.exists():
        raise ValueError("refusing to overwrite contextual chunking machine results")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _prepare_contextual_chunking_gold(args: argparse.Namespace) -> int:
    receipt = build_contextual_chunking_gold_workbook(
        preregistration_path=args.preregistration,
        ingestion_path=args.ingestion,
        results_path=args.results,
        output_dir=args.output,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


def _freeze_contextual_chunking_gold(args: argparse.Namespace) -> int:
    receipt = freeze_contextual_chunking_gold(
        preregistration_path=args.preregistration,
        ingestion_path=args.ingestion,
        results_path=args.results,
        workbook_root=args.workbook,
        output_path=args.output,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["status"] == "COMPLETE" else 2


def _build_controlled_document_rewrites(args: argparse.Namespace) -> int:
    target = args.output.resolve()
    if target.exists():
        raise ValueError("refusing to overwrite controlled document rewrite pack")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    backend = None
    if args.frozen_proposal_root is None:
        backend = VllmKnowledgeExtractionBackend(
            base_url=args.base_url, model_id=args.model,
            timeout_s=args.timeout_s, max_tokens=args.max_tokens,
        )
        backend.discover_model()
    try:
        receipt = build_controlled_rewrite_pack(
            manifest_path=args.manifest, corpus_root=args.corpus_root,
            output_root=temporary, backend=backend,
            frozen_proposal_root=args.frozen_proposal_root,
            maximum_documents=args.maximum_documents,
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["status"] != "BLOCKED" else 2


def _freeze_controlled_document_rewrite_review(args: argparse.Namespace) -> int:
    target = args.output.resolve()
    if target.exists():
        raise ValueError("refusing to overwrite controlled rewrite review freeze")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        receipt = freeze_controlled_rewrite_review(
            rewrite_pack_root=args.rewrite_pack, review_sheet_path=args.review_sheet,
            output_path=temporary, require_complete=not args.allow_incomplete,
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["status"] != "BLOCKED" else 2


def _materialize_reviewed_document_rewrites(args: argparse.Namespace) -> int:
    try:
        receipt = materialize_reviewed_rewrite_dual_use_pack(
            source_manifest_path=args.source_manifest,
            rewrite_pack_root=args.rewrite_pack,
            review_freeze_path=args.review_freeze,
            output_root=args.output,
            release_id=args.release_id,
        )
    except ValueError as exc:
        print(
            json.dumps(
                {
                    "schema_version": (
                        "astrograph-reviewed-rewrite-dual-use-handoff-attempt.v1"
                    ),
                    "status": "BLOCKED",
                    "reason": str(exc),
                    "output_created": args.output.resolve().exists(),
                    "rag_corpus_build_allowed": False,
                    "kg_staging_allowed": False,
                    "publication_allowed": False,
                    "runtime_vote_allowed": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


def _suggest_controlled_document_rewrite_review(args: argparse.Namespace) -> int:
    target = args.output.resolve()
    if target.exists():
        raise ValueError("refusing to overwrite controlled rewrite review suggestions")
    backend = None
    if args.frozen_suggestion_root is None:
        backend = VllmKnowledgeExtractionBackend(
            base_url=args.base_url,
            model_id=args.model,
            timeout_s=args.timeout_s,
            max_tokens=args.max_tokens,
        )
        backend.discover_model()
    receipt = build_rewrite_review_suggestions(
        rewrite_pack_root=args.rewrite_pack,
        review_sheet_path=args.review_sheet,
        output_root=target,
        backend=backend,
        frozen_suggestion_root=args.frozen_suggestion_root,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


def _compile_controlled_document_rewrite_review_workbook(
    args: argparse.Namespace,
) -> int:
    receipt = compile_rewrite_review_workbook(
        rewrite_pack_root=args.rewrite_pack,
        suggestion_root=args.suggestion_pack,
        workbook_path=args.workbook,
        output_root=args.output,
        require_complete=not args.allow_incomplete,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["status"] != "BLOCKED" else 2


def _prepare_document_candidate_correction(args: argparse.Namespace) -> int:
    receipt = prepare_document_candidate_correction(
        candidate_package_root=args.candidate_package,
        output_dir=args.output,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


def _suggest_document_candidate_correction(args: argparse.Namespace) -> int:
    receipt = build_document_candidate_correction_suggestions(
        candidate_package_root=args.candidate_package,
        correction_handoff_root=args.correction_handoff,
        output_dir=args.output,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


def _build_document_candidate_correction_workbook(args: argparse.Namespace) -> int:
    receipt = build_document_candidate_correction_workbook(
        candidate_package_root=args.candidate_package,
        correction_handoff_root=args.correction_handoff,
        suggestion_root=args.suggestions,
        output_dir=args.output,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


def _compile_document_candidate_correction_workbook(args: argparse.Namespace) -> int:
    receipt = compile_document_candidate_correction_workbook(
        candidate_package_root=args.candidate_package,
        workbook_root=args.workbook_root,
        workbook_path=args.workbook,
        output_dir=args.output,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["status"] == "READY_FOR_CORRECTION_REVALIDATION" else 2


def _apply_document_candidate_correction(args: argparse.Namespace) -> int:
    receipt = apply_document_candidate_correction(
        ingestion_package_dir=args.input_package,
        candidate_package_root=args.candidate_package,
        correction_sheet_path=args.correction_sheet,
        review_refreeze_path=args.review_refreeze,
        expected_review_refreeze_file_sha256=args.expected_review_refreeze_sha256,
        graph_id=args.graph_id,
        output_dir=args.output,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["status"] == "CORRECTED_CANDIDATE_READY_FOR_HUMAN_REVIEW" else 2


def _freeze_document_graph_review(args: argparse.Namespace) -> int:
    receipt = freeze_document_graph_review_decisions(
        review_sheet_path=args.review_sheet,
        graph_roots=[args.candidate_package],
        output_path=args.output,
        require_complete=not args.allow_incomplete,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["status"] != "BLOCKED" else 2


def _propose_document_graph_patch(args: argparse.Namespace) -> int:
    target = args.output.resolve()
    if target.exists():
        raise ValueError("refusing to overwrite document graph patch proposal")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        receipt = build_document_graph_patch_proposal(
            review_freeze_path=args.review_freeze,
            base_graph_path=args.base_graph,
            output_path=temporary,
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["status"] != "BLOCKED" else 2


def _compile_document_graph_patch(args: argparse.Namespace) -> int:
    target = args.output.resolve()
    if target.exists():
        raise ValueError("refusing to overwrite compiled document graph patch")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        receipt = compile_document_graph_patch_candidate(
            patch_proposal_path=args.patch_proposal,
            base_graph_path=args.base_graph,
            output_dir=temporary,
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


def _replay_document_graph_patch(args: argparse.Namespace) -> int:
    target = args.output.resolve()
    if target.exists():
        raise ValueError("refusing to overwrite document graph patch replay")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        receipt = replay_document_graph_patch_topology(
            base_graph_path=args.base_graph,
            candidate_graph_path=args.candidate_graph,
            seed_node_ids=args.seed_node_id,
            target_node_types=args.target_node_type,
            expected_added_path_signatures=args.expected_added_path_signature,
            output_path=temporary,
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["status"] == "PASS" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="AstroGraph stable runtime CLI with compatibility commands")
    sub = parser.add_subparsers(dest="command", required=True)

    runtime_group = sub.add_parser("runtime", help="Canonical AstroGraph runtime")
    runtime_sub = runtime_group.add_subparsers(dest="runtime_command", required=True)
    runtime_describe = runtime_sub.add_parser("describe", help="Describe stable runtime pipelines and contracts")
    runtime_describe.add_argument("--project-root", type=Path, default=Path("."))
    runtime_describe.set_defaults(func=_runtime_describe)
    runtime_execute = runtime_sub.add_parser("execute", help="Execute an astrograph-runtime-request.v1 JSON envelope")
    runtime_execute.add_argument("--project-root", type=Path, default=Path("."))
    runtime_execute.add_argument("--request", type=Path, required=True)
    runtime_execute.add_argument("--output", type=Path, default=None)
    runtime_execute.set_defaults(func=_runtime_execute)

    process_document = sub.add_parser(
        "process-document",
        help="Atomically stage one governed document for RAG/OAG and candidate-KG use",
    )
    process_document.add_argument("--source", type=Path, required=True)
    process_document.add_argument("--descriptor", type=Path, required=True)
    process_document.add_argument("--output", type=Path, required=True)
    process_document.add_argument("--independence-group", required=True)
    process_document.add_argument("--license-id", required=True)
    process_document.add_argument(
        "--knowledge-version", default="document-dual-use-v1"
    )
    process_document.set_defaults(func=_process_governed_document)

    build_document_graph = sub.add_parser(
        "build-document-candidate-graph",
        help="Extract and compile a governed document into a review-only candidate graph",
    )
    build_document_graph.add_argument("--input-package", type=Path, required=True)
    build_document_graph.add_argument("--output", type=Path, required=True)
    build_document_graph.add_argument("--review-refreeze", type=Path, required=True)
    build_document_graph.add_argument(
        "--expected-review-refreeze-sha256", required=True
    )
    build_document_graph.add_argument("--graph-id", required=True)
    build_document_graph.add_argument("--proposal", type=Path, default=None)
    build_document_graph.add_argument("--base-url", default="http://127.0.0.1:8000")
    build_document_graph.add_argument("--model", default="auto")
    build_document_graph.add_argument("--timeout-s", type=float, default=180.0)
    build_document_graph.add_argument("--max-tokens", type=int, default=4096)
    build_document_graph.add_argument(
        "--whole-document",
        action="store_true",
        help="Disable the default section-batched extraction mode",
    )
    build_document_graph.set_defaults(func=_build_document_candidate)

    build_document_pack = sub.add_parser(
        "build-document-pack-candidates",
        help="Atomically run a governed manifest through stable per-document ingestion and candidate stages",
    )
    build_document_pack.add_argument("--manifest", type=Path, required=True)
    build_document_pack.add_argument("--corpus-root", type=Path, required=True)
    build_document_pack.add_argument("--output", type=Path, required=True)
    build_document_pack.add_argument("--review-refreeze", type=Path, required=True)
    build_document_pack.add_argument("--expected-review-refreeze-sha256", required=True)
    build_document_pack.add_argument("--frozen-proposal-root", type=Path, default=None)
    build_document_pack.add_argument("--fault-id", action="append", default=None)
    build_document_pack.add_argument("--maximum-documents", type=int, default=None)
    build_document_pack.add_argument("--base-url", default="http://127.0.0.1:8000")
    build_document_pack.add_argument("--model", default="auto")
    build_document_pack.add_argument("--timeout-s", type=float, default=180.0)
    build_document_pack.add_argument("--max-tokens", type=int, default=4096)
    build_document_pack.add_argument("--whole-document", action="store_true")
    build_document_pack.set_defaults(func=_build_document_pack_candidates)

    prepare_candidate_correction = sub.add_parser(
        "prepare-document-candidate-correction",
        help="Create a source-bound human correction sheet for a governed blocked candidate",
    )
    prepare_candidate_correction.add_argument("--candidate-package", type=Path, required=True)
    prepare_candidate_correction.add_argument("--output", type=Path, required=True)
    prepare_candidate_correction.set_defaults(func=_prepare_document_candidate_correction)

    suggest_candidate_correction = sub.add_parser(
        "suggest-document-candidate-correction",
        help="Generate non-binding critic-derived advice for a blank candidate correction handoff",
    )
    suggest_candidate_correction.add_argument("--candidate-package", type=Path, required=True)
    suggest_candidate_correction.add_argument("--correction-handoff", type=Path, required=True)
    suggest_candidate_correction.add_argument("--output", type=Path, required=True)
    suggest_candidate_correction.set_defaults(func=_suggest_document_candidate_correction)

    build_candidate_correction_workbook = sub.add_parser(
        "build-document-candidate-correction-workbook",
        help="Merge a blank correction handoff and critic advice into a locked human workbook",
    )
    build_candidate_correction_workbook.add_argument("--candidate-package", type=Path, required=True)
    build_candidate_correction_workbook.add_argument("--correction-handoff", type=Path, required=True)
    build_candidate_correction_workbook.add_argument("--suggestions", type=Path, required=True)
    build_candidate_correction_workbook.add_argument("--output", type=Path, required=True)
    build_candidate_correction_workbook.set_defaults(
        func=_build_document_candidate_correction_workbook
    )

    compile_candidate_correction_workbook = sub.add_parser(
        "compile-document-candidate-correction-workbook",
        help="Validate a filled correction workbook and restore the canonical correction CSV",
    )
    compile_candidate_correction_workbook.add_argument("--candidate-package", type=Path, required=True)
    compile_candidate_correction_workbook.add_argument("--workbook-root", type=Path, required=True)
    compile_candidate_correction_workbook.add_argument("--workbook", type=Path, required=True)
    compile_candidate_correction_workbook.add_argument("--output", type=Path, required=True)
    compile_candidate_correction_workbook.set_defaults(
        func=_compile_document_candidate_correction_workbook
    )

    apply_candidate_correction = sub.add_parser(
        "apply-document-candidate-correction",
        help="Rebuild and revalidate a blocked document candidate from named corrections",
    )
    apply_candidate_correction.add_argument("--input-package", type=Path, required=True)
    apply_candidate_correction.add_argument("--candidate-package", type=Path, required=True)
    apply_candidate_correction.add_argument("--correction-sheet", type=Path, required=True)
    apply_candidate_correction.add_argument("--review-refreeze", type=Path, required=True)
    apply_candidate_correction.add_argument(
        "--expected-review-refreeze-sha256", required=True
    )
    apply_candidate_correction.add_argument("--graph-id", required=True)
    apply_candidate_correction.add_argument("--output", type=Path, required=True)
    apply_candidate_correction.set_defaults(func=_apply_document_candidate_correction)

    prepare_document_review = sub.add_parser(
        "prepare-document-graph-review",
        help="Create a blank source-bound review sheet for one document candidate package",
    )
    prepare_document_review.add_argument("--candidate-package", type=Path, required=True)
    prepare_document_review.add_argument("--output", type=Path, required=True)
    prepare_document_review.set_defaults(func=_prepare_document_graph_review)

    prepare_pack_review = sub.add_parser(
        "prepare-document-pack-graph-review",
        help="Create one source-bound review sheet for a stable document candidate pack",
    )
    prepare_pack_review.add_argument("--candidate-pack", type=Path, required=True)
    prepare_pack_review.add_argument("--output", type=Path, required=True)
    prepare_pack_review.set_defaults(func=_prepare_document_pack_graph_review)

    suggest_pack_review = sub.add_parser(
        "suggest-document-pack-review",
        help="Generate non-binding exact-match suggestions beside a blank human review sheet",
    )
    suggest_pack_review.add_argument("--candidate-pack", type=Path, required=True)
    suggest_pack_review.add_argument("--review", type=Path, required=True)
    suggest_pack_review.add_argument("--base-graph", type=Path, required=True)
    suggest_pack_review.add_argument("--output", type=Path, required=True)
    suggest_pack_review.set_defaults(func=_suggest_document_pack_review)

    build_pack_review_workbook = sub.add_parser(
        "build-document-pack-review-workbook",
        help="Merge graph candidates and exact-match advice into a fillable priority workbook",
    )
    build_pack_review_workbook.add_argument("--candidate-pack", type=Path, required=True)
    build_pack_review_workbook.add_argument("--review", type=Path, required=True)
    build_pack_review_workbook.add_argument("--suggestions", type=Path, required=True)
    build_pack_review_workbook.add_argument(
        "--component-suggestions", type=Path, default=None
    )
    build_pack_review_workbook.add_argument("--output", type=Path, required=True)
    build_pack_review_workbook.set_defaults(func=_build_document_pack_review_workbook)

    suggest_component_archetypes = sub.add_parser(
        "suggest-document-component-archetypes",
        help="Suggest exact channel-map archetypes for deferred document components",
    )
    suggest_component_archetypes.add_argument("--review", type=Path, required=True)
    suggest_component_archetypes.add_argument("--suggestions", type=Path, required=True)
    suggest_component_archetypes.add_argument(
        "--channel-map-catalog", type=Path, required=True
    )
    suggest_component_archetypes.add_argument("--base-graph", type=Path, required=True)
    suggest_component_archetypes.add_argument("--output", type=Path, required=True)
    suggest_component_archetypes.set_defaults(
        func=_suggest_document_component_archetypes
    )

    compile_pack_review_workbook = sub.add_parser(
        "compile-document-pack-review-workbook",
        help="Validate a filled graph priority workbook and run the canonical review freeze",
    )
    compile_pack_review_workbook.add_argument("--candidate-pack", type=Path, required=True)
    compile_pack_review_workbook.add_argument("--workbook-root", type=Path, required=True)
    compile_pack_review_workbook.add_argument("--workbook", type=Path, required=True)
    compile_pack_review_workbook.add_argument("--output", type=Path, required=True)
    compile_pack_review_workbook.add_argument("--allow-incomplete", action="store_true")
    compile_pack_review_workbook.set_defaults(func=_compile_document_pack_review_workbook)

    audit_corpus_readiness = sub.add_parser(
        "audit-document-corpus-readiness",
        help="Fail closed against the ten P0.5 document-corpus acceptance gates",
    )
    audit_corpus_readiness.add_argument("--project-root", type=Path, default=Path("."))
    audit_corpus_readiness.add_argument("--evidence-spec", type=Path, required=True)
    audit_corpus_readiness.add_argument("--output", type=Path, required=True)
    audit_corpus_readiness.set_defaults(func=_audit_document_corpus_readiness)

    prepare_early_gold = sub.add_parser(
        "prepare-early-design-gold-review",
        help="Create locked human-gold workbooks for the first real design DOCX",
    )
    prepare_early_gold.add_argument("--inventory", type=Path, required=True)
    prepare_early_gold.add_argument("--epistemic-candidates", type=Path, required=True)
    prepare_early_gold.add_argument("--drift-candidates", type=Path, required=True)
    prepare_early_gold.add_argument("--review-template", type=Path, required=True)
    prepare_early_gold.add_argument("--project-root", type=Path, default=Path("."))
    prepare_early_gold.add_argument("--output", type=Path, required=True)
    prepare_early_gold.set_defaults(func=_prepare_early_design_gold_review)

    freeze_early_gold = sub.add_parser(
        "freeze-early-design-gold-review",
        help="Validate named early-design DOCX gold decisions and freeze their result",
    )
    freeze_early_gold.add_argument("--workbook", type=Path, required=True)
    freeze_early_gold.add_argument("--output", type=Path, required=True)
    freeze_early_gold.set_defaults(func=_freeze_early_design_gold_review)

    prepare_multimodal_gold = sub.add_parser(
        "prepare-multimodal-gold-review",
        help="Create a locked formula/table/image/epistemic parser-gold workbook",
    )
    prepare_multimodal_gold.add_argument("--source", type=Path, required=True)
    prepare_multimodal_gold.add_argument("--build-receipt", type=Path, required=True)
    prepare_multimodal_gold.add_argument("--machine-candidate", type=Path, required=True)
    prepare_multimodal_gold.add_argument("--review-template", type=Path, required=True)
    prepare_multimodal_gold.add_argument("--output", type=Path, required=True)
    prepare_multimodal_gold.set_defaults(func=_prepare_multimodal_gold_review)

    freeze_multimodal_gold = sub.add_parser(
        "freeze-multimodal-gold-review",
        help="Validate named parser-gold decisions against the current multimodal fixture",
    )
    freeze_multimodal_gold.add_argument("--source", type=Path, required=True)
    freeze_multimodal_gold.add_argument("--build-receipt", type=Path, required=True)
    freeze_multimodal_gold.add_argument("--machine-candidate", type=Path, required=True)
    freeze_multimodal_gold.add_argument("--review-template", type=Path, required=True)
    freeze_multimodal_gold.add_argument("--workbook", type=Path, required=True)
    freeze_multimodal_gold.add_argument("--output", type=Path, required=True)
    freeze_multimodal_gold.set_defaults(func=_freeze_multimodal_gold_review)

    prepare_chunking_gold = sub.add_parser(
        "prepare-chunking-gold-review",
        help="Create a locked named-human relevance workbook for both chunking systems",
    )
    prepare_chunking_gold.add_argument("--preregistration", type=Path, required=True)
    prepare_chunking_gold.add_argument("--machine-results", type=Path, required=True)
    prepare_chunking_gold.add_argument("--handbook", type=Path, required=True)
    prepare_chunking_gold.add_argument("--canonical-facts", type=Path, required=True)
    prepare_chunking_gold.add_argument("--protocol", type=Path, required=True)
    prepare_chunking_gold.add_argument("--output", type=Path, required=True)
    prepare_chunking_gold.set_defaults(func=_prepare_chunking_gold_review)

    freeze_chunking_gold = sub.add_parser(
        "freeze-chunking-gold-review",
        help="Freeze named relevance decisions and recompute human-gold retrieval metrics",
    )
    freeze_chunking_gold.add_argument("--preregistration", type=Path, required=True)
    freeze_chunking_gold.add_argument("--machine-results", type=Path, required=True)
    freeze_chunking_gold.add_argument("--handbook", type=Path, required=True)
    freeze_chunking_gold.add_argument("--canonical-facts", type=Path, required=True)
    freeze_chunking_gold.add_argument("--protocol", type=Path, required=True)
    freeze_chunking_gold.add_argument("--workbook", type=Path, required=True)
    freeze_chunking_gold.add_argument("--output", type=Path, required=True)
    freeze_chunking_gold.set_defaults(func=_freeze_chunking_gold_review)

    run_contextual_chunking = sub.add_parser(
        "run-contextual-chunking-confirmation",
        help="Run the prospectively frozen real-DOCX section-context confirmation",
    )
    run_contextual_chunking.add_argument("--preregistration", type=Path, required=True)
    run_contextual_chunking.add_argument("--ingestion", type=Path, required=True)
    run_contextual_chunking.add_argument("--output", type=Path, required=True)
    run_contextual_chunking.set_defaults(func=_run_contextual_chunking_confirmation)

    prepare_contextual_gold = sub.add_parser(
        "prepare-contextual-chunking-gold",
        help="Create a blinded human-gold workbook for contextual chunking confirmation",
    )
    prepare_contextual_gold.add_argument("--preregistration", type=Path, required=True)
    prepare_contextual_gold.add_argument("--ingestion", type=Path, required=True)
    prepare_contextual_gold.add_argument("--results", type=Path, required=True)
    prepare_contextual_gold.add_argument("--output", type=Path, required=True)
    prepare_contextual_gold.set_defaults(func=_prepare_contextual_chunking_gold)

    freeze_contextual_gold = sub.add_parser(
        "freeze-contextual-chunking-gold",
        help="Freeze named judgements for contextual chunking confirmation",
    )
    freeze_contextual_gold.add_argument("--preregistration", type=Path, required=True)
    freeze_contextual_gold.add_argument("--ingestion", type=Path, required=True)
    freeze_contextual_gold.add_argument("--results", type=Path, required=True)
    freeze_contextual_gold.add_argument("--workbook", type=Path, required=True)
    freeze_contextual_gold.add_argument("--output", type=Path, required=True)
    freeze_contextual_gold.set_defaults(func=_freeze_contextual_chunking_gold)

    compile_tbox_claims = sub.add_parser(
        "compile-approved-tbox-claims",
        help="Compile the frozen owner-approved TBox associations into scoped publishable claims",
    )
    compile_tbox_claims.add_argument("--graph", type=Path, required=True)
    compile_tbox_claims.add_argument("--construction-bundle", type=Path, required=True)
    compile_tbox_claims.add_argument("--approval", type=Path, required=True)
    compile_tbox_claims.add_argument("--output", type=Path, required=True)
    compile_tbox_claims.set_defaults(func=_compile_approved_tbox_claims)

    build_controlled_rewrites = sub.add_parser(
        "build-controlled-document-rewrites",
        help="Create source-bound LLM paraphrase candidates with deterministic structure gates",
    )
    build_controlled_rewrites.add_argument("--manifest", type=Path, required=True)
    build_controlled_rewrites.add_argument("--corpus-root", type=Path, required=True)
    build_controlled_rewrites.add_argument("--output", type=Path, required=True)
    build_controlled_rewrites.add_argument("--maximum-documents", type=int, default=None)
    build_controlled_rewrites.add_argument("--frozen-proposal-root", type=Path, default=None)
    build_controlled_rewrites.add_argument("--base-url", default="http://127.0.0.1:8000")
    build_controlled_rewrites.add_argument("--model", default="auto")
    build_controlled_rewrites.add_argument("--timeout-s", type=float, default=180.0)
    build_controlled_rewrites.add_argument("--max-tokens", type=int, default=4096)
    build_controlled_rewrites.set_defaults(func=_build_controlled_document_rewrites)

    freeze_controlled_rewrites = sub.add_parser(
        "freeze-controlled-document-rewrite-review",
        help="Freeze source-bound human semantic judgments for LLM rewrite candidates",
    )
    freeze_controlled_rewrites.add_argument("--rewrite-pack", type=Path, required=True)
    freeze_controlled_rewrites.add_argument("--review-sheet", type=Path, required=True)
    freeze_controlled_rewrites.add_argument("--output", type=Path, required=True)
    freeze_controlled_rewrites.add_argument("--allow-incomplete", action="store_true")
    freeze_controlled_rewrites.set_defaults(func=_freeze_controlled_document_rewrite_review)

    materialize_reviewed_rewrites = sub.add_parser(
        "materialize-reviewed-document-rewrites",
        help="Create a review-only RAG and candidate-KG pack from approved paraphrases",
    )
    materialize_reviewed_rewrites.add_argument(
        "--source-manifest", type=Path, required=True
    )
    materialize_reviewed_rewrites.add_argument(
        "--rewrite-pack", type=Path, required=True
    )
    materialize_reviewed_rewrites.add_argument(
        "--review-freeze", type=Path, required=True
    )
    materialize_reviewed_rewrites.add_argument("--output", type=Path, required=True)
    materialize_reviewed_rewrites.add_argument("--release-id", required=True)
    materialize_reviewed_rewrites.set_defaults(
        func=_materialize_reviewed_document_rewrites
    )

    suggest_controlled_rewrites = sub.add_parser(
        "suggest-controlled-document-rewrite-review",
        help="Generate non-binding semantic-risk suggestions beside a blank rewrite review",
    )
    suggest_controlled_rewrites.add_argument("--rewrite-pack", type=Path, required=True)
    suggest_controlled_rewrites.add_argument("--review-sheet", type=Path, required=True)
    suggest_controlled_rewrites.add_argument("--output", type=Path, required=True)
    suggest_controlled_rewrites.add_argument(
        "--frozen-suggestion-root", type=Path, default=None
    )
    suggest_controlled_rewrites.add_argument(
        "--base-url", default="http://127.0.0.1:8000"
    )
    suggest_controlled_rewrites.add_argument("--model", default="auto")
    suggest_controlled_rewrites.add_argument("--timeout-s", type=float, default=180.0)
    suggest_controlled_rewrites.add_argument("--max-tokens", type=int, default=4096)
    suggest_controlled_rewrites.set_defaults(
        func=_suggest_controlled_document_rewrite_review
    )

    compile_rewrite_workbook = sub.add_parser(
        "compile-controlled-document-rewrite-review-workbook",
        help="Validate a filled priority workbook and compile its canonical review freeze",
    )
    compile_rewrite_workbook.add_argument("--rewrite-pack", type=Path, required=True)
    compile_rewrite_workbook.add_argument("--suggestion-pack", type=Path, required=True)
    compile_rewrite_workbook.add_argument("--workbook", type=Path, required=True)
    compile_rewrite_workbook.add_argument("--output", type=Path, required=True)
    compile_rewrite_workbook.add_argument("--allow-incomplete", action="store_true")
    compile_rewrite_workbook.set_defaults(
        func=_compile_controlled_document_rewrite_review_workbook
    )

    freeze_document_review = sub.add_parser(
        "freeze-document-graph-review",
        help="Reconstruct and freeze human decisions for one document candidate package",
    )
    freeze_document_review.add_argument("--candidate-package", type=Path, required=True)
    freeze_document_review.add_argument("--review-sheet", type=Path, required=True)
    freeze_document_review.add_argument("--output", type=Path, required=True)
    freeze_document_review.add_argument("--allow-incomplete", action="store_true")
    freeze_document_review.set_defaults(func=_freeze_document_graph_review)

    freeze_pack_review = sub.add_parser(
        "freeze-document-pack-graph-review",
        help="Reconstruct and freeze decisions for a stable document candidate pack",
    )
    freeze_pack_review.add_argument("--candidate-pack", type=Path, required=True)
    freeze_pack_review.add_argument("--review-sheet", type=Path, required=True)
    freeze_pack_review.add_argument("--output", type=Path, required=True)
    freeze_pack_review.add_argument("--allow-incomplete", action="store_true")
    freeze_pack_review.set_defaults(func=_freeze_document_pack_graph_review)

    propose_document_patch = sub.add_parser(
        "propose-document-graph-patch",
        help="Create a source-bound, non-publishing V3 patch proposal from frozen review",
    )
    propose_document_patch.add_argument("--review-freeze", type=Path, required=True)
    propose_document_patch.add_argument("--base-graph", type=Path, required=True)
    propose_document_patch.add_argument("--output", type=Path, required=True)
    propose_document_patch.set_defaults(func=_propose_document_graph_patch)

    compile_document_patch = sub.add_parser(
        "compile-document-graph-patch",
        help="Compile a READY document V3 patch into an isolated replay-only graph",
    )
    compile_document_patch.add_argument("--patch-proposal", type=Path, required=True)
    compile_document_patch.add_argument("--base-graph", type=Path, required=True)
    compile_document_patch.add_argument("--output", type=Path, required=True)
    compile_document_patch.set_defaults(func=_compile_document_graph_patch)

    replay_document_patch = sub.add_parser(
        "replay-document-graph-patch",
        help="Compare declared causal topology before and after an isolated document patch",
    )
    replay_document_patch.add_argument("--base-graph", type=Path, required=True)
    replay_document_patch.add_argument("--candidate-graph", type=Path, required=True)
    replay_document_patch.add_argument(
        "--seed-node-id", action="append", required=True,
        help="Replay seed node ID; repeat for multiple seeds",
    )
    replay_document_patch.add_argument(
        "--target-node-type", action="append", required=True,
        help="Target schema-v3 node type; repeat for multiple types",
    )
    replay_document_patch.add_argument(
        "--expected-added-path-signature", action="append", default=[],
        help="Expected node/relation alternating path signature; repeat as needed",
    )
    replay_document_patch.add_argument("--output", type=Path, required=True)
    replay_document_patch.set_defaults(func=_replay_document_graph_patch)

    validate = sub.add_parser("validate-data", help="Validate PHM quick dataset protocol and isolation gates")
    validate.add_argument("--data-dir", type=Path, default=Path("data/input_quick"))
    validate.add_argument("--output", type=Path, default=Path("data/outputs/data_quality_report.json"))
    validate.set_defaults(func=_validate)

    diagnose = sub.add_parser("diagnose", help="Run one deterministic diagnosis case")
    diagnose.add_argument("--case", required=True)
    diagnose.add_argument("--request-id", default="")
    diagnose.add_argument("--fault-detection", type=Path, default=Path("data/input_quick/fault_detection.csv"))
    diagnose.add_argument("--edges", type=Path, default=None)
    diagnose.add_argument("--policy", type=Path, default=Path("configs/diagnosis.json"))
    diagnose.add_argument("--output", type=Path, default=Path("data/outputs"))
    diagnose.add_argument("--audit", type=Path, default=Path("data/audit/diagnostic_audit.jsonl"))
    diagnose.add_argument("--telemetry", type=Path, default=None, help="Unified production telemetry CSV")
    diagnose.add_argument("--nominal", type=Path, default=None, help="Optional paired nominal CSV")
    diagnose.add_argument("--backend", choices=["auto", "unified", "composite"], default="auto")
    diagnose.add_argument("--catalog", type=Path, default=None, help="Optional fault knowledge catalog")
    diagnose.add_argument("--catalog-model", type=Path, default=None, help="Explicit catalog model asset")
    diagnose.add_argument("--open-set-model", type=Path, default=None, help="Explicit open-set model asset")
    diagnose.add_argument("--output-json", type=Path, default=None, help="Unified result JSON path")
    diagnose.set_defaults(func=_diagnose)

    diagnose_batch = sub.add_parser("diagnose-batch", help="Run the unified diagnosis service for a frozen JSON manifest")
    diagnose_batch.add_argument("--manifest", type=Path, required=True)
    diagnose_batch.add_argument("--output", type=Path, required=True)
    diagnose_batch.set_defaults(func=_diagnose_unified_batch)

    generate_plugin = sub.add_parser(
        "generate-object-plugin",
        help="Generate a reviewable object-specific deterministic feature-service scaffold",
    )
    generate_plugin.add_argument("--spec", type=Path, required=True)
    generate_plugin.add_argument("--output-root", type=Path, required=True)
    generate_plugin.set_defaults(func=_generate_object_plugin)

    validate_plugin = sub.add_parser(
        "validate-object-plugin",
        help="Validate an Object Plugin Factory package without importing generated code",
    )
    validate_plugin.add_argument("--package-root", type=Path, required=True)
    validate_plugin.set_defaults(func=_validate_object_plugin)


    validate_model_manifest = sub.add_parser(
        "validate-data-model-manifest",
        help="Validate a model-family-neutral data-driven model manifest",
    )
    validate_model_manifest.add_argument("--manifest", type=Path, required=True)
    validate_model_manifest.set_defaults(func=_validate_data_model_manifest)

    audit_runs = sub.add_parser(
        "audit-independent-model-runs",
        help="Fail closed when model train/validation/test runs are not independently separated",
    )
    audit_runs.add_argument("--inventory", type=Path, required=True)
    audit_runs.add_argument("--output", type=Path, default=None)
    audit_runs.add_argument("--minimum-runs-per-class", type=int, default=3)
    audit_runs.add_argument(
        "--required-splits",
        default="TRAIN,VALIDATION,TEST",
        help="Comma-separated split names",
    )
    audit_runs.add_argument("--strict-provenance", action="store_true")
    audit_runs.add_argument("--allow-exact-content-duplicates", action="store_true")
    audit_runs.add_argument("--allow-group-leakage", action="store_true")
    audit_runs.set_defaults(func=_audit_independent_model_runs)

    plan_campaign = sub.add_parser(
        "plan-independent-run-campaign",
        help="Compile an explicit multi-seed, multi-variant independent-run simulation campaign",
    )
    plan_campaign.add_argument("--spec", type=Path, required=True)
    plan_campaign.add_argument("--output", type=Path, required=True)
    plan_campaign.set_defaults(func=_plan_independent_run_campaign)

    execute_campaign = sub.add_parser(
        "execute-independent-run-campaign",
        help="Execute a validated campaign using only the registered external runtime script",
    )
    execute_campaign.add_argument("--plan", type=Path, required=True)
    execute_campaign.add_argument("--runtime-root", type=Path, required=True)
    execute_campaign.add_argument("--output-root", type=Path, required=True)
    execute_campaign.add_argument("--report", type=Path, default=None)
    execute_campaign.add_argument("--continue-on-failure", action="store_true")
    execute_campaign.set_defaults(func=_execute_independent_run_campaign)

    inventory_campaign = sub.add_parser(
        "build-independent-run-inventory",
        help="Hash and inventory completed campaign artifacts for strict split/provenance audit",
    )
    inventory_campaign.add_argument("--plan", type=Path, required=True)
    inventory_campaign.add_argument("--output-root", type=Path, required=True)
    inventory_campaign.add_argument("--output", type=Path, required=True)
    inventory_campaign.set_defaults(func=_build_campaign_inventory)

    compile_graph = sub.add_parser(
        "compile-read-only-graph-query",
        help="Compile a GraphQueryPlan into parameterized, allow-listed read-only Cypher",
    )
    compile_graph.add_argument("--plan", type=Path, required=True)
    compile_graph.add_argument("--output", type=Path, required=True)
    compile_graph.add_argument(
        "--query-kind",
        choices=[item.value for item in GraphQueryKind],
        default=GraphQueryKind.AUTO.value,
    )
    compile_graph.add_argument("--knowledge-version", default="")
    compile_graph.set_defaults(func=_compile_read_only_graph_query)

    evaluate_benchmark = sub.add_parser(
        "evaluate-model-benchmark",
        help="Report model metrics only after a passing independent-run audit",
    )
    evaluate_benchmark.add_argument("--predictions", type=Path, required=True)
    evaluate_benchmark.add_argument("--inventory", type=Path, required=True)
    evaluate_benchmark.add_argument("--audit", type=Path, required=True)
    evaluate_benchmark.add_argument("--output", type=Path, required=True)
    evaluate_benchmark.add_argument(
        "--evaluated-split",
        choices=[item.value for item in SplitName],
        default=SplitName.TEST.value,
    )
    evaluate_benchmark.add_argument(
        "--known-class-ids",
        default="",
        help="Optional comma-separated registered known classes; other truths are treated as unknown/OOD",
    )
    evaluate_benchmark.add_argument(
        "--top-k",
        default="1,3,5",
        help="Comma-separated positive Top-K values",
    )
    evaluate_benchmark.set_defaults(func=_evaluate_model_benchmark)


    dataset_group = sub.add_parser(
        "dataset",
        help="Probe or structurally validate a local read-only dataset source",
    )
    dataset_sub = dataset_group.add_subparsers(dest="dataset_command", required=True)

    dataset_probe = dataset_sub.add_parser(
        "probe",
        help="Generate redacted schema/time/label/quality evidence without copying raw data",
    )
    dataset_probe.add_argument(
        "--source-type",
        choices=[
            DataSourceType.ESA_ADB.value,
            DataSourceType.SIMULATION.value,
            DataSourceType.SAT_AGENT_PLATFORM.value,
            DataSourceType.LOCAL_FILE.value,
        ],
        required=True,
    )
    dataset_probe.add_argument("--root", default="")
    dataset_probe.add_argument("--profile", type=Path, default=None)
    dataset_probe.add_argument("--timestamp-column", default="")
    dataset_probe.add_argument("--label-columns", default="")
    dataset_probe.add_argument(
        "--trust-local-pickle",
        action="store_true",
        help=(
            "Acknowledge that raw ESA-ADB ZIP archives contain trusted pandas "
            "pickle objects that may execute code while loading"
        ),
    )
    dataset_probe.add_argument("--output", type=Path, required=True)
    dataset_probe.add_argument("--max-entities", type=int, default=100)
    dataset_probe.add_argument("--max-rows", type=int, default=10000)
    dataset_probe.add_argument(
        "--hash-mode", choices=["none", "metadata", "full"], default="metadata"
    )
    dataset_probe.add_argument("--include-statistics", action="store_true")
    dataset_probe.add_argument("--include-label-values", action="store_true")
    dataset_probe.set_defaults(func=_dataset_probe)

    dataset_validate = dataset_sub.add_parser(
        "validate",
        help="Run bounded structural quality checks against a read-only dataset source",
    )
    dataset_validate.add_argument(
        "--source-type",
        choices=[
            DataSourceType.ESA_ADB.value,
            DataSourceType.SIMULATION.value,
            DataSourceType.SAT_AGENT_PLATFORM.value,
            DataSourceType.LOCAL_FILE.value,
        ],
        required=True,
    )
    dataset_validate.add_argument("--root", default="")
    dataset_validate.add_argument("--profile", type=Path, default=None)
    dataset_validate.add_argument("--timestamp-column", default="")
    dataset_validate.add_argument("--label-columns", default="")
    dataset_validate.add_argument(
        "--trust-local-pickle",
        action="store_true",
        help=(
            "Acknowledge that raw ESA-ADB ZIP archives contain trusted pandas "
            "pickle objects that may execute code while loading"
        ),
    )
    dataset_validate.add_argument("--output", type=Path, required=True)
    dataset_validate.add_argument("--max-entities", type=int, default=100)
    dataset_validate.add_argument("--max-rows", type=int, default=10000)
    dataset_validate.set_defaults(func=_dataset_validate_source)

    simulate_group = sub.add_parser(
        "simulate",
        help="Generate deterministic profile-compatible synthetic regression data",
    )
    simulate_sub = simulate_group.add_subparsers(dest="simulate_command", required=True)
    simulate_generate = simulate_sub.add_parser("generate")
    simulate_generate.add_argument("--spec", type=Path, required=True)
    simulate_generate.add_argument("--schema-profile", type=Path, default=None)
    simulate_generate.add_argument("--output-root", type=Path, required=True)
    simulate_generate.set_defaults(func=_simulate_generate_dataset)

    diagnosis_group = sub.add_parser(
        "diagnosis",
        help="Run/resume/inspect the data-source diagnosis workflow; legacy 'diagnose' remains supported",
    )
    diagnosis_sub = diagnosis_group.add_subparsers(dest="diagnosis_command", required=True)


    diagnosis_paired = diagnosis_sub.add_parser(
        "paired-sat-sim",
        help="Diagnose one label-isolated Sat-Agent nominal/effect Basilisk pair",
    )
    diagnosis_paired.add_argument("--case-id", required=True)
    diagnosis_paired.add_argument("--backend", choices=["unified", "composite"], required=True)
    diagnosis_paired.add_argument("--nominal", type=Path, required=True)
    diagnosis_paired.add_argument("--telemetry", type=Path, required=True)
    diagnosis_paired.add_argument("--output", type=Path, required=True)
    diagnosis_paired.set_defaults(func=_diagnosis_sat_sim_paired)


    diagnosis_multifault = diagnosis_sub.add_parser(
        "multifault-sat-sim",
        help="Diagnose a label-isolated Sat-Agent nominal/effect Basilisk pair with multiple effects",
    )
    diagnosis_multifault.add_argument("--case-id", required=True)
    diagnosis_multifault.add_argument("--nominal", type=Path, required=True)
    diagnosis_multifault.add_argument("--telemetry", type=Path, required=True)
    diagnosis_multifault.add_argument("--catalog", type=Path, default=None)
    diagnosis_multifault.add_argument("--output", type=Path, required=True)
    diagnosis_multifault.set_defaults(func=_diagnosis_sat_sim_multifault)


    diagnosis_multifault_single = diagnosis_sub.add_parser(
        "multifault-single-sat-sim",
        help="Diagnose one label-isolated Sat-Agent Basilisk trace containing zero, one, or multiple known effects",
    )
    diagnosis_multifault_single.add_argument("--case-id", required=True)
    diagnosis_multifault_single.add_argument("--telemetry", type=Path, required=True)
    diagnosis_multifault_single.add_argument("--catalog", type=Path, default=None)
    diagnosis_multifault_single.add_argument("--output", type=Path, required=True)
    diagnosis_multifault_single.set_defaults(func=_diagnosis_sat_sim_multifault_single)


    diagnosis_catalog = diagnosis_sub.add_parser(
        "catalog-sat-sim",
        help="Diagnose one label-isolated Sat-Agent Basilisk telemetry trace without a paired nominal run",
    )
    diagnosis_catalog.add_argument("--case-id", required=True)
    diagnosis_catalog.add_argument("--backend", choices=["unified", "composite"], required=True)
    diagnosis_catalog.add_argument("--model", type=Path, required=True)
    diagnosis_catalog.add_argument("--telemetry", type=Path, required=True)
    diagnosis_catalog.add_argument("--output", type=Path, required=True)
    diagnosis_catalog.set_defaults(func=_diagnosis_sat_sim_catalog)


    diagnosis_open_set = diagnosis_sub.add_parser(
        "open-set-sat-sim",
        help="Diagnose one Sat-Agent Basilisk trace with calibrated known-catalog rejection",
    )
    diagnosis_open_set.add_argument("--case-id", required=True)
    diagnosis_open_set.add_argument("--backend", required=True)
    diagnosis_open_set.add_argument("--model", type=Path, required=True)
    diagnosis_open_set.add_argument("--telemetry", type=Path, required=True)
    diagnosis_open_set.add_argument("--output", type=Path, required=True)
    diagnosis_open_set.set_defaults(func=_diagnosis_sat_sim_open_set)

    diagnosis_capabilities = diagnosis_sub.add_parser(
        "capabilities",
        help="List unified diagnosis engine capabilities and claim boundaries",
    )
    diagnosis_capabilities.add_argument("--output", type=Path, default=None)
    diagnosis_capabilities.set_defaults(func=_diagnosis_capabilities)

    diagnosis_run = diagnosis_sub.add_parser(
        "run",
        help="Run read-detect-retrieve-diagnose-report against a configured data source",
    )
    diagnosis_run.add_argument(
        "--source-type",
        choices=[
            DataSourceType.ESA_ADB.value,
            DataSourceType.SIMULATION.value,
            DataSourceType.SAT_AGENT_PLATFORM.value,
            DataSourceType.LOCAL_FILE.value,
        ],
        required=True,
    )
    diagnosis_run.add_argument("--root", default="")
    diagnosis_run.add_argument("--profile", type=Path, default=None)
    diagnosis_run.add_argument("--timestamp-column", default="")
    diagnosis_run.add_argument("--label-columns", default="")
    diagnosis_run.add_argument(
        "--trust-local-pickle",
        action="store_true",
        help=(
            "Acknowledge that raw ESA-ADB ZIP archives contain trusted pandas "
            "pickle objects that may execute code while loading"
        ),
    )
    diagnosis_run.add_argument("--entity", required=True)
    diagnosis_run.add_argument("--request-id", default="")
    diagnosis_run.add_argument("--start", default="")
    diagnosis_run.add_argument("--end", default="")
    diagnosis_run.add_argument("--channels", default="")
    diagnosis_run.add_argument("--signal-mapping", type=Path, default=None)
    diagnosis_run.add_argument(
        "--detector",
        choices=["robust-baseline", "rw4-single-fault", "rw4-single-fault-v2"],
        default="robust-baseline",
        help=(
            "Select the generic long-baseline detector, the sample-rate-aware RW4 v3 "
            "authority-loss detector (rw4-single-fault), or the frozen historical v2 detector"
        ),
    )
    diagnosis_run.add_argument("--baseline-fraction", type=float, default=0.2)
    diagnosis_run.add_argument("--z-threshold", type=float, default=6.0)
    diagnosis_run.add_argument("--minimum-baseline-points", type=int, default=10)
    diagnosis_run.add_argument("--edges", type=Path, required=True)
    diagnosis_run.add_argument("--policy", type=Path, default=Path("configs/diagnosis.json"))
    diagnosis_run.add_argument(
        "--state-root", type=Path, default=Path("data/diagnosis_runs")
    )
    diagnosis_run.add_argument(
        "--audit", type=Path, default=Path("data/audit/dataset_diagnosis.jsonl")
    )
    diagnosis_run.add_argument(
        "--output", type=Path, default=Path("local_evidence/dataset_diagnosis")
    )
    diagnosis_run.add_argument("--force", action="store_true")
    diagnosis_run.set_defaults(func=_diagnosis_dataset_run)

    diagnosis_resume = diagnosis_sub.add_parser(
        "resume",
        help="Resume or replay a saved data-source diagnosis run",
    )
    diagnosis_resume.add_argument("--run-id", required=True)
    diagnosis_resume.add_argument(
        "--source-type",
        choices=[
            DataSourceType.ESA_ADB.value,
            DataSourceType.SIMULATION.value,
            DataSourceType.SAT_AGENT_PLATFORM.value,
            DataSourceType.LOCAL_FILE.value,
        ],
        required=True,
    )
    diagnosis_resume.add_argument("--root", default="")
    diagnosis_resume.add_argument("--profile", type=Path, default=None)
    diagnosis_resume.add_argument("--timestamp-column", default="")
    diagnosis_resume.add_argument("--label-columns", default="")
    diagnosis_resume.add_argument(
        "--trust-local-pickle",
        action="store_true",
        help=(
            "Acknowledge that raw ESA-ADB ZIP archives contain trusted pandas "
            "pickle objects that may execute code while loading"
        ),
    )
    diagnosis_resume.add_argument(
        "--detector",
        choices=["auto", "robust-baseline", "rw4-single-fault", "rw4-single-fault-v2"],
        default="auto",
        help="Auto-detect the persisted detector or require an explicit detector",
    )
    diagnosis_resume.add_argument("--edges", type=Path, required=True)
    diagnosis_resume.add_argument("--policy", type=Path, default=Path("configs/diagnosis.json"))
    diagnosis_resume.add_argument(
        "--state-root", type=Path, default=Path("data/diagnosis_runs")
    )
    diagnosis_resume.add_argument(
        "--audit", type=Path, default=Path("data/audit/dataset_diagnosis.jsonl")
    )
    diagnosis_resume.add_argument(
        "--output", type=Path, default=Path("local_evidence/dataset_diagnosis")
    )
    diagnosis_resume.set_defaults(func=_diagnosis_dataset_resume)

    diagnosis_inspect = diagnosis_sub.add_parser(
        "inspect",
        help="Inspect persisted run state without opening the dataset",
    )
    diagnosis_inspect.add_argument("--run-id", required=True)
    diagnosis_inspect.add_argument(
        "--state-root", type=Path, default=Path("data/diagnosis_runs")
    )
    diagnosis_inspect.set_defaults(func=_diagnosis_dataset_inspect)

    trusted_data = sub.add_parser(
        "audit-trusted-data",
        help="Classify Round6 Basilisk runs, scan public traces for truth leakage, and audit reproducibility",
    )
    trusted_data.add_argument("--dataset-root", type=Path, required=True)
    trusted_data.add_argument("--sat-agent-root", type=Path, required=True)
    trusted_data.add_argument("--dataset-archive-sha256", default="")
    trusted_data.add_argument("--sat-agent-archive-sha256", default="")
    trusted_data.add_argument("--output-dir", type=Path, required=True)
    trusted_data.set_defaults(func=_audit_trusted_round6_data)

    governance_group = sub.add_parser(
        "governance",
        help="Validate frozen improvement contracts and formal gate evidence",
    )
    governance_sub = governance_group.add_subparsers(dest="governance_command", required=True)

    governance_validate = governance_sub.add_parser(
        "validate",
        help="Validate contract integrity, candidate version, frozen artifacts, deliverables, and change scope",
    )
    governance_validate.add_argument("--contract", type=Path, required=True)
    governance_validate.add_argument("--expected-contract-sha256", required=True)
    governance_validate.add_argument("--candidate-root", type=Path, default=Path("."))
    governance_validate.add_argument("--output", type=Path, default=None)
    governance_validate.set_defaults(func=_governance_validate)

    governance_decide = governance_sub.add_parser(
        "decide",
        help="Apply the frozen ACCEPT/REJECT/ESCALATE/BLOCKED policy to gate evidence",
    )
    governance_decide.add_argument("--contract", type=Path, required=True)
    governance_decide.add_argument("--expected-contract-sha256", required=True)
    governance_decide.add_argument("--gate-evidence", type=Path, required=True)
    governance_decide.add_argument("--evidence-root", type=Path, required=True)
    governance_decide.add_argument("--query-receipt", type=Path, required=True)
    governance_decide.add_argument("--expected-evaluator-sha256", required=True)
    governance_decide.add_argument("--expected-evaluator-module-sha256", required=True)
    governance_decide.add_argument("--output", type=Path, required=True)
    governance_decide.set_defaults(func=_governance_decide)

    governance_ledger = governance_sub.add_parser(
        "audit-ledger",
        help="Validate the sequence and hash chain of an append-only governance JSONL ledger",
    )
    governance_ledger.add_argument("--ledger", type=Path, required=True)
    governance_ledger.add_argument("--output", type=Path, default=None)
    governance_ledger.set_defaults(func=_governance_audit_ledger)

    problem = governance_sub.add_parser(
        "problem",
        help="Open, govern, inspect and resume hash-chained problem-stack records",
    )
    problem_sub = problem.add_subparsers(dest="problem_action", required=True)
    for action in (
        "open",
        "view",
        "triage",
        "retriage",
        "resolve",
        "verify",
        "close",
        "resume",
    ):
        action_parser = problem_sub.add_parser(action)
        action_parser.add_argument(
            "--events", type=Path, default=Path("governance/problem_stack/events.jsonl")
        )
        action_parser.add_argument(
            "--current", type=Path, default=Path("governance/problem_stack/current.json")
        )
        if action == "open":
            action_parser.add_argument("--record", type=Path, required=True)
        elif action != "view":
            action_parser.add_argument("--problem-id", required=True)
            action_parser.add_argument("--patch", type=Path, default=None)
        action_parser.set_defaults(func=_problem_stack)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
