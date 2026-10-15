#!/usr/bin/env python3
"""Grouped PHM v6.0 train/test regression through AstroGraph's diagnosis workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from astrograph.data_driven.phm_v60 import (
    audit_split, build_dataset, load_records, train_three_stage, train_three_stage_adaptive,
    train_three_stage_promoted_extra_trees, load_three_stage_promoted_bundle,
)

DEFAULT_DATASET = Path(os.environ.get("ASTROGRAPH_PHM_V60_DATA_ROOT", str(ROOT / "data" / "external" / "whole_spacecraft_phm_dataset_v6_0")))
DEFAULT_OUTPUT = ROOT / "scripts" / "integration" / "out" / "phm_v60_regression"

CONFIG_ROOT = ROOT / "configs" / "datasets" / "phm_v6_0"
HIERARCHY_PATH = CONFIG_ROOT / "diagnosis_hierarchy.json"
CLASS_MAP_PATH = CONFIG_ROOT / "fault_class_map.json"
EVIDENCE_PACKAGE = CONFIG_ROOT / "runtime_evidence.json"
DATASET_PROFILE_REGISTRY = ROOT / "configs" / "dataset_profiles" / "registry.json"
RUNTIME_POLICY_REGISTRY = ROOT / "configs" / "runtime_policies" / "registry.json"
KNOWLEDGE_COVERAGE = CONFIG_ROOT / "knowledge_coverage.json"
CANDIDATE_UNIVERSE = CONFIG_ROOT / "candidate_universe.json"
GENERIC_FAULT_ONTOLOGY = ROOT / "configs" / "knowledge" / "generic_fault_ontology.json"
FAULT_ONTOLOGY_BINDINGS = CONFIG_ROOT / "fault_ontology_bindings.json"
COMPONENT_ARCHETYPES = ROOT / "configs" / "semantics" / "core_component_archetypes.json"
SEMANTIC_SCHEMA = ROOT / "configs" / "semantics" / "core_semantic_schema.json"
PROTOCOL_ROOT = ROOT / "protocols" / "phm_project_a"
MODEL_POLICY_PATH = CONFIG_ROOT / "model_policy.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_prediction(
    item: dict[str, Any], class_map: dict[str, str], *, episode_id: str, model_strategy: str
):
    from astrograph.data_driven import (
        DataDrivenModelManifest,
        DataDrivenTaskType,
        ModelCapability,
        ModelInferenceRequest,
        ModelOutputSemantics,
        ModelRuntime,
        RawModelOutput,
        adapt_raw_model_output,
    )

    # This configured evidence package intentionally exposes only one runtime
    # candidate.  The other four strict global mappings remain valid catalog
    # mappings, but are outside this package's evidence/query universe.
    from astrograph.data_driven.candidate_mapping import map_runtime_fault_attribution

    universe = json.loads(CANDIDATE_UNIVERSE.read_text(encoding="utf-8"))
    mapping = map_runtime_fault_attribution(
        item["class_probabilities"], class_map, set(universe["runtime_fault_ids"])
    )
    mapped = mapping.mapped_attribution
    unsupported = mapping.unsupported_probabilities
    model_identity = {
        "promoted_extra_trees": (
            "phm-v60-extra-trees-retrospective-screening",
            "2026.09.14-dev13",
            "Promoted ExtraTrees-500 retrospective known-activation trace screening model; "
            "not an unknown-onset causal or flight/safety model.",
        ),
        "adaptive": (
            "phm-v60-adaptive-classical-regression",
            "2026.09.14-dev12",
            "Validation-selected retrospective known-activation trace screening model; "
            "not an unknown-onset causal or flight/safety model.",
        ),
        "legacy_lr": (
            "phm-v60-three-stage-logistic-regression",
            "2026.08.13",
            "Historical retrospective PHM three-LogisticRegression baseline.",
        ),
    }
    if model_strategy not in model_identity:
        raise ValueError(f"unsupported PHM model strategy identity: {model_strategy}")
    model_id, version, model_claim_boundary = model_identity[model_strategy]
    manifest = DataDrivenModelManifest(
        model_id=model_id,
        model_version=version,
        display_name=model_id,
        task_type=DataDrivenTaskType.ANOMALY_ATTRIBUTION,
        runtime=ModelRuntime.SCIKIT_LEARN,
        output_semantics=ModelOutputSemantics.ANOMALY_ATTRIBUTION,
        capabilities=[ModelCapability.FAULT_RANKING, ModelCapability.NORMAL_CLASS],
        supported_fault_ids=sorted(mapped),
        normal_class_id="NORMAL",
        feature_schema_ids=["phm-v60-post-activation-trace-summary-v1"],
        training_dataset_ids=["whole_spacecraft_phm_dataset_v6_0"],
        claim_boundary=(
            model_claim_boundary + " Only exact ontology mappings feed BiRRF; "
            "probabilities are uncalibrated and are not flight evidence."
        ),
    )
    digest = hashlib.sha256(
        json.dumps(item, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    request = ModelInferenceRequest(
        episode_id=episode_id,
        model_id=model_id,
        model_version=version,
        feature_schema_id="phm-v60-post-activation-trace-summary-v1",
        input_data_sha256=digest,
        registered_fault_ids=sorted(mapped),
    )
    from astrograph.knowledge import load_dataset_fault_bindings, load_generic_fault_ontology
    ontology = load_generic_fault_ontology(GENERIC_FAULT_ONTOLOGY)
    bindings = load_dataset_fault_bindings(FAULT_ONTOLOGY_BINDINGS, ontology)
    binding_by_class = {value.fault_class_id: value for value in bindings.bindings}
    abstract_binding = binding_by_class[item["predicted_class"]]
    raw = RawModelOutput(
        episode_id=episode_id,
        model_id=model_id,
        model_version=version,
        output_semantics=ModelOutputSemantics.ANOMALY_ATTRIBUTION,
        anomaly_score=float(np.clip(item["detection_score"], 0.0, 1.0)),
        fault_attribution=mapped,
        out_of_distribution_score=mapping.unsupported_mass,
        metadata={
            "sample_id": item["sample_id"],
            "predicted_component": item["predicted_component"],
            "predicted_dataset_class": item["predicted_class"],
            "abstract_failure_mechanism_id": abstract_binding.mechanism_id,
            "abstract_expected_effect_pattern_ids": abstract_binding.expected_effect_pattern_ids,
            "abstract_ontology_version": ontology.version,
            "offline_component_probabilities": item["component_probabilities"],
            "offline_dataset_class_probabilities": item["class_probabilities"],
            "unsupported_class_mass": unsupported,
        },
    )
    return adapt_raw_model_output(manifest, request, raw)


def run_chain(
    *, dataset_root: Path, sample_id: str, prediction: Any | None, output_dir: Path,
    reasoner: str, vllm_url: str, vllm_model: str,
    model_registry: Any | None = None, inference_request: Any | None = None,
) -> dict[str, Any]:
    from astrograph.evidence import ConfiguredRuntimeEvidenceBuilder
    from astrograph.dataset_profiles import DatasetProfileRegistry
    from astrograph.policy_profiles import RuntimePolicyRegistry
    from astrograph.llm.openai_compatible import OpenAICompatibleConfig, OpenAICompatibleStructuredBackend
    from astrograph.orchestration.models import ExecutionPolicy, PermissionScope, TaskEnvelope, WorkflowContext
    from astrograph.orchestration.whole_spacecraft_online_workflow import WholeSpacecraftOnlineDiagnosisWorkflow
    from astrograph.simulation import SimulationRunCompleted

    records = load_records(dataset_root)
    sample = next(row for row in records if row.sample_id == sample_id)
    nominal = next(
        row for row in records
        if row.sample_index == sample.sample_index
        and row.subsystem == sample.subsystem
        and row.condition_id == "nominal"
    )
    run_dir = output_dir / "run"
    nominal_dir = output_dir / "nominal"
    artifact_dir = output_dir / "artifacts"
    for path in (run_dir, nominal_dir, artifact_dir):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
    shutil.copy2(sample.telemetry_path, run_dir / "telemetry.csv")
    shutil.copy2(nominal.telemetry_path, nominal_dir / "telemetry.csv")
    source = pd.read_csv(sample.telemetry_path, low_memory=False)
    manifest = {
        "schema_version": "astrograph-phm-v60-runtime.v1",
        "row_count": len(source),
        "field_count": len(source.columns),
        "context_of_use": "E0 synthetic PHM v6.0 grouped regression",
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    event = SimulationRunCompleted(
        event_id=f"PHM-V60-{reasoner}",
        run_id=f"phm-v60-{reasoner}",
        run_dir=str(run_dir),
        nominal_run_dir=str(nominal_dir),
        emitted_at="2026-08-13T00:00:00Z",
        metadata={"source": "whole_spacecraft_phm_dataset_v6_0", "split": "test"},
    )
    bundle, gate = ConfiguredRuntimeEvidenceBuilder(EVIDENCE_PACKAGE).build(event)
    if (prediction is None) == (inference_request is None):
        raise ValueError("provide exactly one of prediction or inference_request")
    if prediction is not None:
        from astrograph.data_driven.adapters import rebind_prediction_episode
        prediction = rebind_prediction_episode(prediction, bundle.episode.episode_id)
        data_input = {"data_driven_prediction": prediction.model_dump(mode="json")}
    else:
        inference_request = inference_request.model_copy(
            update={"episode_id": bundle.episode.episode_id}
        )
        data_input = {
            "data_driven_inference_request": inference_request.model_dump(mode="json")
        }
    permissions = [
        PermissionScope.READ_DATA,
        PermissionScope.READ_KNOWLEDGE,
        PermissionScope.RUN_DIAGNOSIS,
        PermissionScope.WRITE_AUDIT,
        PermissionScope.WRITE_ARTIFACTS,
    ]
    envelope = TaskEnvelope(
        task_id=f"PHM-V60-{reasoner}",
        task_type="simulation.online_diagnosis",
        input_payload={
            "simulation_event": event.model_dump(mode="json"),
            "dataset_profile_id": "phm.v6_0",
            "artifact_output_dir": str(artifact_dir),
            "shadow_mode": True,
            "reasoner_mode": reasoner,
            "open_set_policy": {
                "minimum_top_probability": 0.0,
                "minimum_top_margin": 0.0,
                "maximum_ood_score": json.loads(
                    CANDIDATE_UNIVERSE.read_text(encoding="utf-8")
                )["maximum_unsupported_mass_for_fusion"],
                "suppress_data_branch_on_reject": True,
            },
            **data_input,
        },
        execution_policy=ExecutionPolicy(deterministic_only=reasoner == "rule", allow_llm=reasoner == "llm"),
        permission_scope=permissions,
    )
    context = WorkflowContext(
        workflow_id=WholeSpacecraftOnlineDiagnosisWorkflow.workflow_id,
        run_id=f"phm-v60-{reasoner}",
        permissions=permissions,
    )
    backend = None
    if reasoner == "llm":
        backend = OpenAICompatibleStructuredBackend(OpenAICompatibleConfig(
            base_url=vllm_url,
            provider="vllm",
            model=vllm_model,
            max_tokens=900,
            timeout_s=180,
            prompt_profile="evidence_protocol_v059_compact",
            candidate_top_k=2,
            tokenizer_mode="disabled",
            context_length=16384,
        ))
    result = WholeSpacecraftOnlineDiagnosisWorkflow(
        reasoning_backend=backend,
        dataset_profile_registry=DatasetProfileRegistry(index_path=DATASET_PROFILE_REGISTRY),
        runtime_policy_registry=RuntimePolicyRegistry(RUNTIME_POLICY_REGISTRY),
        data_model_registry=model_registry,
    ).invoke(envelope, context)
    payload = result.model_dump(mode="json")
    payload["configured_evidence_gate"] = gate.model_dump(mode="json")
    payload["configured_feature_results"] = [row.model_dump(mode="json") for row in bundle.feature_results]
    payload["configured_evidence_facts"] = [row.model_dump(mode="json") for row in bundle.evidence_facts]
    return payload


def render_markdown(report: dict[str, Any]) -> str:
    result = report["three_stage_result"]
    chain_rows = []
    for name, chain in report["chains"].items():
        decision = chain.get("output_payload", {}).get("decision_record", {})
        chain_rows.append(
            f"| {name} | {chain.get('status')} | {chain.get('code')} | "
            f"{decision.get('disposition')} | {decision.get('accepted_fault_id')} |"
        )
    return f"""# PHM v6.0 全链路回归报告

性质：单次固定组划分的 E0 合成数据工程回归，不是模型选型或飞行能力声明。

## 数据与模型

- Train/Validation/Test：336/112/112 条轨迹；`sample_index` 组数 6/2/2；组交集 {report['split_audit']['group_overlap_count']}。
- 特征：故障激活后的原始遥测 trace，逐通道 mean/std/min/max/delta/rms；标签、fault、truth、effective 等答案代理字段被排除。
- 数据驱动策略：{result.get('model')}
- Validation 选择回执：`{json.dumps(result.get('selection', {}), ensure_ascii=False)[:1200]}`
- Detection Macro-F1：{result['detection']['macro_f1']:.4f}
- Component localization Macro-F1：{result['localization']['macro_f1']:.4f}
- 17-class oracle-unconditioned Macro-F1：{result['classification_oracle_unconditioned']['macro_f1']:.4f}
- 预测部件路由后的 17-class Macro-F1：{result['classification_predicted_component']['macro_f1']:.4f}
- 端到端 18-class Macro-F1：{result['end_to_end_18_class']['macro_f1']:.4f}

本实验是 trace-level screening，不是未知故障时刻的在线滑窗验证；其结果不能与 ADCS 20 秒在线因果窗直接作公平性能排名。

## 特征构造与知识链路

- 数据驱动特征独立于知识证据：前者服务检测/定位/分类，后者由版本化配置把原始通道映射为语义通道，再计算配对残差证据节点。
- 回放对象：`{report['selected_chain_sample']}`。
- 知识证据：加热器开启状态均值配对残差与加热功率平均绝对配对残差；不读取 labels。
- 严格 KG 映射覆盖 5/17 类；12 类保留为 unsupported，不允许近似改名。

| 链路 | 状态 | 代码 | 决策 | 接受故障 |
|---|---|---|---|---|
{chr(10).join(chain_rows)}

## 组件卡、比较与实验边界

- 组件卡：输入为 post-activation trace summary；局部组件为统计聚合；当前默认回归使用已晋升的 ExtraTrees-500 回顾性筛查级联；保留 dev12 adaptive 与 legacy LR 作为回退对照。限制仍是丢失时间顺序且依赖已知 activation boundary。
- 比较矩阵：三阶段共享同一组划分与 Train-only 预处理；oracle 分类仅是误差上界，正式口径使用预测部件路由。
- 架构方案：Telemetry → trace summary → Detection → Component Localization → component-conditioned Classification → 公共预测协议；并行 Telemetry + paired nominal → configured evidence → KG/BiRRF → Governor。
- 实验计划：本轮覆盖单划分训练/测试和 rule/LLM 链路；未覆盖五折、多种子、校准、未知时刻在线延迟和真实航天器泛化。
- 技术报告扫描：没有提出新算子或新颖性声明。
- 组件登记：登记为项目级 regression bundle，不写入通用神经组件注册表。

## 技能资源追踪

使用 `neural-component-architecture-design` 的 core principles、component ontology、research/design/learning protocol、full design principles 和六份模板；它们约束了信息边界、比较公平性、声明边界，以及组件卡/矩阵/架构/实验/扫描/登记六类交付。
"""


def main() -> int:
    model_policy = json.loads(MODEL_POLICY_PATH.read_text(encoding="utf-8"))
    default_strategy = str(model_policy["default_replay_strategy"])
    allowed_strategies = [str(x) for x in model_policy["allowed_replay_strategies"]]
    promoted_model_dir = ROOT / str(model_policy["promoted_bundle_path"])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8005/v1")
    parser.add_argument("--vllm-model", default="Qwen3.5-9B")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument(
        "--promoted-model-dir", type=Path, default=promoted_model_dir,
        help="frozen dev13 promoted model bundle used by promoted_extra_trees",
    )
    parser.add_argument(
        "--retrain-promoted", action="store_true",
        help="retrain the promoted ExtraTrees family from the frozen Train split instead of loading the bundle",
    )
    parser.add_argument(
        "--model-strategy", choices=allowed_strategies, default=default_strategy,
        help=("promoted_extra_trees is the dev13 retrospective sat/PHM default; adaptive preserves "
              "the dev12 Validation-selected pool; legacy_lr preserves the historical three-LR baseline"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hierarchy = json.loads(HIERARCHY_PATH.read_text(encoding="utf-8"))
    from astrograph.knowledge import GraphShapeValidator, project_generic_fault_ontology
    abstract_graph_report = GraphShapeValidator().validate(project_generic_fault_ontology(
        GENERIC_FAULT_ONTOLOGY, COMPONENT_ARCHETYPES, SEMANTIC_SCHEMA
    ))
    if abstract_graph_report.status != "PASS":
        raise RuntimeError(abstract_graph_report.model_dump(mode="json"))
    class_map = json.loads(CLASS_MAP_PATH.read_text(encoding="utf-8"))["mappings"]
    records = load_records(args.dataset_root)
    split_audit = audit_split(records)
    if split_audit["status"] != "PASS":
        raise RuntimeError(f"split audit failed: {split_audit}")
    dataset = build_dataset(args.dataset_root, hierarchy["fault_to_component"], progress=True)
    if args.model_strategy == "promoted_extra_trees":
        result = (
            train_three_stage_promoted_extra_trees(dataset, hierarchy["fault_to_component"])
            if args.retrain_promoted
            else load_three_stage_promoted_bundle(
                dataset, args.promoted_model_dir, dataset_root=args.dataset_root
            )
        )
    elif args.model_strategy == "adaptive":
        result = train_three_stage_adaptive(dataset, hierarchy["fault_to_component"])
    else:
        result = train_three_stage(dataset, hierarchy["fault_to_component"])
    models = result.pop("models")
    model_dir = args.output_dir / "models"
    model_dir.mkdir(exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, model_dir / f"{name}.joblib")

    selected = next(
        item for item in result["test_predictions"]
        if item["sample_id"].startswith("thermal:active_settled:thermal_heater_stuck_on:")
    )
    prediction = make_prediction(
        selected, class_map, episode_id="preflight-phm-v60", model_strategy=args.model_strategy
    )
    chains = {
        "machine_learning_rule": run_chain(
            dataset_root=args.dataset_root,
            sample_id=selected["sample_id"],
            prediction=prediction,
            output_dir=args.output_dir / "chains" / "rule",
            reasoner="rule",
            vllm_url=args.vllm_url,
            vllm_model=args.vllm_model,
        )
    }
    if not args.skip_llm:
        chains["machine_learning_llm"] = run_chain(
            dataset_root=args.dataset_root,
            sample_id=selected["sample_id"],
            prediction=prediction,
            output_dir=args.output_dir / "chains" / "llm",
            reasoner="llm",
            vllm_url=args.vllm_url,
            vllm_model=args.vllm_model,
        )

    import sklearn
    report = {
        "schema_version": "astrograph-phm-v60-regression-report.v1",
        "generated_at": "2026-08-13T00:00:00+08:00",
        "dataset_root": str(args.dataset_root.resolve()),
        "split_audit": split_audit,
        "model_strategy": args.model_strategy,
        "feature_protocol": {
            "scope": "post_activation_trace_summary",
            "statistics": ["mean", "std", "min", "max", "delta", "rms"],
            "preprocessing_fit": "train_only",
            "excluded_answer_proxy_tokens": [
                "fault", "condition", "label", "active_effect", "event", "remaining_ratio",
                "capacity", "leak_rate", "load_multiplier", "effective", "truth",
            ],
        },
        "three_stage_result": result,
        "model_artifacts": {name: str((model_dir / f"{name}.joblib").resolve()) for name in models},
        "diagnosis_hierarchy": {"path": str(HIERARCHY_PATH), "sha256": sha256(HIERARCHY_PATH)},
        "fault_class_map": {"path": str(CLASS_MAP_PATH), "sha256": sha256(CLASS_MAP_PATH)},
        "runtime_evidence_package": {"path": str(EVIDENCE_PACKAGE), "sha256": sha256(EVIDENCE_PACKAGE)},
        "knowledge_coverage": json.loads(KNOWLEDGE_COVERAGE.read_text(encoding="utf-8")),
        "abstract_fault_ontology": {
            "path": str(GENERIC_FAULT_ONTOLOGY),
            "binding_path": str(FAULT_ONTOLOGY_BINDINGS),
            "shape_report": abstract_graph_report.model_dump(mode="json"),
        },
        "selected_chain_sample": selected["sample_id"],
        "selected_prediction": prediction.model_dump(mode="json"),
        "chains": chains,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "vllm_url": args.vllm_url,
            "vllm_model": args.vllm_model,
        },
        "claim_boundary": (
            "Single fixed grouped split on E0 synthetic traces; trace-level screening only, not "
            "online unknown-onset, calibration, safety, production, or flight evidence."
        ),
    }
    json_path = args.output_dir / "phm_v60_regression_report.json"
    md_path = args.output_dir / "phm_v60_regression_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"[report] {json_path}")
    print(f"[report] {md_path}")
    return 0 if all(str(value.get("status")) in {"SUCCEEDED", "completed"} for value in chains.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
