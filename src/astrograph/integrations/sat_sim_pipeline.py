"""Governed AstroGraph consumer for Satellite Simulation Agent pipeline handoffs.

This module closes the first cross-project vertical slice without weakening either
project's governance boundary. It accepts the data-only simulator handoff,
executes a registered *engineering screening* provider, projects candidate causal
rankings from the supplied mapping asset, runs BiRRF, and renders a review-only
report.  It does not claim a trained or calibrated fault classifier, does not write
to Neo4j, and cannot publish a formal diagnosis.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.assurance import (
    AdmissionSigner,
    FeatureContract,
    FeatureContractRegistry,
    FeatureValueKind,
    InputBinding,
    ModelAdmissionManifest,
    canonical_sha256,
)
from astrograph.assurance.artifact_paths import ArtifactRoot
from astrograph.data_driven import (
    CallableRawModelProvider,
    DataDrivenModelManifest,
    DataDrivenModelRegistry,
    DataDrivenTaskType,
    FaultModelBindingRegistry,
    ModelCapability,
    ModelInferenceRequest,
    ModelOutputSemantics,
    ObservationScope,
    ModelRuntime,
    ModelSelectionRequest,
    ModelSelectionStatus,
    RawModelOutput,
)
from astrograph.evidence.executable import stable_id
from astrograph.evaluation.score_semantics import audit_score_semantics
from astrograph.integrations.rw_jam_promotion import RWJamPromotionGate, RWJamPromotionRequest
from astrograph.fusion import (
    BiRRFConfig,
    FaultRanking,
    RankedFault,
    RankingSource,
    ScoreSemantics,
    fuse_rankings,
)

SAT_SIM_PIPELINE_CLOSURE_SCHEMA_VERSION = "astrograph-sat-sim-pipeline-closure.v1"
_ALLOWED_RUNTIME_SCHEMA = "sat-sim.diagnostic-pipeline-runtime.v1"
_ALLOWED_MAPPING_SCHEMA = "sat-sim.astrograph-diagnostic-mapping-bundle.v1"
_RW_FAULT_IDS = [
    "ADCS_RW_FRICTION_INCREASE",
    "ADCS_RW_JAM",
    "ADCS_RW_TORQUE_AUTHORITY_LOSS",
]
_RW_SCREENING_FEATURE_SCHEMA_ID = "sat-sim.rw-screening-projection.v1"
_RW_SCREENING_FEATURES = (
    "raw.rw_speed.paired_rms",
    "raw.rw_actual_torque.paired_rms",
    "raw.pointing_error.paired_rms",
)


def _sha(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _feature(features: Mapping[str, Any], name: str) -> float:
    if name not in features:
        raise ValueError(f"required screening feature is missing: {name}")
    try:
        numeric = float(features[name])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"screening feature must be numeric: {name}") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"screening feature must be finite: {name}")
    return numeric



def _rw_screening_predict(request: ModelInferenceRequest) -> RawModelOutput:
    """Deterministic engineering screening model for the SAT-SIM vertical slice.

    The rules consume leakage-safe feature-vector values only.  They are not fitted,
    calibrated, or approved for a formal binding.  Their purpose is to prove the
    Model Registry -> common adapter -> BiRRF execution contract.
    """

    features = request.payload.get("features")
    if not isinstance(features, Mapping):
        raise ValueError("sat-sim screening provider requires payload.features")
    if set(features) != set(_RW_SCREENING_FEATURES):
        raise ValueError("sat-sim screening provider received an unadmitted feature set")

    speed = abs(_feature(features, "raw.rw_speed.paired_rms"))
    torque = abs(_feature(features, "raw.rw_actual_torque.paired_rms"))
    pointing = abs(_feature(features, "raw.pointing_error.paired_rms"))

    # Distinct engineering response bands observed in the proxy contract replay.
    # Constants are declared screening thresholds, not learned parameters.
    jam_strength = _clamp((speed - 2.0) / 18.0)
    # The friction response is a *band-pass* candidate signal, not evidence that a
    # zero residual is friction.  Gate the Gaussian by a small observed-deviation
    # floor so the all-zero feature vector remains nominal rather than receiving a
    # high score merely because log1p(0) is numerically close to the broad band.
    friction_presence = _clamp(speed / 0.05)
    friction_band = friction_presence * math.exp(
        -((math.log1p(speed) - math.log1p(0.30)) / 0.85) ** 2
    )
    response_strength = _clamp(torque / 8.0e-4) * 0.55 + _clamp(pointing / 1.0e-3) * 0.45
    torque_loss_strength = _clamp(response_strength * (1.0 - _clamp(speed / 2.0)))
    normal_strength = _clamp(1.0 - max(_clamp(speed / 0.25), response_strength))

    raw_scores = {
        "ADCS_RW_JAM": 3.5 * jam_strength + 1.0 * response_strength,
        "ADCS_RW_FRICTION_INCREASE": 2.8 * friction_band + 0.3 * _clamp(speed / 1.0),
        "ADCS_RW_TORQUE_AUTHORITY_LOSS": 1.4 * torque_loss_strength,
        "NORMAL": 2.0 * normal_strength + 0.05,
    }
    return RawModelOutput(
        episode_id=request.episode_id,
        model_id=request.model_id,
        model_version=request.model_version or "0.1.0-candidate",
        output_semantics=ModelOutputSemantics.NONNEGATIVE_SCORES,
        class_values=raw_scores,
        uncertainty=None,
        metadata={
            "provider": "rw-engineering-screening-provider",
            "screening_thresholds_not_learned": True,
            "formal_model_binding_enabled": False,
            "score_semantics": "ENGINEERING_NONNEGATIVE_SCORES_NORMALIZED_FOR_RANKING",
            "probability_claim_allowed": False,
            "input_feature_names": sorted(str(key) for key in features),
        },
    )


def build_rw_screening_registry() -> tuple[DataDrivenModelRegistry, FaultModelBindingRegistry]:
    manifest = DataDrivenModelManifest(
        model_id="rw-engineering-screening",
        model_version="0.1.0-candidate",
        display_name="RW engineering screening candidate",
        task_type=DataDrivenTaskType.MULTICLASS_FAULT_CLASSIFICATION,
        runtime=ModelRuntime.CUSTOM_CALLABLE,
        output_semantics=ModelOutputSemantics.NONNEGATIVE_SCORES,
        capabilities=[ModelCapability.FAULT_RANKING, ModelCapability.NORMAL_CLASS],
        supported_fault_ids=list(_RW_FAULT_IDS),
        feature_schema_ids=[
            _RW_SCREENING_FEATURE_SCHEMA_ID,
            "adcs.rw_friction_increase.features.v1",
            "adcs.rw_jam.features.v1",
            "adcs.rw_torque_authority_loss.features.v1",
        ],
        input_contract={
            "source": "sat-sim.diagnostic-feature-vector.v1",
            "feature_order": list(_RW_SCREENING_FEATURES),
            "simulator_truth_forbidden": True,
            "missing_features_are_rejected": True,
        },
        training_dataset_ids=[],
        claim_boundary=(
            "Deterministic engineering screening only. This provider is not trained, calibrated, "
            "benchmark-approved, or eligible for formal fault-model binding."
        ),
        metadata={"phase": "3G", "candidate_only": True},
    )
    contract = FeatureContract(
        contract_id="astrograph.sat-sim.rw-screening.features.v1",
        feature_schema_id=_RW_SCREENING_FEATURE_SCHEMA_ID,
        bindings=tuple(
            InputBinding(
                binding_id=f"rw-screening:{name}",
                payload_path=("features", name),
                value_kind=FeatureValueKind.NUMBER,
                source_fields=(name,),
                semantic_signal_ids=(name,),
                transform_chain=("sat-sim.paired-rms", "rw-screening-projection"),
            )
            for name in _RW_SCREENING_FEATURES
        ),
        require_observation_scope=True,
        require_input_hash_match=True,
    )
    model_source_sha256 = hashlib.sha256(
        inspect.getsource(_rw_screening_predict).encode("utf-8")
    ).hexdigest()
    provider = CallableRawModelProvider(
        manifest=manifest,
        predict_fn=_rw_screening_predict,
        requires_admission=True,
        model_artifact_sha256=model_source_sha256,
    )
    admission = ModelAdmissionManifest(
        admission_id="astrograph.sat-sim.rw-screening.admission.v1",
        model_id=manifest.model_id,
        model_version=manifest.model_version,
        provider_id=provider.provider_id,
        model_artifact_sha256=model_source_sha256,
        data_model_manifest_sha256=canonical_sha256(manifest.model_dump(mode="json")),
        feature_contract_id=contract.contract_id,
        feature_contract_sha256=contract.content_sha256,
        supported_fault_ids=tuple(manifest.supported_fault_ids),
        normal_class_id=manifest.normal_class_id,
    )
    contracts = FeatureContractRegistry()
    contracts.register_contract(contract)
    contracts.register_admission(admission)
    models = DataDrivenModelRegistry(
        admission_registry=contracts,
        admission_signer=AdmissionSigner.from_environment_or_ephemeral(),
        require_admission=True,
    )
    models.register(provider)
    return models, FaultModelBindingRegistry()


class SatSimPipelineClosureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runtime: dict[str, Any]
    mapping_bundle: dict[str, Any]
    requested_fault_id: str | None = None
    persist_dir: str | None = None
    birrf_config: BiRRFConfig = Field(default_factory=BiRRFConfig)
    promotion_evidence: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_cross_project_identity(self) -> "SatSimPipelineClosureRequest":
        if self.runtime.get("schema_version") != _ALLOWED_RUNTIME_SCHEMA:
            raise ValueError("unsupported Satellite Simulation runtime schema")
        mapping_schema = str(self.mapping_bundle.get("schema_version") or "")
        if mapping_schema != _ALLOWED_MAPPING_SCHEMA:
            raise ValueError("unsupported Satellite Simulation mapping schema")
        runtime_fault = str(self.runtime.get("fault_id") or "")
        if self.requested_fault_id and self.requested_fault_id != runtime_fault:
            raise ValueError("requested fault does not match runtime fault")
        if runtime_fault not in set(self.mapping_bundle.get("fault_ids") or []):
            raise ValueError("runtime fault is not present in mapping bundle")
        local_outputs = self.runtime.get("local_outputs") or {}
        for output_name in (
            "physics_validation",
            "diagnostic_validation",
            "feature_vector",
            "simulation_evidence_graph",
        ):
            output = local_outputs.get(output_name) or {}
            local_fault = str(output.get("fault_id") or "")
            if local_fault and local_fault != runtime_fault:
                raise ValueError(
                    f"runtime/local output fault identity mismatch: {output_name}"
                )
        return self


class SatSimPipelineClosureResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SAT_SIM_PIPELINE_CLOSURE_SCHEMA_VERSION
    closure_id: str
    fault_id: str
    episode_id: str
    disposition: str
    model_selection: dict[str, Any]
    data_driven_prediction: dict[str, Any]
    kg_projection: dict[str, Any]
    fusion: dict[str, Any]
    causal_chain: list[dict[str, Any]]
    blockers: list[str]
    external_stage_outputs: dict[str, Any]
    report_markdown: str
    formal_diagnosis_publishable: bool = False
    formal_model_binding_enabled: bool = False
    training_ready: bool = False
    claim_boundary: dict[str, Any]
    score_semantics: str = "SCREENING_SCORE"
    promotion_assessment: dict[str, Any]
    score_semantics_audit: dict[str, Any]
    closure_sha256: str = ""


class SatSimPipelineClosureWorkflow:
    workflow_id = "astrograph.sat-sim.pipeline-closure.v1"
    workflow_version = "0.6.6-repair-b003"

    def __init__(
        self,
        *,
        model_registry: DataDrivenModelRegistry | None = None,
        binding_registry: FaultModelBindingRegistry | None = None,
    ) -> None:
        default_models, default_bindings = build_rw_screening_registry()
        self.models = model_registry or default_models
        self.bindings = binding_registry or default_bindings

    @staticmethod
    def _mapping_for(bundle: Mapping[str, Any], fault_id: str) -> dict[str, Any]:
        for item in bundle.get("mappings") or []:
            if str(item.get("fault_id")) == fault_id:
                return dict(item)
        raise KeyError(f"mapping not found for {fault_id}")

    @staticmethod
    def _kg_rankings(
        *,
        episode_id: str,
        fault_id: str,
        mapping: Mapping[str, Any],
        runtime: Mapping[str, Any],
    ) -> tuple[FaultRanking, FaultRanking, dict[str, Any]]:
        """Build label-conditioned audit projections and neutral fusion inputs.

        The Sat-Sim runtime executes one fault-specific diagnostic signature chosen
        from simulator experiment metadata.  That identity is legitimate for
        offline review, but it is target leakage for candidate generation.  The
        projection is therefore retained in the audit payload and deliberately
        replaced by empty KG rankings before BiRRF.
        """

        mechanism = mapping.get("mechanism") or {}
        diagnostic = (runtime.get("local_outputs") or {}).get("diagnostic_validation") or {}
        score = _clamp(float(diagnostic.get("diagnostic_score") or 0.0))
        other = [item for item in _RW_FAULT_IDS if item != fault_id]
        forward_order = [fault_id, *other]
        backward_order = [fault_id, *other]
        mechanism_evidence = []
        mechanism_id = str(mechanism.get("mechanism_id") or "").strip()
        if mechanism_id:
            mechanism_evidence.append(f"mechanism:{mechanism_id}")
        mechanism_evidence.extend(
            f"propagation:{stage_id}"
            for item in mechanism.get("propagation_chain") or []
            if (stage_id := str(item.get("stage_id") or "").strip())
        )
        criterion_evidence = [
            f"criterion:{item.get('criterion_id', '')}"
            for item in diagnostic.get("criteria") or []
            if item.get("status") == "PASS" and item.get("contributes_to_diagnostic_score")
        ]
        forward_candidates: list[RankedFault] = []
        backward_candidates: list[RankedFault] = []
        for index, item in enumerate(forward_order, start=1):
            supported = index == 1 and bool(mechanism_evidence)
            forward_candidates.append(
                RankedFault(
                    fault_id=item,
                    rank=index,
                    score=1.0 if supported else 0.0,
                    source=RankingSource.KG_FORWARD,
                    evidence_fact_ids=mechanism_evidence if supported else [],
                    source_evidence_ids=[],
                    correlation_group_ids=[],
                    lineage_complete=False,
                    contributes_to_fusion=False,
                    metadata={
                        "graph_query_backed": False,
                        "candidate_mapping_projection": True,
                        "label_conditioned_audit_only": True,
                        "fusion_eligible": False,
                    },
                )
            )
        for index, item in enumerate(backward_order, start=1):
            supported = index == 1 and score > 0.0 and bool(criterion_evidence)
            backward_candidates.append(
                RankedFault(
                    fault_id=item,
                    rank=index,
                    score=score if supported else 0.0,
                    source=RankingSource.KG_BACKWARD,
                    evidence_fact_ids=criterion_evidence if supported else [],
                    source_evidence_ids=[],
                    correlation_group_ids=[],
                    lineage_complete=False,
                    contributes_to_fusion=False,
                    metadata={
                        "diagnostic_signature_projection": True,
                        "label_conditioned_audit_only": True,
                        "fusion_eligible": False,
                    },
                )
            )
        audit_forward = FaultRanking(
            ranking_id=stable_id("RANK", {"episode": episode_id, "source": "mapping-forward", "fault": fault_id}),
            episode_id=episode_id,
            source=RankingSource.KG_FORWARD,
            score_semantics=ScoreSemantics.EVIDENCE_STRENGTH,
            candidates=forward_candidates,
            producer_id="sat-sim-mechanism-mapping-projection",
            producer_version="sat-sim-pipeline-v2",
            metadata={
                "graph_query_backed": False,
                "neo4j_write_performed": False,
                "label_conditioned_audit_only": True,
                "fusion_eligible": False,
            },
        )
        audit_backward = FaultRanking(
            ranking_id=stable_id("RANK", {"episode": episode_id, "source": "signature-backward", "fault": fault_id}),
            episode_id=episode_id,
            source=RankingSource.KG_BACKWARD,
            score_semantics=ScoreSemantics.EVIDENCE_COVERAGE,
            candidates=backward_candidates,
            producer_id="sat-sim-signature-evidence-projection",
            producer_version="sat-sim-pipeline-v2",
            metadata={
                "diagnostic_score": score,
                "formal_kg_reasoning": False,
                "label_conditioned_audit_only": True,
                "fusion_eligible": False,
            },
        )
        forward = audit_forward.model_copy(
            update={
                "ranking_id": stable_id(
                    "RANK",
                    {
                        "episode": episode_id,
                        "source": "mapping-forward-fusion-neutral",
                    },
                ),
                "candidates": [],
                "metadata": {
                    **audit_forward.metadata,
                    "excluded_from_fusion_reason": "LABEL_CONDITIONED_RUNTIME_CONTEXT",
                },
            }
        )
        backward = audit_backward.model_copy(
            update={
                "ranking_id": stable_id(
                    "RANK",
                    {
                        "episode": episode_id,
                        "source": "signature-backward-fusion-neutral",
                    },
                ),
                "candidates": [],
                "metadata": {
                    **audit_backward.metadata,
                    "excluded_from_fusion_reason": "LABEL_CONDITIONED_RUNTIME_CONTEXT",
                },
            }
        )
        kg_output = {
            "schema_version": "astrograph-sat-sim-kg-projection.v1",
            "status": "LABEL_CONDITIONED_AUDIT_ONLY",
            "fault_id": fault_id,
            "mechanism_id": mechanism.get("mechanism_id"),
            "causal_chain": mechanism.get("propagation_chain") or [],
            "forward_ranking": audit_forward.model_dump(mode="json"),
            "backward_ranking": audit_backward.model_dump(mode="json"),
            "fusion_input_forward_ranking": forward.model_dump(mode="json"),
            "fusion_input_backward_ranking": backward.model_dump(mode="json"),
            "label_conditioned_runtime_context": True,
            "contributes_to_fusion": False,
            "neo4j_query_executed": False,
            "neo4j_write_performed": False,
            "claim_boundary": (
                "The runtime selected this fault-specific mapping from simulator experiment identity. "
                "It is retained for audit only and is excluded from BiRRF candidate generation."
            ),
        }
        return forward, backward, kg_output


    @staticmethod
    def _report(*, fault_id: str, runtime: Mapping[str, Any], prediction: Mapping[str, Any], fusion: Mapping[str, Any], causal_chain: list[dict[str, Any]], blockers: list[str]) -> str:
        classes = prediction.get("classes") or []
        lines = [
            "# AstroGraph × Sat-Sim 首次端到端候选诊断报告",
            "",
            f"- 目标故障：`{fault_id}`",
            f"- 仿真 Pipeline 状态：`{runtime.get('execution_status')}`",
            f"- 最终处置：**REVIEW_REQUIRED**",
            f"- 融合首选：`{fusion.get('top_fault_id')}`",
            "- 分值语义：`SCREENING_SCORE`（工程筛查分值，不是诊断置信度）",
            "- 正式发布：`false`",
            "",
            "## 一、候选模型输出",
            "",
            "| 类别 | 工程候选分值 | 排名 |",
            "|---|---:|---:|",
        ]
        for item in classes:
            lines.append(f"| `{item.get('class_id')}` | {float(item.get('probability', 0.0)):.6f} | {item.get('rank')} |")
        lines.extend(["", "## 二、因果传播链", ""])
        for item in causal_chain:
            lines.append(f"- `{item.get('from')}` —{item.get('relation')}→ `{item.get('to')}`")
        lines.extend(["", "## 三、融合结果", ""])
        for item in fusion.get("candidates") or []:
            lines.append(f"- `{item.get('fault_id')}`：rank={item.get('final_rank')}，BiRRF={float(item.get('fused_score', 0.0)):.8f}")
        lines.extend(["", "## 四、阻塞项", ""])
        lines.extend([f"- {item}" for item in blockers] or ["- 无"])
        lines.extend([
            "",
            "## 五、结论边界",
            "",
            "本报告只证明跨项目数据合同、候选模型运行、映射投影、BiRRF融合和报告生成链可以执行。",
            "候选模型未训练、未校准、未通过独立Basilisk基准；映射投影不是正式Neo4j推理；具名专家尚未审核。",
        ])
        return "\n".join(lines) + "\n"

    def invoke(self, request: SatSimPipelineClosureRequest) -> SatSimPipelineClosureResult:
        runtime = request.runtime
        fault_id = str(runtime["fault_id"])
        mapping = self._mapping_for(request.mapping_bundle, fault_id)
        feature_output = ((runtime.get("local_outputs") or {}).get("feature_vector") or {})
        features = feature_output.get("feature_vector") or {}
        if not isinstance(features, Mapping) or not features:
            raise ValueError("Satellite Simulation runtime does not contain a feature vector")
        leakage = feature_output.get("leakage_guard") or {}
        if not bool(leakage.get("simulator_truth_excluded")):
            raise ValueError("simulator truth leakage guard is not asserted")

        episode_id = str(runtime.get("pair_id") or runtime.get("fault_dataset_id") or stable_id("EP", runtime))
        feature_schema_id = str((mapping.get("feature_contract") or {}).get("contract_id") or "")
        selection = self.bindings.select(
            ModelSelectionRequest(
                candidate_fault_ids=list(_RW_FAULT_IDS),
                feature_schema_id=feature_schema_id,
                allow_partial_fault_coverage=False,
                allow_unbound_fallback=True,
            ),
            self.models,
        )
        if selection.status != ModelSelectionStatus.SELECTED:
            raise RuntimeError(f"no screening model selected: {selection.status.value}")
        projected_features = {name: _feature(features, name) for name in _RW_SCREENING_FEATURES}
        projected_payload = {"features": projected_features}
        projected_sha256 = _sha(projected_payload)
        inference = ModelInferenceRequest(
            episode_id=episode_id,
            model_id=str(selection.selected_model_id),
            model_version=selection.selected_model_version,
            feature_schema_id=_RW_SCREENING_FEATURE_SCHEMA_ID,
            input_data_sha256=projected_sha256,
            registered_fault_ids=list(_RW_FAULT_IDS),
            payload=projected_payload,
            observation_scope=ObservationScope(
                scope_id=f"sat-sim:{episode_id}:rw-screening",
                source_id="sat-sim.diagnostic-feature-vector.v1",
                source_data_sha256=str(
                    feature_output.get("feature_vector_sha256") or _sha(features)
                ),
                trajectory_group_id=str(runtime.get("pair_id") or "") or None,
            ),
            metadata={},
        )
        prediction = self.models.predict(inference)
        promotion_request = RWJamPromotionRequest.model_validate(request.promotion_evidence or {})
        promotion_assessment = RWJamPromotionGate().assess(promotion_request)
        data_ranking = prediction.to_fault_ranking(list(_RW_FAULT_IDS))
        forward, backward, kg_output = self._kg_rankings(
            episode_id=episode_id,
            fault_id=fault_id,
            mapping=mapping,
            runtime=runtime,
        )
        fused = fuse_rankings(
            data_ranking,
            forward,
            backward,
            config=request.birrf_config,
            normal_probability=prediction.normal_probability,
            normal_rank=prediction.normal_rank,
        )
        blockers = []
        if runtime.get("execution_status") != "READY_FOR_ASTROGRAPH_HANDOFF":
            blockers.append(f"SAT_SIM_RUNTIME_STATUS:{runtime.get('execution_status')}")
        blockers.extend(f"MISSING_REQUIRED_CHANNEL:{item}" for item in ((runtime.get("astrograph_handoff") or {}).get("missing_required_model_channels") or []))
        physics = (runtime.get("local_outputs") or {}).get("physics_validation") or {}
        if not bool(physics.get("formal_training_qualified")):
            blockers.append("BASILISK_A_LEVEL_AND_EXPERT_QUALIFICATION_MISSING")
        blockers.extend(
            [
                "FORMAL_MODEL_BINDING_DISABLED",
                "FORMAL_NEO4J_QUERY_NOT_EXECUTED",
                "LABEL_CONDITIONED_KG_PROJECTION_EXCLUDED_FROM_FUSION",
                "NAMED_EXPERT_REVIEW_PENDING",
            ]
        )
        causal_chain = list((mapping.get("mechanism") or {}).get("propagation_chain") or [])
        fusion_payload = fused.model_dump(mode="json")
        prediction_payload = prediction.model_dump(mode="json")
        manifest = self.models.get(prediction.model_id, prediction.model_version).manifest
        score_audit = audit_score_semantics(
            prediction=prediction, manifest=manifest, ranking=data_ranking, fusion=fused
        ).model_dump(mode="json")
        external = {
            "ml_model": {
                "status": "CANDIDATE_SCREENING_ONLY",
                "prediction": prediction_payload,
                "score_semantics": "SCREENING_SCORE",
                "selection": selection.model_dump(mode="json"),
            },
            "kg_reasoner": kg_output,
            "fusion": {
                "status": "REVIEW_REQUIRED",
                "birrf_result": fusion_payload,
                "formal_diagnosis_publishable": False,
            },
        }
        report = self._report(
            fault_id=fault_id,
            runtime=runtime,
            prediction=prediction_payload,
            fusion=fusion_payload,
            causal_chain=causal_chain,
            blockers=blockers,
        )
        payload = {
            "schema_version": SAT_SIM_PIPELINE_CLOSURE_SCHEMA_VERSION,
            "closure_id": stable_id("SAT-SIM-CLOSURE", {"runtime": runtime.get("runtime_sha256"), "fault": fault_id}),
            "fault_id": fault_id,
            "episode_id": episode_id,
            "disposition": "REVIEW_REQUIRED",
            "model_selection": selection.model_dump(mode="json"),
            "data_driven_prediction": prediction_payload,
            "kg_projection": kg_output,
            "fusion": fusion_payload,
            "causal_chain": causal_chain,
            "blockers": sorted(set(blockers)),
            "external_stage_outputs": external,
            "report_markdown": report,
            "formal_diagnosis_publishable": False,
            "formal_model_binding_enabled": False,
            "training_ready": False,
            "score_semantics": "SCREENING_SCORE",
            "promotion_assessment": promotion_assessment.model_dump(mode="json"),
            "score_semantics_audit": score_audit,
            "claim_boundary": {
                "engineering_vertical_slice_only": True,
                "candidate_model_not_trained_or_calibrated": True,
                "neo4j_projection_not_formal_query": True,
                "basilisk_a_level_required": True,
                "named_expert_required": True,
            },
            "closure_sha256": "",
        }
        payload["closure_sha256"] = _sha({key: value for key, value in payload.items() if key != "closure_sha256"})
        result = SatSimPipelineClosureResult.model_validate(payload)
        if request.persist_dir:
            source_root = Path(__file__).resolve().parents[3]
            artifacts = ArtifactRoot(request.persist_dir, source_root=source_root)
            artifacts.write_json(
                "sat_sim_pipeline_closure.json", result.model_dump(mode="json")
            )
            artifacts.write_text("sat_sim_pipeline_report.md", result.report_markdown)
            artifacts.write_json(
                "sat_sim_external_stage_outputs.json", result.external_stage_outputs
            )
        return result


__all__ = [
    "SAT_SIM_PIPELINE_CLOSURE_SCHEMA_VERSION",
    "SatSimPipelineClosureRequest",
    "SatSimPipelineClosureResult",
    "SatSimPipelineClosureWorkflow",
    "build_rw_screening_registry",
]
