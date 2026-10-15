"""Promotion gates for the RW_JAM diagnostic vertical slice.

The contracts in this module deliberately separate an executable candidate closure
from a publishable diagnostic capability.  No single score, model artifact, graph
query, or expert review can promote the capability by itself.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from enum import Enum
from typing import Any, Iterable, Mapping, Protocol

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from astrograph.data_driven import (
    BenchmarkReleaseStatus,
    BenchmarkStatus,
    CredibilityStatus,
    DataDrivenModelManifest,
    ModelBenchmarkReport,
    ModelCapability,
    ProbabilityCalibrationReceipt,
    ProbabilityCalibrationQualificationReceipt,
    SplitName,
)
from astrograph.schemas import KnowledgeEdge

RW_JAM_FAULT_ID = "ADCS_RW_JAM"
RW_JAM_PROMOTION_SCHEMA_VERSION = "astrograph-rw-jam-promotion-assessment.v1"


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class PromotionStatus(str, Enum):
    BLOCKED = "BLOCKED"
    TECHNICAL_CANDIDATE = "TECHNICAL_CANDIDATE"
    FORMAL_READY = "FORMAL_READY"


class ReviewMode(str, Enum):
    HUMAN = "HUMAN"
    LLM_ASSISTED_HUMAN = "LLM_ASSISTED_HUMAN"
    LLM_ONLY = "LLM_ONLY"


class ALevelRunEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    pair_id: str = Field(min_length=1)
    split: SplitName
    fault_id: str = RW_JAM_FAULT_ID
    simulator_engine: str = "Basilisk"
    simulator_version: str = Field(min_length=1)
    fidelity_level: str = "A"
    provenance_type: str = "SIMULATION_NATIVE"
    independence_group_id: str = Field(min_length=1)
    formal_training_qualified: bool = False
    named_expert_approved: bool = False
    observable_channels: list[str] = Field(default_factory=list)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("observable_channels")
    @classmethod
    def unique_channels(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("observable_channels must be unique")
        return value


class ActuatorTelemetryContractEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wheel_inertia_kg_m2: float = Field(gt=0.0)
    motor_torque_constant_nm_per_a: float = Field(gt=0.0)
    current_limit_a: float = Field(gt=0.0)
    sample_period_s: float = Field(gt=0.0)
    command_actual_time_aligned: bool = False
    saturation_semantics_frozen: bool = False
    contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ALevelDatasetEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_manifest_id: str = Field(min_length=1)
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runs: list[ALevelRunEvidence] = Field(default_factory=list)
    actuator_contract: ActuatorTelemetryContractEvidence | None = None
    split_frozen: bool = False
    leakage_audit_passed: bool = False


class ModelPromotionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest: DataDrivenModelManifest
    benchmark_report: ModelBenchmarkReport
    benchmark_release_status: BenchmarkReleaseStatus
    benchmark_scope_id: str = Field(min_length=1)
    validation_expected_calibration_error: float = Field(ge=0.0)
    calibration_receipt: ProbabilityCalibrationReceipt | ProbabilityCalibrationQualificationReceipt | None = None
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reproducible_training: bool = False
    evaluation_task_id: str | None = None
    label_semantics_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    target_universe_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class KnowledgePromotionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: str
    knowledge_version: str = Field(min_length=1)
    health_status: str
    read_only_query_executed: bool = False
    query_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    matched_fault_ids: list[str] = Field(default_factory=list)
    evidence_paths: list[dict[str, Any]] = Field(
        default_factory=list,
        validation_alias=AliasChoices("evidence_paths", "causal_paths"),
        serialization_alias="evidence_paths",
    )

    @property
    def causal_paths(self) -> list[dict[str, Any]]:
        """Deprecated compatibility view; values are associative evidence paths, not causal effects."""
        return self.evidence_paths
    formal_release_accepted: bool = False


class ExpertPromotionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviewer_id: str = Field(min_length=1)
    reviewer_role: str = Field(min_length=1)
    review_mode: ReviewMode
    approved: bool = False
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    knowledge_query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RWJamPromotionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minimum_train_runs: int = Field(default=30, ge=1)
    minimum_validation_runs: int = Field(default=10, ge=1)
    minimum_test_runs: int = Field(default=10, ge=1)
    minimum_test_recall: float = Field(default=0.90, ge=0.0, le=1.0)
    minimum_test_f1: float = Field(default=0.90, ge=0.0, le=1.0)
    maximum_validation_ece: float = Field(default=0.10, ge=0.0, le=1.0)
    maximum_external_validation_ece: float = Field(default=0.10, ge=0.0, le=1.0)
    minimum_external_calibration_groups: int = Field(default=10, ge=1)
    required_observable_channels: list[str] = Field(
        default_factory=lambda: [
            "sensor.adcs.rw.speed_rad_s_0",
            "command.adcs.rw.motor_torque_nm_0",
            "estimate.adcs.rw.actual_torque_nm_0",
            "sensor.adcs.rw.motor_current_a_0",
            "estimate.adcs.pointing_error_deg",
        ]
    )


class RWJamPromotionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: ALevelDatasetEvidence | None = None
    model: ModelPromotionEvidence | None = None
    knowledge: KnowledgePromotionEvidence | None = None
    expert: ExpertPromotionEvidence | None = None
    policy: RWJamPromotionPolicy = Field(default_factory=RWJamPromotionPolicy)


class RWJamPromotionAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = RW_JAM_PROMOTION_SCHEMA_VERSION
    status: PromotionStatus
    fault_id: str = RW_JAM_FAULT_ID
    dataset_gate_passed: bool
    model_gate_passed: bool
    knowledge_gate_passed: bool
    expert_gate_passed: bool
    score_semantics: str
    blockers: list[str]
    warnings: list[str]
    evidence_summary: dict[str, Any]
    assessment_sha256: str = ""


class RWJamPromotionGate:
    def assess(self, request: RWJamPromotionRequest) -> RWJamPromotionAssessment:
        blockers: list[str] = []
        warnings: list[str] = []
        policy = request.policy

        dataset_ok = self._dataset_gate(request.dataset, policy, blockers, warnings)
        model_ok = self._model_gate(request.model, request.dataset, policy, blockers, warnings)
        knowledge_ok = self._knowledge_gate(request.knowledge, blockers)
        expert_ok = self._expert_gate(request.expert, request.dataset, request.model, request.knowledge, blockers)

        technical = dataset_ok and model_ok and knowledge_ok
        formal = technical and expert_ok
        status = PromotionStatus.FORMAL_READY if formal else (
            PromotionStatus.TECHNICAL_CANDIDATE if technical else PromotionStatus.BLOCKED
        )
        payload = {
            "schema_version": RW_JAM_PROMOTION_SCHEMA_VERSION,
            "status": status.value,
            "fault_id": RW_JAM_FAULT_ID,
            "dataset_gate_passed": dataset_ok,
            "model_gate_passed": model_ok,
            "knowledge_gate_passed": knowledge_ok,
            "expert_gate_passed": expert_ok,
            "score_semantics": "DIAGNOSTIC_CONFIDENCE" if formal else "SCREENING_SCORE",
            "blockers": sorted(set(blockers)),
            "warnings": sorted(set(warnings)),
            "evidence_summary": self._summary(request),
            "assessment_sha256": "",
        }
        payload["assessment_sha256"] = _sha({k: v for k, v in payload.items() if k != "assessment_sha256"})
        return RWJamPromotionAssessment.model_validate(payload)

    @staticmethod
    def _dataset_gate(
        dataset: ALevelDatasetEvidence | None,
        policy: RWJamPromotionPolicy,
        blockers: list[str],
        warnings: list[str],
    ) -> bool:
        if dataset is None:
            blockers.append("A_LEVEL_DATASET_EVIDENCE_MISSING")
            return False
        counts = Counter(run.split for run in dataset.runs)
        required_counts = {
            SplitName.TRAIN: policy.minimum_train_runs,
            SplitName.VALIDATION: policy.minimum_validation_runs,
            SplitName.TEST: policy.minimum_test_runs,
        }
        for split, minimum in required_counts.items():
            if counts[split] < minimum:
                blockers.append(f"INSUFFICIENT_{split.value}_RUNS:{counts[split]}<{minimum}")
        groups = [run.independence_group_id for run in dataset.runs]
        if len(groups) != len(set(groups)):
            blockers.append("INDEPENDENCE_GROUP_REUSED")
        for run in dataset.runs:
            if run.fault_id != RW_JAM_FAULT_ID:
                blockers.append("DATASET_CONTAINS_WRONG_FAULT")
            if run.simulator_engine.lower() != "basilisk" or run.fidelity_level != "A":
                blockers.append("NON_BASILISK_A_LEVEL_RUN_PRESENT")
            if run.provenance_type != "SIMULATION_NATIVE":
                blockers.append("NON_NATIVE_SIMULATION_PROVENANCE")
            if not run.formal_training_qualified or not run.named_expert_approved:
                blockers.append("RUN_NOT_FORMALLY_QUALIFIED")
            missing = sorted(set(policy.required_observable_channels) - set(run.observable_channels))
            blockers.extend(f"RUN_MISSING_REQUIRED_CHANNEL:{item}" for item in missing)
        if not dataset.split_frozen:
            blockers.append("DATASET_SPLIT_NOT_FROZEN")
        if not dataset.leakage_audit_passed:
            blockers.append("TARGET_LEAKAGE_AUDIT_NOT_PASS")
        contract = dataset.actuator_contract
        if contract is None:
            blockers.append("ACTUATOR_TELEMETRY_CONTRACT_MISSING")
        else:
            if not contract.command_actual_time_aligned:
                blockers.append("COMMAND_ACTUAL_TIME_ALIGNMENT_NOT_FROZEN")
            if not contract.saturation_semantics_frozen:
                blockers.append("SATURATION_SEMANTICS_NOT_FROZEN")
        return not any(
            item.startswith(("INSUFFICIENT_", "INDEPENDENCE_", "DATASET_", "NON_", "RUN_", "TARGET_", "ACTUATOR_", "COMMAND_", "SATURATION_"))
            for item in blockers
        )

    @staticmethod
    def _model_gate(
        model: ModelPromotionEvidence | None,
        dataset: ALevelDatasetEvidence | None,
        policy: RWJamPromotionPolicy,
        blockers: list[str],
        warnings: list[str],
    ) -> bool:
        if model is None:
            blockers.append("QUALIFIED_MODEL_EVIDENCE_MISSING")
            return False
        manifest = model.manifest
        report = model.benchmark_report
        if RW_JAM_FAULT_ID not in manifest.supported_fault_ids:
            blockers.append("MODEL_DOES_NOT_SUPPORT_RW_JAM")
        if ModelCapability.CALIBRATED_PROBABILITY not in manifest.capabilities or not manifest.calibration_id:
            blockers.append("MODEL_CALIBRATION_CAPABILITY_MISSING")
        receipt = model.calibration_receipt
        if receipt is None or manifest.calibration_receipt_sha256 is None:
            blockers.append("MODEL_CALIBRATION_RECEIPT_MISSING")
        else:
            if receipt.receipt_sha256 != manifest.calibration_receipt_sha256:
                blockers.append("MODEL_CALIBRATION_RECEIPT_HASH_MISMATCH")
            if receipt.calibration_id != manifest.calibration_id:
                blockers.append("MODEL_CALIBRATION_RECEIPT_ID_MISMATCH")
            if (receipt.model_id, receipt.model_version) != (manifest.model_id, manifest.model_version):
                blockers.append("MODEL_CALIBRATION_RECEIPT_MODEL_MISMATCH")
            if abs(receipt.validation_expected_calibration_error - model.validation_expected_calibration_error) > 1e-12:
                blockers.append("MODEL_CALIBRATION_METRIC_NOT_RECEIPT_BOUND")
            if not isinstance(receipt, ProbabilityCalibrationQualificationReceipt):
                blockers.append("MODEL_EXTERNAL_CALIBRATION_QUALIFICATION_MISSING")
            else:
                if receipt.evaluation_task_id is not None or model.evaluation_task_id is not None or model.label_semantics_sha256 is not None or model.target_universe_sha256 is not None:
                    try:
                        receipt.require_external_task({"evaluation_task_id":model.evaluation_task_id,"label_semantics_sha256":model.label_semantics_sha256,"target_universe_sha256":model.target_universe_sha256})
                    except ValueError:
                        blockers.append("CROSS_TASK_CALIBRATION_REUSE_BLOCKED")
                if receipt.qualification_status != "PASS_FOR_ENGINEERING_CALIBRATION_REVIEW":
                    blockers.append("MODEL_EXTERNAL_CALIBRATION_QUALIFICATION_NOT_PASS")
                if receipt.model_artifact_sha256 != model.artifact_sha256:
                    blockers.append("MODEL_CALIBRATION_RECEIPT_ARTIFACT_MISMATCH")
                if receipt.external_validation_group_count < policy.minimum_external_calibration_groups:
                    blockers.append("MODEL_EXTERNAL_CALIBRATION_GROUPS_BELOW_POLICY")
                if receipt.external_validation_expected_calibration_error > policy.maximum_external_validation_ece:
                    blockers.append("MODEL_EXTERNAL_CALIBRATION_ERROR_ABOVE_POLICY")
        if not manifest.training_dataset_ids:
            blockers.append("MODEL_TRAINING_DATASET_LINEAGE_MISSING")
        if dataset and dataset.dataset_manifest_id not in manifest.training_dataset_ids:
            blockers.append("MODEL_NOT_TRAINED_ON_QUALIFIED_DATASET")
        if report.status != BenchmarkStatus.PASS or report.run_audit_status != CredibilityStatus.PASS:
            blockers.append("MODEL_BENCHMARK_NOT_PASS")
        if report.evaluated_split != SplitName.TEST:
            blockers.append("MODEL_NOT_EVALUATED_ON_HELD_OUT_TEST")
        if model.benchmark_release_status != BenchmarkReleaseStatus.ACCEPTED:
            blockers.append("MODEL_BENCHMARK_RELEASE_NOT_ACCEPTED")
        if not model.reproducible_training:
            blockers.append("MODEL_TRAINING_NOT_REPRODUCIBLE")
        admitted_ece = (
            receipt.validation_expected_calibration_error
            if receipt is not None and manifest.calibration_receipt_sha256 is not None
            else model.validation_expected_calibration_error
        )
        if admitted_ece > policy.maximum_validation_ece:
            blockers.append("MODEL_CALIBRATION_ERROR_ABOVE_POLICY")
        metric = None if report.metrics is None else next(
            (item for item in report.metrics.per_class if item.class_id == RW_JAM_FAULT_ID), None
        )
        if metric is None:
            blockers.append("RW_JAM_TEST_METRIC_MISSING")
        else:
            if metric.support < policy.minimum_test_runs:
                blockers.append("RW_JAM_TEST_SUPPORT_BELOW_POLICY")
            if metric.recall < policy.minimum_test_recall:
                blockers.append("RW_JAM_TEST_RECALL_BELOW_POLICY")
            if metric.f1 < policy.minimum_test_f1:
                blockers.append("RW_JAM_TEST_F1_BELOW_POLICY")
        return "CROSS_TASK_CALIBRATION_REUSE_BLOCKED" not in blockers and not any(item.startswith(("MODEL_", "RW_JAM_")) for item in blockers)

    @staticmethod
    def _knowledge_gate(knowledge: KnowledgePromotionEvidence | None, blockers: list[str]) -> bool:
        if knowledge is None:
            blockers.append("FORMAL_NEO4J_EVIDENCE_MISSING")
            return False
        if knowledge.backend.lower() != "neo4j":
            blockers.append("KNOWLEDGE_BACKEND_NOT_NEO4J")
        if knowledge.health_status != "PASS":
            blockers.append("NEO4J_HEALTH_NOT_PASS")
        if not knowledge.read_only_query_executed or not knowledge.query_sha256:
            blockers.append("FORMAL_NEO4J_QUERY_NOT_EXECUTED")
        if RW_JAM_FAULT_ID not in knowledge.matched_fault_ids:
            blockers.append("RW_JAM_NOT_RETURNED_BY_NEO4J")
        if not knowledge.evidence_paths:
            blockers.append("NEO4J_EVIDENCE_PATH_MISSING")
        if not knowledge.formal_release_accepted:
            blockers.append("KNOWLEDGE_RELEASE_NOT_ACCEPTED")
        return not any(item.startswith(("KNOWLEDGE_", "NEO4J_", "FORMAL_NEO4J", "RW_JAM_NOT")) for item in blockers)

    @staticmethod
    def _expert_gate(
        expert: ExpertPromotionEvidence | None,
        dataset: ALevelDatasetEvidence | None,
        model: ModelPromotionEvidence | None,
        knowledge: KnowledgePromotionEvidence | None,
        blockers: list[str],
    ) -> bool:
        if expert is None:
            blockers.append("NAMED_EXPERT_REVIEW_MISSING")
            return False
        if expert.review_mode == ReviewMode.LLM_ONLY:
            blockers.append("LLM_ONLY_REVIEW_FORBIDDEN")
        if not expert.approved:
            blockers.append("NAMED_EXPERT_NOT_APPROVED")
        if dataset and expert.dataset_manifest_sha256 != dataset.dataset_manifest_sha256:
            blockers.append("EXPERT_DATASET_HASH_MISMATCH")
        if model and expert.model_artifact_sha256 != model.artifact_sha256:
            blockers.append("EXPERT_MODEL_HASH_MISMATCH")
        if knowledge and expert.knowledge_query_sha256 != knowledge.query_sha256:
            blockers.append("EXPERT_KNOWLEDGE_HASH_MISMATCH")
        return not any(item.startswith(("NAMED_", "LLM_", "EXPERT_")) for item in blockers)

    @staticmethod
    def _summary(request: RWJamPromotionRequest) -> dict[str, Any]:
        dataset = request.dataset
        counts = Counter(run.split.value for run in dataset.runs) if dataset else Counter()
        return {
            "dataset_manifest_id": None if dataset is None else dataset.dataset_manifest_id,
            "split_counts": dict(counts),
            "model_manifest_id": None if request.model is None else request.model.manifest.manifest_id,
            "knowledge_backend": None if request.knowledge is None else request.knowledge.backend,
            "expert_reviewer_id": None if request.expert is None else request.expert.reviewer_id,
        }


class _FormalKnowledgeRepository(Protocol):
    backend_id: str
    knowledge_version: str

    def health_check(self) -> dict[str, Any]: ...
    def retrieve(self, anomaly_ids: list[str]) -> list[KnowledgeEdge]: ...


class Neo4jEvidencePathService:
    """Read-only formal Neo4j evidence adapter.

    The service is intentionally small: it records actual repository health and
    retrieved edges.  It never substitutes mapping projection output for a graph
    query and never performs graph writes.
    """

    def __init__(self, repository: _FormalKnowledgeRepository) -> None:
        self.repository = repository

    def reason(self, anomaly_ids: Iterable[str]) -> KnowledgePromotionEvidence:
        anomaly_ids = sorted(set(str(item) for item in anomaly_ids if str(item)))
        if not anomaly_ids:
            raise ValueError("formal Neo4j evidence-path query requires anomaly IDs")
        if getattr(self.repository, "backend_id", "") != "neo4j":
            raise ValueError("formal evidence-path query requires a Neo4j repository")
        health = self.repository.health_check()
        if health.get("status") != "PASS":
            raise RuntimeError(f"Neo4j health check failed: {health}")
        edges = self.repository.retrieve(anomaly_ids)
        query_payload = {
            "knowledge_version": self.repository.knowledge_version,
            "anomaly_ids": anomaly_ids,
            "edge_keys": sorted((edge.fault_id, edge.anomaly_id) for edge in edges),
        }
        by_fault: dict[str, list[KnowledgeEdge]] = defaultdict(list)
        for edge in edges:
            by_fault[edge.fault_id].append(edge)
        paths = [
            {
                "fault_id": fault_id,
                "anomaly_ids": sorted(edge.anomaly_id for edge in values),
                "aggregate_support_score": sum(float(edge.weight) for edge in values),
                "score_semantics": "ASSOCIATIVE_EDGE_WEIGHT_SUM_NOT_CAUSAL_EFFECT",
                "edge_count": len(values),
            }
            for fault_id, values in sorted(by_fault.items())
        ]
        return KnowledgePromotionEvidence(
            backend="neo4j",
            knowledge_version=self.repository.knowledge_version,
            health_status="PASS",
            read_only_query_executed=True,
            query_sha256=_sha(query_payload),
            matched_fault_ids=sorted(by_fault),
            evidence_paths=paths,
            formal_release_accepted=bool(health.get("release_count", 0) >= 1),
        )


# Backward-compatible name. New code must use Neo4jEvidencePathService.
Neo4jCausalReasoningService = Neo4jEvidencePathService


__all__ = [
    "RW_JAM_FAULT_ID",
    "RW_JAM_PROMOTION_SCHEMA_VERSION",
    "PromotionStatus",
    "ReviewMode",
    "ALevelRunEvidence",
    "ActuatorTelemetryContractEvidence",
    "ALevelDatasetEvidence",
    "ModelPromotionEvidence",
    "KnowledgePromotionEvidence",
    "ExpertPromotionEvidence",
    "RWJamPromotionPolicy",
    "RWJamPromotionRequest",
    "RWJamPromotionAssessment",
    "RWJamPromotionGate",
    "Neo4jEvidencePathService",
    "Neo4jCausalReasoningService",
]
