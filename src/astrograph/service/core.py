from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from astrograph.anomaly import (
    AnomalyEpisode,
    EpisodeLifecycleStatus,
    EpisodeTemporalConstraintCatalog,
    IncrementalAnomalyObservation,
    JsonEpisodeLifecycleStore,
    analyze_episode_temporal_constraints,
)
from astrograph.audit.store import AuditStore, NullAuditStore
from astrograph.cases import DiagnosticCaseRecord, DiagnosticCaseRegistry
from astrograph.config import DiagnosisPolicy
from astrograph.evidence import EvidenceConstraintCatalog, EvidenceRegistry, validate_evidence_registry
from astrograph.experiments import ExperimentRecord, JsonExperimentRegistry
from astrograph.idempotency import JsonIdempotencyStore, SqliteIdempotencyStore
from astrograph.knowledge import JsonKnowledgeGovernanceStore
from astrograph.knowledge.release import KnowledgeRelease, validate_knowledge_release
from astrograph.knowledge.repository import KnowledgeRepository
from astrograph.lifecycle import JsonRequestLifecycleStore
from astrograph.persistence import JsonStatePersistenceCoordinator
from astrograph.schemas import WorkflowInput
from astrograph.runtime import (
    AstroGraphRuntime,
    DIAGNOSIS_PIPELINE_ID,
    RuntimeRequest,
    RuntimeStatus,
    build_application_runtime,
)
from astrograph import __version__


class ServiceHealth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    service_version: str
    repository_type: str
    knowledge_version: str
    knowledge_release_status: str
    evidence_registry_status: str
    evidence_constraint_status: str
    temporal_constraint_status: str = "NOT_CONFIGURED"
    online_episode_store_status: str = "NOT_CONFIGURED"
    request_lifecycle_status: str = "NOT_CONFIGURED"
    knowledge_governance_status: str = "NOT_CONFIGURED"
    experiment_registry_status: str = "NOT_CONFIGURED"
    persistence_status: str = "NOT_CONFIGURED"
    runtime_status: str = "NOT_CONFIGURED"
    details: dict[str, Any] = Field(default_factory=dict)


class AstroGraphService:
    """Stable application service. Simulation changes enter through adapters and versioned protocols."""

    def __init__(
        self,
        *,
        repository: KnowledgeRepository,
        policy: DiagnosisPolicy,
        audit_store: AuditStore | None = None,
        service_version: str = __version__,
        project_root: str | Path | None = None,
        active_release_path: str | Path | None = None,
        evidence_registry: EvidenceRegistry | None = None,
        case_registry: DiagnosticCaseRegistry | None = None,
        evidence_constraints: EvidenceConstraintCatalog | None = None,
        temporal_constraints: EpisodeTemporalConstraintCatalog | None = None,
        idempotency_store: JsonIdempotencyStore | SqliteIdempotencyStore | None = None,
        episode_store: JsonEpisodeLifecycleStore | None = None,
        request_lifecycle_store: JsonRequestLifecycleStore | None = None,
        knowledge_governance_store: JsonKnowledgeGovernanceStore | None = None,
        experiment_registry: JsonExperimentRegistry | None = None,
        persistence_coordinator: JsonStatePersistenceCoordinator | None = None,
        runtime: AstroGraphRuntime | None = None,
    ):
        self.repository = repository
        self.policy = policy
        self.audit_store = audit_store or NullAuditStore()
        self.service_version = service_version
        self.project_root = Path(project_root).resolve() if project_root else None
        self.active_release_path = Path(active_release_path) if active_release_path else None
        self.evidence_registry = evidence_registry
        self.case_registry = case_registry
        self.evidence_constraints = evidence_constraints
        self.temporal_constraints = temporal_constraints
        self.idempotency_store = idempotency_store
        self.episode_store = episode_store
        self.request_lifecycle_store = request_lifecycle_store
        self.knowledge_governance_store = knowledge_governance_store
        self.experiment_registry = experiment_registry
        self.persistence_coordinator = persistence_coordinator
        self.runtime = runtime or build_application_runtime(
            repository=self.repository,
            policy=self.policy,
            audit_store=self.audit_store,
            constraints=self.evidence_constraints,
            project_root=self.project_root,
        )

    def execute_runtime(self, request: RuntimeRequest) -> dict[str, Any]:
        return self.runtime.execute(request).model_dump(mode="json")

    def diagnose(self, request: WorkflowInput) -> dict[str, Any]:
        runtime_result = self.runtime.execute(
            RuntimeRequest(
                request_id=request.request_id,
                pipeline_id=DIAGNOSIS_PIPELINE_ID,
                payload=request.model_dump(mode="json"),
                metadata={"entrypoint": "AstroGraphService.diagnose"},
            )
        )
        if runtime_result.status != RuntimeStatus.SUCCEEDED:
            error = runtime_result.error
            if error is None:
                raise RuntimeError("AstroGraph runtime failed without an error record")
            raise RuntimeError(f"{error.code}: {error.message}")
        return runtime_result.output

    def diagnose_idempotent(self, request: WorkflowInput) -> tuple[dict[str, Any], bool]:
        request_payload = request.model_dump(mode="json")
        if self.request_lifecycle_store:
            self.request_lifecycle_store.receive(request.request_id, request_payload)

        def producer() -> dict[str, Any]:
            if self.request_lifecycle_store:
                self.request_lifecycle_store.mark_running(request.request_id)
            try:
                response = self.diagnose(request)
                if self.request_lifecycle_store:
                    self.request_lifecycle_store.mark_completed(request.request_id, response)
                return response
            except Exception as exc:
                if self.request_lifecycle_store:
                    self.request_lifecycle_store.mark_failed(request.request_id, exc)
                raise

        if not self.idempotency_store:
            return producer(), False
        response, replayed = self.idempotency_store.execute(request.request_id, request_payload, producer)
        if replayed and self.request_lifecycle_store:
            self.request_lifecycle_store.mark_replay(request.request_id)
        return response, replayed

    def register_episode_case(
        self,
        *,
        request: WorkflowInput,
        response: dict[str, Any],
        episode_ids: list[str],
        episode_hash: str,
        detector_id: str,
        rule_set_version: str,
    ) -> dict[str, Any] | None:
        if not self.case_registry:
            return None
        metadata = request.metadata
        run_id = str(metadata.get("run_id", ""))
        dataset_id = str(metadata.get("dataset_id", ""))
        audit = response["audit"]
        diagnosis = response["diagnosis"]
        case_record_id = DiagnosticCaseRecord.create_id(
            dataset_id=dataset_id,
            run_id=run_id,
            detector_id=detector_id,
            rule_set_version=rule_set_version,
            policy_version=self.policy.policy_version,
            knowledge_version=self.repository.knowledge_version,
            source_type="anomaly_episode",
            source_fingerprint=episode_hash,
        )
        for existing in self.case_registry.list_records():
            if existing.case_record_id == case_record_id:
                return existing.model_dump(mode="json")
        record = DiagnosticCaseRecord(
            case_record_id=case_record_id,
            source_type="anomaly_episode",
            dataset_id=dataset_id,
            run_id=run_id,
            paired_nominal_run_id=metadata.get("paired_nominal_run_id"),
            split=str(metadata.get("split", "")),
            synthetic_case_id=request.case_id,
            request_id=request.request_id,
            detector_id=detector_id,
            rule_set_version=rule_set_version,
            anomaly_count=len(request.anomalies),
            anomaly_hash=audit["input_hash"],
            episode_count=len(episode_ids),
            episode_ids=episode_ids,
            episode_hash=episode_hash,
            diagnosis_status=diagnosis["status"],
            conclusion_fault_id=diagnosis.get("conclusion_fault_id"),
            conclusion_fault_name=diagnosis.get("conclusion_fault_name"),
            top_score=diagnosis.get("top_score", 0.0),
            score_margin=diagnosis.get("score_margin", 0.0),
            policy_version=self.policy.policy_version,
            knowledge_version=self.repository.knowledge_version,
            audit_id=audit["audit_id"],
            metadata={
                **metadata,
                "episode_source": True,
                "idempotency_request_id": request.request_id,
            },
        )
        self.case_registry.upsert(record)
        return record.model_dump(mode="json")

    def evaluate_episode_temporal(self, episodes: list[AnomalyEpisode]) -> dict[str, Any]:
        if not self.temporal_constraints:
            return {
                "schema_version": "astrograph-episode-temporal-analysis.v1",
                "catalog_version": "NOT_CONFIGURED",
                "relation_count": 0,
                "evaluations_by_fault": {},
                "relations": [],
                "claim_boundary": "Temporal constraints are not configured.",
            }
        return analyze_episode_temporal_constraints(episodes, self.temporal_constraints).model_dump(mode="json")

    # Online anomaly episode lifecycle -------------------------------------------------
    def apply_episode_observation(self, observation: IncrementalAnomalyObservation) -> dict[str, Any]:
        if not self.episode_store:
            raise RuntimeError("online episode store is not configured")
        return self.episode_store.apply(observation).model_dump(mode="json")

    def close_episode(self, episode_id: str, *, end_s: float, reason: str) -> dict[str, Any]:
        if not self.episode_store:
            raise RuntimeError("online episode store is not configured")
        return self.episode_store.close(episode_id, end_s=end_s, reason=reason).model_dump(mode="json")

    def episode(self, episode_id: str) -> dict[str, Any] | None:
        if not self.episode_store:
            return None
        record = self.episode_store.get(episode_id)
        return None if record is None else record.model_dump(mode="json")

    def episodes(self, *, run_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        if not self.episode_store:
            return []
        parsed_status = EpisodeLifecycleStatus(status) if status else None
        return [item.model_dump(mode="json") for item in self.episode_store.list_records(run_id=run_id, status=parsed_status)]

    # Request lifecycle ----------------------------------------------------------------
    def request_lifecycle(self, request_id: str) -> dict[str, Any] | None:
        if not self.request_lifecycle_store:
            return None
        record = self.request_lifecycle_store.get(request_id)
        return None if record is None else record.model_dump(mode="json")

    def request_timeline(self, request_id: str) -> dict[str, Any] | None:
        if not self.request_lifecycle_store:
            return None
        try:
            return self.request_lifecycle_store.replay_timeline(request_id)
        except KeyError:
            return None

    # Persistence and recovery ---------------------------------------------------------
    def persistence_status(self) -> dict[str, Any]:
        if not self.persistence_coordinator:
            return {"schema_version": "astrograph-persistence-status.v1", "status": "NOT_CONFIGURED", "components": []}
        return self.persistence_coordinator.status().model_dump(mode="json")

    def snapshot_state(self, output_dir: str | Path) -> dict[str, Any]:
        if not self.persistence_coordinator:
            raise RuntimeError("persistence coordinator is not configured")
        return self.persistence_coordinator.snapshot(output_dir)

    def recover_state(self) -> dict[str, Any]:
        if not self.persistence_coordinator:
            raise RuntimeError("persistence coordinator is not configured")
        return self.persistence_coordinator.recover()

    # Knowledge governance --------------------------------------------------------------
    def knowledge_changes(self) -> list[dict[str, Any]]:
        if not self.knowledge_governance_store:
            return []
        return [item.model_dump(mode="json") for item in self.knowledge_governance_store.list_records()]

    def knowledge_change(self, change_id: str) -> dict[str, Any] | None:
        if not self.knowledge_governance_store:
            return None
        item = self.knowledge_governance_store.get(change_id)
        return None if item is None else item.model_dump(mode="json")

    def create_knowledge_draft(self, **kwargs: Any) -> dict[str, Any]:
        if not self.knowledge_governance_store:
            raise RuntimeError("knowledge governance store is not configured")
        return self.knowledge_governance_store.create_draft(**kwargs).model_dump(mode="json")

    def submit_knowledge_change(self, change_id: str, **kwargs: Any) -> dict[str, Any]:
        if not self.knowledge_governance_store:
            raise RuntimeError("knowledge governance store is not configured")
        return self.knowledge_governance_store.submit(change_id, **kwargs).model_dump(mode="json")

    def review_knowledge_change(self, change_id: str, **kwargs: Any) -> dict[str, Any]:
        if not self.knowledge_governance_store:
            raise RuntimeError("knowledge governance store is not configured")
        return self.knowledge_governance_store.review(change_id, **kwargs).model_dump(mode="json")

    def publish_knowledge_change(self, change_id: str, **kwargs: Any) -> dict[str, Any]:
        if not self.knowledge_governance_store:
            raise RuntimeError("knowledge governance store is not configured")
        record = self.knowledge_governance_store.publish(change_id, **kwargs).model_dump(mode="json")
        return {
            "change": record,
            "runtime_reload_required": True,
            "message": "The active release manifest changed. Restart/rebuild the repository before serving it.",
        }

    def rollback_knowledge_change(self, change_id: str, **kwargs: Any) -> dict[str, Any]:
        if not self.knowledge_governance_store:
            raise RuntimeError("knowledge governance store is not configured")
        record = self.knowledge_governance_store.rollback(change_id, **kwargs).model_dump(mode="json")
        return {
            "change": record,
            "runtime_reload_required": True,
            "message": "The active release manifest was rolled back. Restart/rebuild the repository before serving it.",
        }

    # Version-combination experiments ---------------------------------------------------
    def register_experiment(self, record: ExperimentRecord) -> dict[str, Any]:
        if not self.experiment_registry:
            raise RuntimeError("experiment registry is not configured")
        return self.experiment_registry.register(record).model_dump(mode="json")

    def experiment(self, experiment_id: str) -> dict[str, Any] | None:
        if not self.experiment_registry:
            return None
        record = self.experiment_registry.get(experiment_id)
        return None if record is None else record.model_dump(mode="json")

    def experiments(self) -> list[dict[str, Any]]:
        if not self.experiment_registry:
            return []
        return [item.model_dump(mode="json") for item in self.experiment_registry.list_records()]

    def health(self) -> ServiceHealth:
        repo_details: dict[str, Any] = {}
        repo_status = "PASS"
        if hasattr(self.repository, "health_check"):
            try:
                repo_details = getattr(self.repository, "health_check")()
                repo_status = str(repo_details.get("status", "FAIL"))
            except Exception as exc:
                repo_status = "FAIL"
                repo_details = {"error": f"{type(exc).__name__}: {exc}"}
        else:
            try:
                repo_details = {"status": "PASS", "content_hash": getattr(self.repository, "content_hash", "")}
            except Exception as exc:
                repo_status = "FAIL"
                repo_details = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}

        release_status = "NOT_CONFIGURED"
        release_details: dict[str, Any] = {}
        if self.active_release_path and self.project_root:
            try:
                release = KnowledgeRelease.from_json(self.active_release_path)
                release_details = validate_knowledge_release(release, self.project_root)
                release_status = release_details["status"]
            except Exception as exc:
                release_status = "FAIL"
                release_details = {"error": f"{type(exc).__name__}: {exc}"}

        evidence_status = "NOT_CONFIGURED"
        evidence_details: dict[str, Any] = {}
        if self.evidence_registry:
            try:
                evidence_details = validate_evidence_registry(self.evidence_registry)
                evidence_status = evidence_details["status"]
            except Exception as exc:
                evidence_status = "FAIL"
                evidence_details = {"error": f"{type(exc).__name__}: {exc}"}

        constraint_status = "PASS" if self.evidence_constraints else "NOT_CONFIGURED"
        temporal_status = "PASS" if self.temporal_constraints else "NOT_CONFIGURED"
        constraint_details = (
            {
                "catalog_version": self.evidence_constraints.catalog_version,
                "critical_requirement_count": len(self.evidence_constraints.critical_requirements),
                "counter_evidence_rule_count": len(self.evidence_constraints.counter_evidence_rules),
            }
            if self.evidence_constraints
            else {}
        )

        persistence_report = self.persistence_status()
        persistence_status = persistence_report.get("status", "NOT_CONFIGURED")
        try:
            runtime_report = self.runtime.describe()
            runtime_status = str(runtime_report.get("status", "FAIL"))
        except Exception as exc:
            runtime_status = "FAIL"
            runtime_report = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}

        configured_stores = {
            "online_episode_store": "PASS" if self.episode_store else "NOT_CONFIGURED",
            "request_lifecycle": "PASS" if self.request_lifecycle_store else "NOT_CONFIGURED",
            "knowledge_governance": "PASS" if self.knowledge_governance_store else "NOT_CONFIGURED",
            "experiment_registry": "PASS" if self.experiment_registry else "NOT_CONFIGURED",
            "persistence": persistence_status,
        }
        critical = [repo_status, runtime_status]
        for item in [release_status, evidence_status, constraint_status, temporal_status, *configured_stores.values()]:
            if item != "NOT_CONFIGURED":
                critical.append(item)
        status = "PASS" if all(item == "PASS" for item in critical) else "DEGRADED"
        return ServiceHealth(
            status=status,
            service_version=self.service_version,
            repository_type=type(self.repository).__name__,
            knowledge_version=self.repository.knowledge_version,
            knowledge_release_status=release_status,
            evidence_registry_status=evidence_status,
            evidence_constraint_status=constraint_status,
            temporal_constraint_status=temporal_status,
            online_episode_store_status=configured_stores["online_episode_store"],
            request_lifecycle_status=configured_stores["request_lifecycle"],
            knowledge_governance_status=configured_stores["knowledge_governance"],
            experiment_registry_status=configured_stores["experiment_registry"],
            persistence_status=configured_stores["persistence"],
            runtime_status=runtime_status,
            details={
                "repository": repo_details,
                "knowledge_release": release_details,
                "evidence_registry": evidence_details,
                "evidence_constraints": constraint_details,
                "temporal_constraints": (
                    {
                        "catalog_version": self.temporal_constraints.catalog_version,
                        "constraint_count": len(self.temporal_constraints.constraints),
                    }
                    if self.temporal_constraints
                    else {}
                ),
                "stores": configured_stores,
                "persistence": persistence_report,
                "runtime": runtime_report,
            },
        )

    def active_knowledge_release(self) -> dict[str, Any]:
        if not self.active_release_path:
            return {"status": "NOT_CONFIGURED"}
        return KnowledgeRelease.from_json(self.active_release_path).model_dump(mode="json")

    def evidence_sources(self) -> list[dict[str, Any]]:
        if not self.evidence_registry:
            return []
        return [item.model_dump(mode="json") for item in self.evidence_registry.list_sources()]

    def diagnostic_case(self, case_record_id: str) -> dict[str, Any] | None:
        if not self.case_registry:
            return None
        record = self.case_registry.get(case_record_id)
        return record.model_dump(mode="json") if record is not None else None

    def diagnostic_cases(self) -> list[dict[str, Any]]:
        if not self.case_registry:
            return []
        return [record.model_dump(mode="json") for record in self.case_registry.list_records()]

    def delete_diagnostic_case(self, case_record_id: str) -> bool:
        if not self.case_registry:
            return False
        return self.case_registry.delete(case_record_id)
