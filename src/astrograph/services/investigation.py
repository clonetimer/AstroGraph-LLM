from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from astrograph.evidence import EpisodeEvidenceBundle, EvidenceDefinition, EvidenceFactStatus
from astrograph.evidence.executable import stable_id
from astrograph.investigation import (
    GraphQueryPlan,
    InvestigationBudget,
    InvestigationPlan,
    InvestigationStep,
    VerificationPlan,
    VerificationTask,
)




class EvidenceNavigationPath(BaseModel):
    """One fault <- evidence-definition <- feature path with observed facts."""

    model_config = ConfigDict(extra="forbid")

    fault_id: str = Field(min_length=1)
    evidence_definition_id: str = Field(min_length=1)
    feature_definition_id: str = Field(min_length=1)
    role: str = Field(default="SUPPORTING")
    evidence_level: str = Field(default="E0")
    critical: bool = False
    activated_fact_ids: list[str] = Field(default_factory=list)
    contradicted_fact_ids: list[str] = Field(default_factory=list)
    missing_fact_count: int = Field(default=0, ge=0)
    activated: bool = False
    path: list[str] = Field(default_factory=list)


class EvidenceNavigationResult(BaseModel):
    """Deterministic walk of the episode evidence graph (fault <- evidence).

    Mirrors the episode-evidence-graph edges: evidence definition USES_FEATURE,
    evidence fact INSTANCE_OF definition, fact DERIVED_FROM feature result.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-evidence-navigation-result.v1"
    episode_id: str = Field(min_length=1)
    navigation_source: str = "episode-evidence-graph"
    paths: list[EvidenceNavigationPath] = Field(default_factory=list)
    fault_summary: dict[str, dict] = Field(default_factory=dict)
    claim_boundary: str = Field(default="Deterministic evidence-graph navigation; associative support only, not causal or flight evidence.")


class InvestigationPlanningService:
    """Deterministic investigation-plan compiler.

    This is a service, not an autonomous Agent.  The historical
    InvestigationOrchestratorAgent delegates here for compatibility.
    """

    service_id = "investigation-planning-service"
    service_version = "0.5.12"

    def plan(
        self,
        *,
        task_id: str,
        run_id: str,
        candidate_fault_ids: list[str],
        object_label: str = "D3反作用轮",
        evidence_tool_id: str = "d3_rw.evidence.build",
        core_tool_id: str = "diagnosis.invoke_core",
    ) -> InvestigationPlan:
        objective = f"对{object_label}运行 {run_id} 执行证据驱动的影子诊断调查"
        plan_id = InvestigationPlan.deterministic_id(
            task_id=task_id,
            objective=objective,
            candidate_fault_ids=candidate_fault_ids,
        )
        return InvestigationPlan(
            plan_id=plan_id,
            task_id=task_id,
            objective=objective,
            candidate_fault_ids=sorted(candidate_fault_ids),
            steps=[
                InvestigationStep(
                    step_id="S1-NAVIGATE-EVIDENCE",
                    agent_id="evidence-navigator-agent",
                    objective="定位候选故障的必要证据、支持证据、反证和数据缺口",
                    allowed_tools=["semantics.validate", "evidence.catalog.read"],
                    expected_output_protocol="astrograph-graph-query-plan.v1",
                ),
                InvestigationStep(
                    step_id="S2-PLAN-VERIFICATION",
                    agent_id="verification-planning-service",
                    objective="将证据要求编译为版本化数值验证任务",
                    allowed_tools=[evidence_tool_id],
                    depends_on=["S1-NAVIGATE-EVIDENCE"],
                    expected_output_protocol="astrograph-verification-plan.v1",
                ),
                InvestigationStep(
                    step_id="S3-REASON",
                    agent_id="diagnostic-reasoner-agent",
                    objective="根据证据事实形成多个竞争假设",
                    allowed_tools=[],
                    depends_on=["S2-PLAN-VERIFICATION"],
                    expected_output_protocol="astrograph-hypothesis-set.v1",
                ),
                InvestigationStep(
                    step_id="S4-CRITIQUE",
                    agent_id="safety-validation-service",
                    objective="检查证据遗漏、逻辑跳跃、能力越界和解释错配",
                    allowed_tools=[],
                    depends_on=["S3-REASON"],
                    expected_output_protocol="astrograph-critique-result.v1",
                ),
                InvestigationStep(
                    step_id="S5-GOVERN",
                    agent_id="decision-governor",
                    objective="执行确定性安全放行与Core对照",
                    allowed_tools=[core_tool_id],
                    depends_on=["S4-CRITIQUE"],
                    expected_output_protocol="astrograph-decision-record.v1",
                ),
            ],
            budget=InvestigationBudget(
                max_reflection_rounds=3,
                max_graph_queries=10,
                max_tool_calls=30,
                max_simulation_calls=0,
                stop_after_no_new_evidence_rounds=2,
            ),
            stop_conditions=[
                "唯一注册故障满足全部关键证据且无阻断审查项",
                "证据不足或冲突无法在预算内解决",
                "Agentic影子结论与确定性Core不一致时转人工复核",
                "连续两轮无新增证据时停止",
            ],
            shadow_mode=True,
        )


class EvidenceNavigationPlanningService:
    """Deterministic compiler from candidate faults to a bounded graph query plan."""

    service_id = "evidence-navigation-planning-service"
    service_version = "0.5.12"

    def navigate(
        self,
        *,
        candidate_fault_ids: list[str],
        definitions: list[EvidenceDefinition],
        object_label: str = "反作用轮",
    ) -> GraphQueryPlan:
        available_faults = {item.target_fault_id for item in definitions}
        unknown = sorted(set(candidate_fault_ids) - available_faults)
        if unknown:
            raise ValueError(f"candidate faults are not present in evidence catalog: {unknown}")
        return GraphQueryPlan(
            query_plan_id=stable_id(
                "GQ",
                {
                    "candidate_fault_ids": sorted(candidate_fault_ids),
                    "definition_ids": sorted(item.evidence_definition_id for item in definitions),
                },
            ),
            purpose=f"收集候选{object_label}故障的支持、反证和缺失证据",
            start_entity_ids=sorted(candidate_fault_ids),
            allowed_relation_types=[
                "REQUIRES_EVIDENCE",
                "SUPPORTED_BY",
                "CONTRADICTED_BY",
                "USES_FEATURE",
                "DOCUMENTED_IN",
            ],
            target_entity_types=["EvidenceDefinition", "FeatureDefinition", "EvidenceSource"],
            include_support=True,
            include_counter_evidence=True,
            max_depth=3,
            max_results=100,
            read_only=True,
        )

    def navigate_episode(
        self,
        bundle: EpisodeEvidenceBundle,
        *,
        candidate_fault_ids: list[str] | None = None,
    ) -> EvidenceNavigationResult:
        """Walk the episode evidence graph and return per-fault navigation paths.

        Unlike ``navigate`` (which compiles a query *plan*), this method walks the
        in-memory episode evidence graph structure itself: evidence definitions
        reference features, evidence facts are instances of definitions, and
        facts carry their evaluated status. For each registered fault the result
        lists every evidence definition path together with its activated,
        contradicted and missing facts.
        """
        definitions = list(bundle.evidence_definitions)
        if candidate_fault_ids is not None:
            wanted = set(candidate_fault_ids)
            definitions = [item for item in definitions if item.target_fault_id in wanted]
        features_by_id = {item.feature_definition_id: item for item in bundle.feature_definitions}
        facts_by_definition: dict[str, list] = {}
        for fact in bundle.evidence_facts:
            facts_by_definition.setdefault(fact.evidence_definition_id, []).append(fact)
        paths: list[EvidenceNavigationPath] = []
        fault_summary: dict[str, dict] = {}
        for definition in definitions:
            facts = facts_by_definition.get(definition.evidence_definition_id, [])
            activated = [
                fact for fact in facts if fact.status == EvidenceFactStatus.SATISFIED
            ]
            contradicted = [
                fact for fact in facts if fact.status == EvidenceFactStatus.CONTRADICTED
            ]
            missing = [
                fact
                for fact in facts
                if fact.status in (
                    EvidenceFactStatus.MISSING_DATA,
                    EvidenceFactStatus.INVALID_DATA,
                    EvidenceFactStatus.UNKNOWN,
                )
            ]
            if not facts:
                missing = [None]
            feature = features_by_id.get(definition.feature_definition_id)
            path = EvidenceNavigationPath(
                fault_id=definition.target_fault_id,
                evidence_definition_id=definition.evidence_definition_id,
                feature_definition_id=definition.feature_definition_id,
                role=definition.role.value,
                evidence_level=definition.evidence_level,
                critical=definition.critical,
                activated_fact_ids=sorted(fact.evidence_fact_id for fact in activated),
                contradicted_fact_ids=sorted(fact.evidence_fact_id for fact in contradicted),
                missing_fact_count=len(missing),
                activated=bool(activated),
                path=[
                    definition.evidence_definition_id,
                    definition.feature_definition_id,
                    feature.name if feature is not None else definition.feature_definition_id,
                ],
            )
            paths.append(path)
            summary = fault_summary.setdefault(
                definition.target_fault_id,
                {"activated_definitions": 0, "total_definitions": 0, "has_support": False},
            )
            summary["total_definitions"] += 1
            if path.activated:
                summary["activated_definitions"] += 1
                summary["has_support"] = True
        return EvidenceNavigationResult(
            episode_id=bundle.episode.episode_id,
            paths=paths,
            fault_summary=fault_summary,
        )


class VerificationPlanningService:
    """Deterministic compiler from an evidence bundle to numerical tasks."""

    service_id = "verification-planning-service"
    service_version = "0.5.12"

    def build_plan(
        self,
        bundle: EpisodeEvidenceBundle,
        *,
        generated_by_id: str | None = None,
    ) -> VerificationPlan:
        tasks: list[VerificationTask] = []
        for definition in bundle.feature_definitions:
            tasks.append(
                VerificationTask(
                    verification_task_id=f"VT-FEATURE-{definition.feature_definition_id}",
                    tool_id="feature.compute",
                    feature_definition_id=definition.feature_definition_id,
                    input_refs={
                        "run_id": bundle.episode.run_id,
                        "paired_nominal_run_id": bundle.episode.paired_nominal_run_id,
                        "episode_id": bundle.episode.episode_id,
                    },
                    parameters={
                        "window_start_s": bundle.episode.start_s,
                        "window_end_s": bundle.episode.end_s,
                        "interval": bundle.episode.interval,
                        "analysis_wheel_index": bundle.analysis_wheel_index,
                        "selected_wheel_index": bundle.selected_wheel_index,
                    },
                )
            )
        for definition in bundle.evidence_definitions:
            tasks.append(
                VerificationTask(
                    verification_task_id=f"VT-EVIDENCE-{definition.evidence_definition_id}",
                    tool_id="evidence.evaluate",
                    evidence_definition_id=definition.evidence_definition_id,
                    input_refs={"episode_id": bundle.episode.episode_id},
                    parameters={
                        "comparison": definition.comparison.value,
                        "threshold": definition.threshold,
                        "unit": definition.threshold_unit,
                    },
                )
            )
        return VerificationPlan(
            verification_plan_id=stable_id(
                "VP",
                {
                    "episode_id": bundle.episode.episode_id,
                    "tasks": [item.verification_task_id for item in tasks],
                },
            ),
            episode_id=bundle.episode.episode_id,
            tasks=tasks,
            generated_by_agent_id=generated_by_id or self.service_id,
        )