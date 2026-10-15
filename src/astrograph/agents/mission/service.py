"""One task facade, existing ToolRegistry, existing autonomous diagnosis kernel.

No new general DAG engine, no direct spacecraft control, no automatic release.
"""
from __future__ import annotations
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any
from astrograph.tools.registry import ToolRegistry, ToolDefinition, ToolSpec
from astrograph.orchestration.models import WorkflowContext, PermissionScope, SideEffectLevel
from astrograph.integrations.process_control import run_managed_subprocess
from astrograph.oag.contracts import OAGInput
from astrograph.oag.kernel import EvidenceDiagnosisKernel
from astrograph.oag.agent import AutonomousDiagnosisService, CaseContext
from astrograph.oag.evidence_tools import MeasurementBank, tools_from_measurements
from astrograph.memory.service import CaseMemoryService, MemoryPolicy
from .models import AgentRequest,AgentSettings,AgentResult
from .storage import Workspace,file_sha,sha
from .planner import Planner
from .documents import DocumentTools
from .modes import resolve_mode, allowed_tools, investigation_brief
from .context import ContextBuilder
from .sources import diagnostic_sources
from .generic_series import GenericSeriesAdapter, GenericSeriesManifest
from .model_capabilities import ModelCapabilityQuery, ModelCapabilityRegistry
from .model_execution import ModelExecutionResolver
from .three_task_contract import normalize_three_task_execution
from .generic_shadow_fusion import build_generic_shadow_fusion
from .model_decision_policy import assess_model_decision_eligibility
from .document_quality import document_provider_quality_receipt
from .literature_context import build_literature_context_assessment
from .reviewed_claims import ReviewedClaimRAGBinding, ReviewedClaimRAGEnricher
from .joint_evidence import build_joint_evidence_package, joint_evidence_report
from .continuous_diagnosis import recommend_next_observations
from .replay import ReplayManifest, run_replay
from astrograph.integrations.execution_budget import ExecutionBudget, execution_scope, remaining_time, interruptible_timeout
from astrograph.lifecycle import DocumentProposal, KnowledgeFeedbackStore, KnowledgeProposal, LifecyclePolicy

CATALOG = {
    "model.infer": "Prepare native ADCS/PHM inference, or generic-series detection with explicit semantic boundaries.",
    "model.capabilities": "Discover compatible model contracts for the current channels; discovery never authorizes execution.",
    "model.execute_compatible": "Execute only a host-bound compatible provider through strict model admission; no binding means no execution.",
    "dataset.replay": "Offline truth-aware dataset replay; never represented as online diagnosis.",
    "graph.reason": "Read-only forward/backward evidence graph reasoning, not LLM diagnosis.",
    "knowledge.graph.mechanism": "Read-only maintenance mechanism lookup in the configured Neo4j knowledge graph.",
    "knowledge.graph.match_observations": "Read-only mapping from observed maintenance phenomena to candidate faults in Neo4j.",
    "diagnosis.fuse": "Existing BiRRF, deterministic decision and bounded evidence acquisition.",
    "document.convert": "Native UTF-8 or host-enabled MinerU; retain source artifacts.",
    "document.clean": "Conservative text cleaning; preserve numbers, units, negation and provenance.",
    "rag.retrieve": "Source-bound retrieval; optional local dense encoder and cross-encoder.",
    "literature.search": "Host-configured PaperSeek HTTP/CLI; external context, never telemetry.",
    "graph.build": "Anything2Ontology or source-quoted LLM extraction into review-only graph candidates.",
    "model.train": "Explicitly approved offline three-task TCN training; candidate release only.",
    "model.infer_candidate": "Truth-free inference using an offline candidate and exact channel contract.",
    "exploration.propose": "Creative-only quarantined hypotheses; no code execution, knowledge release or confirmation.",
    "output.structured": "Validate task output; never let narrative rewrite a diagnosis.",
}


class MissionAgent:
    def __init__(self, settings: AgentSettings, *, backend=None):
        self.settings=AgentSettings.model_validate_json(settings.model_dump_json())
        self.project=Path(settings.project_root).resolve()
        self.read_roots=[Path(p).resolve() for p in settings.read_roots]
        self.backend=backend
        self.backend_injected=backend is not None
        self.llm_config_sha256=None
        if backend is None and settings.planner=="llm_assisted" and settings.llm_config:
            from astrograph.llm import OpenAICompatibleStructuredBackend
            p=Path(settings.llm_config).resolve()
            if not any(p.is_relative_to(root) for root in self.read_roots):raise PermissionError("LLM configuration outside approved roots")
            self.llm_config_sha256=file_sha(p)
            self.backend=OpenAICompatibleStructuredBackend.from_json(p)

    def inventory(self):
        s=self.settings
        statuses={k:"AVAILABLE" for k in CATALOG}
        statuses["document.convert"]="NATIVE_TEXT_AVAILABLE_MINERU_"+("CONFIGURED" if s.allow_mineru and s.mineru_executable and shutil.which(s.mineru_executable) else "UNAVAILABLE")
        statuses["literature.search"]="CONFIGURED_NOT_LIVE_VERIFIED" if s.allow_external_literature and (s.paperseek_url or s.paperseek_source_root) else "NOT_CONFIGURED"
        statuses["graph.build"]="CONFIGURED_NOT_LIVE_VERIFIED" if (s.allow_a2o and s.a2o_source_root) or self.backend else "NOT_CONFIGURED"
        statuses["model.train"]="APPROVAL_REQUIRED" if not s.approved_training_manifest_sha256 else "APPROVED_MANIFESTS_ONLY"
        statuses["model.capabilities"]="CONFIGURED" if s.model_capability_registry else "NOT_CONFIGURED"
        statuses["model.execute_compatible"]="CONFIGURED_STRICT_BINDINGS" if s.model_execution_bindings else "NOT_CONFIGURED"
        statuses["dataset.replay"]="AVAILABLE_OFFLINE_ONLY"
        lifecycle="CONFIGURED_EXPERT_GOVERNED" if s.lifecycle_db else "NOT_CONFIGURED"
        return {"schema_version":"astrograph-mission-capabilities.v1","tools":[{"tool_id":k,"description":v,"availability":statuses[k],"allowed_work_modes":["creative"] if k=="exploration.propose" else s.allowed_work_modes} for k,v in CATALOG.items()],
                "default_task":"diagnose","default_work_mode":"normal","allowed_work_modes":s.allowed_work_modes,"planner":s.planner,"llm_configured":self.backend is not None,
                "memory":"CONFIRMED_CASE_RETRIEVAL_PENDING_WRITE" if s.memory_db else "NOT_CONFIGURED",
                "knowledge_feedback_lifecycle":lifecycle,
                "engine":"EXISTING_TOOLREGISTRY_AND_AUTONOMOUS_DIAGNOSIS_SERVICES","new_product_dag":False,
                "live_external_services_verified":False,"active_release_write_tools":[],"control_tools":[]}

    def run(self, request: AgentRequest) -> AgentResult:
        """Default production path is supervised. Injected Python backends are explicit embedding/test mode."""
        request=AgentRequest.model_validate_json(request.model_dump_json())
        if self.backend_injected:
            return self.run_inline(request)
        return self.run_supervised(request)

    def run_inline(self, request: AgentRequest) -> AgentResult:
        """Explicit embedding/test API: Python interruption only, NOT a C/GPU hard bound."""
        request=AgentRequest.model_validate_json(request.model_dump_json())
        budget=ExecutionBudget(self.settings.task_timeout_s,self.settings.max_http_attempts)
        with execution_scope(budget):
            return _TaskRun(self,request,budget=budget).run()

    def run_supervised(self, request: AgentRequest, *, cancel=None) -> AgentResult:
        """Portable process-supervised production entry; injected Python backends stay test-only."""
        from .supervisor import run_supervised
        if self.backend is not None and not self.settings.llm_config:
            raise ValueError("SUPERVISED_RUN_REQUIRES_SERIALIZABLE_HOST_BACKEND_CONFIGURATION")
        return run_supervised(self.settings,request,cancel=cancel)


class _TaskRun:
    def __init__(self,agent,request,budget=None):
        self.agent=agent;self.request=request
        self.budget=budget or ExecutionBudget(agent.settings.task_timeout_s,agent.settings.max_http_attempts)
        self.plan_versions=[];self.plan_revision_count=0;self.snapshot_count=0
        self.active_tools=[];self.resume_cache={}
        self.s,self.mode_receipt=resolve_mode(agent.settings,request,backend_available=agent.backend is not None)
        self.exploration=None;self.query_cache={};self.optional_calls=0
        self.ws=Workspace(self.s.output_root,agent.read_roots,request.task_id,max_bytes=self.s.max_artifact_bytes)
        limit=getattr(getattr(agent.backend,"config",None),"context_length",None) or self.s.context_token_limit
        reserve=getattr(getattr(agent.backend,"config",None),"max_tokens",2400)
        reserve=min(reserve,limit-1025)
        builder=ContextBuilder(max_tokens=limit,output_reserve=reserve,artifact_writer=self._save_context)
        self.planner=Planner(agent.backend,max_calls=self.s.max_llm_calls,mode=self.s.planner,
            work_mode=request.work_mode,context_builder=builder,timeout_s=self.s.llm_timeout_s,
            audit_writer=self.ws.write_bytes,trace_mode=self.s.llm_trace_mode)
        self.docs=DocumentTools(self.ws,self.s,planner=self.planner)
        self.registry=ToolRegistry();self.step=0;self.limitations=[];self.context_sources=[];self.plugin_tool_ids=set();self.provider_tool_ids=set();self.knowledge_graph_review=None
        self.data=None;self.kernel=None;self.graph_dir=None;self.original=None;self.artifacts={};self.pinned_paths={}
        self.generic_result=None;self.capability_registry=None;self.capability_report=None;self.model_execution_resolver=None;self.model_execution_result=None;self.lifecycle_store=None
        self.reviewed_claim_enricher=None
        self.context=WorkflowContext(workflow_id="mission."+request.task,run_id=request.task_id,actor="spacecraft-diagnostic-agent",
            permissions=[PermissionScope.READ_DATA,PermissionScope.READ_KNOWLEDGE,PermissionScope.RUN_DIAGNOSIS,
                         PermissionScope.READ_ARTIFACTS,PermissionScope.WRITE_ARTIFACTS,PermissionScope.RENDER_REPORT,
                         PermissionScope.READ_MEMORY,PermissionScope.WRITE_MEMORY]+
                        ([PermissionScope.RUN_EXPERIMENT] if self.s.approved_training_manifest_sha256 or request.task=="replay" else []))

    def _bind(self):
        allowed={"diagnose":{"input","graph_dir","source","nominal","model","measurements","series_manifest"},
                 "build_knowledge":set(),"research":set(),"train_model":{"training_manifest"},
                 "infer_candidate":{"candidate_manifest","features"},
                 "replay":{"replay_manifest","dataset_root","dataset_zip"}}[self.request.task]
        if not set(self.request.artifacts)<=allowed:raise ValueError("artifact keys are not valid for this task")
        hashes={}
        for key,value in self.request.artifacts.items():
            path=self.ws.read_path(value,directory=key in {"graph_dir","dataset_root"});self.artifacts[key]=path
            if key=="graph_dir":
                hashes[key]={name:file_sha(self.ws.read_path(str(path/name))) for name in ("nodes.csv","edges.csv")}
                self.pinned_paths.update({str(path/name):digest for name,digest in hashes[key].items()})
            elif key=="dataset_root":
                # Bind a dataset identity without hashing every large trace file in the mission request.
                marker=path/"sample_index.csv"
                if not marker.is_file(): raise ValueError("dataset_root requires sample_index.csv identity")
                hashes[key]={"sample_index_sha256":file_sha(marker),"root_name":path.name}
                self.pinned_paths[str(marker)]=hashes[key]["sample_index_sha256"]
            elif key=="dataset_zip":
                hashes[key]=file_sha(path);self.pinned_paths[str(path)]=hashes[key]
            else:
                hashes[key]=file_sha(path);self.pinned_paths[str(path)]=hashes[key]
        documents=[self.ws.read_path(p) for p in self.request.documents]
        hashes["documents"]=[file_sha(p) for p in documents]
        self.pinned_paths.update({str(p):file_sha(p) for p in documents})
        if self.s.memory_policy:
            policy_path=self.ws.read_path(self.s.memory_policy)
            hashes["memory_policy"]=file_sha(policy_path)
            self.pinned_paths[str(policy_path)]=hashes["memory_policy"]
        if self.agent.llm_config_sha256 is not None:
            llm_path=self.ws.read_path(self.s.llm_config)
            hashes["llm_config"]=file_sha(llm_path)
            if hashes["llm_config"]!=self.agent.llm_config_sha256:
                raise ValueError("LLM configuration changed after backend initialization")
            self.pinned_paths[str(llm_path)]=hashes["llm_config"]
        if "model" in self.artifacts and hashes["model"] not in self.s.approved_model_sha256:
            raise PermissionError("native model not approved by host SHA-256")
        if "series_manifest" in self.artifacts:
            if len(set(self.artifacts) & {"input","source","nominal","model"})>0:
                raise ValueError("series_manifest is mutually exclusive with native diagnosis artifacts")
            manifest=GenericSeriesManifest.model_validate_json(self.artifacts["series_manifest"].read_text(encoding="utf-8"))
            data_path=Path(manifest.data_path)
            if not data_path.is_absolute(): data_path=self.artifacts["series_manifest"].parent/data_path
            data_path=self.ws.read_path(str(data_path));hashes["series_data"]=file_sha(data_path);self.pinned_paths[str(data_path)]=hashes["series_data"]
            if manifest.knowledge_edges_path:
                ep=Path(manifest.knowledge_edges_path)
                if not ep.is_absolute():ep=self.artifacts["series_manifest"].parent/ep
                ep=self.ws.read_path(str(ep));hashes["series_knowledge_edges"]=file_sha(ep);self.pinned_paths[str(ep)]=hashes["series_knowledge_edges"]
        if self.request.task=="replay":
            if "replay_manifest" not in self.artifacts: raise ValueError("replay requires replay_manifest")
            replay_manifest=ReplayManifest.model_validate_json(self.artifacts["replay_manifest"].read_text(encoding="utf-8"))
            if replay_manifest.dataset_kind=="phm_v6_retrospective":
                if "dataset_root" not in self.artifacts or "dataset_zip" in self.artifacts:
                    raise ValueError("PHM replay requires replay_manifest + dataset_root only")
                if replay_manifest.model_strategy=="promoted_extra_trees":
                    policy=self.agent.project/"configs/datasets/phm_v6_0/model_policy.json"
                    bundle=self.agent.project/"models/phm_v6_0/promoted_extra_trees_v1/manifest.json"
                    for label,path in (("phm_model_policy",policy),("phm_model_bundle_manifest",bundle)):
                        path=self.ws.read_path(str(path));hashes[label]=file_sha(path);self.pinned_paths[str(path)]=hashes[label]
                    payload=json.loads(bundle.read_text(encoding="utf-8"))
                    for head,item in payload.get("models",{}).items():
                        model_path=self.ws.read_path(str(bundle.parent/item["path"]))
                        digest=file_sha(model_path)
                        if digest!=item["sha256"]: raise ValueError("PHM promoted model bundle hash mismatch: "+head)
                        hashes["phm_model_"+head]=digest;self.pinned_paths[str(model_path)]=digest
            else:
                if "dataset_zip" not in self.artifacts or "dataset_root" in self.artifacts:
                    raise ValueError("ADCS replay requires replay_manifest + dataset_zip only")
                for label,relative in (("adcs_model","models/adcs_v5_2_2/p2_innovation_v2.pt"),
                                       ("adcs_runtime_evidence","configs/datasets/adcs_v5_2_2/runtime_sensor_checks.research.json"),
                                       ("adcs_projection","configs/datasets/adcs_v5_2_2/telemetry_projection.json")):
                    bound=self.ws.read_path(str(self.agent.project/relative));hashes[label]=file_sha(bound);self.pinned_paths[str(bound)]=hashes[label]
        if self.s.model_capability_registry:
            cp=Path(self.s.model_capability_registry)
            if not cp.is_absolute(): cp=self.agent.project/cp
            cp=self.ws.read_path(str(cp));hashes["model_capability_registry"]=file_sha(cp);self.pinned_paths[str(cp)]=hashes["model_capability_registry"]
            self.capability_registry=ModelCapabilityRegistry.from_json(cp,project_root=self.agent.project)
        if self.s.model_execution_bindings:
            bp=Path(self.s.model_execution_bindings)
            if not bp.is_absolute(): bp=self.agent.project/bp
            bp=self.ws.read_path(str(bp));hashes["model_execution_bindings"]=file_sha(bp);self.pinned_paths[str(bp)]=hashes["model_execution_bindings"]
            self.model_execution_resolver=ModelExecutionResolver.from_json(bp,project_root=self.agent.project,read_path=self.ws.read_path)
        if self.s.lifecycle_policy:
            lp=self.ws.read_path(self.s.lifecycle_policy);hashes["lifecycle_policy"]=file_sha(lp);self.pinned_paths[str(lp)]=hashes["lifecycle_policy"]
            policy=LifecyclePolicy.model_validate_json(lp.read_text(encoding="utf-8"));self.lifecycle_store=KnowledgeFeedbackStore(self.s.lifecycle_db,policy)
        if self.s.reviewed_claim_trace_path:
            tp=Path(self.s.reviewed_claim_trace_path);mp=Path(self.s.reviewed_claim_mapping_path or "")
            if not tp.is_absolute():tp=self.agent.project/tp
            if not mp.is_absolute():mp=self.agent.project/mp
            tp=self.ws.read_path(str(tp));mp=self.ws.read_path(str(mp))
            hashes["reviewed_claim_trace"]=file_sha(tp);hashes["reviewed_claim_mapping"]=file_sha(mp)
            if hashes["reviewed_claim_trace"]!=self.s.reviewed_claim_trace_sha256 or hashes["reviewed_claim_mapping"]!=self.s.reviewed_claim_mapping_sha256:
                raise ValueError("reviewed claim RAG asset hash mismatch")
            self.pinned_paths[str(tp)]=hashes["reviewed_claim_trace"];self.pinned_paths[str(mp)]=hashes["reviewed_claim_mapping"]
            self.reviewed_claim_enricher=ReviewedClaimRAGEnricher.from_binding(ReviewedClaimRAGBinding(
                claim_trace_path=str(tp),claim_trace_sha256=hashes["reviewed_claim_trace"],
                mapping_bundle_path=str(mp),mapping_bundle_sha256=hashes["reviewed_claim_mapping"]))
        if self.request.task=="train_model":
            if "training_manifest" not in self.artifacts:raise ValueError("training_manifest artifact required")
            if hashes["training_manifest"] not in self.s.approved_training_manifest_sha256:
                raise PermissionError("TRAINING_APPROVAL_REQUIRED: bind the exact manifest hash in host settings")
            from .training import TrainingManifest
            m=TrainingManifest.model_validate_json(self.artifacts["training_manifest"].read_text(encoding="utf-8"))
            p=Path(m.data_path)
            if not p.is_absolute():p=self.artifacts["training_manifest"].parent/p
            p=self.ws.read_path(str(p));hashes["training_dataset"]=file_sha(p)
            if hashes["training_dataset"]!=m.data_sha256:raise ValueError("approved dataset changed")
            self.pinned_paths[str(p)]=hashes["training_dataset"]
        if self.request.task=="infer_candidate":
            if not {"candidate_manifest","features"}<=set(self.artifacts):raise ValueError("candidate_manifest and features are required")
            weights=self.ws.read_path(str(self.artifacts["candidate_manifest"].parent/"weights.pt"))
            hashes["candidate_weights"]=file_sha(weights);self.pinned_paths[str(weights)]=hashes["candidate_weights"]
        self.bound_hashes=hashes
        return {"request_sha256":sha(self.request.model_dump(mode="json")),"settings_sha256":sha(self.agent.settings.model_dump(mode="json")),"mode_policy_sha256":self.mode_receipt["policy_sha256"],"input_hashes":hashes}

    def _register(self,tool_id,handler, *, input_schema=None,permissions=None,timeout=None):
        schema=input_schema or {"type":"object","additionalProperties":False}
        self.registry.register(ToolDefinition(ToolSpec(tool_id=tool_id,tool_version="post48.dev16",input_schema=schema,
            output_schema={"type":"object","required":["status"]},
            deterministic=tool_id not in {"graph.build","literature.search"} and not (tool_id=="diagnosis.fuse" and self.s.planner=="llm_assisted"),
            side_effect_level=SideEffectLevel.EXTERNAL_IO if tool_id=="literature.search" else SideEffectLevel.WRITE_ARTIFACT,
            required_permissions=permissions or [PermissionScope.READ_ARTIFACTS],timeout_s=timeout or self.s.subprocess_timeout_s+30,
            max_payload_bytes=4_000_000,max_output_bytes=64_000_000),handler))

    def _setup_tools(self):
        query_schema={"type":"object","properties":{"query":{"type":"string","minLength":1,"maxLength":4000}},"required":["query"],"additionalProperties":False}
        self._register("model.infer",lambda p:self._infer_native(),permissions=[PermissionScope.READ_DATA,PermissionScope.RUN_DIAGNOSIS])
        self._register("model.capabilities",lambda p:self._model_capabilities(),permissions=[PermissionScope.READ_DATA])
        self._register("model.execute_compatible",lambda p:self._execute_compatible(),permissions=[PermissionScope.READ_DATA,PermissionScope.RUN_DIAGNOSIS])
        self._register("dataset.replay",lambda p:self._replay_dataset(),permissions=[PermissionScope.READ_DATA,PermissionScope.RUN_EXPERIMENT])
        self._register("graph.reason",lambda p:self._graph_reason(),permissions=[PermissionScope.READ_KNOWLEDGE])
        from astrograph.tools.neo4j_maintenance import get_fault_mechanism, match_observations
        self._register(
            "knowledge.graph.mechanism",
            get_fault_mechanism,
            input_schema={
                "type":"object",
                "properties":{
                    "fault_id":{"type":"string","minLength":1,"maxLength":200},
                    "max_nodes":{"type":"integer","minimum":1,"maximum":100}
                },
                "required":["fault_id"],
                "additionalProperties":False
            },
            permissions=[PermissionScope.READ_KNOWLEDGE],
        )
        self._register(
            "knowledge.graph.match_observations",
            match_observations,
            input_schema={
                "type":"object",
                "properties":{
                    "anomaly_ids":{"type":"array","items":{"type":"string","minLength":1,"maxLength":200},"minItems":1,"maxItems":20}
                },
                "required":["anomaly_ids"],
                "additionalProperties":False
            },
            permissions=[PermissionScope.READ_KNOWLEDGE],
        )
        self._register("diagnosis.fuse",lambda p:self._diagnose(),permissions=[PermissionScope.READ_DATA,PermissionScope.READ_KNOWLEDGE,PermissionScope.RUN_DIAGNOSIS])
        self._register("document.convert",lambda p:self.docs.convert(self.request.documents))
        self._register("document.clean",lambda p:self.docs.clean(self.docs.documents))
        self._register("rag.retrieve",lambda p:self.docs.retrieve(p["query"]),input_schema=query_schema)
        self._register("literature.search",lambda p:self._literature(p["query"]),input_schema=query_schema)
        self._register("graph.build",lambda p:self._build_graph(),permissions=[PermissionScope.READ_ARTIFACTS,PermissionScope.WRITE_ARTIFACTS])
        self._register("model.train",lambda p:self._train(),permissions=[PermissionScope.RUN_EXPERIMENT])
        self._register("model.infer_candidate",lambda p:self._candidate_infer(),permissions=[PermissionScope.READ_DATA])
        self._register("output.structured",lambda p:self._structured_output(),permissions=[PermissionScope.RENDER_REPORT])
        if self.request.work_mode == "creative":
            self._register("exploration.propose",lambda p:self._explore(),permissions=[PermissionScope.READ_ARTIFACTS,PermissionScope.WRITE_ARTIFACTS])
        # Optional administrator-owned plugins are snapshotted before sealing. Hot reloads
        # only affect later tasks; dynamic tools always execute in isolated worker processes.
        try:
            from astrograph.tools.plugins import get_trusted_agent_tool_plugin_manager
            manager = get_trusted_agent_tool_plugin_manager()
            self.plugin_tool_ids = set(manager.register_for_task(
                self.registry, task=self.request.task, work_mode=self.request.work_mode
            ))
        except Exception as exc:
            self.limitations.append("tool_plugins:"+type(exc).__name__+":"+str(exc)[:300])
            self.plugin_tool_ids = set()
        # External MCP providers are a protocol boundary, not Python plugins. Only
        # providers explicitly registered as read-only in the SQLite control plane
        # are snapshotted into this task. Write-capable MCP tools stay blocked until
        # a dedicated approval/side-effect policy is introduced.
        try:
            from astrograph.integrations.platform_state import PlatformStateRepository
            from astrograph.tools.providers import ToolProviderManager
            provider_manager = ToolProviderManager(PlatformStateRepository.from_environment())
            provider_ids, provider_limits = provider_manager.register_read_only_for_task(self.registry)
            self.provider_tool_ids = set(provider_ids)
            self.limitations.extend(provider_limits)
        except Exception as exc:
            self.limitations.append("tool_providers:"+type(exc).__name__+":"+str(exc)[:300])
            self.provider_tool_ids = set()
        self.registry.seal()

    def _call(self,tool_id,payload=None,*,optional=False):
        if (
            tool_id not in allowed_tools(self.request.task,self.request.work_mode)
            and tool_id not in self.plugin_tool_ids
            and tool_id not in self.provider_tool_ids
        ):
            raise PermissionError("TOOL_NOT_ALLOWED_FOR_TASK_AND_MODE")
        # A zero optional budget means zero retrieval/network actions in ALL modes.
        cached_key=None
        if tool_id in {"rag.retrieve","literature.search"}:
            cached_key=(tool_id,sha(payload or {}))
            if cached_key in self.query_cache:
                self.ws.event("QUERY_CACHE_HIT",tool_id=tool_id,input_sha256=cached_key[1])
                return self.query_cache[cached_key]
            if self.optional_calls >= self.s.max_optional_actions:
                self.ws.event("OPTIONAL_BUDGET_EXHAUSTED",tool_id=tool_id)
                return {"status":"BUDGET_EXHAUSTED","hits":[]}
            self.optional_calls+=1
        resume_key=(tool_id,sha(payload or {}))
        if resume_key in self.resume_cache:
            result=self.resume_cache.pop(resume_key)
            if tool_id=="model.infer":
                if "series_manifest" in self.artifacts:
                    self.generic_result=result.get("generic_series_result")
                    if self.generic_result is None: raise RuntimeError("generic-series checkpoint lacks reconstructable result")
                else:
                    self.data=OAGInput.model_validate_json(self.artifacts["input"].read_text(encoding="utf-8"))
                    self.graph_dir=self.artifacts["graph_dir"];self.kernel=EvidenceDiagnosisKernel(self.graph_dir)
            elif tool_id=="model.capabilities":
                self.capability_report=result
            elif tool_id in {"document.convert","document.clean"}:
                self.docs.documents=result["documents"]
            self.ws.event("READ_ONLY_STEP_REUSED",tool_id=tool_id,result_sha256=sha(result))
            return result
        self.budget.remaining()
        self.step+=1
        step_id=self.step
        self.ws.event("TOOL_STARTED",step=step_id,tool_id=tool_id,input_sha256=sha(payload or {}))
        self.active_tools.append({"tool_id":tool_id,"deadline":time.monotonic()+self.s.subprocess_timeout_s+30})
        self._progress()
        try:
            if tool_id in {"model.infer","model.capabilities","model.execute_compatible","dataset.replay","graph.reason","diagnosis.fuse","graph.build","model.train","model.infer_candidate","output.structured"}:
                for path,digest in self.pinned_paths.items():
                    if file_sha(Path(path))!=digest:raise ValueError("task input changed during execution")
            result=self.registry.invoke(tool_id,payload or {},self.context)
            self.ws.write_json(f"step-{step_id:03d}.json",{"tool_id":tool_id,"input_sha256":sha(payload or {}),"result":result})
            self.ws.event("TOOL_FINISHED",step=step_id,tool_id=tool_id,status=result.get("status","UNKNOWN"),output_sha256=sha(result))
            if cached_key is not None:self.query_cache[cached_key]=result
            return result
        except Exception as exc:
            self.ws.event("TOOL_FAILED",step=step_id,tool_id=tool_id,error_type=type(exc).__name__,message=str(exc)[:600])
            if optional:
                self.limitations.append(tool_id+":"+type(exc).__name__+":"+str(exc)[:300])
                return {"status":"UNAVAILABLE","error_type":type(exc).__name__}
            raise
        finally:
            self.active_tools.pop()
            self._progress()

    def _progress(self):
        if self.ws.root.exists():
            self.ws.replace_json("execution-progress.json",{"active_tools":self.active_tools,
                "task_deadline":self.budget.deadline,"budget":self.budget.snapshot()})

    def _save_context(self,digest,payload):
        if not self.ws.root.exists():return None
        name="context-"+digest+".json"
        path=self.ws.root/name
        if not path.exists():self.ws.write_json(name,payload)
        return {"name":name,"sha256":file_sha(path)}

    def _snapshot(self,result,trace,phase):
        self.snapshot_count+=1
        value={"diagnosis_result":result,"planning_trace":trace,"phase":phase,
               "complete_investigation":False,"status":"PARTIAL"}
        name=f"diagnostic-snapshot-{self.snapshot_count:03d}.json"
        path=self.ws.write_json(name,value)
        self.ws.replace_json("diagnostic-snapshot-receipt.json",{"name":name,"sha256":file_sha(path)})
        self.latest_snapshot=value

    def _subprocess(self,args):
        result=run_managed_subprocess([sys.executable,*args],cwd=self.agent.project,capture_output=True,text=True,
                                     timeout=self.s.subprocess_timeout_s,check=False)
        if result.returncode:raise RuntimeError("WORKER_FAILED:"+(result.stderr or result.stdout or "")[-1200:])
        return result

    def _model_capabilities(self):
        if self.capability_registry is None:
            return {"status":"NOT_CONFIGURED","tasks":{},"execution_authorized":False}
        if "series_manifest" in self.artifacts:
            manifest=GenericSeriesManifest.model_validate_json(self.artifacts["series_manifest"].read_text(encoding="utf-8"))
            hint=self.model_execution_resolver.profile_hint(manifest) if self.model_execution_resolver is not None else {"domain":"generic","bound_artifact_contract":False}
            domain=str(hint.get("domain") or "generic");channels=tuple(x.name for x in manifest.channels);units={x.name:x.unit for x in manifest.channels if x.unit}
            online=True;retrospective=False
        elif self.request.task=="replay":
            manifest=ReplayManifest.model_validate_json(self.artifacts["replay_manifest"].read_text(encoding="utf-8"))
            domain="phm" if manifest.dataset_kind=="phm_v6_retrospective" else "adcs"
            channels=();units={};online=False;retrospective=True
        else:
            domain=self.s.domain;channels=();units={};online=True;retrospective=False
        bound_contract = self.request.task=="replay" or (self.request.task=="diagnose" and self.generic_result is None and bool(set(self.artifacts) & {"input","model"}))
        if "series_manifest" in self.artifacts and self.model_execution_resolver is not None:
            bound_contract = bool(self.model_execution_resolver.profile_hint(manifest).get("bound_artifact_contract"))
        profile=self.capability_registry.profile(ModelCapabilityQuery(domain=domain,channel_names=channels,units=units,
            tasks=("detection","localization","classification"),online_required=online,allow_retrospective=retrospective,
            bound_artifact_contract=bound_contract))
        result={"status":"SUCCEEDED","domain":domain,"tasks":profile["heads"],
                "coverage_tier":profile["coverage_tier"],"available_heads":profile["available_heads"],
                "missing_heads":profile["missing_heads"],"shared_three_task_model_id":profile["shared_three_task_model_id"],
                "recommended_next_step":profile["recommended_next_step"],"execution_authorized":False,
                "claim_boundary":"Per-head compatibility discovery only; model execution still requires its existing admission path."}
        self.capability_report=result;return result

    def _execute_compatible(self):
        if "series_manifest" not in self.artifacts:
            result={"status":"NOT_APPLICABLE","execution_authorized":False,"reason":"GENERIC_SERIES_ONLY_IN_DEV18_BRIDGE"}
            self.model_execution_result=result;return result
        if self.capability_report is None:
            raise RuntimeError("model capability discovery must run before compatible execution")
        if self.model_execution_resolver is None:
            result={"status":"NOT_CONFIGURED","execution_authorized":False,"reason":"NO_HOST_EXECUTION_BINDING_CATALOG"}
            self.model_execution_result=result;return result
        result=self.model_execution_resolver.execute_generic(series_manifest_path=self.artifacts["series_manifest"],
            capability_profile=self.capability_report,episode_id=self.request.task_id)
        if result.get("status")=="SUCCEEDED":
            result["three_task_contract"]=normalize_three_task_execution(result)
            result["decision_eligibility"]=assess_model_decision_eligibility(result)
            if self.generic_result is not None:
                result["shadow_fusion"]=build_generic_shadow_fusion(result,self.generic_result.get("diagnosis",{}))
        self.model_execution_result=result;return result

    def _replay_dataset(self):
        if self.request.task!="replay": raise PermissionError("dataset replay is an offline replay task only")
        manifest=ReplayManifest.model_validate_json(self.artifacts["replay_manifest"].read_text(encoding="utf-8"))
        return run_replay(project_root=self.agent.project,workspace_root=self.ws.root,
                          dataset_root=self.artifacts.get("dataset_root"),dataset_zip=self.artifacts.get("dataset_zip"),
                          manifest=manifest,run_subprocess=self._subprocess)

    def _load_approved_library(self):
        if not (self.lifecycle_store and self.s.use_approved_document_library): return {"loaded":0}
        approved=self.lifecycle_store.approved_documents()
        if not approved:return {"loaded":0}
        existing=list(self.docs.documents);converted=self.docs.convert([x["source_path"] for x in approved])["documents"]
        for doc,entry in zip(converted,approved,strict=True):
            if doc["source_sha256"]!=entry["source_sha256"]:raise ValueError("approved document hash changed during conversion")
            doc["review_status"]="APPROVED_LIBRARY";doc["library_document_id"]=entry["document_id"]
        self.docs.documents=existing+converted
        return {"loaded":len(converted),"document_ids":[x["document_id"] for x in approved]}

    def _stage_input_documents(self):
        if not self.lifecycle_store or not self.request.documents:return []
        receipts=[]
        by_sha={d["source_sha256"]:d for d in self.docs.documents}
        from datetime import timezone
        for path_value in self.request.documents:
            path=self.ws.read_path(path_value);digest=file_sha(path);doc=by_sha.get(digest)
            if doc is None: continue
            proposal=DocumentProposal(document_id=doc["document_id"],source_path=str(path),source_sha256=digest,
                title=path.name,media_type=path.suffix.lower().lstrip(".") or "unknown",parser=doc["parser"],
                proposer_actor=self.context.actor,proposed_at=datetime.now(timezone.utc),
                metadata={"task_id":self.request.task_id,"diagnostic_vote_allowed":False})
            receipts.append(self.lifecycle_store.stage_document(proposal,event_id="stage-document:"+self.request.task_id+":"+doc["document_id"]))
        if receipts:self.ws.write_json("document-lifecycle-proposals.json",receipts)
        return receipts


    def _stage_knowledge_proposal(self, candidate: dict):
        if self.lifecycle_store is None:return None
        if candidate.get("active_kg_modified") is not False:
            raise ValueError("only review-only knowledge candidates may enter lifecycle staging")
        from datetime import timezone
        candidate_sha=sha(candidate)
        proposal=KnowledgeProposal(
            proposal_id="KPROP-"+candidate_sha[:20].upper(),task_id=self.request.task_id,
            candidate_sha256=candidate_sha,candidate_payload=candidate,
            source_bindings=tuple(candidate.get("source_bindings") or candidate.get("input_source_bindings") or []),
            proposer_actor=self.context.actor,proposed_at=datetime.now(timezone.utc),
        )
        receipt=self.lifecycle_store.stage_knowledge_proposal(proposal,event_id="stage-knowledge:"+self.request.task_id+":"+proposal.proposal_id)
        self.ws.write_json("knowledge-lifecycle-proposal.json",{
            "proposal":proposal.model_dump(mode="json"),"receipt":receipt,
            "active_kg_modified":False,"active_kg_write_allowed":False,
        })
        return receipt

    def _ground_retrieval_query(self, query: str) -> str:
        """Add host-observed diagnostic semantics to diagnosis-time retrieval queries.

        This is retrieval grounding only: appended terms never become diagnostic
        evidence and documents remain non-voting.  It applies to deterministic
        and LLM-proposed RAG actions so an under-specified model query cannot
        discard the observation/candidate semantics already known to the host.
        """
        terms: list[str] = [str(query or "").strip()]
        if self.request.task != "diagnose":
            return terms[0][:4000]
        if self.data is not None:
            faults=[x.fault_id for x in self.data.diagnostic_ranking.candidates[:5]]
            names=[d.name for d in self.data.bundle.evidence_definitions if d.target_fault_id in faults]
            terms.extend([*faults,*names])
        elif self.generic_result is not None:
            layered=self.generic_result.get("diagnosis",{})
            for item in layered.get("detections",[])[:8]:
                terms.extend([str(item.get("signal") or ""),str(item.get("anomaly_id") or ""),str(item.get("anomaly_name") or "")])
            for item in layered.get("fault_candidates",[])[:5]:
                terms.extend([str(item.get("fault_id") or ""),str(item.get("fault_name") or "")])
        seen=set(); ordered=[]
        for term in terms:
            term=" ".join(str(term).split())
            key=term.casefold()
            if term and key not in seen:
                seen.add(key);ordered.append(term)
        return " ".join(ordered)[:4000]

    def _generic_core_result(self):
        layered=self.generic_result["diagnosis"]; conclusion=layered["conclusion"]; level=conclusion["level"]
        selected=conclusion.get("fault_id"); detected=bool(layered.get("detections"))
        if selected and level in {"probable","possible"}:
            # Generic seed knowledge is provisional E0 unless a future published knowledge contract states otherwise.
            # Keep the possible cause visible without upgrading it to independently verified support.
            status="CANDIDATE_SET"
        elif detected: status="INSUFFICIENT_EVIDENCE"
        else: status="NO_ANOMALY"
        candidates=[x.get("fault_id") for x in layered.get("fault_candidates",[]) if x.get("fault_id")]
        explanations=[]
        for index,item in enumerate(layered.get("fault_candidates",[])):
            explanations.append({"fault_id":item.get("fault_id"),"selected":item.get("fault_id")==selected,
                "knowledge_state":"PROVISIONAL_E0" if item.get("fault_id") else "UNCOVERED",
                "supporting_evidence_ids":[],"contradicting_evidence_ids":[],"missing_required_definition_ids":[],
                "fusion_rank":index+1,"fusion_score":item.get("score",0.0),"source_ranks":{"GENERIC_KG":index+1},
                "reciprocal_contributions":{},"dependency_adjusted":False,"graph_paths":[]})
        diagnosis={"schema_version":"astrograph-generic-series-decision.v1","policy_id":"dataset-diagnosis-policy",
            "status":status,"selected_fault_id":selected,"candidate_fault_ids":candidates,"working_hypothesis_fault_id":selected,
            "working_hypothesis_basis":"GENERIC_SERIES_ROBUST_DETECTION_AND_PROVISIONAL_KG",
            "working_hypothesis_is_confirmed":False,"requires_discriminating_evidence":detected and not selected,
            "alternatives":[x for x in candidates if x!=selected],"candidate_overflow":False,"candidate_set_is_confirmed_class":False,
            "knowledge_verified":False,"detected":detected,"data_top_fault_ids":[],"knowledge_top_fault_ids":candidates[:3],
            "fusion_top_fault_id":selected,"decision_changed_data_top":False,"blocking_conflicts":[],
            "quality_warnings":["GENERIC_ROBUST_SCORE_NOT_CALIBRATED","GENERIC_KG_EDGES_MAY_BE_PROVISIONAL"],
            "reason_codes":["GENERIC_SERIES_LAYERED_DIAGNOSIS","PROVISIONAL_CAUSE_NOT_INDEPENDENTLY_VERIFIED"],"candidate_explanations":explanations,
            "score_semantics":"ROBUST_DEVIATION_AND_DETERMINISTIC_KG_NOT_PROBABILITY","calibrated_probability":None,
            "control_actions_authorized":False,"root_cause_confirmed":False,
            "claim_boundary":conclusion.get("claim_boundary","Generic engineering diagnosis; not independently confirmed ground truth.")}
        observed_channels=[x.get("name") for x in self.generic_result.get("series_manifest",{}).get("channels",[]) if x.get("name")]
        active_diagnosis=recommend_next_observations(candidate_fault_ids=candidates,observed_channels=observed_channels,
            fault_catalog_path=self.agent.project/"configs/fault_modes/lab_fault_catalog.json")
        return {"status":"SUCCEEDED","diagnosis":diagnosis,"generic_series":layered,
                "active_diagnosis":active_diagnosis,
                "semantic_mapping_receipt":self.generic_result.get("mapping_receipt",[]),
                "single_signal_identifiability":self.generic_result.get("identifiability"),
                "generic_input_contract":self.generic_result.get("input_contract"),
                "generic_capability_boundary":self.generic_result.get("capability_boundary",{}),
                "model_capabilities":self.capability_report,"model_execution":self.model_execution_result,"planning_trace":[],
                "stopping_reason":"GENERIC_SERIES_DIAGNOSIS_COMPLETE",
                "explanation":{"conclusion":conclusion,"report_markdown":layered.get("report_markdown","")}}

    def _infer_native(self):
        if "series_manifest" in self.artifacts:
            adapter=GenericSeriesAdapter(project_root=self.agent.project,workspace_root=self.ws.root,read_path=self.ws.read_path)
            self.generic_result=adapter.run(self.artifacts["series_manifest"],request_id=self.request.task_id)
            return {"status":"SUCCEEDED","mode":"GENERIC_SERIES_ADAPTER_EXECUTED",
                    "detection_count":len(self.generic_result["diagnosis"].get("detections",[])),
                    "conclusion":self.generic_result["diagnosis"].get("conclusion"),
                    "capability_boundary":self.generic_result["capability_boundary"],
                    "generic_series_result":self.generic_result}
        if "input" in self.artifacts:
            if "graph_dir" not in self.artifacts:raise ValueError("prepared diagnosis needs graph_dir")
            if "model" in self.artifacts:raise ValueError("prepared input and model inference are mutually exclusive")
            self.data=OAGInput.model_validate_json(self.artifacts["input"].read_text(encoding="utf-8"))
            self.graph_dir=self.artifacts["graph_dir"]
            mode="BOUND_PRECOMPUTED_MODEL_OUTPUT_VALIDATED_NOT_RERUN"
        else:
            if not {"source","nominal","model"}<=set(self.artifacts):
                raise ValueError("diagnose requires bound input+graph_dir OR source+nominal+approved model")
            out=self.ws.root/"prepared-input.json";self.graph_dir=self.ws.root/"prepared-graph"
            self._subprocess(["-m","astrograph.oag","prepare-native","--domain",self.s.domain,
                "--evidence-profile",self.s.evidence_profile,"--no-mechanism-confirmation",
                "--source",str(self.artifacts["source"]),"--nominal",str(self.artifacts["nominal"]),
                "--model",str(self.artifacts["model"]),"--project-root",str(self.agent.project),
                "--graph-output",str(self.graph_dir),"--output",str(out)])
            self.data=OAGInput.model_validate_json(out.read_text(encoding="utf-8"));mode="EXISTING_NATIVE_THREE_TASK_PROVIDER_EXECUTED"
        self.kernel=EvidenceDiagnosisKernel(self.graph_dir)
        return {"status":"SUCCEEDED","mode":mode,"input_sha256":sha(self.data.model_dump(mode="json")),
                "knowledge_version":self.kernel.knowledge_version,"diagnostic_ranking":self.data.diagnostic_ranking.model_dump(mode="json"),
                "three_task_result":self.data.three_task_result.model_dump(mode="json") if self.data.three_task_result else None}

    def _graph_reason(self):
        if self.generic_result is not None:
            layered=self.generic_result["diagnosis"]
            return {"status":"SUCCEEDED","mode":"GENERIC_DATASET_KG_REASONING",
                    "fault_candidates":layered.get("fault_candidates",[]),"evidence":layered.get("evidence",[]),
                    "observations_added":False,"final_diagnosis_generated":False,
                    "claim_boundary":"Read-only generic knowledge reasoning already executed inside DatasetDiagnosisRunner."}
        from astrograph.knowledge import build_forward_evidence_query_plan, build_backward_fault_query_plan
        definitions=sorted(d.evidence_definition_id for d in self.data.bundle.evidence_definitions)
        faults=sorted({d.target_fault_id for d in self.data.bundle.evidence_definitions})
        plans=([build_forward_evidence_query_plan(definitions),build_backward_fault_query_plan(faults)]
               if definitions and faults else [])
        results=[self.kernel.executor.execute(p,knowledge_version=self.kernel.knowledge_version) for p in plans]
        if any(x.truncated for x in results):raise ValueError("graph query results truncated")
        return {"status":"SUCCEEDED","knowledge_version":self.kernel.knowledge_version,
                "query_results":[x.model_dump(mode="json") for x in results],
                "observations_added":False,"final_diagnosis_generated":False}

    def _diagnose(self):
        if self.generic_result is not None:
            result=self._generic_core_result();self.original=result
            path=self.ws.write_json("core-result.json",result);self.ws.write_json("core-result-receipt.json",{"sha256":file_sha(path),"status":"CORE_DIAGNOSIS_COMMITTED"})
            self.ws.event("CORE_COMMITTED",sha256=file_sha(path));return result
        tools=()
        if "measurements" in self.artifacts:
            bank=MeasurementBank.model_validate_json(self.artifacts["measurements"].read_text(encoding="utf-8"))
            tools=tools_from_measurements(bank,self.data.bundle)
        if ("source" in self.artifacts)!=("nominal" in self.artifacts):raise ValueError("paired sensor checks need both source and nominal")
        if "source" in self.artifacts:
            import pandas as pd
            from astrograph.oag.sensor_checks import sensor_check_tools
            tools+=sensor_check_tools(self.data,source=pd.read_csv(self.artifacts["source"],low_memory=False),
                                      nominal=pd.read_csv(self.artifacts["nominal"],low_memory=False))
        memory=None;case=None
        if self.s.memory_db and self.request.case:
            policy=MemoryPolicy.model_validate_json(self.ws.read_path(self.s.memory_policy).read_text(encoding="utf-8"))
            memory=CaseMemoryService(self.s.memory_db,policy)
            case=CaseContext(**self.request.case.model_dump())
        def selector(*,options,diagnosis,history=None,remaining_checks=None):
            # Re-query after evidence changes; no text is ever converted into evidence here.
            local=[]
            if self.docs.documents:
                query=" ".join(diagnosis.get("candidate_fault_ids",[])+diagnosis.get("data_top_fault_ids",[])) or self.request.goal
                result=self._call("rag.retrieve",{"query":query[:4000]},optional=True)
                local=result.get("hits",[])
            if self.request.work_mode!="normal" and (history or local) and self.plan_revision_count<self.s.max_plan_revisions and self.optional_calls<self.s.max_optional_actions:
                self.plan_revision_count+=1
                revised=self.planner.prepare(self.request,["rag.retrieve"] if self.docs.documents else [],
                    max_actions=max(0,self.s.max_optional_actions-self.optional_calls),
                    diagnostic_context={"diagnosis":diagnosis,"planning_trace":history or [],"new_context":local})
                version={"version":len(self.plan_versions)+1,"trigger":"CHECK_BOUNDARY_WITH_CURRENT_EVIDENCE",
                    "plan":revised.model_dump(mode="json"),"history_sha256":sha(history or [])}
                self.plan_versions.append(version)
                self.ws.write_json(f"plan-revision-{version['version']:03d}.json",version)
                for action in revised.actions:
                    extra=self._call(action.tool_id,{"query":action.query},optional=True)
                    if action.tool_id=="rag.retrieve":self.context_sources+=extra.get("hits",[])
            return self.planner.choose_check(options=options,diagnosis=diagnosis,context=local+self.context_sources,
                                            history=history,remaining_checks=remaining_checks)
        def execute_check(tool,requested):
            self.budget.remaining()
            self.active_tools.append({"tool_id":tool.tool_id,"deadline":time.monotonic()+self.s.subprocess_timeout_s})
            self._progress()
            try:
                with interruptible_timeout(self.s.subprocess_timeout_s,"CHECK_DEADLINE_EXCEEDED"):
                    return list(tool.handler(requested))
            finally:
                self.active_tools.pop();self._progress()
        result=AutonomousDiagnosisService(self.kernel,tools=tools,memory=memory,max_checks=self.s.max_checks,
            plan_selector=selector if self.s.planner=="llm_assisted" else None,
            check_executor=execute_check,progress_callback=self._snapshot).run(self.data,context=self.context,case=case)
        self.original=result
        path=self.ws.write_json("core-result.json",result)
        self.ws.write_json("core-result-receipt.json",{"sha256":file_sha(path),"status":"CORE_DIAGNOSIS_COMMITTED"})
        self.ws.event("CORE_COMMITTED",sha256=file_sha(path))
        return result

    def _literature(self,query):
        if not self.s.allow_external_literature:raise PermissionError("external literature disabled by host")
        if self.s.paperseek_source_root:
            from astrograph.oag.paperseek_cli import PaperSeekCLIClient,PaperSeekCLIConfig
            client=PaperSeekCLIClient(PaperSeekCLIConfig(source_root=self.s.paperseek_source_root,timeout_s=min(self.s.subprocess_timeout_s,180)))
        elif self.s.paperseek_url:
            from astrograph.oag.paperseek_http import PaperSeekHTTPClient,PaperSeekConfig
            client=PaperSeekHTTPClient(PaperSeekConfig(base_url=self.s.paperseek_url,timeout_s=min(self.s.subprocess_timeout_s,180)))
        else:raise RuntimeError("PAPERSEEK_NOT_CONFIGURED")
        result=client.search(query)
        for d in result.get("documents",[]):
            self.context_sources.append({"citation_id":d["document_id"],"text":d["text"],"source_uri":d["source_url"],
                "source_sha256":d["source_content_sha256"],"diagnostic_vote_allowed":False})
        return result

    def _build_graph(self):
        a2o=None
        if self.s.allow_a2o:
            from astrograph.integrations.platform_knowledge import Anything2OntologyCLIAdapter
            a2o=Anything2OntologyCLIAdapter(source_root=self.s.a2o_source_root,timeout_s=self.s.subprocess_timeout_s)
            if not a2o.available():raise RuntimeError("ANYTHING2ONTOLOGY_NOT_CONFIGURED")
        return self.docs.build_graph(a2o=a2o)

    def _train(self):
        p=self.artifacts["training_manifest"]
        if file_sha(p) not in self.s.approved_training_manifest_sha256:raise PermissionError("training manifest approval changed")
        out=self.ws.root/"model-candidate"
        self._subprocess(["-m","astrograph.agents.mission.training","train","--manifest",str(p),"--output",str(out)])
        return json.loads((out/"training_report.json").read_text(encoding="utf-8"))

    def _candidate_infer(self):
        out=self.ws.root/"candidate-inference.json"
        self._subprocess(["-m","astrograph.agents.mission.training","infer","--manifest",str(self.artifacts["candidate_manifest"]),
                          "--features",str(self.artifacts["features"]),"--output",str(out)])
        return json.loads(out.read_text(encoding="utf-8"))

    def _explore(self):
        before=sha(self.original)
        diagnosis=self.original.get("diagnosis",{}) if isinstance(self.original,dict) else {}
        proposal=self.planner.explore(diagnosis=diagnosis,
            sources=diagnostic_sources(self.original, exploration=True)+self.context_sources,
            max_proposals=self.mode_receipt["max_proposals"])
        if sha(self.original)!=before:raise ValueError("EXPLORATION_CHANGED_DIAGNOSIS")
        self.exploration={**proposal,"storage_class":"EXPLORATION_ONLY", "diagnostic_vote_allowed":False,
                          "core_output_sha256":before,"automatic_execution":False,"memory_write":"NONE"}
        self.ws.write_json("exploration.json",self.exploration)
        return self.exploration

    def _reviewed_enrich_hits(self, hits):
        if not hits or self.reviewed_claim_enricher is None:
            return list(hits or [])
        return self.reviewed_claim_enricher.enrich_hits(list(hits))

    def _structured_output(self):
        if self.request.task=="diagnose":
            if self.original is None:raise ValueError("no validated diagnosis to render")
            # Exact decision equality is checked; prose is a separate optional attachment.
            literature_context=build_literature_context_assessment(self.original,self.context_sources)
            joint=build_joint_evidence_package(core_result=self.original,model_execution=self.model_execution_result,
                generic_result=self.generic_result,literature_context=literature_context,sources=self.context_sources)
            result={"status":"SUCCEEDED","diagnosis_result":self.original,"sources":self.context_sources,
                    "literature_context":literature_context,
                    "joint_evidence_package":joint.model_dump(mode="json"),
                    "joint_evidence_report":joint_evidence_report(joint)}
            if self.s.planner=="llm_assisted":
                sources=diagnostic_sources(self.original)+self.context_sources
                result["narrative"]=self.planner.explain(self.original["diagnosis"],sources)
            if self.request.work_mode != "normal":result["investigation"]=investigation_brief(self.original)
            if self.knowledge_graph_review is not None:
                result["knowledge_graph_review"]={**self.knowledge_graph_review,"diagnostic_vote_allowed":False,"core_diagnosis_modified":False}
            if self.exploration is not None:result["exploration"]=self.exploration
            return result
        result={"status":"SUCCEEDED","task_output":self.original,"sources":self.context_sources}
        if self.exploration is not None:result["exploration"]=self.exploration
        return result

    def run(self):
        identity=self._bind()
        can_resume=self.s.resume_read_only and self.s.planner=="deterministic" and not self.s.memory_db
        saved=self.ws.begin(identity,resume_read_only=can_resume)
        if saved is not None and not saved.get("__resume_read_only__"):
            return AgentResult.model_validate(saved)
        if saved is not None:
            self.step=saved["max_step"]
            self.resume_cache={(x["tool_id"],x.get("input_sha256",sha({}))):x["result"] for x in saved["steps"]}
        if not (self.ws.root/"mode.json").exists():self.ws.write_json("mode.json",self.mode_receipt)
        self.ws.event("MODE_RESOLVED",**self.mode_receipt)
        self._setup_tools();plan=None;output={};status="SUCCEEDED"
        try:
            diagnostic_context=None
            if self.request.task=="diagnose":
                self._call("model.infer")
                self._call("model.capabilities")
                if self.s.model_execution_bindings:
                    self._call("model.execute_compatible")
                else:
                    self.model_execution_result={"status":"NOT_CONFIGURED","execution_authorized":False,"reason":"NO_HOST_EXECUTION_BINDING_CATALOG"}
                self._call("graph.reason")
                if self.generic_result is not None:
                    diagnostic_context={"generic_conclusion":self.generic_result["diagnosis"].get("conclusion"),
                        "detections":self.generic_result["diagnosis"].get("detections",[]),
                        "model_capabilities":self.capability_report,"model_execution":self.model_execution_result}
                else:
                    diagnostic_context={
                        "data_ranking":self.data.diagnostic_ranking.model_dump(mode="json"),
                        "evidence_states":[{"definition_id":f.evidence_definition_id,"fault_id":f.target_fault_id,
                            "role":f.role.value,"status":f.status.value} for f in self.data.bundle.evidence_facts],
                        "window_end_s":self.data.bundle.episode.end_s,
                        "knowledge_version":self.kernel.knowledge_version,
                    }
            # Document work is optional for diagnosis, mandatory for a requested knowledge build.
            if self.request.documents:
                optional=self.request.task=="diagnose"
                converted=self._call("document.convert",optional=optional)
                if converted.get("status")=="SUCCEEDED":
                    self._call("document.clean",optional=optional)
                    self._stage_input_documents()
            library_receipt=self._load_approved_library()
            if library_receipt.get("loaded"):
                self.ws.write_json("approved-library-load.json",library_receipt)
            available=[]
            if self.docs.documents:available.append("rag.retrieve")
            if self.s.allow_external_literature and (self.s.paperseek_url or self.s.paperseek_source_root):available.append("literature.search")
            plan=self.planner.prepare(self.request,available,max_actions=self.s.max_optional_actions,diagnostic_context=diagnostic_context)
            if not (self.ws.root/"plan.json").exists():self.ws.write_json("plan.json",plan.model_dump(mode="json"))
            elif json.loads((self.ws.root/"plan.json").read_text(encoding="utf-8"))!=plan.model_dump(mode="json"):
                raise ValueError("read-only resume plan changed")
            self.plan_versions.append({"version":1,"trigger":"INITIAL","plan":plan.model_dump(mode="json")})
            # Deterministic template still retrieves supplied context; optional actions never bypass core.
            actions=[x.model_dump(mode="json") for x in plan.actions]
            if self.s.max_optional_actions>0 and self.docs.documents and not any(x["tool_id"]=="rag.retrieve" for x in actions):
                actions=[{"tool_id":"rag.retrieve","query":self._ground_retrieval_query(self.request.goal),"rationale":"Retrieve source context grounded in observed evidence and competing candidates."}]+actions
            if self.request.task=="research" and not actions:
                research_actions=[]
                if "rag.retrieve" in available:
                    research_actions.append({"tool_id":"rag.retrieve","query":self.request.goal,
                        "rationale":"Search the host-approved local/approved document library for the explicit research task."})
                if "literature.search" in available and len(research_actions)<self.s.max_optional_actions:
                    research_actions.append({"tool_id":"literature.search","query":self.request.goal,
                        "rationale":"Search the host-configured external literature source for the explicit research task."})
                actions=research_actions
            for action in actions[:self.s.max_optional_actions]:
                action_query=action["query"]
                if action["tool_id"]=="rag.retrieve":
                    action_query=self._ground_retrieval_query(action_query)
                result=self._call(action["tool_id"],{"query":action_query},optional=True)
                if action["tool_id"]=="rag.retrieve":
                    self.context_sources+=self._reviewed_enrich_hits(result.get("hits",[]))
            if self.request.task=="diagnose":
                self._call("diagnosis.fuse")
                # Deep/extended review may consult the configured local Neo4j graph
                # after the core diagnosis is frozen. This read-only result is an
                # attachment for maintenance review and cannot change the diagnosis.
                if self.request.work_mode != "normal":
                    selected = ((self.original or {}).get("diagnosis") or {}).get("selected_fault_id")
                    if selected:
                        self.knowledge_graph_review = self._call(
                            "knowledge.graph.mechanism", {"fault_id": selected, "max_nodes": 40}, optional=True
                        )
                        if self.ws.root.exists():
                            self.ws.write_json("knowledge-graph-review.json", self.knowledge_graph_review)
            elif self.request.task=="build_knowledge":
                if not self.docs.documents:raise ValueError("graph construction requires source documents")
                self.original=self._call("graph.build")
                if self.original.get("status") in {"REVIEW_REQUIRED","PARTIAL_REVIEW_REQUIRED"}:
                    lifecycle_receipt=self._stage_knowledge_proposal(self.original)
                    if lifecycle_receipt:self.original={**self.original,"lifecycle":lifecycle_receipt}
                if self.original.get("status")!="REVIEW_REQUIRED":status="PARTIAL"
            elif self.request.task=="train_model":self.original=self._call("model.train")
            elif self.request.task=="infer_candidate":self.original=self._call("model.infer_candidate")
            elif self.request.task=="replay":
                self._call("model.capabilities")
                self.original=self._call("dataset.replay")
            else:
                self.original={"status":"SUCCEEDED" if self.context_sources else "NO_RESULTS","sources":self.context_sources,
                               "diagnostic_conclusion_generated":False}
                if not self.context_sources:status="PARTIAL"
            if self.request.work_mode == "creative":self._call("exploration.propose",optional=True)
            output=self._call("output.structured")
            if self.limitations and status=="SUCCEEDED":status="PARTIAL"
        except Exception as exc:
            status="BLOCKED" if isinstance(exc,(PermissionError,ValueError)) else "FAILED"
            output={"error_type":type(exc).__name__,"message":str(exc)[:1500]}
            if self.request.task=="diagnose" and self.original is not None:
                output["diagnosis_result"]=self.original
                output["attachment_status"]="FAILED_CORE_PRESERVED"
                status="PARTIAL"
            elif self.request.task=="diagnose" and hasattr(self,"latest_snapshot"):
                output["diagnosis_result"]=self.latest_snapshot["diagnosis_result"]
                output["investigation_completion"]="INTERRUPTED_SNAPSHOT_ONLY"
                status="PARTIAL"
            self.limitations.append("TASK_STAGE_FAILURE:"+type(exc).__name__)
            self.ws.event("TASK_ERROR",error_type=type(exc).__name__,message=str(exc)[:600])
        llm_records=self.planner.records
        accepted=sum(x["outcome"] in {"ACCEPTED","STRUCTURE_AND_CITATIONS_VALIDATED","QUARANTINED_PROPOSAL"} for x in llm_records)
        failed=sum(x["outcome"]=="DETERMINISTIC_FALLBACK" for x in llm_records)
        abstained=sum(x["outcome"]=="ABSTAINED" for x in llm_records)
        if self.s.planner=="llm_assisted":
            llm_status="SUCCEEDED" if accepted and not (failed or abstained) else ("PARTIAL" if accepted else "UNAVAILABLE_OR_FAILED")
            self.mode_receipt["planner_effective"]="llm_assisted" if accepted and not (failed or abstained) else ("mixed_llm_and_fallback" if accepted else "deterministic_fallback")
            if failed or abstained:
                reason="LLM_CAPABILITY_DEGRADED: inspect planner_records; fallback is not LLM acceptance"
                self.limitations.append(reason);self.mode_receipt["capability_degradations"].append(reason)
                if status=="SUCCEEDED":status="PARTIAL"
        else:llm_status="NOT_REQUESTED"
        self.ws.write_json("context-receipts.json",self.planner.context_builder.records)
        capability={"core_diagnosis":"SUCCEEDED" if self.original and self.request.task=="diagnose" else "NOT_COMPLETED",
                    "llm":{"status":llm_status,"accepted":accepted,"fallbacks":failed,"abstentions":abstained},
                    "rag_cache":self.docs.cache_stats,"attachment":"FAILED" if output.get("attachment_status") else "COMPLETED_OR_NOT_REQUESTED",
                    "document_provider_quality":document_provider_quality_receipt(self.s)}
        result=AgentResult(task_id=self.request.task_id,task=self.request.task,status=status,
            work_mode=self.request.work_mode,mode_receipt=self.mode_receipt,
            request_sha256=identity["request_sha256"],settings_sha256=identity["settings_sha256"],
            plan={"template":self.request.task,"proposal":plan.model_dump(mode="json") if plan else None,
                  "planner_records":self.planner.records,"llm_calls":self.planner.calls,"plan_versions":self.plan_versions,
                  "default_diagnosis_chain":["model.infer","model.capabilities","model.execute_compatible","graph.reason","diagnosis.fuse / eligible checks / recompute","output.structured","pending case only"]},
            output=output,tool_receipts=self.registry.executor.receipts(),limitations=self.limitations,
            capability_status=capability,execution_budget=self.budget.snapshot())
        result.execution_budget["enforcement"]="PROCESS_SUPERVISOR" if os.environ.get("ASTROGRAPH_PROCESS_SUPERVISED")=="1" else "INLINE_PYTHON_INTERRUPTION_ONLY"
        self.ws.finish(result.model_dump(mode="json"))
        return result
