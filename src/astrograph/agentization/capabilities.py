from __future__ import annotations
from dataclasses import dataclass
import datetime as dt, json, os, sys
from pathlib import Path
from typing import Any, Callable
from .artifacts import verify_file, scan_zip, resolve_scoped, sha256_file
from .evidence_agent import propose_card, validate_card
from .policy import authorize
from .profiles import AgentProfile, PROFILE_BOUNDARIES
from .protocol import CapabilityProtocolError, PROJECT_STATUS, PROTOCOL_VERSION, sha256_json
from .readiness import audit_roots
from .runtime_bridge import AgentRuntimeBridge, RUNTIME_EXECUTE_INPUT_SCHEMA
from .shadow import compare_plan
from .workflows import load_workflows, validate_workflow

Handler=Callable[[dict[str,Any],"CapabilityEnvironment"],dict[str,Any]]

@dataclass(frozen=True)
class CapabilityDefinition:
    name: str
    description: str
    input_schema: dict[str,Any]
    output_schema: dict[str,Any]
    allowed_profiles: frozenset[str]
    handler: Handler
    side_effect: str="NONE"
    label_access: str="NONE"
    policy_selection: bool=False
    availability: str="AVAILABLE"

    def public(self):
        return {"name":self.name,"description":self.description,"input_schema":self.input_schema,"output_schema":self.output_schema,"allowed_profiles":sorted(self.allowed_profiles),"side_effect":self.side_effect,"label_access":self.label_access,"policy_selection":self.policy_selection,"availability":self.availability}

@dataclass
class CapabilityEnvironment:
    project_root: Path
    allowed_roots: tuple[Path,...]
    artifact_dir: Path
    workflow_root: Path
    external_audit_roots: tuple[Path,...]
    runtime_factory: Callable[[], Any] | None = None

class CapabilityRegistry:
    def __init__(self,env:CapabilityEnvironment): self.env=env; self._defs={}
    def register(self,d:CapabilityDefinition):
        if d.name in self._defs: raise RuntimeError(f"duplicate capability {d.name}")
        self._defs[d.name]=d
    def manifest(self):
        caps=[self._defs[k].public() for k in sorted(self._defs)]
        payload={"protocol_version":PROTOCOL_VERSION,"project_status":PROJECT_STATUS,"capabilities":caps}
        payload["manifest_sha256"]=sha256_json(payload)
        return payload
    def invoke(self,name:str,profile:str,args:dict[str,Any]):
        d=self._defs.get(name)
        if d is None: raise CapabilityProtocolError("UNKNOWN_CAPABILITY",f"unknown capability {name!r}")
        authorize(d,profile,args)
        return d.handler(args,self.env)

OBJ={"type":"object","additionalProperties":True}
EMPTY={"type":"object","properties":{},"additionalProperties":False}
ALL=frozenset(p.value for p in AgentProfile)
NO_TRUTH=frozenset({AgentProfile.DIAGNOSIS.value,AgentProfile.EVIDENCE_CURATION.value,AgentProfile.MAINTAINER.value,AgentProfile.SHADOW.value})

def build_registry(env:CapabilityEnvironment)->CapabilityRegistry:
    r=CapabilityRegistry(env)
    def reg(name,desc,handler,profiles=ALL,side="NONE",label="NONE",select=False,availability="AVAILABLE",schema=OBJ):
        r.register(CapabilityDefinition(name,desc,schema,OBJ,frozenset(profiles),handler,side,label,select,availability))

    reg("system.describe","Describe immutable project status and scientific boundaries.",lambda a,e:{"protocol_version":PROTOCOL_VERSION,"project_status":PROJECT_STATUS,"stable_baseline":"B005","current_snapshot":"B014_SEALED_CAMPAIGN_PREREGISTERED_NOT_EXECUTED","development_groups":["3","8"],"permanent_posthoc_groups":["4","9"],"new_confirmatory_evidence":"NONE","qualification_changes":"NONE"})
    reg("profiles.describe","Return profile-specific truth and mutation boundaries.",lambda a,e:{"profiles":PROFILE_BOUNDARIES})
    reg("artifact.verify","Verify a file SHA-256 within configured roots.",lambda a,e:verify_file(str(a.get("path","")),str(a.get("expected_sha256","")),e.allowed_roots),schema={"type":"object","required":["path","expected_sha256"],"properties":{"path":{"type":"string"},"expected_sha256":{"type":"string","pattern":"^[0-9a-fA-F]{64}$"}},"additionalProperties":False})
    reg("dataset.intake_archive","Read-only ZIP safety and provenance intake; never deserializes payloads.",lambda a,e:scan_zip(str(a.get("path","")),e.allowed_roots),schema={"type":"object","required":["path"],"properties":{"path":{"type":"string"}},"additionalProperties":False})
    reg("workflows.list","List persistent AstroGraph agent workflows.",lambda a,e:{"workflows":sorted(load_workflows(e.workflow_root))})
    reg("workflows.get","Return a content-addressed workflow DAG.",lambda a,e:load_workflows(e.workflow_root).get(str(a.get("workflow_id",""))) or (_ for _ in ()).throw(CapabilityProtocolError("WORKFLOW_NOT_FOUND","unknown workflow")),schema={"type":"object","required":["workflow_id"],"properties":{"workflow_id":{"type":"string"}},"additionalProperties":False})
    reg("workflows.validate","Validate DAG acyclicity and references; this is structural, not scientific validation.",lambda a,e:validate_workflow(dict(a.get("workflow",{}))),schema={"type":"object","required":["workflow"],"properties":{"workflow":{"type":"object"}},"additionalProperties":False})
    runtime_bridge=AgentRuntimeBridge(env)
    reg("runtime.describe","Describe the canonical deterministic Runtime pipelines, stages and Agent execution boundary.",lambda a,e:runtime_bridge.describe(),profiles={AgentProfile.DIAGNOSIS.value,AgentProfile.SHADOW.value,AgentProfile.MAINTAINER.value},schema=EMPTY)
    reg("runtime.reference_plan","Return the executable atomic diagnosis workflow without granting execution authority.",lambda a,e:{"workflow_id":"astrograph-diagnosis-v1","stages":["runtime.describe","runtime.diagnosis.execute"],"execution":"AVAILABLE_TO_DIAGNOSIS_PROFILE_VIA_DETERMINISTIC_RUNTIME","individual_stage_invocation_allowed":False},profiles={AgentProfile.DIAGNOSIS.value,AgentProfile.SHADOW.value,AgentProfile.MAINTAINER.value})
    reg("runtime.diagnosis.execute","Execute an admitted RuntimeRequest atomically through AstroGraphRuntime and return all governed stage receipts.",lambda a,e:runtime_bridge.execute(a),profiles={AgentProfile.DIAGNOSIS.value,AgentProfile.MAINTAINER.value},side="DIAGNOSTIC_EXECUTION",schema=RUNTIME_EXECUTE_INPUT_SCHEMA)
    def shadow(a,e):
        reference=load_workflows(e.workflow_root).get(str(a.get("workflow_id","astrograph-diagnosis-v1")))
        if not reference: raise CapabilityProtocolError("WORKFLOW_NOT_FOUND","shadow workflow not found")
        order=validate_workflow(reference)["topological_order"]
        allowed={str(n.get("capability",n.get("id"))) for n in reference["nodes"]}
        return compare_plan(list(a.get("proposed_tools",[])),order,allowed).to_mapping()
    reg("agent.shadow.validate_plan","Validate an LLM-proposed plan against a deterministic reference without granting execution authority.",shadow,profiles={AgentProfile.SHADOW.value,AgentProfile.MAINTAINER.value},schema={"type":"object","required":["proposed_tools"],"properties":{"workflow_id":{"type":"string"},"proposed_tools":{"type":"array","items":{"type":"string"}}},"additionalProperties":False})
    reg("authoritative_p2.audit","Search supplied roots and archives for model/temporal candidates without promoting filename matches to authoritative evidence.",lambda a,e:audit_roots(e.external_audit_roots),profiles={AgentProfile.MAINTAINER.value})
    def verify_json_artifact(a,e):
        rec=verify_file(str(a.get("path","")),str(a.get("expected_sha256","")),e.allowed_roots)
        raw=json.loads(Path(rec["path"]).read_text(encoding="utf-8"))
        return {"artifact":rec,"schema_version":raw.get("schema_version"),"group_fields":[{"path":p,"value":g} for p,g in _groups(raw)],"content_sha256":sha256_json(raw)}
    def _groups(value,path=""):
        out=[]
        if isinstance(value,dict):
            for k,v in value.items():
                kp=f"{path}.{k}" if path else str(k)
                if "group" in str(k).lower():
                    if isinstance(v,(str,int)): out.append((kp,str(v)))
                    elif isinstance(v,list): out.extend((kp,str(x)) for x in v if isinstance(x,(str,int)))
                out.extend(_groups(v,kp))
        elif isinstance(value,list):
            for i,v in enumerate(value): out.extend(_groups(v,f"{path}[{i}]"))
        return out
    reg("research.artifact.verify_json","Verify and inspect a research JSON artifact before evaluator use.",verify_json_artifact,profiles={AgentProfile.DEVELOPMENT.value,AgentProfile.POSTHOC_AUDIT.value,AgentProfile.MAINTAINER.value},label="OFFLINE_ONLY",schema={"type":"object","required":["path","expected_sha256"],"properties":{"path":{"type":"string"},"expected_sha256":{"type":"string"}},"additionalProperties":False})
    def selection_gate(a,e):
        required=a.get("required_artifacts",[]); missing=[]; receipts=[]
        for item in required:
            try: receipts.append(verify_file(str(item.get("path","")),str(item.get("sha256","")),e.allowed_roots))
            except CapabilityProtocolError as exc: missing.append({"path":item.get("path"),"code":exc.code})
        return {"status":"READY" if not missing else "BLOCKED","verified_artifacts":receipts,"missing_or_invalid":missing,"selection_groups":["3","8"],"posthoc_groups_excluded":["4","9"],"policy_selected":False}
    reg("research.route_a.selection_gate","Verify frozen development inputs before deterministic Route A selection.",selection_gate,profiles={AgentProfile.DEVELOPMENT.value,AgentProfile.MAINTAINER.value},label="OFFLINE_ONLY",select=True,schema={"type":"object","required":["required_artifacts","selection_groups"],"properties":{"required_artifacts":{"type":"array","items":{"type":"object"}},"selection_groups":{"type":"array","items":{"type":["string","integer"]}}},"additionalProperties":True})
    def evidence_propose(a,e):
        src=verify_file(str(a.get("source_path","")),str(a.get("source_sha256","")),e.allowed_roots)
        return propose_card(a,src)
    reg("evidence.card.propose","Draft a source-pinned evidence card. Agent output cannot become frozen evidence.",evidence_propose,profiles={AgentProfile.EVIDENCE_CURATION.value,AgentProfile.MAINTAINER.value},side="ARTIFACT_WRITE",schema={"type":"object","required":["source_path","source_sha256","fault_id","evidence_type","telemetry_fields","predicate","applicability_envelope"],"additionalProperties":True})
    reg("evidence.card.validate","Validate evidence-card structure and enforce the automatic promotion ceiling.",lambda a,e:validate_card(dict(a.get("card",{})),a.get("requested_state")),profiles={AgentProfile.EVIDENCE_CURATION.value,AgentProfile.MAINTAINER.value},schema={"type":"object","required":["card"],"properties":{"card":{"type":"object"},"requested_state":{"type":"string"}},"additionalProperties":False})
    def confirmatory(a,e):
        manifest=dict(a.get("preregistration",{})); required=("source_tree_sha256","generator_sha256","model_package_sha256","temporal_policy_sha256","evidence_package_sha256","fusion_policy_sha256","rejection_policy_sha256","evaluator_sha256","statistics_plan_sha256")
        missing=[k for k in required if not isinstance(manifest.get(k),str) or len(manifest[k])!=64]
        flags={"new_campaign_id":bool(manifest.get("campaign_id")),"hidden_truth":manifest.get("hidden_truth") is True,"one_time_evaluation":manifest.get("one_time_evaluation") is True,"not_previously_accessed":manifest.get("not_previously_accessed") is True}
        ready=not missing and all(flags.values())
        return {"status":"READY_TO_EXECUTE_ONCE" if ready else "BLOCKED_NOT_READY_TO_GENERATE","missing_hashes":missing,"required_flags":flags,"execution_performed":False,"prohibited_shortcut":"No policy mutation or use of groups 4/9 is allowed."}
    reg("confirmatory.campaign.verify_preregistration","Verify a sealed confirmatory preregistration. Does not generate or evaluate a campaign.",confirmatory,profiles={AgentProfile.CONFIRMATORY_SEALED.value,AgentProfile.MAINTAINER.value},label="OFFLINE_ONLY",schema={"type":"object","required":["preregistration"],"properties":{"preregistration":{"type":"object"}},"additionalProperties":False})
    def temporal_campaign_status(a,e):
        plan=e.project_root/"configs/research/adcs_temporal_confirmatory_analysis_plan_r001.json"
        return {"status":"FROZEN_RUNTIME_BLOCKED" if plan.exists() else "NOT_PREPARED","analysis_plan":json.loads(plan.read_text(encoding="utf-8")) if plan.exists() else None,"groups_4_9_read":False,"execution_performed":False,"generation_consumed":False,"evaluation_consumed":False,"new_confirmatory_evidence":"NONE"}
    reg("confirmatory.temporal_campaign.status","Return the merged B014 temporal campaign status without generating or evaluating data.",temporal_campaign_status,profiles={AgentProfile.CONFIRMATORY_SEALED.value,AgentProfile.MAINTAINER.value},label="OFFLINE_ONLY",schema=EMPTY)
    reg("confirmatory.campaign.execute_once","Execute B014 only through a separately authorized one-time external runner; in-process Agent execution remains prohibited.",lambda a,e:{"status":"FROZEN_NOT_EXECUTED","reason":"SEPARATE_ONE_TIME_EXTERNAL_AUTHORIZATION_REQUIRED","execution_performed":False,"generation_consumed":False,"evaluation_consumed":False},profiles={AgentProfile.CONFIRMATORY_SEALED.value},side="SEALED_ONE_TIME_EXECUTION",availability="EXTERNAL_ONE_TIME_AUTHORIZATION_REQUIRED")
    def model_independent_describe(a,e):
        policy_path=e.project_root/"configs/research/adcs_model_independent_temporal_policy_r001.json"
        assessment_path=e.project_root/"research/model_independent_temporal_r001/thesis_contribution_assessment.json"
        policy=json.loads(policy_path.read_text(encoding="utf-8")) if policy_path.exists() else None
        assessment=json.loads(assessment_path.read_text(encoding="utf-8")) if assessment_path.exists() else None
        return {"status":"DEVELOPMENT_EVIDENCE","selection_groups":["3","8"],"excluded_groups":["4","9"],"runtime_ground_truth_access":False,"historical_p2_required":False,"selected_policy":policy,"thesis_assessment":assessment,"new_confirmatory_evidence":"NONE"}
    reg("temporal.model_independent.describe","Describe the frozen groups-3/8 classifier-independent temporal route and its claim boundary.",model_independent_describe,profiles={AgentProfile.DIAGNOSIS.value,AgentProfile.DEVELOPMENT.value,AgentProfile.SHADOW.value,AgentProfile.MAINTAINER.value})
    def verify_trace_index(a,e):
        rec=verify_file(str(a.get("path","")),str(a.get("expected_sha256","")),e.allowed_roots)
        raw=json.loads(Path(rec["path"]).read_text(encoding="utf-8"))
        if set(str(x) for x in raw.get("development_groups",[]))!={"3","8"}: raise CapabilityProtocolError("DEVELOPMENT_GROUP_CONTRACT_VIOLATION","trace index must contain exactly groups 3/8")
        if set(str(x) for x in raw.get("excluded_groups",[]))!={"4","9"}: raise CapabilityProtocolError("POSTHOC_GROUP_CONTRACT_VIOLATION","trace index must exclude groups 4/9")
        if raw.get("truth_access_during_generation") is not False: raise CapabilityProtocolError("LABEL_ACCESS_FORBIDDEN","trace generation truth access must be false")
        return {"status":"VERIFIED","artifact":rec,"index_sha256":raw.get("index_sha256"),"truth_access_during_generation":False,"policy_selection_performed":False}
    reg("temporal.model_independent.verify_trace_index","Verify a frozen label-free trace index before offline evaluation.",verify_trace_index,profiles={AgentProfile.DEVELOPMENT.value,AgentProfile.MAINTAINER.value},label="OFFLINE_ONLY",schema={"type":"object","required":["path","expected_sha256"],"properties":{"path":{"type":"string"},"expected_sha256":{"type":"string","pattern":"^[0-9a-fA-F]{64}$"}},"additionalProperties":False})
    def mismatch_tolerant_describe(a,e):
        policy_path=e.project_root/"configs/research/adcs_mismatch_tolerant_temporal_policy_r001.json"
        assessment_path=e.project_root/"research/mismatch_tolerant_temporal_r001/thesis_contribution_assessment.json"
        policy=json.loads(policy_path.read_text(encoding="utf-8")) if policy_path.exists() else None
        assessment=json.loads(assessment_path.read_text(encoding="utf-8")) if assessment_path.exists() else None
        return {"status":"DEVELOPMENT_EVIDENCE","selection_groups":["3","8"],"excluded_groups":["4","9"],"groups_4_9_read":False,"runtime_ground_truth_access":False,"historical_p2_required":False,"method_scope":"BOUNDED_PREFIX_ALIGNMENT_AND_SELECTIVE_REFERENCE_ADMISSION","selected_policy":policy,"thesis_assessment":assessment,"arbitrary_cross_reference_robustness_established":False,"new_confirmatory_evidence":"NONE"}
    reg("temporal.mismatch_tolerant.describe","Describe the frozen groups-3/8 bounded mismatch-tolerant temporal route and its claim boundary.",mismatch_tolerant_describe,profiles={AgentProfile.DIAGNOSIS.value,AgentProfile.DEVELOPMENT.value,AgentProfile.SHADOW.value,AgentProfile.MAINTAINER.value})
    def verify_mismatch_trace_index(a,e):
        rec=verify_file(str(a.get("path","")),str(a.get("expected_sha256","")),e.allowed_roots)
        raw=json.loads(Path(rec["path"]).read_text(encoding="utf-8"))
        if set(str(x) for x in raw.get("development_groups",[]))!={"3","8"}: raise CapabilityProtocolError("DEVELOPMENT_GROUP_CONTRACT_VIOLATION","trace index must contain exactly groups 3/8")
        if set(str(x) for x in raw.get("excluded_groups",[]))!={"4","9"}: raise CapabilityProtocolError("POSTHOC_GROUP_CONTRACT_VIOLATION","trace index must exclude groups 4/9")
        if raw.get("truth_access_during_generation") is not False: raise CapabilityProtocolError("LABEL_ACCESS_FORBIDDEN","trace generation truth access must be false")
        if raw.get("outcome_data_read_during_generation") is not False: raise CapabilityProtocolError("OUTCOME_ACCESS_FORBIDDEN","trace generation outcome access must be false")
        if raw.get("full_trajectory_alignment_used") is not False: raise CapabilityProtocolError("NONCAUSAL_ALIGNMENT_FORBIDDEN","full-trajectory alignment is forbidden")
        if raw.get("padded_edge_action") != "INVALIDATE": raise CapabilityProtocolError("ALIGNMENT_EDGE_CONTRACT_VIOLATION","padded alignment edges must be invalidated")
        return {"status":"VERIFIED","artifact":rec,"index_sha256":raw.get("index_sha256"),"truth_access_during_generation":False,"full_trajectory_alignment_used":False,"policy_selection_performed":False}
    reg("temporal.mismatch_tolerant.verify_trace_index","Verify a frozen label-free bounded-alignment trace index before offline evaluation.",verify_mismatch_trace_index,profiles={AgentProfile.DEVELOPMENT.value,AgentProfile.MAINTAINER.value},label="OFFLINE_ONLY",schema={"type":"object","required":["path","expected_sha256"],"properties":{"path":{"type":"string"},"expected_sha256":{"type":"string","pattern":"^[0-9a-fA-F]{64}$"}},"additionalProperties":False})
    def temporal_development_describe(policy_name,assessment_name,method_scope):
        def describe(a,e):
            policy_path=e.project_root/"configs/research"/policy_name
            assessment_path=e.project_root/"research"/assessment_name/"thesis_contribution_assessment.json"
            return {"status":"DEVELOPMENT_EVIDENCE","selection_groups":["3","8"],"calibration_groups":["0","1","2","5","6","7"],"excluded_groups":["4","9"],"groups_4_9_read":False,"runtime_ground_truth_access":False,"method_scope":method_scope,"selected_policy":json.loads(policy_path.read_text(encoding="utf-8")) if policy_path.exists() else None,"thesis_assessment":json.loads(assessment_path.read_text(encoding="utf-8")) if assessment_path.exists() else None,"fault_identification_established":False,"probabilistic_or_conformal_guarantee":False,"new_confirmatory_evidence":"NONE"}
        return describe
    def verify_development_index(a,e,*,require_reference_isolation=False):
        rec=verify_file(str(a.get("path","")),str(a.get("expected_sha256","")),e.allowed_roots)
        raw=json.loads(Path(rec["path"]).read_text(encoding="utf-8"))
        if set(str(x) for x in raw.get("development_groups",[]))!={"3","8"}: raise CapabilityProtocolError("DEVELOPMENT_GROUP_CONTRACT_VIOLATION","groups 3/8 required")
        if set(str(x) for x in raw.get("excluded_groups",[]))!={"4","9"} or raw.get("groups_4_9_read") is not False: raise CapabilityProtocolError("POSTHOC_GROUP_ACCESS_FORBIDDEN","groups 4/9 forbidden")
        calibration={str(x) for x in raw.get("calibration_groups",[])}
        if calibration and calibration!={"0","1","2","5","6","7"}: raise CapabilityProtocolError("CALIBRATION_GROUP_CONTRACT_VIOLATION","calibration groups differ")
        if raw.get("runtime_ground_truth_access",raw.get("truth_access_during_generation")) is not False or raw.get("outcome_data_read_during_generation") is not False: raise CapabilityProtocolError("LABEL_ACCESS_FORBIDDEN","runtime/evaluator access forbidden")
        if require_reference_isolation:
            if raw.get("target_provenance_group_exclusion_enforced") is not True or raw.get("prefix_template_identity_verified") is not True: raise CapabilityProtocolError("REFERENCE_PROVENANCE_LEAKAGE","target provenance group exclusion or prefix identity is incomplete")
            if raw.get("full_trajectory_alignment_used") is not False: raise CapabilityProtocolError("NONCAUSAL_ALIGNMENT_FORBIDDEN","full-trajectory alignment is forbidden")
        elif raw.get("candidate_selections_frozen_before_truth_join") is not True:
            raise CapabilityProtocolError("PREDICTION_FREEZE_REQUIRED","candidate selections must be frozen")
        return {"status":"VERIFIED","artifact":rec,"index_sha256":raw.get("index_sha256"),"candidate_count":raw.get("candidate_count"),"policy_selection_performed":False}
    reg("temporal.reference_set.describe","Describe B010 leave-one-provenance-group-out reference-set temporal evidence.",temporal_development_describe("adcs_reference_set_temporal_policy_r001.json","reference_set_temporal_r001","LEAVE_ONE_PROVENANCE_GROUP_OUT_REFERENCE_ENVELOPE"),profiles={AgentProfile.DIAGNOSIS.value,AgentProfile.DEVELOPMENT.value,AgentProfile.SHADOW.value,AgentProfile.MAINTAINER.value})
    reg("temporal.reference_set.verify_trace_index","Verify a frozen B010 reference-set trace index.",lambda a,e:verify_development_index(a,e,require_reference_isolation=True),profiles={AgentProfile.DEVELOPMENT.value,AgentProfile.MAINTAINER.value},label="OFFLINE_ONLY",schema={"type":"object","required":["path","expected_sha256"],"properties":{"path":{"type":"string"},"expected_sha256":{"type":"string","pattern":"^[0-9a-fA-F]{64}$"}},"additionalProperties":False})
    reg("temporal.mechanism_evidence.describe","Describe B011 label-free causal local mechanism-change evidence.",temporal_development_describe("adcs_mechanism_evidence_temporal_policy_r001.json","mechanism_evidence_temporal_r001","CAUSAL_LOCAL_MECHANISM_CHANGE_PLUS_B010"),profiles={AgentProfile.DIAGNOSIS.value,AgentProfile.DEVELOPMENT.value,AgentProfile.SHADOW.value,AgentProfile.MAINTAINER.value})
    reg("temporal.mechanism_evidence.verify_trace_index","Verify frozen B011 candidate selections.",verify_development_index,profiles={AgentProfile.DEVELOPMENT.value,AgentProfile.MAINTAINER.value},label="OFFLINE_ONLY",schema={"type":"object","required":["path","expected_sha256"],"properties":{"path":{"type":"string"},"expected_sha256":{"type":"string","pattern":"^[0-9a-fA-F]{64}$"}},"additionalProperties":False})
    reg("temporal.so3_star_gyro.describe","Describe B012 SO(3) star/gyro consistency evidence.",temporal_development_describe("adcs_so3_star_gyro_temporal_policy_r001.json","so3_star_gyro_temporal_r001","SO3_GEODESIC_STAR_GYRO_CONSISTENCY_PLUS_B011_OOF"),profiles={AgentProfile.DIAGNOSIS.value,AgentProfile.DEVELOPMENT.value,AgentProfile.SHADOW.value,AgentProfile.MAINTAINER.value})
    reg("temporal.so3_star_gyro.verify_trace_index","Verify frozen B012 candidate selections.",verify_development_index,profiles={AgentProfile.DEVELOPMENT.value,AgentProfile.MAINTAINER.value},label="OFFLINE_ONLY",schema={"type":"object","required":["path","expected_sha256"],"properties":{"path":{"type":"string"},"expected_sha256":{"type":"string","pattern":"^[0-9a-fA-F]{64}$"}},"additionalProperties":False})
    reg("temporal.control_authority.describe","Describe B013 command/execution control-authority evidence.",temporal_development_describe("adcs_control_authority_temporal_policy_r001.json","control_authority_temporal_r001","ACTIVITY_GATED_ACTUATOR_DELIVERY_LOSS_PLUS_B012_OOF"),profiles={AgentProfile.DIAGNOSIS.value,AgentProfile.DEVELOPMENT.value,AgentProfile.SHADOW.value,AgentProfile.MAINTAINER.value})
    reg("temporal.control_authority.verify_trace_index","Verify frozen B013 candidate selections.",verify_development_index,profiles={AgentProfile.DEVELOPMENT.value,AgentProfile.MAINTAINER.value},label="OFFLINE_ONLY",schema={"type":"object","required":["path","expected_sha256"],"properties":{"path":{"type":"string"},"expected_sha256":{"type":"string","pattern":"^[0-9a-fA-F]{64}$"}},"additionalProperties":False})
    def temporal_stack_describe(a,e):
        path=e.project_root/"configs/research/adcs_temporal_stack_freeze_r001.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status":"NOT_FROZEN"}
    reg("temporal.stack.describe","Describe the frozen B010-B013 ADCS development temporal stack and B014 boundary.",temporal_stack_describe,profiles={AgentProfile.DEVELOPMENT.value,AgentProfile.CONFIRMATORY_SEALED.value,AgentProfile.SHADOW.value,AgentProfile.MAINTAINER.value},label="OFFLINE_ONLY",schema=EMPTY)
    catalog_path=env.project_root/"configs/agentization/optional_signal_toolkits.json"
    reg("optional_toolkits.catalog","Return selectively admitted optional signal/anomaly libraries; none are imported by default.",lambda a,e:json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.exists() else {"toolkits":[]})
    def workflow_capability_audit(a,e):
        rows=[]
        for workflow_id,workflow in sorted(load_workflows(e.workflow_root).items()):
            profile=str(workflow.get("profile",""))
            names=[str(node.get("capability",node.get("id",""))) for node in workflow.get("nodes",[])]
            missing=[name for name in names if name not in r._defs]
            forbidden=[name for name in names if name in r._defs and profile not in r._defs[name].allowed_profiles]
            unavailable=[{"capability":name,"availability":r._defs[name].availability} for name in names if name in r._defs and r._defs[name].availability!="AVAILABLE"]
            status="INVALID"
            if not missing and not forbidden:
                status="BLOCKED_BY_DECLARED_PREREQUISITES" if unavailable else "EXECUTABLE"
            rows.append({"workflow_id":workflow_id,"profile":profile,"status":status,"capabilities":names,"missing_capabilities":missing,"profile_forbidden_capabilities":forbidden,"unavailable_capabilities":unavailable})
        invalid=[row["workflow_id"] for row in rows if row["status"]=="INVALID"]
        return {"status":"PASS" if not invalid else "FAIL","invalid_workflows":invalid,"workflow_count":len(rows),"workflows":rows,"scientific_validation_performed":False}
    reg("workflows.audit_capabilities","Audit every persistent workflow against registered capabilities, profile boundaries and declared availability.",workflow_capability_audit,profiles={AgentProfile.MAINTAINER.value,AgentProfile.SHADOW.value},schema=EMPTY)
    return r
