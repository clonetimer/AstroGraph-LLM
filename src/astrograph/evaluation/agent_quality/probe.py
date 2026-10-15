"""Finite numeric choice sensitivity probe; scripted strategies are NOT LLMs."""
from __future__ import annotations
from pathlib import Path
from astrograph.agents.mission.models import AgentSettings,AgentRequest,InvestigationPlan,CheckChoice,AgentNarrative,ExplorationProposal
from astrograph.agents.mission.service import MissionAgent
from .io import read_json,write_new
from .metrics import inspect_task


class ScriptedCommonFirst:
    def generate_structured(self, *, agent_id,output_model,payload):
        if output_model is InvestigationPlan:return output_model(task=payload["fixed_task"],actions=[],rationale="Scripted numeric quality probe, not a real model.")
        if output_model is CheckChoice:
            opts=payload["eligible_checks"]
            chosen=next((x for x in opts if x["tool_id"].endswith("F_common")),opts[0])
            return output_model(tool_id=chosen["tool_id"],rationale="Deliberately consume budget on common support to test evaluator sensitivity.")
        if output_model is AgentNarrative:return output_model(candidate_fault_id=payload["required_candidate_fault_id"],excerpt_ids=[payload["sources_untrusted"][0]["excerpt_id"]])
        if output_model is ExplorationProposal:return output_model(hypotheses=[])
        raise ValueError("UNSCRIPTED_TASK")


def numeric_probe(cases_path:Path,output:Path,project:Path) -> dict:
    if output.exists():raise FileExistsError("NEW_NUMERIC_PROBE_DIRECTORY_REQUIRED")
    cases=read_json(cases_path)
    if not cases or any(c.get("scope")!="SYNTHETIC_NUMERIC_CHOICE_FIXTURE_NOT_ADCS_PERFORMANCE" for c in cases):
        raise ValueError("NUMERIC_PROBE_ONLY_ACCEPTS_EXPLICIT_SYNTHETIC_SCOPE")
    output.mkdir(parents=True);rows=[]
    for case in cases:
        request=AgentRequest.model_validate(read_json(Path(case["request_path"])))
        # No labels / judgement files enter the request or granted case root.
        for policy in ("EXISTING_DETERMINISTIC","SCRIPTED_COMMON_FIRST_NOT_LLM"):
            rid=case["case_id"]+("-det" if policy=="EXISTING_DETERMINISTIC" else "-common")
            req=request.model_copy(update={"task_id":rid,"work_mode":"thinking","case":None})
            settings=AgentSettings(project_root=str(project),read_roots=[str(Path(case["request_path"]).parent)],
                output_root=str(output/"tasks"),planner="deterministic" if policy=="EXISTING_DETERMINISTIC" else "llm_assisted",
                max_checks=2,max_optional_actions=0,max_llm_calls=6)
            result=MissionAgent(settings,backend=None if policy=="EXISTING_DETERMINISTIC" else ScriptedCommonFirst()).run(req)
            report=inspect_task(Path(settings.output_root)/rid)
            write_new(output/(rid+".quality.json"),report)
            rows.append({"case_id":case["case_id"],"policy":policy,"status":result.status,"check_budget":2,
                         "checks_executed":report["checks"]["executed"],"choice_opportunities":report["checks"]["multiple_option_opportunities"],
                         "selected":report["diagnosis"]["selected"],"diagnosis_status":report["diagnosis"]["status"],
                         "outcomes":report["checks"]["outcomes"],"http_attempts":report["http_attempts"]})
    result={"scope":"SYNTHETIC_SENSITIVITY_PROBE_NOT_LIVE_LLM_OR_ADCS_BENCHMARK", "rows":rows,
            "generation_requests_sent":0,"same_check_budget":2,"production_model_or_graph_modified":False,
            "interpretation":"A legal less-discriminating choice can consume the same budget yet leave a different decision; no claim that an LLM beats deterministic selection."}
    write_new(output/"report.json",result);return result
