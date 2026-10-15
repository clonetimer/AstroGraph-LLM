"""Spacecraft task Agent CLI. Existing OAG commands remain supported."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .models import AgentRequest,AgentSettings
from .service import MissionAgent


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    for name in ("run","doctor"):
        q=sub.add_parser(name);q.add_argument("--settings",type=Path,required=True)
        if name=="run":
            q.add_argument("--request",type=Path,required=True)
            q.add_argument("--work-mode",choices=["normal","thinking","creative"],help="Override only requested strategy; cannot grant host permissions.")
    args=p.parse_args(argv)
    try:
        settings=AgentSettings.model_validate_json(args.settings.read_text(encoding="utf-8"))
        agent=MissionAgent(settings)
        if args.command=="doctor":print(json.dumps(agent.inventory(),ensure_ascii=False,indent=2));return 0
        req=AgentRequest.model_validate_json(args.request.read_text(encoding="utf-8"))
        if args.work_mode:req=AgentRequest.model_validate({**req.model_dump(),"work_mode":args.work_mode})
        result=agent.run_supervised(req)
        print(json.dumps({"task_id":req.task_id,"status":result.status,"work_mode":result.work_mode,
                          "result_path":str(Path(settings.output_root).resolve()/req.task_id/"result.json")},ensure_ascii=False))
        return 0 if result.status in {"SUCCEEDED","PARTIAL"} else 2
    except Exception as exc:
        print(json.dumps({"status":"BLOCKED","error_type":type(exc).__name__,"message":str(exc)},ensure_ascii=False))
        return 2
