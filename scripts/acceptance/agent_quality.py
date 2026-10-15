#!/usr/bin/env python3
"""Quality evaluation entry point. Never changes stage-39 acceptance or prior tasks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"src"),str(ROOT)]
from astrograph.evaluation.agent_quality.io import read_json,write_new,file_sha
from astrograph.evaluation.agent_quality.metrics import inspect_task,review_template
from astrograph.evaluation.agent_quality.experiment import create_experiment,run_experiment,score_experiment
from astrograph.evaluation.agent_quality.corpus import evaluate_corpus
from astrograph.evaluation.agent_quality.reviews import score_review
from astrograph.evaluation.agent_quality.probe import numeric_probe
from astrograph.evaluation.agent_quality.fixtures import build_numeric_cases,build_corpus_fixture


def from_workspace(workspace: Path, output: Path, settings: Path, repeats, modes, scope, check_budget=None):
    workspace=workspace.resolve()
    config=read_json(workspace/"acceptance.json")
    prepared=workspace/"stages/10-data-inference/prepared-input.json"
    if not prepared.exists():raise ValueError("RUN_CORE_IN_THE_SOURCE_WORKSPACE_FIRST")
    # Construct the immutable request in a preparation sibling, not old workspace.
    staging=output.with_name(output.name+"-preparation")
    staging.mkdir(parents=True,exist_ok=False)
    previous=workspace/"stages/39-llm-planning/llm-thinking.request.json"
    old_goal=read_json(previous)["goal"] if previous.exists() else "Diagnose spacecraft faults using data and verified mechanism evidence."
    req={"task_id":"adcs-slice","task":"diagnose","work_mode":"thinking",
         "goal":old_goal,
         "artifacts":{"input":str(prepared),"graph_dir":str(prepared.parent/"graph"),
                      "source":config["source"],"nominal":config["nominal"]},
         "documents":config.get("documents",[])}
    write_new(staging/"request.json",req)
    return create_experiment(output,project=ROOT,settings_path=settings,repeats=repeats,modes=tuple(modes),
            cases=[{"case_id":"adcs-slice","request_path":str(staging/"request.json"),"scope":scope}],check_budget=check_budget)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    audit=sub.add_parser("audit",help="Read-only completed task quality observations")
    audit.add_argument("--task-dir",type=Path,required=True);audit.add_argument("--output",type=Path,required=True)
    prepare=sub.add_parser("init",help="Freeze repeated paired trials; sends no requests")
    prepare.add_argument("--workspace",type=Path,required=True);prepare.add_argument("--settings",type=Path,required=True)
    source=prepare.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-workspace",type=Path);source.add_argument("--cases",type=Path)
    prepare.add_argument("--repeats",type=int,default=10)
    prepare.add_argument("--modes",nargs="+",choices=["normal","thinking","creative"],default=["thinking","creative"])
    prepare.add_argument("--check-budget",type=int,help="Optional cap, applied equally to both arms; never raises a host budget")
    prepare.add_argument("--scope",default="ALREADY_USED_ADCS_SLICE_NOT_NEW_HOLDOUT")
    run=sub.add_parser("run",help="Explicitly request live repeated trials; never silently rerun failures")
    run.add_argument("--workspace",type=Path,required=True);run.add_argument("--allow-live",action="store_true")
    run.add_argument("--max-runs",type=int);run.add_argument("--output",type=Path,required=True)
    score=sub.add_parser("score",help="Recompute all scheduled trial rows without requests")
    score.add_argument("--workspace",type=Path,required=True);score.add_argument("--output",type=Path,required=True)
    corpus=sub.add_parser("retrieval",help="Run current local retriever against evaluator-side source-bound qrels")
    corpus.add_argument("--manifest",type=Path,required=True);corpus.add_argument("--output",type=Path,required=True)
    corpus.add_argument("--settings",type=Path)
    fixtures=sub.add_parser("fixtures",help="Prepare synthetic cases and source-quoted corpus; no services or production writes")
    fixtures.add_argument("--output",type=Path,required=True)
    review=sub.add_parser("review-score",help="Score result-hash-bound human judgements; never feed them to a planner")
    review.add_argument("--quality",type=Path,required=True);review.add_argument("--review",type=Path,required=True);review.add_argument("--output",type=Path,required=True)
    probe=sub.add_parser("numeric-probe",help="Offline scripted sensitivity, not a real LLM comparison")
    probe.add_argument("--cases",type=Path,required=True);probe.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        if args.command in {"audit","run","score"} and args.output.exists():
            raise FileExistsError("OUTPUT_EXISTS_PRESERVE_PRIOR_REPORT")
        if args.command=="audit":
            report=inspect_task(args.task_dir);write_new(args.output,report)
            write_new(args.output.with_name(args.output.stem+".review-template.json"),review_template(report))
            answer={"status":"AUDIT_COMPLETED_NOT_QUALITY_PASS","generation_requests_sent":0,"output":str(args.output)}
        elif args.command=="init":
            if args.from_workspace:
                report=from_workspace(args.from_workspace,args.workspace,args.settings,args.repeats,args.modes,args.scope,args.check_budget)
            else:
                cases=read_json(args.cases)
                report=create_experiment(args.workspace,project=ROOT,settings_path=args.settings,repeats=args.repeats,modes=tuple(args.modes),cases=cases,check_budget=args.check_budget)
            answer={"status":"FROZEN_READY_NOT_RUN","scheduled":len(report["cells"]),"generation_requests_sent":0,
                    "manifest":str(args.workspace/"experiment.json"),"case_memory":"DISABLED_FOR_ALL_ARMS"}
        elif args.command=="run":
            report=run_experiment(args.workspace,allow_live=args.allow_live,max_runs=args.max_runs)
            write_new(args.output,report)
            answer={"status":"MEASURED_NOT_AUTOMATICALLY_PASSED","counts":report["status_counts"],"output":str(args.output)}
        elif args.command=="score":
            report=score_experiment(args.workspace);write_new(args.output,report)
            answer={"status":"RECOMPUTED","counts":report["status_counts"],"generation_requests_sent":0}
        elif args.command=="retrieval":
            report=evaluate_corpus(args.manifest,args.output,settings_path=args.settings)
            answer={k:report[k] for k in ("execution_status","quality_verdict","scope","queries","mean_passage_recall_at_k","no_answer_rejection_rate")}
        elif args.command=="review-score":
            report=score_review(read_json(args.quality),read_json(args.review));write_new(args.output,report)
            answer={"status":"REVIEW_SCORED_WITH_COVERAGE","generation_requests_sent":0,"output":str(args.output)}
        elif args.command=="numeric-probe":
            report=numeric_probe(args.cases,args.output,ROOT)
            answer={"status":"PROBE_COMPLETED_NOT_LLM_GAIN","cases":len(report["rows"]),"generation_requests_sent":0}
        else:
            if args.output.exists():raise FileExistsError("NEW_FIXTURE_ROOT_REQUIRED")
            args.output.mkdir(parents=True)
            cases=build_numeric_cases(ROOT,args.output/"numeric")
            write_new(args.output/"cases.json",cases)
            corpus_path=build_corpus_fixture(args.output/"corpus")
            answer={"status":"SYNTHETIC_FIXTURES_PREPARED","cases":str(args.output/"cases.json"),"retrieval_manifest":str(corpus_path),
                    "generation_requests_sent":0,"production_admitted":False}
        print(json.dumps(answer,ensure_ascii=False,indent=2));return 0
    except Exception as exc:
        # Details may include private path names but never dump settings/credentials.
        print(json.dumps({"status":"BLOCKED_OR_ERROR","error_type":type(exc).__name__,"message":str(exc)[:800]},ensure_ascii=False));return 2

if __name__=="__main__":raise SystemExit(main())
