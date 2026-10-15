"""Headless interface for the same upload-to-report workflow used by /platform."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import sys

from astrograph.integrations.platform_tools import platform_tool_catalog
from astrograph.integrations.upload_report import UploadReportJobStore


def _store(root: str | None) -> UploadReportJobStore:
    return UploadReportJobStore(root or os.environ.get("ASTROGRAPH_PLATFORM_ROOT", "runs/platform"))


def _print_json(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False))


def _summary(report: dict) -> dict:
    analysis = report.get("analysis") or {}
    candidate = None
    disposition = report.get("status")
    kind = report.get("artifact_kind")
    if kind == "PUBLIC_CASE":
        diag = ((analysis.get("closure") or {}).get("diagnosis") or {})
        items = diag.get("classification_candidates") or []
        candidate = items[0] if items else None
        disposition = diag.get("status") or disposition
    elif kind == "RESULT_LOG_REVIEW":
        review = analysis.get("result_log_review") or {}
        candidate = review.get("top_fault_id")
        disposition = review.get("status")
    elif kind == "PIPELINE_RUNTIME":
        result = analysis.get("pipeline_result") or {}
        candidate = (result.get("fusion") or {}).get("top_fault_id")
        disposition = result.get("disposition") or disposition
    elif kind == "OAG_DIAGNOSTIC":
        summary = (((analysis.get("oag_result") or {}).get("output") or {}).get("summary") or {})
        candidate = summary.get("suspected_fault_id")
        disposition = summary.get("status") or disposition
    return {
        "job_id": report.get("job_id"),
        "status": report.get("status"),
        "artifact_kind": kind,
        "work_mode": report.get("work_mode"),
        "candidate": candidate,
        "disposition": disposition,
    }


def cmd_diagnose(args: argparse.Namespace) -> int:
    store = _store(args.root)
    job = store.create(args.task, args.mode)
    for raw in args.files:
        path = Path(raw).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"input is not a file: {path}")
        store.put_file(job["job_id"], path.name, path.read_bytes())
    report = store.run(job["job_id"])
    _print_json(report if args.full else _summary(report))
    return 0 if report.get("status") == "SUCCEEDED" else 2


def cmd_review(args: argparse.Namespace) -> int:
    """Explicitly assign roles; upload the same manifest used by the Web UI."""
    role_args = {
        "diagnosis_results": args.result,
        "log_reports": args.log,
        "references": args.reference,
    }
    paths = {role: [Path(raw).expanduser().resolve(strict=True) for raw in values]
             for role, values in role_args.items()}
    for group in paths.values():
        if any(not x.is_file() for x in group):
            raise ValueError("review inputs must be regular files")
    names = [path.name for values in paths.values() for path in values]
    if len({x.casefold() for x in names}) != len(names) or "analysis_input.json" in names:
        raise ValueError("review input filenames must be distinct and cannot be analysis_input.json")
    store = _store(args.root)
    job = store.create(args.task, args.mode)
    for group in paths.values():
        for path in group:
            store.put_file(job["job_id"], path.name, path.read_bytes())
    manifest = {"schema_version": "astrograph-result-log-review.v1",
                **{role: [path.name for path in values] for role, values in paths.items()}}
    if args.equipment_scope:
        manifest["equipment_scope"] = args.equipment_scope
    store.put_file(job["job_id"], "analysis_input.json", json.dumps(manifest, ensure_ascii=False).encode("utf-8"))
    preflight = store.preflight(job["job_id"])
    if not preflight["can_start"]:
        _print_json(preflight)
        return 2
    report = store.run(job["job_id"])
    _print_json(report if args.full else _summary(report))
    return 0 if report.get("status") == "SUCCEEDED" else 2


def _material_memory(args: argparse.Namespace):
    from astrograph.integrations.material_case_memory import MaterialCaseMemory
    root = _store(args.root).root
    reviewers=tuple(x.strip() for x in os.environ.get('ASTROGRAPH_MATERIAL_REVIEWERS','').split(',') if x.strip())
    return MaterialCaseMemory(root / 'material_knowledge' / 'cases.sqlite3',reviewers=reviewers)


def cmd_memory_pending(args: argparse.Namespace) -> int:
    _print_json({'cases':_material_memory(args).pending(limit=args.limit)})
    return 0


def cmd_memory_decide(args: argparse.Namespace) -> int:
    # Host-side privileged CLI, not a user-upload or model-controlled tool.
    actor=os.environ.get('ASTROGRAPH_MATERIAL_REVIEWER','').strip()
    if not actor:
        raise PermissionError('ASTROGRAPH_MATERIAL_REVIEWER must be set by local administrator')
    _print_json(_material_memory(args).decide(
        case_id=args.case_id,reviewer=actor,decision=args.decision,
        truth_source_ref=args.truth_source_ref,
        verified_fault_id=args.verified_fault_id,
        equipment_scope=args.equipment_scope,
        approved_actions=args.approved_action or [],
        expected_snapshot=args.expected_snapshot,
    ))
    return 0


def cmd_document_pending(args: argparse.Namespace) -> int:
    _print_json({'documents':_material_memory(args).pending_documents(limit=args.limit)})
    return 0


def cmd_document_decide(args: argparse.Namespace) -> int:
    actor=os.environ.get('ASTROGRAPH_MATERIAL_REVIEWER','').strip()
    if not actor:
        raise PermissionError('ASTROGRAPH_MATERIAL_REVIEWER must be configured by local administrator')
    _print_json(_material_memory(args).decide_document(
        document_id=args.document_id,reviewer=actor,decision=args.decision,
        truth_source_ref=args.truth_source_ref,equipment_scope=args.equipment_scope,
        expected_snapshot=args.expected_snapshot,
    ))
    return 0


def cmd_graph_build(args: argparse.Namespace) -> int:
    """Trusted-host standalone reference graph construction, no diagnosis needed."""
    from astrograph.integrations.material_graph_builder import build_material_graph
    from astrograph.integrations.platform_knowledge import UploadedDocumentKnowledgeIndex
    memory=_material_memory(args)
    memory._scope(args.equipment_scope)  # require unambiguous intended applicability
    inputs=[Path(p).expanduser().resolve(strict=True) for p in args.reference]
    if len(inputs)>24 or len({p.name.casefold() for p in inputs})!=len(inputs):
        raise ValueError('graph-build accepts at most 24 distinct filenames')
    if any(not p.is_file() or p.stat().st_size > 32*1024*1024 for p in inputs):
        raise ValueError('graph-build requires regular source files at most 32 MiB each')
    chunks,ingestion=UploadedDocumentKnowledgeIndex(top_k=8).build(inputs)
    group='JOB-'+secrets.token_hex(12).upper()
    result=build_material_graph(chunks=chunks, source_group_id=group,
                                memory=memory, enabled=True,max_chunks=args.max_chunks)
    result['equipment_scope_requested']=args.equipment_scope.upper()
    result['source_group_id']=group
    result['document_ingestion']=ingestion
    _print_json(result)
    return 0 if result['status'] in {'SUCCEEDED','NO_VALID_RELATIONS'} else 2


def cmd_graph_pending(args: argparse.Namespace) -> int:
    mem=_material_memory(args)
    _print_json({'snapshot':mem.snapshot(),'graph_proposals':mem.pending_graph(limit=args.limit)})
    return 0


def cmd_graph_status(args: argparse.Namespace) -> int:
    _print_json(_material_memory(args).graph_status(equipment_scope=args.equipment_scope))
    return 0


def cmd_graph_decide(args: argparse.Namespace) -> int:
    actor=os.environ.get('ASTROGRAPH_MATERIAL_REVIEWER','').strip()
    if not actor:
        raise PermissionError('ASTROGRAPH_MATERIAL_REVIEWER required for graph approval')
    _print_json(_material_memory(args).decide_graph(
        proposal_id=args.proposal_id,reviewer=actor,decision=args.decision,
        truth_source_ref=args.truth_source_ref,equipment_scope=args.equipment_scope,
        expected_snapshot=args.expected_snapshot))
    return 0


def cmd_graph_retract(args: argparse.Namespace) -> int:
    """Audited tombstone: future graph snapshots stop exposing this relation."""
    actor=os.environ.get('ASTROGRAPH_MATERIAL_REVIEWER','').strip()
    if not actor:
        raise PermissionError('ASTROGRAPH_MATERIAL_REVIEWER required for graph retraction')
    _print_json(_material_memory(args).retract_graph(
        proposal_id=args.proposal_id,reviewer=actor,reason=args.reason,
        truth_source_ref=args.truth_source_ref,expected_snapshot=args.expected_snapshot))
    return 0


def cmd_graph_export(args: argparse.Namespace) -> int:
    mem=_material_memory(args)
    snapshot=mem.snapshot() if args.snapshot is None else args.snapshot
    output=Path(args.output).expanduser().resolve() if args.output else (
        _store(args.root).root/'material_knowledge'/'graph_releases')
    _print_json(mem.export_graph_release(equipment_scope=args.equipment_scope,
                                          snapshot=snapshot,output_root=output))
    return 0


def cmd_graph_verify(args: argparse.Namespace) -> int:
    from astrograph.integrations.material_graph_release import verify_graph_release
    manifest,_=verify_graph_release(args.release)
    _print_json({**manifest,'verification_status':'VERIFIED'})
    return 0


def cmd_graph_publish_neo4j(args: argparse.Namespace) -> int:
    from astrograph.integrations.material_graph_release import publish_release_neo4j
    _print_json(publish_release_neo4j(args.release,confirm_write=args.confirm_write))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    _print_json(_store(args.root).overview(limit=args.limit))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    store = _store(args.root)
    if args.format == "md":
        sys.stdout.write(store.report_markdown_path(args.job_id).read_text(encoding="utf-8"))
    else:
        _print_json(store.report(args.job_id))
    return 0


def cmd_tools(args: argparse.Namespace) -> int:
    _print_json(platform_tool_catalog())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AstroGraph headless diagnosis platform")
    parser.add_argument("--root", help="Platform job directory; defaults to ASTROGRAPH_PLATFORM_ROOT or runs/platform")
    sub = parser.add_subparsers(dest="command", required=True)

    diagnose = sub.add_parser("diagnose", help="Create, upload and run one diagnosis synchronously")
    diagnose.add_argument("files", nargs="+", help="Diagnostic artifacts and optional PDF/DOCX/TXT/MD documents")
    diagnose.add_argument("--task", default="分析上传的诊断工件并给出证据驱动结论。")
    diagnose.add_argument("--mode", choices=["normal", "thinking", "creative"], default="normal")
    diagnose.add_argument("--full", action="store_true", help="Print the complete report JSON")
    diagnose.set_defaults(func=cmd_diagnose)

    review = sub.add_parser("review", help="Root-cause review from upstream diagnosis, log and reference files")
    review.add_argument("--result", required=True, action="append", help="Upstream diagnosis result (repeatable)")
    review.add_argument("--log", required=True, action="append", help="Operations log/report (repeatable)")
    review.add_argument("--reference", required=True, action="append", help="Reference document (repeatable)")
    review.add_argument("--task", default="核对故障原因并给出有证据出处的人工处置建议")
    review.add_argument("--mode", choices=["normal", "thinking", "creative"], default="normal")
    review.add_argument("--equipment-scope", help="Expert-known model/device scope for exact historical case retrieval")
    review.add_argument("--full", action="store_true", help="Print the complete review JSON")
    review.set_defaults(func=cmd_review)

    pending=sub.add_parser('memory-pending',help='List material review cases pending independent expert confirmation (local host)')
    pending.add_argument('--limit',type=int,default=30)
    pending.set_defaults(func=cmd_memory_pending)

    decision=sub.add_parser('memory-decide',help='Confirm or reject an unverified case as local approved reviewer')
    decision.add_argument('case_id')
    decision.add_argument('--decision',choices=['CONFIRM','REJECT'],required=True)
    decision.add_argument('--truth-source-ref',required=True)
    decision.add_argument('--verified-fault-id')
    decision.add_argument('--equipment-scope')
    decision.add_argument('--approved-action',action='append')
    decision.add_argument('--expected-snapshot',type=int)
    decision.set_defaults(func=cmd_memory_decide)

    docs=sub.add_parser('document-pending',help='List unverified reference documents')
    docs.add_argument('--limit',type=int,default=30)
    docs.set_defaults(func=cmd_document_pending)
    docdecision=sub.add_parser('document-decide',help='Independently approve/reject a staged reference document')
    docdecision.add_argument('document_id')
    docdecision.add_argument('--decision',choices=['CONFIRM','REJECT'],required=True)
    docdecision.add_argument('--truth-source-ref',required=True)
    docdecision.add_argument('--equipment-scope')
    docdecision.add_argument('--expected-snapshot',type=int)
    docdecision.set_defaults(func=cmd_document_decide)

    graph_build=sub.add_parser('graph-build',help='Standalone LLM graph construction from references (trusted host)')
    graph_build.add_argument('--reference',required=True,action='append',help='PDF/DOCX/MD/TXT (repeatable)')
    graph_build.add_argument('--equipment-scope',required=True,help='Precise model/configuration identifier')
    graph_build.add_argument('--max-chunks',type=int,default=12)
    graph_build.set_defaults(func=cmd_graph_build)

    gp=sub.add_parser('graph-pending',help='List model-extracted graph assertions awaiting individual review')
    gp.add_argument('--limit',type=int,default=30)
    gp.set_defaults(func=cmd_graph_pending)

    gs=sub.add_parser('graph-status',help='Audit graph proposal/approval/retraction snapshot counters')
    gs.add_argument('--equipment-scope',help='Filter approved graph knowledge by exact equipment scope')
    gs.set_defaults(func=cmd_graph_status)

    gd=sub.add_parser('graph-decide',help='Independently confirm/reject one reference graph relation')
    gd.add_argument('proposal_id')
    gd.add_argument('--decision',choices=['CONFIRM','REJECT'],required=True)
    gd.add_argument('--truth-source-ref',required=True)
    gd.add_argument('--equipment-scope',help='Required when CONFIRM')
    gd.add_argument('--expected-snapshot',type=int)
    gd.set_defaults(func=cmd_graph_decide)

    gr=sub.add_parser('graph-retract',help='Withdraw one approved graph claim from future snapshots')
    gr.add_argument('proposal_id')
    gr.add_argument('--reason',required=True,help='Explicit correction/scope change rationale')
    gr.add_argument('--truth-source-ref',required=True,help='Independent engineering revocation record')
    gr.add_argument('--expected-snapshot',type=int)
    gr.set_defaults(func=cmd_graph_retract)

    ge=sub.add_parser('graph-export',help='Export immutable graph JSON/CSV release of reviewed claims')
    ge.add_argument('--equipment-scope',required=True)
    ge.add_argument('--snapshot',type=int,help='Fixed memory snapshot (default current head)')
    ge.add_argument('--output',help='Immutable release directory parent')
    ge.set_defaults(func=cmd_graph_export)

    gv=sub.add_parser('graph-verify',help='Verify release manifest, hashes and approved ontology claims')
    gv.add_argument('release')
    gv.set_defaults(func=cmd_graph_verify)

    pub=sub.add_parser('graph-publish-neo4j',help='Dry-run or separately authorized Neo4j material namespace import')
    pub.add_argument('release')
    pub.add_argument('--confirm-write',action='store_true',help='Require host write gate and dedicated database')
    pub.set_defaults(func=cmd_graph_publish_neo4j)

    listing = sub.add_parser("list", help="List recent jobs")
    listing.add_argument("--limit", type=int, default=30)
    listing.set_defaults(func=cmd_list)

    report = sub.add_parser("report", help="Read a completed report")
    report.add_argument("job_id")
    report.add_argument("--format", choices=["json", "md"], default="json")
    report.set_defaults(func=cmd_report)

    tools = sub.add_parser("tools", help="Inspect built-in and trusted plugin tools")
    tools.set_defaults(func=cmd_tools)

    args = parser.parse_args(argv)
    try:
        # Honour settings saved from the UI even in a standalone CLI process.
        if args.command in {'graph-build','review','diagnose'}:
            from astrograph.integrations.material_engine_settings import load_active_engine_environment
            from pathlib import Path
            platform_root = Path(args.root or os.environ.get('ASTROGRAPH_PLATFORM_ROOT') or 'runs/platform')
            load_active_engine_environment(platform_root)

        return int(args.func(args))
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
