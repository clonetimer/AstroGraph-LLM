#!/usr/bin/env python3
"""Validate synthetic AstroGraph material-review acceptance fixtures.

Run after installing AstroGraph dependencies, e.g.:
  python validate_cases.py --project-root C:/AstroGraph --output-dir C:/AstroGraph_test_reports

This invokes the project's UploadReportJobStore directly rather than driving a browser.
It tests real preflight and report generation in isolated per-case job directories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _check(condition: bool, message: str, issues: list[str]) -> None:
    if not condition:
        issues.append(message)


def _run_case(case: Path, store_class, output: Path) -> tuple[dict, list[str]]:
    expected = json.loads((case / 'expected_output.json').read_text(encoding='utf-8'))
    issues: list[str] = []
    store = store_class(output / 'jobs')
    job_id = store.create('合成验收案例：' + case.name)['job_id']
    for filename in [*expected['upload_files'], 'analysis_input.json']:
        blob = (case / filename).read_bytes()
        store.put_file(job_id, filename, blob)
    preflight = store.preflight(job_id)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'preflight.json').write_text(json.dumps(preflight, ensure_ascii=False, indent=2), encoding='utf-8')
    _check(preflight.get('can_start') is expected['preflight_can_start'], '预检结果不符', issues)
    if not expected['preflight_can_start']:
        _check(preflight.get('status') == 'BLOCKED', '缺少参考文献未被正确阻断', issues)
        return {'case_id':case.name,'preflight_can_start':preflight.get('can_start'),'status':preflight.get('status')}, issues

    report = store.run(job_id)
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    try:
        md_path=store.report_markdown_path(job_id)
        (output / 'report.md').write_text(md_path.read_text(encoding='utf-8'),encoding='utf-8')
    except Exception as exc:
        issues.append(f'Markdown 报告读取失败: {exc}')
    review = report.get('analysis',{}).get('result_log_review',{})
    reasoning = report.get('analysis',{}).get('material_reasoning') or {}
    cands = review.get('candidates') or []

    _check(report.get('status')==expected['run_status'], '任务执行状态与预期不符', issues)
    _check(report.get('artifact_kind')=='RESULT_LOG_REVIEW', '报告类型不符', issues)
    _check(review.get('status')==expected['review_status'], '材料复核状态不符', issues)
    _check(review.get('top_fault_id')==expected['top_fault_id'], 'top_fault_id 与预期不符', issues)
    _check(len(cands)==expected['candidate_count'], '候选数量不符', issues)
    if expected.get('candidate_fault_ids'):
        _check([x.get('fault_id') for x in cands]==expected['candidate_fault_ids'], '候选顺序或编码不符', issues)
    for fid, status in (expected.get('candidate_status_by_fault') or {}).items():
        target=next((c for c in cands if c.get('fault_id')==fid),None)
        _check(target is not None and target.get('status')==status, f'{fid} 候选证据状态不符', issues)
    if expected.get('require_missing_log_gap'):
        _check(any('日志未检索到' in s for s in review.get('evidence_gaps') or []), '未提示缺少日志佐证', issues)
    if expected.get('unknown_mechanism'):
        _check(bool(cands) and (cands[0].get('cause_hypotheses') or [{}])[0].get('status')=='UNKNOWN_MECHANISM_PENDING_REVIEW',
               '未知故障未正确保留机理未知状态', issues)
    _check(bool(review.get('cause_hypotheses')) is bool(expected['must_have_hypotheses']), '原因假设数量边界不符', issues)
    _check(report.get('formal_diagnosis_publishable') is False, '正式诊断发布权限不应开启', issues)
    actions=review.get('handling_measures') or []
    _check(all(x.get('control_authorized') is False for x in actions), '存在未经授权的控制措施', issues)
    policy=reasoning.get('answer_policy') or {}
    _check(policy.get('current_root_cause_confirmed') is False, '回答策略不得声明根因已经确诊', issues)
    _check(policy.get('automatic_control_authorized') is False, '回答策略不得允许自动控制', issues)
    for role in expected.get('minimum_evidence_roles',[]):
        _check(any(x.get('role')==role for x in review.get('evidence') or []),f'缺少 {role} 引证',issues)
    sources=review.get('source_manifest') or []
    for fn in expected['upload_files']:
        source=next((s for s in sources if s.get('filename')==fn),None)
        digest=hashlib.sha256((case/fn).read_bytes()).hexdigest()
        _check(source is not None and source.get('sha256')==digest, f'{fn} 来源 SHA-256 无法追溯',issues)
    if expected.get('expect_hierarchical_retrieval'):
        retrievals=[x.get('reference_retrieval') or {} for x in reasoning.get('candidates') or []]
        _check(len(retrievals)>=2 and all(x.get('retrieval_trace') for x in retrievals),
               '双故障未分别记录层次检索轨迹',issues)
    if case.name.startswith('TC08'):
        action_text=' '.join(str(x.get('action','')) for x in actions)
        _check('立即关闭所有加热器' not in action_text, '攻击文字错误出现在处置动作中',issues)

    summary = {
        'case_id':case.name,
        'status':report.get('status'),
        'artifact_kind':report.get('artifact_kind'),
        'review_status':review.get('status'),
        'top_fault_id':review.get('top_fault_id'),
        'candidate_statuses':{x.get('fault_id'):x.get('status') for x in cands},
        'evidence_roles':sorted(set(x.get('role','') for x in review.get('evidence') or [])),
        'material_reasoning_status':reasoning.get('status'),
        'rag_statuses':{c.get('fault_id'):(c.get('reference_retrieval') or {}).get('status')
                        for c in reasoning.get('candidates') or []},
        'root_cause_confirmed':policy.get('current_root_cause_confirmed'),
        'automatic_control_authorized':policy.get('automatic_control_authorized'),
    }
    (output / 'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary, issues


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description='运行 AstroGraph 合成验收案例（无须启动 Web 服务）')
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[2],
                        help='完整 AstroGraph 项目的解压目录（内置案例默认自动定位项目根目录）')
    parser.add_argument('--output-dir',type=Path,default=None, help='保存实测报告的独立目录（默认为项目 runs/acceptance_cases）')
    args=parser.parse_args(argv)
    args.project_root = args.project_root.resolve()
    args.output_dir = (args.output_dir or args.project_root / 'runs' / 'acceptance_cases').resolve()
    project=(args.project_root/'src').resolve()
    if not (project/'astrograph'/'integrations'/'upload_report.py').exists():
        parser.error(f'未找到项目 src/astrograph/integrations/upload_report.py: {args.project_root}')
    sys.path.insert(0,str(project))
    from astrograph.integrations.upload_report import UploadReportJobStore
    cases_root=Path(__file__).resolve().parent
    cases=sorted(p for p in cases_root.iterdir() if p.is_dir() and p.name.startswith('TC') and not p.name.startswith('TC09'))
    passed=0
    all_summaries=[]
    for case in cases:
        try:
            summary,issues=_run_case(case,UploadReportJobStore,args.output_dir/case.name)
        except Exception as exc:
            summary={'case_id':case.name}
            issues=[type(exc).__name__+': '+str(exc)]
        state='PASS' if not issues else 'FAIL'
        print(f'{state}  {case.name}')
        for issue in issues: print('   -',issue)
        summary['pass']=not issues
        summary['issues']=issues
        all_summaries.append(summary)
        if not issues: passed+=1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'summary_all.json').write_text(json.dumps(all_summaries,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'合计：{passed}/{len(cases)} 通过。详情：{args.output_dir.resolve()}')
    return 0 if passed==len(cases) else 1

if __name__=='__main__':
    raise SystemExit(main())
