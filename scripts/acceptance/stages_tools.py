"""Tool-level checks and explicit real-service / contract-fixture boundaries."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

from .common import ROOT, Blocked, Context, check, digest, load, object_sha, parse_junit, table
from .stages_core import mission


def text_convert(c: Context) -> dict:
    tools = c.documents()
    out = tools.convert(c.config['documents'])
    c.write('converted.json', out)
    check(bool(out['documents']), 'No document converted')
    return {'documents': len(out['documents']), 'parsers': [d['parser'] for d in out['documents']]}


def text_clean(c: Context) -> dict:
    tools = c.documents()
    before = load(c.previous('30-text-convert', 'converted.json'))['documents']
    out = tools.clean(before)
    c.write('cleaned.json', out)
    c.write('cleaning-diff.json', [{'document_id': x['document_id'], 'changes': x['changes'],
                                  'semantic_rewrite_performed': x['semantic_rewrite_performed'],
                                  'raw_text_sha256': x['raw_text_sha256'], 'clean_text_sha256': x['clean_text_sha256']}
                                 for x in out['documents']])
    for i, doc in enumerate(out['documents']):
        c.path(f'cleaned-{i:02d}.md').write_text(doc['text'], encoding='utf-8')
    return {'documents': len(out['documents']), 'semantic_rewrite': False,
            'changes': sum(len(x['changes']) for x in out['documents'])}


def rag(c: Context) -> dict:
    external = c.stage['id'] == '37-rag-hybrid'
    settings = c.settings(external=external)
    if external and not (settings.embedding_model_path and settings.reranker_model_path):
        raise Blocked('Configure both embedding_model_path and reranker_model_path with real local weights in host settings; no automatic downloads')
    tools = c.documents(external=external)
    tools.documents = load(c.previous('31-text-clean', 'cleaned.json'))['documents']
    out = tools.retrieve(c.config['query'])
    c.write('retrieval.json', out)
    table(c.path('hits.csv'), out.get('hits', []))
    if not out.get('hits'):
        return {'_status': 'PARTIAL', 'reason': 'NO_MATCH', 'query': c.config['query']}
    check(all(x['source_sha256'] and x['source_locator'] and x['diagnostic_vote_allowed'] is False
              for x in out['hits']), 'Missing citation provenance or unsafe diagnostic vote')
    if external and out['retrieval_mode'] != 'BM25_DENSE_RRF_CROSS_ENCODER':
        return {'_status': 'PARTIAL', 'reason': 'Requested hybrid/reranker did not execute fully',
                'actual_mode': out['retrieval_mode'], 'warnings': out.get('warnings', [])}
    return {'hits': len(out['hits']), 'actual_mode': out['retrieval_mode'], 'diagnostic_vote_allowed': False}


def rag_metrics(c: Context) -> dict:
    # Explicitly small, fixed fixture; never call it a real corpus or diagnosis benchmark.
    tools = c.documents()
    fixture = c.workspace / 'inputs' / 'retrieval-gold.md'
    tools.clean(tools.convert([str(fixture)])['documents'])
    queries = [('q1', 'gyro bias residual', 'GYRO_SENTINEL'),
               ('q2', 'reaction wheel torque speed', 'WHEEL_SENTINEL')]
    rows = []
    for qid, query, marker in queries:
        out = tools.retrieve(query, top_k=5)
        positions = [i + 1 for i, hit in enumerate(out['hits']) if marker in hit['text']]
        rows.append({'query_id': qid, 'query': query, 'expected_marker': marker,
                     'recall_at_5': float(bool(positions)),
                     'reciprocal_rank': 1.0 / positions[0] if positions else 0.0,
                     'citation_integrity': all(x['source_sha256'] == digest(fixture) for x in out['hits']),
                     'result': out})
    negative = tools.retrieve('zzzz_no_such_term_129944')
    report = {'scope': 'TWO_POSITIVE_ONE_NEGATIVE_SYNTHETIC_FIXTURE_NOT_CORPUS_OR_DIAGNOSIS_BENCHMARK',
              'mean_recall_at_5': sum(x['recall_at_5'] for x in rows) / len(rows),
              'mrr': sum(x['reciprocal_rank'] for x in rows) / len(rows),
              'negative_rejected': not negative['hits'],
              'citation_integrity': all(x['citation_integrity'] for x in rows), 'cases': rows,
              'ragas_used': False, 'negative_result': negative}
    c.write('rag-metrics.json', report)
    check(report['mean_recall_at_5'] == 1 and report['negative_rejected'] and report['citation_integrity'], 'Retrieval fixture failed')
    return {k: v for k, v in report.items() if k not in ('cases', 'negative_result')}


def mineru(c: Context) -> dict:
    s = c.settings(external=True)
    if not (s.allow_mineru and s.mineru_executable):
        raise Blocked('MinerU not enabled/configured; protocol fixtures are separate in stage 36')
    source = c.required('nontext_document')
    if source.suffix.lower() in {'.txt', '.md', '.markdown'}:
        raise Blocked('MinerU acceptance requires a real PDF/image/other supported non-text input')
    out = c.documents(external=True).convert([str(source)])
    c.write('conversion.json', out)
    check(out['documents'] and all(d['parser'] == 'MINERU_CLI' for d in out['documents']), 'MinerU was not executed')
    return {'parsed_documents': len(out['documents']), 'visual_quality_manually_review_required': True,
            'semantic_accuracy_evaluated': False}


def external_task(c: Context) -> dict:
    from astrograph.agents.mission.models import AgentRequest
    from astrograph.agents.mission.service import MissionAgent
    s = c.settings(external=True)
    build = c.stage['id'] == '35-graph-build'
    if build:
        if not ((s.allow_a2o and s.a2o_source_root) or (s.planner == 'llm_assisted' and s.llm_config)):
            raise Blocked('Configure real Anything2Ontology source or structured LLM; no synthetic ontology will be substituted')
        if not s.allow_a2o and s.llm_config:
            check(Path(s.llm_config).is_file(), 'Missing configured LLM file')
    else:
        if not (s.allow_external_literature and (s.paperseek_url or s.paperseek_source_root)):
            raise Blocked('PaperSeek endpoint/source is not configured and authorized')
        # Direct deterministic research query verifies PaperSeek independently of LLM planning.
        s = s.model_copy(update={'planner': 'deterministic', 'llm_config': None})
    request = AgentRequest(task_id='acceptance-' + ('graph-build' if build else 'paperseek'),
                           task='build_knowledge' if build else 'research', goal=c.config['query'],
                           documents=c.config['documents'] if build else [])
    c.write('request.json', request.model_dump(mode='json'))
    result = MissionAgent(s).run_supervised(request)
    c.write('result.json', result.model_dump(mode='json'))
    if result.status not in ('SUCCEEDED', 'PARTIAL'):
        raise RuntimeError(f'External task returned {result.status}: {result.output}')
    if build:
        candidate = result.output.get('task_output', {})
        c.write('graph-candidate.json', candidate)
        check(candidate.get('active_kg_modified') is False, 'Graph release boundary violated')
        return {'_status': 'PASSED' if candidate.get('status') == 'REVIEW_REQUIRED' else 'PARTIAL',
                'candidate_status': candidate.get('status'), 'backend': candidate.get('backend', 'A2O_ADAPTER'),
                'domain_quality_not_evaluated': True, 'review_required': True}
    sources = result.output.get('sources', [])
    c.write('literature.json', {'sources': sources, 'limitations': result.limitations})
    return {'_status': 'PASSED' if sources and result.status == 'SUCCEEDED' else 'PARTIAL',
            'sources': len(sources), 'limitations': result.limitations, 'full_text_download_verified': False}


def real_llm(c: Context) -> dict:
    s = c.settings(external=True)
    if s.planner != 'llm_assisted' or not s.llm_config:
        raise Blocked('Provide real llm_assisted settings and llm_config; no fixture backend is used by this stage')
    rows = []
    for mode in ('thinking', 'creative'):
        result = mission(c, mode, external=True, task_id='llm-' + mode, documents=True)
        c.write(mode + '-result.json', result.model_dump(mode='json'))
        records = result.plan.get('llm_records', result.plan.get('planner_records', []))
        rows.append({'mode': mode, 'mission_status': result.status, 'mode_receipt': result.mode_receipt,
                     'plan': result.plan, 'records': records,
                     'exploration': result.output.get('exploration'), 'narrative':result.output.get('narrative'),
                     'capability_status':result.capability_status})
    from astrograph.llm.trace_verification import verify_task
    for row in rows:
        row['wire_audit']=verify_task(c.out/'tasks'/('llm-'+row['mode']))
    # Any fallback/abstention remains visible; structural acceptance is not benefit.
    accepted = live_llm_accepted(rows) and all(r['wire_audit']['status']=='VERIFIED' for r in rows)
    out = {'_status': 'PASSED' if accepted else 'PARTIAL',
           'rows': rows, 'domain_planning_quality_evaluated': False,
           'note': 'Live transport/output-contract check, not an accuracy or scientific-discovery benchmark.'}
    c.write('llm-acceptance.json', out)
    return out



def live_llm_accepted(rows: list[dict]) -> bool:
    """Inspect typed receipt fields; untrusted narrative cannot claim acceptance."""
    if {r.get('mode') for r in rows} != {'thinking', 'creative'}:
        return False
    for row in rows:
        records = row.get('records', [])
        if row.get('mission_status') != 'SUCCEEDED' or row.get('plan', {}).get('llm_calls', 0) < 1:
            return False
        if not any(r.get('phase') == 'TASK_PLAN' and r.get('outcome') == 'ACCEPTED' for r in records):
            return False
        if any(r.get('outcome') in {'DETERMINISTIC_FALLBACK','ABSTAINED'} for r in records):
            return False
        if not any(r.get('phase')=='EXPLANATION' and r.get('outcome')=='STRUCTURE_AND_CITATIONS_VALIDATED' for r in records):
            return False
        if not (row.get('narrative') or {}).get('content',{}).get('paragraphs'):
            return False
        if row['mode'] == 'creative':
            e = row.get('exploration') or {}
            if not (e.get('status') == 'UNVERIFIED_PROPOSAL' and e.get('hypotheses') and e.get('diagnostic_vote_allowed') is False
                    and e.get('memory_write') == 'NONE' and e.get('automatic_execution') is False):
                return False
    return True


def tests(c: Context) -> dict:
    targets = ['tests/implementation/mission/test_external_contracts.py'] if c.stage['id'] == '36-external-contracts' else [
        'tests/implementation/mission', 'tests/implementation/acceptance', 'tests/implementation/p1_p4']
    if c.stage['id']=='62-agent-reliability':targets=['tests/implementation/reliability']
    if c.stage['id']=='63-context-budget':targets=['tests/implementation/context_transport']
    if c.stage['id']=='64-citation-grounding':targets=['tests/implementation/grounding']
    if c.stage['id']=='65-agent-quality':targets=['tests/implementation/agent_quality']
    error = None
    try:
        c.worker(['-m', 'pytest', '-q', '-ra', '-p', 'no:cacheprovider', *targets,
                  '--junitxml=' + str(c.path('junit.xml'))], timeout=900, name='pytest')
    except RuntimeError as exc:
        error = exc
    report = parse_junit(c.path('junit.xml')) if c.path('junit.xml').exists() else {'tests': 0, 'errors': 1}
    report['verification_level'] = 'EXTERNAL_PROTOCOL_FIXTURES_ONLY' if len(targets) == 1 else 'SELECTED_MODE_AND_ACCEPTANCE_REGRESSION'
    if c.stage['id']=='62-agent-reliability':report['verification_level']='FAULT_INJECTION_AND_LOOPBACK_CONTRACTS_NOT_LIVE_LLM'
    if c.stage['id']=='63-context-budget':report['verification_level']='LOSSLESS_CONTEXT_CONTRACT_NOT_LIVE_LLM'
    if c.stage['id']=='64-citation-grounding':report['verification_level']='SOURCE_GROUNDING_WIRE_LOOPBACK_NOT_LIVE_LLM'
    if c.stage['id']=='65-agent-quality':report['verification_level']='QUALITY_EVALUATOR_CONTRACTS_NOT_LIVE_MODEL_GAIN'
    c.write('test-summary.json', report)
    if error:
        raise error
    check(report['tests'] > 0 and report.get('failures', 0) == 0 and report.get('errors', 0) == 0, 'Regression failed or collected no tests')
    return report


def memory(c: Context) -> dict:
    from astrograph.memory.service import CaseMemoryService, MemoryPolicy, PendingCase
    from astrograph.orchestration.models import WorkflowContext, PermissionScope as P
    t = datetime(2026, 9, 13, tzinfo=timezone.utc)
    policy = MemoryPolicy(profile_sha256=object_sha({'purpose': 'synthetic-local-acceptance'}),
                          approved_reviewers=('acceptance-review-fixture',))
    store = CaseMemoryService(c.path('acceptance-only.sqlite'), policy)
    actor = WorkflowContext(workflow_id='local-acceptance', run_id='memory-fixture', actor='fixture-agent',
                            permissions=[P.READ_MEMORY, P.WRITE_MEMORY])
    reviewer = WorkflowContext(workflow_id='local-acceptance', run_id='memory-review-fixture',
                               actor='acceptance-review-fixture', permissions=[P.REVIEW_MEMORY])
    case = PendingCase(case_id='synthetic-case', source_group_id='synthetic-group-a', producer_actor=actor.actor,
                       symptom_ids=('GYRO_RESIDUAL_FIXTURE',), operating_mode='tracking',
                       proposed_fault_id='UNCONFIRMED_FIXTURE', evidence_ids=('synthetic-evidence',),
                       observed_end=t, end_reason='completed', output_valid=True, knowledge_version='fixture-v1')
    staged = store.stage_case(case, event_id='stage-fixture', context=actor)
    def retrieve(**changes):
        kwargs = dict(operating_mode='tracking', snapshot=store.snapshot(), cutoff_at=t + timedelta(seconds=5),
                      exclude_case_id='new-case', exclude_source_group_id='new-group', context=actor,
                      knowledge_version='fixture-v1')
        kwargs.update(changes)
        return store.retrieve(('GYRO_RESIDUAL_FIXTURE',), **kwargs)
    check(not retrieve(), 'Pending case leaked into confirmed retrieval')
    conf = dict(case_id=case.case_id, event_id='independent-fixture-review', verified_fault_id='FIXTURE_FAULT',
                truth_source_ref='fixture://synthetic-label-not-real-review', verified_at=t + timedelta(seconds=1),
                available_at=t + timedelta(seconds=2), expected_snapshot=0, context=reviewer)
    self_rejected = False
    try:
        store.confirm_case(**{**conf, 'context': actor})
    except PermissionError:
        self_rejected = True
    first = store.confirm_case(**conf)
    second = store.confirm_case(**conf)
    hits = retrieve()
    checks = {'pending_hidden': True, 'self_confirmation_rejected': self_rejected,
              'idempotent': first == second and store.snapshot() == 1,
              'source_isolated': not retrieve(exclude_source_group_id=case.source_group_id),
              'time_isolated': not retrieve(cutoff_at=t), 'retrieved_after_confirmation': len(hits) == 1,
              'no_diagnostic_vote': all(x['diagnostic_vote_allowed'] is False for x in hits)}
    check(all(checks.values()), 'Memory contract failed')
    out = {'scope': 'SYNTHETIC_ISOLATED_FIXTURE_NOT_REAL_DIAGNOSIS_CONFIRMATION', 'checks': checks,
           'pending_receipt': staged, 'confirmation_receipt': first, 'retrieved_cases': hits,
           'production_memory_touched': False, 'ppl_invoked': False}
    c.write('memory-report.json', out)
    return {'checks': checks, 'production_memory_touched': False}


HANDLERS = {'25-memory': memory, '30-text-convert': text_convert, '31-text-clean': text_clean,
            '32-rag': rag, '33-mineru': mineru, '34-paperseek': external_task, '35-graph-build': external_task,
            '36-external-contracts': tests, '37-rag-hybrid': rag, '38-rag-evaluation': rag_metrics,
            '39-llm-planning': real_llm, '60-regression': tests, '62-agent-reliability': tests, '63-context-budget': tests, '64-citation-grounding':tests, '65-agent-quality':tests}
