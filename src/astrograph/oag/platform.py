"""Bind the existing upload service to native graph/three-task diagnostic review."""
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from .contracts import OAGInput, LiteratureDocument, content_hash
from .harness import execute_harness
from astrograph.integrations.platform_knowledge import UploadedDocumentKnowledgeIndex


class _KnowledgeClient:
    def __init__(self, paths, service, output):
        self.index = UploadedDocumentKnowledgeIndex(top_k=service.config.local_top_k)
        self.chunks, self.receipts = self.index.build(paths)
        if any(r['status'] != 'PARSED' for r in self.receipts):
            raise ValueError('uploaded knowledge document could not be parsed')
        self.remote = service.paperseek_client
        self.source_tools = None
        if service.config.source_tool_url:
            from astrograph.training.source_tools import SourceToolHTTPClient
            if not service.config.source_tool_adapter_sha256:
                raise ValueError('configured source adapter requires its SHA-256 identity')
            self.source_tools = SourceToolHTTPClient(service.config.source_tool_url,
                adapter_sha256=service.config.source_tool_adapter_sha256)
        self.output = output
        # Immutable content/configuration identity for harness replay.
        self.config = {'sources': [r['source_sha256'] for r in self.receipts],
                       'source_tool_url': service.config.source_tool_url,
                       'source_tool_adapter_sha256': service.config.source_tool_adapter_sha256,
                       'remote_config': str(getattr(self.remote, 'config', None))}

    def search(self, query):
        documents = []
        source_results = []
        hits = self.index.retrieve(query, self.chunks)
        for h in hits:
            text = h['quote']
            digest = hashlib.sha256(text.encode()).hexdigest()
            documents.append(LiteratureDocument(
                document_id=h['knowledge_id'], title=h['filename'], text=text,
                source_url='urn:sha256:' + digest,
                source_content_sha256=digest, retrieved_at=datetime.now(timezone.utc).isoformat(),
            ).model_dump(mode='json'))
            if self.source_tools and len(source_results) < 4:
                from astrograph.training.source_tools import SourceTaskRequest, source_target
                import re
                # Only explicit source-code lines enter the trained task scope.
                lines = [line for line in text.splitlines() if re.match(r'^\s*ParameterEffect\(.*\),?\s*$', line)]
                for index in range(0, min(len(lines), 2), 2):
                    excerpt = '\n'.join(lines[index:index+2])
                    for task in ('ONTOLOGY_EXTRACTION','RETRIEVAL_QUERY'):
                        request = SourceTaskRequest(task=task, component='simulation_component',
                            document_id=h['knowledge_id'], text=excerpt)
                        try:
                            source_target(request)
                        except (ValueError,SyntaxError):
                            source_results.append({'status':'OUTSIDE_TRAINED_SCOPE','document_id':h['knowledge_id']})
                            continue
                        result = self.source_tools.run(request)
                        source_results.append(result)
        refinement_queries = []
        for result in source_results:
            if result.get('task') != 'RETRIEVAL_QUERY':
                continue
            refined = result['output']['query']
            refinement_queries.append(refined)
            # One bounded refinement pass; its output never recursively invokes
            # the planner or acquires diagnostic voting authority.
            for h in self.index.retrieve(refined, self.chunks):
                digest = hashlib.sha256(h['quote'].encode()).hexdigest()
                if any(d['document_id'] == h['knowledge_id'] for d in documents):
                    continue
                documents.append(LiteratureDocument(document_id=h['knowledge_id'],
                    title=h['filename'], text=h['quote'], source_url='urn:sha256:'+digest,
                    source_content_sha256=digest, retrieved_at=datetime.now(timezone.utc).isoformat()).model_dump(mode='json'))
        remote = self.remote.search(query) if self.remote else {'status': 'NOT_CONFIGURED', 'documents': []}
        if self.remote and remote.get('status') != 'SUCCEEDED':
            raise ValueError('configured PaperSeek retrieval failed')
        documents.extend(remote.get('documents', []))
        unique = {}
        for d in documents:
            item = LiteratureDocument.model_validate(d)
            prior = unique.setdefault(item.document_id, item.model_dump(mode='json'))
            if prior != item.model_dump(mode='json'):
                raise ValueError('conflicting retrieved document identities')
        result = {'status': 'SUCCEEDED', 'documents': list(unique.values()),
                  'source_task_candidates': source_results,
                  'refinement_queries': refinement_queries,
                  'local_receipts': self.receipts, 'local_hits': hits,
                  'paperseek_status': remote['status']}
        (self.output / 'retrieval.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        return result


def run_uploaded_oag(path: Path, *, document_paths: list[Path], output: Path, job_id: str, service) -> dict:
    """Validated uploads select no executable providers or filesystem paths."""
    data = OAGInput.model_validate_json(path.read_text(encoding='utf-8'))
    data = OAGInput.model_validate({**data.model_dump(), 'review_evidence': True})
    output.mkdir(parents=True, exist_ok=True)
    if service._paperseek_configuration_error:
        raise ValueError(service._paperseek_configuration_error)
    client = _KnowledgeClient(document_paths, service, output) if document_paths or service.paperseek_client else None
    graph = Path(__file__).resolve().parents[3] / 'benchmarks/knowledge_graph/online_extended'
    if data.three_task_result and data.three_task_result.evidence_projection:
        from astrograph.evidence import ConfiguredRuntimeEvidenceBuilder
        from .native_bridge import write_package_graph
        root = Path(__file__).resolve().parents[3]
        binding = data.three_task_result.evidence_projection
        matched = None
        for relative in ('adcs_v5_2_2/runtime_evidence.json',
                         'adcs_v5_2_2/runtime_window_integrated_evidence.json',
                         'adcs_v5_2_2/runtime_window_evidence.json', 'phm_v6_0/runtime_evidence.json'):
            package = root / 'configs/datasets' / relative
            builder = ConfiguredRuntimeEvidenceBuilder(package)
            if builder.package_sha256 == binding.evidence_package_sha256:
                if [d.model_dump() for d in builder.package.evidence_definitions] != [d.model_dump() for d in data.bundle.evidence_definitions]:
                    raise ValueError('uploaded native definitions differ from the configured package')
                matched = package
                break
        if matched is None:
            raise ValueError('native projection refers to an unconfigured evidence package')
        graph = write_package_graph(matched, output / 'native-graph')

    ontology_context = None
    if service.config.a2o_enabled and document_paths:
        a2o = service.a2o_adapter.run(document_paths, output / 'ontology-toolchain')
        (output / 'anything2ontology.json').write_text(json.dumps(a2o, ensure_ascii=False, indent=2), encoding='utf-8')
        if a2o.get('status') != 'REVIEW_REQUIRED' or a2o.get('active_kg_modified') is not False:
            raise ValueError('configured Anything2Ontology did not produce review-only artifacts')
        ontology_context = {'status':'REVIEW_REQUIRED','active_kg_modified':False,
            'receipt_sha256':content_hash(a2o), 'candidate_material':[{
                k:item[k] for k in ('ontology_id','sha256','quote','knowledge_validation_status','role',
                                   'diagnostic_vote_allowed','active_kg_write_allowed')
            } for item in a2o['review_material']]}
    result = execute_harness(data, graph_dir=graph, ontology_context=ontology_context, output_root=output / 'harness', request_id=job_id,
                             paperseek_client=client, llm_backend=service.llm_backend,
                             ontology_backend=service.llm_backend if data.require_ontology_extraction else None)
    if result['status'] != 'SUCCEEDED':
        raise ValueError('OAG diagnostic review failed: ' + str(result.get('error')))
    return result


def render_oag_report(result: dict) -> str:
    output = result['output']
    summary, assessment = output['summary'], output['assessment']
    lines = ['# 三任务与图谱证据诊断报告', '',
             '状态：' + summary['status'], '候选：' + str(summary['suspected_fault_id']), '',
             '## 证据评估', '',
             '|候选|评估|支持证据|反证|尚未满足的必要条件|', '|---|---|---|---|---|']
    for h in assessment['hypotheses']:
        lines.append('|'+ '|'.join([h['fault_id'], h['verdict'], ', '.join(h['supporting_evidence_ids']),
                                   ', '.join(h['contradicting_evidence_ids']), ', '.join(h['missing_required_definition_ids'])])+'|')
    if 'diagnostic_decision' in summary:
        decision = summary['diagnostic_decision']
        lines += ['', '## 有限候选与区分观测', '',
                  '状态：' + str(decision.get('status') or 'UNKNOWN'),
                  '候选集合：' + (', '.join(decision.get('candidate_fault_ids') or []) or '证据不足或候选范围过大'),
                  '候选集合不等于已确认的具体故障。']
        diagnostic_scopes = decision.get('diagnostic_scopes')
        if diagnostic_scopes is not None:
            lines += ['', json.dumps(diagnostic_scopes, ensure_ascii=False, indent=2)]
        additional_observations = decision.get('additional_observations')
        if additional_observations is not None:
            lines += ['', json.dumps(additional_observations, ensure_ascii=False, indent=2)]
        elif decision.get('candidate_explanations') is not None:
            # evidence-aware-fusion.v1 carries bounded candidate explanations
            # and reason codes rather than complementary_diagnosis' proposed
            # observation list.  Render the schema actually present instead of
            # manufacturing an additional-observation claim.
            lines += ['', json.dumps({
                'reason_codes': decision.get('reason_codes') or [],
                'candidate_explanations': decision.get('candidate_explanations') or [],
            }, ensure_ascii=False, indent=2)]
    lines += ['', '## 模型解释', '', json.dumps(output['generation'], ensure_ascii=False, indent=2),
              '', '## 三任务输出', '', json.dumps(summary.get('three_task_results', {'status': 'NOT_PROVIDED'}), ensure_ascii=False, indent=2),
              '', '软件诊断候选；缺少因果辨识/部署资格时不作确认或控制授权。',
              '', '## 报告自动审计', '', json.dumps(output['generation'].get('report_audit', {}), ensure_ascii=False, indent=2),
              '', '## 执行阶段', '']
    lines.extend(f"- {s['stage_id']}: {s['status']} ({s['elapsed_ms']:.1f} ms)" for s in result['stages'])
    return '\n'.join(lines)
