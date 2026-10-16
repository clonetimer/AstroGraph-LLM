"""Evidence-fusion companion for upstream-result + logs + references.

Candidate IDs are *never* generated from literature, graph or memory. Those
sources enrich hypotheses with located review cues, possible counterevidence
and independently confirmed prior cases. Graph lookups are read-only. No
control advice is auto-approved and no numeric 'fault probability' is made up.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

from astrograph.integrations.platform_knowledge import UploadedDocumentKnowledgeIndex, _LocalChunk
from astrograph.integrations.result_log_review import SourceLine, _rules, load_manifest, read_lines
from astrograph.integrations.material_case_memory import MaterialCaseMemory
from astrograph.integrations.material_graph_builder import build_material_graph
from astrograph.integrations.material_multihop import (
    evaluate_multihop, evaluate_bidirectional_hypotheses, evaluate_causal_chains, default_public_graph_path,
)
from astrograph.integrations.material_hierarchical_rag import HierarchicalMaterialIndex, RetrievalBudgets

MATERIAL_REASONING_VERSION = 'astrograph-material-knowledge-fusion.v1'


def _graph_review_hops() -> int:
    """Administrator-only opt-in; historical default remains three hops."""
    value = os.environ.get('ASTROGRAPH_MATERIAL_GRAPH_MAX_HOPS', '3').strip()
    if value not in {'3', '4'}:
        raise ValueError('ASTROGRAPH_MATERIAL_GRAPH_MAX_HOPS must be 3 or 4')
    return int(value)
_DATE = re.compile(r'\b(20\d\d-\d\d-\d\d[T ]\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:?\d\d)?)\b')
_NEGATIVE = re.compile(r'\b(?:no|not|normal|nominal|cleared|clear|resolved|recovered|within.range)\b|(?:正常|恢复|未出现|未发现|无异常|已排除|已解除)', re.I)
_WARNING = re.compile(r'\b(?:warn|warning|error|critical|failed|fault|timeout|alarm)\b|(?:告警|异常|故障|超时|报警|失效)', re.I)


def _date(text: str) -> str | None:
    match = _DATE.search(text)
    if not match:
        return None
    raw = match.group(1).replace(' ', 'T').replace('Z','+00:00')
    try:
        # Naive timestamps are explicitly left timezone-unverified.
        return datetime.fromisoformat(raw).isoformat()
    except ValueError:
        return None


def _receipt(line: SourceLine, index: int) -> dict[str, Any]:
    return {'evidence_id':f'MLOG-{index:05d}', 'role':'log_reports',
            'filename':line.filename, 'locator':line.locator, 'sha256':line.sha256,
            'excerpt':line.text[:420], 'timestamp':_date(line.text),
            'severity':'WARNING_OR_ERROR' if _WARNING.search(line.text) else 'INFORMATIONAL'}


def _ngrams(text: str) -> set[str]:
    # Latin subsystem/fault identifiers + Chinese character 2/3-grams.
    latin = set(re.findall(r'[A-Za-z][A-Za-z0-9_]{2,}', text.casefold()))
    chinese = re.findall(r'[\u4e00-\u9fff]+', text)
    grams = {segment[i:i+n] for segment in chinese for n in (2,3)
             for i in range(max(0,len(segment)-n+1))}
    return latin | grams


def _ngram_hits(query: str, chunks: list[Any]) -> list[str]:
    terms = _ngrams(query)
    if not terms:
        return []
    ranked = []
    for c in chunks:
        target = _ngrams(' '.join(c.section_path)+' '+c.text[:6000])
        overlap = terms & target
        if overlap:
            # Penalise coincidences involving only common short CJK bigrams.
            score = len(overlap) / math.sqrt(max(1,len(terms))*max(1,len(target)))
            if score > 0:
                ranked.append((score,c.chunk_id))
    return [cid for _score,cid in sorted(ranked, key=lambda x:(-x[0],x[1]))[:16]]


def _semantic_rank(queries: list[str], chunks: list[Any], *,
                   equipment_scope: str | None = None) -> tuple[list[list[str]], dict[str, Any]]:
    """Optional local embeddings. Offline revision-bound disk cache if configured."""
    path_value = os.environ.get('ASTROGRAPH_MATERIAL_EMBEDDING_MODEL','').strip()
    if not path_value:
        return ([[] for _ in queries], {'status':'NOT_CONFIGURED','model':'NONE'})
    model_path = Path(path_value).expanduser()
    if not model_path.is_dir():
        return ([[] for _ in queries], {'status':'UNAVAILABLE_LOCAL_MODEL_MISSING'})
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(str(model_path), local_files_only=True, trust_remote_code=False)
        qdrant_url=os.environ.get('ASTROGRAPH_MATERIAL_QDRANT_URL','').strip()
        if qdrant_url:
            from astrograph.retrieval.qdrant_vectors import qdrant_rank
            orders, status = qdrant_rank(
                queries=queries,chunks=chunks,model=model,model_path=model_path,
                scope=equipment_scope or '',url=qdrant_url,
                collection=os.environ.get('ASTROGRAPH_MATERIAL_QDRANT_COLLECTION','astrograph_material_embeddings'),
                api_key=os.environ.get('ASTROGRAPH_MATERIAL_QDRANT_API_KEY') or None)
        else:
            from astrograph.retrieval.persistent_vectors import semantic_rank_with_cache
            orders, status = semantic_rank_with_cache(
                queries, chunks, model=model, model_path=model_path, scope=equipment_scope,
                cache_path=os.environ.get('ASTROGRAPH_MATERIAL_VECTOR_CACHE_DB') or None,
            )
        return orders, status
    except Exception as exc:
        return ([[] for _ in queries], {'status':'FAILED','error_type':type(exc).__name__,
                                        'network_download_allowed':False})


def _retrieval_query(candidate: dict[str, Any], rule: dict[str, Any] | None) -> str:
    terms = [str(candidate['fault_id']), str(candidate['fault_name'])]
    # Rule aliases and diagnostic vocabulary are question-expansion hints,
    # NOT independent knowledge or a verified diagnosis.
    if rule:
        terms.extend(rule.get('aliases', [])[:5])
        terms.extend(rule.get('reference_terms', [])[:8])
    return ' '.join(dict.fromkeys(x.strip() for x in terms if x.strip()))[:900]


def _hybrid_retrieve(index: UploadedDocumentKnowledgeIndex, chunks: list[Any], query: str,
                     semantic_ids: list[str], *, top_k: int = 4) -> dict[str, Any]:
    lexical = index.retrieve(query, chunks)
    lexical_ids = [x['knowledge_id'] for x in lexical]
    char_ids = _ngram_hits(query, chunks)
    rankings = [('BM25',lexical_ids,1.0), ('CHAR_NGRAM',char_ids,0.65)]
    if semantic_ids:
        rankings.append(('LOCAL_EMBEDDING',semantic_ids,1.2))
    scores: dict[str,float] = {}
    channels: dict[str,list[str]] = {}
    for channel, order, weight in rankings:
        for i, cid in enumerate(order,1):
            scores[cid] = scores.get(cid,0.0) + weight/(60+i)
            channels.setdefault(cid,[]).append(channel)
    by_id = {c.chunk_id:c for c in chunks}
    hits = []
    for cid in sorted(scores,key=lambda c:(-scores[c],c))[:top_k]:
        c=by_id[cid]
        # The quote is an excerpt of this precise source chunk, never generated.
        hits.append({'knowledge_id':cid,'filename':c.filename,'source_sha256':c.source_sha256,
                     'source_locators':list(c.source_locators),'section_path':list(c.section_path),
                     'quote':c.text[:900],'fusion_score':round(scores[cid],8),
                     'retrieval_channels':channels[cid],
                     'knowledge_validation_status':('EXPERT_APPROVED_HISTORICAL_REFERENCE' if c.review_status=='EXPERT_APPROVED_REFERENCE' else 'UNREVIEWED_UPLOADED_REFERENCE'),
                     'support_scope':'TOPICAL_KNOWLEDGE_ONLY_NOT_OBSERVED_CAUSE',
                     'diagnostic_vote_allowed':False})
    return {'status':'SUCCEEDED' if hits else 'NO_RELEVANT_PASSAGES', 'query':query,
            'methods':[x[0] for x in rankings], 'hits':hits}


def _log_events(lines: list[SourceLine], candidate: dict[str, Any],
                rule: dict[str, Any] | None, limit: int = 12) -> dict[str, Any]:
    terms = [str(candidate['fault_id']),str(candidate['fault_name'])]
    if rule:
        terms += list(rule.get('log_signals') or [])
    # Don't use tiny ambiguous aliases as observations.
    terms = [x.casefold() for x in terms if len(x.strip()) >= 4]
    events: list[dict[str, Any]] = []
    for i,line in enumerate(lines,1):
        if not any(term in line.text.casefold() for term in terms):
            continue
        receipt = _receipt(line,i)
        receipt['assessment'] = ('NEGATIVE_OR_RECOVERY_CUE' if _NEGATIVE.search(line.text)
                                  else 'POSSIBLE_SUPPORTING_CUE')
        receipt['temporal_causality_verified'] = False
        events.append(receipt)
        if len(events)>=limit:
            break
    timed = sorted([e for e in events if e['timestamp']],key=lambda e:e['timestamp'])
    # Untimed observations remain visible rather than being assigned fabricated timestamps.
    return {'status':'CUES_FOUND' if events else 'NO_MATCH', 'events':timed+[e for e in events if not e['timestamp']],
            'supporting_ids':[e['evidence_id'] for e in events if e['assessment']=='POSSIBLE_SUPPORTING_CUE'],
            'contradicting_ids':[e['evidence_id'] for e in events if e['assessment']=='NEGATIVE_OR_RECOVERY_CUE'],
            'unparseable_timestamp_count':len(events)-len(timed), 'time_base_verified':False,
            'temporal_causality_verified':False,
            'claim_boundary':'Keyword cues can be supporting or opposing checks; no independent time synchronisation or root-cause proof.'}


def _graph_retrieve(candidate_ids: list[str]) -> dict[str, Any]:
    """Existing version-bound Neo4j/CSV read-only query, opt-in via host config."""
    binding_path = os.environ.get('ASTROGRAPH_MATERIAL_KG_BINDING_JSON','').strip()
    if not binding_path:
        return {'status':'NOT_CONFIGURED','paths':[],'active_kg_modified':False}
    try:
        from astrograph.knowledge.runtime import Neo4jKGEvidenceBindingConfig, KGEvidenceRuntimeService
        from astrograph.knowledge.query import (ReadOnlyCypherCompiler, ReadOnlyGraphQueryExecutor,
            CsvEvidenceGraphReferenceExecutor, build_backward_fault_query_plan)
        binding_file = Path(binding_path).expanduser().resolve(strict=True)
        data = json.loads(binding_file.read_text(encoding='utf-8'))
        for key in ('fallback_nodes_csv','fallback_edges_csv'):
            if data.get(key):
                path = Path(data[key])
                data[key] = str((binding_file.parent/path).resolve()) if not path.is_absolute() else str(path)
        binding = Neo4jKGEvidenceBindingConfig.model_validate(data)
        if not candidate_ids:
            return {'status':'NO_CANDIDATE','paths':[],'active_kg_modified':False}
        ids=sorted(set(candidate_ids))[:5]
        plan=build_backward_fault_query_plan(ids,max_results=80)
        compiler=ReadOnlyCypherCompiler()
        compiled=compiler.compile(plan,knowledge_version=binding.knowledge_version)
        mode=os.environ.get('ASTROGRAPH_MATERIAL_KG_MODE','auto').lower()
        if mode not in {'auto','csv','neo4j'}:
            raise ValueError('invalid ASTROGRAPH_MATERIAL_KG_MODE')
        backend='neo4j'
        failure_type=None
        if mode in {'auto','neo4j'}:
            try:
                password=os.environ.get(binding.read_password_env)
                if not password:
                    raise RuntimeError('Neo4j read credential is not configured')
                from neo4j import GraphDatabase
                driver=GraphDatabase.driver(binding.uri,auth=(binding.read_user,password),connection_timeout=3.0)
                try:
                    driver.verify_connectivity()
                    res=ReadOnlyGraphQueryExecutor(driver).execute(compiled,knowledge_version=binding.knowledge_version)
                finally:
                    driver.close()
            except Exception as exc:
                failure_type=type(exc).__name__
                if mode=='neo4j' or binding.fallback_mode!='csv_on_unavailable':
                    raise
                mode='csv'
        if mode=='csv':
            if binding.fallback_mode!='csv_on_unavailable':
                raise PermissionError('CSV graph reference mode requires explicit binding fallback policy')
            nodes,edges=Path(str(binding.fallback_nodes_csv)),Path(str(binding.fallback_edges_csv))
            if (KGEvidenceRuntimeService._sha256_file(nodes).lower()!=binding.nodes_sha256.lower() or
                KGEvidenceRuntimeService._sha256_file(edges).lower()!=binding.edges_sha256.lower()):
                raise ValueError('bound graph CSV digest mismatch')
            res=CsvEvidenceGraphReferenceExecutor(nodes,edges).execute(plan,knowledge_version=binding.knowledge_version,compiler=compiler)
            backend='csv_reference_read_only'
        paths=[]
        for row in res.rows:
            if row.start_entity_id not in ids:
                continue
            path_id='KG-'+hashlib.sha256((binding.release_id+'|'+row.start_entity_id+'|'+str(row.relation_type)+'|'+row.target_entity_id).encode()).hexdigest()[:16].upper()
            paths.append({'graph_path_id':path_id,'fault_id':row.start_entity_id, 'evidence_definition_id':row.target_entity_id,
                          'relation_type':row.relation_type,
                          'path_entity_ids':row.path_entity_ids or [row.start_entity_id,row.target_entity_id],
                          'evidence_label':str(row.target_properties.get('name') or row.target_properties.get('label') or '')[:180],
                          'relation_provenance':row.relationship_properties,
                          'knowledge_version':binding.knowledge_version,
                          'review_status':'KG_RELATION_NOT_LOG_OBSERVATION'})
        return {'status':'SUCCEEDED', 'backend':backend, 'paths':paths[:80],
                'fallback_reason_type':failure_type,'release_id':binding.release_id,
                'knowledge_version':binding.knowledge_version, 'nodes_sha256':binding.nodes_sha256,
                'edges_sha256':binding.edges_sha256,'query_sha256':res.query_sha256,
                'truncated':res.truncated,'release_hashes_verified_against_backend': backend=='csv_reference_read_only',
                'release_identity_externally_declared':backend=='neo4j','active_kg_modified':False,
                'claim_boundary':'Read-only bound graph associations describe expected evidence, not observed hardware failure.'}
    except Exception as exc:
        return {'status':'FAILED','paths':[],'error_type':type(exc).__name__,
                'error_message':str(exc)[:300],'active_kg_modified':False}


def build_competing_hypothesis_review(items: list[dict[str, Any]],
                                      shared_symptoms: list[dict[str, Any]]) -> dict[str, Any]:
    """Deterministic competition table; references cannot change the diagnosis.

    Absence of a retrieved passage, log keyword or expected symptom is a
    missing check, never an observed negative or proof of another candidate.
    """
    rows = []
    for item in items[:12]:
        bidir = item.get('bidirectional_graph_review') or {}
        log = item.get('log_evidence_matrix') or {}
        reviewed = ((item.get('reference_retrieval') or {})
                    .get('reviewed_counterevidence') or {})
        rows.append({
            'fault_id': item['fault_id'],
            'fault_name': item['fault_name'],
            'supporting_log_cue_ids': list(dict.fromkeys(log.get('supporting_ids') or []))[:12],
            'opposing_log_cue_ids': list(dict.fromkeys(log.get('contradicting_ids') or []))[:12],
            'approved_negative_relation_ids': [x['edge_id'] for x in reviewed.get('items', [])[:12]],
            'missing_expected_symptom_ids': list(bidir.get('missing_symptom_cue_ids') or [])[:16],
            'shared_symptom_ids': sorted({x['symptom_id'] for x in shared_symptoms
                 if item['fault_id'] in x.get('candidate_fault_ids', [])})[:16],
            'contrastive_reference': dict(item.get('contrastive_reference_review') or {}),
            'current_root_cause_confirmed': False,
            'status': 'NON_VOTING_DIFFERENTIAL_REVIEW',
        })
    # Causal path endpoints offer a non-voting DIFFERENTIAL over the already
    # frozen candidate set, not a new diagnosis or a counterfactual proof.
    # Avoid awarding extra votes for multiple citations of the same symptom.
    causal_index: dict[str, dict[str, list[dict[str, Any]]]] = {}
    symptom_owners: dict[str, set[str]] = defaultdict(set)
    truncated_candidates: list[str] = []
    for item in items[:12]:
        fid = item['fault_id']
        review = ((item.get('multi_hop') or {}).get('causal_chain_review') or {})
        if review.get('search_truncated'):
            truncated_candidates.append(fid)
        symptom_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for chain in (review.get('chains') or [])[:32]:
            sid = str(chain.get('terminal_symptom_id') or '')
            if not sid or not chain.get('structural_claim_chain_valid'):
                continue
            symptom_groups[sid].append(chain)
            symptom_owners[sid].add(fid)
        causal_index[fid] = symptom_groups
    causal_rows = []
    for item in items[:12]:
        fid = item['fault_id']
        groups = causal_index[fid]
        causal_rows.append({
            'fault_id': fid,
            'candidate_specific_reference_symptoms': [
                {'symptom_id':sid, 'chain_ids':sorted({p['path_id'] for p in paths})[:8],
                 'cross_document_context_unverified':any(
                     p.get('cross_document_mechanism_splice',False) for p in paths)}
                for sid,paths in sorted(groups.items()) if symptom_owners[sid] == {fid}
            ][:24],
            'shared_reference_symptom_ids': sorted(
                sid for sid in groups if len(symptom_owners[sid]) > 1)[:24],
            'missing_reference_or_observation_is_not_counterevidence': True,
            'current_fault_observation_verified': False,
            'causal_path_count': sum(len(paths) for paths in groups.values()),
        })
    causal_differential = {
        'schema_version':'astrograph-causal-hypothesis-differential.v1',
        'status': ('PARTIAL_REFERENCE_COVERAGE' if truncated_candidates
                   else 'REFERENCE_PATHS_FOR_COMPARISON' if symptom_owners
                   else 'NO_REVIEWED_CAUSAL_PATHS'),
        'candidates':causal_rows,
        'shared_reference_symptom_ids': sorted(
            sid for sid, owners in symptom_owners.items() if len(owners) > 1)[:24],
        'truncated_candidate_fault_ids':truncated_candidates,
        'candidate_specific_only_within_supplied_frozen_set':True,
        'no_counterfactual_causation_inferred':True,
        'current_root_cause_proven':False,
        'diagnostic_vote_allowed':False,
        'automatic_control_authorized':False,
    }
    return {
        'schema_version': 'astrograph-competing-hypothesis-review.v1',
        'status': 'READY_FOR_EXPERT_COMPARISON' if len(rows) >= 2 else
            'SINGLE_OR_NO_HYPOTHESIS',
        'candidates': rows, 'shared_symptoms': shared_symptoms[:24],
        'causal_path_differential': causal_differential,
        'missing_cue_is_not_negative_evidence': True,
        'reviewed_negative_claim_is_not_current_observation': True,
        'diagnostic_vote_allowed': False,
        'automatic_control_authorized': False,
    }


def analyze_material_knowledge(*, expanded: Path, manifest_path: Path,
                               review: dict[str, Any], memory: MaterialCaseMemory | None = None,
                               source_group_id: str = '') -> dict[str, Any]:
    """Turn 3 uploaded roles into fault-specific evidence comparison and reuse."""
    manifest, roles = load_manifest(expanded,manifest_path)
    candidates = list(review.get('candidates') or [])
    rules={rule['fault_id']:rule for rule in _rules()}
    log_lines=[line for path in roles['log_reports'] for line in read_lines(path,'log_reports')]
    index=UploadedDocumentKnowledgeIndex(top_k=12)
    chunks,ingestion=index.build(roles['references'])
    # One immutable snapshot per task, and no current-task self-retrieval.
    prior_snapshot=memory.snapshot() if memory else None
    historical_chunks=[]
    if memory is not None and prior_snapshot is not None:
        for prior in memory.retrieve_reference_chunks(
                equipment_scope=manifest.equipment_scope,
                exclude_source_group_id=source_group_id,snapshot=prior_snapshot):
            historical_chunks.append(_LocalChunk(
                chunk_id='approved:'+prior['document_id']+':'+prior['knowledge_id'],
                source_id='approved:'+prior['document_id'],filename=prior['filename'],
                source_sha256=prior['source_sha256'],text=prior['quote'],
                section_path=tuple(prior['section_path']),
                source_locators=tuple(prior['source_locators']),
                review_status='EXPERT_APPROVED_REFERENCE',content_safety_status='REVIEWED',
                modalities=tuple(prior.get('modalities') or ('TEXT',)),
            ))
    all_chunks=chunks+historical_chunks
    # Reuse a single immutable lexical index for all bounded candidate-specific
    # plans. The graph part remains a separately reviewed, read-only evidence lane.
    hierarchy=HierarchicalMaterialIndex(all_chunks)
    rag_limits=RetrievalBudgets(
        documents=int(os.environ.get('ASTROGRAPH_MATERIAL_RAG_DOC_BUDGET','8')),
        sections=int(os.environ.get('ASTROGRAPH_MATERIAL_RAG_SECTION_BUDGET','16')),
        hits=int(os.environ.get('ASTROGRAPH_MATERIAL_RAG_HIT_BUDGET','5')),
    )
    queries=[_retrieval_query(c,rules.get(c['fault_id'])) for c in candidates]
    semantic_orders,semantic_status=_semantic_rank(queries,all_chunks,
        equipment_scope=manifest.equipment_scope)
    graph=_graph_retrieve([c['fault_id'] for c in candidates])
    # The entire pinned, independently reviewed snapshot is required for
    # genuine 3-hop reasoning. The legacy retrieval API is only 2-hop and
    # cannot safely be treated as a full neighborhood.
    approved_edges=[]
    approved_graph_error=None
    if memory is not None and prior_snapshot is not None and manifest.equipment_scope:
        try:
            approved_edges=memory.approved_graph_for_reasoning(
                equipment_scope=manifest.equipment_scope, snapshot=prior_snapshot,
                exclude_source_group_id=source_group_id)
        except Exception as exc:
            approved_graph_error=type(exc).__name__
    # Global community search uses ONLY the already-frozen reviewed snapshot.
    # No LLM-generated summary, pending OCR, disputed or public-seed relation
    # may become a factual premise here.
    from astrograph.retrieval.approved_community_search import (
        build_approved_community_reports, global_community_search,
    )
    community_index=build_approved_community_reports(
        reviewed_edges=approved_edges, equipment_scope=manifest.equipment_scope,
        snapshot=prior_snapshot)
    if os.environ.get('ASTROGRAPH_MATERIAL_COMMUNITY_LLM_ENABLED','0').lower() in {'1','true','yes'}:
        try:
            from astrograph.integrations.material_graph_builder import _backend_from_environment
            from astrograph.retrieval.approved_community_search import llm_extract_community_reports
            community_backend = _backend_from_environment()
            if community_backend is not None:
                community_index = llm_extract_community_reports(
                    community_index,backend=community_backend,max_communities=8)
            else:
                community_index['llm_summary_status']='MODEL_NOT_CONFIGURED'
        except Exception as exc:
            community_index['llm_summary_status']='FAILED_'+type(exc).__name__
    community_global=global_community_search(community_index,
        ' '.join(queries)[:4000])
    # Graph evidence is compared across the host-supplied candidate set only.
    # Reverse symptom lookup is non-voting and never proposes new fault IDs.
    graph_hops = _graph_review_hops()
    bidirectional = evaluate_bidirectional_hypotheses(
        candidates=candidates[:12], equipment_scope=manifest.equipment_scope,
        log_lines=log_lines, reviewed_edges=approved_edges, snapshot=prior_snapshot,
        max_hops=graph_hops,
    ) if candidates else {'status': 'NO_CANDIDATES', 'candidates': []}
    bidirectional_by_id = {row['fault_id']: row for row in bidirectional['candidates']}
    contrastive_by_id = {}
    if len(candidates) >= 2:
        first, second = candidates[:2]
        comparison = hierarchy.retrieve_contrastive(
            base_query=' '.join(queries[:2])[:900],
            fault_id=first['fault_id'], fault_name=first['fault_name'],
            competitor_fault_id=second['fault_id'],
            competitor_fault_name=second['fault_name'],
            approved_edges=approved_edges, equipment_scope=manifest.equipment_scope,
            graph_snapshot=prior_snapshot, budgets=rag_limits,
        )
        # Report bounded pointers, not a second duplicate source-text payload.
        contrastive_by_id[first['fault_id']] = {
            'competitor_fault_id': second['fault_id'],
            'primary_only_chunk_ids': comparison['primary_only_chunk_ids'],
            'competitor_only_chunk_ids': comparison['competitor_only_chunk_ids'],
            'shared_source_sha256': comparison['shared_source_sha256'],
            'diagnostic_vote_allowed': False,
            'status': comparison['status'],
        }
    items=[]
    for n,candidate in enumerate(candidates):
        rule=rules.get(candidate['fault_id'])
        query=queries[n]
        retrieved=hierarchy.retrieve(
            base_query=query, fault_id=candidate['fault_id'],
            fault_name=candidate['fault_name'],
            reference_terms=list(rule.get('reference_terms') or []) if rule else [],
            semantic_ids=semantic_orders[n],
            approved_edges=approved_edges,
            equipment_scope=manifest.equipment_scope,
            graph_snapshot=prior_snapshot,
            budgets=rag_limits,
        )
        retrieved['reviewed_counterevidence'] = hierarchy.retrieve_reviewed_counterevidence(
            fault_id=candidate['fault_id'], fault_name=candidate['fault_name'],
            approved_edges=approved_edges, equipment_scope=manifest.equipment_scope,
            graph_snapshot=prior_snapshot,
        )
        global_candidate=global_community_search(community_index, query)
        logs=_log_events(log_lines,candidate,rule)
        kg_paths=[p for p in graph['paths'] if p['fault_id']==candidate['fault_id']]
        # Explicit evidence-definition-ID mentions, nothing inferred from a path alone.
        graph_cross_checks=[]
        for path in kg_paths:
            ev_id=path['evidence_definition_id']
            hits=[_receipt(line,i)['evidence_id'] for i,line in enumerate(log_lines,1)
                  if ev_id.casefold() in line.text.casefold()]
            graph_cross_checks.append({**path,'matched_log_ids':hits[:6],
                                       'observation_status':'UNVERIFIED_TEXT_ID_MENTION' if hits else 'NOT_EVALUATED_NO_MATCH'})
        if memory is None:
            cases={'status':'NOT_CONFIGURED','hits':[]}
        else:
            cases=memory.retrieve_confirmed(fault_id=candidate['fault_id'],
                         equipment_scope=manifest.equipment_scope,exclude_source_group_id=source_group_id,
                         snapshot=prior_snapshot)
        reviewed_graph = ({'status':'NOT_CONFIGURED','paths':[]} if memory is None else
              memory.retrieve_graph(fault_id=candidate['fault_id'],fault_name=candidate['fault_name'],
                    equipment_scope=manifest.equipment_scope,exclude_source_group_id=source_group_id,
                    snapshot=prior_snapshot))
        ids=[x['knowledge_id'] for x in retrieved['hits']]
        counter_ids=logs['contradicting_ids']
        multi_hop=evaluate_multihop(
            fault_id=candidate['fault_id'],fault_name=candidate['fault_name'],
            equipment_scope=manifest.equipment_scope,log_lines=log_lines,
            reviewed_edges=approved_edges,snapshot=prior_snapshot,
            public_graph_path=default_public_graph_path(),max_hops=graph_hops)
        # Independent R7 layer: association paths and causal document claims
        # must never share a field that implies causal proof or an authorized vote.
        multi_hop['causal_chain_review'] = evaluate_causal_chains(
            fault_id=candidate['fault_id'], fault_name=candidate['fault_name'],
            equipment_scope=manifest.equipment_scope, reviewed_edges=approved_edges,
            snapshot=prior_snapshot, max_hops=graph_hops,
        )
        if hierarchy.truncated:
            multi_hop['knowledge_gap_codes'].append('REFERENCE_CORPUS_NOT_FULLY_INDEXED')
            # Retrieval truncation does not erase independent graph evidence,
            # but must be visible in the answer policy and expert gap queue.
            multi_hop['reference_completeness']='PARTIAL_REVIEW_REQUIRED'
        if approved_graph_error:
            # Never downgrade a failed approved graph read into 'no knowledge'
            # without explicitly reporting that the backend was unavailable.
            multi_hop['decision']='ABSTAIN_APPROVED_GRAPH_READ_FAILED'
            multi_hop['response_mode']='ABSTAIN_FROM_ROOT_CAUSE'
            multi_hop['knowledge_gap_codes'].append('APPROVED_GRAPH_READ_FAILED')
            multi_hop['approved_graph_error_type']=approved_graph_error
        # Independent historical corrections or unresolved log contradictions
        # are counterevidence to the upstream candidate, not votes for a new
        # automatically confirmed diagnosis.
        historical_corrections=[case['case_id'] for case in cases['hits']
          if case['relation_status']=='PRIOR_UPSTREAM_MISCLASSIFICATION']
        if historical_corrections:
            multi_hop['historical_misclassification_case_ids']=historical_corrections
            if 'HISTORICAL_UPSTREAM_MISCLASSIFICATION' not in multi_hop['knowledge_gap_codes']:
                multi_hop['knowledge_gap_codes'].append('HISTORICAL_UPSTREAM_MISCLASSIFICATION')
        if (counter_ids or historical_corrections) and multi_hop['response_mode']=='EVIDENCE_BOUND_PROVISIONAL_ONLY':
            multi_hop['decision']=('ABSTAIN_HISTORICAL_CORRECTION_REQUIRES_REVIEW' if historical_corrections
                                   else 'ABSTAIN_CANDIDATE_LOG_CONTRADICTION')
            multi_hop['response_mode']='ABSTAIN_FROM_ROOT_CAUSE'
            if counter_ids:
                multi_hop['knowledge_gap_codes'].append('LOG_CONTRADICTION_REQUIRES_REVIEW')
        items.append({
            'fault_id':candidate['fault_id'],'fault_name':candidate['fault_name'],
            'hypothesis_status':'NOT_INDEPENDENTLY_VERIFIED',
            'hypothesis_texts':[x['text'] for x in candidate.get('cause_hypotheses',[])],
            'reference_retrieval':retrieved,'community_retrieval':global_candidate,
            'log_evidence_matrix':logs,
            'graph_evidence_paths':graph_cross_checks,'historical_confirmed_cases':cases,
            'approved_document_graph':reviewed_graph,
            'multi_hop':multi_hop,
            'bidirectional_graph_review': bidirectional_by_id.get(candidate['fault_id'], {}),
            'contrastive_reference_review': contrastive_by_id.get(candidate['fault_id'], {}),
            'evidence_assessment':{
                'upstream_assertion_ids':list(candidate.get('diagnosis_source_ids') or []),
                'supporting_log_cue_ids':logs['supporting_ids'],
                'contradicting_or_recovery_log_cue_ids':counter_ids,
                'topical_reference_ids':ids,
                'reference_retrieval_status':retrieved['status'],
                'reference_retrieval_proves_cause':False,
                'graph_association_count':len(kg_paths),
                'approved_document_graph_edge_count':len(reviewed_graph['paths']),
                'approved_document_graph_disputes':reviewed_graph.get('contradicted_relation_groups',0),
                'independently_confirmed_historical_case_ids':[x['case_id'] for x in cases['hits']],
                'historical_correction_ids':[x['case_id'] for x in cases['hits'] if x['relation_status']=='PRIOR_UPSTREAM_MISCLASSIFICATION'],
                'verified_current_root_cause':False, 'numeric_confidence_available':False,
                'multi_hop_decision':multi_hop['decision'],
                'assessment':'CONTRADICTORY_CUES_REQUIRE_REVIEW' if counter_ids else
                    'LOG_CUES_WITH_CONTEXT' if logs['supporting_ids'] else 'NO_LOG_CUES',
            },
            'review_action_options':[{'action':action,'source':'APPROVED_HISTORICAL_CASE',
                       'case_id':case['case_id'],'truth_source_ref':case['truth_source_ref'],
                       'control_authorized':False,'status':'APPLICABILITY_AND_HUMAN_REVIEW_REQUIRED'}
                       for case in cases['hits'] if case['relation_status']=='CONSISTENT_HISTORICAL_CASE'
                       for action in case['approved_actions']][:12],
            'knowledge_candidate_relations':[
                {'source':h['knowledge_id'],'relation':'TOPICALLY_RELATED_TO',
                 'target':candidate['fault_id'],'review_status':'UNREVIEWED_NOT_ACTIVE_KG',
                 'source_sha256':h['source_sha256']} for h in retrieved['hits']
            ],
        })
    document_learning={'status':'NOT_CONFIGURED','proposals':[]}
    if memory is not None and source_group_id:
        try:
            doc_proposals=memory.stage_reference_documents(
                source_group_id=source_group_id,chunks=chunks)
            document_learning={'status':'PENDING_EXPERT_REVIEW','proposals':doc_proposals}
        except Exception as exc:
            document_learning={'status':'STAGING_FAILED','error_type':type(exc).__name__,'proposals':[]}
    gap_learning={'status':'NOT_STAGED','proposals':[], 'active_kg_modified':False}
    if memory is not None and source_group_id:
        try:
            gap_candidates=[{'fault_id':c['fault_id'],
                             'knowledge_gap_codes':c['multi_hop']['knowledge_gap_codes']}
                            for c in items]
            if not gap_candidates:
                gap_candidates=[{'fault_id':'NO_UPSTREAM_CANDIDATE',
                                 'knowledge_gap_codes':['NO_UPSTREAM_CANDIDATE']}]
            staged=memory.stage_reasoning_gaps(
                source_group_id=source_group_id,
                equipment_scope=manifest.equipment_scope or 'UNKNOWN',
                candidates=gap_candidates,
                source_digests=[x['sha256'] for x in review.get('source_manifest',[])],
                snapshot=prior_snapshot)
            gap_learning={'status':'PENDING_EXPERT_REVIEW' if staged else 'NO_GAPS',
                          'proposals':staged,'active_kg_modified':False}
        except Exception as exc:
            gap_learning={'status':'STAGING_FAILED','proposals':[],
                          'error_type':type(exc).__name__,'active_kg_modified':False}
    graph_build = build_material_graph(chunks=chunks, source_group_id=source_group_id,
                                       memory=memory) if source_group_id else {
        'status':'NOT_APPLICABLE','proposals':[],'active_kg_modified':False}
    status = ('NO_UPSTREAM_CANDIDATE' if not candidates else
              'PARTIAL' if approved_graph_error or graph['status']=='FAILED' or semantic_status['status']=='FAILED'
                         or graph_build['status']=='PARTIAL'
                         or any(x.get('status')=='PARSE_FAILED' for x in ingestion)
              else 'SUCCEEDED')
    return {'schema_version':MATERIAL_REASONING_VERSION,'status':status,
            'source_group_id':source_group_id,'equipment_scope':manifest.equipment_scope or 'UNKNOWN',
            'document_ingestion':ingestion,'semantic_backend':semantic_status,
            'global_community_retrieval':community_global,
            'competing_hypothesis_review': build_competing_hypothesis_review(
                items, bidirectional.get('shared_symptoms', [])),
            'bidirectional_graph_comparison': {
                'schema_version': bidirectional['schema_version'] if 'schema_version' in bidirectional
                    else 'astrograph-bidirectional-graph-review.v1',
                'status': bidirectional['status'],
                'shared_symptoms': bidirectional.get('shared_symptoms', []),
                'reverse_observation_index': bidirectional.get('reverse_observation_index', []),
                'current_root_cause_proven': False,
                'automatic_control_authorized': False,
                'diagnostic_vote_allowed': False,
            },
            'global_community_index':{
                key: value for key,value in community_index.items() if key!='reports'},
            'retrieval_architecture':{
                'mode':'HIERARCHICAL_HYBRID_RAG_WITH_SEPARATE_APPROVED_GRAPH_REASONING',
                'index_levels':['DOCUMENT','SECTION','PROVENANCE_BOUND_CHUNK'],
                'keyword_channels':['BM25','CHAR_NGRAM'],
                'vector_channel':'LOCAL_SENTENCE_TRANSFORMERS_WITH_OPTIONAL_SQLITE_PERSISTENCE',
                'graph_channel':'APPROVED_GRAPH_COMMUNITY_GLOBAL_SEARCH_AND_SEPARATE_MULTIHOP',
                'r3_enhancements':['REVIEWED_SIGNED_COUNTERCLAIMS',
                    'BOUNDED_CONTRASTIVE_DOCUMENT_RETRIEVAL',
                    'BIDIRECTIONAL_NONCAUSAL_GRAPH_REVIEW'],
                'global_community_summaries_implemented':'EXTRACTIVE_REVIEWED_EDGE_COMMUNITY_REPORTS',
                'cross_document_community_clustering_implemented':bool(community_index.get('community_count')),
                'index_truncated':hierarchy.truncated,
                'unreviewed_multimodal_fragments_excluded':len(hierarchy.suppressed),
                'claim_boundary':'EXTRACTIVE_GLOBAL_COMMUNITY_SEARCH_NOT_FULL_MICROSOFT_GRAPHRAG; NO_CAUSAL_PROOF',
            },
            'approved_library_chunks_used':len(historical_chunks),
            'document_learning':document_learning,
            'reasoning_gap_learning':gap_learning,
            'answer_policy':{'current_root_cause_confirmed':False,
                'all_candidates_abstained':all(x['multi_hop']['response_mode']=='ABSTAIN_FROM_ROOT_CAUSE' for x in items),
                'provisional_hypotheses_allowed':any(x['multi_hop']['response_mode']=='EVIDENCE_BOUND_PROVISIONAL_ONLY' for x in items),
                'claim_boundary':'PROVISIONAL_NOT_CONFIRMED; NEVER_INFER_MISSING_GRAPH_EDGE',
                'automatic_control_authorized':False},
            'graph_construction':graph_build,
            'graph_query':{key:value for key,value in graph.items() if key!='paths'},
            'memory_snapshot':prior_snapshot,
            'candidates':items,'active_kg_modified':False,
            'formal_diagnosis_publishable':False,'automatic_control_authorized':False,
            'claim_boundary':'Multi-source context evaluates upstream candidates; no telemetry diagnosis, causal proof or Active KG promotion.'}
